#include "ship/resource/archive/O2rArchive.h"

#include "ship/Context.h"
#include "ship/window/Window.h"
#include "spdlog/spdlog.h"
#include <unordered_map>
#include <chrono>
#include <algorithm>
#include <cerrno>
#include <cstring>

#ifdef __3DS__
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#endif

namespace Ship {

#ifdef __3DS__
namespace {

// A libzip source that reads the archive file in 64 KB blocks and keeps the last 2 MB of them.
// Each read the 3DS file system serves costs milliseconds, and libzip reads an entry in several
// small pieces, so loading many small files that sit together (a sound bank, a level's models)
// becomes a few large reads instead of thousands of small ones.
class CachedFileSource {
  public:
    static constexpr size_t kBlockSize = 64 * 1024;
    static constexpr size_t kBlockCount = 32;

    static zip_source_t* Create(const std::string& path, zip_error_t* error) {
        int fd = open(path.c_str(), O_RDONLY);
        if (fd < 0) {
            zip_error_set(error, ZIP_ER_OPEN, errno);
            return nullptr;
        }
        struct stat st;
        if (fstat(fd, &st) != 0) {
            zip_error_set(error, ZIP_ER_OPEN, errno);
            close(fd);
            return nullptr;
        }
        auto* self = new CachedFileSource(fd, (zip_uint64_t)st.st_size);
        zip_source_t* source = zip_source_function_create(&Callback, self, error);
        if (source == nullptr) {
            delete self;
        }
        return source;
    }

  private:
    struct Block {
        int64_t index = -1;
        uint64_t lastUse = 0;
        size_t size = 0;
        std::vector<uint8_t> data;
    };

    CachedFileSource(int fd, zip_uint64_t size) : mFd(fd), mSize(size), mBlocks(kBlockCount) {
        zip_error_init(&mError);
    }

    ~CachedFileSource() {
        close(mFd);
        zip_error_fini(&mError);
    }

    const Block* GetBlock(int64_t index) {
        Block* victim = &mBlocks[0];
        for (auto& block : mBlocks) {
            if (block.index == index) {
                block.lastUse = ++mUseCounter;
                return &block;
            }
            if (block.lastUse < victim->lastUse) {
                victim = &block;
            }
        }
        victim->data.resize(kBlockSize);
        const off_t offset = (off_t)index * (off_t)kBlockSize;
        size_t got = 0;
        if (lseek(mFd, offset, SEEK_SET) == offset) {
            while (got < kBlockSize) {
                const ssize_t n = read(mFd, victim->data.data() + got, kBlockSize - got);
                if (n <= 0) {
                    break;
                }
                got += (size_t)n;
            }
        }
        victim->index = got > 0 ? index : -1;
        victim->size = got;
        victim->lastUse = ++mUseCounter;
        return got > 0 ? victim : nullptr;
    }

    zip_int64_t Read(uint8_t* out, zip_uint64_t len) {
        zip_uint64_t done = 0;
        while (done < len && mPos < mSize) {
            const Block* block = GetBlock((int64_t)(mPos / kBlockSize));
            if (block == nullptr) {
                zip_error_set(&mError, ZIP_ER_READ, EIO);
                return -1;
            }
            const size_t inBlock = (size_t)(mPos % kBlockSize);
            if (inBlock >= block->size) {
                break;
            }
            const size_t n = std::min<size_t>(block->size - inBlock, (size_t)(len - done));
            memcpy(out + done, block->data.data() + inBlock, n);
            done += n;
            mPos += n;
        }
        return (zip_int64_t)done;
    }

    static zip_int64_t Callback(void* userdata, void* data, zip_uint64_t len, zip_source_cmd_t cmd) {
        auto* self = static_cast<CachedFileSource*>(userdata);
        switch (cmd) {
            case ZIP_SOURCE_OPEN:
                self->mPos = 0;
                return 0;
            case ZIP_SOURCE_READ:
                return self->Read(static_cast<uint8_t*>(data), len);
            case ZIP_SOURCE_CLOSE:
                return 0;
            case ZIP_SOURCE_STAT: {
                auto* st = ZIP_SOURCE_GET_ARGS(zip_stat_t, data, len, &self->mError);
                if (st == nullptr) {
                    return -1;
                }
                zip_stat_init(st);
                st->size = self->mSize;
                st->valid |= ZIP_STAT_SIZE;
                return sizeof(*st);
            }
            case ZIP_SOURCE_SEEK: {
                const zip_int64_t pos = zip_source_seek_compute_offset(self->mPos, self->mSize, data, len, &self->mError);
                if (pos < 0) {
                    return -1;
                }
                self->mPos = (zip_uint64_t)pos;
                return 0;
            }
            case ZIP_SOURCE_TELL:
                return (zip_int64_t)self->mPos;
            case ZIP_SOURCE_ERROR:
                return zip_error_to_data(&self->mError, data, len);
            case ZIP_SOURCE_FREE:
                delete self;
                return 0;
            case ZIP_SOURCE_SUPPORTS:
                return zip_source_make_command_bitmap(ZIP_SOURCE_OPEN, ZIP_SOURCE_READ, ZIP_SOURCE_CLOSE,
                                                      ZIP_SOURCE_STAT, ZIP_SOURCE_SEEK, ZIP_SOURCE_TELL,
                                                      ZIP_SOURCE_ERROR, ZIP_SOURCE_FREE, ZIP_SOURCE_SUPPORTS, -1);
            default:
                zip_error_set(&self->mError, ZIP_ER_OPNOTSUPP, 0);
                return -1;
        }
    }

    int mFd;
    zip_uint64_t mSize;
    zip_uint64_t mPos = 0;
    zip_error_t mError;
    std::vector<Block> mBlocks;
    uint64_t mUseCounter = 0;
};

} // namespace
#endif

O2rArchive::O2rArchive(const std::string& archivePath) : Archive(archivePath) {
    mZipArchive = nullptr;
}

O2rArchive::~O2rArchive() {
    SPDLOG_TRACE("destruct o2rarchive: {}", GetPath());
    Close();
}

zip_t* O2rArchive::GetZipHandle() {
#ifdef __3DS__
    // Opening another handle parses the archive's whole file list again, which takes seconds
    // for a large archive on the 3DS, and its SD card serves one read at a time anyway: every
    // reader waits its turn for the handle Open made
    {
        std::unique_lock<std::mutex> lock(mPoolMutex);
        mZipArchiveFree.wait(lock, [this] { return !mZipArchiveBusy; });
        mZipArchiveBusy = true;
        return mZipArchive;
    }
#endif
    std::lock_guard<std::mutex> lock(mPoolMutex);
    if (!mZipArchivePool.empty()) {
        zip_t* handle = mZipArchivePool.back();
        mZipArchivePool.pop_back();
        return handle;
    }
    return zip_open(GetPath().c_str(), ZIP_RDONLY, nullptr);
}

void O2rArchive::ReleaseZipHandle(zip_t* handle) {
    if (handle == nullptr) {
        return;
    }

#ifdef __3DS__
    {
        std::lock_guard<std::mutex> lock(mPoolMutex);
        mZipArchiveBusy = false;
    }
    mZipArchiveFree.notify_one();
    return;
#endif
    std::lock_guard<std::mutex> lock(mPoolMutex);
    mZipArchivePool.push_back(handle);
}

std::shared_ptr<File> O2rArchive::LoadFile(uint64_t hash) {
    const std::string& filePath =
        *Context::GetRawInstance()->GetResourceManager()->GetArchiveManager()->HashToString(hash);
    return LoadFile(filePath);
}

std::atomic<uint32_t> O2rArchive::sFilesRead = 0;
std::atomic<uint64_t> O2rArchive::sReadMicroseconds = 0;

namespace {
struct ReadTimer {
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    ~ReadTimer() {
        const auto us =
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count();
        O2rArchive::sReadMicroseconds += (uint64_t)us;
        O2rArchive::sFilesRead++;
    }
};
} // namespace

std::shared_ptr<File> O2rArchive::LoadFile(const std::string& filePath) {
    ReadTimer timer;
    zip_t* zipArchive = GetZipHandle();
    if (zipArchive == nullptr) {
        SPDLOG_TRACE("Failed to open file {} from zip archive {}. Archive not open.", filePath, GetPath());
        return nullptr;
    }

    auto zipEntryIndex = zip_name_locate(zipArchive, filePath.c_str(), 0);
    if (zipEntryIndex < 0) {
        SPDLOG_TRACE("Failed to find file {} in zip archive  {}.", filePath, GetPath());
        ReleaseZipHandle(zipArchive);
        return nullptr;
    }

    struct zip_stat zipEntryStat;
    zip_stat_init(&zipEntryStat);
    if (zip_stat_index(zipArchive, zipEntryIndex, 0, &zipEntryStat) != 0) {
        SPDLOG_TRACE("Failed to get entry information for file {} in zip archive  {}.", filePath, GetPath());
        ReleaseZipHandle(zipArchive);
        return nullptr;
    }

    // Filesize 0, no logging needed
    if (zipEntryStat.size == 0) {
        SPDLOG_TRACE("Failed to load file {}; filesize 0", filePath, GetPath());
        ReleaseZipHandle(zipArchive);
        return nullptr;
    }

    struct zip_file* zipEntryFile = zip_fopen_index(zipArchive, zipEntryIndex, 0);
    if (!zipEntryFile) {
        SPDLOG_TRACE("Failed to open file {} in zip archive  {}.", filePath, GetPath());
        ReleaseZipHandle(zipArchive);
        return nullptr;
    }

    auto fileToLoad = std::make_shared<File>();
    fileToLoad->Buffer = std::make_shared<std::vector<char>>(zipEntryStat.size);

    if (zip_fread(zipEntryFile, fileToLoad->Buffer->data(), zipEntryStat.size) < 0) {
        SPDLOG_TRACE("Error reading file {} in zip archive  {}.", filePath, GetPath());
    }

    if (zip_fclose(zipEntryFile) != 0) {
        SPDLOG_TRACE("Error closing file {} in zip archive  {}.", filePath, GetPath());
    }

    ReleaseZipHandle(zipArchive);

    fileToLoad->IsLoaded = true;

    return fileToLoad;
}

bool O2rArchive::Open() {
#ifdef __3DS__
    // Read-only through the block cache; nothing writes archives on the 3DS
    zip_error_t error;
    zip_error_init(&error);
    zip_source_t* source = CachedFileSource::Create(GetPath(), &error);
    mZipArchive = source != nullptr ? zip_open_from_source(source, ZIP_RDONLY, &error) : nullptr;
    if (mZipArchive == nullptr && source != nullptr) {
        zip_source_free(source);
    }
    zip_error_fini(&error);
#else
    mZipArchive = zip_open(GetPath().c_str(), ZIP_CREATE, nullptr);
#endif
    if (mZipArchive == nullptr) {
        SPDLOG_ERROR("Failed to load zip file \"{}\"", GetPath());
        return false;
    }

    auto zipNumEntries = zip_get_num_entries(mZipArchive, 0);
    for (auto i = 0; i < zipNumEntries; i++) {
        auto zipEntryName = zip_get_name(mZipArchive, i, 0);

        // It is possible for directories to have entries in a zip
        // file, we don't want those indexed as files in the archive
        if (zipEntryName[strlen(zipEntryName) - 1] == '/') {
            continue;
        }

        IndexFile(zipEntryName);
    }

    return true;
}

bool O2rArchive::Close() {
    bool success = true;

    if (mZipArchive != nullptr) {
        if (zip_close(mZipArchive) == -1) {
            SPDLOG_ERROR("Failed to close zip file \"{}\"", GetPath());
            success = false;
        }
        mZipArchive = nullptr;
    }

    std::lock_guard<std::mutex> lock(mPoolMutex);
    for (auto* handle : mZipArchivePool) {
        if (zip_close(handle) == -1) {
            SPDLOG_ERROR("Failed to close pooled zip file \"{}\"", GetPath());
            success = false;
        }
    }
    mZipArchivePool.clear();

    return success;
}

bool O2rArchive::WriteFile(const std::string& filePath, const std::vector<uint8_t>& data) {
    if (!mZipArchive) {
        SPDLOG_ERROR("Cannot write to zip: Archive is not open.");
        return false;
    }

    // Create a new zip source from the data buffer
    zip_source_t* source = zip_source_buffer(mZipArchive, data.data(), data.size(), 0);
    if (!source) {
        SPDLOG_ERROR("Failed to create zip source for file \"{}\"", filePath);
        return false;
    }

    // Add or replace the file in the zip archive
    if (zip_file_add(mZipArchive, filePath.c_str(), source, ZIP_FL_ENC_UTF_8 | ZIP_FL_OVERWRITE) < 0) {
        SPDLOG_ERROR("Failed to add file \"{}\" to ZIP", filePath);
        zip_source_free(source);
        return false;
    }

    // Save changes to disk
    if (zip_close(mZipArchive) < 0) {
        zip_error_t* error = zip_get_error(mZipArchive);
        SPDLOG_ERROR("Failed to save changes to zip archive: {} ({})", zip_error_strerror(error),
                     zip_error_code_zip(error));
        zip_discard(mZipArchive); // Close zip and discard changes
        return false;
    }

    // Clear the pool as the file on disk has likely changed
    {
        std::lock_guard<std::mutex> lock(mPoolMutex);
        for (auto* handle : mZipArchivePool) {
            zip_close(handle);
        }
        mZipArchivePool.clear();
    }

    SPDLOG_INFO("Successfully wrote file: {}", filePath);

    // Reopen the zip file so that it may continued to be used by libultraship
    mZipArchive = zip_open(GetPath().c_str(), ZIP_CREATE, nullptr);
    if (mZipArchive == nullptr) {
        SPDLOG_ERROR("Failed to reopen zip file after writing.");
        return false;
    }

    IndexFile(filePath);

    // Success
    return true;
}

} // namespace Ship
