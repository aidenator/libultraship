#ifdef __3DS__

#include "fast/backends/gfx_3ds.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <malloc.h>
#include <sys/iosupport.h>

// The build renames libultra's osGetTime; lift that while libctru declares its own
#pragma push_macro("osGetTime")
#undef osGetTime
#include <3ds.h>
#pragma pop_macro("osGetTime")
#include <citro3d.h>
#include <SDL2/SDL.h>

#include "fast/Fast3dGui.h"
#include "ship/Context.h"
#include "ship/controller/controldeck/ControlDeck.h"

#include <pthread.h>

// Main thread stack; libctru's default of 32 KB is too small for the interpreter and game code
extern "C" u32 __stacksize__;
u32 __stacksize__ = 256 * 1024;

// devkitPro starts pthreads, and so every std::thread, with a 32 KB heap-allocated stack. The
// game and engine threads need more, and overflowing one silently corrupts neighbouring heap
// objects. The link wraps pthread_create (see src/CMakeLists.txt) to raise small stacks.
static constexpr size_t kMinThreadStackSize = 256 * 1024;

extern "C" int __real_pthread_create(pthread_t* thread, const pthread_attr_t* attr, void* (*start)(void*), void* arg);

extern "C" int __wrap_pthread_create(pthread_t* thread, const pthread_attr_t* attr, void* (*start)(void*), void* arg) {
    pthread_attr_t sized;
    size_t stackSize = 0;
    if (attr != nullptr) {
        sized = *attr;
        pthread_attr_getstacksize(&sized, &stackSize);
    } else {
        pthread_attr_init(&sized);
    }
    if (stackSize < kMinThreadStackSize) {
        pthread_attr_setstacksize(&sized, kMinThreadStackSize);
    }
    return __real_pthread_create(thread, &sized, start, arg);
}

// SDL's main() mounts the app's romfs before calling SDL_main. The app has no romfs (everything is
// read from sdmc:/3ds/lighthouse), but looking for one still takes about a second, so skip it.
extern "C" Result __wrap_romfsMountSelf(const char* name) {
    return MAKERESULT(RL_PERMANENT, RS_NOTFOUND, RM_ROMFS, RD_NOT_FOUND);
}

namespace Fast {

void* Start3dsThreadOnCore(void (*entry)(void*), void* arg, int core) {
    bool isNew3ds = false;
    APT_CheckNew3DS(&isNew3ds);
    if (core >= 2 && !isNew3ds) {
        return nullptr;
    }
    // Core 1 is the system's, and an app may only use the share of it the system grants. 80% is
    // the most the system accepts.
    static bool sCore1Granted = false;
    if (core == 1 && !sCore1Granted) {
        const Result res = APT_SetAppCpuTimeLimit(80);
        if (R_FAILED(res)) {
            printf("No time on core 1 (0x%08lx)\n", (unsigned long)res);
            return nullptr;
        }
        sCore1Granted = true;
    }
    s32 priority = 0x30;
    svcGetThreadPriority(&priority, CUR_THREAD_HANDLE);
    return threadCreate(entry, arg, kMinThreadStackSize, priority, core, false);
}

bool Is3dsNew() {
    bool isNew3ds = false;
    APT_CheckNew3DS(&isNew3ds);
    return isNew3ds;
}

void Join3dsThread(void* thread) {
    threadJoin((Thread)thread, U64_MAX);
    threadFree((Thread)thread);
}

// The console's own output device, kept so the bottom screen can draw after stdout moves away
static const devoptab_t* sConsoleDevice = nullptr;
// Where stdout went before the loading screen took it: the 3dslink host, or nothing
static const devoptab_t* sStdoutDevice = nullptr;
static bool sStdoutToHost = false;
static LightLock sConsoleLock;
static bool sLoadingScreenUp = false;
static bool sStatusScreenUp = false;
static int sShownFps = -1;
static int sShownBarCells = -1;
static uint64_t sStartTick = 0;
// When the screens were up and when stdout reached the 3dslink host, for the startup log
static uint64_t sScreensTick = 0;
static uint64_t sLinkTick = 0;

// Bring up the screens and the bottom-screen console before main(), so output from startup
// (spdlog's default logger writes to stdout until the game configures logging) is visible.
static struct EarlyConsole {
    EarlyConsole() {
        sStartTick = svcGetSystemTick();
        LightLock_Init(&sConsoleLock);
        // New 3DS: 804 MHz and the extra L2 cache (no effect on older models)
        osSetSpeedupEnable(true);
        gfxInitDefault();
        consoleInit(GFX_BOTTOM, nullptr);
        sConsoleDevice = devoptab_list[STD_OUT];
        sScreensTick = svcGetSystemTick();

        // Started with `3dslink -s`: send stdout and stderr to the host's terminal instead
        if (__3dslink_host.s_addr != 0) {
            static constexpr u32 kSocBufferSize = 0x40000;
            void* socBuffer = memalign(0x1000, kSocBufferSize);
            if (socBuffer != nullptr && R_SUCCEEDED(socInit((u32*)socBuffer, kSocBufferSize))) {
                printf("Logging to 3dslink host\n");
                if (link3dsStdio() >= 0) {
                    setvbuf(stdout, nullptr, _IOLBF, 0);
                    sStdoutToHost = true;
                    sLinkTick = svcGetSystemTick();
                }
            }
        }
    }
} sEarlyConsole;

static ssize_t DiscardWrite(struct _reent* r, void* fd, const char* ptr, size_t len) {
    return len;
}

static void ConsoleWrite(const char* text) {
    LightLock_Lock(&sConsoleLock);
    sConsoleDevice->write_r(_REENT, nullptr, text, strlen(text));
    LightLock_Unlock(&sConsoleLock);
}

// Text centred on a row of the 40x30 console, replacing whatever the row held
static void ConsoleRow(int row, const char* color, const char* text) {
    char line[128];
    const int length = std::min((int)strlen(text), 40);
    const int column = (40 - length) / 2;
    snprintf(line, sizeof(line), "\x1b[%d;1H\x1b[2K\x1b[%d;%dH%s%.*s\x1b[0m", row + 1, row + 1, column + 1, color,
             length, text);
    ConsoleWrite(line);
}

// Bottom-screen rows, counted from 0
static constexpr int kTitleRow = 2;
static constexpr int kVersionRow = 3;
static constexpr int kLoadStepRow = 11;
static constexpr int kBarRow = 13;
static constexpr int kBarCells = 30;
static constexpr int kLogRow = 16;
static constexpr int kErrorRow = 19; // and the row under it
static constexpr int kFirstStatusLine = 6;
static constexpr int kStatusLineCount = 14;
static constexpr int kFpsRow = 23;
static constexpr int kNoteRow = 25;
static constexpr int kHintRow = 28;

// While loading, stdout lands here: the latest line shows under the bar, errors stay up in red
static char sLogLine[160];
static size_t sLogLength = 0;

static void ShowLogLine(char* line) {
    const bool isError = strstr(line, "[error]") != nullptr || strstr(line, "[critical]") != nullptr;
    // Drop spdlog's "[time] [file:line] [level] " prefix
    char* text = line;
    for (char* close = strstr(text, "] "); close != nullptr; close = strstr(text, "] ")) {
        text = close + 2;
    }
    // The step is already shown above the bar
    if (*text == '\0' || strncmp(text, "Loading: ", 9) == 0) {
        return;
    }
    if (isError) {
        ConsoleRow(kErrorRow, "\x1b[31;1m", text);
        ConsoleRow(kErrorRow + 1, "\x1b[31;1m", strlen(text) > 40 ? text + 40 : "");
    } else {
        ConsoleRow(kLogRow, "\x1b[2m", text);
    }
}

static ssize_t LoadingWrite(struct _reent* r, void* fd, const char* ptr, size_t len) {
    if (sStdoutDevice != nullptr && sStdoutDevice->write_r != nullptr) {
        sStdoutDevice->write_r(r, fd, ptr, len);
    }
    for (size_t i = 0; i < len; i++) {
        if (ptr[i] == '\n' || sLogLength == sizeof(sLogLine) - 1) {
            sLogLine[sLogLength] = '\0';
            if (sLoadingScreenUp) {
                ShowLogLine(sLogLine);
            }
            sLogLength = 0;
        } else if (ptr[i] != '\r') {
            sLogLine[sLogLength++] = ptr[i];
        }
    }
    return len;
}

static void TakeStdout(ssize_t (*write)(struct _reent*, void*, const char*, size_t)) {
    static devoptab_t device = {};
    fflush(stdout);
    fflush(stderr);
    if (sStdoutDevice == nullptr) {
        static devoptab_t discard = {};
        discard.name = "discard";
        discard.write_r = DiscardWrite;
        sStdoutDevice = sStdoutToHost ? devoptab_list[STD_OUT] : &discard;
    }
    device = *sStdoutDevice;
    device.name = "bottom";
    device.write_r = write;
    devoptab_list[STD_OUT] = &device;
    devoptab_list[STD_ERR] = &device;
}

static void ShowTitle(const char* title, const char* version) {
    ConsoleWrite("\x1b[2J");
    ConsoleRow(kTitleRow, "\x1b[33;1m", title);
    ConsoleRow(kVersionRow, "\x1b[2m", version);
}

void Show3dsLoadingScreen(const char* title, const char* version) {
    if (sConsoleDevice == nullptr || sLoadingScreenUp || sStatusScreenUp) {
        return;
    }
    TakeStdout(LoadingWrite);
    sLoadingScreenUp = true;
    ShowTitle(title, version);
    const auto ms = [](uint64_t from, uint64_t to) { return (to - from) * 1000.0 / SYSCLOCK_ARM11; };
    printf("Before main: screens %.0f ms, 3dslink %.0f ms, the rest %.0f ms\n", ms(sStartTick, sScreensTick),
           sLinkTick != 0 ? ms(sScreensTick, sLinkTick) : 0.0,
           ms(sLinkTick != 0 ? sLinkTick : sScreensTick, svcGetSystemTick()));
    Set3dsLoadProgress(0.0f, "Starting");
}

void Set3dsLoadProgress(float fraction, const char* step) {
    printf("Loading: %s (%.0f%%) at %.0f ms\n", step, fraction * 100.0f,
           (svcGetSystemTick() - sStartTick) * 1000.0 / SYSCLOCK_ARM11);
    if (!sLoadingScreenUp) {
        return;
    }
    char text[48];
    snprintf(text, sizeof(text), "%s...", step);
    ConsoleRow(kLoadStepRow, "", text);

    const int cells = std::clamp((int)(fraction * kBarCells + 0.5f), 0, kBarCells);
    if (cells == sShownBarCells) {
        return;
    }
    sShownBarCells = cells;
    char bar[160];
    char* out = bar + snprintf(bar, sizeof(bar), "\x1b[%d;%dH\x1b[47m", kBarRow + 1, (40 - kBarCells) / 2 + 1);
    for (int i = 0; i < kBarCells; i++) {
        if (i == cells) {
            out += sprintf(out, "\x1b[44m");
        }
        *out++ = ' ';
    }
    strcpy(out, "\x1b[0m");
    ConsoleWrite(bar);
}

void Show3dsStatusScreen(const char* title, const char* version) {
    if (sConsoleDevice == nullptr || sStatusScreenUp) {
        return;
    }
    if (!sLoadingScreenUp) {
        TakeStdout(LoadingWrite);
    }
    // From here on the log goes only to the 3dslink host (if any) and the log file
    sLoadingScreenUp = false;
    sStatusScreenUp = true;
    ShowTitle(title, version);
    ConsoleRow(kHintRow, "\x1b[2m", "START + SELECT to quit");
}

void Update3dsStatusLine(int index, const char* text) {
    static char sShown[kStatusLineCount][48];
    if (!sStatusScreenUp || index < 0 || index >= kStatusLineCount || strncmp(sShown[index], text, 47) == 0) {
        return;
    }
    snprintf(sShown[index], sizeof(sShown[index]), "%s", text);
    // A line starting with '*' is a heading
    if (text[0] == '*') {
        ConsoleRow(kFirstStatusLine + index, "\x1b[36;1m", text + 1);
    } else {
        ConsoleRow(kFirstStatusLine + index, "", text);
    }
}

void Update3dsStatusFps(float fps) {
    const int rounded = (int)(fps + 0.5f);
    if (!sStatusScreenUp || rounded == sShownFps) {
        return;
    }
    sShownFps = rounded;
    char text[16];
    snprintf(text, sizeof(text), "%d fps", rounded);
    ConsoleRow(kFpsRow, "\x1b[32m", text);
}

volatile uint64_t g3dsVBlankWaitTicks = 0;

void Update3dsStatusNote(const char* text) {
    if (sStatusScreenUp) {
        ConsoleRow(kNoteRow, "\x1b[33m", text);
    }
}

void GfxWindowBackend3DS::Init(const char* gameName, const char* apiName, bool startFullScreen, uint32_t width,
                               uint32_t height, int32_t posX, int32_t posY) {
    bool isNew3ds = false;
    APT_CheckNew3DS(&isNew3ds);
    printf("%s (%s) on %s 3DS\n", gameName, apiName, isNew3ds ? "New" : "Old");
    printf("heap %lu KB, linear %lu KB free\n", (unsigned long)(envGetHeapSize() / 1024),
           (unsigned long)(linearSpaceFree() / 1024));

    // The SDL backend does this too: it creates the ImGui context and links the GUI to the
    // interpreter. On the 3DS the GUI is headless, so no window handles are needed.
    GuiWindowInitData windowImpl = {};
    windowImpl.Backend = WindowBackend::FAST3D_3DS_CITRO3D;
    std::dynamic_pointer_cast<Fast3dGui>(Ship::Context::GetRawInstance()->GetWindow()->GetGui())->Init(windowImpl);
}

void GfxWindowBackend3DS::Close() {
    mIsRunning = false;
}

void GfxWindowBackend3DS::SetKeyboardCallbacks(bool (*onKeyDown)(int scancode), bool (*onKeyUp)(int scancode),
                                               void (*onAllKeysUp)()) {
}

void GfxWindowBackend3DS::SetMouseCallbacks(bool (*onMouseButtonDown)(int btn), bool (*onMouseButtonUp)(int btn)) {
}

void GfxWindowBackend3DS::SetFullscreenChangedCallback(void (*onFullscreenChanged)(bool isNowFullscreen)) {
}

void GfxWindowBackend3DS::SetFullscreen(bool fullscreen) {
}

void GfxWindowBackend3DS::GetActiveWindowRefreshRate(uint32_t* refreshRate) {
    *refreshRate = 60;
}

void GfxWindowBackend3DS::SetCursorVisibility(bool visible) {
}

void GfxWindowBackend3DS::SetMousePos(int32_t posX, int32_t posY) {
}

void GfxWindowBackend3DS::GetMousePos(int32_t* x, int32_t* y) {
    *x = 0;
    *y = 0;
}

void GfxWindowBackend3DS::GetMouseDelta(int32_t* x, int32_t* y) {
    *x = 0;
    *y = 0;
}

void GfxWindowBackend3DS::GetMouseWheel(float* x, float* y) {
    *x = 0.0f;
    *y = 0.0f;
}

bool GfxWindowBackend3DS::GetMouseState(uint32_t btn) {
    return false;
}

void GfxWindowBackend3DS::SetMouseCapture(bool capture) {
}

bool GfxWindowBackend3DS::IsMouseCaptured() {
    return false;
}

void GfxWindowBackend3DS::GetDimensions(uint32_t* width, uint32_t* height, int32_t* posX, int32_t* posY) {
    *width = kTopScreenWidth;
    *height = kTopScreenHeight;
    *posX = 0;
    *posY = 0;
}

void GfxWindowBackend3DS::SetDimensions(uint32_t width, uint32_t height, int32_t posX, int32_t posY) {
}

Ship::WindowRect GfxWindowBackend3DS::GetPrimaryMonitorRect() {
    return { 0, 0, (int32_t)kTopScreenWidth, (int32_t)kTopScreenHeight };
}

void GfxWindowBackend3DS::HandleEvents() {
    // Services HOME, sleep and power; returns false once the system asks the app to quit
    if (!aptMainLoop() && mIsRunning) {
        printf("System requested exit\n");
        mIsRunning = false;
    }
    // SDL scans the buttons in its 3DS video driver, which this backend does not use, so scan here.
    // SDL's joystick driver then reads the result in SDL_PumpEvents.
    hidScanInput();
    SDL_PumpEvents();
    // The SDL backend drains the queue each frame, and the headless GUI never runs the element
    // that connects gamepads, so do both here: connect the 3DS's own buttons, drop the rest.
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
        auto devices = Ship::Context::GetRawInstance()->GetControlDeck()->GetConnectedPhysicalDeviceManager();
        if (event.type == SDL_CONTROLLERDEVICEADDED) {
            devices->HandlePhysicalDeviceConnect(event.cdevice.which);
            printf("Controller connected: %s\n", SDL_GameControllerNameForIndex(event.cdevice.which));
        } else if (event.type == SDL_CONTROLLERDEVICEREMOVED) {
            devices->HandlePhysicalDeviceDisconnect(event.cdevice.which);
        }
    }
    // START + SELECT quits, so a .3dsx returns to the Homebrew Launcher. SELECT has no N64 button.
    if (mIsRunning && (hidKeysHeld() & (KEY_START | KEY_SELECT)) == (KEY_START | KEY_SELECT)) {
        printf("START + SELECT: exiting\n");
        mIsRunning = false;
    }
}

bool GfxWindowBackend3DS::IsFrameReady() {
    return true;
}

void GfxWindowBackend3DS::SwapBuffersBegin() {
    // At the game's own rate (20 or 30 fps) the game paces itself; citro3d flips the screen during
    // vblank (see GfxRenderingAPICitro3D::StartFrame). Drawing in-between frames, the frame pacer
    // presents as fast as it can draw and counts on the swap waiting for the screen, as vsync does
    // on PC: without that it draws frames the screen never shows. One frame per refresh is enough;
    // when a refresh has already passed since the last present, drawing the next frame while the
    // GPU finishes this one beats waiting for another.
    if (mTargetFps >= 60) {
        if (C3D_FrameCounter(0) == mLastPresentVBlank) {
            const uint64_t waitStart = svcGetSystemTick();
            gspWaitForVBlank();
            g3dsVBlankWaitTicks += svcGetSystemTick() - waitStart;
        }
        mLastPresentVBlank = C3D_FrameCounter(0);
    }
}

void GfxWindowBackend3DS::SwapBuffersEnd() {
}

double GfxWindowBackend3DS::GetTime() {
    return (double)svcGetSystemTick() / SYSCLOCK_ARM11;
}

int GfxWindowBackend3DS::GetTargetFps() {
    return mTargetFps;
}

void GfxWindowBackend3DS::SetTargetFps(int fps) {
    mTargetFps = fps;
}

void GfxWindowBackend3DS::SetMaxFrameLatency(int latency) {
}

const char* GfxWindowBackend3DS::GetKeyName(int scancode) {
    return "";
}

bool GfxWindowBackend3DS::CanDisableVsync() {
    return false;
}

bool GfxWindowBackend3DS::IsRunning() {
    return mIsRunning;
}

void GfxWindowBackend3DS::Destroy() {
    // SDL stops its audio thread and the 3DS sound service (NDSP) only here. Left running, they
    // can run once more while the exiting program's memory is released, and crash on their stacks.
    SDL_Quit();
    gfxExit();
}

bool GfxWindowBackend3DS::IsFullscreen() {
    return true;
}

} // namespace Fast

#endif
