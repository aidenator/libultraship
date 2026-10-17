#ifdef __3DS__

#include "fast/backends/gfx_3ds.h"

#include <cstdio>
#include <malloc.h>

// The build renames libultra's osGetTime; lift that while libctru declares its own
#pragma push_macro("osGetTime")
#undef osGetTime
#include <3ds.h>
#pragma pop_macro("osGetTime")
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

namespace Fast {

void* Start3dsThreadOnCore(void (*entry)(void*), void* arg, int core) {
    bool isNew3ds = false;
    APT_CheckNew3DS(&isNew3ds);
    if (core == 2 && !isNew3ds) {
        return nullptr;
    }
    s32 priority = 0x30;
    svcGetThreadPriority(&priority, CUR_THREAD_HANDLE);
    return threadCreate(entry, arg, kMinThreadStackSize, priority, core, false);
}

void Join3dsThread(void* thread) {
    threadJoin((Thread)thread, U64_MAX);
    threadFree((Thread)thread);
}

// Bring up the screens and the bottom-screen console before main(), so output from startup
// (spdlog's default logger writes to stdout until the game configures logging) is visible.
static struct EarlyConsole {
    EarlyConsole() {
        // New 3DS: 804 MHz and the extra L2 cache (no effect on older models)
        osSetSpeedupEnable(true);
        gfxInitDefault();
        consoleInit(GFX_BOTTOM, nullptr);

        // Started with `3dslink -s`: send stdout and stderr to the host's terminal instead
        if (__3dslink_host.s_addr != 0) {
            static constexpr u32 kSocBufferSize = 0x40000;
            void* socBuffer = memalign(0x1000, kSocBufferSize);
            if (socBuffer != nullptr && R_SUCCEEDED(socInit((u32*)socBuffer, kSocBufferSize))) {
                printf("Logging to 3dslink host\n");
                if (link3dsStdio() >= 0) {
                    setvbuf(stdout, nullptr, _IOLBF, 0);
                }
            }
        }
    }
} sEarlyConsole;

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
    // The game paces itself; citro3d flips the screen during vblank (see GfxRenderingAPICitro3D::StartFrame)
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
