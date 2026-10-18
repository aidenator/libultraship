#pragma once
#ifdef __3DS__

#include "gfx_window_manager_api.h"

namespace Fast {

/**
 * @brief Starts entry(arg) on another CPU core, at the calling thread's priority.
 *
 * Every pthread (and so every std::thread) runs on the app's main core. On a New 3DS, apps may
 * also use core 2, so heavy threads can run alongside the main thread instead of taking turns.
 * @return the thread, or nullptr if the core is not available (such as on an Old 3DS).
 */
void* Start3dsThreadOnCore(void (*entry)(void*), void* arg, int core);
void Join3dsThread(void* thread);

/**
 * @brief Turns the bottom screen from the startup log into a status screen.
 *
 * Until this is called, the bottom screen shows stdout, so startup problems are readable on the
 * device. Afterwards it shows the title, version and frame rate, and stdout goes only to a
 * 3dslink host (if one is connected) and the log file.
 */
void Show3dsStatusScreen(const char* title, const char* version);

/// Shows a new frame rate on the status screen (if it is up). Call from the render thread.
void Update3dsStatusFps(float fps);

/**
 * @brief Window backend for the Nintendo 3DS.
 *
 * The game draws to the 400x240 top screen; the bottom screen is a text console that shows
 * stdout during startup, then a status screen (see Show3dsStatusScreen). Input and audio stay with SDL2, which this
 * backend pumps each frame. There is no keyboard, mouse or windowing, so those calls do nothing.
 */
class GfxWindowBackend3DS final : public GfxWindowBackend {
  public:
    GfxWindowBackend3DS() = default;
    ~GfxWindowBackend3DS() override = default;

    void Init(const char* gameName, const char* apiName, bool startFullScreen, uint32_t width, uint32_t height,
              int32_t posX, int32_t posY) override;
    void Close() override;
    void SetKeyboardCallbacks(bool (*onKeyDown)(int scancode), bool (*onKeyUp)(int scancode),
                              void (*onAllKeysUp)()) override;
    void SetMouseCallbacks(bool (*onMouseButtonDown)(int btn), bool (*onMouseButtonUp)(int btn)) override;
    void SetFullscreenChangedCallback(void (*onFullscreenChanged)(bool isNowFullscreen)) override;
    void SetFullscreen(bool fullscreen) override;
    void GetActiveWindowRefreshRate(uint32_t* refreshRate) override;
    void SetCursorVisibility(bool visible) override;
    void SetMousePos(int32_t posX, int32_t posY) override;
    void GetMousePos(int32_t* x, int32_t* y) override;
    void GetMouseDelta(int32_t* x, int32_t* y) override;
    void GetMouseWheel(float* x, float* y) override;
    bool GetMouseState(uint32_t btn) override;
    void SetMouseCapture(bool capture) override;
    bool IsMouseCaptured() override;
    void GetDimensions(uint32_t* width, uint32_t* height, int32_t* posX, int32_t* posY) override;
    void SetDimensions(uint32_t width, uint32_t height, int32_t posX, int32_t posY) override;
    Ship::WindowRect GetPrimaryMonitorRect() override;
    void HandleEvents() override;
    bool IsFrameReady() override;
    void SwapBuffersBegin() override;
    void SwapBuffersEnd() override;
    double GetTime() override;
    int GetTargetFps() override;
    void SetTargetFps(int fps) override;
    void SetMaxFrameLatency(int latency) override;
    const char* GetKeyName(int scancode) override;
    bool CanDisableVsync() override;
    bool IsRunning() override;
    void Destroy() override;
    bool IsFullscreen() override;

    static constexpr uint32_t kTopScreenWidth = 400;
    static constexpr uint32_t kTopScreenHeight = 240;

  private:
    bool mIsRunning = true;
    int mTargetFps = 60;
};

} // namespace Fast

#endif
