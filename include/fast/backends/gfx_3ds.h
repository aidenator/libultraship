#pragma once
#ifdef __3DS__

#include "gfx_window_manager_api.h"

namespace Fast {

/**
 * @brief Window backend for the Nintendo 3DS.
 *
 * The game draws to the 400x240 top screen; the bottom screen is a text console that shows
 * stdout, so log output is readable on the device. Input and audio stay with SDL2, which this
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
