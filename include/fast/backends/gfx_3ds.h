#pragma once
#ifdef __3DS__

#include "gfx_window_manager_api.h"

namespace Fast {

/**
 * @brief Starts entry(arg) on another CPU core, at the calling thread's priority.
 *
 * Every pthread (and so every std::thread) runs on the app's main core. Apps may also use 80% of
 * core 1 (asked for here on its first use) and, on a New 3DS, core 2 and (from the Homebrew
 * Launcher) usually core 3, so heavy threads can run alongside the main thread instead of taking
 * turns.
 * @return the thread, or nullptr if the core is not available (such as on an Old 3DS).
 */
void* Start3dsThreadOnCore(void (*entry)(void*), void* arg, int core);
void Join3dsThread(void* thread);

/** @brief Whether this is a New 3DS (faster CPU, more cores) rather than an Old 3DS. */
bool Is3dsNew();

/**
 * @brief Turns the bottom screen from the startup log into a loading screen.
 *
 * Until this is called, the bottom screen shows stdout. Afterwards it shows the title, version, a
 * progress bar (see Set3dsLoadProgress) and the latest log line, with errors kept up in red.
 * stdout still reaches a 3dslink host, if one is connected.
 */
void Show3dsLoadingScreen(const char* title, const char* version);

/// Whether the system asked the app to close (from the HOME Menu), rather than the player quitting
bool Is3dsSystemClose();

/// Ends the app at once after a system close, skipping the shutdown that would wait on the GPU and
/// sound the suspended app no longer has. Save what matters first.
[[noreturn]] void Exit3dsForSystemClose();

/// Moves the loading bar to fraction (0 to 1) and names the step under way; also logs the time
void Set3dsLoadProgress(float fraction, const char* step);

/// Uses convergence (the distance drawn at the screen's depth in 3D) for the world instead of the
/// usual one, for a scene that looks better with its own, such as the file select; 0 goes back
void Set3dsSceneConvergence(float convergence);

/// The player's 3D settings: convergence (the distance drawn at the screen's depth) and the HUD's
/// depth, as a share of the farthest shift (+ is into the screen)
void Set3dsStereoDepth(float convergence, float hudDepth);

/**
 * @brief Switches the bottom screen between the status screen and a page the game draws, such as
 * a menu.
 *
 * Switching clears the screen. The status lines, frame rate and note keep updating while the
 * game's page is up, and come back as they are; the game then draws its own parts of the status
 * screen again, such as a button.
 */
void Set3dsBottomPage(bool gamePage);

/// Writes text at a cell of the 40x30 bottom screen (column, row, from 0), over what is there.
/// ANSI codes set the colours, and are reset after the text.
void Draw3dsBottomText(int column, int row, const char* text);

/// A tap on the bottom screen since the last call, as the cell tapped; false when none
bool Take3dsTap(int* column, int* row);

/**
 * @brief Turns the bottom screen into a status screen, once loading is done.
 *
 * It shows the title, version, lines of game status (see Update3dsStatusLine) and the frame rate.
 * stdout then goes only to a 3dslink host (if one is connected) and the log file.
 */
void Show3dsStatusScreen(const char* title, const char* version);

/**
 * @brief Shows an error on the bottom screen until START is pressed, then exits.
 *
 * For problems the player has to fix, such as a missing file: title says what's wrong and message
 * how to fix it ('\n' starts a new line; long lines wrap at spaces).
 */
[[noreturn]] void Show3dsFatalError(const char* title, const char* message);

/// Sets one of the status screen's 14 lines; a line starting with '*' is a heading
void Update3dsStatusLine(int index, const char* text);

/// Shows a new frame rate on the status screen (if it is up). Call from the render thread.
void Update3dsStatusFps(float fps);

/// Shows a line of text under the frame rate on the status screen (if it is up)
void Update3dsStatusNote(const char* text);
// Time the renderer has spent waiting for vblank before presenting, in system ticks
extern volatile uint64_t g3dsVBlankWaitTicks;

/**
 * @brief Window backend for the Nintendo 3DS.
 *
 * The game draws to the 400x240 top screen; the bottom screen is a text console that shows
 * stdout during startup, then a loading and a status screen (see Show3dsLoadingScreen). Input and audio stay with SDL2, which this
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
    uint32_t mLastPresentVBlank = 0; // the top screen's refresh count at the last present
};

} // namespace Fast

#endif
