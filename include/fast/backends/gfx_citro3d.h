#pragma once
#ifdef __3DS__

#include "gfx_rendering_api.h"

#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <imgui.h>

#include "fast/interpreter.h"

struct C3D_RenderTarget_tag;

namespace Fast {

namespace Citro3D {

// A value the combiner reads, after the formula has been split into TEV stages
enum class Src : uint8_t { Const, Primary, Tex0, Tex1, Tex2, Prev, Buffer, Combined };

struct Operand {
    Src src = Src::Prev;
    bool alpha = false;    // read the source's alpha (replicated into RGB for colour stages)
    bool oneMinus = false; // 1 - value
    int16_t constExpr = -1; // for Src::Const: index into TevProgram::constExprs
};

// One channel (colour or alpha) of a TEV stage
struct ChannelStage {
    uint8_t func = 0; // GPU_COMBINEFUNC
    Operand ops[3];
    uint8_t numOps = 1;
    bool updateBuffer = false;
};

// A constant the CPU works out per draw: the N64 combiner inputs that stay the same across a
// batch (prim and env colour, usually) and anything built only from them
struct ConstExpr {
    enum class Op : uint8_t { Leaf, Mul, Add, Sub, OneMinus } op = Op::Leaf;
    uint8_t item = 0;  // Leaf: SHADER_* item
    bool alpha = false; // Leaf: read the input's alpha
    int16_t a = -1, b = -1;
};

struct TevProgram {
    std::vector<ChannelStage> rgb;
    std::vector<ChannelStage> alpha; // same length as rgb
    std::vector<ConstExpr> constExprs;
};

} // namespace Citro3D

struct ShaderProgram {
    uint64_t shaderId0;
    uint64_t shaderId1;
    CCFeatures cc;
    uint8_t numInputs;
    bool usedTextures[2];
    bool inputUsedRgb[7];
    bool inputUsedAlpha[7];
    uint8_t numFloats; // per vertex in the interpreter's buffer
    // Compiled TEV programs, keyed by which input (if any) arrives as the per-vertex colour
    std::unordered_map<uint32_t, Citro3D::TevProgram> programs;
};

/**
 * @brief Rendering backend for the 3DS GPU (PICA200) through citro3d.
 *
 * Fast3D transforms and lights vertices on the CPU, so the vertex shader only passes them
 * through. Each N64 colour combiner is translated into fixed-function TEV stages: combiner
 * inputs that stay constant across a draw (usually prim and env colour) become stage constants,
 * and the one that varies (usually shade) becomes the vertex colour.
 */
class GfxRenderingAPICitro3D final : public GfxRenderingAPI {
  public:
    ~GfxRenderingAPICitro3D() override;
    const char* GetName() override;
    int GetMaxTextureSize() override;
    GfxClipParameters GetClipParameters() override;
    void UnloadShader(ShaderProgram* oldPrg) override;
    void LoadShader(ShaderProgram* newPrg) override;
    void ClearShaderCache() override;
    ShaderProgram* CreateAndLoadNewShader(uint64_t shaderId0, uint64_t shaderId1) override;
    ShaderProgram* LookupShader(uint64_t shaderId0, uint64_t shaderId1) override;
    void ShaderGetInfo(ShaderProgram* prg, uint8_t* numInputs, bool usedTextures[2]) override;
    uint32_t NewTexture() override;
    void SelectTexture(int tile, uint32_t textureId) override;
    void UploadTexture(const uint8_t* rgba32Buf, uint32_t width, uint32_t height) override;
    void SetSamplerParameters(int sampler, bool linear_filter, uint32_t cms, uint32_t cmt) override;
    void SetDepthTestAndMask(bool depth_test, bool z_upd) override;
    void SetZmodeDecal(bool decal) override;
    void SetViewport(int x, int y, int width, int height) override;
    void SetScissor(int x, int y, int width, int height) override;
    void SetUseAlpha(bool useAlpha) override;
    void DrawTriangles(float buf_vbo[], size_t buf_vbo_len, size_t buf_vbo_num_tris) override;
    void Init() override;
    void OnResize() override;
    void StartFrame() override;
    void EndFrame() override;
    void FinishRender() override;
    int CreateFramebuffer() override;
    void UpdateFramebufferParameters(int fb_id, uint32_t width, uint32_t height, uint32_t msaa_level,
                                     bool opengl_invertY, bool render_target, bool has_depth_buffer,
                                     bool can_extract_depth) override;
    void StartDrawToFramebuffer(int fbId, float noiseScale) override;
    void CopyFramebuffer(int fbDstId, int fbSrcId, int srcX0, int srcY0, int srcX1, int srcY1, int dstX0, int dstY0,
                         int dstX1, int dstY1) override;
    void ClearFramebuffer(bool color, bool depth) override;
    void ReadFramebufferToCPU(int fbId, uint32_t width, uint32_t height, uint16_t* rgba16Buf) override;
    void ResolveMSAAColorBuffer(int fbIdTarger, int fbIdSrc) override;
    std::unordered_map<std::pair<float, float>, uint16_t, hash_pair_ff>
    GetPixelDepth(int fb_id, const std::set<std::pair<float, float>>& coordinates) override;
    void* GetFramebufferTextureId(int fbId) override;
    void SelectTextureFb(int fbId) override;
    void DeleteTexture(uint32_t texId) override;
    void SetTextureFilter(FilteringMode mode) override;
    FilteringMode GetTextureFilter() override;
    void SetSrgbMode() override;
    ImTextureID GetTextureById(int id) override;
    void SetCurrentPrimDepth(float depth) override;

  private:
    struct Texture {
        void* tex = nullptr; // C3D_Tex, an anonymous struct that cannot be forward-declared
        float scaleS = 1.0f; // the image's share of the power-of-two texture it was padded into
        float scaleT = 1.0f;
    };

    // An offscreen framebuffer: a 256x512 texture holding an image stored sideways like the screen
    struct Framebuffer {
        void* tex = nullptr; // C3D_Tex
        C3D_RenderTarget_tag* target = nullptr; // created when first drawn into
        uint32_t width = 0;  // the size the game sees
        uint32_t height = 0;
        uint32_t contentWidth = 0; // the size of the image it holds: a copied screen, or its own drawing
        uint32_t contentHeight = 0;
    };

    const Citro3D::TevProgram& GetTevProgram(ShaderProgram* prg, int primaryRgb, int primaryAlpha);
    void ApplyTevProgram(const Citro3D::TevProgram& program, const float* inputs, bool hasFog, const float* fog);
    void ApplyDepth();
    void ApplyBlendAndAlphaTest();
    void DrawRun(const float* buf, size_t firstTri, size_t numTris, int primaryRgb, int primaryAlpha);
    void SubmitDraws();

    // Deferred drawing. During a frame the interpreter's calls are only recorded; at the frame's
    // end a replay runs the same methods for real, waits for the GPU to finish the previous frame
    // and sends this one. On a New 3DS the replay runs on its own thread, while the interpreter
    // records the next frame; otherwise EndFrame replays. Only the replay calls citro3d.
    enum class OpType : uint8_t {
        LoadShader,
        SelectTexture,
        SelectTextureFb,
        SetSampler,
        DepthTestAndMask,
        ZmodeDecal,
        Viewport,
        Scissor,
        UseAlpha,
        PrimDepth,
        Draw,
        DrawToFramebuffer,
        CopyFramebuffer,
        Clear,
        Upload,
        FramebufferSize,
    };
    struct Op {
        OpType type;
        int args[10];
        float value;
        ShaderProgram* shader;
    };
    // One frame's recorded calls
    struct FrameRecord {
        std::vector<Op> ops;
        std::vector<float> floats;    // the vertices of the draws
        std::vector<uint8_t> pixels;  // the RGBA32 images of the uploads
        std::vector<uint32_t> deleted; // textures deleted this frame
        uint64_t recordTicks = 0;     // the interpreter's time on it
        uint64_t handoffWaitTicks = 0; // the interpreter waiting for the last replay to finish
        uint64_t gapTicks = 0;        // the renderer thread between the last frame's end and this one's start
        uint64_t vblankWaitTicks = 0; // of that, waiting for vblank
        uint32_t textureCount = 0;    // ids handed out by the frame's end
        int framebufferCount = 1;
    };
    Op& Record(OpType type);
    void BeginGpuFrame();
    void Replay(FrameRecord& frame);
    void FinishFrame(FrameRecord& frame);
    void ReplayLoop();
    void UploadTextureNow(uint32_t texId, const uint8_t* rgba32Buf, uint32_t width, uint32_t height);
    void SetFramebufferSize(int fbId, uint32_t width, uint32_t height);
    void UpdateGpuTest();
    void BindUnit(int unit, void* tex);
    void ApplyTextureUnits(const Citro3D::TevProgram& program);

    C3D_RenderTarget_tag* mTopTarget = nullptr;
    C3D_RenderTarget_tag* mCurrentTarget = nullptr;
    uint32_t mTargetWidth = 400; // logical size of what is being drawn to
    uint32_t mTargetHeight = 240;
    std::vector<Framebuffer> mFramebuffers = std::vector<Framebuffer>(1); // 0 is the screen
    int mBoundFramebuffer = -1; // framebuffer bound as texture 0, if any
    std::vector<ShaderProgram*> mShaders;
    ShaderProgram* mCurrentShader = nullptr;

    std::vector<Texture> mTextures;
    uint32_t mBoundTexture[2] = { 0, 0 };
    int mLastSelectedUnit = 0;
    void* mFogRamp = nullptr; // C3D_Tex
    // What each texture unit should hold (C3D_Tex) and what the GPU has. A unit is only switched
    // on for draws whose combiner reads it: the GPU fetches from every enabled unit for each pixel.
    void* mUnitTex[3] = {};
    void* mDummyTex = nullptr; // C3D_Tex: 8x8, for units 1 and 2, which citro3d cannot leave empty
    void* mUnitApplied[3] = {};
    bool mUnitStale[3] = {};

    float* mVbo = nullptr; // linear memory, refilled every frame
    size_t mVboUsed = 0;   // vertices
    uint32_t mDrawsSinceSubmit = 0; // draws still in the command list, not yet sent to the GPU
    FrameRecord mFrames[2];
    FrameRecord* mRec = &mFrames[0]; // being recorded; the other one is being replayed
    int mRecSelectedUnit = 0;       // the texture the game last selected, as it records
    uint32_t mRecBoundTexture[2] = { 0, 0 };
    uint32_t mRecTextureCount = 0;  // ids handed out; the replay grows mTextures to match
    int mRecFramebufferCount = 1;
    std::vector<uint32_t> mTexturesFreeable; // deleted before the frame the GPU is drawing

    // The replay thread, if there is one
    void* mReplayThread = nullptr;
    std::mutex mReplayMutex;
    std::condition_variable mReplayCv;
    FrameRecord* mReplayFrame = nullptr; // handed to the thread, until it is done with it
    bool mReplayQuit = false;
    volatile uint32_t* mPhase = &gRenderPhase; // where the replay reports its phase to the profiler
    size_t mVboSubmitted = 0;       // vertices already flushed from the CPU cache for the GPU
    bool mVboFullReported = false;

    bool mDepthTest = false;
    bool mDepthMask = true;
    bool mDecal = false;
    bool mUseAlpha = false;
    // The GPU state set by the last draw, so a draw that repeats it sends nothing
    const Citro3D::TevProgram* mLastTevProgram = nullptr;
    uint32_t mLastTevColors[6] = {};
    int mLastDepthKey = -1;
    int mLastBlendKey = -1;
    float mPrimDepth = 0.0f;
    FilteringMode mFilterMode = FILTER_LINEAR;

    // A test that turns one GPU cost off, to find what the GPU spends its time on: cycled with
    // SELECT + R
    enum class GpuTest { Off, TinyScissor, NoBlending, NoDraws, Count };
    GpuTest mGpuTest = GpuTest::Off;
    uint32_t mHeldKeys = 0;
    int mScissor[4] = { 0, 0, 400, 240 }; // what the game asked for: x, y, width, height

    // Statistics, printed every few seconds
    uint32_t mFrameCount = 0;
    uint32_t mFrameTris = 0;
    uint32_t mFrameDraws = 0;
    uint32_t mFrameUploads = 0;
    uint32_t mTexturesByBits[3] = {}; // textures uploaded as 8, 16 and 32 bits a pixel
    uint64_t mStatsTick = 0;
    uint32_t mFpsFrames = 0; // for the bottom screen, once a second
    uint64_t mFpsTick = 0;

    // Slow-frame report: one line for any frame that takes much longer than the game's rate
    uint64_t mFrameStartTick = 0;
    uint64_t mDrawTicksTotal = 0; // since the last summary
    uint64_t mDrawTicksMax = 0;
    uint64_t mWaitTicksTotal = 0;
    uint64_t mReplayTicksTotal = 0;
    uint64_t mHandoffTicksTotal = 0;
    uint64_t mGapTicksTotal = 0;
    uint64_t mVBlankTicksTotal = 0;
    uint64_t mMainEndTick = 0;        // when the renderer thread's EndFrame returned
    uint64_t mVBlankWaitAtEnd = 0;
    uint64_t mLastFrameEndTick = 0;
    uint64_t mUploadTicks = 0;
    uint32_t mFilesReadAtFrameEnd = 0;
    uint64_t mReadMicrosAtFrameEnd = 0;
    uint32_t mLoadedAtFrameEnd = 0;
    uint64_t mLoadMicrosAtFrameEnd = 0;
    uint32_t mImportedAtFrameEnd = 0;
    uint64_t mImportMicrosAtFrameEnd = 0;
};

} // namespace Fast

#endif
