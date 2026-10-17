#pragma once
#ifdef __3DS__

#include "gfx_rendering_api.h"

#include <cstdint>
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

    float* mVbo = nullptr; // linear memory, refilled every frame
    size_t mVboUsed = 0;   // vertices
    uint32_t mDrawsSinceSubmit = 0; // draws still in the command list, not yet sent to the GPU
    bool mVboFullReported = false;

    bool mDepthTest = false;
    bool mDepthMask = true;
    bool mDecal = false;
    bool mUseAlpha = false;
    float mPrimDepth = 0.0f;
    FilteringMode mFilterMode = FILTER_LINEAR;

    // Statistics, printed every few seconds
    uint32_t mFrameCount = 0;
    uint32_t mFrameTris = 0;
    uint32_t mFrameDraws = 0;
    uint32_t mFrameUploads = 0;
    uint64_t mStatsTick = 0;

    // Slow-frame report: one line for any frame that takes much longer than the game's rate
    uint64_t mFrameStartTick = 0;
    uint64_t mDrawTicksTotal = 0; // since the last summary
    uint64_t mDrawTicksMax = 0;
    uint64_t mWaitTicksTotal = 0;
    uint64_t mVertexMicrosAtSummary = 0;
    uint64_t mBackendMicrosAtSummary = 0;
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
