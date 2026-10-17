#ifdef __3DS__

#include "fast/backends/gfx_citro3d.h"

#include <cstdio>
#include <cstring>

// The build renames libultra's osGetTime; lift that while libctru declares its own
#pragma push_macro("osGetTime")
#undef osGetTime
#include <3ds.h>
#include <citro3d.h>
#pragma pop_macro("osGetTime")

#include "fast/interpreter.h"

// Render the RGBA8 target into the RGB8 screen framebuffer without scaling
#define DISPLAY_TRANSFER_FLAGS                                                                                    \
    (GX_TRANSFER_FLIP_VERT(0) | GX_TRANSFER_OUT_TILED(0) | GX_TRANSFER_RAW_COPY(0) |                              \
     GX_TRANSFER_IN_FORMAT(GX_TRANSFER_FMT_RGBA8) | GX_TRANSFER_OUT_FORMAT(GX_TRANSFER_FMT_RGB8) |                \
     GX_TRANSFER_SCALING(GX_TRANSFER_SCALE_NO))

namespace Fast {

static constexpr uint32_t kClearColor = 0x203040FF; // dark blue: frames are running
static constexpr double kStatsIntervalSeconds = 3.0;

GfxRenderingAPICitro3D::~GfxRenderingAPICitro3D() {
    for (auto* prg : mShaders) {
        delete prg;
    }
}

const char* GfxRenderingAPICitro3D::GetName() {
    return "citro3d";
}

int GfxRenderingAPICitro3D::GetMaxTextureSize() {
    return 1024; // PICA200 limit
}

GfxClipParameters GfxRenderingAPICitro3D::GetClipParameters() {
    return { false, false };
}

void GfxRenderingAPICitro3D::UnloadShader(ShaderProgram* oldPrg) {
}

void GfxRenderingAPICitro3D::LoadShader(ShaderProgram* newPrg) {
    mCurrentShader = newPrg;
}

void GfxRenderingAPICitro3D::ClearShaderCache() {
}

ShaderProgram* GfxRenderingAPICitro3D::CreateAndLoadNewShader(uint64_t shaderId0, uint64_t shaderId1) {
    CCFeatures features;
    gfx_cc_get_features(shaderId0, shaderId1, &features);

    auto* prg = new ShaderProgram();
    prg->shaderId0 = shaderId0;
    prg->shaderId1 = shaderId1;
    prg->numInputs = (uint8_t)features.numInputs;
    prg->usedTextures[0] = features.usedTextures[0];
    prg->usedTextures[1] = features.usedTextures[1];
    mShaders.push_back(prg);
    LoadShader(prg);
    return prg;
}

ShaderProgram* GfxRenderingAPICitro3D::LookupShader(uint64_t shaderId0, uint64_t shaderId1) {
    for (auto* prg : mShaders) {
        if (prg->shaderId0 == shaderId0 && prg->shaderId1 == shaderId1) {
            return prg;
        }
    }
    return nullptr;
}

void GfxRenderingAPICitro3D::ShaderGetInfo(ShaderProgram* prg, uint8_t* numInputs, bool usedTextures[2]) {
    *numInputs = prg->numInputs;
    usedTextures[0] = prg->usedTextures[0];
    usedTextures[1] = prg->usedTextures[1];
}

uint32_t GfxRenderingAPICitro3D::NewTexture() {
    return mNextTextureId++;
}

void GfxRenderingAPICitro3D::SelectTexture(int tile, uint32_t textureId) {
}

void GfxRenderingAPICitro3D::UploadTexture(const uint8_t* rgba32Buf, uint32_t width, uint32_t height) {
    mFrameUploads++;
}

void GfxRenderingAPICitro3D::SetSamplerParameters(int sampler, bool linear_filter, uint32_t cms, uint32_t cmt) {
}

void GfxRenderingAPICitro3D::SetDepthTestAndMask(bool depth_test, bool z_upd) {
}

void GfxRenderingAPICitro3D::SetZmodeDecal(bool decal) {
}

void GfxRenderingAPICitro3D::SetViewport(int x, int y, int width, int height) {
}

void GfxRenderingAPICitro3D::SetScissor(int x, int y, int width, int height) {
}

void GfxRenderingAPICitro3D::SetUseAlpha(bool useAlpha) {
}

void GfxRenderingAPICitro3D::DrawTriangles(float buf_vbo[], size_t buf_vbo_len, size_t buf_vbo_num_tris) {
    mFrameDraws++;
    mFrameTris += buf_vbo_num_tris;
}

void GfxRenderingAPICitro3D::Init() {
    C3D_Init(C3D_DEFAULT_CMDBUF_SIZE);
    // The top screen is 400x240 but its framebuffer is stored rotated, hence 240x400
    mTopTarget = C3D_RenderTargetCreate(240, 400, GPU_RB_RGBA8, GPU_RB_DEPTH24_STENCIL8);
    C3D_RenderTargetSetOutput(mTopTarget, GFX_TOP, GFX_LEFT, DISPLAY_TRANSFER_FLAGS);
    mStatsTick = svcGetSystemTick();
}

void GfxRenderingAPICitro3D::OnResize() {
}

void GfxRenderingAPICitro3D::StartFrame() {
    C3D_FrameBegin(C3D_FRAME_SYNCDRAW);
    C3D_RenderTargetClear(mTopTarget, C3D_CLEAR_ALL, kClearColor, 0);
    C3D_FrameDrawOn(mTopTarget);
}

void GfxRenderingAPICitro3D::EndFrame() {
    C3D_FrameEnd(0);

    mFrameCount++;
    const uint64_t now = svcGetSystemTick();
    const double elapsed = (double)(now - mStatsTick) / SYSCLOCK_ARM11;
    if (elapsed >= kStatsIntervalSeconds) {
        printf("frame %lu: %.1f fps, %lu draws, %lu tris, %lu uploads/frame\n", (unsigned long)mFrameCount,
               mFrameCount / elapsed, (unsigned long)mFrameDraws, (unsigned long)mFrameTris,
               (unsigned long)mFrameUploads);
        mFrameCount = 0;
        mStatsTick = now;
    }
    mFrameDraws = 0;
    mFrameTris = 0;
    mFrameUploads = 0;
}

void GfxRenderingAPICitro3D::FinishRender() {
}

int GfxRenderingAPICitro3D::CreateFramebuffer() {
    return mNextFramebufferId++;
}

void GfxRenderingAPICitro3D::UpdateFramebufferParameters(int fb_id, uint32_t width, uint32_t height,
                                                         uint32_t msaa_level, bool opengl_invertY, bool render_target,
                                                         bool has_depth_buffer, bool can_extract_depth) {
}

void GfxRenderingAPICitro3D::StartDrawToFramebuffer(int fbId, float noiseScale) {
}

void GfxRenderingAPICitro3D::CopyFramebuffer(int fbDstId, int fbSrcId, int srcX0, int srcY0, int srcX1, int srcY1,
                                             int dstX0, int dstY0, int dstX1, int dstY1) {
}

void GfxRenderingAPICitro3D::ClearFramebuffer(bool color, bool depth) {
}

void GfxRenderingAPICitro3D::ReadFramebufferToCPU(int fbId, uint32_t width, uint32_t height, uint16_t* rgba16Buf) {
    memset(rgba16Buf, 0, width * height * sizeof(uint16_t));
}

void GfxRenderingAPICitro3D::ResolveMSAAColorBuffer(int fbIdTarger, int fbIdSrc) {
}

std::unordered_map<std::pair<float, float>, uint16_t, hash_pair_ff>
GfxRenderingAPICitro3D::GetPixelDepth(int fb_id, const std::set<std::pair<float, float>>& coordinates) {
    std::unordered_map<std::pair<float, float>, uint16_t, hash_pair_ff> depths;
    for (const auto& coordinate : coordinates) {
        depths[coordinate] = 0;
    }
    return depths;
}

void* GfxRenderingAPICitro3D::GetFramebufferTextureId(int fbId) {
    return nullptr;
}

void GfxRenderingAPICitro3D::SelectTextureFb(int fbId) {
}

void GfxRenderingAPICitro3D::DeleteTexture(uint32_t texId) {
}

void GfxRenderingAPICitro3D::SetTextureFilter(FilteringMode mode) {
    mFilterMode = mode;
}

FilteringMode GfxRenderingAPICitro3D::GetTextureFilter() {
    return mFilterMode;
}

void GfxRenderingAPICitro3D::SetSrgbMode() {
}

ImTextureID GfxRenderingAPICitro3D::GetTextureById(int id) {
    return (ImTextureID)0;
}

void GfxRenderingAPICitro3D::SetCurrentPrimDepth(float depth) {
}

} // namespace Fast

#endif
