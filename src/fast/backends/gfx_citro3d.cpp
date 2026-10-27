#ifdef __3DS__

#include "fast/backends/gfx_citro3d.h"
#include "fast/backends/gfx_3ds.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <malloc.h>

// The build renames libultra's osGetTime; lift that while libctru declares its own
#pragma push_macro("osGetTime")
#undef osGetTime
#include <3ds.h>
#include <citro3d.h>
#pragma pop_macro("osGetTime")

#include "fast/interpreter.h"
#include "fast/lus_gbi.h"
#include "fast3d_shbin.h"
#include "ship/resource/archive/O2rArchive.h"
#include "ship/resource/ResourceManager.h"

// Render the RGBA8 target into the RGB8 screen framebuffer without scaling
#define DISPLAY_TRANSFER_FLAGS                                                                                    \
    (GX_TRANSFER_FLIP_VERT(0) | GX_TRANSFER_OUT_TILED(0) | GX_TRANSFER_RAW_COPY(0) |                              \
     GX_TRANSFER_IN_FORMAT(GX_TRANSFER_FMT_RGBA8) | GX_TRANSFER_OUT_FORMAT(GX_TRANSFER_FMT_RGB8) |                \
     GX_TRANSFER_SCALING(GX_TRANSFER_SCALE_NO))

// citro3d's context, whose first member is its GPU command queue (source/internal.h in citro3d
// 1.7.1). citro3d runs the queue only from the frame's end; starting it after the left eye lets the
// GPU draw that eye while the right one is built (see Replay).
extern "C" gxCmdQueue_s __C3D_Context;

namespace Fast {

// Set by the game for a scene with its own convergence (see Set3dsSceneConvergence); 0 when none
static std::atomic<float> sSceneConvergence{ 0.0f };

void Set3dsSceneConvergence(float convergence) {
    sSceneConvergence.store(convergence, std::memory_order_relaxed);
}

using namespace Citro3D;

// Texture decoding in the interpreter (interpreter.cpp), for the slow-frame report
extern std::atomic<uint32_t> gTexturesImported;
extern std::atomic<uint64_t> gTextureImportMicroseconds;

namespace {
// Set on the thread running a replay: the backend's methods then act instead of recording. Per
// thread, as the interpreter records the next frame meanwhile.
thread_local bool tReplaying = false;

// A sampling profiler: a thread on another core reads gRenderPhase every 200 us and counts what
// the renderer was doing. Reading it costs the renderer nothing, unlike timing each command. The
// replay thread, if there is one, reports in sReplayPhase.
uint32_t sPhaseSamples[RENDER_PHASE_COUNT] = {};
volatile uint32_t sReplayPhase = RENDER_PHASE_OUTSIDE_LIST;
uint32_t sReplaySamples[RENDER_PHASE_COUNT] = {};

void SamplePhases(void*) {
    for (;;) {
        svcSleepThread(200000);
        const uint32_t phase = gRenderPhase;
        if (phase < RENDER_PHASE_COUNT) {
            sPhaseSamples[phase]++;
        }
        const uint32_t replayPhase = sReplayPhase;
        if (replayPhase < RENDER_PHASE_COUNT) {
            sReplaySamples[replayPhase]++;
        }
    }
}

const char* PhaseName(uint32_t phase) {
    switch (phase) {
        case RENDER_PHASE_OUTSIDE_LIST:
            return "outside lists";
        case RENDER_PHASE_LIST_SETUP:
            return "list setup";
        case RENDER_PHASE_TEXTURES:
            return "textures";
        case RENDER_PHASE_BACKEND:
            return "backend";
        case RENDER_PHASE_BACKEND_SCAN:
            return "backend scan";
        case RENDER_PHASE_BACKEND_STATE:
            return "backend state";
        case RENDER_PHASE_BACKEND_VERTICES:
            return "backend vertices";
        case RENDER_PHASE_BACKEND_DRAW:
            return "backend draw call";
        case RENDER_PHASE_BACKEND_SUBMIT:
            return "backend submit";
        case RENDER_PHASE_TRI_STATE:
            return "triangle state";
        default: {
            const char* name = GfxGetOpcodeName((int8_t)phase);
            return name != nullptr ? name : "?";
        }
    }
}

// The phases that took the most samples since the last call, as a share of all samples
void PrintPhaseProfile(const char* label, const uint32_t* samples, uint32_t* atLastPrint) {
    uint32_t counts[RENDER_PHASE_COUNT];
    uint32_t total = 0;
    for (uint32_t i = 0; i < RENDER_PHASE_COUNT; i++) {
        const uint32_t now = samples[i];
        counts[i] = now - atLastPrint[i];
        atLastPrint[i] = now;
        total += counts[i];
    }
    if (total == 0) {
        return;
    }
    char line[512];
    int len = snprintf(line, sizeof(line), "  %s, %lu samples:", label, (unsigned long)total);
    for (int shown = 0; shown < 10; shown++) {
        uint32_t best = 0;
        for (uint32_t i = 1; i < RENDER_PHASE_COUNT; i++) {
            if (counts[i] > counts[best]) {
                best = i;
            }
        }
        if (counts[best] == 0 || len >= (int)sizeof(line)) {
            break;
        }
        len += snprintf(line + len, sizeof(line) - len, " %s %.1f%%", PhaseName(best),
                        100.0 * counts[best] / total);
        counts[best] = 0;
    }
    printf("%s\n", line);
}

constexpr double kStatsIntervalSeconds = 3.0;
constexpr double kSlowFrameMs = 70.0; // the game runs at 20 to 30 fps, 50 ms at worst
constexpr int kMaxTevStages = 6;
constexpr int kMaxBufferUpdateStage = 3; // only stages 0-3 can write the combiner buffer
constexpr uint8_t kFogItem = 0xFE;      // ConstExpr leaf for the fog colour

// The top screen is 400x240, stored sideways as a 240x400 framebuffer
constexpr int kScreenWidth = 400;
constexpr int kScreenHeight = 240;

// Per vertex in our buffer: position, three texture coordinates (the third carries the fog
// factor), colour
constexpr int kVertexFloats = 4 + 2 + 2 + 2 + 4;
constexpr size_t kVboBytes = 1536 * 1024;
constexpr size_t kVboVertices = kVboBytes / (kVertexFloats * sizeof(float));

constexpr int kFogRampWidth = 256;

// Stereoscopic 3D at full slider: how far each eye's view of a distant point moves, as a share of
// half the screen's width (0.03 is 6 pixels), and how far out of the screen a close-up may come,
// as a multiple of that
constexpr float kMaxEyeShift = 0.03f;
constexpr float kPopOutCap = 1.0f;
// The convergence distance for G_EX_STEREO_OWN_CAMERA draws: the jigsaw transition's picture is at
// w = 150 (measured), so it starts at the screen's depth; its pieces fly in to about 50
constexpr float kOwnCameraConvergence = 150.0f;

// VRAM kept free for offscreen framebuffers' depth buffers, which are made when first drawn into
constexpr size_t kVramReserve = 1536 * 1024;
// Each VRAM upload is a copy in citro3d's GPU command queue, which holds 32 entries; a frame's
// clears, screen copies and transfers to the screens need the rest. A frame that loads a level
// uploads far more: past this, textures go to linear memory.
constexpr uint32_t kMaxVramUploadsPerFrame = 8;

size_t TextureBytes(uint32_t width, uint32_t height, GPU_TEXCOLOR fmt) {
    const size_t bits = fmt == GPU_RGBA8 ? 32 : (fmt == GPU_L8 || fmt == GPU_LA4) ? 8 : 16;
    return (size_t)width * height * bits / 8;
}

// Offscreen framebuffers are textures that hold a 240x400 sideways image, like the screen's
constexpr int kFbTexWidth = 256;
constexpr int kFbTexHeight = 512;
// Framebuffers and textures are both stored bottom-up, so a copied screen goes in the texture's
// last 400 rows for its rows to line up with the texture's (512 - 400) / 8 tile rows
constexpr int kCopyTileRowOffset = (kFbTexHeight - kScreenWidth) / 8;

DVLB_s* sShaderDvlb = nullptr;
shaderProgram_s sShaderProgram;

bool IsInput(int item) {
    return item >= SHADER_INPUT_1 && item <= SHADER_INPUT_7;
}

// An operand while a formula is being split into stages: a TEV source, or a constant expression
struct Term {
    bool isConst = false;
    int16_t constExpr = -1;
    Operand op;
};

// Builds the TEV stages for one channel of one combiner cycle: (a - b) * c + d
class ChannelCompiler {
  public:
    ChannelCompiler(TevProgram& program, std::vector<ChannelStage>& out, bool alphaChannel, int primaryInput,
                    bool secondCycle)
        : mProgram(program), mOut(out), mAlpha(alphaChannel), mPrimary(primaryInput), mSecondCycle(secondCycle) {
    }

    void Compile(int a, int b, int c, int d) {
        if (c == SHADER_0 || a == b) {
            Replace(Item(d));
            return;
        }
        Term A = Item(a), B = Item(b), C = Item(c), D = Item(d);
        if (A.isConst && B.isConst && C.isConst && D.isConst) {
            Replace(K(ConstExpr::Op::Add, K(ConstExpr::Op::Mul, K(ConstExpr::Op::Sub, A, B), C), D));
        } else if (b == SHADER_0 && d == SHADER_0) {
            Mul(A, C);
        } else if (b == d) {
            Lerp(A, B, C);
        } else if (b == SHADER_0) {
            MulAdd(A, C, D);
        } else if (d == SHADER_0) {
            if (A.isConst && B.isConst) {
                // A negative difference clamps to 0, as the product would
                Mul(K(ConstExpr::Op::Sub, A, B), C);
            } else {
                Stage(GPU_SUBTRACT, A, B);
                Mul(Prev(), C);
            }
        } else if (c == SHADER_1) {
            if (A.isConst && D.isConst) {
                Term sum = K(ConstExpr::Op::Add, A, D);
                if (B.isConst) {
                    Replace(K(ConstExpr::Op::Sub, sum, B));
                } else {
                    Stage(GPU_SUBTRACT, sum, B);
                }
            } else {
                Stage(GPU_ADD, A, D);
                Stage(GPU_SUBTRACT, Prev(), B);
            }
        } else if (B.isConst && C.isConst) {
            // a*c + d - b*c, with b*c folded on the CPU
            Term bc = K(ConstExpr::Op::Mul, B, C);
            MulAdd(A, C, D);
            Stage(GPU_SUBTRACT, Prev(), bc);
        } else {
            // a*c + d - b*c: b*c goes to the combiner buffer, readable two stages later
            Stage(GPU_MODULATE, B, C);
            mOut.back().updateBuffer = true;
            MulAdd(A, C, D);
            Stage(GPU_SUBTRACT, Prev(), Buffer());
        }
    }

  private:
    Term Item(int item) {
        Term t;
        switch (item) {
            case SHADER_TEXEL0:
            case SHADER_TEXEL0A:
            case SHADER_TEXEL1:
            case SHADER_TEXEL1A: {
                // The second cycle sees the textures swapped, as on the RDP
                bool first = item == SHADER_TEXEL0 || item == SHADER_TEXEL0A;
                t.op.src = (first != mSecondCycle) ? Src::Tex0 : Src::Tex1;
                t.op.alpha = mAlpha || item == SHADER_TEXEL0A || item == SHADER_TEXEL1A;
                return t;
            }
            case SHADER_COMBINED:
                t.op.src = Src::Combined;
                t.op.alpha = mAlpha;
                return t;
            default:
                break;
        }
        if (IsInput(item) && item - SHADER_INPUT_1 == mPrimary) {
            t.op.src = Src::Primary;
            t.op.alpha = mAlpha;
            return t;
        }
        // 0, 1, noise and the inputs that stay constant across the draw
        t.isConst = true;
        t.constExpr = Leaf((uint8_t)item);
        return t;
    }

    int16_t Leaf(uint8_t item) {
        for (size_t i = 0; i < mProgram.constExprs.size(); i++) {
            const ConstExpr& e = mProgram.constExprs[i];
            if (e.op == ConstExpr::Op::Leaf && e.item == item && e.alpha == mAlpha) {
                return (int16_t)i;
            }
        }
        ConstExpr e;
        e.item = item;
        e.alpha = mAlpha;
        mProgram.constExprs.push_back(e);
        return (int16_t)(mProgram.constExprs.size() - 1);
    }

    Term K(ConstExpr::Op op, const Term& a, const Term& b = Term()) {
        ConstExpr e;
        e.op = op;
        e.a = a.constExpr;
        e.b = b.constExpr;
        mProgram.constExprs.push_back(e);
        Term t;
        t.isConst = true;
        t.constExpr = (int16_t)(mProgram.constExprs.size() - 1);
        return t;
    }

    static Term Prev() {
        Term t;
        t.op.src = Src::Prev;
        return t;
    }

    static Term Buffer() {
        Term t;
        t.op.src = Src::Buffer;
        return t;
    }

    Term OneMinus(Term t) {
        if (t.isConst) {
            return K(ConstExpr::Op::OneMinus, t);
        }
        t.op.oneMinus = !t.op.oneMinus;
        return t;
    }

    static bool SameConst(const Term& a, const Term& b) {
        return a.isConst && b.isConst && a.constExpr == b.constExpr;
    }

    static int CountConsts(const Term& a, const Term& b, const Term& c) {
        int n = a.isConst + b.isConst + c.isConst;
        if (SameConst(a, b) || SameConst(a, c) || SameConst(b, c)) {
            n--;
        }
        return n;
    }

    void Stage(uint8_t func, const Term& a, const Term& b = Term(), const Term& c = Term(), uint8_t numOps = 0) {
        ChannelStage s;
        s.func = func;
        const Term* terms[3] = { &a, &b, &c };
        s.numOps = numOps != 0 ? numOps : (func == GPU_REPLACE ? 1 : (func == GPU_MULTIPLY_ADD || func == GPU_INTERPOLATE) ? 3 : 2);
        for (int i = 0; i < s.numOps; i++) {
            if (terms[i]->isConst) {
                s.ops[i].src = Src::Const;
                s.ops[i].constExpr = terms[i]->constExpr;
            } else {
                s.ops[i] = terms[i]->op;
            }
        }
        mOut.push_back(s);
    }

    void Replace(const Term& t) {
        Stage(GPU_REPLACE, t);
    }

    void Mul(const Term& a, const Term& b) {
        if (a.isConst && b.isConst) {
            Replace(K(ConstExpr::Op::Mul, a, b));
        } else {
            Stage(GPU_MODULATE, a, b);
        }
    }

    // a*b + c
    void MulAdd(const Term& a, const Term& b, const Term& c) {
        if (CountConsts(a, b, c) <= 1) {
            Stage(GPU_MULTIPLY_ADD, a, b, c);
        } else if (a.isConst && b.isConst) {
            Term ab = K(ConstExpr::Op::Mul, a, b);
            if (c.isConst) {
                Replace(K(ConstExpr::Op::Add, ab, c));
            } else {
                Stage(GPU_ADD, ab, c);
            }
        } else {
            Mul(a, b);
            Stage(GPU_ADD, Prev(), c);
        }
    }

    // a*f + b*(1-f)
    void Lerp(const Term& a, const Term& b, const Term& f) {
        if (CountConsts(a, b, f) <= 1) {
            Stage(GPU_INTERPOLATE, a, b, f);
        } else if (a.isConst && b.isConst && f.isConst) {
            Replace(K(ConstExpr::Op::Add, K(ConstExpr::Op::Mul, a, f), K(ConstExpr::Op::Mul, b, OneMinus(f))));
        } else if (a.isConst && b.isConst) {
            Stage(GPU_MODULATE, a, f);
            Stage(GPU_MULTIPLY_ADD, b, OneMinus(f), Prev());
        } else if (a.isConst) {
            MulAdd(b, OneMinus(f), K(ConstExpr::Op::Mul, a, f));
        } else {
            MulAdd(a, f, K(ConstExpr::Op::Mul, b, OneMinus(f)));
        }
    }

    TevProgram& mProgram;
    std::vector<ChannelStage>& mOut;
    bool mAlpha;
    int mPrimary;
    bool mSecondCycle;
};

GPU_TEVSRC ToTevSrc(Src src) {
    switch (src) {
        case Src::Const:
            return GPU_CONSTANT;
        case Src::Primary:
            return GPU_PRIMARY_COLOR;
        case Src::Tex0:
            return GPU_TEXTURE0;
        case Src::Tex1:
            return GPU_TEXTURE1;
        case Src::Tex2:
            return GPU_TEXTURE2;
        case Src::Buffer:
            return GPU_PREVIOUS_BUFFER;
        case Src::Prev:
        case Src::Combined:
        default:
            return GPU_PREVIOUS;
    }
}

GPU_TEVOP_RGB ToRgbOp(const Operand& op) {
    if (op.alpha) {
        return op.oneMinus ? GPU_TEVOP_RGB_ONE_MINUS_SRC_ALPHA : GPU_TEVOP_RGB_SRC_ALPHA;
    }
    return op.oneMinus ? GPU_TEVOP_RGB_ONE_MINUS_SRC_COLOR : GPU_TEVOP_RGB_SRC_COLOR;
}

GPU_TEVOP_A ToAlphaOp(const Operand& op) {
    return op.oneMinus ? GPU_TEVOP_A_ONE_MINUS_SRC_ALPHA : GPU_TEVOP_A_SRC_ALPHA;
}

uint8_t ToByte(float v) {
    if (!(v > 0.0f)) {
        return 0;
    }
    if (v >= 1.0f) {
        return 255;
    }
    return (uint8_t)(v * 255.0f + 0.5f);
}

// Position of pixel (x, y) inside an 8x8 PICA texture tile: the coordinate bits interleaved
// (Morton order), indexed by y * 8 + x
struct TileOrder {
    uint8_t pos[64];
    constexpr TileOrder() : pos() {
        for (int i = 0; i < 64; i++) {
            const int x = i & 7, y = i >> 3;
            pos[i] = (uint8_t)((x & 1) | ((y & 1) << 1) | ((x & 2) << 1) | ((y & 2) << 2) | ((x & 4) << 2) |
                               ((y & 4) << 3));
        }
    }
};
constexpr TileOrder kTileOrder;

// The channels of a pixel from the interpreter: RGBA bytes in memory, so R is the low byte
uint32_t R(uint32_t c) {
    return c & 0xFF;
}
uint32_t G(uint32_t c) {
    return (c >> 8) & 0xFF;
}
uint32_t B(uint32_t c) {
    return (c >> 16) & 0xFF;
}
uint32_t A(uint32_t c) {
    return c >> 24;
}

// Whether a channel survives being cut to a number of bits and expanded back by the GPU (which
// repeats the top bits). Off by one is let through: decoders that expand with a multiply round
// differently, and the difference doesn't show.
bool FitsBits(uint32_t v, int bits) {
    const uint32_t cut = v >> (8 - bits);
    uint32_t expanded = cut << (8 - bits);
    for (int shift = bits; shift < 8; shift += bits) {
        expanded |= (cut << (8 - bits)) >> shift;
    }
    return (expanded > v ? expanded - v : v - expanded) <= 1;
}

// The smallest texture format that holds an image without visible loss. N64 textures are mostly
// 16-bit colour or intensity (with or without alpha), so most fit in 8 or 16 bits a pixel, and the
// GPU then reads half or a quarter as much memory for every pixel it draws.
GPU_TEXCOLOR ChooseTextureFormat(const uint32_t* src, size_t numPixels) {
    bool gray = true, opaque = true, fits5551 = true, fits4 = true;
    for (size_t i = 0; i < numPixels && (gray || fits5551 || fits4); i++) {
        const uint32_t c = src[i];
        const uint32_t r = R(c), g = G(c), b = B(c), a = A(c);
        gray = gray && r == g && g == b;
        opaque = opaque && a == 255;
        fits5551 = fits5551 && (a == 0 || a == 255) && FitsBits(r, 5) && FitsBits(g, 5) && FitsBits(b, 5);
        fits4 = fits4 && FitsBits(r, 4) && FitsBits(g, 4) && FitsBits(b, 4) && FitsBits(a, 4);
    }
    if (gray) {
        if (opaque) {
            return GPU_L8;
        }
        return fits4 ? GPU_LA4 : GPU_LA8;
    }
    if (fits5551) {
        return GPU_RGBA5551;
    }
    if (fits4) {
        return GPU_RGBA4;
    }
    return GPU_RGBA8;
}

// Writes an image into a texture's 8x8 Morton-ordered tiles, padding to the texture's size by
// repeating the edge pixels
template <typename T, typename Convert>
void SwizzleTiles(const uint32_t* src, uint32_t width, uint32_t height, uint32_t texWidth, uint32_t texHeight,
                  void* data, Convert convert) {
    T* dst = (T*)data;
    for (uint32_t ty = 0; ty < texHeight; ty += 8) {
        for (uint32_t tx = 0; tx < texWidth; tx += 8) {
            for (int i = 0; i < 64; i++) {
                const uint32_t x = std::min(tx + (i & 7), width - 1);
                const uint32_t y = std::min(ty + (i >> 3), height - 1);
                dst[kTileOrder.pos[i]] = convert(src[y * width + x]);
            }
            dst += 64;
        }
    }
}

C3D_Tex* AsTex(void* p) {
    return static_cast<C3D_Tex*>(p);
}

uint32_t NextPow2(uint32_t v) {
    uint32_t p = 8;
    while (p < v) {
        p <<= 1;
    }
    return p;
}

} // namespace

GfxRenderingAPICitro3D::~GfxRenderingAPICitro3D() {
    if (mReplayThread != nullptr) {
        {
            std::lock_guard<std::mutex> lock(mReplayMutex);
            mReplayQuit = true;
        }
        mReplayCv.notify_all();
        Join3dsThread(mReplayThread);
    }
    for (auto* prg : mShaders) {
        delete prg;
    }
    for (auto& t : mTextures) {
        if (t.tex != nullptr) {
            C3D_TexDelete(AsTex(t.tex));
            delete AsTex(t.tex);
        }
    }
}

const char* GfxRenderingAPICitro3D::GetName() {
    return "citro3d";
}

int GfxRenderingAPICitro3D::GetMaxTextureSize() {
    return 1024; // PICA200 limit
}

GfxClipParameters GfxRenderingAPICitro3D::GetClipParameters() {
    // DrawRun maps z from [-w, w] to the PICA's [-w, 0] itself
    return { false, false };
}

void GfxRenderingAPICitro3D::UnloadShader(ShaderProgram* oldPrg) {
}

void GfxRenderingAPICitro3D::LoadShader(ShaderProgram* newPrg) {
    if (!tReplaying) {
        Record(OpType::LoadShader).shader = newPrg;
        return;
    }
    mCurrentShader = newPrg;
}

void GfxRenderingAPICitro3D::ClearShaderCache() {
}

ShaderProgram* GfxRenderingAPICitro3D::CreateAndLoadNewShader(uint64_t shaderId0, uint64_t shaderId1) {
    auto* prg = new ShaderProgram();
    prg->shaderId0 = shaderId0;
    prg->shaderId1 = shaderId1;
    gfx_cc_get_features(shaderId0, shaderId1, &prg->cc);
    const CCFeatures& cc = prg->cc;
    prg->numInputs = (uint8_t)cc.numInputs;
    prg->usedTextures[0] = cc.usedTextures[0];
    prg->usedTextures[1] = cc.usedTextures[1];

    for (int cycle = 0; cycle < (cc.opt_2cyc ? 2 : 1); cycle++) {
        for (int k = 0; k < 4; k++) {
            if (IsInput(cc.c[cycle][0][k])) {
                prg->inputUsedRgb[cc.c[cycle][0][k] - SHADER_INPUT_1] = true;
            }
            if (cc.opt_alpha && IsInput(cc.c[cycle][1][k])) {
                prg->inputUsedAlpha[cc.c[cycle][1][k] - SHADER_INPUT_1] = true;
            }
        }
    }

    // Mirrors the interpreter's vertex layout (Interpreter::GfxSpTri)
    int floats = 4;
    for (int t = 0; t < 2; t++) {
        if (cc.usedTextures[t]) {
            floats += 2 + (cc.clamp[t][0] ? 1 : 0) + (cc.clamp[t][1] ? 1 : 0);
        }
    }
    floats += cc.opt_fog ? 4 : 0;
    floats += cc.opt_grayscale ? 4 : 0;
    floats += cc.numInputs * (cc.opt_alpha ? 4 : 3);
    prg->numFloats = (uint8_t)floats;

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

const TevProgram& GfxRenderingAPICitro3D::GetTevProgram(ShaderProgram* prg, int primaryRgb, int primaryAlpha) {
    const uint32_t key = (uint32_t)(primaryRgb + 1) | ((uint32_t)(primaryAlpha + 1) << 4);
    auto it = prg->programs.find(key);
    if (it != prg->programs.end()) {
        return it->second;
    }

    TevProgram& program = prg->programs[key];
    const CCFeatures& cc = prg->cc;
    int cycle2Start = -1;
    for (int cycle = 0; cycle < (cc.opt_2cyc ? 2 : 1); cycle++) {
        std::vector<ChannelStage> rgb, alpha;
        ChannelCompiler(program, rgb, false, primaryRgb, cycle == 1)
            .Compile(cc.c[cycle][0][0], cc.c[cycle][0][1], cc.c[cycle][0][2], cc.c[cycle][0][3]);
        if (cc.opt_alpha) {
            ChannelCompiler(program, alpha, true, primaryAlpha, cycle == 1)
                .Compile(cc.c[cycle][1][0], cc.c[cycle][1][1], cc.c[cycle][1][2], cc.c[cycle][1][3]);
        } else {
            ChannelCompiler(program, alpha, true, -1, cycle == 1).Compile(SHADER_0, SHADER_0, SHADER_0, SHADER_1);
        }
        // Both channels share stages: pad the shorter one with pass-through stages
        while (alpha.size() < rgb.size()) {
            alpha.push_back(ChannelStage());
        }
        while (rgb.size() < alpha.size()) {
            rgb.push_back(ChannelStage());
        }
        if (cycle == 1) {
            cycle2Start = (int)program.rgb.size();
            // Keep the first cycle's result in the combiner buffer for later second-cycle stages
            program.rgb.back().updateBuffer = true;
            program.alpha.back().updateBuffer = true;
        }
        program.rgb.insert(program.rgb.end(), rgb.begin(), rgb.end());
        program.alpha.insert(program.alpha.end(), alpha.begin(), alpha.end());
    }

    // COMBINED is the previous stage's output for the first second-cycle stage, and the
    // buffered first-cycle result after that
    for (auto* channel : { &program.rgb, &program.alpha }) {
        for (int s = 0; s < (int)channel->size(); s++) {
            for (auto& op : (*channel)[s].ops) {
                if (op.src == Src::Combined) {
                    op.src = (s == cycle2Start) ? Src::Prev : Src::Buffer;
                }
            }
        }
    }

    if (cc.opt_fog) {
        // mix(colour, fog colour, fog factor): the factor arrives through the fog ramp texture
        ChannelStage fogRgb;
        fogRgb.func = GPU_INTERPOLATE;
        fogRgb.numOps = 3;
        ConstExpr fogLeaf;
        fogLeaf.item = kFogItem;
        program.constExprs.push_back(fogLeaf);
        fogRgb.ops[0].src = Src::Const;
        fogRgb.ops[0].constExpr = (int16_t)(program.constExprs.size() - 1);
        fogRgb.ops[1].src = Src::Prev;
        fogRgb.ops[2].src = Src::Tex2;
        fogRgb.ops[2].alpha = true;
        program.rgb.push_back(fogRgb);
        program.alpha.push_back(ChannelStage());
    }

    if ((int)program.rgb.size() > kMaxTevStages || (cycle2Start > kMaxBufferUpdateStage + 1)) {
        printf("citro3d: combiner %016llx/%016llx needs %d TEV stages\n", (unsigned long long)prg->shaderId0,
               (unsigned long long)prg->shaderId1, (int)program.rgb.size());
        program.rgb.resize(std::min<size_t>(program.rgb.size(), kMaxTevStages));
        program.alpha.resize(program.rgb.size());
    }
    return program;
}

void GfxRenderingAPICitro3D::ApplyTevProgram(const TevProgram& program, const float* inputs, bool hasFog,
                                             const float* fog) {
    // Work out every constant this draw needs, in order (operands always precede their users)
    float values[64][3];
    const size_t numExprs = std::min<size_t>(program.constExprs.size(), 64);
    for (size_t i = 0; i < numExprs; i++) {
        const ConstExpr& e = program.constExprs[i];
        float* v = values[i];
        switch (e.op) {
            case ConstExpr::Op::Leaf:
                if (e.item == SHADER_1) {
                    v[0] = v[1] = v[2] = 1.0f;
                } else if (e.item == SHADER_NOISE) {
                    v[0] = v[1] = v[2] = (float)(rand() & 0xFF) / 255.0f;
                } else if (e.item == kFogItem) {
                    v[0] = fog[0];
                    v[1] = fog[1];
                    v[2] = fog[2];
                } else if (IsInput(e.item)) {
                    const float* in = &inputs[(e.item - SHADER_INPUT_1) * 4];
                    if (e.alpha) {
                        v[0] = v[1] = v[2] = in[3];
                    } else {
                        v[0] = in[0];
                        v[1] = in[1];
                        v[2] = in[2];
                    }
                } else {
                    v[0] = v[1] = v[2] = 0.0f;
                }
                break;
            case ConstExpr::Op::Mul:
                for (int c = 0; c < 3; c++) {
                    v[c] = values[e.a][c] * values[e.b][c];
                }
                break;
            case ConstExpr::Op::Add:
                for (int c = 0; c < 3; c++) {
                    v[c] = values[e.a][c] + values[e.b][c];
                }
                break;
            case ConstExpr::Op::Sub:
                for (int c = 0; c < 3; c++) {
                    v[c] = values[e.a][c] - values[e.b][c];
                }
                break;
            case ConstExpr::Op::OneMinus:
                for (int c = 0; c < 3; c++) {
                    v[c] = 1.0f - values[e.a][c];
                }
                break;
        }
    }

    // Each stage's constant colour; a draw that repeats the last one's program and colours leaves
    // the GPU's combiner as it is
    const int numStages = std::min((int)program.rgb.size(), kMaxTevStages);
    uint32_t colors[kMaxTevStages] = {};
    for (int i = 0; i < numStages; i++) {
        const ChannelStage& rgb = program.rgb[i];
        const ChannelStage& alpha = program.alpha[i];
        uint8_t r = 0, g = 0, b = 0, a = 0;
        for (int k = 0; k < rgb.numOps; k++) {
            if (rgb.ops[k].src == Src::Const && rgb.ops[k].constExpr < (int)numExprs) {
                const float* v = values[rgb.ops[k].constExpr];
                r = ToByte(v[0]);
                g = ToByte(v[1]);
                b = ToByte(v[2]);
            }
        }
        for (int k = 0; k < alpha.numOps; k++) {
            if (alpha.ops[k].src == Src::Const && alpha.ops[k].constExpr < (int)numExprs) {
                a = ToByte(values[alpha.ops[k].constExpr][0]);
            }
        }
        colors[i] = (uint32_t)r | ((uint32_t)g << 8) | ((uint32_t)b << 16) | ((uint32_t)a << 24);
    }
    if (&program == mLastTevProgram && memcmp(colors, mLastTevColors, sizeof(colors)) == 0) {
        return;
    }
    mLastTevProgram = &program;
    memcpy(mLastTevColors, colors, sizeof(colors));

    int rgbBufferMask = 0;
    int alphaBufferMask = 0;
    for (int i = 0; i < kMaxTevStages; i++) {
        C3D_TexEnv* env = C3D_GetTexEnv(i);
        C3D_TexEnvInit(env);
        if (i >= numStages) {
            continue;
        }
        const ChannelStage& rgb = program.rgb[i];
        const ChannelStage& alpha = program.alpha[i];
        C3D_TexEnvSrc(env, C3D_RGB, ToTevSrc(rgb.ops[0].src), ToTevSrc(rgb.ops[1].src), ToTevSrc(rgb.ops[2].src));
        C3D_TexEnvOpRgb(env, ToRgbOp(rgb.ops[0]), ToRgbOp(rgb.ops[1]), ToRgbOp(rgb.ops[2]));
        C3D_TexEnvFunc(env, C3D_RGB, (GPU_COMBINEFUNC)rgb.func);
        C3D_TexEnvSrc(env, C3D_Alpha, ToTevSrc(alpha.ops[0].src), ToTevSrc(alpha.ops[1].src),
                      ToTevSrc(alpha.ops[2].src));
        C3D_TexEnvOpAlpha(env, ToAlphaOp(alpha.ops[0]), ToAlphaOp(alpha.ops[1]), ToAlphaOp(alpha.ops[2]));
        C3D_TexEnvFunc(env, C3D_Alpha, (GPU_COMBINEFUNC)alpha.func);

        C3D_TexEnvColor(env, colors[i]);

        if (i <= kMaxBufferUpdateStage) {
            rgbBufferMask |= rgb.updateBuffer ? (1 << i) : 0;
            alphaBufferMask |= alpha.updateBuffer ? (1 << i) : 0;
        }
    }
    C3D_TexEnvBufUpdate(C3D_RGB, rgbBufferMask);
    C3D_TexEnvBufUpdate(C3D_Alpha, alphaBufferMask);
}

void GfxRenderingAPICitro3D::BindUnit(int unit, void* tex) {
    mUnitTex[unit] = tex;
    // Rebind even the same texture: its filter or wrap may have changed
    mUnitStale[unit] = true;
}

void GfxRenderingAPICitro3D::ApplyTextureUnits(const TevProgram& program) {
    bool used[3] = {};
    for (size_t i = 0; i < program.rgb.size(); i++) {
        for (const ChannelStage* stage : { &program.rgb[i], &program.alpha[i] }) {
            for (int k = 0; k < stage->numOps; k++) {
                const Src src = stage->ops[k].src;
                used[0] |= src == Src::Tex0;
                used[1] |= src == Src::Tex1;
                used[2] |= src == Src::Tex2;
            }
        }
    }
    for (int unit = 0; unit < 3; unit++) {
        // Unit 0 can be switched off; units 1 and 2 get a tiny texture that stays in the GPU's cache
        void* want = used[unit] ? mUnitTex[unit] : (unit == 0 ? nullptr : mDummyTex);
        if (want != mUnitApplied[unit] || (want != nullptr && mUnitStale[unit])) {
            C3D_TexBind(unit, AsTex(want));
            mUnitApplied[unit] = want;
            if (want != nullptr) {
                mUnitStale[unit] = false;
            }
        }
    }
}

uint32_t GfxRenderingAPICitro3D::NewTexture() {
    return mRecTextureCount++;
}

void GfxRenderingAPICitro3D::SelectTexture(int tile, uint32_t textureId) {
    if (!tReplaying) {
        // Uploads go straight to the texture the game last selected
        mRecSelectedUnit = tile;
        if (tile <= 1) {
            mRecBoundTexture[tile] = textureId;
        }
        Op& op = Record(OpType::SelectTexture);
        op.args[0] = tile;
        op.args[1] = (int)textureId;
        return;
    }
    mLastSelectedUnit = tile;
    if (tile == 0) {
        mBoundFramebuffer = -1;
    }
    if (tile > 1) {
        return; // mask and blend textures (texture packs) are not supported
    }
    mBoundTexture[tile] = textureId;
    if (textureId < mTextures.size() && mTextures[textureId].tex != nullptr) {
        BindUnit(tile, AsTex(mTextures[textureId].tex));
    }
}

void GfxRenderingAPICitro3D::UploadTexture(const uint8_t* rgba32Buf, uint32_t width, uint32_t height) {
    // To the texture the game last selected, in order with the draws
    if (mRecSelectedUnit > 1 || width == 0 || height == 0) {
        return;
    }
    Op& op = Record(OpType::Upload);
    op.args[0] = (int)mRecBoundTexture[mRecSelectedUnit];
    op.args[1] = (int)width;
    op.args[2] = (int)height;
    op.args[3] = (int)mRec->pixels.size();
    mRec->pixels.insert(mRec->pixels.end(), rgba32Buf, rgba32Buf + width * height * 4);
}

void GfxRenderingAPICitro3D::UploadTextureNow(uint32_t texId, const uint8_t* rgba32Buf, uint32_t width,
                                              uint32_t height) {
    mFrameUploads++;
    const uint64_t uploadStart = svcGetSystemTick();
    struct AddTicks {
        uint64_t& total;
        uint64_t start;
        ~AddTicks() {
            total += svcGetSystemTick() - start;
        }
    } addTicks{ mUploadTicks, uploadStart };
    if (texId >= mTextures.size()) {
        return;
    }
    Texture& t = mTextures[texId];

    // The PICA needs power-of-two sizes of at least 8; pad by repeating the edge pixels
    const uint32_t texWidth = NextPow2(width);
    const uint32_t texHeight = NextPow2(height);
    if (texWidth > 1024 || texHeight > 1024) {
        return;
    }
    const GPU_TEXCOLOR fmt = ChooseTextureFormat((const uint32_t*)rgba32Buf, width * height);
    mTexturesByBits[fmt == GPU_RGBA8 ? 2 : (fmt == GPU_L8 || fmt == GPU_LA4) ? 0 : 1]++;
    const bool canCopy = mVramUploads < kMaxVramUploadsPerFrame;
    void* replaced = nullptr;
    if (t.tex != nullptr && (AsTex(t.tex)->width != texWidth || AsTex(t.tex)->height != texHeight ||
                             AsTex(t.tex)->fmt != fmt || (t.inVram && !canCopy))) {
        // The last frame's draws used it: free it once the GPU is done
        mTexObjectsFreeable.push_back(t.tex);
        mVramTextures -= t.inVram ? 1 : 0;
        replaced = t.tex;
        t.tex = nullptr;
    }
    if (t.tex == nullptr) {
        // VRAM while it lasts: the GPU samples it faster than linear memory
        C3D_Tex* tex = new C3D_Tex();
        t.inVram = canCopy && vramSpaceFree() >= TextureBytes(texWidth, texHeight, fmt) + kVramReserve &&
                   C3D_TexInitVRAM(tex, (u16)texWidth, (u16)texHeight, fmt);
        if (!t.inVram && !C3D_TexInit(tex, (u16)texWidth, (u16)texHeight, fmt)) {
            printf("citro3d: out of memory for a %lux%lu texture\n", (unsigned long)texWidth,
                   (unsigned long)texHeight);
            delete tex;
            return;
        }
        mVramTextures += t.inVram ? 1 : 0;
        C3D_TexSetFilter(tex, GPU_LINEAR, GPU_LINEAR);
        t.tex = tex;
        // A unit still holding the replaced texture takes the new one
        for (int unit = 0; unit < 3; unit++) {
            if (replaced != nullptr && mUnitTex[unit] == replaced) {
                BindUnit(unit, tex);
            }
        }
    }
    t.scaleS = (float)width / (float)texWidth;
    t.scaleT = (float)height / (float)texHeight;

    // The CPU writes a VRAM texture through a copy in linear memory, which the GPU's copy engine
    // moves over in order with the draws
    C3D_Tex* dst = AsTex(t.tex);
    if (t.inVram) {
        dst = new C3D_Tex();
        if (!C3D_TexInit(dst, (u16)texWidth, (u16)texHeight, fmt)) {
            printf("citro3d: out of memory for a %lux%lu texture upload\n", (unsigned long)texWidth,
                   (unsigned long)texHeight);
            delete dst;
            return;
        }
    }

    // Swizzle into 8x8 tiles, converting each pixel to the texture's format
    const uint32_t* src = (const uint32_t*)rgba32Buf;
    switch (fmt) {
        case GPU_RGBA8:
            // RGBA bytes to the PICA's ABGR word order
            SwizzleTiles<uint32_t>(src, width, height, texWidth, texHeight, dst->data,
                                   [](uint32_t c) { return __builtin_bswap32(c); });
            break;
        case GPU_RGBA5551:
            SwizzleTiles<uint16_t>(src, width, height, texWidth, texHeight, dst->data, [](uint32_t c) {
                return (uint16_t)(((R(c) >> 3) << 11) | ((G(c) >> 3) << 6) | ((B(c) >> 3) << 1) | (A(c) >> 7));
            });
            break;
        case GPU_RGBA4:
            SwizzleTiles<uint16_t>(src, width, height, texWidth, texHeight, dst->data, [](uint32_t c) {
                return (uint16_t)(((R(c) >> 4) << 12) | ((G(c) >> 4) << 8) | ((B(c) >> 4) << 4) | (A(c) >> 4));
            });
            break;
        case GPU_LA8:
            SwizzleTiles<uint16_t>(src, width, height, texWidth, texHeight, dst->data,
                                   [](uint32_t c) { return (uint16_t)((R(c) << 8) | A(c)); });
            break;
        case GPU_LA4:
            SwizzleTiles<uint8_t>(src, width, height, texWidth, texHeight, dst->data,
                                  [](uint32_t c) { return (uint8_t)(((R(c) >> 4) << 4) | (A(c) >> 4)); });
            break;
        case GPU_L8:
        default:
            SwizzleTiles<uint8_t>(src, width, height, texWidth, texHeight, dst->data,
                                  [](uint32_t c) { return (uint8_t)R(c); });
            break;
    }
    C3D_TexFlush(dst);
    if (dst != AsTex(t.tex)) {
        // Uploads come before the frame's draws (see Replay), so the copy lands before all of
        // them. The linear copy is freed once the GPU is done.
        const u32 bytes = (u32)TextureBytes(texWidth, texHeight, fmt);
        GX_TextureCopy((u32*)dst->data, 0, (u32*)AsTex(t.tex)->data, 0, bytes, GX_TRANSFER_RAW_COPY(1));
        mTexObjectsFreeable.push_back(dst);
        mVramUploads++;
    }
    // The replay's SelectTexture binds it
}

void GfxRenderingAPICitro3D::SetSamplerParameters(int sampler, bool linear_filter, uint32_t cms, uint32_t cmt) {
    if (!tReplaying) {
        Op& op = Record(OpType::SetSampler);
        op.args[0] = sampler;
        op.args[1] = linear_filter;
        op.args[2] = (int)cms;
        op.args[3] = (int)cmt;
        return;
    }
    if (sampler > 1) {
        return;
    }
    if (sampler == 0 && mBoundFramebuffer >= 0) {
        C3D_Tex* fbTex = AsTex(EyeTexture(mFramebuffers[mBoundFramebuffer]));
        const bool linear = linear_filter && mFilterMode != FILTER_NONE;
        C3D_TexSetFilter(fbTex, linear ? GPU_LINEAR : GPU_NEAREST, linear ? GPU_LINEAR : GPU_NEAREST);
        C3D_TexSetWrap(fbTex, GPU_CLAMP_TO_EDGE, GPU_CLAMP_TO_EDGE);
        BindUnit(0, fbTex);
        return;
    }
    if (mBoundTexture[sampler] >= mTextures.size()) {
        return;
    }
    Texture& t = mTextures[mBoundTexture[sampler]];
    if (t.tex == nullptr) {
        return;
    }
    auto wrap = [](uint32_t cm) {
        if (cm & G_TX_CLAMP) {
            return GPU_CLAMP_TO_EDGE;
        }
        return (cm & G_TX_MIRROR) ? GPU_MIRRORED_REPEAT : GPU_REPEAT;
    };
    const bool linear = linear_filter && mFilterMode != FILTER_NONE;
    C3D_TexSetFilter(AsTex(t.tex), linear ? GPU_LINEAR : GPU_NEAREST, linear ? GPU_LINEAR : GPU_NEAREST);
    C3D_TexSetWrap(AsTex(t.tex), wrap(cms), wrap(cmt));
    BindUnit(sampler, AsTex(t.tex));
}

void GfxRenderingAPICitro3D::SetDepthTestAndMask(bool depth_test, bool z_upd) {
    if (!tReplaying) {
        Op& op = Record(OpType::DepthTestAndMask);
        op.args[0] = depth_test;
        op.args[1] = z_upd;
        return;
    }
    mDepthTest = depth_test;
    mDepthMask = z_upd;
}

void GfxRenderingAPICitro3D::SetZmodeDecal(bool decal) {
    if (!tReplaying) {
        Record(OpType::ZmodeDecal).args[0] = decal;
        return;
    }
    mDecal = decal;
}

void GfxRenderingAPICitro3D::SetViewport(int x, int y, int width, int height) {
    if (!tReplaying) {
        Op& op = Record(OpType::Viewport);
        op.args[0] = x;
        op.args[1] = y;
        op.args[2] = width;
        op.args[3] = height;
        return;
    }
    // Screen x runs down the sideways framebuffer, from its far end
    C3D_SetViewport((u32)y, (u32)((int)mTargetWidth - x - width), (u32)height, (u32)width);
}

void GfxRenderingAPICitro3D::SetScissor(int x, int y, int width, int height) {
    if (!tReplaying) {
        Op& op = Record(OpType::Scissor);
        op.args[0] = x;
        op.args[1] = y;
        op.args[2] = width;
        op.args[3] = height;
        return;
    }
    mScissor[0] = x;
    mScissor[1] = y;
    mScissor[2] = width;
    mScissor[3] = height;
    if (mGpuTest == GpuTest::TinyScissor) {
        width = std::min(width, 8);
        height = std::min(height, 8);
    }
    const int w = (int)mTargetWidth;
    const int h = (int)mTargetHeight;
    const int left = std::clamp(y, 0, h);
    const int right = std::clamp(y + height, 0, h);
    const int top = std::clamp(w - x - width, 0, w);
    const int bottom = std::clamp(w - x, 0, w);
    C3D_SetScissor(GPU_SCISSOR_NORMAL, (u32)left, (u32)top, (u32)right, (u32)bottom);
}

void GfxRenderingAPICitro3D::SetUseAlpha(bool useAlpha) {
    if (!tReplaying) {
        Record(OpType::UseAlpha).args[0] = useAlpha;
        return;
    }
    mUseAlpha = useAlpha;
}

void GfxRenderingAPICitro3D::ApplyDepth() {
    const CCFeatures& cc = mCurrentShader->cc;
    // The PICA only writes depth while testing it, so "no test" is ALWAYS
    int writeMask = (cc.opt_invisible ? 0 : GPU_WRITE_COLOR) | (mDepthMask ? GPU_WRITE_DEPTH : 0);
    const int key = writeMask | (mDepthTest ? 0x100 : 0) | (mDecal ? 0x200 : 0);
    if (key == mLastDepthKey) {
        return;
    }
    mLastDepthKey = key;
    C3D_DepthTest(true, mDepthTest ? GPU_LEQUAL : GPU_ALWAYS, (GPU_WRITEMASK)writeMask);
    C3D_DepthMap(true, -1.0f, mDecal ? -0.0005f : 0.0f);
}

void GfxRenderingAPICitro3D::ApplyBlendAndAlphaTest() {
    const CCFeatures& cc = mCurrentShader->cc;
    const bool blend = mUseAlpha && mGpuTest != GpuTest::NoBlending;
    const int alphaTest = (cc.opt_texture_edge && cc.opt_alpha) ? 1 : (cc.opt_alpha_threshold && cc.opt_alpha) ? 2 : 0;
    const int key = (blend ? 1 : 0) | (alphaTest << 1);
    if (key == mLastBlendKey) {
        return;
    }
    mLastBlendKey = key;
    if (blend) {
        C3D_AlphaBlend(GPU_BLEND_ADD, GPU_BLEND_ADD, GPU_SRC_ALPHA, GPU_ONE_MINUS_SRC_ALPHA, GPU_SRC_ALPHA,
                       GPU_ONE_MINUS_SRC_ALPHA);
    } else {
        C3D_AlphaBlend(GPU_BLEND_ADD, GPU_BLEND_ADD, GPU_ONE, GPU_ZERO, GPU_ONE, GPU_ZERO);
    }
    if (alphaTest == 1) {
        C3D_AlphaTest(true, GPU_GREATER, 48); // 0.19
    } else if (alphaTest == 2) {
        C3D_AlphaTest(true, GPU_GEQUAL, 8);
    } else {
        C3D_AlphaTest(false, GPU_ALWAYS, 0);
    }
}

void GfxRenderingAPICitro3D::DrawRun(const float* buf, size_t firstTri, size_t numTris, int primaryRgb,
                                     int primaryAlpha) {
    ShaderProgram* prg = mCurrentShader;
    const CCFeatures& cc = prg->cc;
    const size_t numVerts = numTris * 3;
    if (mVboUsed + numVerts > kVboVertices) {
        if (!mVboFullReported) {
            printf("citro3d: vertex buffer full\n");
            mVboFullReported = true;
        }
        return;
    }

    // Offsets into the interpreter's vertex layout
    int texOff[2] = { -1, -1 };
    int off = 4;
    for (int t = 0; t < 2; t++) {
        if (cc.usedTextures[t]) {
            texOff[t] = off;
            off += 2 + (cc.clamp[t][0] ? 1 : 0) + (cc.clamp[t][1] ? 1 : 0);
        }
    }
    const int fogOff = cc.opt_fog ? off : -1;
    off += cc.opt_fog ? 4 : 0;
    off += cc.opt_grayscale ? 4 : 0;
    const int inputOff = off;
    const int inputFloats = cc.opt_alpha ? 4 : 3;
    const int stride = prg->numFloats;

    // Constant inputs come from the run's first vertex
    const float* first = buf + firstTri * 3 * stride;
    float inputs[7 * 4];
    for (int j = 0; j < prg->numInputs; j++) {
        const float* in = first + inputOff + j * inputFloats;
        inputs[j * 4 + 0] = in[0];
        inputs[j * 4 + 1] = in[1];
        inputs[j * 4 + 2] = in[2];
        inputs[j * 4 + 3] = cc.opt_alpha ? in[3] : 1.0f;
    }
    const float* fog = fogOff >= 0 ? first + fogOff : nullptr;

    *mPhase = RENDER_PHASE_BACKEND_STATE;
    const TevProgram& program = GetTevProgram(prg, primaryRgb, primaryAlpha);
    ApplyTevProgram(program, inputs, cc.opt_fog, fog);
    ApplyTextureUnits(program);
    ApplyDepth();
    ApplyBlendAndAlphaTest();

    const Texture* tex[2] = { nullptr, nullptr };
    for (int t = 0; t < 2; t++) {
        if (texOff[t] >= 0 && mBoundTexture[t] < mTextures.size()) {
            tex[t] = &mTextures[mBoundTexture[t]];
        }
    }
    // A framebuffer bound as texture 0 is sampled in its own sideways layout
    const Framebuffer* fbTex = (texOff[0] >= 0 && mBoundFramebuffer >= 0) ? &mFramebuffers[mBoundFramebuffer] : nullptr;
    ApplyEyeShift(first, fbTex != nullptr);

    *mPhase = RENDER_PHASE_BACKEND_VERTICES;
    float* dst = mVbo + mVboUsed * kVertexFloats;
    // The second eye draws the same vertices, which the first eye wrote
    for (size_t i = 0; i < (mSecondEye ? 0 : numVerts); i++) {
        const float* v = first + i * stride;
        const float x = v[0], y = v[1], z = v[2], w = v[3];
        // Rotate for the sideways screen; map z from [-w, w] to [-w, 0] (near to far)
        dst[0] = y;
        dst[1] = -x;
        dst[2] = cc.opt_prim_depth ? -mPrimDepth * w : -(z + w) * 0.5f;
        dst[3] = w;
        for (int t = 0; t < 2; t++) {
            if (t == 0 && fbTex != nullptr) {
                // Texture coordinates span the framebuffer's game-side size, top-down
                const float sx = v[texOff[0]] * (float)fbTex->width;
                const float sy = v[texOff[0] + 1] * (float)fbTex->height;
                dst[4] = ((float)fbTex->contentHeight - sy) / kFbTexWidth;
                dst[5] = ((float)fbTex->contentWidth - sx) / kFbTexHeight;
            } else if (tex[t] != nullptr) {
                dst[4 + t * 2] = v[texOff[t]] * tex[t]->scaleS;
                dst[5 + t * 2] = 1.0f - v[texOff[t] + 1] * tex[t]->scaleT;
            } else {
                dst[4 + t * 2] = 0.0f;
                dst[5 + t * 2] = 0.0f;
            }
        }
        if (fogOff >= 0) {
            dst[8] = (v[fogOff + 3] * (kFogRampWidth - 1) + 0.5f) / kFogRampWidth;
        } else {
            dst[8] = 0.0f;
        }
        dst[9] = 0.5f;
        const float* in = v + inputOff;
        if (primaryRgb >= 0) {
            dst[10] = in[primaryRgb * inputFloats + 0];
            dst[11] = in[primaryRgb * inputFloats + 1];
            dst[12] = in[primaryRgb * inputFloats + 2];
        } else {
            dst[10] = dst[11] = dst[12] = 1.0f;
        }
        dst[13] = primaryAlpha >= 0 ? in[primaryAlpha * inputFloats + 3] : 1.0f;
        dst += kVertexFloats;
    }

    *mPhase = RENDER_PHASE_BACKEND_DRAW;
    if (mGpuTest != GpuTest::NoDraws) {
        C3D_DrawArrays(GPU_TRIANGLES, (int)mVboUsed, (int)numVerts);
    }
    *mPhase = RENDER_PHASE_BACKEND;
    mVboUsed += numVerts;
    mFrameDraws += mSecondEye ? 0 : 1;
    mDrawsSinceSubmit++;
}

void GfxRenderingAPICitro3D::SubmitDraws() {
    if (mDrawsSinceSubmit == 0) {
        return;
    }
    const uint32_t prevPhase = *mPhase;
    *mPhase = RENDER_PHASE_BACKEND_SUBMIT;
    // The GPU reads the vertex buffer from memory, not the CPU's cache
    if (mVboUsed > mVboSubmitted) {
        svcFlushProcessDataCache(CUR_PROCESS_HANDLE, (u32)(mVbo + mVboSubmitted * kVertexFloats),
                                 (mVboUsed - mVboSubmitted) * kVertexFloats * sizeof(float));
        mVboSubmitted = mVboUsed;
    }
    // Flush only this part of the command list for the GPU (see FinishFrame)
    C3D_FrameSplit(GX_CMDLIST_FLUSH);
    mDrawsSinceSubmit = 0;
    *mPhase = prevPhase;
}

GfxRenderingAPICitro3D::FloatArena::~FloatArena() {
    free(data);
}

bool GfxRenderingAPICitro3D::FloatArena::Reserve(size_t extra) {
    if (size + extra <= capacity) {
        return true;
    }
    const size_t newCapacity = std::max({ capacity * 2, size + extra, (size_t)256 * 1024 });
    float* grown = (float*)realloc(data, newCapacity * sizeof(float));
    if (grown == nullptr) {
        return false;
    }
    data = grown;
    capacity = newCapacity;
    return true;
}

float* GfxRenderingAPICitro3D::GetVertexBuffer(size_t floats) {
    // The interpreter writes each batch straight into the frame's record, so the draw needs no copy
    FloatArena& arena = mRec->floats;
    return arena.Reserve(floats) ? arena.data + arena.size : nullptr;
}

void GfxRenderingAPICitro3D::DrawTriangles(float buf_vbo[], size_t buf_vbo_len, size_t buf_vbo_num_tris) {
    if (!tReplaying) {
        // Keep the vertices for the replay: the interpreter wrote them into the record (see
        // GetVertexBuffer) unless it had to use its own buffer
        FloatArena& floats = mRec->floats;
        if (buf_vbo != floats.data + floats.size) {
            // Not written where GetVertexBuffer said: keep a copy
            if (!floats.Reserve(buf_vbo_len)) {
                return;
            }
            memcpy(floats.data + floats.size, buf_vbo, buf_vbo_len * sizeof(float));
        }
        Op& op = Record(OpType::Draw);
        op.args[0] = (int)floats.size;
        op.args[1] = (int)buf_vbo_len;
        op.args[2] = (int)buf_vbo_num_tris;
        floats.size += buf_vbo_len;
        return;
    }
    ShaderProgram* prg = mCurrentShader;
    if (prg == nullptr || buf_vbo_num_tris == 0) {
        return;
    }
    mFrameTris += mSecondEye ? 0 : buf_vbo_num_tris;
    const CCFeatures& cc = prg->cc;
    const int stride = prg->numFloats;
    const int inputFloats = cc.opt_alpha ? 4 : 3;
    const int inputOff = stride - prg->numInputs * inputFloats;
    const size_t numVerts = buf_vbo_num_tris * 3;

    *mPhase = RENDER_PHASE_BACKEND_SCAN;
    // Which inputs vary across the batch: one per channel can be the vertex colour
    bool varyRgb[7] = {}, varyAlpha[7] = {};
    for (int j = 0; j < prg->numInputs; j++) {
        const float* in0 = buf_vbo + inputOff + j * inputFloats;
        for (size_t i = 1; i < numVerts && !(varyRgb[j] && (varyAlpha[j] || !cc.opt_alpha)); i++) {
            const float* in = buf_vbo + i * stride + inputOff + j * inputFloats;
            if (in[0] != in0[0] || in[1] != in0[1] || in[2] != in0[2]) {
                varyRgb[j] = true;
            }
            if (cc.opt_alpha && in[3] != in0[3]) {
                varyAlpha[j] = true;
            }
        }
    }
    *mPhase = RENDER_PHASE_BACKEND;
    int primaryRgb = -1, primaryAlpha = -1;
    int extraVarying = 0;
    for (int j = 0; j < prg->numInputs; j++) {
        if (prg->inputUsedRgb[j] && varyRgb[j]) {
            if (primaryRgb < 0) {
                primaryRgb = j;
            } else {
                extraVarying++;
            }
        }
        if (prg->inputUsedAlpha[j] && varyAlpha[j]) {
            if (primaryAlpha < 0) {
                primaryAlpha = j;
            } else {
                extraVarying++;
            }
        }
    }

    if (extraVarying == 0) {
        DrawRun(buf_vbo, 0, buf_vbo_num_tris, primaryRgb, primaryAlpha);
        return;
    }

    // More than one input varies: split into runs of triangles whose other inputs match, each
    // taking those inputs from its first vertex
    auto sameConstants = [&](size_t triA, size_t triB) {
        const float* a = buf_vbo + triA * 3 * stride + inputOff;
        const float* b = buf_vbo + triB * 3 * stride + inputOff;
        for (int j = 0; j < prg->numInputs; j++) {
            for (int c = 0; c < inputFloats; c++) {
                const bool isAlpha = c == 3;
                if (isAlpha ? (j == primaryAlpha || !prg->inputUsedAlpha[j])
                            : (j == primaryRgb || !prg->inputUsedRgb[j])) {
                    continue;
                }
                if (a[j * inputFloats + c] != b[j * inputFloats + c]) {
                    return false;
                }
            }
        }
        return true;
    };
    size_t runStart = 0;
    for (size_t tri = 1; tri <= buf_vbo_num_tris; tri++) {
        if (tri == buf_vbo_num_tris || !sameConstants(runStart, tri)) {
            DrawRun(buf_vbo, runStart, tri - runStart, primaryRgb, primaryAlpha);
            runStart = tri;
        }
    }
}

void GfxRenderingAPICitro3D::Init() {
    C3D_Init(C3D_DEFAULT_CMDBUF_SIZE * 4);
    // The top screen is 400x240 but its framebuffer is stored rotated, hence 240x400
    mTopTarget = C3D_RenderTargetCreate(kScreenHeight, kScreenWidth, GPU_RB_RGBA8, GPU_RB_DEPTH24_STENCIL8);
    C3D_RenderTargetSetOutput(mTopTarget, GFX_TOP, GFX_LEFT, DISPLAY_TRANSFER_FLAGS);
    mCurrentTarget = mTopTarget;
    mEyeTarget = mTopTarget;
    mRightTarget = C3D_RenderTargetCreate(kScreenHeight, kScreenWidth, GPU_RB_RGBA8, GPU_RB_DEPTH24_STENCIL8);
    if (mRightTarget != nullptr) {
        C3D_RenderTargetSetOutput(mRightTarget, GFX_TOP, GFX_RIGHT, DISPLAY_TRANSFER_FLAGS);
    }
    printf("3D %s, VRAM %lu KB free\n", mRightTarget != nullptr ? "ready" : "unavailable",
           (unsigned long)(vramSpaceFree() / 1024));

    sShaderDvlb = DVLB_ParseFile((u32*)fast3d_shbin, fast3d_shbin_size);
    shaderProgramInit(&sShaderProgram);
    shaderProgramSetVsh(&sShaderProgram, &sShaderDvlb->DVLE[0]);
    C3D_BindProgram(&sShaderProgram);
    mStereoUniform = shaderInstanceGetUniformLocation(sShaderProgram.vertexShader, "stereo");

    C3D_AttrInfo* attrInfo = C3D_GetAttrInfo();
    AttrInfo_Init(attrInfo);
    AttrInfo_AddLoader(attrInfo, 0, GPU_FLOAT, 4); // position
    AttrInfo_AddLoader(attrInfo, 1, GPU_FLOAT, 2); // texcoord 0
    AttrInfo_AddLoader(attrInfo, 2, GPU_FLOAT, 2); // texcoord 1
    AttrInfo_AddLoader(attrInfo, 3, GPU_FLOAT, 2); // texcoord 2: fog factor
    AttrInfo_AddLoader(attrInfo, 4, GPU_FLOAT, 4); // colour

    mVbo = (float*)linearAlloc(kVboBytes);
    C3D_BufInfo* bufInfo = C3D_GetBufInfo();
    BufInfo_Init(bufInfo);
    BufInfo_Add(bufInfo, mVbo, kVertexFloats * sizeof(float), 5, 0x43210);

    // A ramp from 0 to 1 that turns texture coordinate 2 into the fog factor
    C3D_Tex* fogRamp = new C3D_Tex();
    mFogRamp = fogRamp;
    C3D_TexInit(fogRamp, kFogRampWidth, 8, GPU_RGBA8);
    uint32_t* ramp = (uint32_t*)fogRamp->data;
    for (int tx = 0; tx < kFogRampWidth; tx += 8) {
        for (int i = 0; i < 64; i++) {
            const uint32_t v = (uint32_t)((tx + (i & 7)) * 255 / (kFogRampWidth - 1));
            ramp[kTileOrder.pos[i]] = (v << 24) | (v << 16) | (v << 8) | v;
        }
        ramp += 64;
    }
    C3D_TexFlush(fogRamp);
    C3D_TexSetFilter(fogRamp, GPU_LINEAR, GPU_LINEAR);
    C3D_TexSetWrap(fogRamp, GPU_CLAMP_TO_EDGE, GPU_CLAMP_TO_EDGE);
    BindUnit(2, fogRamp);

    C3D_Tex* dummy = new C3D_Tex();
    mDummyTex = dummy;
    C3D_TexInit(dummy, 8, 8, GPU_RGBA8);
    memset(dummy->data, 0xFF, 8 * 8 * 4);
    C3D_TexFlush(dummy);

    C3D_CullFace(GPU_CULL_NONE);
    mStatsTick = svcGetSystemTick();

    // The profiler shares core 1 with the audio. An Old 3DS has no time to spare there, so it only
    // runs on a New 3DS
    if (!Is3dsNew() || Start3dsThreadOnCore(SamplePhases, nullptr, 1) == nullptr) {
        printf("No renderer profiler\n");
    }

    // The replay gets core 2, beside the game logic, which leaves most of it idle at 30 fps
    mPhase = &sReplayPhase;
    mReplayThread = Start3dsThreadOnCore([](void* self) { ((GfxRenderingAPICitro3D*)self)->ReplayLoop(); }, this, 2);
    if (mReplayThread == nullptr) {
        mPhase = &gRenderPhase;
    } else {
        // Ahead of the game thread: threads of equal priority on a core do not preempt each other,
        // and a long game update would hold up the frame being shown
        s32 priority = 0x30;
        svcGetThreadPriority(&priority, CUR_THREAD_HANDLE);
        svcSetThreadPriority(threadGetHandle((Thread)mReplayThread), priority - 1);
    }
    printf("Replay %s\n", mReplayThread != nullptr ? "on core 2" : "in the renderer thread");
}

void GfxRenderingAPICitro3D::OnResize() {
}

void GfxRenderingAPICitro3D::StartFrame() {
    // Record this frame's drawing while the GPU is still busy with the last one
    mFrameStartTick = svcGetSystemTick();
    // The 3D slider, read once per frame so the culling margin matches the frame's eye shift
    const float eyeShift = mRightTarget != nullptr ? osGet3DSliderState() * kMaxEyeShift : 0.0f;
    mRec->eyeShift = eyeShift;
    gClipScaleX = 1.0f + eyeShift * std::max(1.0f, kPopOutCap) + 0.01f;
    if (mMainEndTick != 0) {
        mRec->gapTicks = mFrameStartTick - mMainEndTick;
        const uint64_t vblankWait = g3dsVBlankWaitTicks;
        mRec->vblankWaitTicks = vblankWait - mVBlankWaitAtEnd;
        mVBlankWaitAtEnd = vblankWait;
    }
}

void GfxRenderingAPICitro3D::BeginGpuFrame() {
    const uint64_t beginTick = svcGetSystemTick();
    // Wait for the last frame's GPU work (the vertex buffer and command list are reused). Not
    // SYNCDRAW: that also waits for the display's next vblank, which the game's own frame timer is
    // not in step with. The screen still only flips during vblank.
    C3D_FrameBegin(0);
    mWaitTicksTotal += svcGetSystemTick() - beginTick;
    // The GPU is done with the last frame: free the textures deleted before it was sent
    for (uint32_t texId : mTexturesFreeable) {
        mVramTextures -= mTextures[texId].inVram ? 1 : 0;
        ForgetTexObject(mTextures[texId].tex);
        C3D_TexDelete(AsTex(mTextures[texId].tex));
        delete AsTex(mTextures[texId].tex);
        mTextures[texId].tex = nullptr;
    }
    mTexturesFreeable.clear();
    for (void* tex : mTexObjectsFreeable) {
        ForgetTexObject(tex);
        C3D_TexDelete(AsTex(tex));
        delete AsTex(tex);
    }
    mTexObjectsFreeable.clear();
    mVramUploads = 0;
}

void GfxRenderingAPICitro3D::ForgetTexObject(void* tex) {
    // A unit left holding a freed texture would hand the GPU freed memory on its next draw
    for (int unit = 0; unit < 3; unit++) {
        if (mUnitTex[unit] == tex) {
            mUnitTex[unit] = unit == 0 ? nullptr : mDummyTex;
            mUnitStale[unit] = true;
        }
        if (mUnitApplied[unit] == tex) {
            mUnitApplied[unit] = &mUnitApplied; // unknown: rebind on the next draw
        }
        if (mEyeStart.unitTex[unit] == tex) {
            mEyeStart.unitTex[unit] = mUnitTex[unit];
        }
    }
}

void GfxRenderingAPICitro3D::Replay(FrameRecord& frame, C3D_RenderTarget_tag* target, int eyeSign, bool secondEye) {
    if (!secondEye) {
        if (mTextures.size() < frame.textureCount) {
            mTextures.resize(frame.textureCount);
        }
        if ((int)mFramebuffers.size() < frame.framebufferCount) {
            mFramebuffers.resize(frame.framebufferCount);
        }
        mEyeStart = { mCurrentShader, { mBoundTexture[0], mBoundTexture[1] }, mBoundFramebuffer, mLastSelectedUnit,
                      { mUnitTex[0], mUnitTex[1], mUnitTex[2] }, mPrimDepth, mDepthTest, mDepthMask, mDecal,
                      mUseAlpha, mStereoOwnCamera, { mScissor[0], mScissor[1], mScissor[2], mScissor[3] } };
        mVboUsed = 0;
        mVboSubmitted = 0;
    } else {
        // Start from the state the first eye started from, and draw from its vertices, already
        // flushed for the GPU
        const EyeStartState& st = mEyeStart;
        mCurrentShader = st.shader;
        mBoundTexture[0] = st.boundTexture[0];
        mBoundTexture[1] = st.boundTexture[1];
        mBoundFramebuffer = st.boundFramebuffer;
        mLastSelectedUnit = st.lastSelectedUnit;
        for (int unit = 0; unit < 3; unit++) {
            mUnitTex[unit] = st.unitTex[unit];
            mUnitStale[unit] = true;
        }
        mPrimDepth = st.primDepth;
        mDepthTest = st.depthTest;
        mDepthMask = st.depthMask;
        mDecal = st.decal;
        mUseAlpha = st.useAlpha;
        mStereoOwnCamera = st.stereoOwnCamera;
        SubmitDraws(); // flushes the first eye's last vertices
        mVboSubmitted = mVboUsed;
        mVboUsed = 0;
        // Start the GPU on the first eye now; what is added from here on starts as it arrives.
        // The queue is never stopped while it runs: citro3d stops it at the next frame's start,
        // once it has finished (libctru loses count of commands still running when stopped).
        gxCmdQueueRun(&__C3D_Context);
    }
    C3D_FrameDrawOn(target);
    mEyeTarget = target;
    mCurrentTarget = target;
    mTargetWidth = kScreenWidth;
    mTargetHeight = kScreenHeight;
    mEyeSign = eyeSign;
    mEyeShift = frame.eyeShift;
    mSecondEye = secondEye;
    mLastEyeKind = -1;
    mLastTevProgram = nullptr;
    mLastDepthKey = -1;
    mLastBlendKey = -1;

    if (!secondEye) {
        // Find the frame's last depth-tested draw on the screen (see ApplyEyeShift)
        bool depthTest = mDepthTest;
        bool onScreen = mCurrentTarget == target;
        mHudFromOp = 0;
        for (size_t i = 0; i < frame.ops.size(); i++) {
            const Op& op = frame.ops[i];
            if (op.type == OpType::DepthTestAndMask) {
                depthTest = op.args[0] != 0;
            } else if (op.type == OpType::DrawToFramebuffer) {
                onScreen = op.args[0] <= 0 || op.args[0] >= (int)mFramebuffers.size() ||
                           mFramebuffers[op.args[0]].tex == nullptr;
            } else if (op.type == OpType::Draw && depthTest && onScreen) {
                mHudFromOp = i + 1;
            }
        }
    }

    tReplaying = true;
    if (!secondEye) {
        // Texture uploads first: each VRAM upload is then one queued copy ahead of every draw,
        // rather than a copy plus a send of the draws before it. Every draw sees the frame's final
        // image of a texture, as with linear memory, where the GPU reads it only once the frame ends.
        for (const Op& op : frame.ops) {
            if (op.type == OpType::Upload) {
                const int* a = op.args;
                UploadTextureNow((uint32_t)a[0], frame.pixels.data() + a[3], (uint32_t)a[1], (uint32_t)a[2]);
            }
        }
    }
    if (secondEye) {
        SetScissor(mEyeStart.scissor[0], mEyeStart.scissor[1], mEyeStart.scissor[2], mEyeStart.scissor[3]);
    }
    for (mOpIndex = 0; mOpIndex < frame.ops.size(); mOpIndex++) {
        const Op& op = frame.ops[mOpIndex];
        const int* a = op.args;
        switch (op.type) {
            case OpType::LoadShader:
                LoadShader(op.shader);
                break;
            case OpType::SelectTexture:
                SelectTexture(a[0], (uint32_t)a[1]);
                break;
            case OpType::SelectTextureFb:
                SelectTextureFb(a[0]);
                break;
            case OpType::SetSampler:
                SetSamplerParameters(a[0], a[1] != 0, (uint32_t)a[2], (uint32_t)a[3]);
                break;
            case OpType::DepthTestAndMask:
                SetDepthTestAndMask(a[0] != 0, a[1] != 0);
                break;
            case OpType::ZmodeDecal:
                SetZmodeDecal(a[0] != 0);
                break;
            case OpType::Viewport:
                SetViewport(a[0], a[1], a[2], a[3]);
                break;
            case OpType::Scissor:
                SetScissor(a[0], a[1], a[2], a[3]);
                break;
            case OpType::UseAlpha:
                SetUseAlpha(a[0] != 0);
                break;
            case OpType::PrimDepth:
                SetCurrentPrimDepth(op.value);
                break;
            case OpType::Draw:
                DrawTriangles(frame.floats.data + a[0], (size_t)a[1], (size_t)a[2]);
                break;
            case OpType::DrawToFramebuffer:
                StartDrawToFramebuffer(a[0], 1.0f);
                break;
            case OpType::CopyFramebuffer:
                CopyFramebuffer(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9]);
                break;
            case OpType::Clear:
                ClearFramebuffer(a[0] != 0, a[1] != 0);
                break;
            case OpType::Upload:
                break; // done before the draws
            case OpType::StereoOwnCamera:
                mStereoOwnCamera = a[0] != 0;
                break;
            case OpType::FramebufferSize:
                if (!secondEye) {
                    SetFramebufferSize(a[0], (uint32_t)a[1], (uint32_t)a[2]);
                }
                break;
        }
    }
    tReplaying = false;
    mSecondEye = false;
}

void GfxRenderingAPICitro3D::ApplyEyeShift(const float* firstVertex, bool sampledFramebuffer) {
    // Only the screen gets shifted: not offscreen framebuffers, nor a copy of the screen drawn
    // back (the pause menu's background), which is already this eye's image. The HUD is
    // rectangles and orthographic draws, which have w = 1, and what is drawn without depth testing
    // after the world: the game draws its counters and the pause menu in perspective.
    const bool screen = mCurrentTarget == mEyeTarget && !sampledFramebuffer;
    const bool hud = firstVertex[3] == 1.0f || (!mDepthTest && mOpIndex >= mHudFromOp);
    // Under the game's own camera, rectangles (w = 1) stay with the HUD
    const bool ownCamera = mStereoOwnCamera && firstVertex[3] != 1.0f;
    const int kind = (mEyeSign == 0 || !screen) ? 0 : ownCamera ? 3 : hud ? 1 : 2;
    if (ownCamera && screen && !mSecondEye) {
        mOwnCameraWMin = std::min(mOwnCameraWMin, firstVertex[3]);
        mOwnCameraWMax = std::max(mOwnCameraWMax, firstVertex[3]);
    }
    if (screen && !hud && !mSecondEye) {
        const float w = firstVertex[3];
        int band = 0;
        for (float limit = 100.0f; band < 7 && w >= limit; limit *= 2.0f) {
            band++;
        }
        mWorldDepths[band]++;
    }
    if (kind == mLastEyeKind) {
        return;
    }
    mLastEyeKind = kind;
    // The screen's x is the sideways framebuffer's -y; a distant point moves left for the left eye
    const float shift = -(float)mEyeSign * mEyeShift;
    if (kind == 2 || kind == 3) {
        const float scene = sSceneConvergence.load(std::memory_order_relaxed);
        const float convergence = kind == 3 ? kOwnCameraConvergence : (scene > 0.0f ? scene : mConvergence);
        C3D_FVUnifSet(GPU_VERTEX_SHADER, mStereoUniform, shift, -convergence, -kPopOutCap, 0.0f);
    } else if (kind == 1) {
        C3D_FVUnifSet(GPU_VERTEX_SHADER, mStereoUniform, 0.0f, 0.0f, 0.0f, shift * mHudDepth);
    } else {
        C3D_FVUnifSet(GPU_VERTEX_SHADER, mStereoUniform, 0.0f, 0.0f, 0.0f, 0.0f);
    }
}

void GfxRenderingAPICitro3D::UpdateStereoTuning() {
    // A hidden tuning control for 3D: hold SELECT and tap the bottom screen's top half to move the
    // convergence distance (left nearer, right farther), or its bottom half to move the HUD (left
    // out, right in)
    const bool held = (hidKeysHeld() & (KEY_TOUCH | KEY_SELECT)) == (KEY_TOUCH | KEY_SELECT);
    const bool tapped = held && !mTouchHeld;
    mTouchHeld = held;
    if (!tapped) {
        return;
    }
    touchPosition touch;
    hidTouchRead(&touch);
    const bool right = touch.px >= 160;
    // A scene with its own convergence tunes that one
    const float scene = sSceneConvergence.load(std::memory_order_relaxed);
    float convergence = scene > 0.0f ? scene : mConvergence;
    if (touch.py < 120) {
        convergence = std::clamp(convergence * (right ? 1.25f : 0.8f), 25.0f, 20000.0f);
        if (scene > 0.0f) {
            sSceneConvergence.store(convergence, std::memory_order_relaxed);
        } else {
            mConvergence = convergence;
        }
    } else {
        mHudDepth = std::clamp(mHudDepth + (right ? 0.25f : -0.25f), -2.0f, 2.0f);
    }
    char note[48];
    snprintf(note, sizeof(note), "3D: convergence %.0f, HUD %+.2f", convergence, mHudDepth);
    printf("%s\n", note);
    Update3dsStatusNote(note);
}

void GfxRenderingAPICitro3D::PromoteTextures() {
    // Move textures that went to linear memory, in a frame with too many uploads, to VRAM with
    // what is left of this frame's copies. The copies run after this frame's draws; the linear
    // copy is freed once the GPU is done with it.
    constexpr uint32_t kChecksPerFrame = 64;
    for (uint32_t checked = 0; checked < kChecksPerFrame && mVramUploads < kMaxVramUploadsPerFrame &&
                               !mTextures.empty();
         checked++) {
        mPromoteCursor = mPromoteCursor + 1 < mTextures.size() ? mPromoteCursor + 1 : 0;
        Texture& t = mTextures[mPromoteCursor];
        if (t.tex == nullptr || t.inVram) {
            continue;
        }
        C3D_Tex* linear = AsTex(t.tex);
        const size_t bytes = TextureBytes(linear->width, linear->height, linear->fmt);
        if (vramSpaceFree() < bytes + kVramReserve) {
            continue; // a smaller one may still fit
        }
        C3D_Tex* vram = new C3D_Tex();
        if (!C3D_TexInitVRAM(vram, linear->width, linear->height, linear->fmt)) {
            delete vram;
            continue;
        }
        vram->param = linear->param; // filter and wrap
        GX_TextureCopy((u32*)linear->data, 0, (u32*)vram->data, 0, (u32)bytes, GX_TRANSFER_RAW_COPY(1));
        mTexObjectsFreeable.push_back(linear);
        t.tex = vram;
        t.inVram = true;
        mVramTextures++;
        mVramUploads++;
        // A unit still holding the linear copy rebinds
        for (int unit = 0; unit < 3; unit++) {
            if (mUnitTex[unit] == linear) {
                BindUnit(unit, vram);
            }
        }
    }
}

// A wait for the other side of the main/replay handoff. After hours of play the two once waited
// on each other with the frame handed over: a wake-up went missing (libctru's condition variable,
// as far as could be told). So wait in short slices and check again; a missed wake-up then costs
// one slice instead of the game.
template <typename Ready> void GfxRenderingAPICitro3D::WaitForReplay(std::unique_lock<std::mutex>& lock, Ready ready) {
    while (!ready()) {
        if (mReplayCv.wait_for(lock, std::chrono::milliseconds(50)) == std::cv_status::timeout && ready()) {
            static int sMissed = 0;
            if (++sMissed <= 20) {
                printf("Renderer handoff: a wake-up was missed (%d so far)\n", sMissed);
                SPDLOG_WARN("Renderer handoff: a wake-up was missed ({} so far)", sMissed);
            }
        }
    }
}

void GfxRenderingAPICitro3D::ReplayLoop() {
    for (;;) {
        FrameRecord* frame;
        {
            std::unique_lock<std::mutex> lock(mReplayMutex);
            WaitForReplay(lock, [this] { return mReplayFrame != nullptr || mReplayQuit; });
            if (mReplayFrame == nullptr) {
                return;
            }
            frame = mReplayFrame;
        }
        *mPhase = RENDER_PHASE_LIST_SETUP;
        FinishFrame(*frame);
        *mPhase = RENDER_PHASE_OUTSIDE_LIST; // idle
        {
            std::lock_guard<std::mutex> lock(mReplayMutex);
            mReplayFrame = nullptr;
        }
        mReplayCv.notify_all();
    }
}

GfxRenderingAPICitro3D::Op& GfxRenderingAPICitro3D::Record(OpType type) {
    Op& op = mRec->ops.emplace_back();
    op.type = type;
    return op;
}

void GfxRenderingAPICitro3D::UpdateGpuTest() {
    // Fires once when both buttons are down, pressed in either order
    constexpr uint32_t kCombo = KEY_SELECT | KEY_R;
    const uint32_t held = hidKeysHeld() & kCombo;
    const bool fire = held == kCombo && mHeldKeys != kCombo;
    mHeldKeys = held;
    if (!fire) {
        return;
    }
    static const char* const kNames[] = { "GPU test off", "GPU test: tiny scissor", "GPU test: no blending",
                                          "GPU test: no draws" };
    mGpuTest = (GpuTest)(((int)mGpuTest + 1) % (int)GpuTest::Count);
    printf("%s\n", kNames[(int)mGpuTest]);
    Update3dsStatusNote(kNames[(int)mGpuTest]);
    // Start the next stats line fresh
    mStatsTick = svcGetSystemTick();
    mFrameCount = 0;
    mWaitTicksTotal = 0;
    mReplayTicksTotal = 0;
    mHandoffTicksTotal = 0;
    mGapTicksTotal = 0;
    mVBlankTicksTotal = 0;
    mDrawTicksTotal = 0;
    mDrawTicksMax = 0;
}

void GfxRenderingAPICitro3D::EndFrame() {
    FrameRecord& rec = *mRec;
    rec.recordTicks = svcGetSystemTick() - mFrameStartTick;
    rec.textureCount = mRecTextureCount;
    rec.framebufferCount = mRecFramebufferCount;
    if (mReplayThread == nullptr) {
        FinishFrame(rec);
    } else {
        // Hand the frame over once the replay thread is done with the last one, and record the
        // next one into that
        const uint64_t waitStart = svcGetSystemTick();
        {
            std::unique_lock<std::mutex> lock(mReplayMutex);
            WaitForReplay(lock, [this] { return mReplayFrame == nullptr; });
            rec.handoffWaitTicks = svcGetSystemTick() - waitStart;
            mReplayFrame = &rec;
            mRec = &rec == &mFrames[0] ? &mFrames[1] : &mFrames[0];
        }
        mReplayCv.notify_all();
    }
    mRec->ops.clear();
    mRec->floats.size = 0;
    mRec->pixels.clear();
    mRec->deleted.clear();
    mRec->handoffWaitTicks = 0;
    mRec->gapTicks = 0;
    mRec->vblankWaitTicks = 0;
    mMainEndTick = svcGetSystemTick();
}

void GfxRenderingAPICitro3D::FinishFrame(FrameRecord& frame) {
    const uint64_t replayStart = svcGetSystemTick();
    // Stereoscopic 3D when the slider is up: the frame is replayed once per eye. Otherwise the
    // screen shows one image, at the cost of one.
    const bool stereo = frame.eyeShift > 0.0f && mRightTarget != nullptr;
    gfxSet3D(stereo);
    BeginGpuFrame();
    Replay(frame, mTopTarget, stereo ? -1 : 0, false);
    if (stereo) {
        Replay(frame, mRightTarget, 1, true);
    }
    // Textures deleted this frame are freed once the GPU has drawn it
    mTexturesFreeable.insert(mTexturesFreeable.end(), frame.deleted.begin(), frame.deleted.end());
    SubmitDraws();
    PromoteTextures();
    // C3D_FrameEnd(0) would flush all of linear memory, tens of MB, from the CPU's cache, through
    // the GPU service on core 1. Flush only what the CPU wrote for the GPU or the screens: the
    // command list (here and in SubmitDraws), the vertex buffer (SubmitDraws), textures (on
    // upload), and the bottom screen's text console.
    u16 bottomWidth = 0;
    u16 bottomHeight = 0;
    u8* bottom = gfxGetFramebuffer(GFX_BOTTOM, GFX_LEFT, &bottomWidth, &bottomHeight);
    svcFlushProcessDataCache(CUR_PROCESS_HANDLE, (u32)bottom,
                             (u32)bottomWidth * bottomHeight * gspGetBytesPerPixel(gfxGetScreenFormat(GFX_BOTTOM)));
    C3D_FrameEnd(GX_CMDLIST_FLUSH);
    UpdateGpuTest();
    UpdateStereoTuning();

    mFrameCount++;
    const uint64_t now = svcGetSystemTick();

    // Report a slow frame: how long since the last one, how much of it went to drawing (from
    // StartFrame, which includes waiting for the previous frame's GPU work), and what loading
    // happened in between, on any thread
    const uint32_t filesRead = Ship::O2rArchive::sFilesRead.load();
    const uint64_t readMicros = Ship::O2rArchive::sReadMicroseconds.load();
    const uint32_t loaded = Ship::ResourceManager::sResourcesLoaded.load();
    const uint64_t loadMicros = Ship::ResourceManager::sLoadMicroseconds.load();
    const uint32_t imported = gTexturesImported.load();
    const uint64_t importMicros = gTextureImportMicroseconds.load();
    if (mLastFrameEndTick != 0) {
        const double toMs = 1000.0 / SYSCLOCK_ARM11;
        const double frameMs = (double)(now - mLastFrameEndTick) * toMs;
        if (frameMs > kSlowFrameMs) {
            printf("slow frame: %.0f ms (draw %.0f, gpu %.1f, %lu draws, %lu tris)\n"
                   "  loaded %lu resources in %.0f ms (reading %lu files: %.0f ms), "
                   "decoded %lu textures in %.0f ms (uploading: %.0f ms)\n",
                   frameMs, (double)frame.recordTicks * toMs, C3D_GetDrawingTime(),
                   (unsigned long)mFrameDraws, (unsigned long)mFrameTris,
                   (unsigned long)(loaded - mLoadedAtFrameEnd), (loadMicros - mLoadMicrosAtFrameEnd) / 1000.0,
                   (unsigned long)(filesRead - mFilesReadAtFrameEnd), (readMicros - mReadMicrosAtFrameEnd) / 1000.0,
                   (unsigned long)(imported - mImportedAtFrameEnd), (importMicros - mImportMicrosAtFrameEnd) / 1000.0,
                   (double)mUploadTicks * toMs);
        }
    }
    mFpsFrames++;
    if (mFpsTick == 0) {
        mFpsTick = now;
        mFpsFrames = 0;
    } else if (now - mFpsTick >= SYSCLOCK_ARM11) {
        Update3dsStatusFps(mFpsFrames * (float)SYSCLOCK_ARM11 / (now - mFpsTick));
        mFpsFrames = 0;
        mFpsTick = now;
    }
    const uint64_t drawTicks = frame.recordTicks;
    mDrawTicksTotal += drawTicks;
    mReplayTicksTotal += now - replayStart;
    mHandoffTicksTotal += frame.handoffWaitTicks;
    mGapTicksTotal += frame.gapTicks;
    mVBlankTicksTotal += frame.vblankWaitTicks;
    mDrawTicksMax = std::max(mDrawTicksMax, drawTicks);
    mLastFrameEndTick = now;
    mFilesReadAtFrameEnd = filesRead;
    mReadMicrosAtFrameEnd = readMicros;
    mLoadedAtFrameEnd = loaded;
    mLoadMicrosAtFrameEnd = loadMicros;
    mImportedAtFrameEnd = imported;
    mImportMicrosAtFrameEnd = importMicros;
    mUploadTicks = 0;
    const double elapsed = (double)(now - mStatsTick) / SYSCLOCK_ARM11;
    if (elapsed >= kStatsIntervalSeconds) {
        const double toMs = 1000.0 / SYSCLOCK_ARM11;
        const struct mallinfo heap = mallinfo();
        printf("%sframe %lu: %.1f fps, %lu draws, %lu tris, %lu uploads/frame, gpu %.1f ms, "
               "draw %.1f ms (worst %.0f), replay %.1f ms, waiting for the GPU %.1f ms, for the replay %.1f ms, "
               "between frames %.1f ms (vblank %.1f), "
               "heap %lu/%lu KB in use, linear %lu KB free\n",
               mGpuTest == GpuTest::Off ? "" : "[test] ", (unsigned long)mFrameCount, mFrameCount / elapsed,
               (unsigned long)mFrameDraws,
               (unsigned long)mFrameTris, (unsigned long)mFrameUploads, C3D_GetDrawingTime(),
               (double)mDrawTicksTotal * toMs / mFrameCount, (double)mDrawTicksMax * toMs,
               (double)mReplayTicksTotal * toMs / mFrameCount, (double)mWaitTicksTotal * toMs / mFrameCount,
               (double)mHandoffTicksTotal * toMs / mFrameCount, (double)mGapTicksTotal * toMs / mFrameCount,
               (double)mVBlankTicksTotal * toMs / mFrameCount, (unsigned long)(heap.uordblks / 1024),
               (unsigned long)(envGetHeapSize() / 1024), (unsigned long)(linearSpaceFree() / 1024));
        static uint32_t sRendererAtLastPrint[RENDER_PHASE_COUNT] = {};
        static uint32_t sReplayAtLastPrint[RENDER_PHASE_COUNT] = {};
        PrintPhaseProfile("renderer", sPhaseSamples, sRendererAtLastPrint);
        if (mReplayThread != nullptr) {
            PrintPhaseProfile("replay", sReplaySamples, sReplayAtLastPrint);
        }
        printf("  textures so far: %lu 8-bit, %lu 16-bit, %lu 32-bit; %lu in VRAM, %lu KB VRAM free\n",
               (unsigned long)mTexturesByBits[0], (unsigned long)mTexturesByBits[1],
               (unsigned long)mTexturesByBits[2], (unsigned long)mVramTextures,
               (unsigned long)(vramSpaceFree() / 1024));
        const uint32_t* d = mWorldDepths;
        const float scene = sSceneConvergence.load(std::memory_order_relaxed);
        printf("  3D: shift %.3f, convergence %.0f, HUD %+.2f; world draws by w: <100 %lu, <200 %lu, <400 %lu, "
               "<800 %lu, <1600 %lu, <3200 %lu, <6400 %lu, more %lu\n",
               frame.eyeShift, scene > 0.0f ? scene : mConvergence, mHudDepth, (unsigned long)d[0], (unsigned long)d[1],
               (unsigned long)d[2], (unsigned long)d[3], (unsigned long)d[4], (unsigned long)d[5],
               (unsigned long)d[6], (unsigned long)d[7]);
        memset(mWorldDepths, 0, sizeof(mWorldDepths));
        if (mOwnCameraWMax > 0.0f) {
            printf("  3D: own-camera draws at w %.0f to %.0f\n", mOwnCameraWMin, mOwnCameraWMax);
        }
        mOwnCameraWMin = 1e9f;
        mOwnCameraWMax = 0.0f;
        mWaitTicksTotal = 0;
        mReplayTicksTotal = 0;
        mHandoffTicksTotal = 0;
        mGapTicksTotal = 0;
        mVBlankTicksTotal = 0;
        mDrawTicksTotal = 0;
        mDrawTicksMax = 0;
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
    return mRecFramebufferCount++;
}

void GfxRenderingAPICitro3D::UpdateFramebufferParameters(int fb_id, uint32_t width, uint32_t height,
                                                         uint32_t msaa_level, bool opengl_invertY, bool render_target,
                                                         bool has_depth_buffer, bool can_extract_depth) {
    Op& op = Record(OpType::FramebufferSize);
    op.args[0] = fb_id;
    op.args[1] = (int)width;
    op.args[2] = (int)height;
}

void GfxRenderingAPICitro3D::SetFramebufferSize(int fb_id, uint32_t width, uint32_t height) {
    if (fb_id <= 0 || fb_id >= (int)mFramebuffers.size()) {
        return; // the screen keeps its size
    }
    Framebuffer& fb = mFramebuffers[fb_id];
    fb.width = width;
    fb.height = height;
    if (fb.tex == nullptr) {
        C3D_Tex* tex = new C3D_Tex();
        if (!C3D_TexInit(tex, kFbTexWidth, kFbTexHeight, GPU_RGBA8)) {
            printf("citro3d: out of memory for framebuffer %d\n", fb_id);
            delete tex;
            return;
        }
        memset(tex->data, 0, tex->size);
        C3D_TexFlush(tex);
        C3D_TexSetFilter(tex, GPU_LINEAR, GPU_LINEAR);
        C3D_TexSetWrap(tex, GPU_CLAMP_TO_EDGE, GPU_CLAMP_TO_EDGE);
        fb.tex = tex;
    }
}

void GfxRenderingAPICitro3D::StartDrawToFramebuffer(int fbId, float noiseScale) {
    if (!tReplaying) {
        Record(OpType::DrawToFramebuffer).args[0] = fbId;
        return;
    }
    if (fbId <= 0 || fbId >= (int)mFramebuffers.size() || mFramebuffers[fbId].tex == nullptr) {
        C3D_FrameDrawOn(mEyeTarget);
        mCurrentTarget = mEyeTarget;
        mTargetWidth = kScreenWidth;
        mTargetHeight = kScreenHeight;
        return;
    }
    Framebuffer& fb = mFramebuffers[fbId];
    if (fb.target == nullptr) {
        fb.target = C3D_RenderTargetCreateFromTex(AsTex(fb.tex), GPU_TEXFACE_2D, 0, GPU_RB_DEPTH24_STENCIL8);
        if (fb.target == nullptr) {
            printf("citro3d: cannot draw into framebuffer %d\n", fbId);
            return;
        }
    }
    C3D_FrameDrawOn(fb.target);
    mCurrentTarget = fb.target;
    mTargetWidth = std::min<uint32_t>(fb.width, kFbTexHeight);
    mTargetHeight = std::min<uint32_t>(fb.height, kFbTexWidth);
    fb.contentWidth = mTargetWidth;
    fb.contentHeight = mTargetHeight;
}

void GfxRenderingAPICitro3D::CopyFramebuffer(int fbDstId, int fbSrcId, int srcX0, int srcY0, int srcX1, int srcY1,
                                             int dstX0, int dstY0, int dstX1, int dstY1) {
    if (!tReplaying) {
        Op& op = Record(OpType::CopyFramebuffer);
        const int args[10] = { fbDstId, fbSrcId, srcX0, srcY0, srcX1, srcY1, dstX0, dstY0, dstX1, dstY1 };
        memcpy(op.args, args, sizeof(args));
        return;
    }
    // The game only copies the screen (pause background, transition snapshot), always whole
    if (fbSrcId != 0 || fbDstId <= 0 || fbDstId >= (int)mFramebuffers.size() ||
        mFramebuffers[fbDstId].tex == nullptr) {
        return;
    }
    Framebuffer& fb = mFramebuffers[fbDstId];
    // The game copies the screen once and draws the copy for as long as it needs it (the pause
    // menu's background): in 3D each eye keeps its own
    void* dstTex = fb.tex;
    if (mEyeSign > 0) {
        if (fb.rightTex == nullptr) {
            C3D_Tex* tex = new C3D_Tex();
            if (!C3D_TexInit(tex, kFbTexWidth, kFbTexHeight, GPU_RGBA8)) {
                printf("citro3d: out of memory for framebuffer %d's right eye\n", fbDstId);
                delete tex;
                return;
            }
            C3D_TexSetFilter(tex, GPU_LINEAR, GPU_LINEAR);
            C3D_TexSetWrap(tex, GPU_CLAMP_TO_EDGE, GPU_CLAMP_TO_EDGE);
            fb.rightTex = tex;
        }
        dstTex = fb.rightTex;
        fb.rightValid = true;
    } else if (mEyeSign == 0) {
        fb.rightValid = false;
    }
    // The copy engine runs from the GX queue, after anything already submitted: submit the
    // frame's draws so far first
    SubmitDraws();
    // Both are 8x8-tiled RGBA8, so copy tile row by tile row: 240 pixels of each 256-wide row
    C3D_FrameBuf& screen = mEyeTarget->frameBuf;
    const u32 rowBytes = (u32)screen.width * 8 * 4;
    const u32 texRowBytes = (u32)kFbTexWidth * 8 * 4;
    u8* dst = (u8*)AsTex(dstTex)->data + kCopyTileRowOffset * texRowBytes;
    GX_TextureCopy((u32*)screen.colorBuf, GX_BUFFER_DIM(rowBytes >> 4, 0), (u32*)dst,
                   GX_BUFFER_DIM(rowBytes >> 4, (texRowBytes - rowBytes) >> 4), rowBytes * (screen.height / 8),
                   GX_TRANSFER_RAW_COPY(1));
    fb.contentWidth = kScreenWidth;
    fb.contentHeight = kScreenHeight;
}

void GfxRenderingAPICitro3D::ClearFramebuffer(bool color, bool depth) {
    if (!tReplaying) {
        Op& op = Record(OpType::Clear);
        op.args[0] = color;
        op.args[1] = depth;
        return;
    }
    const int bits = (color ? C3D_CLEAR_COLOR : 0) | (depth ? C3D_CLEAR_DEPTH : 0);
    if (bits != 0) {
        // A clear is a memory fill queued straight to the GPU, while draws wait in the command
        // list until the frame ends; send them first, or a mid-frame depth clear (before the HUD,
        // say) lands before the world is drawn
        SubmitDraws();
        C3D_RenderTargetClear(mCurrentTarget, (C3D_ClearBits)bits, 0x000000FF, 0x00FFFFFF);
    }
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

void* GfxRenderingAPICitro3D::EyeTexture(const Framebuffer& fb) const {
    return (mEyeSign > 0 && fb.rightValid) ? fb.rightTex : fb.tex;
}

void GfxRenderingAPICitro3D::SelectTextureFb(int fbId) {
    if (!tReplaying) {
        Record(OpType::SelectTextureFb).args[0] = fbId;
        return;
    }
    if (fbId <= 0 || fbId >= (int)mFramebuffers.size() || mFramebuffers[fbId].tex == nullptr) {
        return;
    }
    BindUnit(0, AsTex(EyeTexture(mFramebuffers[fbId])));
    mBoundFramebuffer = fbId;
}

void GfxRenderingAPICitro3D::DeleteTexture(uint32_t texId) {
    // This frame's recorded draws, and the GPU's last frame, may still use it: free it once both
    // are done. Texture ids are never reused, so nothing selects it again.
    if (texId < mRecTextureCount) {
        mRec->deleted.push_back(texId);
    }
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

void GfxRenderingAPICitro3D::SetStereoOwnCamera(bool ownCamera) {
    // Called before every batch: record only changes
    if (!tReplaying && ownCamera != mRecStereoOwnCamera) {
        mRecStereoOwnCamera = ownCamera;
        Record(OpType::StereoOwnCamera).args[0] = ownCamera;
    }
}

void GfxRenderingAPICitro3D::SetCurrentPrimDepth(float depth) {
    if (!tReplaying) {
        Record(OpType::PrimDepth).value = depth;
        return;
    }
    mPrimDepth = depth;
}

} // namespace Fast

#endif
