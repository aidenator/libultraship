#ifdef __3DS__

#include "fast/backends/gfx_citro3d.h"
#include "fast/backends/gfx_3ds.h"

#include <algorithm>
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

namespace Fast {

using namespace Citro3D;

// Texture decoding in the interpreter (interpreter.cpp), for the slow-frame report
extern std::atomic<uint32_t> gTexturesImported;
extern std::atomic<uint64_t> gTextureImportMicroseconds;

namespace {
// A sampling profiler: a thread on another core reads gRenderPhase every 200 us and counts what
// the renderer was doing. Reading it costs the renderer nothing, unlike timing each command.
uint32_t sPhaseSamples[RENDER_PHASE_COUNT] = {};

void SamplePhases(void*) {
    for (;;) {
        svcSleepThread(200000);
        const uint32_t phase = gRenderPhase;
        if (phase < RENDER_PHASE_COUNT) {
            sPhaseSamples[phase]++;
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
        default: {
            const char* name = GfxGetOpcodeName((int8_t)phase);
            return name != nullptr ? name : "?";
        }
    }
}

// The phases that took the most samples since the last call, as a share of all samples
void PrintPhaseProfile() {
    static uint32_t sAtLastPrint[RENDER_PHASE_COUNT] = {};
    uint32_t counts[RENDER_PHASE_COUNT];
    uint32_t total = 0;
    for (uint32_t i = 0; i < RENDER_PHASE_COUNT; i++) {
        const uint32_t now = sPhaseSamples[i];
        counts[i] = now - sAtLastPrint[i];
        sAtLastPrint[i] = now;
        total += counts[i];
    }
    if (total == 0) {
        return;
    }
    char line[512];
    int len = snprintf(line, sizeof(line), "  renderer, %lu samples:", (unsigned long)total);
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
    if (!mReplaying) {
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
    mTextures.emplace_back();
    return (uint32_t)(mTextures.size() - 1);
}

void GfxRenderingAPICitro3D::SelectTexture(int tile, uint32_t textureId) {
    if (!mReplaying) {
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
    mFrameUploads++;
    const uint64_t uploadStart = svcGetSystemTick();
    struct AddTicks {
        uint64_t& total;
        uint64_t start;
        ~AddTicks() {
            total += svcGetSystemTick() - start;
        }
    } addTicks{ mUploadTicks, uploadStart };
    if (mRecSelectedUnit > 1 || width == 0 || height == 0) {
        return;
    }
    // The GPU may still be drawing the last frame with this texture
    BeginGpuFrame();
    Texture& t = mTextures[mRecBoundTexture[mRecSelectedUnit]];

    // The PICA needs power-of-two sizes of at least 8; pad by repeating the edge pixels
    const uint32_t texWidth = NextPow2(width);
    const uint32_t texHeight = NextPow2(height);
    if (texWidth > 1024 || texHeight > 1024) {
        return;
    }
    const GPU_TEXCOLOR fmt = ChooseTextureFormat((const uint32_t*)rgba32Buf, width * height);
    mTexturesByBits[fmt == GPU_RGBA8 ? 2 : (fmt == GPU_L8 || fmt == GPU_LA4) ? 0 : 1]++;
    if (t.tex != nullptr && (AsTex(t.tex)->width != texWidth || AsTex(t.tex)->height != texHeight ||
                             AsTex(t.tex)->fmt != fmt)) {
        C3D_TexDelete(AsTex(t.tex));
        delete AsTex(t.tex);
        t.tex = nullptr;
    }
    if (t.tex == nullptr) {
        t.tex = new C3D_Tex();
        if (!C3D_TexInit(AsTex(t.tex), (u16)texWidth, (u16)texHeight, fmt)) {
            printf("citro3d: out of memory for a %lux%lu texture\n", (unsigned long)texWidth,
                   (unsigned long)texHeight);
            delete AsTex(t.tex);
            t.tex = nullptr;
            return;
        }
        C3D_TexSetFilter(AsTex(t.tex), GPU_LINEAR, GPU_LINEAR);
    }
    t.scaleS = (float)width / (float)texWidth;
    t.scaleT = (float)height / (float)texHeight;

    // Swizzle into 8x8 tiles, converting each pixel to the texture's format
    const uint32_t* src = (const uint32_t*)rgba32Buf;
    switch (fmt) {
        case GPU_RGBA8:
            // RGBA bytes to the PICA's ABGR word order
            SwizzleTiles<uint32_t>(src, width, height, texWidth, texHeight, AsTex(t.tex)->data,
                                   [](uint32_t c) { return __builtin_bswap32(c); });
            break;
        case GPU_RGBA5551:
            SwizzleTiles<uint16_t>(src, width, height, texWidth, texHeight, AsTex(t.tex)->data, [](uint32_t c) {
                return (uint16_t)(((R(c) >> 3) << 11) | ((G(c) >> 3) << 6) | ((B(c) >> 3) << 1) | (A(c) >> 7));
            });
            break;
        case GPU_RGBA4:
            SwizzleTiles<uint16_t>(src, width, height, texWidth, texHeight, AsTex(t.tex)->data, [](uint32_t c) {
                return (uint16_t)(((R(c) >> 4) << 12) | ((G(c) >> 4) << 8) | ((B(c) >> 4) << 4) | (A(c) >> 4));
            });
            break;
        case GPU_LA8:
            SwizzleTiles<uint16_t>(src, width, height, texWidth, texHeight, AsTex(t.tex)->data,
                                   [](uint32_t c) { return (uint16_t)((R(c) << 8) | A(c)); });
            break;
        case GPU_LA4:
            SwizzleTiles<uint8_t>(src, width, height, texWidth, texHeight, AsTex(t.tex)->data,
                                  [](uint32_t c) { return (uint8_t)(((R(c) >> 4) << 4) | (A(c) >> 4)); });
            break;
        case GPU_L8:
        default:
            SwizzleTiles<uint8_t>(src, width, height, texWidth, texHeight, AsTex(t.tex)->data,
                                  [](uint32_t c) { return (uint8_t)R(c); });
            break;
    }
    C3D_TexFlush(AsTex(t.tex));
    // The replay's SelectTexture binds it
}

void GfxRenderingAPICitro3D::SetSamplerParameters(int sampler, bool linear_filter, uint32_t cms, uint32_t cmt) {
    if (!mReplaying) {
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
        C3D_Tex* fbTex = AsTex(mFramebuffers[mBoundFramebuffer].tex);
        const bool linear = linear_filter && mFilterMode != FILTER_NONE;
        C3D_TexSetFilter(fbTex, linear ? GPU_LINEAR : GPU_NEAREST, linear ? GPU_LINEAR : GPU_NEAREST);
        C3D_TexSetWrap(fbTex, GPU_CLAMP_TO_EDGE, GPU_CLAMP_TO_EDGE);
        BindUnit(0, fbTex);
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
    if (!mReplaying) {
        Op& op = Record(OpType::DepthTestAndMask);
        op.args[0] = depth_test;
        op.args[1] = z_upd;
        return;
    }
    mDepthTest = depth_test;
    mDepthMask = z_upd;
}

void GfxRenderingAPICitro3D::SetZmodeDecal(bool decal) {
    if (!mReplaying) {
        Record(OpType::ZmodeDecal).args[0] = decal;
        return;
    }
    mDecal = decal;
}

void GfxRenderingAPICitro3D::SetViewport(int x, int y, int width, int height) {
    if (!mReplaying) {
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
    if (!mReplaying) {
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
    if (!mReplaying) {
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

    gRenderPhase = RENDER_PHASE_BACKEND_STATE;
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

    gRenderPhase = RENDER_PHASE_BACKEND_VERTICES;
    float* dst = mVbo + mVboUsed * kVertexFloats;
    for (size_t i = 0; i < numVerts; i++) {
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

    gRenderPhase = RENDER_PHASE_BACKEND_DRAW;
    if (mGpuTest != GpuTest::NoDraws) {
        C3D_DrawArrays(GPU_TRIANGLES, (int)mVboUsed, (int)numVerts);
    }
    gRenderPhase = RENDER_PHASE_BACKEND;
    mVboUsed += numVerts;
    mFrameDraws++;
    mDrawsSinceSubmit++;
}

void GfxRenderingAPICitro3D::SubmitDraws() {
    if (mDrawsSinceSubmit == 0) {
        return;
    }
    RenderPhaseScope phase{ RENDER_PHASE_BACKEND_SUBMIT };
    // The GPU reads the vertex buffer from memory, not the CPU's cache
    if (mVboUsed > mVboSubmitted) {
        svcFlushProcessDataCache(CUR_PROCESS_HANDLE, (u32)(mVbo + mVboSubmitted * kVertexFloats),
                                 (mVboUsed - mVboSubmitted) * kVertexFloats * sizeof(float));
        mVboSubmitted = mVboUsed;
    }
    C3D_FrameSplit(0);
    mDrawsSinceSubmit = 0;
}

void GfxRenderingAPICitro3D::DrawTriangles(float buf_vbo[], size_t buf_vbo_len, size_t buf_vbo_num_tris) {
    if (!mReplaying) {
        // The interpreter reuses its buffer, so keep a copy for the replay
        Op& op = Record(OpType::Draw);
        op.args[0] = (int)mOpFloats.size();
        op.args[1] = (int)buf_vbo_len;
        op.args[2] = (int)buf_vbo_num_tris;
        mOpFloats.insert(mOpFloats.end(), buf_vbo, buf_vbo + buf_vbo_len);
        return;
    }
    ShaderProgram* prg = mCurrentShader;
    if (prg == nullptr || buf_vbo_num_tris == 0) {
        return;
    }
    mFrameTris += buf_vbo_num_tris;
    const CCFeatures& cc = prg->cc;
    const int stride = prg->numFloats;
    const int inputFloats = cc.opt_alpha ? 4 : 3;
    const int inputOff = stride - prg->numInputs * inputFloats;
    const size_t numVerts = buf_vbo_num_tris * 3;

    gRenderPhase = RENDER_PHASE_BACKEND_SCAN;
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
    gRenderPhase = RENDER_PHASE_BACKEND;
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

    sShaderDvlb = DVLB_ParseFile((u32*)fast3d_shbin, fast3d_shbin_size);
    shaderProgramInit(&sShaderProgram);
    shaderProgramSetVsh(&sShaderProgram, &sShaderDvlb->DVLE[0]);
    C3D_BindProgram(&sShaderProgram);

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

    // The profiler shares core 1 with the audio; on an Old 3DS, or without core 1, it just doesn't run
    if (Start3dsThreadOnCore(SamplePhases, nullptr, 1) == nullptr) {
        printf("No renderer profiler\n");
    }
}

void GfxRenderingAPICitro3D::OnResize() {
}

void GfxRenderingAPICitro3D::StartFrame() {
    // Record this frame's drawing while the GPU is still busy with the last one
    mFrameStartTick = svcGetSystemTick();
    mOps.clear();
    mOpFloats.clear();
    mGpuFrameBegun = false;
}

void GfxRenderingAPICitro3D::BeginGpuFrame() {
    if (mGpuFrameBegun) {
        return;
    }
    mGpuFrameBegun = true;
    const uint64_t beginTick = svcGetSystemTick();
    // Wait for the last frame's GPU work (the vertex buffer and command list are reused). Not
    // SYNCDRAW: that also waits for the display's next vblank, which the game's own frame timer is
    // not in step with. The screen still only flips during vblank.
    C3D_FrameBegin(0);
    mWaitTicksTotal += svcGetSystemTick() - beginTick;
    // The GPU is done with the last frame: free the textures deleted before it was sent
    for (uint32_t texId : mTexturesFreeable) {
        C3D_TexDelete(AsTex(mTextures[texId].tex));
        delete AsTex(mTextures[texId].tex);
        mTextures[texId].tex = nullptr;
    }
    mTexturesFreeable.clear();
}

void GfxRenderingAPICitro3D::Replay() {
    BeginGpuFrame();
    C3D_FrameDrawOn(mTopTarget);
    mCurrentTarget = mTopTarget;
    mTargetWidth = kScreenWidth;
    mTargetHeight = kScreenHeight;
    mVboUsed = 0;
    mVboSubmitted = 0;
    mDrawsSinceSubmit = 0;
    mLastTevProgram = nullptr;
    mLastDepthKey = -1;
    mLastBlendKey = -1;

    mReplaying = true;
    for (const Op& op : mOps) {
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
                DrawTriangles(mOpFloats.data() + a[0], (size_t)a[1], (size_t)a[2]);
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
        }
    }
    mReplaying = false;
    // Textures deleted this frame are freed once the GPU has drawn it
    mTexturesFreeable.insert(mTexturesFreeable.end(), mTexturesToFree.begin(), mTexturesToFree.end());
    mTexturesToFree.clear();
}

GfxRenderingAPICitro3D::Op& GfxRenderingAPICitro3D::Record(OpType type) {
    Op& op = mOps.emplace_back();
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
    mDrawTicksTotal = 0;
    mDrawTicksMax = 0;
}

void GfxRenderingAPICitro3D::EndFrame() {
    Replay();
    SubmitDraws();
    C3D_FrameEnd(0);
    UpdateGpuTest();

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
                   frameMs, (double)(now - mFrameStartTick) * toMs, C3D_GetDrawingTime(),
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
    const uint64_t drawTicks = now - mFrameStartTick;
    mDrawTicksTotal += drawTicks;
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
               "draw %.1f ms (worst %.0f), waiting for the GPU %.1f ms, "
               "heap %lu/%lu KB in use, linear %lu KB free\n",
               mGpuTest == GpuTest::Off ? "" : "[test] ", (unsigned long)mFrameCount, mFrameCount / elapsed,
               (unsigned long)mFrameDraws,
               (unsigned long)mFrameTris, (unsigned long)mFrameUploads, C3D_GetDrawingTime(),
               (double)mDrawTicksTotal * toMs / mFrameCount, (double)mDrawTicksMax * toMs,
               (double)mWaitTicksTotal * toMs / mFrameCount, (unsigned long)(heap.uordblks / 1024),
               (unsigned long)(envGetHeapSize() / 1024), (unsigned long)(linearSpaceFree() / 1024));
        PrintPhaseProfile();
        printf("  textures so far: %lu 8-bit, %lu 16-bit, %lu 32-bit\n", (unsigned long)mTexturesByBits[0],
               (unsigned long)mTexturesByBits[1], (unsigned long)mTexturesByBits[2]);
        mWaitTicksTotal = 0;
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
    mFramebuffers.emplace_back();
    return (int)mFramebuffers.size() - 1;
}

void GfxRenderingAPICitro3D::UpdateFramebufferParameters(int fb_id, uint32_t width, uint32_t height,
                                                         uint32_t msaa_level, bool opengl_invertY, bool render_target,
                                                         bool has_depth_buffer, bool can_extract_depth) {
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
    if (!mReplaying) {
        Record(OpType::DrawToFramebuffer).args[0] = fbId;
        return;
    }
    if (fbId <= 0 || fbId >= (int)mFramebuffers.size() || mFramebuffers[fbId].tex == nullptr) {
        C3D_FrameDrawOn(mTopTarget);
        mCurrentTarget = mTopTarget;
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
    if (!mReplaying) {
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
    // The copy engine runs from the GX queue, after anything already submitted: submit the
    // frame's draws so far first
    SubmitDraws();
    // Both are 8x8-tiled RGBA8, so copy tile row by tile row: 240 pixels of each 256-wide row
    C3D_FrameBuf& screen = mTopTarget->frameBuf;
    const u32 rowBytes = (u32)screen.width * 8 * 4;
    const u32 texRowBytes = (u32)kFbTexWidth * 8 * 4;
    u8* dst = (u8*)AsTex(fb.tex)->data + kCopyTileRowOffset * texRowBytes;
    GX_TextureCopy((u32*)screen.colorBuf, GX_BUFFER_DIM(rowBytes >> 4, 0), (u32*)dst,
                   GX_BUFFER_DIM(rowBytes >> 4, (texRowBytes - rowBytes) >> 4), rowBytes * (screen.height / 8),
                   GX_TRANSFER_RAW_COPY(1));
    fb.contentWidth = kScreenWidth;
    fb.contentHeight = kScreenHeight;
}

void GfxRenderingAPICitro3D::ClearFramebuffer(bool color, bool depth) {
    if (!mReplaying) {
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

void GfxRenderingAPICitro3D::SelectTextureFb(int fbId) {
    if (!mReplaying) {
        Record(OpType::SelectTextureFb).args[0] = fbId;
        return;
    }
    if (fbId <= 0 || fbId >= (int)mFramebuffers.size() || mFramebuffers[fbId].tex == nullptr) {
        return;
    }
    BindUnit(0, AsTex(mFramebuffers[fbId].tex));
    mBoundFramebuffer = fbId;
}

void GfxRenderingAPICitro3D::DeleteTexture(uint32_t texId) {
    // This frame's recorded draws, and the GPU's last frame, may still use it: free it once both
    // are done. Texture ids are never reused, so nothing selects it again.
    if (texId < mTextures.size() && mTextures[texId].tex != nullptr) {
        mTexturesToFree.push_back(texId);
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

void GfxRenderingAPICitro3D::SetCurrentPrimDepth(float depth) {
    if (!mReplaying) {
        Record(OpType::PrimDepth).value = depth;
        return;
    }
    mPrimDepth = depth;
}

} // namespace Fast

#endif
