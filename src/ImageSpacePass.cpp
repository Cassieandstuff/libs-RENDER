// The post pass with an image space (ImageSpace::enabled): what a game does between the lit frame and the screen.
//   1. luminance  the HDR frame at quarter size → log luminance × coverage and coverage (RGBA16F), mipped down to 1×1:
//                 its last level is the frame's geometric-mean luminance over the geometry (the background doesn't count)
//   2. adapt      a 1×1 R32F ping-pong: the adapted luminance closes 1 − e^(−speed·dt) of the gap to that mean
//   3. bloom      half size: a bright pass (after exposure, above the threshold), then a separable Gaussian, H and V
//   4. post       exposure from the adapted luminance, bloom added, the white-point tone, composited over the
//                 background, then the cinematic grade on display values, into the output
// Its tone curve replaces the view transform. Everything is fullscreen triangles; no Diligent type leaves this file.
#include "SceneRendererImpl.h"
#include "Shaders.h"

#include "CommonlyUsedStates.h"
#include "GraphicsAccessories.hpp"
#include "MapHelper.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <string>
#include <vector>

namespace render::diligent {

using namespace Diligent;

namespace {

constexpr TEXTURE_FORMAT kLumFormat   = TEX_FORMAT_RGBA16_FLOAT;
constexpr TEXTURE_FORMAT kBloomFormat = TEX_FORMAT_RGBA16_FLOAT;

#define RENDER_IS_CB R"(
cbuffer ImageSpaceCB {
    float4 g_A;          // adapt key, strength, min exposure, max exposure
    float4 g_B;          // adapt blend (0..1), last luminance mip, first frame, base exposure
    float4 g_C;          // bloom threshold (linear), blur sigma (texels), step x, step y
    float4 g_D;          // white, bloom scale, saturation, brightness
    float4 g_E;          // contrast, tint amount, unused, unused
    float4 g_Tint;       // tint colour (display)
    float4 g_Background; // linear rgb
    float4 g_Size;       // HDR width, height, luminance width, height
};
static const float3 kLumW = float3(0.2126, 0.7152, 0.0722);
struct PSIn { float4 pos : SV_POSITION; float2 uv : TEX_COORD; };
float Exposure(float adapted) {
    return clamp(pow(g_A.x / max(adapted, 1e-5), g_A.y), g_A.z, g_A.w) * g_B.w;
}
)"

constexpr const char* kLumPS = RENDER_IS_CB R"(
Texture2D<float4> g_HDR;
float4 main(in PSIn i) : SV_Target {
    int2  base = int2(i.pos.xy) * 4;
    int2  lim  = int2(g_Size.xy) - 1;
    float sumL = 0.0, sumA = 0.0;
    for (int y = 1; y < 4; y += 2)
        for (int x = 1; x < 4; x += 2) {
            float4 h = g_HDR.Load(int3(min(base + int2(x, y), lim), 0));
            float  a = saturate(h.a);
            float  l = dot(h.rgb / max(h.a, 1e-4), kLumW);
            sumL += log(max(l, 1e-4)) * a;
            sumA += a;
        }
    return float4(sumL * 0.25, sumA * 0.25, 0.0, 1.0);
}
)";

constexpr const char* kAdaptPS = RENDER_IS_CB R"(
Texture2D<float4> g_Lum;
Texture2D<float>  g_Prev;
float main(in PSIn i) : SV_Target {
    float4 s    = g_Lum.Load(int3(0, 0, (int)g_B.y));
    float  mean = s.g > 1e-4 ? exp(s.r / s.g) : g_A.x;   // nothing drawn: no pull either way
    float  prev = g_Prev.Load(int3(0, 0, 0));
    return g_B.z > 0.5 ? mean : lerp(prev, mean, g_B.x);
}
)";

constexpr const char* kBrightPS = RENDER_IS_CB R"(
Texture2D<float4> g_HDR;
Texture2D<float>  g_Adapted;
float4 main(in PSIn i) : SV_Target {
    int2   base = int2(i.pos.xy) * 2;
    int2   lim  = int2(g_Size.xy) - 1;
    float3 sum  = float3(0.0, 0.0, 0.0);
    for (int y = 0; y < 2; ++y)
        for (int x = 0; x < 2; ++x) sum += g_HDR.Load(int3(min(base + int2(x, y), lim), 0)).rgb;   // premultiplied
    float3 lit = sum * 0.25 * Exposure(g_Adapted.Load(int3(0, 0, 0)));
    float  l   = dot(lit, kLumW);
    return float4(lit * (max(l - g_C.x, 0.0) / max(l, 1e-4)), 1.0);
}
)";

constexpr const char* kBlurPS = RENDER_IS_CB R"(
Texture2D<float4> g_Src;
float4 main(in PSIn i) : SV_Target {
    int2   p     = int2(i.pos.xy);
    int2   lim   = int2(g_Size.zw) - 1;   // the bloom targets' size travels in zw here
    int2   step  = int2(g_C.zw);
    float  sigma = max(g_C.y, 0.5);
    int    r     = min((int)ceil(sigma * 2.5), 24);
    float3 sum   = float3(0.0, 0.0, 0.0);
    float  wsum  = 0.0;
    for (int k = -r; k <= r; ++k) {
        float w = exp(-(k * k) / (2.0 * sigma * sigma));
        sum  += g_Src.Load(int3(clamp(p + step * k, int2(0, 0), lim), 0)).rgb * w;
        wsum += w;
    }
    return float4(sum / wsum, 1.0);
}
)";

constexpr const char* kIsPostPS = RENDER_IS_CB R"(
Texture2D<float4> g_HDR;
Texture2D<float>  g_Adapted;
Texture2D<float4> g_Bloom;
SamplerState      g_Bloom_sampler;
float3 LinearToSrgb(float3 c) {
    c = max(c, 0.0);
    return lerp(c * 12.92, 1.055 * pow(c, 1.0 / 2.4) - 0.055, step(0.0031308, c));
}
float3 SrgbToLinear(float3 c) {
    c = max(c, 0.0);
    return lerp(c / 12.92, pow((c + 0.055) / 1.055, 2.4), step(0.04045, c));
}
float3 Tone(float3 c) {
    float w = max(g_D.x, 1e-3);
    return c * (1.0 + c / (w * w)) / (1.0 + c);
}
float4 main(in PSIn i) : SV_Target {
    float4 hdr   = g_HDR.Load(int3(int2(i.pos.xy), 0));
    float  a     = saturate(hdr.a);
    float  e     = Exposure(g_Adapted.Load(int3(0, 0, 0)));
    float3 bloom = g_Bloom.Sample(g_Bloom_sampler, i.uv).rgb * g_D.y;
    float3 lit   = a > 0.0 ? hdr.rgb / max(hdr.a, 1e-4) * e : float3(0.0, 0.0, 0.0);
    float3 scene = saturate(abs(g_D.x - 1.0) < 1e-4 ? lit + bloom : Tone(lit + bloom));
    float3 ldr   = lerp(saturate(g_Background.rgb + bloom), scene, a);
    // The grade, on display values.
    float3 d = LinearToSrgb(ldr);
    float  l = dot(d, kLumW);
    d = lerp(float3(l, l, l), d, g_D.z);
    d *= g_D.w;
    d = (d - 0.5) * g_E.x + 0.5;
    d = lerp(d, l * g_Tint.rgb, g_E.y);
    ldr = SrgbToLinear(saturate(d));
#if OUTPUT_GAMMA
    ldr = pow(ldr, 1.0 / 2.2);
#endif
    return float4(ldr, 1.0);
}
)";

#undef RENDER_IS_CB

struct IsConstants {
    glm::vec4 a, b, c, d, e, tint, background, size;
};

float SrgbToLinear(float c) { return c <= 0.04045f ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f); }

} // namespace

struct ImageSpaceState {
    struct Pass {
        RefCntAutoPtr<IPipelineState>         pso;
        RefCntAutoPtr<IShaderResourceBinding> srb;
    };
    Pass lum, adapt, bright, blur;
    std::map<int, Pass> post;   // by output format
    RefCntAutoPtr<IBuffer>  cb;
    RefCntAutoPtr<ITexture> lumTex, adapted[2], bloomA, bloomB;
    int  current = 0;           // adapted[current] holds the last frame's value
    bool first   = true;
    int  width = 0, height = 0, lumW = 0, lumH = 0, lumMips = 1, bloomW = 0, bloomH = 0;
    bool failed = false;
};

void DestroyImageSpaceState(ImageSpaceState* p) { delete p; }

ImageSpacePtr CreateImageSpaceState(DiligentSceneRenderer::Impl& r) {
    ImageSpacePtr st(new ImageSpaceState{}, &DestroyImageSpaceState);
    BufferDesc d;
    d.Name           = "render image space CB";
    d.Size           = sizeof(IsConstants);
    d.Usage          = USAGE_DYNAMIC;
    d.BindFlags      = BIND_UNIFORM_BUFFER;
    d.CPUAccessFlags = CPU_ACCESS_WRITE;
    r.device->CreateBuffer(d, nullptr, &st->cb);
    return st;
}

namespace {

ImageSpaceState::Pass MakePass(DiligentSceneRenderer::Impl& r, ImageSpaceState& st, const char* name, const char* ps,
                               TEXTURE_FORMAT format, std::initializer_list<const char*> textures, bool bloomSampler,
                               bool outputGamma = false) {
    ImageSpaceState::Pass pass;
    const ShaderMacro macros[] = { { "OUTPUT_GAMMA", outputGamma ? "1" : "0" } };
    auto vs = r.CompileShader("render fullscreen VS", SHADER_TYPE_VERTEX, shaders::kFullscreenVS);
    auto p  = r.CompileShader(name, SHADER_TYPE_PIXEL, ps, macros, 1);
    if (!vs || !p) return pass;
    GraphicsPipelineStateCreateInfo ci;
    ci.PSODesc.Name = name;
    GraphicsPipelineDesc& g = ci.GraphicsPipeline;
    g.NumRenderTargets              = 1;
    g.RTVFormats[0]                 = format;
    g.DSVFormat                     = TEX_FORMAT_UNKNOWN;
    g.PrimitiveTopology             = PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    g.RasterizerDesc.CullMode       = CULL_MODE_NONE;
    g.DepthStencilDesc.DepthEnable  = False;
    ci.pVS = vs;
    ci.pPS = p;
    std::vector<ShaderResourceVariableDesc> vars;
    for (const char* t : textures) vars.push_back({ SHADER_TYPE_PIXEL, t, SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC });
    const ImmutableSamplerDesc sampler{ SHADER_TYPE_PIXEL, "g_Bloom", Sam_LinearClamp };
    ci.PSODesc.ResourceLayout.DefaultVariableType = SHADER_RESOURCE_VARIABLE_TYPE_STATIC;
    ci.PSODesc.ResourceLayout.Variables           = vars.data();
    ci.PSODesc.ResourceLayout.NumVariables        = static_cast<Uint32>(vars.size());
    if (bloomSampler) {
        ci.PSODesc.ResourceLayout.ImmutableSamplers    = &sampler;
        ci.PSODesc.ResourceLayout.NumImmutableSamplers = 1;
    }
    r.device->CreateGraphicsPipelineState(ci, &pass.pso);
    if (!pass.pso) return pass;
    if (auto* var = pass.pso->GetStaticVariableByName(SHADER_TYPE_PIXEL, "ImageSpaceCB")) var->Set(st.cb);
    pass.pso->CreateShaderResourceBinding(&pass.srb, true);
    return pass;
}

RefCntAutoPtr<ITexture> MakeTexture(IRenderDevice* device, const char* name, int w, int h, TEXTURE_FORMAT format, Uint32 mips = 1,
                                    bool generateMips = false) {
    TextureDesc d;
    d.Name      = name;
    d.Type      = RESOURCE_DIM_TEX_2D;
    d.Width     = static_cast<Uint32>(std::max(w, 1));
    d.Height    = static_cast<Uint32>(std::max(h, 1));
    d.MipLevels = mips;
    d.Format    = format;
    d.BindFlags = BIND_RENDER_TARGET | BIND_SHADER_RESOURCE;
    if (generateMips) d.MiscFlags = MISC_TEXTURE_FLAG_GENERATE_MIPS;
    RefCntAutoPtr<ITexture> t;
    device->CreateTexture(d, nullptr, &t);
    return t;
}

bool Prepare(DiligentSceneRenderer::Impl& r, ImageSpaceState& st) {
    if (st.failed) return false;
    if (!st.lum.pso) {
        st.lum    = MakePass(r, st, "render image space luminance", kLumPS, kLumFormat, { "g_HDR" }, false);
        st.adapt  = MakePass(r, st, "render image space adapt", kAdaptPS, TEX_FORMAT_R32_FLOAT, { "g_Lum", "g_Prev" }, false);
        st.bright = MakePass(r, st, "render image space bright", kBrightPS, kBloomFormat, { "g_HDR", "g_Adapted" }, false);
        st.blur   = MakePass(r, st, "render image space blur", kBlurPS, kBloomFormat, { "g_Src" }, false);
        if (!st.lum.srb || !st.adapt.srb || !st.bright.srb || !st.blur.srb) {
            st.failed = true;
            return false;
        }
    }
    if (st.width != r.width || st.height != r.height || !st.lumTex) {
        st.width   = r.width;
        st.height  = r.height;
        st.lumW    = std::max(r.width / 4, 1);
        st.lumH    = std::max(r.height / 4, 1);
        st.lumMips = 1;
        for (int s = std::max(st.lumW, st.lumH); s > 1; s >>= 1) ++st.lumMips;
        st.lumTex  = MakeTexture(r.device, "render image space luminance", st.lumW, st.lumH, kLumFormat, static_cast<Uint32>(st.lumMips), true);
        st.bloomW  = std::max(r.width / 2, 1);
        st.bloomH  = std::max(r.height / 2, 1);
        st.bloomA  = MakeTexture(r.device, "render bloom A", st.bloomW, st.bloomH, kBloomFormat);
        st.bloomB  = MakeTexture(r.device, "render bloom B", st.bloomW, st.bloomH, kBloomFormat);
    }
    if (!st.adapted[0]) {
        st.adapted[0] = MakeTexture(r.device, "render adapted luminance 0", 1, 1, TEX_FORMAT_R32_FLOAT);
        st.adapted[1] = MakeTexture(r.device, "render adapted luminance 1", 1, 1, TEX_FORMAT_R32_FLOAT);
        st.first      = true;
    }
    return st.lumTex && st.bloomA && st.bloomB && st.adapted[0] && st.adapted[1];
}

void Fullscreen(DiligentSceneRenderer::Impl& r, ITexture* target, const ImageSpaceState::Pass& pass) {
    ITextureView* rtv = target->GetDefaultView(TEXTURE_VIEW_RENDER_TARGET);
    r.context->SetRenderTargets(1, &rtv, nullptr, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    r.context->SetViewports(1, nullptr, 0, 0);
    r.context->SetPipelineState(pass.pso);
    r.context->CommitShaderResources(pass.srb, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    DrawAttribs draw{ 3, DRAW_FLAG_VERIFY_ALL };
    r.context->Draw(draw);
}

void Bind(const ImageSpaceState::Pass& pass, const char* name, ITexture* tex) {
    if (auto* var = pass.srb->GetVariableByName(SHADER_TYPE_PIXEL, name)) var->Set(tex->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE));
}

} // namespace

ITextureView* ImageSpaceAdaptedView(DiligentSceneRenderer::Impl& r) {
    ImageSpaceState* st = r.imageSpaceState.get();
    if (!st || st->failed || st->first || !st->adapted[st->current]) return nullptr;
    return st->adapted[st->current]->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE);
}

bool ImageSpacePost(DiligentSceneRenderer::Impl& r, ITextureView* outputRTV) {
    ImageSpaceState* st = r.imageSpaceState.get();
    if (!st || !st->cb || !Prepare(r, *st)) return false;
    const TEXTURE_FORMAT format = outputRTV->GetDesc().Format;
    auto [it, inserted] = st->post.try_emplace(static_cast<int>(format));
    if (inserted) {
        const bool srgb = GetTextureFormatAttribs(format).ComponentType == COMPONENT_TYPE_UNORM_SRGB;
        it->second = MakePass(r, *st, "render image space post", kIsPostPS, format, { "g_HDR", "g_Adapted", "g_Bloom" }, true, !srgb);
    }
    const ImageSpaceState::Pass& post = it->second;
    if (!post.srb) return false;

    const ImageSpace& is = r.imageSpace;
    const float blend   = is.deltaSeconds <= 0.f ? 1.f : 1.f - std::exp(-std::max(is.adaptSpeed, 0.f) * is.deltaSeconds);
    // The threshold is a display value; the bright pass compares linear luminance.
    const float threshold = SrgbToLinear(std::clamp(is.bloomThreshold, 0.f, 1.f)) * std::max(1.f, is.bloomThreshold);
    const float sigma     = std::max(is.bloomRadius, 0.f) * (static_cast<float>(r.height) / 1080.f) * 0.5f / 2.f;   // half size, σ = radius / 2
    // False when out of dynamic memory (MapDiscard): the pass then gives up and the plain post pass is tried.
    auto write = [&](float stepX, float stepY, glm::vec2 blurSize) {
        MapHelper<IsConstants> map;
        IsConstants* c = MapDiscard(r, map, st->cb);
        if (!c) return false;
        c->a         = { std::max(is.adaptKey, 1e-5f), std::clamp(is.adaptStrength, 0.f, 4.f), is.minExposure, is.maxExposure };
        c->b          = { blend, static_cast<float>(st->lumMips - 1), st->first ? 1.f : 0.f, r.exposure };
        c->c          = { threshold, sigma, stepX, stepY };
        c->d          = { is.white, std::max(is.bloomScale, 0.f), is.saturation, is.brightness };
        c->e          = { is.contrast, std::clamp(is.tintAmount, 0.f, 1.f), 0.f, 0.f };
        c->tint       = glm::vec4(is.tintColor, 0.f);
        c->background = { SrgbToLinear(r.style.background[0]), SrgbToLinear(r.style.background[1]), SrgbToLinear(r.style.background[2]), 1.f };
        c->size       = { static_cast<float>(r.width), static_cast<float>(r.height), blurSize.x, blurSize.y };
        return true;
    };
    const glm::vec2 bloomSize(static_cast<float>(st->bloomW), static_cast<float>(st->bloomH));

    // 1. Luminance, mipped to its mean.
    if (!write(0.f, 0.f, bloomSize)) return false;
    Bind(st->lum, "g_HDR", r.hdrColor);
    Fullscreen(r, st->lumTex, st->lum);
    r.context->GenerateMips(st->lumTex->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE));

    // 2. Adapt.
    const int next = 1 - st->current;
    Bind(st->adapt, "g_Lum", st->lumTex);
    Bind(st->adapt, "g_Prev", st->adapted[st->current]);
    Fullscreen(r, st->adapted[next], st->adapt);
    st->current = next;
    st->first   = false;
    ITexture* adapted = st->adapted[st->current];

    // 3. Bloom (skipped at scale 0; the post pass still reads a target, cleared).
    if (is.bloomScale > 0.f) {
        Bind(st->bright, "g_HDR", r.hdrColor);
        Bind(st->bright, "g_Adapted", adapted);
        Fullscreen(r, st->bloomA, st->bright);
        if (!write(1.f, 0.f, bloomSize)) return false;
        Bind(st->blur, "g_Src", st->bloomA);
        Fullscreen(r, st->bloomB, st->blur);
        if (!write(0.f, 1.f, bloomSize)) return false;
        Bind(st->blur, "g_Src", st->bloomB);
        Fullscreen(r, st->bloomA, st->blur);
    } else {
        ITextureView* rtv     = st->bloomA->GetDefaultView(TEXTURE_VIEW_RENDER_TARGET);
        const float   zero[4] = { 0.f, 0.f, 0.f, 0.f };
        r.context->SetRenderTargets(1, &rtv, nullptr, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
        r.context->ClearRenderTarget(rtv, zero, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    }

    // 4. Post into the output.
    if (!write(0.f, 0.f, bloomSize)) return false;
    Bind(post, "g_HDR", r.hdrColor);
    Bind(post, "g_Adapted", adapted);
    Bind(post, "g_Bloom", st->bloomA);
    r.context->SetRenderTargets(1, &outputRTV, nullptr, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    r.context->SetViewports(1, nullptr, 0, 0);
    r.context->SetPipelineState(post.pso);
    r.context->CommitShaderResources(post.srb, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    DrawAttribs draw{ 3, DRAW_FLAG_VERIFY_ALL };
    r.context->Draw(draw);
    return true;
}

} // namespace render::diligent
