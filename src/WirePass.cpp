// Flat passes over the finished frame, after the tone map and the grid, before the overlays. Both re-draw draws of the
// frame (kept by FlushQueue) in a flat display colour (exact, like an overlay), depth-tested against the scene:
//   wireframe over shaded (DrawSurface::wireColor): the surface's edges, with a small bias towards the camera so its own
//     edges show. Needs the device's wireframe fill; without it the request is skipped (said once).
//   outline (DrawSurface::outlineColor): the visible part of the surface is drawn into a mask, then an edge pass paints
//     ViewStyle::outlineWidth pixels around it in its colour (where two outlines meet, the stronger alpha wins).
#include "SceneRendererImpl.h"
#include "Shaders.h"

#include "GraphicsAccessories.hpp"
#include "MapHelper.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>

namespace render::diligent {

using namespace Diligent;

namespace {

constexpr Uint32 kMaxJoints = 256;
// sRGB storage: the mask holds linear colour, and 8 bits encoded keep dark display values exact.
constexpr TEXTURE_FORMAT kMaskFormat = TEX_FORMAT_RGBA8_UNORM_SRGB;

constexpr const char* kFlatVS = R"(
cbuffer FlatDraw {
    float4x4 g_ViewProj;
    float4x4 g_World;
    float4   g_Color;
};
#if SKINNED
cbuffer FlatBones { float4x4 g_Bones[256]; };
#endif
struct VSIn {
    float3 pos : ATTRIB0;
#if SKINNED
    float4 joints  : ATTRIB3;
    float4 weights : ATTRIB4;
#endif
};
void main(in VSIn i, out float4 pos : SV_POSITION) {
    float4x4 model = g_World;
#if SKINNED
    float4x4 skin = g_Bones[(int)i.joints.x] * i.weights.x + g_Bones[(int)i.joints.y] * i.weights.y +
                    g_Bones[(int)i.joints.z] * i.weights.z + g_Bones[(int)i.joints.w] * i.weights.w;
    model = mul(g_World, skin);
#endif
    pos = mul(g_ViewProj, mul(model, float4(i.pos, 1.0)));
}
)";

constexpr const char* kFlatPS = R"(
cbuffer FlatDraw {
    float4x4 g_ViewProj;
    float4x4 g_World;
    float4   g_Color;   // premultiplied, in the target's encoding
};
float4 main(in float4 pos : SV_POSITION) : SV_Target { return g_Color; }
)";

// The outline's edge pass: a pixel outside every mask takes the strongest mask colour within the radius.
constexpr const char* kOutlinePS = R"(
Texture2D<float4> g_Mask;
cbuffer OutlineCB { int4 g_Params; };   // x = radius (pixels), y = width, z = height
struct PSIn { float4 pos : SV_POSITION; float2 uv : TEX_COORD; };
float4 main(in PSIn i) : SV_Target {
    int2 p = int2(i.pos.xy);
    if (g_Mask.Load(int3(p, 0)).a > 0.0) discard;
    int    r    = g_Params.x;
    float4 best = float4(0.0, 0.0, 0.0, 0.0);
    for (int dy = -r; dy <= r; ++dy)
        for (int dx = -r; dx <= r; ++dx) {
            if (dx * dx + dy * dy > r * r) continue;
            int2   q = clamp(p + int2(dx, dy), int2(0, 0), g_Params.yz - 1);
            float4 c = g_Mask.Load(int3(q, 0));
            if (c.a > best.a) best = c;
        }
    if (best.a <= 0.0) discard;
#if OUTPUT_GAMMA
    best.rgb = pow(best.rgb, 1.0 / 2.2);   // a non-sRGB output takes display values
#endif
    return best;
}
)";

struct FlatConstants {
    glm::mat4 viewProj, world;
    glm::vec4 color;
};

float SrgbToLinear(float c) { return c <= 0.04045f ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f); }

// A display colour, premultiplied, in a target's encoding.
glm::vec4 Encode(Rgba8 c, bool srgbTarget) {
    const float a = c.a / 255.f;
    auto ch = [&](std::uint8_t x) { const float f = x / 255.f; return (srgbTarget ? SrgbToLinear(f) : f) * a; };
    return { ch(c.r), ch(c.g), ch(c.b), a };
}

} // namespace

struct WireState {
    struct Variant {
        RefCntAutoPtr<IPipelineState>         pso;
        RefCntAutoPtr<IShaderResourceBinding> srb;
    };
    std::map<int, Variant> variants;   // key: target format << 3 | outline mask << 2 | wire << 1 | skinned
    std::map<int, Variant> edges;      // by output format
    RefCntAutoPtr<IBuffer>  cb, bonesCB, outlineCB;
    RefCntAutoPtr<ITexture> mask;
    int  maskW = 0, maskH = 0;
    bool warnedNoWire = false, warnedSize = false;
};

void DestroyWireState(WireState* p) { delete p; }

WirePtr CreateWireState(DiligentSceneRenderer::Impl& r) {
    WirePtr st(new WireState{}, &DestroyWireState);
    auto cb = [&](const char* name, Uint32 size) {
        BufferDesc d;
        d.Name           = name;
        d.Size           = size;
        d.Usage          = USAGE_DYNAMIC;
        d.BindFlags      = BIND_UNIFORM_BUFFER;
        d.CPUAccessFlags = CPU_ACCESS_WRITE;
        RefCntAutoPtr<IBuffer> b;
        r.device->CreateBuffer(d, nullptr, &b);
        return b;
    };
    st->cb        = cb("render flat CB", sizeof(FlatConstants));
    st->bonesCB   = cb("render flat bones CB", sizeof(glm::mat4) * kMaxJoints);
    st->outlineCB = cb("render outline CB", sizeof(int) * 4);
    return st;
}

namespace {

// wire: edges over the output (biased, blended). Otherwise the outline mask: solid, unbiased, replacing.
WireState::Variant& FlatVariant(DiligentSceneRenderer::Impl& r, WireState& st, TEXTURE_FORMAT format, bool skinned, bool wire) {
    const int key = (static_cast<int>(format) << 3) | (wire ? 0 : 4) | (wire ? 2 : 0) | (skinned ? 1 : 0);
    auto [it, inserted] = st.variants.try_emplace(key);
    WireState::Variant& v = it->second;
    if (!inserted) return v;
    const ShaderMacro macros[] = { { "SKINNED", skinned ? "1" : "0" } };
    auto vs = r.CompileShader("render flat VS", SHADER_TYPE_VERTEX, kFlatVS, macros, 1);
    auto ps = r.CompileShader("render flat PS", SHADER_TYPE_PIXEL, kFlatPS, macros, 1);
    if (!vs || !ps) return v;
    std::vector<LayoutElement> layout = { { 0, 0, 3, VT_FLOAT32, False } };
    if (skinned) {
        layout.push_back({ 3, 4, 4, VT_FLOAT32, False });
        layout.push_back({ 4, 5, 4, VT_FLOAT32, False });
    }
    GraphicsPipelineStateCreateInfo ci;
    ci.PSODesc.Name = wire ? "render wire PSO" : "render outline mask PSO";
    GraphicsPipelineDesc& g = ci.GraphicsPipeline;
    g.NumRenderTargets                  = 1;
    g.RTVFormats[0]                     = format;
    g.DSVFormat                         = kSceneDepthFormat;
    g.PrimitiveTopology                 = PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    g.InputLayout                       = { layout.data(), static_cast<Uint32>(layout.size()) };
    g.RasterizerDesc.FillMode           = wire ? FILL_MODE_WIREFRAME : FILL_MODE_SOLID;
    g.RasterizerDesc.CullMode           = CULL_MODE_NONE;
    g.DepthStencilDesc.DepthEnable      = True;
    g.DepthStencilDesc.DepthWriteEnable = False;
    g.DepthStencilDesc.DepthFunc        = COMPARISON_FUNC_LESS_EQUAL;
    if (wire) {
        g.RasterizerDesc.DepthBias            = -256;   // the surface's own edges sit on its depth: pull them in front
        g.RasterizerDesc.SlopeScaledDepthBias = -2.f;
        RenderTargetBlendDesc& rt = g.BlendDesc.RenderTargets[0];   // premultiplied over
        rt.BlendEnable = True;
        rt.SrcBlend = rt.SrcBlendAlpha = BLEND_FACTOR_ONE;
        rt.DestBlend = rt.DestBlendAlpha = BLEND_FACTOR_INV_SRC_ALPHA;
    } else {
        // The mask is drawn where the surface is visible: its own depth is in the scene depth, equal within a bias.
        g.RasterizerDesc.DepthBias            = -64;
        g.RasterizerDesc.SlopeScaledDepthBias = -1.f;
    }
    ci.pVS = vs;
    ci.pPS = ps;
    ci.PSODesc.ResourceLayout.DefaultVariableType = SHADER_RESOURCE_VARIABLE_TYPE_STATIC;
    r.device->CreateGraphicsPipelineState(ci, &v.pso);
    if (!v.pso) return v;
    for (SHADER_TYPE stage : { SHADER_TYPE_VERTEX, SHADER_TYPE_PIXEL })
        if (auto* var = v.pso->GetStaticVariableByName(stage, "FlatDraw")) var->Set(st.cb);
    if (skinned)
        if (auto* var = v.pso->GetStaticVariableByName(SHADER_TYPE_VERTEX, "FlatBones")) var->Set(st.bonesCB);
    v.pso->CreateShaderResourceBinding(&v.srb, true);
    return v;
}

WireState::Variant& EdgeVariant(DiligentSceneRenderer::Impl& r, WireState& st, TEXTURE_FORMAT format) {
    auto [it, inserted] = st.edges.try_emplace(static_cast<int>(format));
    WireState::Variant& v = it->second;
    if (!inserted) return v;
    const bool        srgbOut  = GetTextureFormatAttribs(format).ComponentType == COMPONENT_TYPE_UNORM_SRGB;
    const ShaderMacro macros[] = { { "OUTPUT_GAMMA", srgbOut ? "0" : "1" } };
    auto vs = r.CompileShader("render fullscreen VS", SHADER_TYPE_VERTEX, shaders::kFullscreenVS);
    auto ps = r.CompileShader("render outline PS", SHADER_TYPE_PIXEL, kOutlinePS, macros, 1);
    if (!vs || !ps) return v;
    GraphicsPipelineStateCreateInfo ci;
    ci.PSODesc.Name = "render outline PSO";
    GraphicsPipelineDesc& g = ci.GraphicsPipeline;
    g.NumRenderTargets                  = 1;
    g.RTVFormats[0]                     = format;
    g.DSVFormat                         = TEX_FORMAT_UNKNOWN;
    g.PrimitiveTopology                 = PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    g.RasterizerDesc.CullMode           = CULL_MODE_NONE;
    g.DepthStencilDesc.DepthEnable      = False;
    RenderTargetBlendDesc& rt = g.BlendDesc.RenderTargets[0];
    rt.BlendEnable = True;
    rt.SrcBlend = rt.SrcBlendAlpha = BLEND_FACTOR_ONE;
    rt.DestBlend = rt.DestBlendAlpha = BLEND_FACTOR_INV_SRC_ALPHA;
    ci.pVS = vs;
    ci.pPS = ps;
    const ShaderResourceVariableDesc vars[] = { { SHADER_TYPE_PIXEL, "g_Mask", SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC } };
    ci.PSODesc.ResourceLayout.DefaultVariableType = SHADER_RESOURCE_VARIABLE_TYPE_STATIC;
    ci.PSODesc.ResourceLayout.Variables           = vars;
    ci.PSODesc.ResourceLayout.NumVariables        = 1;
    r.device->CreateGraphicsPipelineState(ci, &v.pso);
    if (!v.pso) return v;
    if (auto* var = v.pso->GetStaticVariableByName(SHADER_TYPE_PIXEL, "OutlineCB")) var->Set(st.outlineCB);
    v.pso->CreateShaderResourceBinding(&v.srb, true);
    return v;
}

// Draws `draws` flat into the bound target, each in the colour `colorOf` gives (encoded for the target).
template <typename ColorOf>
void DrawFlat(DiligentSceneRenderer::Impl& r, WireState& st, TEXTURE_FORMAT format, bool wire, const std::vector<QueuedDraw>& draws,
              const std::vector<glm::mat4>& bones, ColorOf&& colorOf) {
    for (const QueuedDraw& d : draws) {
        const GpuMesh* mesh = r.FindMesh(static_cast<MeshHandle>(d.mesh));
        if (!mesh) continue;
        const bool skinned = d.boneCount > 0 && mesh->joints && mesh->weights;
        WireState::Variant& v = FlatVariant(r, st, format, skinned, wire);
        if (!v.pso || !v.srb) continue;
        {
            MapHelper<FlatConstants> map;
            FlatConstants* c = MapDiscard(r, map, st.cb);
            if (!c) continue;
            c->viewProj = r.viewProj;
            c->world    = d.world;
            c->color    = colorOf(d.surface);
        }
        if (skinned) {
            const Uint32 n = std::min<Uint32>(d.boneCount, kMaxJoints);
            MapHelper<glm::mat4> map;
            glm::mat4* b = MapDiscard(r, map, st.bonesCB);
            if (!b) continue;
            std::memcpy(b, bones.data() + d.boneOffset, n * sizeof(glm::mat4));
        }
        IBuffer* vbs[6] = { mesh->positions, nullptr, nullptr, nullptr, skinned ? mesh->joints.RawPtr() : nullptr,
                            skinned ? mesh->weights.RawPtr() : nullptr };
        const Uint64 offsets[6] = {};
        r.context->SetPipelineState(v.pso);
        r.context->CommitShaderResources(v.srb, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
        r.context->SetVertexBuffers(0, 6, vbs, offsets, RESOURCE_STATE_TRANSITION_MODE_TRANSITION, SET_VERTEX_BUFFERS_FLAG_RESET);
        r.context->SetIndexBuffer(mesh->indices, 0, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
        DrawIndexedAttribs draw{ mesh->indexCount, mesh->index32 ? VT_UINT32 : VT_UINT16, DRAW_FLAG_VERIFY_ALL };
        r.context->DrawIndexed(draw);
    }
}

} // namespace

void WireRun(DiligentSceneRenderer::Impl& r, ITextureView* outputRTV) {
    WireState* st = r.wire.get();
    if (!st || (r.wireDraws.empty() && r.outlineDraws.empty())) return;
    const TextureDesc& od = outputRTV->GetTexture()->GetDesc();
    if (static_cast<int>(od.Width) != r.width || static_cast<int>(od.Height) != r.height) {
        if (!st->warnedSize) std::fprintf(stderr, "[render] output target size differs from the frame size; wireframe and outlines skipped\n");
        st->warnedSize = true;
    } else {
        const TEXTURE_FORMAT format  = outputRTV->GetDesc().Format;
        const bool           srgbOut = GetTextureFormatAttribs(format).ComponentType == COMPONENT_TYPE_UNORM_SRGB;
        ITextureView*        dsv     = r.hdrDepth->GetDefaultView(TEXTURE_VIEW_DEPTH_STENCIL);

        // Outlines: the mask, then the edge pass onto the output.
        if (!r.outlineDraws.empty()) {
            if (!st->mask || st->maskW != r.width || st->maskH != r.height) {
                TextureDesc d;
                d.Name      = "render outline mask";
                d.Type      = RESOURCE_DIM_TEX_2D;
                d.Width     = static_cast<Uint32>(r.width);
                d.Height    = static_cast<Uint32>(r.height);
                d.Format    = kMaskFormat;
                d.BindFlags = BIND_RENDER_TARGET | BIND_SHADER_RESOURCE;
                st->mask.Release();
                r.device->CreateTexture(d, nullptr, &st->mask);
                st->maskW = r.width;
                st->maskH = r.height;
            }
            WireState::Variant& edge = EdgeVariant(r, *st, format);
            if (st->mask && edge.pso && edge.srb) {
                ITextureView* maskRTV = st->mask->GetDefaultView(TEXTURE_VIEW_RENDER_TARGET);
                const float   clear[4] = { 0.f, 0.f, 0.f, 0.f };
                r.context->SetRenderTargets(1, &maskRTV, dsv, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
                r.context->SetViewports(1, nullptr, 0, 0);
                r.context->ClearRenderTarget(maskRTV, clear, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
                // Linear colour into the sRGB mask; the edge pass reads it back linear and encodes for the output.
                DrawFlat(r, *st, kMaskFormat, false, r.outlineDraws, r.outlineBones,
                         [&](const DrawSurface& s) { return Encode(s.outlineColor, true); });
                bool edgeMapped = false;   // out of dynamic memory (MapDiscard): no edge pass this frame
                {
                    MapHelper<int> map;
                    if (int* c = MapDiscard(r, map, st->outlineCB)) {
                        c[0] = std::clamp(r.style.outlineWidth, 1, 8);
                        c[1] = r.width;
                        c[2] = r.height;
                        c[3] = 0;
                        edgeMapped = true;
                    }
                }
                if (edgeMapped) {
                    if (auto* var = edge.srb->GetVariableByName(SHADER_TYPE_PIXEL, "g_Mask"))
                        var->Set(st->mask->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE));
                    r.context->SetRenderTargets(1, &outputRTV, nullptr, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
                    r.context->SetViewports(1, nullptr, 0, 0);
                    r.context->SetPipelineState(edge.pso);
                    r.context->CommitShaderResources(edge.srb, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
                    DrawAttribs draw{ 3, DRAW_FLAG_VERIFY_ALL };
                    r.context->Draw(draw);
                }
            }
        }

        // Wireframe over shaded.
        if (!r.wireDraws.empty()) {
            if (!r.owner.supportsWireframe()) {
                if (!st->warnedNoWire) std::fprintf(stderr, "[render] wireframe over shaded needs wireframe fill; the device has none\n");
                st->warnedNoWire = true;
            } else {
                r.context->SetRenderTargets(1, &outputRTV, dsv, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
                r.context->SetViewports(1, nullptr, 0, 0);
                DrawFlat(r, *st, format, true, r.wireDraws, r.wireBones,
                         [&](const DrawSurface& s) { return Encode(s.wireColor, srgbOut); });
            }
        }
    }
    r.wireDraws.clear();
    r.wireBones.clear();
    r.outlineDraws.clear();
    r.outlineBones.clear();
}

} // namespace render::diligent
