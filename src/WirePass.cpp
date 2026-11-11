// Wireframe over shaded (DrawSurface::wireColor): after the tone map and the grid, before the overlays, every draw of the
// frame that asked for it is drawn again as edges in its flat colour (a display value, exact like an overlay), depth-tested
// against the scene with a small bias towards the camera so the edges of the surface itself show. Needs the device's
// wireframe fill; without it the request is skipped (said once).
#include "SceneRendererImpl.h"

#include "GraphicsAccessories.hpp"
#include "MapHelper.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>

namespace render::diligent {

using namespace Diligent;

namespace {

constexpr Uint32 kMaxJoints = 256;

constexpr const char* kWireVS = R"(
cbuffer WireDraw {
    float4x4 g_ViewProj;
    float4x4 g_World;
    float4   g_Color;
};
#if SKINNED
cbuffer WireBones { float4x4 g_Bones[256]; };
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

constexpr const char* kWirePS = R"(
cbuffer WireDraw {
    float4x4 g_ViewProj;
    float4x4 g_World;
    float4   g_Color;   // premultiplied, in the output's encoding
};
float4 main(in float4 pos : SV_POSITION) : SV_Target { return g_Color; }
)";

struct WireConstants {
    glm::mat4 viewProj, world;
    glm::vec4 color;
};

float SrgbToLinear(float c) { return c <= 0.04045f ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f); }

} // namespace

struct WireState {
    struct Variant {
        RefCntAutoPtr<IPipelineState>         pso;
        RefCntAutoPtr<IShaderResourceBinding> srb;
    };
    std::map<int, Variant> variants;   // key: output format << 1 | skinned
    RefCntAutoPtr<IBuffer> cb, bonesCB;
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
    st->cb      = cb("render wire CB", sizeof(WireConstants));
    st->bonesCB = cb("render wire bones CB", sizeof(glm::mat4) * kMaxJoints);
    return st;
}

namespace {

WireState::Variant& VariantFor(DiligentSceneRenderer::Impl& r, WireState& st, TEXTURE_FORMAT format, bool skinned) {
    const int key = (static_cast<int>(format) << 1) | (skinned ? 1 : 0);
    auto [it, inserted] = st.variants.try_emplace(key);
    WireState::Variant& v = it->second;
    if (!inserted) return v;
    const ShaderMacro macros[] = { { "SKINNED", skinned ? "1" : "0" } };
    auto vs = r.CompileShader("render wire VS", SHADER_TYPE_VERTEX, kWireVS, macros, 1);
    auto ps = r.CompileShader("render wire PS", SHADER_TYPE_PIXEL, kWirePS, macros, 1);
    if (!vs || !ps) return v;
    std::vector<LayoutElement> layout = { { 0, 0, 3, VT_FLOAT32, False } };
    if (skinned) {
        layout.push_back({ 3, 4, 4, VT_FLOAT32, False });
        layout.push_back({ 4, 5, 4, VT_FLOAT32, False });
    }
    GraphicsPipelineStateCreateInfo ci;
    ci.PSODesc.Name = "render wire PSO";
    GraphicsPipelineDesc& g = ci.GraphicsPipeline;
    g.NumRenderTargets                    = 1;
    g.RTVFormats[0]                       = format;
    g.DSVFormat                           = kSceneDepthFormat;
    g.PrimitiveTopology                   = PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    g.InputLayout                         = { layout.data(), static_cast<Uint32>(layout.size()) };
    g.RasterizerDesc.FillMode             = FILL_MODE_WIREFRAME;
    g.RasterizerDesc.CullMode             = CULL_MODE_NONE;
    g.RasterizerDesc.DepthBias            = -256;   // the surface's own edges sit on its depth: pull them in front
    g.RasterizerDesc.SlopeScaledDepthBias = -2.f;
    g.DepthStencilDesc.DepthEnable        = True;
    g.DepthStencilDesc.DepthWriteEnable   = False;
    g.DepthStencilDesc.DepthFunc          = COMPARISON_FUNC_LESS_EQUAL;
    RenderTargetBlendDesc& rt = g.BlendDesc.RenderTargets[0];   // premultiplied over
    rt.BlendEnable = True;
    rt.SrcBlend = rt.SrcBlendAlpha = BLEND_FACTOR_ONE;
    rt.DestBlend = rt.DestBlendAlpha = BLEND_FACTOR_INV_SRC_ALPHA;
    ci.pVS = vs;
    ci.pPS = ps;
    ci.PSODesc.ResourceLayout.DefaultVariableType = SHADER_RESOURCE_VARIABLE_TYPE_STATIC;
    r.device->CreateGraphicsPipelineState(ci, &v.pso);
    if (!v.pso) return v;
    for (SHADER_TYPE stage : { SHADER_TYPE_VERTEX, SHADER_TYPE_PIXEL })
        if (auto* var = v.pso->GetStaticVariableByName(stage, "WireDraw")) var->Set(st.cb);
    if (skinned)
        if (auto* var = v.pso->GetStaticVariableByName(SHADER_TYPE_VERTEX, "WireBones")) var->Set(st.bonesCB);
    v.pso->CreateShaderResourceBinding(&v.srb, true);
    return v;
}

} // namespace

void WireRun(DiligentSceneRenderer::Impl& r, ITextureView* outputRTV) {
    WireState* st = r.wire.get();
    if (!st || r.wireDraws.empty()) return;
    if (!r.owner.supportsWireframe()) {
        if (!st->warnedNoWire) std::fprintf(stderr, "[render] wireframe over shaded needs wireframe fill; the device has none\n");
        st->warnedNoWire = true;
        return;
    }
    const TextureDesc& od = outputRTV->GetTexture()->GetDesc();
    if (static_cast<int>(od.Width) != r.width || static_cast<int>(od.Height) != r.height) {
        if (!st->warnedSize) std::fprintf(stderr, "[render] output target size differs from the frame size; wireframe skipped\n");
        st->warnedSize = true;
        return;
    }
    const TEXTURE_FORMAT format = outputRTV->GetDesc().Format;
    const bool           srgbOut = GetTextureFormatAttribs(format).ComponentType == COMPONENT_TYPE_UNORM_SRGB;
    ITextureView*        dsv     = r.hdrDepth->GetDefaultView(TEXTURE_VIEW_DEPTH_STENCIL);
    r.context->SetRenderTargets(1, &outputRTV, dsv, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    r.context->SetViewports(1, nullptr, 0, 0);

    for (const QueuedDraw& d : r.wireDraws) {
        const GpuMesh* mesh = r.FindMesh(static_cast<MeshHandle>(d.mesh));
        if (!mesh) continue;
        const bool skinned = d.boneCount > 0 && mesh->joints && mesh->weights;
        WireState::Variant& v = VariantFor(r, *st, format, skinned);
        if (!v.pso || !v.srb) continue;
        {
            const Rgba8  c = d.surface.wireColor;
            const float  a = c.a / 255.f;
            auto channel   = [&](std::uint8_t x) { const float f = x / 255.f; return (srgbOut ? SrgbToLinear(f) : f) * a; };
            MapHelper<WireConstants> w(r.context, st->cb, MAP_WRITE, MAP_FLAG_DISCARD);
            w->viewProj = r.viewProj;
            w->world    = d.world;
            w->color    = { channel(c.r), channel(c.g), channel(c.b), a };
        }
        if (skinned) {
            const Uint32 n = std::min<Uint32>(d.boneCount, kMaxJoints);
            MapHelper<glm::mat4> b(r.context, st->bonesCB, MAP_WRITE, MAP_FLAG_DISCARD);
            std::memcpy(static_cast<glm::mat4*>(b), r.wireBones.data() + d.boneOffset, n * sizeof(glm::mat4));
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
    r.wireDraws.clear();
    r.wireBones.clear();
}

} // namespace render::diligent
