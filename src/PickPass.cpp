// Picking (ISceneRenderer::RequestPick / TakePicks). At EndFrame, after the scene flush, the frame's pickable draws
// (DrawSurface::pickId != 0, kept by FlushQueue) are drawn again into an id target (R32_UINT) and a depth-value target
// (R32_FLOAT) over a depth buffer of their own; then only the picked rect is copied to staging textures and read back.
// Alpha-tested and blended surfaces cut at their threshold through the diffuse alpha, so a pick goes through the holes in
// a leaf card as the eye does. TakePicks waits for the GPU.
#include "MathConv.h"
#include "SceneRendererImpl.h"

#include "CommonlyUsedStates.h"
#include "MapHelper.hpp"

#include <algorithm>
#include <cstring>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>

namespace render::diligent {

using namespace Diligent;

namespace {

constexpr Uint32 kMaxJoints = 256;

constexpr const char* kPickVS = R"(
cbuffer PickDraw {
    float4x4 g_ViewProj;
    float4x4 g_World;
    float4   g_Params;   // y = alpha cut, z = tint alpha
    uint4    g_Id;
};
#if SKINNED
cbuffer PickBones { float4x4 g_Bones[256]; };
#endif
struct VSIn {
    float3 pos : ATTRIB0;
    float2 uv  : ATTRIB2;
#if SKINNED
    float4 joints  : ATTRIB3;
    float4 weights : ATTRIB4;
#endif
};
struct PSIn { float4 pos : SV_POSITION; float2 uv : TEX_COORD; };
void main(in VSIn i, out PSIn o) {
    float4x4 model = g_World;
#if SKINNED
    float4x4 skin = g_Bones[(int)i.joints.x] * i.weights.x + g_Bones[(int)i.joints.y] * i.weights.y +
                    g_Bones[(int)i.joints.z] * i.weights.z + g_Bones[(int)i.joints.w] * i.weights.w;
    model = mul(g_World, skin);
#endif
    o.pos = mul(g_ViewProj, mul(model, float4(i.pos, 1.0)));
    o.uv  = i.uv;
}
)";

constexpr const char* kPickPS = R"(
cbuffer PickDraw {
    float4x4 g_ViewProj;
    float4x4 g_World;
    float4   g_Params;
    uint4    g_Id;
};
#if ALPHA_CUT
Texture2D<float4> g_Diffuse; SamplerState g_Diffuse_sampler;
#endif
struct PSIn { float4 pos : SV_POSITION; float2 uv : TEX_COORD; };
struct PSOut { uint id : SV_Target0; float depth : SV_Target1; };
PSOut main(in PSIn i) {
#if ALPHA_CUT
    if (g_Diffuse.Sample(g_Diffuse_sampler, i.uv).a * g_Params.z <= g_Params.y) discard;
#endif
    PSOut o;
    o.id    = g_Id.x;
    o.depth = i.pos.z;
    return o;
}
)";

struct PickConstants {
    glm::mat4  viewProj, world;
    glm::vec4  params;
    glm::uvec4 id;
};

} // namespace

struct PickState {
    struct Variant {
        RefCntAutoPtr<IPipelineState>         pso;
        RefCntAutoPtr<IShaderResourceBinding> srb;
        IShaderResourceVariable*              diffuse = nullptr;
    };
    std::map<int, Variant>  variants;   // key: skinned | alphaCut << 1 | twoSided << 2
    RefCntAutoPtr<IBuffer>  cb, bonesCB;
    RefCntAutoPtr<ITexture> id, depthValue, depth;   // frame-sized
    RefCntAutoPtr<ITexture> idStaging, depthStaging; // rect-sized
    int width = 0, height = 0, rectW = 0, rectH = 0;
};

void DestroyPickState(PickState* p) { delete p; }

namespace {

RefCntAutoPtr<ITexture> MakeTarget(IRenderDevice* device, const char* name, int w, int h, TEXTURE_FORMAT format, BIND_FLAGS bind,
                                   USAGE usage = USAGE_DEFAULT) {
    TextureDesc d;
    d.Name      = name;
    d.Type      = RESOURCE_DIM_TEX_2D;
    d.Width     = static_cast<Uint32>(w);
    d.Height    = static_cast<Uint32>(h);
    d.Format    = format;
    d.BindFlags = bind;
    d.Usage     = usage;
    if (usage == USAGE_STAGING) d.CPUAccessFlags = CPU_ACCESS_READ;
    RefCntAutoPtr<ITexture> t;
    device->CreateTexture(d, nullptr, &t);
    return t;
}

PickState::Variant& VariantFor(DiligentSceneRenderer::Impl& r, PickState& st, bool skinned, bool alphaCut, bool twoSided) {
    const int key = (skinned ? 1 : 0) | (alphaCut ? 2 : 0) | (twoSided ? 4 : 0);
    auto [it, inserted] = st.variants.try_emplace(key);
    PickState::Variant& v = it->second;
    if (!inserted) return v;

    const ShaderMacro macros[] = { { "SKINNED", skinned ? "1" : "0" }, { "ALPHA_CUT", alphaCut ? "1" : "0" } };
    auto vs = r.CompileShader("render pick VS", SHADER_TYPE_VERTEX, kPickVS, macros, _countof(macros));
    auto ps = r.CompileShader("render pick PS", SHADER_TYPE_PIXEL, kPickPS, macros, _countof(macros));
    if (!vs || !ps) return v;

    std::vector<LayoutElement> layout = { { 0, 0, 3, VT_FLOAT32, False }, { 2, 2, 2, VT_FLOAT32, False } };
    if (skinned) {
        layout.push_back({ 3, 4, 4, VT_FLOAT32, False });
        layout.push_back({ 4, 5, 4, VT_FLOAT32, False });
    }
    GraphicsPipelineStateCreateInfo ci;
    ci.PSODesc.Name = "render pick PSO";
    GraphicsPipelineDesc& g = ci.GraphicsPipeline;
    g.NumRenderTargets                     = 2;
    g.RTVFormats[0]                        = TEX_FORMAT_R32_UINT;
    g.RTVFormats[1]                        = TEX_FORMAT_R32_FLOAT;
    g.DSVFormat                            = kSceneDepthFormat;
    g.PrimitiveTopology                    = PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    g.InputLayout                          = { layout.data(), static_cast<Uint32>(layout.size()) };
    g.RasterizerDesc.FrontCounterClockwise = True;
    g.RasterizerDesc.CullMode              = twoSided ? CULL_MODE_NONE : CULL_MODE_BACK;
    g.DepthStencilDesc.DepthEnable         = True;
    g.DepthStencilDesc.DepthWriteEnable    = True;
    g.DepthStencilDesc.DepthFunc           = COMPARISON_FUNC_LESS_EQUAL;
    ci.pVS = vs;
    ci.pPS = ps;
    const ShaderResourceVariableDesc vars[]     = { { SHADER_TYPE_PIXEL, "g_Diffuse", SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC } };
    const ImmutableSamplerDesc       samplers[] = { { SHADER_TYPE_PIXEL, "g_Diffuse", Sam_LinearWrap } };
    ci.PSODesc.ResourceLayout.DefaultVariableType = SHADER_RESOURCE_VARIABLE_TYPE_STATIC;
    if (alphaCut) {
        ci.PSODesc.ResourceLayout.Variables            = vars;
        ci.PSODesc.ResourceLayout.NumVariables         = 1;
        ci.PSODesc.ResourceLayout.ImmutableSamplers    = samplers;
        ci.PSODesc.ResourceLayout.NumImmutableSamplers = 1;
    }
    r.device->CreateGraphicsPipelineState(ci, &v.pso);
    if (!v.pso) return v;
    for (SHADER_TYPE stage : { SHADER_TYPE_VERTEX, SHADER_TYPE_PIXEL })
        if (auto* var = v.pso->GetStaticVariableByName(stage, "PickDraw")) var->Set(st.cb);
    if (skinned)
        if (auto* var = v.pso->GetStaticVariableByName(SHADER_TYPE_VERTEX, "PickBones")) var->Set(st.bonesCB);
    v.pso->CreateShaderResourceBinding(&v.srb, true);
    if (v.srb && alphaCut) v.diffuse = v.srb->GetVariableByName(SHADER_TYPE_PIXEL, "g_Diffuse");
    return v;
}

void EnsureTargets(DiligentSceneRenderer::Impl& r, PickState& st, int rectW, int rectH) {
    if (!st.id || st.width != r.width || st.height != r.height) {
        st.width      = r.width;
        st.height     = r.height;
        st.id         = MakeTarget(r.device, "render pick id", r.width, r.height, TEX_FORMAT_R32_UINT, BIND_RENDER_TARGET);
        st.depthValue = MakeTarget(r.device, "render pick depth value", r.width, r.height, TEX_FORMAT_R32_FLOAT, BIND_RENDER_TARGET);
        st.depth      = MakeTarget(r.device, "render pick depth", r.width, r.height, kSceneDepthFormat, BIND_DEPTH_STENCIL);
    }
    if (!st.idStaging || st.rectW != rectW || st.rectH != rectH) {
        st.rectW        = rectW;
        st.rectH        = rectH;
        st.idStaging    = MakeTarget(r.device, "render pick id readback", rectW, rectH, TEX_FORMAT_R32_UINT, BIND_NONE, USAGE_STAGING);
        st.depthStaging = MakeTarget(r.device, "render pick depth readback", rectW, rectH, TEX_FORMAT_R32_FLOAT, BIND_NONE, USAGE_STAGING);
    }
}

} // namespace

PickPtr CreatePickState(DiligentSceneRenderer::Impl& r) {
    PickPtr st(new PickState{}, &DestroyPickState);
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
    st->cb      = cb("render pick CB", sizeof(PickConstants));
    st->bonesCB = cb("render pick bones CB", sizeof(glm::mat4) * kMaxJoints);
    return st;
}

void PickRun(DiligentSceneRenderer::Impl& r) {
    r.pickRequested = false;
    r.pickHits.clear();
    PickState* st = r.pick.get();
    const int x0 = std::clamp(r.pickX0, 0, r.width), x1 = std::clamp(r.pickX1, 0, r.width);
    const int y0 = std::clamp(r.pickY0, 0, r.height), y1 = std::clamp(r.pickY1, 0, r.height);
    if (!st || !st->cb || x1 <= x0 || y1 <= y0 || r.pickDraws.empty()) {
        r.pickDraws.clear();
        r.pickBones.clear();
        return;
    }
    EnsureTargets(r, *st, x1 - x0, y1 - y0);
    if (!st->id || !st->depthValue || !st->depth || !st->idStaging || !st->depthStaging) return;

    ITextureView* rtvs[2] = { st->id->GetDefaultView(TEXTURE_VIEW_RENDER_TARGET), st->depthValue->GetDefaultView(TEXTURE_VIEW_RENDER_TARGET) };
    ITextureView* dsv     = st->depth->GetDefaultView(TEXTURE_VIEW_DEPTH_STENCIL);
    r.context->SetRenderTargets(2, rtvs, dsv, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    r.context->SetViewports(1, nullptr, 0, 0);
    const float clearId[4] = { 0.f, 0.f, 0.f, 0.f }, clearDepth[4] = { 1.f, 0.f, 0.f, 0.f };
    r.context->ClearRenderTarget(rtvs[0], clearId, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    r.context->ClearRenderTarget(rtvs[1], clearDepth, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    r.context->ClearDepthStencil(dsv, CLEAR_DEPTH_FLAG, 1.f, 0, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);

    using Blend = DrawSurface::BlendMode;
    for (const QueuedDraw& d : r.pickDraws) {
        const GpuMesh* mesh = r.FindMesh(static_cast<MeshHandle>(d.mesh));
        if (!mesh) continue;
        const DrawSurface& s = d.surface;
        GpuTexture* diffuse  = r.FindTexture(s.diffuse);
        if (diffuse && diffuse->cube) diffuse = nullptr;
        const bool skinned  = d.boneCount > 0 && mesh->joints && mesh->weights;
        const bool cutting  = s.blendMode != Blend::Opaque || s.xray;
        const bool alphaCut = cutting && diffuse;
        PickState::Variant& v = VariantFor(r, *st, skinned, alphaCut, s.twoSided || s.xray);
        if (!v.pso || !v.srb) continue;
        {
            MapHelper<PickConstants> c(r.context, st->cb, MAP_WRITE, MAP_FLAG_DISCARD);
            c->viewProj = r.viewProj;
            c->world    = d.world;
            // Tested surfaces cut where the eye sees a hole; blended ones where they are nearly clear.
            const bool tested = s.blendMode == Blend::AlphaTest || s.blendMode == Blend::AlphaTestAndBlend;
            c->params = { 0.f, tested ? s.alphaThreshold : 0.1f, s.tint.a, 0.f };
            c->id     = { s.pickId, 0u, 0u, 0u };
        }
        if (skinned) {
            const Uint32 n = std::min<Uint32>(d.boneCount, kMaxJoints);
            MapHelper<glm::mat4> b(r.context, st->bonesCB, MAP_WRITE, MAP_FLAG_DISCARD);
            std::memcpy(static_cast<glm::mat4*>(b), r.pickBones.data() + d.boneOffset, n * sizeof(glm::mat4));
        }
        if (v.diffuse) v.diffuse->Set(diffuse->srv);
        IBuffer* vbs[6] = { mesh->positions, nullptr, mesh->uvs, nullptr, skinned ? mesh->joints.RawPtr() : nullptr,
                            skinned ? mesh->weights.RawPtr() : nullptr };
        const Uint64 offsets[6] = {};
        r.context->SetPipelineState(v.pso);
        r.context->CommitShaderResources(v.srb, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
        r.context->SetVertexBuffers(0, 6, vbs, offsets, RESOURCE_STATE_TRANSITION_MODE_TRANSITION, SET_VERTEX_BUFFERS_FLAG_RESET);
        r.context->SetIndexBuffer(mesh->indices, 0, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
        DrawIndexedAttribs draw{ mesh->indexCount, mesh->index32 ? VT_UINT32 : VT_UINT16, DRAW_FLAG_VERIFY_ALL };
        r.context->DrawIndexed(draw);
    }
    r.pickDraws.clear();
    r.pickBones.clear();

    // Only the rect goes to the CPU.
    Box box{ static_cast<Uint32>(x0), static_cast<Uint32>(x1), static_cast<Uint32>(y0), static_cast<Uint32>(y1) };
    for (auto [src, dst] : { std::pair{ st->id.RawPtr(), st->idStaging.RawPtr() }, std::pair{ st->depthValue.RawPtr(), st->depthStaging.RawPtr() } }) {
        CopyTextureAttribs copy{ src, RESOURCE_STATE_TRANSITION_MODE_TRANSITION, dst, RESOURCE_STATE_TRANSITION_MODE_TRANSITION };
        copy.pSrcBox = &box;
        r.context->CopyTexture(copy);
    }
    r.pickX0 = x0;
    r.pickY0 = y0;
    r.BindSceneTargets();
}

void DiligentSceneRenderer::RequestPick(int x0, int y0, int x1, int y1) {
    Impl& r = *m_impl;
    if (!r.inFrame) return;
    r.pickRequested = true;
    r.pickX0 = std::min(x0, x1);
    r.pickY0 = std::min(y0, y1);
    r.pickX1 = std::max(x0, x1);
    r.pickY1 = std::max(y0, y1);
    r.pickHits.clear();
}

std::vector<PickHit> DiligentSceneRenderer::TakePicks() {
    Impl&      r  = *m_impl;
    PickState* st = r.pick.get();
    if (!st || !st->idStaging || st->rectW <= 0) return {};
    if (!r.pickHits.empty()) return std::exchange(r.pickHits, {});
    r.context->WaitForIdle();

    MappedTextureSubresource ids{}, depths{};
    r.context->MapTextureSubresource(st->idStaging, 0, 0, MAP_READ, MAP_FLAG_DO_NOT_WAIT, nullptr, ids);
    r.context->MapTextureSubresource(st->depthStaging, 0, 0, MAP_READ, MAP_FLAG_DO_NOT_WAIT, nullptr, depths);
    std::unordered_map<std::uint32_t, PickHit> nearest;
    if (ids.pData && depths.pData) {
        const glm::mat4 inv = glm::inverse(r.viewProj);
        for (int y = 0; y < st->rectH; ++y) {
            const auto* idRow    = reinterpret_cast<const std::uint32_t*>(static_cast<const std::uint8_t*>(ids.pData) + y * ids.Stride);
            const auto* depthRow = reinterpret_cast<const float*>(static_cast<const std::uint8_t*>(depths.pData) + y * depths.Stride);
            for (int x = 0; x < st->rectW; ++x) {
                const std::uint32_t id = idRow[x];
                if (!id) continue;
                const float depth = depthRow[x];
                auto [it, fresh] = nearest.try_emplace(id);
                if (!fresh && it->second.depth <= depth) continue;
                PickHit& hit = it->second;
                hit.id    = id;
                hit.depth = depth;
                hit.x     = r.pickX0 + x;
                hit.y     = r.pickY0 + y;
                // Pixel centre to NDC (y up), depth 0..1 as the scene's projection writes it.
                const float     nx  = (static_cast<float>(hit.x) + 0.5f) / static_cast<float>(r.width) * 2.f - 1.f;
                const float     ny  = 1.f - (static_cast<float>(hit.y) + 0.5f) / static_cast<float>(r.height) * 2.f;
                const glm::vec4 w   = inv * glm::vec4(nx, ny, depth, 1.f);
                hit.position        = glm::vec3(w) / w.w;
            }
        }
    }
    if (ids.pData) r.context->UnmapTextureSubresource(st->idStaging, 0, 0);
    if (depths.pData) r.context->UnmapTextureSubresource(st->depthStaging, 0, 0);
    st->rectW = 0;   // answered: a second TakePicks returns nothing until the next pick
    st->idStaging.Release();
    st->depthStaging.Release();

    std::vector<PickHit> out;
    out.reserve(nearest.size());
    for (auto& [id, hit] : nearest) out.push_back(hit);
    std::ranges::sort(out, {}, &PickHit::depth);
    return out;
}

} // namespace render::diligent
