// Mesh shading through DiligentFX's PBR_Renderer.
//
// We use the renderer's shaders, signature, PSO caches and default textures, but
// drive it ourselves: the caller's render targets are bound, constant buffers are
// written per draw, and DrawIndexed is ours. See MaterialMapping.h for how a
// DrawSurface becomes PSO flags + material constants.
#include "MaterialMapping.h"
#include "MathConv.h"
#include "SceneRendererImpl.h"

#include "BasicMath.hpp"
#include "GLTF_PBR_Renderer.hpp"
#include "MapHelper.hpp"
#include "PBR_Renderer.hpp"

#include <glm/gtc/packing.hpp>

#include <array>
#include <cstdio>
#include <cstring>

namespace Diligent::HLSL {
#include "Shaders/Common/public/BasicStructures.fxh"
#include "Shaders/PBR/public/PBR_Structures.fxh"
#include "Shaders/PBR/private/RenderPBR_Structures.fxh"
} // namespace Diligent::HLSL

namespace render::diligent {

using namespace Diligent;

namespace {

constexpr Uint32 kMaxJoints = 256;

// One buffer slot per attribute, slot = PBR attribute index. Static storage:
// PBR_Renderer copies the layout, but keep it valid regardless.
const LayoutElement kInputLayout[] = {
    { PBR_Renderer::VERTEX_ATTRIB_ID_POSITION, 0, 3, VT_FLOAT32, False },
    { PBR_Renderer::VERTEX_ATTRIB_ID_NORMAL, 1, 3, VT_FLOAT32, False },
    { PBR_Renderer::VERTEX_ATTRIB_ID_TEXCOORD0, 2, 2, VT_FLOAT32, False },
    { PBR_Renderer::VERTEX_ATTRIB_ID_JOINTS, 4, 4, VT_FLOAT32, False },
    { PBR_Renderer::VERTEX_ATTRIB_ID_WEIGHTS, 5, 4, VT_FLOAT32, False },
    { PBR_Renderer::VERTEX_ATTRIB_ID_COLOR, 6, 4, VT_UINT8, True },
};
constexpr Uint32 kNumVertexSlots = 7;

// The material textures we bind, in a fixed order (also the SRB cache key order).
constexpr PBR_Renderer::TEXTURE_ATTRIB_ID kTexAttribs[] = {
    PBR_Renderer::TEXTURE_ATTRIB_ID_BASE_COLOR, PBR_Renderer::TEXTURE_ATTRIB_ID_NORMAL,
    PBR_Renderer::TEXTURE_ATTRIB_ID_PHYS_DESC,  PBR_Renderer::TEXTURE_ATTRIB_ID_OCCLUSION,
    PBR_Renderer::TEXTURE_ATTRIB_ID_EMISSIVE,   PBR_Renderer::TEXTURE_ATTRIB_ID_SPECULAR,
};
constexpr std::size_t kNumTex = std::size(kTexAttribs);

// IBL source: v2's sky/ground gradient, at unit intensity (SetLight's ambient
// tints it via IBLScale, as u_ambientColor does in the GL renderer).
constexpr int kIblFaceSize = 32;

// Direction for texel (u, v) of cube face `face` (D3D/Vulkan face order and
// orientation, the same as GL's). World axes: +Z is the sky.
glm::vec3 CubeFaceDir(int face, float u, float v) {
    switch (face) {
        case 0: return glm::normalize(glm::vec3(1.f, -v, -u));    // +X
        case 1: return glm::normalize(glm::vec3(-1.f, -v, u));    // -X
        case 2: return glm::normalize(glm::vec3(u, 1.f, v));      // +Y
        case 3: return glm::normalize(glm::vec3(u, -1.f, -v));    // -Y
        case 4: return glm::normalize(glm::vec3(u, -v, 1.f));     // +Z
        default: return glm::normalize(glm::vec3(-u, -v, -1.f));  // -Z
    }
}

glm::vec3 SkyColor(const glm::vec3& d) {
    const glm::vec3 skyTop{ 0.55f, 0.68f, 1.00f };
    const glm::vec3 skyHorizon{ 0.85f, 0.89f, 1.00f };
    const glm::vec3 groundCol{ 0.28f, 0.25f, 0.22f };
    if (d.z >= 0.f) return glm::mix(skyHorizon, skyTop, glm::sqrt(d.z));
    const float t = glm::clamp(-d.z * 4.f, 0.f, 1.f);   // soften the terminator
    return glm::mix(skyHorizon * 0.65f, groundCol, t);
}

float4x4 ToFloat4x4(const glm::mat4& m) {
    float4x4 out;
    StoreMatrix(&out, m);
    return out;
}

} // namespace

struct PbrState {
    std::unique_ptr<PBR_Renderer> renderer;
    PBR_Renderer::PSO_FLAGS       enabledFlags = PBR_Renderer::PSO_FLAG_NONE;
    // [wireframe][depthWrite]
    PBR_Renderer::PsoCacheAccessor accessors[2][2];

    RefCntAutoPtr<IBuffer>  frameCB;
    RefCntAutoPtr<ITexture> envCube, irradiance, prefiltered;
    bool                    frameDirty = true;

    // SRBs keyed by the texture ids bound to kTexAttribs (0 = renderer default).
    // The specular slot's key is id·4 + view kind (0 normal alpha, 1 RMAOS, 2 red → alpha).
    using SrbKey = std::array<std::uint32_t, kNumTex>;
    std::map<SrbKey, RefCntAutoPtr<IShaderResourceBinding>> srbs;

    bool warnedJoints = false;
};

void DestroyPbrState(PbrState* p) { delete p; }

namespace {

void BuildIbl(DiligentSceneRenderer::Impl& r, PbrState& st) {
    TextureDesc desc;
    desc.Name      = "render IBL source";
    desc.Type      = RESOURCE_DIM_TEX_CUBE;
    desc.Width     = kIblFaceSize;
    desc.Height    = kIblFaceSize;
    desc.ArraySize = 6;
    desc.MipLevels = 0;   // full chain
    desc.Format    = TEX_FORMAT_RGBA16_FLOAT;
    desc.BindFlags = BIND_SHADER_RESOURCE | BIND_RENDER_TARGET;
    desc.MiscFlags = MISC_TEXTURE_FLAG_GENERATE_MIPS;
    r.device->CreateTexture(desc, nullptr, &st.envCube);
    if (!st.envCube) return;

    std::vector<std::uint16_t> face(kIblFaceSize * kIblFaceSize * 4);
    for (int f = 0; f < 6; ++f) {
        for (int y = 0; y < kIblFaceSize; ++y)
            for (int x = 0; x < kIblFaceSize; ++x) {
                const float     u = ((x + 0.5f) / kIblFaceSize) * 2.f - 1.f;
                const float     v = ((y + 0.5f) / kIblFaceSize) * 2.f - 1.f;
                const glm::vec3 c = SkyColor(CubeFaceDir(f, u, v));
                std::uint16_t*  t = &face[(static_cast<std::size_t>(y) * kIblFaceSize + x) * 4];
                t[0] = glm::packHalf1x16(c.r);
                t[1] = glm::packHalf1x16(c.g);
                t[2] = glm::packHalf1x16(c.b);
                t[3] = glm::packHalf1x16(1.f);
            }
        TextureSubResData sub{ face.data(), kIblFaceSize * 4 * sizeof(std::uint16_t) };
        Box               box{ 0, kIblFaceSize, 0, kIblFaceSize };
        r.context->UpdateTexture(st.envCube, 0, static_cast<Uint32>(f), box, sub,
                                 RESOURCE_STATE_TRANSITION_MODE_TRANSITION, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    }
    r.context->GenerateMips(st.envCube->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE));

    st.irradiance  = st.renderer->CreateIrradianceCube(r.context);
    st.prefiltered = st.renderer->CreatePrefilteredEnvMap(r.context);
    PBR_Renderer::PrecomputeCubemapsAttribs pre;
    pre.pEnvironmentMapSRV = st.envCube->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE);
    pre.pIrradianceCube    = st.irradiance;
    pre.pPrefilteredEnvMap = st.prefiltered;
    st.renderer->PrecomputeCubemaps(r.context, pre);
}

void WriteFrameAttribs(DiligentSceneRenderer::Impl& r, PbrState& st) {
    MapHelper<Uint8> dst(r.context, st.frameCB, MAP_WRITE, MAP_FLAG_DISCARD);
    std::memset(dst, 0, st.renderer->GetPRBFrameAttribsSize());
    auto* frame  = reinterpret_cast<HLSL::PBRFrameAttribs*>(static_cast<Uint8*>(dst));
    auto* lights = reinterpret_cast<HLSL::PBRLightAttribs*>(frame + 1);

    HLSL::CameraAttribs& cam = frame->Camera;
    cam.f4Position     = float4{ r.cameraPos.x, r.cameraPos.y, r.cameraPos.z, 1.f };
    cam.f4ViewportSize = float4{ static_cast<float>(r.width), static_cast<float>(r.height), 1.f / r.width, 1.f / r.height };
    cam.fHandness      = glm::determinant(r.view) > 0.f ? 1.f : -1.f;
    cam.fFocusDistance = 10.f;
    cam.fFStop         = 5.6f;
    cam.fFocalLength   = 50.f;
    cam.fSensorWidth   = 36.f;
    cam.fSensorHeight  = 24.f;
    StoreMatrix(&cam.mView, r.view);
    StoreMatrix(&cam.mProj, r.projZO);
    StoreMatrix(&cam.mViewProj, r.viewProj);
    StoreMatrix(&cam.mViewInv, glm::inverse(r.view));
    StoreMatrix(&cam.mProjInv, glm::inverse(r.projZO));
    StoreMatrix(&cam.mViewProjInv, glm::inverse(r.viewProj));
    frame->PrevCamera = cam;

    HLSL::PBRRendererShaderParameters& p = frame->Renderer;
    st.renderer->SetInternalShaderParameters(p, st.prefiltered ? st.prefiltered->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE) : nullptr);
    p.AverageLogLum     = 0.3f;   // unused (no PBR tone mapping) but must be sane
    p.MiddleGray        = 0.18f;
    p.WhitePoint        = 3.f;
    p.IBLScale          = float4{ r.ambient.x, r.ambient.y, r.ambient.z, 1.f };
    p.OcclusionStrength = 1.f;
    p.EmissionScale     = 1.f;
    p.PointSize         = 1.f;
    p.LightCount        = 1;

    // Our light direction points towards the light; PBR's is the light's travel.
    const glm::vec3 travel = -glm::normalize(r.lightDir);
    HLSL::PBRLightAttribs& L = lights[0];
    L.Type           = PBR_Renderer::LIGHT_TYPE_DIRECTIONAL;
    L.DirectionX     = travel.x;
    L.DirectionY     = travel.y;
    L.DirectionZ     = travel.z;
    L.ShadowMapIndex = -1;
    // Both renderers divide the Lambert term by π, so the colour maps 1:1.
    L.IntensityR = r.lightColor.r;
    L.IntensityG = r.lightColor.g;
    L.IntensityB = r.lightColor.b;
}

PBR_Renderer::PSO_FLAGS FlagsFor(const MappedMaterial& m) {
    using F = PBR_Renderer::PSO_FLAGS;
    F f = PBR_Renderer::PSO_FLAG_USE_VERTEX_NORMALS | PBR_Renderer::PSO_FLAG_USE_TEXCOORD0 | PBR_Renderer::PSO_FLAG_USE_LIGHTS |
          PBR_Renderer::PSO_FLAG_USE_IBL | PBR_Renderer::PSO_FLAG_ENABLE_SPECULAR;
    if (m.colorMap) f |= PBR_Renderer::PSO_FLAG_USE_COLOR_MAP;
    if (m.normalMap) f |= PBR_Renderer::PSO_FLAG_USE_NORMAL_MAP;
    if (m.physDescMap) f |= PBR_Renderer::PSO_FLAG_USE_PHYS_DESC_MAP;
    if (m.aoMap) f |= PBR_Renderer::PSO_FLAG_USE_AO_MAP;
    if (m.emissiveMap) f |= PBR_Renderer::PSO_FLAG_USE_EMISSIVE_MAP;
    if (m.specSource != MappedMaterial::SpecSource::None) f |= PBR_Renderer::PSO_FLAG_USE_SPECULAR_MAP;
    if (m.vertexColors) f |= PBR_Renderer::PSO_FLAG_USE_VERTEX_COLORS;
    if (m.joints) f |= PBR_Renderer::PSO_FLAG_USE_JOINTS;
    return f;
}

PBR_Renderer::ALPHA_MODE AlphaFor(const MappedMaterial& m) {
    switch (m.alpha) {
        case MappedMaterial::Alpha::Mask: return PBR_Renderer::ALPHA_MODE_MASK;
        case MappedMaterial::Alpha::Blend: return PBR_Renderer::ALPHA_MODE_BLEND;
        default: return PBR_Renderer::ALPHA_MODE_OPAQUE;
    }
}

// PBRMaterialShaderInfo for these flags: Basic, Specular (ENABLE_SPECULAR), then
// one PBRMaterialTextureAttribs per enabled texture, in flag-bit order.
void WriteMaterialAttribs(Uint8* dst, PBR_Renderer::PSO_FLAGS flags, const MappedMaterial& m, PBR_Renderer::ALPHA_MODE alpha) {
    auto* basic = reinterpret_cast<HLSL::PBRMaterialBasicAttribs*>(dst);
    *basic = {};
    basic->BaseColorFactor = float4{ m.baseColorFactor.r, m.baseColorFactor.g, m.baseColorFactor.b, m.baseColorFactor.a };
    basic->EmissiveFactorR = m.emissiveFactor.r;
    basic->EmissiveFactorG = m.emissiveFactor.g;
    basic->EmissiveFactorB = m.emissiveFactor.b;
    basic->NormalScale     = 1.f;
    basic->SpecularFactorR = basic->SpecularFactorG = basic->SpecularFactorB = 1.f;
    basic->Workflow        = PBR_Renderer::PBR_WORKFLOW_METALL_ROUGH;
    basic->AlphaMode       = alpha;
    basic->AlphaMaskCutoff = m.alphaCutoff;
    basic->MetallicFactor  = m.metallicFactor;
    basic->RoughnessFactor = m.roughnessFactor;
    basic->OcclusionFactor = 1.f;
    dst += sizeof(HLSL::PBRMaterialBasicAttribs);

    if (flags & PBR_Renderer::PSO_FLAG_ENABLE_SPECULAR) {
        auto* spec = reinterpret_cast<HLSL::PBRMaterialSpecularAttribs*>(dst);
        spec->ColorFactorR = spec->ColorFactorG = spec->ColorFactorB = 1.f;
        spec->Factor = m.specularFactor;
        dst += sizeof(HLSL::PBRMaterialSpecularAttribs);
    }

    auto* tex = reinterpret_cast<HLSL::PBRMaterialTextureAttribs*>(dst);
    PBR_Renderer::ProcessTexturAttribs(flags, [&](int index, PBR_Renderer::TEXTURE_ATTRIB_ID) {
        HLSL::PBRMaterialTextureAttribs& t = tex[index];
        t = {};
        t.PackedProps         = 1u | (15u << 9u);   // UV0, wrap/wrap, all mips
        t.UVScaleAndRotation  = float4{ 1.f, 0.f, 0.f, 1.f };
        t.AtlasUVScaleAndBias = float4{ 1.f, 1.f, 0.f, 0.f };
    });
}

} // namespace

PbrPtr CreatePbrState(DiligentSceneRenderer::Impl& r) {
    PbrPtr st(new PbrState{}, &DestroyPbrState);

    PBR_Renderer::CreateInfo ci;
    ci.EnableIBL                      = true;
    ci.EnableAO                       = true;
    ci.EnableEmissive                 = true;
    ci.EnableSpecular                 = true;
    ci.PackMatrixRowMajor             = true;   // raw glm bytes (see MathConv.h)
    ci.PackVertexColors               = true;
    ci.MaxLightCount                  = 1;
    ci.MaxJointCount                  = kMaxJoints;
    ci.JointsBufferMode               = PBR_Renderer::JOINTS_BUFFER_MODE_UNIFORM;
    ci.MaxActiveMorphTargetCount      = 0;      // morphs are blended on the CPU
    ci.CreateDefaultMorphTargetBuffer = false;
    ci.TexColorConversionMode         = PBR_Renderer::CreateInfo::TEX_COLOR_CONVERSION_MODE_NONE;   // sRGB views
    ci.InputLayout                    = InputLayoutDesc{ kInputLayout, static_cast<Uint32>(std::size(kInputLayout)) };
    for (auto id : kTexAttribs) ci.TextureAttribIndices[id] = static_cast<int>(id);

    st->renderer     = std::make_unique<PBR_Renderer>(r.device, nullptr, r.context, ci);
    st->enabledFlags = PBR_Renderer::GetEnabledPSOFlags(ci);

    BufferDesc fb;
    fb.Name           = "render PBR frame attribs";
    fb.Size           = st->renderer->GetPRBFrameAttribsSize();
    fb.Usage          = USAGE_DYNAMIC;
    fb.BindFlags      = BIND_UNIFORM_BUFFER;
    fb.CPUAccessFlags = CPU_ACCESS_WRITE;
    r.device->CreateBuffer(fb, nullptr, &st->frameCB);

    for (int wire = 0; wire < 2; ++wire) {
        if (wire && !r.owner.supportsWireframe()) continue;
        for (int depthWrite = 0; depthWrite < 2; ++depthWrite) {
            GraphicsPipelineDesc gd;
            gd.NumRenderTargets                     = 1;
            gd.RTVFormats[0]                        = TEX_FORMAT_RGBA16_FLOAT;
            gd.DSVFormat                            = TEX_FORMAT_D32_FLOAT;
            gd.PrimitiveTopology                    = PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
            gd.RasterizerDesc.FrontCounterClockwise = True;
            gd.RasterizerDesc.FillMode              = wire ? FILL_MODE_WIREFRAME : FILL_MODE_SOLID;
            gd.DepthStencilDesc.DepthEnable         = True;
            gd.DepthStencilDesc.DepthWriteEnable    = depthWrite ? True : False;
            gd.DepthStencilDesc.DepthFunc           = COMPARISON_FUNC_LESS_EQUAL;
            st->accessors[wire][depthWrite]         = st->renderer->GetPsoCacheAccessor(gd);
        }
    }

    BuildIbl(r, *st);

    // Prewarm the common permutations (static textured, alpha-tested, skinned).
    const PBR_Renderer::PSO_FLAGS base = (PBR_Renderer::PSO_FLAG_USE_VERTEX_NORMALS | PBR_Renderer::PSO_FLAG_USE_TEXCOORD0 |
                                          PBR_Renderer::PSO_FLAG_USE_LIGHTS | PBR_Renderer::PSO_FLAG_USE_IBL |
                                          PBR_Renderer::PSO_FLAG_ENABLE_SPECULAR | PBR_Renderer::PSO_FLAG_USE_COLOR_MAP |
                                          PBR_Renderer::PSO_FLAG_USE_NORMAL_MAP | PBR_Renderer::PSO_FLAG_USE_SPECULAR_MAP) &
                                         st->enabledFlags;
    const auto get = PBR_Renderer::PsoCacheAccessor::GET_FLAG_CREATE_IF_NULL | PBR_Renderer::PsoCacheAccessor::GET_FLAG_ASYNC_COMPILE;
    for (auto flags : { base, base | PBR_Renderer::PSO_FLAG_USE_JOINTS })
        for (auto alpha : { PBR_Renderer::ALPHA_MODE_OPAQUE, PBR_Renderer::ALPHA_MODE_MASK })
            st->accessors[0][1].Get({ PBR_Renderer::RenderPassType::Main, flags & st->enabledFlags, alpha, CULL_MODE_BACK }, get);
    return st;
}

void PbrBeginFrame(DiligentSceneRenderer::Impl& r) {
    if (r.pbr) r.pbr->frameDirty = true;
}

void PbrInvalidateTexture(DiligentSceneRenderer::Impl& r, std::uint32_t textureId) {
    if (!r.pbr) return;
    auto& srbs = r.pbr->srbs;
    for (auto it = srbs.begin(); it != srbs.end();) {
        const auto& k   = it->first;
        bool        hit = false;
        for (std::size_t i = 0; i < kNumTex; ++i) {
            const std::uint32_t id = kTexAttribs[i] == PBR_Renderer::TEXTURE_ATTRIB_ID_SPECULAR ? k[i] >> 2 : k[i];
            hit |= id == textureId;
        }
        it = hit ? srbs.erase(it) : std::next(it);
    }
}

void PbrDraw(DiligentSceneRenderer::Impl& r, const GpuMesh& mesh, const glm::mat4& world, std::span<const glm::mat4> bones,
             const DrawSurface& surface) {
    if (!r.pbr || !r.pbr->renderer) return;
    PbrState&     st  = *r.pbr;
    PBR_Renderer& pbr = *st.renderer;

    GpuTexture* diffuse = r.FindTexture(surface.diffuse);
    GpuTexture* normal  = r.FindTexture(surface.normalMap);
    GpuTexture* rmaos   = r.FindTexture(surface.rmaosMap);
    GpuTexture* glow    = r.FindTexture(surface.glowMap);
    GpuTexture* specular = r.FindTexture(surface.specularMap);
    ITextureView* rmaosView = rmaos ? r.RmaosView(*rmaos) : nullptr;
    ITextureView* specView  = specular ? r.SpecularView(*specular) : nullptr;

    MaterialInputs in;
    in.hasDiffuse  = diffuse != nullptr;
    in.hasNormal   = normal != nullptr;
    in.hasRmaos    = rmaosView != nullptr;
    in.hasSpecular = specView != nullptr;
    in.hasGlow     = glow != nullptr;
    in.meshColors  = mesh.colors != nullptr;
    in.skinned     = !bones.empty() && mesh.joints && mesh.weights;
    in.wireframeOk = r.owner.supportsWireframe();
    const MappedMaterial m = MapMaterial(surface, in);

    const PBR_Renderer::PSO_FLAGS  flags = FlagsFor(m) & st.enabledFlags;
    const PBR_Renderer::ALPHA_MODE alpha = AlphaFor(m);
    const PBR_Renderer::PSOKey     key{ PBR_Renderer::RenderPassType::Main, flags, alpha, m.cullBack ? CULL_MODE_BACK : CULL_MODE_NONE };
    const auto& accessor = st.accessors[m.wireframe ? 1 : 0][m.depthWrite ? 1 : 0];
    IPipelineState* pso = accessor ? accessor.Get(key, PBR_Renderer::PsoCacheAccessor::GET_FLAG_CREATE_IF_NULL |
                                                           PBR_Renderer::PsoCacheAccessor::GET_FLAG_ASYNC_COMPILE)
                                   : nullptr;
    if (!pso || pso->GetStatus() != PIPELINE_STATE_STATUS_READY) {
        ++r.stats.skippedCompiling;
        return;
    }

    if (st.frameDirty) {
        WriteFrameAttribs(r, st);
        st.frameDirty = false;
    }

    // ── Textures → SRB (cached by the bound texture ids) ──
    auto id = [](TextureHandle h, const GpuTexture* t) { return t ? static_cast<std::uint32_t>(h) : 0u; };
    PbrState::SrbKey srbKey{};
    ITextureView*    views[kNumTex] = {};
    views[0] = diffuse ? diffuse->arraySrv.RawPtr() : pbr.GetWhiteTexSRV();
    srbKey[0] = id(surface.diffuse, diffuse);
    views[1] = normal ? normal->arraySrv.RawPtr() : pbr.GetDefaultNormalMapSRV();
    srbKey[1] = id(surface.normalMap, normal);
    views[2] = views[3] = in.hasRmaos ? rmaosView : pbr.GetWhiteTexSRV();
    srbKey[2] = srbKey[3] = in.hasRmaos ? id(surface.rmaosMap, rmaos) : 0u;
    views[4] = glow ? glow->arraySrv.RawPtr() : pbr.GetBlackTexSRV();
    srbKey[4] = id(surface.glowMap, glow);
    switch (m.specSource) {
        case MappedMaterial::SpecSource::Rmaos:
            views[5]  = rmaosView;
            srbKey[5] = srbKey[2] * 4 + 1;
            break;
        case MappedMaterial::SpecSource::NormalAlpha:
            views[5]  = normal->arraySrv;
            srbKey[5] = srbKey[1] * 4;
            break;
        case MappedMaterial::SpecSource::External:
            views[5]  = specView;
            srbKey[5] = id(surface.specularMap, specular) * 4 + 2;
            break;
        default: views[5] = pbr.GetWhiteTexSRV(); break;
    }

    RefCntAutoPtr<IShaderResourceBinding>& srb = st.srbs[srbKey];
    if (!srb) {
        pbr.CreateResourceBinding(&srb);
        if (!srb) return;
        PBR_Renderer::InitCommonSRBVarsAttribs common;
        common.pFrameAttribs = st.frameCB;
        pbr.InitCommonSRBVars(srb, common);
        if (st.irradiance && st.prefiltered)
            pbr.SetIBLResourceViews(srb, st.irradiance->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE),
                                    st.prefiltered->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE));
        for (std::size_t i = 0; i < kNumTex; ++i) pbr.SetMaterialTexture(srb, views[i], kTexAttribs[i]);
    }

    // ── Per-draw constants ──
    Uint32 jointCount = 0;
    if (flags & PBR_Renderer::PSO_FLAG_USE_JOINTS) {
        jointCount = static_cast<Uint32>(std::min<std::size_t>(bones.size(), kMaxJoints));
        if (bones.size() > kMaxJoints && !st.warnedJoints) {
            st.warnedJoints = true;
            std::fprintf(stderr, "[render] skin palette of %zu bones clamped to %u\n", bones.size(), kMaxJoints);
        }
        MapHelper<Uint8> joints(r.context, pbr.GetJointsBuffer(), MAP_WRITE, MAP_FLAG_DISCARD);
        std::memcpy(joints, bones.data(), jointCount * sizeof(glm::mat4));   // row-major packing: raw glm bytes
    }
    {
        const float4x4 node     = ToFloat4x4(world);
        const float4   fallback = { 1.f, 1.f, 1.f, 1.f };
        GLTF_PBR_Renderer::PBRPrimitiveShaderAttribsData attribs;
        attribs.PSOFlags      = flags;
        attribs.NodeMatrix    = &node;
        attribs.JointCount    = jointCount;
        attribs.FallbackColor = &fallback;
        MapHelper<Uint8> prim(r.context, pbr.GetPBRPrimitiveAttribsCB(), MAP_WRITE, MAP_FLAG_DISCARD);
        GLTF_PBR_Renderer::WritePBRPrimitiveShaderAttribs(prim, attribs, /*TransposeMatrices*/ false, /*UseSkinPreTransform*/ false,
                                                          PBR_Renderer::VERTEX_POS_PACK_MODE_NONE, 0);
    }
    {
        MapHelper<Uint8> mat(r.context, pbr.GetPBRMaterialAttribsCB(), MAP_WRITE, MAP_FLAG_DISCARD);
        WriteMaterialAttribs(mat, flags, m, alpha);
    }

    // ── Draw ──
    IBuffer* vbs[kNumVertexSlots] = { mesh.positions, mesh.normals, mesh.uvs, nullptr,
                                      (flags & PBR_Renderer::PSO_FLAG_USE_JOINTS) ? mesh.joints.RawPtr() : nullptr,
                                      (flags & PBR_Renderer::PSO_FLAG_USE_JOINTS) ? mesh.weights.RawPtr() : nullptr,
                                      (flags & PBR_Renderer::PSO_FLAG_USE_VERTEX_COLORS) ? mesh.colors.RawPtr() : nullptr };
    const Uint64 offsets[kNumVertexSlots] = {};
    r.context->SetPipelineState(pso);
    r.context->CommitShaderResources(srb, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    r.context->SetVertexBuffers(0, kNumVertexSlots, vbs, offsets, RESOURCE_STATE_TRANSITION_MODE_TRANSITION,
                                SET_VERTEX_BUFFERS_FLAG_RESET);
    r.context->SetIndexBuffer(mesh.indices, 0, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    DrawIndexedAttribs draw{ mesh.indexCount, VT_UINT16, DRAW_FLAG_VERIFY_ALL };
    r.context->DrawIndexed(draw);
    ++r.stats.draws;
}

} // namespace render::diligent
