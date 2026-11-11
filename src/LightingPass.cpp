// The Lighting material model (DrawSurface::model == Lighting): the classic game
// look, in our own HLSL (LightingShaders.h). One pipeline per feature set and
// fixed-function state, made on first use and compiled in the background; a draw
// whose pipeline is still compiling is skipped (FrameStats::skippedCompiling), as
// PbrPass does. Textures are dynamic variables, set per draw.
#include "LightingShaders.h"
#include "SceneRendererImpl.h"

#include "CommonlyUsedStates.h"
#include "GraphicsAccessories.hpp"
#include "MapHelper.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

namespace render::diligent {

using namespace Diligent;

namespace {

constexpr Uint32 kMaxJoints = 256;
constexpr int    kMaxLights = 32;

// Feature bits (each one a shader macro).
enum Feature : std::uint32_t {
    kSkinned          = 1u << 0,
    kVertexColors     = 1u << 1,
    kNormalMap        = 1u << 2,
    kModelSpaceNormals = 1u << 3,
    kSpecNormalAlpha  = 1u << 4,
    kSpecMap          = 1u << 5,
    kGlowMap          = 1u << 6,
    kOwnEmit          = 1u << 7,
    kEnvMap           = 1u << 8,
    kEnvMask          = 1u << 9,
    kEnvMaskNormalAlpha = 1u << 10,
    kRimLight         = 1u << 11,
    kSoftLight        = 1u << 12,
    kLightingMap      = 1u << 13,
    kBackLight        = 1u << 14,
    kBackLightMap     = 1u << 15,
    kParallax         = 1u << 16,
    kAlphaTest        = 1u << 17,
    kVertexTangents   = 1u << 18,
};
constexpr const char* kFeatureMacros[] = {
    "SKINNED", "VERTEX_COLORS", "NORMAL_MAP", "MODEL_SPACE_NORMALS", "SPEC_NORMAL_ALPHA", "SPEC_MAP",
    "GLOW_MAP", "OWN_EMIT", "ENV_MAP", "ENV_MASK", "ENV_MASK_NORMAL_ALPHA", "RIM_LIGHT", "SOFT_LIGHT",
    "LIGHTING_MAP", "BACK_LIGHT", "BACK_LIGHT_MAP", "PARALLAX", "ALPHA_TEST", "VERTEX_TANGENTS",
};
constexpr int kNumFeatures = static_cast<int>(std::size(kFeatureMacros));

// The fixed-function half of a pipeline key.
struct FixedState {
    bool        blend      = false;
    BlendFactor src        = BlendFactor::SrcAlpha;
    BlendFactor dst        = BlendFactor::InvSrcAlpha;
    bool        depthWrite = true;
    bool        depthTest  = true;
    int         depthBias  = 0;
    bool        cullBack   = true;
    bool        wireframe  = false;
    std::uint32_t Bits() const {
        return (blend ? 1u : 0u) | (static_cast<std::uint32_t>(src) << 1) | (static_cast<std::uint32_t>(dst) << 5) |
               (depthWrite ? 1u << 9 : 0u) | (depthTest ? 1u << 10 : 0u) |
               (static_cast<std::uint32_t>(std::clamp(depthBias, 0, 15)) << 11) | (cullBack ? 1u << 15 : 0u) |
               (wireframe ? 1u << 16 : 0u);
    }
};

BLEND_FACTOR ToDiligent(BlendFactor f) {
    switch (f) {
        case BlendFactor::One: return BLEND_FACTOR_ONE;
        case BlendFactor::Zero: return BLEND_FACTOR_ZERO;
        case BlendFactor::SrcColor: return BLEND_FACTOR_SRC_COLOR;
        case BlendFactor::InvSrcColor: return BLEND_FACTOR_INV_SRC_COLOR;
        case BlendFactor::DestColor: return BLEND_FACTOR_DEST_COLOR;
        case BlendFactor::InvDestColor: return BLEND_FACTOR_INV_DEST_COLOR;
        case BlendFactor::SrcAlpha: return BLEND_FACTOR_SRC_ALPHA;
        case BlendFactor::InvSrcAlpha: return BLEND_FACTOR_INV_SRC_ALPHA;
        case BlendFactor::DestAlpha: return BLEND_FACTOR_DEST_ALPHA;
        case BlendFactor::InvDestAlpha: return BLEND_FACTOR_INV_DEST_ALPHA;
        case BlendFactor::SrcAlphaSaturate: return BLEND_FACTOR_SRC_ALPHA_SAT;
    }
    return BLEND_FACTOR_ONE;
}

struct FrameConstants {
    glm::mat4 viewProj;
    glm::vec4 cameraPos, sunDir, sunColor;
    glm::vec4 ambient[6];
    glm::vec4 fog, fogNearColor, fogFarColor;
    glm::vec4 lightInfo;
    glm::vec4 lightPos[kMaxLights];
    glm::vec4 lightColor[kMaxLights];
};

struct DrawConstants {
    glm::mat4 world;
    glm::vec4 tint, specColor, emissive, effects, uv, alpha, srgb;
};

// The textures a pipeline may bind, in order (name, its feature gate).
struct TextureSlot {
    const char*   name;
    std::uint32_t gate;   // 0 = always
};
constexpr TextureSlot kSlots[] = {
    { "g_Diffuse", 0 },          { "g_Normal", kNormalMap | kModelSpaceNormals },
    { "g_Specular", kSpecMap },  { "g_Glow", kGlowMap },
    { "g_Env", kEnvMap },        { "g_EnvMask", kEnvMask },
    { "g_LightingMap", kLightingMap }, { "g_BackLightMap", kBackLightMap },
    { "g_Height", kParallax },
};
constexpr std::size_t kNumSlots = std::size(kSlots);

} // namespace

struct LightingState {
    struct Variant {
        RefCntAutoPtr<IPipelineState>         pso;
        RefCntAutoPtr<IShaderResourceBinding> srb;
        IShaderResourceVariable*              textures[kNumSlots] = {};
        bool                                  failed = false;
    };
    std::unordered_map<std::uint64_t, Variant> variants;
    RefCntAutoPtr<IBuffer>  frameCB, drawCB, bonesCB;
    RefCntAutoPtr<ITexture> white, flatNormal;
    bool frameDirty = true;
    bool warnedJoints = false;
};

void DestroyLightingState(LightingState* p) { delete p; }

namespace {

RefCntAutoPtr<IBuffer> MakeCB(IRenderDevice* device, const char* name, Uint32 size) {
    BufferDesc d;
    d.Name           = name;
    d.Size           = size;
    d.Usage          = USAGE_DYNAMIC;
    d.BindFlags      = BIND_UNIFORM_BUFFER;
    d.CPUAccessFlags = CPU_ACCESS_WRITE;
    RefCntAutoPtr<IBuffer> b;
    device->CreateBuffer(d, nullptr, &b);
    return b;
}

RefCntAutoPtr<ITexture> MakePixel(IRenderDevice* device, const char* name, std::uint32_t rgba) {
    TextureDesc d;
    d.Name      = name;
    d.Type      = RESOURCE_DIM_TEX_2D;
    d.Width     = 1;
    d.Height    = 1;
    d.Format    = TEX_FORMAT_RGBA8_UNORM;
    d.Usage     = USAGE_IMMUTABLE;
    d.BindFlags = BIND_SHADER_RESOURCE;
    TextureSubResData sub{ &rgba, 4 };
    TextureData       init{ &sub, 1 };
    RefCntAutoPtr<ITexture> t;
    device->CreateTexture(d, &init, &t);
    return t;
}

RefCntAutoPtr<IShader> Compile(DiligentSceneRenderer::Impl& r, const char* name, SHADER_TYPE type, const std::string& source,
                               const std::vector<ShaderMacro>& macros) {
    ShaderCreateInfo ci;
    ci.SourceLanguage = SHADER_SOURCE_LANGUAGE_HLSL;
    ci.Desc           = { name, type, true };
    ci.EntryPoint     = "main";
    ci.Source         = source.c_str();
    ci.Macros         = { macros.data(), static_cast<Uint32>(macros.size()) };
    ci.CompileFlags   = SHADER_COMPILE_FLAG_ASYNCHRONOUS;
    RefCntAutoPtr<IShader> shader;
    r.device->CreateShader(ci, &shader);
    if (!shader) std::fprintf(stderr, "[render] shader '%s' failed to compile\n", name);
    return shader;
}

LightingState::Variant& VariantFor(DiligentSceneRenderer::Impl& r, LightingState& st, std::uint32_t features,
                                   const FixedState& fixed) {
    const std::uint64_t key = (static_cast<std::uint64_t>(fixed.Bits()) << 32) | features;
    auto [it, inserted]     = st.variants.try_emplace(key);
    LightingState::Variant& var = it->second;
    if (!inserted) return var;

    std::vector<ShaderMacro> macros;
    for (int b = 0; b < kNumFeatures; ++b) macros.push_back({ kFeatureMacros[b], (features & (1u << b)) ? "1" : "0" });
    macros.push_back({ "NORMAL_MAP_GREEN_UP", "0" });
    const std::string common = shaders::kLightingCommon;
    auto vs = Compile(r, "render lighting VS", SHADER_TYPE_VERTEX, common + shaders::kLightingVS, macros);
    auto ps = Compile(r, "render lighting PS", SHADER_TYPE_PIXEL, common + shaders::kLightingPS, macros);
    if (!vs || !ps) {
        var.failed = true;
        return var;
    }

    std::vector<LayoutElement> layout = {
        { 0, 0, 3, VT_FLOAT32, False },
        { 1, 1, 3, VT_FLOAT32, False },
        { 2, 2, 2, VT_FLOAT32, False },
    };
    if (features & kSkinned) {
        layout.push_back({ 3, 4, 4, VT_FLOAT32, False });
        layout.push_back({ 4, 5, 4, VT_FLOAT32, False });
    }
    if (features & kVertexColors) layout.push_back({ 5, 6, 4, VT_UINT8, True });
    if (features & kVertexTangents) layout.push_back({ 6, 3, 4, VT_FLOAT32, False });

    GraphicsPipelineStateCreateInfo ci;
    ci.PSODesc.Name = "render lighting PSO";
    ci.Flags        = PSO_CREATE_FLAG_ASYNCHRONOUS;
    GraphicsPipelineDesc& g = ci.GraphicsPipeline;
    g.NumRenderTargets                     = 1;
    g.RTVFormats[0]                        = TEX_FORMAT_RGBA16_FLOAT;
    g.DSVFormat                            = kSceneDepthFormat;
    g.PrimitiveTopology                    = PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    g.InputLayout                          = { layout.data(), static_cast<Uint32>(layout.size()) };
    g.RasterizerDesc.FrontCounterClockwise = True;
    g.RasterizerDesc.CullMode              = fixed.cullBack ? CULL_MODE_BACK : CULL_MODE_NONE;
    g.RasterizerDesc.FillMode              = fixed.wireframe ? FILL_MODE_WIREFRAME : FILL_MODE_SOLID;
    const int bias                          = std::clamp(fixed.depthBias, 0, 15);
    g.RasterizerDesc.DepthBias             = -64 * bias;   // as PbrPass: towards the camera
    g.RasterizerDesc.SlopeScaledDepthBias  = -2.f * static_cast<float>(bias);
    g.DepthStencilDesc.DepthEnable         = True;
    g.DepthStencilDesc.DepthWriteEnable    = fixed.depthWrite ? True : False;
    g.DepthStencilDesc.DepthFunc           = fixed.depthTest ? COMPARISON_FUNC_LESS_EQUAL : COMPARISON_FUNC_ALWAYS;
    if (fixed.blend) {
        RenderTargetBlendDesc& rt = g.BlendDesc.RenderTargets[0];
        rt.BlendEnable = True;
        rt.SrcBlend    = ToDiligent(fixed.src);
        rt.DestBlend   = ToDiligent(fixed.dst);
        rt.BlendOp     = BLEND_OPERATION_ADD;
        // The target's alpha is coverage (the post pass composites the background by it). An "over" blend adds
        // coverage; any other (additive, multiplicative) leaves it as it was.
        const bool over   = fixed.dst == BlendFactor::InvSrcAlpha;
        rt.SrcBlendAlpha  = over ? BLEND_FACTOR_ONE : BLEND_FACTOR_ZERO;
        rt.DestBlendAlpha = over ? BLEND_FACTOR_INV_SRC_ALPHA : BLEND_FACTOR_ONE;
        rt.BlendOpAlpha   = BLEND_OPERATION_ADD;
    }
    ci.pVS = vs;
    ci.pPS = ps;

    std::vector<ShaderResourceVariableDesc> vars;
    std::vector<ImmutableSamplerDesc>       samplers;
    SamplerDesc aniso = Sam_Aniso8xWrap;
    for (const TextureSlot& slot : kSlots) {
        if (slot.gate && !(features & slot.gate)) continue;
        vars.push_back({ SHADER_TYPE_PIXEL, slot.name, SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC });
        samplers.push_back({ SHADER_TYPE_PIXEL, slot.name, std::strcmp(slot.name, "g_Env") == 0 ? Sam_LinearClamp : aniso });
    }
    ci.PSODesc.ResourceLayout.DefaultVariableType  = SHADER_RESOURCE_VARIABLE_TYPE_STATIC;
    ci.PSODesc.ResourceLayout.Variables            = vars.data();
    ci.PSODesc.ResourceLayout.NumVariables         = static_cast<Uint32>(vars.size());
    ci.PSODesc.ResourceLayout.ImmutableSamplers    = samplers.data();
    ci.PSODesc.ResourceLayout.NumImmutableSamplers = static_cast<Uint32>(samplers.size());

    r.device->CreateGraphicsPipelineState(ci, &var.pso);
    if (!var.pso) {
        var.failed = true;
        return var;
    }
    return var;
}

// Binds the static constant buffers and finds the texture variables once the pipeline is ready.
bool PrepareVariant(LightingState& st, LightingState::Variant& var, bool skinned) {
    if (var.srb) return true;
    for (SHADER_TYPE stage : { SHADER_TYPE_VERTEX, SHADER_TYPE_PIXEL }) {
        if (auto* v = var.pso->GetStaticVariableByName(stage, "LightingFrame")) v->Set(st.frameCB);
        if (auto* v = var.pso->GetStaticVariableByName(stage, "LightingDraw")) v->Set(st.drawCB);
    }
    if (skinned)
        if (auto* v = var.pso->GetStaticVariableByName(SHADER_TYPE_VERTEX, "LightingBones")) v->Set(st.bonesCB);
    var.pso->CreateShaderResourceBinding(&var.srb, true);
    if (!var.srb) return false;
    for (std::size_t i = 0; i < kNumSlots; ++i) var.textures[i] = var.srb->GetVariableByName(SHADER_TYPE_PIXEL, kSlots[i].name);
    return true;
}

void WriteFrame(DiligentSceneRenderer::Impl& r, LightingState& st) {
    MapHelper<FrameConstants> f(r.context, st.frameCB, MAP_WRITE, MAP_FLAG_DISCARD);
    const SceneEnvironment& e = r.environment;
    std::memset(static_cast<FrameConstants*>(f), 0, sizeof(FrameConstants));
    f->viewProj  = r.viewProj;
    f->cameraPos = glm::vec4(r.cameraPos, 1.f);
    f->sunDir    = glm::vec4(glm::normalize(e.sunDirection), 0.f);
    f->sunColor  = glm::vec4(e.sunColor, 0.f);
    for (int i = 0; i < 6; ++i) f->ambient[i] = glm::vec4(e.ambient[i], 0.f);
    f->fog          = { e.fogNear, e.fogFar, e.fogPower, e.fogMax };
    f->fogNearColor = glm::vec4(e.fogNearColor, e.fog ? 1.f : 0.f);
    f->fogFarColor  = glm::vec4(e.fogFarColor, 0.f);
    const int n     = static_cast<int>(std::min<std::size_t>(e.lights.size(), kMaxLights));
    f->lightInfo    = { static_cast<float>(n), 0.f, 0.f, 0.f };
    for (int i = 0; i < n; ++i) {
        f->lightPos[i]   = glm::vec4(e.lights[i].position, e.lights[i].radius);
        f->lightColor[i] = glm::vec4(e.lights[i].color, 0.f);
    }
}

} // namespace

LightingPtr CreateLightingState(DiligentSceneRenderer::Impl& r) {
    LightingPtr st(new LightingState{}, &DestroyLightingState);
    st->frameCB    = MakeCB(r.device, "render lighting frame CB", sizeof(FrameConstants));
    st->drawCB     = MakeCB(r.device, "render lighting draw CB", sizeof(DrawConstants));
    st->bonesCB    = MakeCB(r.device, "render lighting bones CB", sizeof(glm::mat4) * kMaxJoints);
    st->white      = MakePixel(r.device, "render lighting white", 0xFFFFFFFFu);
    st->flatNormal = MakePixel(r.device, "render lighting flat normal", 0xFFFF8080u);   // (128, 128, 255, 255)
    return st;
}

void LightingBeginFrame(DiligentSceneRenderer::Impl& r) {
    if (r.lighting) r.lighting->frameDirty = true;
}

void LightingDraw(DiligentSceneRenderer::Impl& r, const GpuMesh& mesh, const glm::mat4& world, std::span<const glm::mat4> bones,
                  const DrawSurface& s) {
    LightingState* st = r.lighting.get();
    if (!st || !st->frameCB || !st->drawCB) return;
    const LightingParams& L = s.lighting;

    GpuTexture* diffuse   = r.FindTexture(s.diffuse);
    GpuTexture* normal    = r.FindTexture(s.normalMap);
    GpuTexture* specular  = r.FindTexture(s.specularMap);
    GpuTexture* glow      = r.FindTexture(s.glowMap);
    GpuTexture* env       = r.FindTexture(L.envMap);
    GpuTexture* envMask   = r.FindTexture(L.envMask);
    GpuTexture* lightMap  = r.FindTexture(L.lightingMap);
    GpuTexture* backMap   = r.FindTexture(L.backLightMap);
    GpuTexture* height    = r.FindTexture(L.heightMap);
    if (env && !env->cube) env = nullptr;

    using Blend = DrawSurface::BlendMode;
    const bool skinned = !bones.empty() && mesh.joints && mesh.weights;
    const bool tested  = s.blendMode == Blend::AlphaTest || s.blendMode == Blend::AlphaTestAndBlend;
    const bool blended = s.blendMode == Blend::AlphaBlend || s.blendMode == Blend::Additive ||
                         s.blendMode == Blend::AlphaTestAndBlend || s.xray;

    std::uint32_t f = 0;
    if (skinned) f |= kSkinned;
    if (s.useVertexColor && mesh.colors) f |= kVertexColors;
    if (normal) f |= L.modelSpaceNormals ? kModelSpaceNormals : kNormalMap;
    if (specular) f |= kSpecMap;
    else if (normal && s.specFromNormalAlpha) f |= kSpecNormalAlpha;
    if (glow) f |= kGlowMap;
    else if (L.ownEmit) f |= kOwnEmit;
    if (env) {
        f |= kEnvMap;
        if (envMask) f |= kEnvMask;
        else if (normal && L.envMaskFromNormalAlpha) f |= kEnvMaskNormalAlpha;
    }
    if (L.rimLight) f |= kRimLight;
    if (L.softLight) f |= kSoftLight;
    if ((L.rimLight || L.softLight) && lightMap) f |= kLightingMap;
    if (L.backLight) {
        f |= kBackLight;
        if (backMap) f |= kBackLightMap;
    }
    if (height) f |= kParallax;
    if (mesh.tangents && (f & (kNormalMap | kParallax))) f |= kVertexTangents;
    if (tested) f |= kAlphaTest;

    FixedState fixed;
    fixed.blend      = blended;
    fixed.src        = s.xray ? BlendFactor::SrcAlpha : s.srcBlend;
    fixed.dst        = s.xray ? BlendFactor::InvSrcAlpha : s.dstBlend;
    fixed.depthWrite = !blended || (s.blendWritesDepth && !s.xray);
    fixed.depthTest  = s.depthTest;
    fixed.depthBias  = s.depthBias;
    fixed.cullBack   = !s.twoSided && !s.xray;
    fixed.wireframe  = s.wireframe && r.owner.supportsWireframe();
    if (!fixed.blend) fixed.src = fixed.dst = BlendFactor::One;   // irrelevant: one key

    LightingState::Variant& var = VariantFor(r, *st, f, fixed);
    if (var.failed || !var.pso) return;
    if (var.pso->GetStatus() != PIPELINE_STATE_STATUS_READY) {
        if (var.pso->GetStatus() == PIPELINE_STATE_STATUS_FAILED) var.failed = true;
        else ++r.stats.skippedCompiling;
        return;
    }
    if (!PrepareVariant(*st, var, skinned)) return;

    if (st->frameDirty) {
        WriteFrame(r, *st);
        st->frameDirty = false;
    }
    {
        MapHelper<DrawConstants> d(r.context, st->drawCB, MAP_WRITE, MAP_FLAG_DISCARD);
        d->world     = world;
        d->tint      = s.tint;
        if (s.xray) d->tint.a = 0.35f;
        d->specColor = glm::vec4(L.specularColor, s.specularStrength);
        d->emissive  = glm::vec4(s.emissiveColor * s.emissiveMultiple, L.glossiness);
        d->effects   = { L.effect1, L.effect2, L.envScale, L.parallaxScale };
        d->uv        = { L.uvOffset.x, L.uvOffset.y, L.uvScale.x, L.uvScale.y };
        d->alpha     = { s.alphaThreshold, static_cast<float>(s.alphaTestFunc),
                         blended && fixed.dst == BlendFactor::One ? 1.f : 0.f, blended ? 1.f : 0.f };
        auto srgb    = [](const GpuTexture* t) { return t && t->srgb ? 1.f : 0.f; };
        d->srgb      = { srgb(diffuse), srgb(glow), srgb(env), srgb(backMap) };
    }
    if (skinned) {
        const Uint32 n = static_cast<Uint32>(std::min<std::size_t>(bones.size(), kMaxJoints));
        if (bones.size() > kMaxJoints && !st->warnedJoints) {
            st->warnedJoints = true;
            std::fprintf(stderr, "[render] skin palette of %zu bones clamped to %u\n", bones.size(), kMaxJoints);
        }
        MapHelper<glm::mat4> b(r.context, st->bonesCB, MAP_WRITE, MAP_FLAG_DISCARD);
        std::memcpy(static_cast<glm::mat4*>(b), bones.data(), n * sizeof(glm::mat4));
    }

    // Textures: what the variant reads; a missing optional map never reaches here (its feature is off).
    ITextureView* white = st->white->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE);
    ITextureView* flat  = st->flatNormal->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE);
    const GpuTexture* bound[kNumSlots] = { diffuse, normal, specular, glow, env, envMask, lightMap, backMap, height };
    for (std::size_t i = 0; i < kNumSlots; ++i) {
        if (!var.textures[i]) continue;
        ITextureView* view = bound[i] ? bound[i]->srv.RawPtr() : (i == 1 ? flat : white);
        var.textures[i]->Set(view);
    }

    IBuffer* vbs[7] = { mesh.positions, mesh.normals, mesh.uvs, (f & kVertexTangents) ? mesh.tangents.RawPtr() : nullptr,
                        (f & kSkinned) ? mesh.joints.RawPtr() : nullptr,
                        (f & kSkinned) ? mesh.weights.RawPtr() : nullptr, (f & kVertexColors) ? mesh.colors.RawPtr() : nullptr };
    const Uint64 offsets[7] = {};
    r.context->SetPipelineState(var.pso);
    r.context->CommitShaderResources(var.srb, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    r.context->SetVertexBuffers(0, 7, vbs, offsets, RESOURCE_STATE_TRANSITION_MODE_TRANSITION, SET_VERTEX_BUFFERS_FLAG_RESET);
    r.context->SetIndexBuffer(mesh.indices, 0, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    DrawIndexedAttribs draw{ mesh.indexCount, mesh.index32 ? VT_UINT32 : VT_UINT16, DRAW_FLAG_VERIFY_ALL };
    r.context->DrawIndexed(draw);
    ++r.stats.draws;
}

} // namespace render::diligent
