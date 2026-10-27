// The face-tint composite: ping-pong between two RGBA8 sRGB targets the size of
// the base diffuse — pass 0 copies the base, then one fullscreen pass per layer
// (GL's maths plus Skyrim's soft light, see Shaders.h kTintPS; soft-light layers
// read the original base as well). The final target gets its mips generated
// and is registered as a new texture. Callable inside a frame: the scene targets
// are re-bound afterwards.
#include "SceneRendererImpl.h"
#include "Shaders.h"

#include "CommonlyUsedStates.h"
#include "MapHelper.hpp"

#include <utility>

namespace render::diligent {

using namespace Diligent;

namespace {

constexpr TEXTURE_FORMAT kTintFormat = TEX_FORMAT_RGBA8_UNORM_SRGB;

struct TintCB {
    float tint[4];
    int   params[4];    // blend, coverage from alpha, has mask, copy
    int   params2[4];   // has tint map
};

} // namespace

struct FaceTintState {
    RefCntAutoPtr<IPipelineState>         pso;
    RefCntAutoPtr<IShaderResourceBinding> srb;
    RefCntAutoPtr<IBuffer>                cb;
};

void DestroyFaceTintState(FaceTintState* t) { delete t; }

FaceTintPtr CreateFaceTintState(DiligentSceneRenderer::Impl& r) {
    FaceTintPtr st(new FaceTintState{}, &DestroyFaceTintState);

    BufferDesc cbDesc;
    cbDesc.Name           = "render face tint CB";
    cbDesc.Size           = sizeof(TintCB);
    cbDesc.Usage          = USAGE_DYNAMIC;
    cbDesc.BindFlags      = BIND_UNIFORM_BUFFER;
    cbDesc.CPUAccessFlags = CPU_ACCESS_WRITE;
    r.device->CreateBuffer(cbDesc, nullptr, &st->cb);

    auto vs = r.CompileShader("render fullscreen VS", SHADER_TYPE_VERTEX, shaders::kFullscreenVS);
    auto ps = r.CompileShader("render face tint PS", SHADER_TYPE_PIXEL, shaders::kTintPS);
    if (!vs || !ps || !st->cb) return st;

    GraphicsPipelineStateCreateInfo ci;
    ci.PSODesc.Name = "render face tint PSO";
    GraphicsPipelineDesc& gp = ci.GraphicsPipeline;
    gp.NumRenderTargets             = 1;
    gp.RTVFormats[0]                = kTintFormat;
    gp.DSVFormat                    = TEX_FORMAT_UNKNOWN;
    gp.PrimitiveTopology            = PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    gp.RasterizerDesc.CullMode      = CULL_MODE_NONE;
    gp.DepthStencilDesc.DepthEnable = False;   // the shader blends; no fixed-function blend
    ci.pVS = vs;
    ci.pPS = ps;

    const ShaderResourceVariableDesc vars[] = {
        { SHADER_TYPE_PIXEL, "g_Base", SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC },
        { SHADER_TYPE_PIXEL, "g_Current", SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC },
        { SHADER_TYPE_PIXEL, "g_Mask", SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC },
        { SHADER_TYPE_PIXEL, "g_TintMap", SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC },
    };
    const ImmutableSamplerDesc samplers[] = { { SHADER_TYPE_PIXEL, "g_Mask", Sam_LinearClamp },
                                              { SHADER_TYPE_PIXEL, "g_TintMap", Sam_LinearClamp } };
    ci.PSODesc.ResourceLayout.DefaultVariableType  = SHADER_RESOURCE_VARIABLE_TYPE_STATIC;
    ci.PSODesc.ResourceLayout.Variables            = vars;
    ci.PSODesc.ResourceLayout.NumVariables         = _countof(vars);
    ci.PSODesc.ResourceLayout.ImmutableSamplers    = samplers;
    ci.PSODesc.ResourceLayout.NumImmutableSamplers = _countof(samplers);

    r.device->CreateGraphicsPipelineState(ci, &st->pso);
    if (!st->pso) return st;
    if (auto* var = st->pso->GetStaticVariableByName(SHADER_TYPE_PIXEL, "TintCB")) var->Set(st->cb);
    st->pso->CreateShaderResourceBinding(&st->srb, true);
    return st;
}

TextureHandle FaceTintComposite(DiligentSceneRenderer::Impl& r, TextureHandle base, std::span<const FaceTintLayer> layers) {
    FaceTintState* st = r.tint.get();
    GpuTexture*    b  = r.FindTexture(base);
    if (!st || !st->srb || !b || layers.empty() || b->width <= 0 || b->height <= 0) return TextureHandle::Invalid;

    TextureDesc desc;
    desc.Name      = "render face tint";
    desc.Type      = RESOURCE_DIM_TEX_2D;
    desc.Width     = static_cast<Uint32>(b->width);
    desc.Height    = static_cast<Uint32>(b->height);
    desc.MipLevels = 0;   // full chain, generated at the end
    desc.Format    = kTintFormat;
    desc.BindFlags = BIND_RENDER_TARGET | BIND_SHADER_RESOURCE;
    desc.MiscFlags = MISC_TEXTURE_FLAG_GENERATE_MIPS;
    RefCntAutoPtr<ITexture> src, dst;
    r.device->CreateTexture(desc, nullptr, &src);
    r.device->CreateTexture(desc, nullptr, &dst);
    if (!src || !dst) return TextureHandle::Invalid;

    IShaderResourceVariable* baseVar = st->srb->GetVariableByName(SHADER_TYPE_PIXEL, "g_Base");
    IShaderResourceVariable* curVar  = st->srb->GetVariableByName(SHADER_TYPE_PIXEL, "g_Current");
    IShaderResourceVariable* maskVar = st->srb->GetVariableByName(SHADER_TYPE_PIXEL, "g_Mask");
    IShaderResourceVariable* tintVar = st->srb->GetVariableByName(SHADER_TYPE_PIXEL, "g_TintMap");
    if (!baseVar || !curVar || !maskVar || !tintVar) return TextureHandle::Invalid;

    ITextureView* baseSrv = b->texture->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE);
    auto pass = [&](ITexture* target, ITextureView* current, ITextureView* mask, ITextureView* tintMap, const TintCB& c) {
        {
            MapHelper<TintCB> cb(r.context, st->cb, MAP_WRITE, MAP_FLAG_DISCARD);
            *cb = c;
        }
        ITextureView* rtv = target->GetDefaultView(TEXTURE_VIEW_RENDER_TARGET);
        r.context->SetRenderTargets(1, &rtv, nullptr, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
        r.context->SetViewports(1, nullptr, 0, 0);
        baseVar->Set(baseSrv);
        curVar->Set(current);
        maskVar->Set(mask);
        tintVar->Set(tintMap);
        r.context->SetPipelineState(st->pso);
        r.context->CommitShaderResources(st->srb, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
        DrawAttribs draw{ 3, DRAW_FLAG_VERIFY_ALL };
        r.context->Draw(draw);
    };

    // Pass 0: copy the base into the accumulator.
    pass(src, baseSrv, baseSrv, baseSrv, TintCB{ { 0, 0, 0, 0 }, { 0, 0, 0, 1 }, { 0, 0, 0, 0 } });

    for (const FaceTintLayer& layer : layers) {
        // A layer may be maskless (full coverage, e.g. body skin tone); an
        // unresolvable mask or tint map skips the layer.
        ITextureView* mask = nullptr;
        if (layer.mask != TextureHandle::Invalid) {
            GpuTexture* m = r.FindTexture(layer.mask);
            if (!m) continue;
            mask = m->texture->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE);
        }
        ITextureView* tintMap = nullptr;
        const bool    usesMap = layer.blend == FaceTintLayer::Blend::SoftLight || layer.blend == FaceTintLayer::Blend::Detail;
        if (usesMap && layer.tintMap != TextureHandle::Invalid) {
            GpuTexture* t = r.FindTexture(layer.tintMap);
            if (!t) continue;
            tintMap = t->texture->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE);
        }
        if (layer.blend == FaceTintLayer::Blend::Detail && !tintMap) continue;   // nothing to scale by
        ITextureView* cur = src->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE);
        const TintCB  c{ { layer.color.r, layer.color.g, layer.color.b, layer.color.a },
                         { static_cast<int>(layer.blend), layer.coverageFromAlpha ? 1 : 0, mask ? 1 : 0, 0 },
                         { tintMap ? 1 : 0, 0, 0, 0 } };
        // The accumulator is a harmless bind for an absent mask / tint map.
        pass(dst, cur, mask ? mask : cur, tintMap ? tintMap : cur, c);
        std::swap(src, dst);   // result now in src
    }

    r.context->GenerateMips(src->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE));
    if (r.inFrame) r.BindSceneTargets();

    GpuTexture out;
    out.texture  = src;
    out.rendered = true;
    return r.RegisterTexture(std::move(out));
}

TextureHandle DiligentSceneRenderer::CompositeFaceTint(TextureHandle base, std::span<const FaceTintLayer> layers) {
    return FaceTintComposite(*m_impl, base, layers);
}

} // namespace render::diligent
