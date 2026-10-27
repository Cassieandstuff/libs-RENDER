// DiligentSceneRenderer: frame lifecycle, output binding, the post pass (exposure
// + ACES into the bound output) and the infinite grid. Overlays: Overlay.cpp.
#include "MathConv.h"
#include "SceneRendererImpl.h"
#include "Shaders.h"

#include "GraphicsAccessories.hpp"
#include "MapHelper.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace render::diligent {

using namespace Diligent;

namespace {

constexpr TEXTURE_FORMAT kHdrFormat   = TEX_FORMAT_RGBA16_FLOAT;
constexpr TEXTURE_FORMAT kDepthFormat = kSceneDepthFormat;
constexpr float          kClearColor[4] = { 0.f, 0.f, 0.f, 0.f };   // transparent: post composites over the background

// sRGB-encoded (display) value !92 linear.
float SrgbToLinear(float c) { return c <= 0.04045f ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f); }

void LogOnce(bool& flag, const char* what) {
    if (flag) return;
    flag = true;
    std::fprintf(stderr, "[render] %s\n", what);
}

RefCntAutoPtr<IBuffer> CreateDynamicCB(IRenderDevice* device, const char* name, Uint32 size) {
    BufferDesc desc;
    desc.Name           = name;
    desc.Size           = size;
    desc.Usage          = USAGE_DYNAMIC;
    desc.BindFlags      = BIND_UNIFORM_BUFFER;
    desc.CPUAccessFlags = CPU_ACCESS_WRITE;
    RefCntAutoPtr<IBuffer> buf;
    device->CreateBuffer(desc, nullptr, &buf);
    return buf;
}

} // namespace

// ── Impl ──────────────────────────────────────────────────────────────────────

DiligentSceneRenderer::Impl::Impl(RenderDevice& dev)
    : owner(dev), device(dev.impl().device), context(dev.impl().context) {
    postCB = CreateDynamicCB(device, "render post CB", sizeof(float) * 8);
    gridCB = CreateDynamicCB(device, "render grid CB", sizeof(float) * (16 * 2 + 4 * 6));
    pbr    = CreatePbrState(*this);
    tint   = CreateFaceTintState(*this);
    overlay = CreateOverlayState(*this);
}

DiligentSceneRenderer::Impl::~Impl() {
    // GPU work may still reference what's about to be released.
    owner.WaitIdle();
}

RefCntAutoPtr<IShader> DiligentSceneRenderer::Impl::CompileShader(const char* name, SHADER_TYPE type, const char* source,
                                                                  const ShaderMacro* macros, Uint32 numMacros) {
    ShaderCreateInfo ci;
    ci.SourceLanguage = SHADER_SOURCE_LANGUAGE_HLSL;
    ci.Desc           = { name, type, true };
    ci.EntryPoint     = "main";
    ci.Source         = source;
    ci.Macros         = { macros, numMacros };
    RefCntAutoPtr<IShader> shader;
    device->CreateShader(ci, &shader);
    if (!shader) std::fprintf(stderr, "[render] shader '%s' failed to compile\n", name);
    return shader;
}

ITextureView* DiligentSceneRenderer::Impl::OutputRTV() const {
    if (outSurface)   return outSurface->impl().swapChain->GetCurrentBackBufferRTV();
    if (outOffscreen) return outOffscreen->impl().color->GetDefaultView(TEXTURE_VIEW_RENDER_TARGET);
    return nullptr;
}

void DiligentSceneRenderer::Impl::EnsureSceneTargets(int w, int h) {
    if (hdrColor && w == width && h == height) return;
    hdrColor.Release();
    hdrDepth.Release();
    width  = w;
    height = h;

    TextureDesc desc;
    desc.Type      = RESOURCE_DIM_TEX_2D;
    desc.Width     = static_cast<Uint32>(w);
    desc.Height    = static_cast<Uint32>(h);
    desc.MipLevels = 1;

    desc.Name      = "render scene HDR";
    desc.Format    = kHdrFormat;
    desc.BindFlags = BIND_RENDER_TARGET | BIND_SHADER_RESOURCE;
    desc.ClearValue.Format = kHdrFormat;
    for (int i = 0; i < 4; ++i) desc.ClearValue.Color[i] = kClearColor[i];
    device->CreateTexture(desc, nullptr, &hdrColor);

    desc.Name      = "render scene depth";
    desc.Format    = kDepthFormat;
    desc.BindFlags = BIND_DEPTH_STENCIL;
    desc.ClearValue.Format = kDepthFormat;
    desc.ClearValue.DepthStencil.Depth   = 1.f;
    desc.ClearValue.DepthStencil.Stencil = 0;
    device->CreateTexture(desc, nullptr, &hdrDepth);
}

void DiligentSceneRenderer::Impl::BindSceneTargets() {
    ITextureView* rtv = hdrColor->GetDefaultView(TEXTURE_VIEW_RENDER_TARGET);
    ITextureView* dsv = hdrDepth->GetDefaultView(TEXTURE_VIEW_DEPTH_STENCIL);
    context->SetRenderTargets(1, &rtv, dsv, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    context->SetViewports(1, nullptr, 0, 0);   // full target
}

// ── Post pass ─────────────────────────────────────────────────────────────────

DiligentSceneRenderer::Impl::PostPipeline& DiligentSceneRenderer::Impl::PostFor(TEXTURE_FORMAT format) {
    auto [it, inserted] = postPipelines.try_emplace(static_cast<int>(format));
    PostPipeline& pp = it->second;
    if (!inserted) return pp;

    const bool srgb = GetTextureFormatAttribs(format).ComponentType == COMPONENT_TYPE_UNORM_SRGB;
    const ShaderMacro macros[] = { { "OUTPUT_GAMMA", srgb ? "0" : "1" } };
    auto vs = CompileShader("render fullscreen VS", SHADER_TYPE_VERTEX, shaders::kFullscreenVS);
    auto ps = CompileShader("render post PS", SHADER_TYPE_PIXEL, shaders::kPostPS, macros, _countof(macros));
    if (!vs || !ps) return pp;

    GraphicsPipelineStateCreateInfo ci;
    ci.PSODesc.Name = "render post PSO";
    GraphicsPipelineDesc& gp = ci.GraphicsPipeline;
    gp.NumRenderTargets  = 1;
    gp.RTVFormats[0]     = format;
    gp.DSVFormat         = TEX_FORMAT_UNKNOWN;
    gp.PrimitiveTopology = PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    gp.RasterizerDesc.CullMode      = CULL_MODE_NONE;
    gp.DepthStencilDesc.DepthEnable = False;
    ci.pVS = vs;
    ci.pPS = ps;

    const ShaderResourceVariableDesc vars[] = {
        { SHADER_TYPE_PIXEL, "g_HDR", SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC },   // re-pointed on resize
    };
    ci.PSODesc.ResourceLayout.DefaultVariableType = SHADER_RESOURCE_VARIABLE_TYPE_STATIC;
    ci.PSODesc.ResourceLayout.Variables           = vars;
    ci.PSODesc.ResourceLayout.NumVariables        = _countof(vars);

    device->CreateGraphicsPipelineState(ci, &pp.pso);
    if (!pp.pso) return pp;
    if (auto* var = pp.pso->GetStaticVariableByName(SHADER_TYPE_PIXEL, "PostCB")) var->Set(postCB);
    pp.pso->CreateShaderResourceBinding(&pp.srb, true);
    return pp;
}

void DiligentSceneRenderer::Impl::RunPost(ITextureView* outputRTV) {
    PostPipeline& pp = PostFor(outputRTV->GetDesc().Format);
    if (!pp.pso) return;
    {
        MapHelper<float> cb(context, postCB, MAP_WRITE, MAP_FLAG_DISCARD);
        cb[0] = exposure;
        cb[1] = cb[2] = cb[3] = 0.f;
        for (int i = 0; i < 3; ++i) cb[4 + i] = SrgbToLinear(style.background[i]);
        cb[7] = 1.f;
    }
    context->SetRenderTargets(1, &outputRTV, nullptr, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    context->SetViewports(1, nullptr, 0, 0);
    pp.srb->GetVariableByName(SHADER_TYPE_PIXEL, "g_HDR")->Set(hdrColor->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE));
    context->SetPipelineState(pp.pso);
    context->CommitShaderResources(pp.srb, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    DrawAttribs draw{ 3, DRAW_FLAG_VERIFY_ALL };
    context->Draw(draw);
}

// ── Infinite grid ─────────────────────────────────────────────────────────────

DiligentSceneRenderer::Impl::PostPipeline& DiligentSceneRenderer::Impl::GridFor(TEXTURE_FORMAT format) {
    auto [it, inserted] = gridPipelines.try_emplace(static_cast<int>(format));
    PostPipeline& gp = it->second;
    if (!inserted) return gp;

    auto vs = CompileShader("render fullscreen VS", SHADER_TYPE_VERTEX, shaders::kFullscreenVS);
    auto ps = CompileShader("render grid PS", SHADER_TYPE_PIXEL, shaders::kGridPS);
    if (!vs || !ps) return gp;

    GraphicsPipelineStateCreateInfo ci;
    ci.PSODesc.Name = "render grid PSO";
    GraphicsPipelineDesc& g = ci.GraphicsPipeline;
    g.NumRenderTargets                  = 1;
    g.RTVFormats[0]                     = format;
    g.DSVFormat                         = kDepthFormat;
    g.PrimitiveTopology                 = PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    g.RasterizerDesc.CullMode           = CULL_MODE_NONE;
    g.DepthStencilDesc.DepthEnable      = True;
    g.DepthStencilDesc.DepthWriteEnable = False;   // geometry below the plane stays visible
    g.DepthStencilDesc.DepthFunc        = COMPARISON_FUNC_LESS_EQUAL;
    RenderTargetBlendDesc& rt = g.BlendDesc.RenderTargets[0];   // premultiplied coverage
    rt.BlendEnable    = True;
    rt.SrcBlend       = BLEND_FACTOR_ONE;
    rt.DestBlend      = BLEND_FACTOR_INV_SRC_ALPHA;
    rt.SrcBlendAlpha  = BLEND_FACTOR_ONE;
    rt.DestBlendAlpha = BLEND_FACTOR_INV_SRC_ALPHA;
    ci.pVS = vs;
    ci.pPS = ps;
    ci.PSODesc.ResourceLayout.DefaultVariableType = SHADER_RESOURCE_VARIABLE_TYPE_STATIC;

    device->CreateGraphicsPipelineState(ci, &gp.pso);
    if (!gp.pso) return gp;
    if (auto* var = gp.pso->GetStaticVariableByName(SHADER_TYPE_PIXEL, "GridCB")) var->Set(gridCB);
    gp.pso->CreateShaderResourceBinding(&gp.srb, true);
    return gp;
}

void DiligentSceneRenderer::Impl::RunGrid(ITextureView* outputRTV) {
    // The grid tests against the scene depth, so the output must match the scene size.
    const TextureDesc& od = outputRTV->GetTexture()->GetDesc();
    if (static_cast<int>(od.Width) != width || static_cast<int>(od.Height) != height) {
        LogOnce(warnedGridSize, "output target size differs from the frame size; grid skipped");
        return;
    }
    PostPipeline& gp = GridFor(outputRTV->GetDesc().Format);
    if (!gp.pso) return;

    // An sRGB output blends in linear; the style's colours are display values.
    const bool srgbOut = GetTextureFormatAttribs(outputRTV->GetDesc().Format).ComponentType == COMPONENT_TYPE_UNORM_SRGB;
    auto color = [srgbOut](const std::array<float, 3>& c) {
        return srgbOut ? glm::vec4(SrgbToLinear(c[0]), SrgbToLinear(c[1]), SrgbToLinear(c[2]), 1.f)
                       : glm::vec4(c[0], c[1], c[2], 1.f);
    };
    {
        struct GridConstants {
            glm::mat4 viewProj, invViewProj;
            glm::vec4 cameraPos, grid, minor, major, axisX, axisY;
        };
        MapHelper<GridConstants> cb(context, gridCB, MAP_WRITE, MAP_FLAG_DISCARD);
        cb->viewProj    = viewProj;
        cb->invViewProj = glm::inverse(viewProj);
        cb->cameraPos   = glm::vec4(cameraPos, 1.f);
        cb->grid        = { gridSpacing, static_cast<float>(std::max(style.gridMajorEvery, 1)), gridFade, 0.f };
        cb->minor       = color(style.gridMinor);
        cb->major       = color(style.gridMajor);
        cb->axisX       = color(style.axisX);
        cb->axisY       = color(style.axisY);
    }
    ITextureView* dsv = hdrDepth->GetDefaultView(TEXTURE_VIEW_DEPTH_STENCIL);
    context->SetRenderTargets(1, &outputRTV, dsv, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    context->SetViewports(1, nullptr, 0, 0);
    context->SetPipelineState(gp.pso);
    context->CommitShaderResources(gp.srb, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    DrawAttribs draw{ 3, DRAW_FLAG_VERIFY_ALL };
    context->Draw(draw);
}

// ── DiligentSceneRenderer ─────────────────────────────────────────────────────

DiligentSceneRenderer::DiligentSceneRenderer(RenderDevice& device) : m_impl(std::make_unique<Impl>(device)) {}
DiligentSceneRenderer::~DiligentSceneRenderer() = default;

void DiligentSceneRenderer::SetOutputTarget(ViewportSurface& surface) {
    m_impl->outSurface   = &surface;
    m_impl->outOffscreen = nullptr;
}
void DiligentSceneRenderer::SetOutputTarget(OffscreenTarget& target) {
    m_impl->outSurface   = nullptr;
    m_impl->outOffscreen = &target;
}
void DiligentSceneRenderer::ClearOutputTarget() {
    m_impl->outSurface   = nullptr;
    m_impl->outOffscreen = nullptr;
}

void             DiligentSceneRenderer::SetExposure(float exposure) { m_impl->exposure = exposure; }
void             DiligentSceneRenderer::SetViewStyle(const ViewStyle& style) { m_impl->style = style; }
const ViewStyle& DiligentSceneRenderer::GetViewStyle() const { return m_impl->style; }
FrameStats       DiligentSceneRenderer::LastFrameStats() const { return m_impl->lastStats; }

void DiligentSceneRenderer::BeginFrame(int width, int height) {
    Impl& r = *m_impl;
    r.EnsureSceneTargets(std::max(width, 1), std::max(height, 1));
    r.stats   = {};
    r.inFrame = true;
    r.gridRequested = false;
    OverlayBeginFrame(r);

    r.BindSceneTargets();
    r.context->ClearRenderTarget(r.hdrColor->GetDefaultView(TEXTURE_VIEW_RENDER_TARGET), kClearColor,
                                 RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    r.context->ClearDepthStencil(r.hdrDepth->GetDefaultView(TEXTURE_VIEW_DEPTH_STENCIL), CLEAR_DEPTH_FLAG, 1.f, 0,
                                 RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    PbrBeginFrame(r);
}

void DiligentSceneRenderer::SetCamera(const glm::mat4& view, const glm::mat4& proj) {
    Impl& r     = *m_impl;
    r.view      = view;
    r.proj      = proj;
    r.projZO    = DepthZeroToOne(proj);
    r.viewProj  = r.projZO * view;
    r.cameraPos = glm::vec3(glm::inverse(view)[3]);
}

void DiligentSceneRenderer::SetLight(const glm::vec3& dir, const glm::vec3& color, const glm::vec3& ambient) {
    m_impl->lightDir   = dir;
    m_impl->lightColor = color;
    m_impl->ambient    = ambient;
}

void DiligentSceneRenderer::EndFrame() {
    Impl& r = *m_impl;
    if (!r.inFrame) return;
    if (ITextureView* out = r.OutputRTV()) {
        r.RunPost(out);
        if (r.gridRequested) r.RunGrid(out);
        OverlayRun(r, out);
    }
    else LogOnce(r.warnedNoOutput, "EndFrame with no output target bound; the frame is not shown");

    r.stats.meshes             = static_cast<int>(r.meshes.size());
    r.stats.textures           = static_cast<int>(r.textures.size());
    r.stats.failedTextureLoads = r.failedTextureLoads;
    r.lastStats = r.stats;
    r.inFrame   = false;
}

// ── Grid ──────────────────────────────────────────────────────────────────────

void DiligentSceneRenderer::DrawGrid(float cellSize, int halfExtent) {
    // Deferred to EndFrame: the grid is drawn over the tone-mapped image, against
    // the finished scene depth.
    Impl& r         = *m_impl;
    r.gridRequested = cellSize > 0.f;
    r.gridSpacing   = cellSize;
    r.gridFade      = cellSize * static_cast<float>(std::max(halfExtent, 1));
}

// ── Draws ─────────────────────────────────────────────────────────────────────

void DiligentSceneRenderer::DrawMesh(MeshHandle mesh, const glm::mat4& world, const DrawSurface& surface) {
    if (!m_impl->inFrame) return;
    if (const GpuMesh* m = m_impl->FindMesh(mesh)) PbrDraw(*m_impl, *m, world, {}, surface);
}

void DiligentSceneRenderer::DrawSkinnedMesh(MeshHandle mesh, const glm::mat4& world,
                                            std::span<const glm::mat4> boneTransforms, const DrawSurface& surface) {
    if (!m_impl->inFrame) return;
    if (const GpuMesh* m = m_impl->FindMesh(mesh)) PbrDraw(*m_impl, *m, world, boneTransforms, surface);
}

void DiligentSceneRenderer::DrawTerrainTile(MeshHandle, const TerrainSurface&) {
    static bool once = false;
    LogOnce(once, "DrawTerrainTile is not implemented yet; ignored");
}

} // namespace render::diligent
