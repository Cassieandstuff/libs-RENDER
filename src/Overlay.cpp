// The overlay layers (IOverlayRenderer + DrawLines): every call
// appends to a per-frame batch; EndFrame draws them over the tone-mapped output,
// after the grid, in three layers — Tested (scene depth, read only), OnTop (the
// scene depth cleared, then read + written, so on-top primitives sort among
// themselves), Screen (no depth) — each in call order. Colours are display values
// (see Shaders.h), so they land exactly.
#include "MathConv.h"
#include "OverlayGeometry.h"
#include "SceneRendererImpl.h"
#include "Shaders.h"

#include "CommonlyUsedStates.h"
#include "GraphicsAccessories.hpp"
#include "MapHelper.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <map>

namespace render::diligent {

using namespace Diligent;

namespace {

std::uint32_t Pack(Rgba8 c) {
    return std::uint32_t(c.r) | std::uint32_t(c.g) << 8 | std::uint32_t(c.b) << 16 | std::uint32_t(c.a) << 24;
}
std::uint32_t Pack(const LineVertex& v) { return Pack(Rgba8{ v.r, v.g, v.b, v.a }); }

// GPU layouts (see the input layouts in PipelineFor).
struct LineInstance {
    glm::vec3     a;
    std::uint32_t ca;
    glm::vec3     b;
    std::uint32_t cb;
    float         width;
};
struct TriVertex {
    glm::vec3     pos;
    std::uint32_t col;
};
struct TextVertex {
    glm::vec2     pos;
    glm::vec2     uv;
    std::uint32_t col;
};
static_assert(sizeof(LineInstance) == 36 && sizeof(TriVertex) == 16 && sizeof(TextVertex) == 20,
              "overlay vertex structs must match their input layouts");

enum class Layer : std::uint8_t { Tested, OnTop, Screen };
enum class Prim : std::uint8_t { Lines, Triangles, Text };
// Per layer: Tested reads the scene depth; Sorted reads + writes the on-top depth; None.
enum class DepthState : std::uint8_t { Tested, Sorted, None };

struct Cmd {
    Layer         layer;
    Prim          prim;
    std::uint32_t first, count;   // instances (lines) / vertices
};

struct OverlayCB {
    glm::mat4 viewProj;
    glm::vec4 viewport;   // w, h, 1/w, 1/h
    glm::vec4 mode;       // x = screen space
};

struct GrowBuffer {
    RefCntAutoPtr<IBuffer> buf;
    Uint64                 capacity = 0;   // bytes
};

Layer LayerOf(OverlayDepth d) { return d == OverlayDepth::OnTop ? Layer::OnTop : Layer::Tested; }

void LogOnce(bool& flag, const char* what) {
    if (flag) return;
    flag = true;
    std::fprintf(stderr, "[render] %s\n", what);
}

} // namespace

struct OverlayState {
    std::vector<LineInstance> lines;
    std::vector<TriVertex>    tris;
    std::vector<TextVertex>   text;
    std::vector<Cmd>          cmds;

    GrowBuffer             lineVB, triVB, textVB;
    RefCntAutoPtr<IBuffer> cb;

    struct Pipeline {
        RefCntAutoPtr<IPipelineState>         pso;
        RefCntAutoPtr<IShaderResourceBinding> srb;
    };
    std::map<int, Pipeline> pipelines;   // keyed by (output format, prim, depth state)

    ovl::Font               font;
    RefCntAutoPtr<ITexture> atlas;
    SkeletonStyle           skeleton;

    bool warnedSize = false, warnedNoFont = false;

    // Append a command, merging with the previous one when it continues it.
    void Push(Layer layer, Prim prim, std::size_t first, std::size_t count) {
        if (count == 0) return;
        const auto f = static_cast<std::uint32_t>(first), c = static_cast<std::uint32_t>(count);
        if (!cmds.empty()) {
            Cmd& last = cmds.back();
            if (last.layer == layer && last.prim == prim && last.first + last.count == f) {
                last.count += c;
                return;
            }
        }
        cmds.push_back({ layer, prim, f, c });
    }
};

void DestroyOverlayState(OverlayState* o) { delete o; }

OverlayPtr CreateOverlayState(DiligentSceneRenderer::Impl& r) {
    OverlayPtr st(new OverlayState{}, &DestroyOverlayState);
    BufferDesc desc;
    desc.Name           = "render overlay CB";
    desc.Size           = sizeof(OverlayCB);
    desc.Usage          = USAGE_DYNAMIC;
    desc.BindFlags      = BIND_UNIFORM_BUFFER;
    desc.CPUAccessFlags = CPU_ACCESS_WRITE;
    r.device->CreateBuffer(desc, nullptr, &st->cb);
    return st;
}

void OverlayBeginFrame(DiligentSceneRenderer::Impl& r) {
    OverlayState& o = *r.overlay;
    o.lines.clear();
    o.tris.clear();
    o.text.clear();
    o.cmds.clear();
}

namespace {

OverlayState::Pipeline& PipelineFor(DiligentSceneRenderer::Impl& r, TEXTURE_FORMAT format, Prim prim, DepthState depth) {
    OverlayState& o   = *r.overlay;
    const int     key = static_cast<int>(format) << 8 | static_cast<int>(prim) << 4 | static_cast<int>(depth);
    auto [it, inserted] = o.pipelines.try_emplace(key);
    OverlayState::Pipeline& p = it->second;
    if (!inserted) return p;

    const bool        srgb     = GetTextureFormatAttribs(format).ComponentType == COMPONENT_TYPE_UNORM_SRGB;
    const ShaderMacro macros[] = { { "LINEAR_OUT", srgb ? "1" : "0" } };
    const char*       vsSrc    = prim == Prim::Lines ? shaders::kOverlayLineVS : prim == Prim::Triangles ? shaders::kOverlayTriVS : shaders::kOverlayTextVS;
    const char*       psSrc    = prim == Prim::Lines ? shaders::kOverlayLinePS : prim == Prim::Triangles ? shaders::kOverlayTriPS : shaders::kOverlayTextPS;
    auto vs = r.CompileShader("render overlay VS", SHADER_TYPE_VERTEX, vsSrc);
    auto ps = r.CompileShader("render overlay PS", SHADER_TYPE_PIXEL, psSrc, macros, _countof(macros));
    if (!vs || !ps) return p;

    GraphicsPipelineStateCreateInfo ci;
    ci.PSODesc.Name = "render overlay PSO";
    GraphicsPipelineDesc& g = ci.GraphicsPipeline;
    g.NumRenderTargets                  = 1;
    g.RTVFormats[0]                     = format;
    g.DSVFormat                         = depth == DepthState::None ? TEX_FORMAT_UNKNOWN : kSceneDepthFormat;
    g.PrimitiveTopology                 = PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    g.RasterizerDesc.CullMode           = CULL_MODE_NONE;
    g.RasterizerDesc.FrontCounterClockwise = True;
    g.DepthStencilDesc.DepthEnable      = depth != DepthState::None;
    g.DepthStencilDesc.DepthWriteEnable = depth == DepthState::Sorted;
    g.DepthStencilDesc.DepthFunc        = COMPARISON_FUNC_LESS_EQUAL;
    RenderTargetBlendDesc& rt = g.BlendDesc.RenderTargets[0];
    rt.BlendEnable    = True;
    rt.SrcBlend       = BLEND_FACTOR_SRC_ALPHA;
    rt.DestBlend      = BLEND_FACTOR_INV_SRC_ALPHA;
    rt.BlendOp        = BLEND_OPERATION_ADD;
    rt.SrcBlendAlpha  = BLEND_FACTOR_ONE;
    rt.DestBlendAlpha = BLEND_FACTOR_INV_SRC_ALPHA;
    rt.BlendOpAlpha   = BLEND_OPERATION_ADD;

    constexpr auto kInst = INPUT_ELEMENT_FREQUENCY_PER_INSTANCE;
    const LayoutElement lineLayout[] = {
        { 0, 0, 3, VT_FLOAT32, False, kInst }, { 1, 0, 4, VT_UINT8, True, kInst },  { 2, 0, 3, VT_FLOAT32, False, kInst },
        { 3, 0, 4, VT_UINT8, True, kInst },    { 4, 0, 1, VT_FLOAT32, False, kInst },
    };
    const LayoutElement triLayout[]  = { { 0, 0, 3, VT_FLOAT32, False }, { 1, 0, 4, VT_UINT8, True } };
    const LayoutElement textLayout[] = { { 0, 0, 2, VT_FLOAT32, False }, { 1, 0, 2, VT_FLOAT32, False }, { 2, 0, 4, VT_UINT8, True } };
    if (prim == Prim::Lines) g.InputLayout = { lineLayout, _countof(lineLayout) };
    else if (prim == Prim::Triangles) g.InputLayout = { triLayout, _countof(triLayout) };
    else g.InputLayout = { textLayout, _countof(textLayout) };

    ci.pVS = vs;
    ci.pPS = ps;
    const ShaderResourceVariableDesc vars[]     = { { SHADER_TYPE_PIXEL, "g_Atlas", SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC } };
    const ImmutableSamplerDesc       samplers[] = { { SHADER_TYPE_PIXEL, "g_Atlas", Sam_LinearClamp } };
    ci.PSODesc.ResourceLayout.DefaultVariableType = SHADER_RESOURCE_VARIABLE_TYPE_STATIC;
    if (prim == Prim::Text) {
        ci.PSODesc.ResourceLayout.Variables            = vars;
        ci.PSODesc.ResourceLayout.NumVariables         = _countof(vars);
        ci.PSODesc.ResourceLayout.ImmutableSamplers    = samplers;
        ci.PSODesc.ResourceLayout.NumImmutableSamplers = _countof(samplers);
    }

    r.device->CreateGraphicsPipelineState(ci, &p.pso);
    if (!p.pso) return p;
    if (auto* var = p.pso->GetStaticVariableByName(SHADER_TYPE_VERTEX, "OverlayCB")) var->Set(o.cb);
    p.pso->CreateShaderResourceBinding(&p.srb, true);
    return p;
}

void Upload(DiligentSceneRenderer::Impl& r, GrowBuffer& gb, const void* data, std::size_t bytes, const char* name) {
    if (bytes == 0) return;
    if (bytes > gb.capacity) {
        gb.buf.Release();
        gb.capacity = std::max<Uint64>(static_cast<Uint64>(bytes) * 3 / 2, 64 * 1024);
        BufferDesc desc;
        desc.Name           = name;
        desc.Size           = gb.capacity;
        desc.Usage          = USAGE_DYNAMIC;
        desc.BindFlags      = BIND_VERTEX_BUFFER;
        desc.CPUAccessFlags = CPU_ACCESS_WRITE;
        r.device->CreateBuffer(desc, nullptr, &gb.buf);
        if (!gb.buf) {
            gb.capacity = 0;
            return;
        }
    }
    MapHelper<std::uint8_t> dst(r.context, gb.buf, MAP_WRITE, MAP_FLAG_DISCARD);
    std::memcpy(dst, data, bytes);
}

} // namespace

void OverlayRun(DiligentSceneRenderer::Impl& r, ITextureView* outputRTV) {
    OverlayState& o = *r.overlay;
    if (o.cmds.empty() || !o.cb) return;

    const TextureDesc& od     = outputRTV->GetTexture()->GetDesc();
    const bool         sameSz = static_cast<int>(od.Width) == r.width && static_cast<int>(od.Height) == r.height;
    const TEXTURE_FORMAT format = outputRTV->GetDesc().Format;

    Upload(r, o.lineVB, o.lines.data(), o.lines.size() * sizeof(LineInstance), "render overlay lines");
    Upload(r, o.triVB, o.tris.data(), o.tris.size() * sizeof(TriVertex), "render overlay triangles");
    Upload(r, o.textVB, o.text.data(), o.text.size() * sizeof(TextVertex), "render overlay text");

    ITextureView* dsv = r.hdrDepth->GetDefaultView(TEXTURE_VIEW_DEPTH_STENCIL);
    for (Layer layer : { Layer::Tested, Layer::OnTop, Layer::Screen }) {
        if (std::none_of(o.cmds.begin(), o.cmds.end(), [layer](const Cmd& c) { return c.layer == layer; })) continue;
        const bool screen = layer == Layer::Screen;
        if (!screen && !sameSz) {   // the world layers test against the scene depth
            LogOnce(o.warnedSize, "output target size differs from the frame size; world-space overlays skipped");
            continue;
        }
        {
            MapHelper<OverlayCB> cb(r.context, o.cb, MAP_WRITE, MAP_FLAG_DISCARD);
            const float w = static_cast<float>(od.Width), h = static_cast<float>(od.Height);
            cb->viewProj = r.viewProj;
            cb->viewport = { w, h, 1.f / w, 1.f / h };
            cb->mode     = { screen ? 1.f : 0.f, 0.f, 0.f, 0.f };
        }
        r.context->SetRenderTargets(1, &outputRTV, screen ? nullptr : dsv, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
        r.context->SetViewports(1, nullptr, 0, 0);
        // The tested layer is done with the scene depth: on-top primitives get it
        // cleared, to sort among themselves.
        if (layer == Layer::OnTop)
            r.context->ClearDepthStencil(dsv, CLEAR_DEPTH_FLAG, 1.f, 0, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
        const DepthState depth = layer == Layer::Tested ? DepthState::Tested : layer == Layer::OnTop ? DepthState::Sorted : DepthState::None;

        for (const Cmd& c : o.cmds) {
            if (c.layer != layer) continue;
            if (c.prim == Prim::Text && !o.atlas) continue;
            OverlayState::Pipeline& p = PipelineFor(r, format, c.prim, depth);
            if (!p.pso) continue;
            IBuffer* vb = c.prim == Prim::Lines ? o.lineVB.buf.RawPtr() : c.prim == Prim::Triangles ? o.triVB.buf.RawPtr() : o.textVB.buf.RawPtr();
            if (!vb) continue;
            if (c.prim == Prim::Text)
                p.srb->GetVariableByName(SHADER_TYPE_PIXEL, "g_Atlas")->Set(o.atlas->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE));
            r.context->SetPipelineState(p.pso);
            r.context->CommitShaderResources(p.srb, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
            const Uint64 offset = 0;
            r.context->SetVertexBuffers(0, 1, &vb, &offset, RESOURCE_STATE_TRANSITION_MODE_TRANSITION, SET_VERTEX_BUFFERS_FLAG_RESET);
            DrawAttribs draw;
            draw.Flags = DRAW_FLAG_VERIFY_ALL;
            if (c.prim == Prim::Lines) {
                draw.NumVertices           = 6;
                draw.NumInstances          = c.count;
                draw.FirstInstanceLocation = c.first;
            } else {
                draw.NumVertices         = c.count;
                draw.StartVertexLocation = c.first;
            }
            r.context->Draw(draw);
        }
    }
}

// ── DiligentSceneRenderer: overlay calls ──────────────────────────────────────

void DiligentSceneRenderer::DrawLines(std::span<const LineVertex> segments, float widthPx) {
    if (!m_impl->inFrame) return;
    OverlayState&     o     = *m_impl->overlay;
    const std::size_t first = o.lines.size();
    for (std::size_t i = 0; i + 1 < segments.size(); i += 2)
        o.lines.push_back({ segments[i].pos, Pack(segments[i]), segments[i + 1].pos, Pack(segments[i + 1]), widthPx });
    o.Push(Layer::Tested, Prim::Lines, first, o.lines.size() - first);
}

void DiligentSceneRenderer::DrawOverlayLines(std::span<const OverlayVertex> segments, float widthPx, OverlayDepth depth) {
    if (!m_impl->inFrame) return;
    OverlayState&     o     = *m_impl->overlay;
    const std::size_t first = o.lines.size();
    for (std::size_t i = 0; i + 1 < segments.size(); i += 2)
        o.lines.push_back({ segments[i].pos, Pack(segments[i].color), segments[i + 1].pos, Pack(segments[i + 1].color), widthPx });
    o.Push(LayerOf(depth), Prim::Lines, first, o.lines.size() - first);
}

void DiligentSceneRenderer::DrawOverlayTriangles(std::span<const OverlayVertex> triangles, OverlayDepth depth) {
    if (!m_impl->inFrame) return;
    OverlayState&     o     = *m_impl->overlay;
    const std::size_t first = o.tris.size();
    for (std::size_t i = 0; i + 2 < triangles.size(); i += 3)
        for (std::size_t k = 0; k < 3; ++k) o.tris.push_back({ triangles[i + k].pos, Pack(triangles[i + k].color) });
    o.Push(LayerOf(depth), Prim::Triangles, first, o.tris.size() - first);
}

void DiligentSceneRenderer::DrawScreenLines(std::span<const ScreenVertex> segments, float widthPx) {
    if (!m_impl->inFrame) return;
    OverlayState&     o     = *m_impl->overlay;
    const std::size_t first = o.lines.size();
    for (std::size_t i = 0; i + 1 < segments.size(); i += 2)
        o.lines.push_back({ glm::vec3(segments[i].pos, 0.f), Pack(segments[i].color), glm::vec3(segments[i + 1].pos, 0.f),
                            Pack(segments[i + 1].color), widthPx });
    o.Push(Layer::Screen, Prim::Lines, first, o.lines.size() - first);
}

void DiligentSceneRenderer::DrawScreenTriangles(std::span<const ScreenVertex> triangles) {
    if (!m_impl->inFrame) return;
    OverlayState&     o     = *m_impl->overlay;
    const std::size_t first = o.tris.size();
    for (std::size_t i = 0; i + 2 < triangles.size(); i += 3)
        for (std::size_t k = 0; k < 3; ++k) o.tris.push_back({ glm::vec3(triangles[i + k].pos, 0.f), Pack(triangles[i + k].color) });
    o.Push(Layer::Screen, Prim::Triangles, first, o.tris.size() - first);
}

void DiligentSceneRenderer::DrawScreenText(glm::vec2 pos, std::string_view utf8, Rgba8 color, TextAnchor anchor, const TextPill* pill) {
    if (!m_impl->inFrame) return;
    OverlayState& o = *m_impl->overlay;
    if (o.font.Empty() || !o.atlas) {
        LogOnce(o.warnedNoFont, "DrawScreenText with no glyph atlas set; text skipped");
        return;
    }
    const ovl::TextLayout layout = ovl::Layout(o.font, pos, utf8, anchor);
    if (pill) {
        std::vector<ScreenVertex> box;
        const glm::vec2           pad(pill->padX, pill->padY);
        ovl::AppendRoundedRect(box, layout.boxMin - pad, layout.boxMax + pad, pill->radius, pill->color);
        DrawScreenTriangles(box);
    }
    const std::uint32_t col   = Pack(color);
    const std::size_t   first = o.text.size();
    for (const ovl::GlyphQuad& q : layout.quads) {
        const TextVertex tl{ q.p0, q.uv0, col }, tr{ { q.p1.x, q.p0.y }, { q.uv1.x, q.uv0.y }, col };
        const TextVertex br{ q.p1, q.uv1, col }, bl{ { q.p0.x, q.p1.y }, { q.uv0.x, q.uv1.y }, col };
        for (const TextVertex& v : { tl, tr, br, tl, br, bl }) o.text.push_back(v);
    }
    o.Push(Layer::Screen, Prim::Text, first, o.text.size() - first);
}

TextExtent DiligentSceneRenderer::MeasureText(std::string_view utf8) const { return ovl::Measure(m_impl->overlay->font, utf8); }

void DiligentSceneRenderer::SetGlyphAtlas(const GlyphAtlas& atlas) {
    OverlayState& o = *m_impl->overlay;
    o.atlas.Release();
    o.font = {};
    const std::size_t bpp = atlas.format == GlyphAtlas::Format::R8 ? 1 : 4;
    if (atlas.width <= 0 || atlas.height <= 0 || atlas.glyphs.empty() ||
        atlas.pixels.size() != static_cast<std::size_t>(atlas.width) * atlas.height * bpp)
        return;

    // Uploaded as RGBA8 either way: R8 coverage c becomes (255, 255, 255, c), so
    // one shader serves both (and no swizzle support is needed).
    std::vector<std::uint8_t> rgba;
    const std::uint8_t*       src = atlas.pixels.data();
    if (bpp == 1) {
        rgba.resize(atlas.pixels.size() * 4);
        for (std::size_t i = 0; i < atlas.pixels.size(); ++i) {
            rgba[i * 4 + 0] = rgba[i * 4 + 1] = rgba[i * 4 + 2] = 255;
            rgba[i * 4 + 3] = atlas.pixels[i];
        }
        src = rgba.data();
    }
    TextureDesc desc;
    desc.Name      = "render glyph atlas";
    desc.Type      = RESOURCE_DIM_TEX_2D;
    desc.Width     = static_cast<Uint32>(atlas.width);
    desc.Height    = static_cast<Uint32>(atlas.height);
    desc.MipLevels = 1;
    desc.Format    = TEX_FORMAT_RGBA8_UNORM;
    desc.Usage     = USAGE_IMMUTABLE;
    desc.BindFlags = BIND_SHADER_RESOURCE;
    TextureSubResData sub{ src, static_cast<Uint64>(atlas.width) * 4 };
    TextureData       init{ &sub, 1 };
    m_impl->device->CreateTexture(desc, &init, &o.atlas);
    if (o.atlas) o.font.Assign(atlas);
}

bool DiligentSceneRenderer::HasGlyphAtlas() const { return m_impl->overlay->atlas && !m_impl->overlay->font.Empty(); }

void DiligentSceneRenderer::SetSkeletonStyle(const SkeletonStyle& style) { m_impl->overlay->skeleton = style; }

void DiligentSceneRenderer::DrawBones(std::span<const glm::vec3> joints, std::span<const int> parents, const glm::mat4& world) {
    if (!m_impl->inFrame) return;
    const SkeletonStyle& st = m_impl->overlay->skeleton;
    ovl::BoneGeometry    geo;
    ovl::AppendSkeleton(geo, joints, parents, world, st, m_impl->cameraPos);
    DrawOverlayTriangles(geo.triangles, st.depth);
    DrawOverlayLines(geo.lines, st.rootLinePx, st.depth);
}

} // namespace render::diligent
