#pragma once
// The CPU half of the overlays: UTF-8 decoding, text layout against a glyph
// atlas, the text pill, and the octahedral skeleton (after Dance Card Creator's
// DebugDraw::AppendSkeleton). Pure functions, no graphics API — unit-tested by
// tests/OverlayGeometryTest.cpp; Overlay.cpp turns their output into draws.
#include <render/Diligent.h>

#include <glm/glm.hpp>

#include <cstdint>
#include <span>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace render::diligent::ovl {

// UTF-8 → code points. Malformed or truncated sequences become U+FFFD.
std::vector<std::uint32_t> DecodeUtf8(std::string_view s);

// A glyph atlas indexed for layout.
struct Font {
    std::unordered_map<std::uint32_t, GlyphMetrics> glyphs;
    float         lineHeight = 0.f, ascent = 0.f;
    int           atlasW = 0, atlasH = 0;
    std::uint32_t fallback = '?';

    void Assign(const GlyphAtlas& atlas);
    bool Empty() const { return glyphs.empty() || atlasW <= 0 || atlasH <= 0; }
    // The glyph for `cp`: itself, else the fallback, else null.
    const GlyphMetrics* Find(std::uint32_t cp) const;
};

struct GlyphQuad {
    glm::vec2 p0, p1;     // pixels, top-left / bottom-right
    glm::vec2 uv0, uv1;
};

struct TextLayout {
    std::vector<GlyphQuad> quads;
    glm::vec2              boxMin{ 0.f }, boxMax{ 0.f };   // the line boxes' union
};

TextExtent Measure(const Font& font, std::string_view utf8);
// Pens and baselines snap to whole pixels, so a font baked at the display's
// density maps texels 1:1 to pixels.
TextLayout Layout(const Font& font, glm::vec2 pos, std::string_view utf8, TextAnchor anchor);

// A filled rounded rectangle (convex, a fan around its centre) as triangles.
void AppendRoundedRect(std::vector<ScreenVertex>& out, glm::vec2 min, glm::vec2 max, float radius, Rgba8 color);

// Octahedral bones for model-space joints placed by `world` (assumed unscaled:
// widths and ring radii are world units). Triangles get baked flat shading from a
// light at the eye (`eye` = camera position), so faces turned to the viewer read
// brightest; root rings are lines, emitted separately (they carry a pixel width).
// The ring lies in the plane normal to world's +Z, its tick along world's +Y.
struct BoneGeometry {
    std::vector<OverlayVertex> triangles;   // triples
    std::vector<OverlayVertex> lines;       // pairs (root rings)
};
void AppendSkeleton(BoneGeometry& out, std::span<const glm::vec3> modelJoints, std::span<const int> parents,
                    const glm::mat4& world, const SkeletonStyle& style, const glm::vec3& eye);

} // namespace render::diligent::ovl
