#include "OverlayGeometry.h"

#include <algorithm>
#include <cmath>

namespace render::diligent::ovl {

// ── UTF-8 ─────────────────────────────────────────────────────────────────────

std::vector<std::uint32_t> DecodeUtf8(std::string_view s) {
    constexpr std::uint32_t kBad = 0xFFFD;
    std::vector<std::uint32_t> out;
    out.reserve(s.size());
    std::size_t i = 0;
    while (i < s.size()) {
        const auto b0 = static_cast<unsigned char>(s[i]);
        int           extra;
        std::uint32_t cp;
        if (b0 < 0x80) { out.push_back(b0); ++i; continue; }
        if ((b0 & 0xE0) == 0xC0) { extra = 1; cp = b0 & 0x1Fu; }
        else if ((b0 & 0xF0) == 0xE0) { extra = 2; cp = b0 & 0x0Fu; }
        else if ((b0 & 0xF8) == 0xF0) { extra = 3; cp = b0 & 0x07u; }
        else { out.push_back(kBad); ++i; continue; }   // a stray continuation / invalid lead byte
        std::size_t j = i + 1;
        bool        ok = true;
        for (int k = 0; k < extra; ++k, ++j) {
            if (j >= s.size() || (static_cast<unsigned char>(s[j]) & 0xC0) != 0x80) { ok = false; break; }
            cp = (cp << 6) | (static_cast<unsigned char>(s[j]) & 0x3Fu);
        }
        static constexpr std::uint32_t kMin[4] = { 0, 0x80, 0x800, 0x10000 };
        if (!ok || cp < kMin[extra] || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) {
            out.push_back(kBad);
            i = j;   // past the sequence, or on the byte that broke it (resync there)
            continue;
        }
        out.push_back(cp);
        i = j;
    }
    return out;
}

// ── Font + layout ─────────────────────────────────────────────────────────────

void Font::Assign(const GlyphAtlas& atlas) {
    glyphs.clear();
    for (const GlyphMetrics& g : atlas.glyphs) glyphs[g.codepoint] = g;
    lineHeight = atlas.lineHeight;
    ascent     = atlas.ascent;
    atlasW     = atlas.width;
    atlasH     = atlas.height;
    fallback   = atlas.fallback;
}

const GlyphMetrics* Font::Find(std::uint32_t cp) const {
    auto it = glyphs.find(cp);
    if (it != glyphs.end()) return &it->second;
    it = glyphs.find(fallback);
    return it != glyphs.end() ? &it->second : nullptr;
}

namespace {

// Code points split into lines at '\n' ('\r' is dropped).
std::vector<std::vector<std::uint32_t>> Lines(std::string_view utf8) {
    std::vector<std::vector<std::uint32_t>> lines(1);
    for (std::uint32_t cp : DecodeUtf8(utf8)) {
        if (cp == '\n') lines.emplace_back();
        else if (cp != '\r') lines.back().push_back(cp);
    }
    return lines;
}

float LineAdvance(const Font& font, const std::vector<std::uint32_t>& line) {
    float w = 0.f;
    for (std::uint32_t cp : line)
        if (const GlyphMetrics* g = font.Find(cp)) w += g->advance;
    return w;
}

} // namespace

TextExtent Measure(const Font& font, std::string_view utf8) {
    TextExtent e;
    if (font.Empty()) return e;
    const auto lines = Lines(utf8);
    for (const auto& l : lines) e.width = std::max(e.width, LineAdvance(font, l));
    e.lineHeight = font.lineHeight;
    e.ascent     = font.ascent;
    e.height     = font.lineHeight * static_cast<float>(lines.size());
    return e;
}

TextLayout Layout(const Font& font, glm::vec2 pos, std::string_view utf8, TextAnchor anchor) {
    TextLayout out;
    if (font.Empty()) return out;
    const auto  lines  = Lines(utf8);
    const float totalH = font.lineHeight * static_cast<float>(lines.size());
    float top = pos.y;
    if (anchor.v == TextAlignV::Middle) top = pos.y - totalH * 0.5f;
    else if (anchor.v == TextAlignV::Baseline) top = pos.y - font.ascent;
    top = std::round(top);

    const float invW = 1.f / static_cast<float>(font.atlasW), invH = 1.f / static_cast<float>(font.atlasH);
    float       minX = 0.f, maxX = 0.f;
    for (std::size_t k = 0; k < lines.size(); ++k) {
        const float lineW = LineAdvance(font, lines[k]);
        float       x     = pos.x;
        if (anchor.h == TextAlignH::Center) x -= lineW * 0.5f;
        else if (anchor.h == TextAlignH::Right) x -= lineW;
        x = std::round(x);
        minX = k == 0 ? x : std::min(minX, x);
        maxX = k == 0 ? x + lineW : std::max(maxX, x + lineW);

        const float baseline = top + std::round(font.ascent + font.lineHeight * static_cast<float>(k));
        float       pen      = x;
        for (std::uint32_t cp : lines[k]) {
            const GlyphMetrics* g = font.Find(cp);
            if (!g) continue;
            if (g->w > 0 && g->h > 0) {
                GlyphQuad q;
                q.p0  = { std::round(pen + g->bearingX), baseline - std::round(g->bearingY) };
                q.p1  = q.p0 + glm::vec2(g->sizeW > 0.f ? g->sizeW : static_cast<float>(g->w),
                                         g->sizeH > 0.f ? g->sizeH : static_cast<float>(g->h));
                q.uv0 = { static_cast<float>(g->x) * invW, static_cast<float>(g->y) * invH };
                q.uv1 = { static_cast<float>(g->x + g->w) * invW, static_cast<float>(g->y + g->h) * invH };
                out.quads.push_back(q);
            }
            pen += g->advance;
        }
    }
    out.boxMin = { minX, top };
    out.boxMax = { maxX, top + totalH };
    return out;
}

void AppendRoundedRect(std::vector<ScreenVertex>& out, glm::vec2 mn, glm::vec2 mx, float radius, Rgba8 color) {
    const glm::vec2 size = mx - mn;
    if (size.x <= 0.f || size.y <= 0.f) return;
    const float half = 0.5f * std::min(size.x, size.y);
    const float r    = radius < 0.f ? half : std::min(radius, half);
    if (r < 0.5f) {
        const glm::vec2 c[4] = { mn, { mx.x, mn.y }, mx, { mn.x, mx.y } };
        for (int i : { 0, 1, 2, 0, 2, 3 }) out.push_back({ c[i], color });
        return;
    }
    // The perimeter, corner arcs clockwise on screen from the top-right corner.
    constexpr int   kSeg = 8;
    constexpr float kHalfPi = 1.57079633f;
    const glm::vec2 centres[4] = { { mx.x - r, mn.y + r }, { mx.x - r, mx.y - r }, { mn.x + r, mx.y - r }, { mn.x + r, mn.y + r } };
    std::vector<glm::vec2> ring;
    for (int c = 0; c < 4; ++c)
        for (int i = 0; i <= kSeg; ++i) {
            const float a = kHalfPi * (static_cast<float>(c) - 1.f + static_cast<float>(i) / kSeg);   // -90°.. per corner
            ring.push_back(centres[c] + r * glm::vec2(std::cos(a), std::sin(a)));
        }
    const glm::vec2 centre = 0.5f * (mn + mx);
    for (std::size_t i = 0; i < ring.size(); ++i) {
        out.push_back({ centre, color });
        out.push_back({ ring[i], color });
        out.push_back({ ring[(i + 1) % ring.size()], color });
    }
}

// ── Skeleton ──────────────────────────────────────────────────────────────────

namespace {

glm::vec3 SafeNormalize(const glm::vec3& v, const glm::vec3& fallback) {
    const float l = glm::length(v);
    return l > 1e-6f ? v / l : fallback;
}

Rgba8 Shade(Rgba8 c, const glm::vec3& normal, const glm::vec3& toLight) {
    const float k  = 0.45f + 0.55f * std::max(0.f, glm::dot(normal, toLight));
    auto        ch = [k](std::uint8_t v) { return static_cast<std::uint8_t>(std::min(255.f, std::round(static_cast<float>(v) * k))); };
    return { ch(c.r), ch(c.g), ch(c.b), c.a };
}

struct SkelBuilder {
    BoneGeometry&   out;
    const glm::vec3 eye;

    void Face(const glm::vec3& a, const glm::vec3& b, const glm::vec3& c, Rgba8 color, const glm::vec3& centre) {
        glm::vec3 n = SafeNormalize(glm::cross(b - a, c - a), { 0.f, 0.f, 1.f });
        if (glm::dot(n, a - centre) < 0.f) n = -n;   // outward, whatever the winding
        const glm::vec3 faceCentre = (a + b + c) / 3.f;
        const Rgba8     s          = Shade(color, n, SafeNormalize(eye - faceCentre, n));
        out.triangles.push_back({ a, s });
        out.triangles.push_back({ b, s });
        out.triangles.push_back({ c, s });
    }

    void Octahedron(const glm::vec3& head, const glm::vec3& tail, Rgba8 color, const SkeletonStyle& st) {
        const glm::vec3 axis   = tail - head;
        const float     length = glm::length(axis);
        if (length < 1e-3f) return;
        const glm::vec3 dir    = axis / length;
        const glm::vec3 helper = std::abs(dir.z) < 0.9f ? glm::vec3(0.f, 0.f, 1.f) : glm::vec3(1.f, 0.f, 0.f);
        const glm::vec3 u      = glm::normalize(glm::cross(dir, helper));
        const glm::vec3 v      = glm::cross(dir, u);
        const float     w      = std::clamp(st.widthRatio * length, std::min(st.minWidth, 0.3f * length), st.maxWidth);
        const glm::vec3 mid    = head + dir * (0.2f * length);
        const glm::vec3 ring[4] = { mid + u * w, mid + v * w, mid - u * w, mid - v * w };
        for (int i = 0; i < 4; ++i) {
            const glm::vec3& a = ring[i];
            const glm::vec3& b = ring[(i + 1) % 4];
            Face(head, a, b, color, mid);
            Face(tail, b, a, color, mid);
        }
    }

    void Diamond(const glm::vec3& c, float r, Rgba8 color) {
        const glm::vec3 tip[6] = { c + glm::vec3(r, 0, 0), c + glm::vec3(0, r, 0), c + glm::vec3(-r, 0, 0),
                                   c + glm::vec3(0, -r, 0), c + glm::vec3(0, 0, r), c + glm::vec3(0, 0, -r) };
        for (int i = 0; i < 4; ++i) {
            Face(tip[i], tip[(i + 1) % 4], tip[4], color, c);
            Face(tip[(i + 1) % 4], tip[i], tip[5], color, c);
        }
    }

    void RootMarker(const glm::vec3& at, const glm::vec3& up, const glm::vec3& fwd, float r, Rgba8 color) {
        const glm::vec3 side = SafeNormalize(glm::cross(fwd, up), { 1.f, 0.f, 0.f });
        constexpr int   kSeg = 32;
        auto point = [&](int i) {
            const float a = 6.2831853f * static_cast<float>(i) / kSeg;
            return at + r * (std::cos(a) * side + std::sin(a) * fwd);
        };
        for (int i = 0; i < kSeg; ++i) {
            out.lines.push_back({ point(i), color });
            out.lines.push_back({ point(i + 1), color });
        }
        out.lines.push_back({ at, color });
        out.lines.push_back({ at + fwd * (r * 1.6f), color });   // facing
    }
};

} // namespace

void AppendSkeleton(BoneGeometry& out, std::span<const glm::vec3> modelJoints, std::span<const int> parents,
                    const glm::mat4& world, const SkeletonStyle& style, const glm::vec3& eye) {
    const std::size_t n = std::min(modelJoints.size(), parents.size());
    std::vector<glm::vec3> joints(n);
    for (std::size_t i = 0; i < n; ++i) joints[i] = glm::vec3(world * glm::vec4(modelJoints[i], 1.f));
    const glm::vec3 up  = SafeNormalize(glm::vec3(world * glm::vec4(0.f, 0.f, 1.f, 0.f)), { 0.f, 0.f, 1.f });
    const glm::vec3 fwd = SafeNormalize(glm::vec3(world * glm::vec4(0.f, 1.f, 0.f, 0.f)), { 0.f, 1.f, 0.f });

    SkelBuilder b{ out, eye };
    auto colorOf = [&](int i, Rgba8 base) {
        if (i == style.selectedBone) return style.selected;
        if (i == style.hoveredBone) return style.hovered;
        return base;
    };
    for (std::size_t i = 0; i < n; ++i) {
        if (i < style.visible.size() && !style.visible[i]) continue;
        const int bone = static_cast<int>(i);
        const int p    = parents[i];
        if (p < 0 || static_cast<std::size_t>(p) >= n) {
            if (p >= 0) continue;   // a parent outside the joint list: nothing to draw from
            b.RootMarker(joints[i], up, fwd, style.rootSize, colorOf(bone, style.root));
            if (bone == style.selectedBone || bone == style.hoveredBone)   // a ring alone is easy to miss
                b.Diamond(joints[i], std::max(style.rootSize * 0.2f, style.minWidth), colorOf(bone, style.root));
            continue;
        }
        if (parents[static_cast<std::size_t>(p)] < 0) continue;   // root -> child: an offset, not a limb
        b.Octahedron(joints[static_cast<std::size_t>(p)], joints[i], colorOf(bone, style.bone), style);
    }
}

} // namespace render::diligent::ovl
