// The overlays' CPU half (src/OverlayGeometry.h): UTF-8 decoding, text layout and
// anchors against a synthetic atlas, the pill, and the octahedral skeleton.
// No GPU. Exit code = number of failed checks.
#include "OverlayGeometry.h"

#include <glm/gtc/matrix_transform.hpp>

#include <cmath>
#include <cstdio>

using namespace render;
using namespace render::diligent;

namespace {

int g_failures = 0;

void check(const char* label, bool ok) {
    std::printf("[ %s ] %s\n", ok ? "PASS" : "FAIL", label);
    if (!ok) ++g_failures;
}

bool Eq(glm::vec2 a, glm::vec2 b) { return std::abs(a.x - b.x) < 1e-4f && std::abs(a.y - b.y) < 1e-4f; }

void TestUtf8() {
    std::printf("-- UTF-8 --\n");
    const auto a = ovl::DecodeUtf8("A\xC3\x84\xE2\x80\x94\xF0\x9F\x98\x80");   // A Ä — 😀
    check("ASCII, 2-, 3- and 4-byte sequences decode",
          a.size() == 4 && a[0] == 'A' && a[1] == 0xC4 && a[2] == 0x2014 && a[3] == 0x1F600);
    const auto b = ovl::DecodeUtf8("\xC3");   // truncated
    check("a truncated sequence is U+FFFD", b.size() == 1 && b[0] == 0xFFFD);
    const auto c = ovl::DecodeUtf8("\x80Z");   // stray continuation, then resync
    check("a stray continuation byte is U+FFFD and decoding resyncs", c.size() == 2 && c[0] == 0xFFFD && c[1] == 'Z');
    const auto d = ovl::DecodeUtf8("\xC0\xAF");   // overlong '/'
    check("an overlong encoding is U+FFFD", d.size() == 1 && d[0] == 0xFFFD);
}

// Two glyphs in a 16×8 atlas: 'A' 4×6 at (0,0), '?' 3×3 at (8,0); a space.
GlyphAtlas MakeAtlas() {
    GlyphAtlas a;
    a.width = 16;
    a.height = 8;
    a.pixels.assign(16 * 8, 0);
    a.lineHeight = 10.f;
    a.ascent     = 8.f;
    a.glyphs = {
        { 'A', 0, 0, 4, 6, 0.f, 0.f, 1.f, 6.f, 6.f },
        { '?', 8, 0, 3, 3, 0.f, 0.f, 0.f, 5.f, 4.f },
        { ' ', 0, 0, 0, 0, 0.f, 0.f, 0.f, 0.f, 3.f },
    };
    return a;
}

void TestLayout() {
    std::printf("-- text layout --\n");
    ovl::Font font;
    font.Assign(MakeAtlas());

    const TextExtent e = ovl::Measure(font, "AA A");
    check("Measure: width = sum of advances", std::abs(e.width - 21.f) < 1e-4f);
    check("Measure: one line's height and ascent", e.height == 10.f && e.ascent == 8.f && e.lineHeight == 10.f);
    check("Measure: lines stack", ovl::Measure(font, "A\nAA").height == 20.f && ovl::Measure(font, "A\nAA").width == 12.f);
    check("Measure without a font is zero", ovl::Measure(ovl::Font{}, "A").width == 0.f);

    // Top-left at (10, 20): baseline 28; 'A' ink at x 11..15, y 22..28.
    auto l = ovl::Layout(font, { 10.f, 20.f }, "A", {});
    check("top-left: one quad", l.quads.size() == 1);
    if (l.quads.size() == 1) {
        check("top-left: quad placed from bearing and baseline", Eq(l.quads[0].p0, { 11.f, 22.f }) && Eq(l.quads[0].p1, { 15.f, 28.f }));
        check("quad UVs cover the atlas rect", Eq(l.quads[0].uv0, { 0.f, 0.f }) && Eq(l.quads[0].uv1, { 4.f / 16.f, 6.f / 8.f }));
    }
    check("top-left: box", Eq(l.boxMin, { 10.f, 20.f }) && Eq(l.boxMax, { 16.f, 30.f }));

    l = ovl::Layout(font, { 50.f, 40.f }, "A", { TextAlignH::Right, TextAlignV::Baseline });
    check("right/baseline: the advance ends at x, the baseline at y",
          l.quads.size() == 1 && Eq(l.quads[0].p0, { 45.f, 34.f }) && Eq(l.boxMax, { 50.f, 42.f }));

    l = ovl::Layout(font, { 32.f, 32.f }, "A", { TextAlignH::Center, TextAlignV::Middle });
    check("centre/middle: the box is centred on the point",
          l.quads.size() == 1 && Eq(l.boxMin, { 29.f, 27.f }) && Eq(l.boxMax, { 35.f, 37.f }) && Eq(l.quads[0].p0, { 30.f, 29.f }));

    l = ovl::Layout(font, { 0.f, 0.f }, "A B", {});
    check("an unknown glyph draws the fallback; a space advances without a quad",
          l.quads.size() == 2 && Eq(l.quads[1].p0, { 9.f, 3.f }) && Eq(l.quads[1].uv0, { 0.5f, 0.f }));

    l = ovl::Layout(font, { 0.3f, 0.6f }, "A", {});
    check("positions snap to whole pixels", l.quads.size() == 1 && Eq(l.quads[0].p0, { 1.f, 3.f }));

    l = ovl::Layout(font, { 20.f, 0.f }, "AA\nA", { TextAlignH::Right, TextAlignV::Top });
    check("multi-line: each line aligns on its own, the second a line lower",
          l.quads.size() == 3 && Eq(l.quads[0].p0, { 9.f, 2.f }) && Eq(l.quads[2].p0, { 15.f, 12.f }) && Eq(l.boxMin, { 8.f, 0.f }));
}

void TestPill() {
    std::printf("-- pill --\n");
    std::vector<ScreenVertex> v;
    ovl::AppendRoundedRect(v, { 0.f, 0.f }, { 40.f, 10.f }, -1.f, Rgba8{ 1, 2, 3, 4 });
    bool inside = !v.empty() && v.size() % 3 == 0;
    glm::vec2 mn(1e9f), mx(-1e9f);
    for (const ScreenVertex& s : v) {
        mn = glm::min(mn, s.pos);
        mx = glm::max(mx, s.pos);
        inside = inside && s.color.r == 1 && s.color.a == 4;
    }
    check("a rounded box is triangles in its colour", inside);
    check("it spans its rect", glm::length(mn - glm::vec2(0.f)) < 1e-3f && glm::length(mx - glm::vec2(40.f, 10.f)) < 1e-3f);
    v.clear();
    ovl::AppendRoundedRect(v, { 0.f, 0.f }, { 4.f, 4.f }, 0.f, Rgba8{});
    check("radius 0 is a plain quad", v.size() == 6);
}

void TestSkeleton() {
    std::printf("-- skeleton geometry --\n");
    // root(0) -> com(1) -> spine(2) -> {arm(3), arm(4)}
    const std::vector<glm::vec3> joints = { { 0, 0, 0 }, { 0, 0, 2 }, { 0, 0, 6 }, { 4, 0, 6 }, { -4, 0, 6 } };
    const std::vector<int>       parents = { -1, 0, 1, 2, 2 };
    SkeletonStyle st;
    ovl::BoneGeometry g;
    ovl::AppendSkeleton(g, joints, parents, glm::mat4(1.f), st, { 0.f, -30.f, 4.f });
    check("three bones (root -> child skipped): 3 octahedra x 8 faces", g.triangles.size() == 3 * 8 * 3);
    check("the root gets a ring + facing tick", g.lines.size() == (32 + 1) * 2);

    st.selectedBone = 3;
    st.visible      = { 1, 1, 1, 1, 0 };
    g = {};
    ovl::AppendSkeleton(g, joints, parents, glm::mat4(1.f), st, { 0.f, -30.f, 4.f });
    check("the visibility mask hides bone 4", g.triangles.size() == 2 * 8 * 3);
    int selected = 0;
    for (const OverlayVertex& v : g.triangles)   // shaded, so compare the hue: selected is warm, bones are blue
        if (v.color.r > v.color.b) ++selected;
    check("the selected bone's faces are in the selected colour", selected == 8 * 3);

    // Placed by a world matrix: joints move, the ring stays in the world's ground plane.
    g = {};
    st = {};
    const glm::mat4 world = glm::translate(glm::mat4(1.f), glm::vec3(100.f, 0.f, 0.f));
    ovl::AppendSkeleton(g, joints, parents, world, st, { 100.f, -30.f, 4.f });
    bool moved = !g.triangles.empty();
    for (const OverlayVertex& v : g.triangles) moved = moved && v.pos.x > 90.f && v.pos.x < 110.f;
    bool flat = !g.lines.empty();
    for (const OverlayVertex& v : g.lines) flat = flat && std::abs(v.pos.z) < 1e-4f;
    check("world placement moves the bones; the root ring stays flat", moved && flat);

    // Selecting the root adds a diamond at it.
    g  = {};
    st.selectedBone = 0;
    ovl::AppendSkeleton(g, joints, parents, glm::mat4(1.f), st, { 0.f, -30.f, 4.f });
    check("a selected root gets a diamond (8 more faces)", g.triangles.size() == 4 * 8 * 3 && g.lines.front().color.g == st.selected.g);
}

} // namespace

int main() {
    TestUtf8();
    TestLayout();
    TestPill();
    TestSkeleton();
    std::printf(g_failures == 0 ? "\nALL PASSED\n" : "\n%d CHECK(S) FAILED\n", g_failures);
    return g_failures;
}
