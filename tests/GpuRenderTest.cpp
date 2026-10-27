// ============================================================================
// RENDER GPU test — renders through one backend into an off-screen
// target and checks the read-back pixels.
//
//   render-gpu-tests --backend vulkan|d3d12 [--warp] [--validation] [--load-dds <dir>]
//
// --warp runs D3D12 on the WARP software adapter (no GPU needed). --load-dds
// skips the suite and instead loads every .dds in <dir> (e.g. files pulled
// from the game's BSAs). Exit code = number of failed checks.
// ============================================================================

#include <render/Diligent.h>

#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <thread>
#include <vector>

using namespace render;
using namespace render::diligent;

namespace {

int g_failures = 0;

void check(const char* label, bool ok) {
    std::printf("[ %s ] %s\n", ok ? "PASS" : "FAIL", label);
    if (!ok) ++g_failures;
}

struct Rgba { int r, g, b, a; };

Rgba PixelAt(const std::vector<std::uint8_t>& px, int w, int x, int y) {
    const size_t i = (static_cast<size_t>(y) * w + x) * 4;
    return { px[i], px[i + 1], px[i + 2], px[i + 3] };
}

bool Near(int a, int b, int tol) { return std::abs(a - b) <= tol; }
int  Lum(Rgba p) { return p.r + p.g + p.b; }
bool Differs(Rgba a, Rgba b, int tol = 10) { return !Near(a.r, b.r, tol) || !Near(a.g, b.g, tol) || !Near(a.b, b.b, tol); }

// ── 1. Device + off-screen target round trip ─────────────────────────────────
void TestClearReadback(RenderDevice& dev) {
    std::printf("-- off-screen clear + readback --\n");
    auto target = OffscreenTarget::Create(dev, 64, 32);
    check("off-screen target created", target != nullptr);
    if (!target) return;
    target->Clear(0.2f, 0.4f, 0.6f, 1.f);
    const auto px = target->ReadbackRGBA8();
    check("readback size is w*h*4", px.size() == 64u * 32u * 4u);
    if (px.size() != 64u * 32u * 4u) return;
    bool all = true;
    for (int y = 0; y < 32 && all; y += 7)
        for (int x = 0; x < 64 && all; x += 9) {
            const Rgba p = PixelAt(px, 64, x, y);
            all = Near(p.r, 51, 1) && Near(p.g, 102, 1) && Near(p.b, 153, 1) && p.a == 255;
        }
    check("cleared colour reads back (sRGB round trip)", all);
}

// ── 2. Frame lifecycle: an empty frame is the background colour ───────────────
double SrgbEncode(double c) { return c <= 0.0031308 ? c * 12.92 : 1.055 * std::pow(c, 1.0 / 2.4) - 0.055; }
double Aces(double x) { return std::clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0); }
int    Expected(double linear) { return static_cast<int>(std::lround(SrgbEncode(Aces(linear)) * 255.0)); }

// The style's background as it reads back (display values, exact).
Rgba Background(const ViewStyle& s = {}) {
    auto b = [](float v) { return static_cast<int>(std::lround(v * 255.0)); };
    return { b(s.background[0]), b(s.background[1]), b(s.background[2]), 255 };
}

void TestEmptyFrame(RenderDevice& dev) {
    std::printf("-- empty frame --\n");
    auto target = OffscreenTarget::Create(dev, 64, 64);
    DiligentSceneRenderer r(dev);
    r.SetOutputTarget(*target);
    r.BeginFrame(64, 64);
    r.SetCamera(glm::mat4(1.f), glm::mat4(1.f));
    r.EndFrame();
    const auto px = target->ReadbackRGBA8();
    const Rgba p  = PixelAt(px, 64, 32, 32);
    const Rgba e = Background();
    const int  er = e.r, eg = e.g, eb = e.b;
    std::printf("   got (%d,%d,%d,%d), expected (%d,%d,%d,255)\n", p.r, p.g, p.b, p.a, er, eg, eb);
    check("empty frame = the background colour within 2/255", Near(p.r, er, 2) && Near(p.g, eg, 2) && Near(p.b, eb, 2) && p.a == 255);
}

// ── 3. Grid: axis colours and image orientation ───────────────────────────────
// A camera above the origin looking down -Z with +Y up: +X is to the right of the
// image, +Y towards the top.
glm::ivec2 Project(const glm::mat4& viewProj, glm::vec3 p, int w, int h) {
    const glm::vec4 c = viewProj * glm::vec4(p, 1.f);
    const glm::vec2 ndc(c.x / c.w, c.y / c.w);
    return { static_cast<int>((ndc.x * 0.5f + 0.5f) * w), static_cast<int>((0.5f - ndc.y * 0.5f) * h) };
}

// The most `dominant`-channel pixel in a 5×5 neighbourhood (lines are 1 px wide).
Rgba Strongest(const std::vector<std::uint8_t>& px, int w, int h, glm::ivec2 at, int channel) {
    Rgba best{ 0, 0, 0, 0 };
    int  bestScore = -1000;
    for (int dy = -2; dy <= 2; ++dy)
        for (int dx = -2; dx <= 2; ++dx) {
            const int x = at.x + dx, y = at.y + dy;
            if (x < 0 || y < 0 || x >= w || y >= h) continue;
            const Rgba p = PixelAt(px, w, x, y);
            const int  c[3] = { p.r, p.g, p.b };
            const int  score = 2 * c[channel] - c[(channel + 1) % 3] - c[(channel + 2) % 3];
            if (score > bestScore) { bestScore = score; best = p; }
        }
    return best;
}

void TestGridAxes(RenderDevice& dev) {
    std::printf("-- grid axes + orientation --\n");
    constexpr int W = 128, H = 128;
    auto target = OffscreenTarget::Create(dev, W, H);
    DiligentSceneRenderer r(dev);
    r.SetOutputTarget(*target);
    const glm::mat4 view = glm::lookAt(glm::vec3(0.f, 0.f, 10.f), glm::vec3(0.f), glm::vec3(0.f, 1.f, 0.f));
    const glm::mat4 proj = glm::perspective(glm::radians(60.f), 1.f, 0.1f, 100.f);
    r.BeginFrame(W, H);
    r.SetCamera(view, proj);
    r.DrawGrid(1.f, 40);
    r.EndFrame();
    const auto px = target->ReadbackRGBA8();

    const glm::ivec2 xAt = Project(proj * view, { 3.5f, 0.f, 0.f }, W, H);
    const glm::ivec2 yAt = Project(proj * view, { 0.f, 3.5f, 0.f }, W, H);
    const Rgba xp = Strongest(px, W, H, xAt, 0);
    const Rgba yp = Strongest(px, W, H, yAt, 1);
    std::printf("   +X at (%d,%d) = (%d,%d,%d)   +Y at (%d,%d) = (%d,%d,%d)\n", xAt.x, xAt.y, xp.r, xp.g, xp.b, yAt.x,
                yAt.y, yp.r, yp.g, yp.b);
    check("+X axis is red", xp.r > xp.g + 60 && xp.r > xp.b + 60);
    check("+Y axis is green", yp.g > yp.r + 60 && yp.g > yp.b + 60);
    check("+X lands right of centre, +Y above it", xAt.x > W / 2 && yAt.y < H / 2);

    // A grid-cell interior stays background.
    const glm::ivec2 cell = Project(proj * view, { 2.5f, 2.5f, 0.f }, W, H);
    const Rgba       bg   = PixelAt(px, W, cell.x, cell.y);
    check("cell interior is background", !Differs(bg, Background(), 2));

    // A custom background replaces the default.
    ViewStyle style;
    style.background = { 0.5f, 0.1f, 0.1f };
    r.SetViewStyle(style);
    r.BeginFrame(W, H);
    r.SetCamera(view, proj);
    r.EndFrame();
    const auto px2 = target->ReadbackRGBA8();
    check("SetViewStyle sets the background", !Differs(PixelAt(px2, W, 5, 5), Background(style), 2));
}


// ── 4. Resources: meshes, textures, handle hygiene ────────────────────────────
// A 4×4 uncompressed BGRA8 DDS (legacy header, no DX10 extension).
std::vector<std::uint8_t> MakeDds4x4(std::uint8_t b, std::uint8_t g, std::uint8_t r, std::uint8_t a) {
    std::vector<std::uint32_t> h(32, 0);   // 'DDS ' + 124-byte header
    h[0]  = 0x20534444;                    // 'DDS '
    h[1]  = 124;                           // dwSize
    h[2]  = 0x1 | 0x2 | 0x4 | 0x1000;      // CAPS | HEIGHT | WIDTH | PIXELFORMAT
    h[3]  = 4;                             // height
    h[4]  = 4;                             // width
    h[5]  = 16;                            // pitch
    h[19] = 32;                            // ddspf.dwSize
    h[20] = 0x40 | 0x1;                    // DDPF_RGB | DDPF_ALPHAPIXELS
    h[22] = 32;                            // bit count
    h[23] = 0x00FF0000;                    // R mask
    h[24] = 0x0000FF00;                    // G mask
    h[25] = 0x000000FF;                    // B mask
    h[26] = 0xFF000000;                    // A mask
    h[27] = 0x1000;                        // DDSCAPS_TEXTURE
    std::vector<std::uint8_t> bytes(h.size() * 4);
    std::memcpy(bytes.data(), h.data(), bytes.size());
    for (int i = 0; i < 16; ++i) bytes.insert(bytes.end(), { b, g, r, a });
    return bytes;
}

void TestResources(RenderDevice& dev) {
    std::printf("-- resources --\n");
    DiligentSceneRenderer r(dev);

    const auto dds = MakeDds4x4(10, 20, 30, 255);
    const TextureHandle srgb   = r.LoadTextureFromMemory(dds, TexColorSpace::Srgb);
    const TextureHandle linear = r.LoadTextureFromMemory(dds, TexColorSpace::Linear);
    check("DDS loads as sRGB", srgb != TextureHandle::Invalid);
    check("DDS loads as linear", linear != TextureHandle::Invalid);
    check("handles are distinct", srgb != linear);

    const std::vector<std::uint8_t> garbage(300, 0xAB);
    check("garbage blob -> Invalid", r.LoadTextureFromMemory(garbage, TexColorSpace::Srgb) == TextureHandle::Invalid);
    check("missing file -> Invalid", r.LoadTexture("Z:/no/such/file.dds", TexColorSpace::Linear) == TextureHandle::Invalid);

    std::vector<float> blend(33 * 33, 0.5f);
    const TextureHandle bm = r.UploadBlendMap(blend.data());
    check("blend map uploads", bm != TextureHandle::Invalid);

    std::vector<std::uint8_t> px;
    int w = 0, h = 0;
    check("DebugReadback refuses a loaded texture", !r.DebugReadbackRGBA8(srgb, px, w, h));

    MeshData quad;
    quad.positions = { { -1, -1, 0 }, { 1, -1, 0 }, { 1, 1, 0 }, { -1, 1, 0 } };
    quad.normals   = std::vector<glm::vec3>(4, { 0, 0, 1 });
    quad.uvs       = { { 0, 1 }, { 1, 1 }, { 1, 0 }, { 0, 0 } };
    quad.indices   = { 0, 1, 2, 0, 2, 3 };
    const MeshHandle m = r.UploadMesh(quad);
    check("mesh uploads", m != MeshHandle::Invalid);
    check("empty mesh -> Invalid", r.UploadMesh(MeshData{}) == MeshHandle::Invalid);

    MeshData skinned = quad;
    skinned.boneIndices  = std::vector<glm::u8vec4>(4, { 0, 1, 0, 0 });
    skinned.boneWeights  = std::vector<glm::vec4>(4, { 0.5f, 0.5f, 0, 0 });
    skinned.vertexColors = std::vector<glm::u8vec3>(4, { 255, 128, 0 });
    const MeshHandle sm = r.UploadMesh(skinned);
    check("skinned + coloured mesh uploads", sm != MeshHandle::Invalid);

    const std::vector<glm::vec3> moved(4, glm::vec3(0.f));
    r.UpdateMeshPositions(m, moved);
    r.UpdateMeshPositions(m, std::span<const glm::vec3>(moved.data(), 2));   // wrong length: ignored

    r.BeginFrame(8, 8);
    r.EndFrame();
    FrameStats st = r.LastFrameStats();
    check("stats count live meshes / textures", st.meshes == 2 && st.textures == 3);
    check("stats count failed loads", st.failedTextureLoads == 2);

    r.FreeTexture(srgb);
    r.FreeTexture(srgb);                   // double free: no-op
    r.FreeTexture(TextureHandle::Invalid);
    r.FreeMesh(m);
    r.FreeMesh(m);
    r.FreeMesh(MeshHandle::Invalid);
    r.DrawMesh(m, glm::mat4(1.f));         // freed handle, outside a frame: no-op

    const TextureHandle again = r.LoadTextureFromMemory(dds, TexColorSpace::Srgb);
    check("handles are never reused", again != srgb && again != linear && again != bm);

    r.BeginFrame(8, 8);
    r.DrawMesh(m, glm::mat4(1.f));         // freed handle inside a frame: no-op
    r.EndFrame();
    st = r.LastFrameStats();
    check("frees are reflected in stats", st.meshes == 1 && st.textures == 3);
}


// ── 5. PBR meshes ─────────────────────────────────────────────────────────────
// A camera on +Z looking at the origin; a 2×2 quad in the XY plane facing it.
struct PbrScene {
    static constexpr int W = 96, H = 96;
    RenderDevice&                    dev;
    std::unique_ptr<OffscreenTarget> target;
    DiligentSceneRenderer            r;
    glm::mat4 view = glm::lookAt(glm::vec3(0.f, 0.f, 3.f), glm::vec3(0.f), glm::vec3(0.f, 1.f, 0.f));
    glm::mat4 proj = glm::perspective(glm::radians(60.f), 1.f, 0.1f, 100.f);
    glm::vec3 lightDir{ 0.f, 0.f, 1.f };
    int       frames = 0;

    explicit PbrScene(RenderDevice& d) : dev(d), target(OffscreenTarget::Create(d, W, H)), r(d) { r.SetOutputTarget(*target); }

    // Renders until no draw was skipped for a compiling pipeline (or ~20 s).
    template <typename Fn>
    std::vector<std::uint8_t> Render(Fn&& draw) {
        for (frames = 1; frames <= 2000; ++frames) {
            r.BeginFrame(W, H);
            r.SetCamera(view, proj);
            r.SetLight(lightDir, glm::vec3(1.f), glm::vec3(0.3f));
            draw(r);
            r.EndFrame();
            dev.FinishFrame();
            if (r.LastFrameStats().skippedCompiling == 0) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        return target->ReadbackRGBA8();
    }
    Rgba At(const std::vector<std::uint8_t>& px, glm::vec3 world) const {
        const glm::ivec2 p = Project(proj * view, world, W, H);
        return PixelAt(px, W, p.x, p.y);
    }
};

MeshData MakeQuad(float z = 0.f, bool reversed = false) {
    MeshData q;
    q.positions = { { -1, -1, z }, { 1, -1, z }, { 1, 1, z }, { -1, 1, z } };
    q.normals   = std::vector<glm::vec3>(4, { 0, 0, 1 });
    q.uvs       = { { 0, 1 }, { 1, 1 }, { 1, 0 }, { 0, 0 } };
    q.indices   = reversed ? std::vector<std::uint16_t>{ 0, 2, 1, 0, 3, 2 } : std::vector<std::uint16_t>{ 0, 1, 2, 0, 2, 3 };
    return q;
}


void TestPbr(RenderDevice& dev) {
    std::printf("-- PBR meshes --\n");
    PbrScene s(dev);
    const Rgba bg = Background();
    const glm::vec3 centre(0.f), right(1.2f, 0.f, 0.f);

    const MeshHandle quad = s.r.UploadMesh(MakeQuad());
    check("quad uploads", quad != MeshHandle::Invalid);

    // Lit from the front vs from behind: darker, but IBL ambient keeps it off black.
    auto px = s.Render([&](auto& r) { r.DrawMesh(quad, glm::mat4(1.f)); });
    const FrameStats st = s.r.LastFrameStats();
    std::printf("   pipelines ready after %d frame(s); draws=%d skipped=%d\n", s.frames, st.draws, st.skippedCompiling);
    check("pipelines finish compiling (skippedCompiling reaches 0)", st.skippedCompiling == 0 && st.draws == 1);
    const Rgba lit = s.At(px, centre);
    s.lightDir     = { 0.f, 0.f, -1.f };
    px             = s.Render([&](auto& r) { r.DrawMesh(quad, glm::mat4(1.f)); });
    const Rgba back = s.At(px, centre);
    s.lightDir      = { 0.f, 0.f, 1.f };
    std::printf("   lit (%d,%d,%d)  back-lit (%d,%d,%d)  bg (%d,%d,%d)\n", lit.r, lit.g, lit.b, back.r, back.g, back.b, bg.r, bg.g, bg.b);
    check("front-lit quad is drawn", Differs(lit, bg));
    check("back-lit quad is darker than front-lit", Lum(back) + 30 < Lum(lit));
    check("back-lit quad is not black (IBL ambient)", Lum(back) > 30);
    check("quad does not cover the image edge", !Differs(PixelAt(px, PbrScene::W, 2, 2), bg));

    // Culling: a clockwise (back-facing) quad is culled unless twoSided.
    const MeshHandle rev = s.r.UploadMesh(MakeQuad(0.f, true));
    px = s.Render([&](auto& r) { r.DrawMesh(rev, glm::mat4(1.f)); });
    check("back face is culled", !Differs(s.At(px, centre), bg));
    DrawSurface twoSided;
    twoSided.twoSided = true;
    px = s.Render([&](auto& r) { r.DrawMesh(rev, glm::mat4(1.f), twoSided); });
    check("twoSided draws the back face", Differs(s.At(px, centre), bg));

    // Depth: the nearer (red) quad wins even when drawn first.
    const MeshHandle nearQ = s.r.UploadMesh(MakeQuad(0.5f));
    DrawSurface red, green;
    red.tint   = { 1.f, 0.05f, 0.05f, 1.f };
    green.tint = { 0.05f, 1.f, 0.05f, 1.f };
    px = s.Render([&](auto& r) {
        r.DrawMesh(nearQ, glm::mat4(1.f), red);
        r.DrawMesh(quad, glm::mat4(1.f), green);
    });
    const Rgba depth = s.At(px, centre);
    check("depth test keeps the nearer surface", depth.r > depth.g + 30);

    // The grid (Z = 0) hides behind geometry above the plane, but draws over geometry below it.
    const MeshHandle belowQ = s.r.UploadMesh(MakeQuad(-0.5f));
    const glm::vec3  onAxis(0.5f, 0.f, 0.f);
    px = s.Render([&](auto& r) {
        r.DrawMesh(nearQ, glm::mat4(1.f), green);
        r.DrawGrid(0.25f, 40);
    });
    const Rgba covered = Strongest(px, PbrScene::W, PbrScene::H, Project(s.proj * s.view, onAxis, PbrScene::W, PbrScene::H), 0);
    check("grid is hidden behind geometry above the plane", covered.g > covered.r + 30);
    px = s.Render([&](auto& r) {
        r.DrawMesh(belowQ, glm::mat4(1.f));
        r.DrawGrid(0.25f, 40);
    });
    const Rgba over = Strongest(px, PbrScene::W, PbrScene::H, Project(s.proj * s.view, onAxis, PbrScene::W, PbrScene::H), 0);
    std::printf("   grid over a mesh below the plane: (%d,%d,%d)\n", over.r, over.g, over.b);
    check("grid draws over geometry below the plane", over.r > over.g + 40 && over.r > over.b + 40);

    // Alpha blend: half-transparent white lands between background and opaque.
    DrawSurface half;
    half.tint      = { 1.f, 1.f, 1.f, 0.5f };
    half.blendMode = DrawSurface::BlendMode::AlphaBlend;
    px = s.Render([&](auto& r) { r.DrawMesh(quad, glm::mat4(1.f), half); });
    const Rgba blended = s.At(px, centre);
    std::printf("   blended (%d,%d,%d)\n", blended.r, blended.g, blended.b);
    check("alpha blend is between background and opaque", Lum(blended) > Lum(bg) + 15 && Lum(blended) + 15 < Lum(lit));

    // Skinning: a translated palette moves the coverage.
    MeshData skinnedData    = MakeQuad();
    skinnedData.boneIndices = std::vector<glm::u8vec4>(4, { 0, 0, 0, 0 });
    skinnedData.boneWeights = std::vector<glm::vec4>(4, { 1, 0, 0, 0 });
    const MeshHandle skinned = s.r.UploadMesh(skinnedData);
    const std::vector<glm::mat4> identity{ glm::mat4(1.f) };
    const std::vector<glm::mat4> moved{ glm::translate(glm::mat4(1.f), glm::vec3(1.5f, 0.f, 0.f)) };
    px = s.Render([&](auto& r) { r.DrawSkinnedMesh(skinned, glm::mat4(1.f), identity); });
    const bool skinCentre = Differs(s.At(px, centre), bg);
    px = s.Render([&](auto& r) { r.DrawSkinnedMesh(skinned, glm::mat4(1.f), moved); });
    check("skinned quad covers the centre with an identity palette", skinCentre);
    check("a translated palette moves the coverage", !Differs(s.At(px, centre), bg) && Differs(s.At(px, right), bg));

    // UpdateMeshPositions moves the coverage the same way.
    const MeshHandle morph = s.r.UploadMesh(MakeQuad());
    std::vector<glm::vec3> shifted = MakeQuad().positions;
    for (auto& p : shifted) p.x += 1.5f;
    s.r.UpdateMeshPositions(morph, shifted);
    px = s.Render([&](auto& r) { r.DrawMesh(morph, glm::mat4(1.f)); });
    check("UpdateMeshPositions moves the coverage", !Differs(s.At(px, centre), bg) && Differs(s.At(px, right), bg));

    // Wireframe: the interior of a triangle (off the diagonal) stays background.
    if (dev.supportsWireframe()) {
        DrawSurface wire;
        wire.wireframe = true;
        px = s.Render([&](auto& r) { r.DrawMesh(quad, glm::mat4(1.f), wire); });
        check("wireframe leaves triangle interiors empty", !Differs(s.At(px, { 0.5f, -0.5f, 0.f }), bg));
    } else {
        std::printf("   (wireframe not supported on this device; skipped)\n");
    }
}


// ── 6. Normal maps ────────────────────────────────────────────────────────────
// The quad's U runs along +X, so a normal map tilted to +U (red) must face the
// surface towards +X: lit from +X it is brighter than lit from -X.
void TestNormalMap(RenderDevice& dev) {
    std::printf("-- normal map --\n");
    PbrScene s(dev);
    const MeshHandle quad = s.r.UploadMesh(MakeQuad());
    const glm::vec3  fromRight = glm::normalize(glm::vec3(0.8f, 0.f, 0.6f));
    const glm::vec3  fromLeft  = glm::normalize(glm::vec3(-0.8f, 0.f, 0.6f));
    const glm::vec3  fromUp    = glm::normalize(glm::vec3(0.f, 0.8f, 0.6f));
    const glm::vec3  fromDown  = glm::normalize(glm::vec3(0.f, -0.8f, 0.6f));

    auto shade = [&](const DrawSurface& surf, glm::vec3 light) {
        s.lightDir = light;
        const auto px = s.Render([&](auto& r) { r.DrawMesh(quad, glm::mat4(1.f), surf); });
        return Lum(s.At(px, glm::vec3(0.f)));
    };

    DrawSurface flat;
    const int flatR = shade(flat, fromRight), flatL = shade(flat, fromLeft);
    check("without a normal map, left/right light is symmetric", Near(flatR, flatL, 6));

    // Tangent-space (0.8, 0, 0.6) → RGB (230, 128, 204); BGRA byte order.
    DrawSurface tiltU;
    tiltU.normalMap = s.r.LoadTextureFromMemory(MakeDds4x4(204, 128, 230, 255), TexColorSpace::Linear);
    const int uR = shade(tiltU, fromRight), uL = shade(tiltU, fromLeft);
    std::printf("   +U tilt: lit from +X %d, from -X %d\n", uR, uL);
    check("a +U (red) normal tilt faces the surface towards +X", uR > uL + 30);

    // Green: report which way +V tilts (DirectX vs OpenGL normal-map convention).
    DrawSurface tiltV;
    tiltV.normalMap = s.r.LoadTextureFromMemory(MakeDds4x4(204, 230, 128, 255), TexColorSpace::Linear);
    const int vUp = shade(tiltV, fromUp), vDown = shade(tiltV, fromDown);
    std::printf("   +G tilt: lit from +Y %d, from -Y %d -> green points %s (texture V runs -Y here)\n", vUp, vDown,
                vUp > vDown ? "+Y (up the image: OpenGL-style)" : "-Y (down the image: DirectX-style)");
    check("a green normal tilt changes the shading", !Near(vUp, vDown, 30));
}


// ── 7. Face tint ──────────────────────────────────────────────────────────────
double SrgbDecode(int c) {
    const double v = c / 255.0;
    return v <= 0.04045 ? v / 12.92 : std::pow((v + 0.055) / 1.055, 2.4);
}
int SrgbByte(double linear) { return static_cast<int>(std::lround(SrgbEncode(std::clamp(linear, 0.0, 1.0)) * 255.0)); }

void TestFaceTint(RenderDevice& dev) {
    std::printf("-- face tint --\n");
    DiligentSceneRenderer r(dev);
    // Base diffuse (sRGB) RGB (200, 100, 50); BGRA byte order.
    const TextureHandle base = r.LoadTextureFromMemory(MakeDds4x4(50, 100, 200, 255), TexColorSpace::Srgb);
    check("tint base loads", base != TextureHandle::Invalid);

    auto composite = [&](std::span<const FaceTintLayer> layers, Rgba& out) {
        const TextureHandle t = r.CompositeFaceTint(base, layers);
        std::vector<std::uint8_t> px;
        int w = 0, h = 0;
        const bool ok = t != TextureHandle::Invalid && r.DebugReadbackRGBA8(t, px, w, h) && w == 4 && h == 4;
        out = ok ? PixelAt(px, w, 1, 1) : Rgba{ -1, -1, -1, -1 };
        r.FreeTexture(t);
        return ok;
    };

    check("Invalid base -> Invalid", r.CompositeFaceTint(TextureHandle::Invalid, {}) == TextureHandle::Invalid);
    FaceTintLayer neutral;
    neutral.blend = FaceTintLayer::Blend::Multiply;
    neutral.color = { 0.5f, 0.5f, 0.5f, 1.f };
    check("no layers -> Invalid", r.CompositeFaceTint(base, {}) == TextureHandle::Invalid);

    Rgba p{};
    check("multiply by grey 0.5 composites", composite({ &neutral, 1 }, p));
    std::printf("   multiply 0.5 -> (%d,%d,%d)\n", p.r, p.g, p.b);
    check("multiply-2x by grey 0.5 is the identity", Near(p.r, 200, 2) && Near(p.g, 100, 2) && Near(p.b, 50, 2));

    FaceTintLayer red;
    red.color = { 1.f, 0.f, 0.f, 1.f };
    composite({ &red, 1 }, p);
    check("overlay red at full strength -> (255,0,0)", Near(p.r, 255, 1) && p.g <= 1 && p.b <= 1);

    // Alpha-0 mask (coverage from alpha): no change.
    FaceTintLayer masked = red;
    masked.mask = r.LoadTextureFromMemory(MakeDds4x4(255, 255, 255, 0), TexColorSpace::Linear);
    composite({ &masked, 1 }, p);
    check("an alpha-0 mask is a no-op", Near(p.r, 200, 2) && Near(p.g, 100, 2) && Near(p.b, 50, 2));

    // Luminance mask of ~0.5 (linear byte 128): half coverage, mixed in linear space.
    FaceTintLayer lum = red;
    lum.mask              = r.LoadTextureFromMemory(MakeDds4x4(128, 128, 128, 255), TexColorSpace::Linear);
    lum.coverageFromAlpha = false;
    composite({ &lum, 1 }, p);
    const double k = 128.0 / 255.0;
    const int er = SrgbByte(SrgbDecode(200) * (1 - k) + k), eg = SrgbByte(SrgbDecode(100) * (1 - k)), eb = SrgbByte(SrgbDecode(50) * (1 - k));
    std::printf("   luminance mask -> (%d,%d,%d), expected (%d,%d,%d)\n", p.r, p.g, p.b, er, eg, eb);
    check("a luminance mask gives half coverage", Near(p.r, er, 3) && Near(p.g, eg, 3) && Near(p.b, eb, 3));

    // A composite in the middle of a frame leaves that frame intact.
    constexpr int W = 128, H = 128;
    auto target = OffscreenTarget::Create(dev, W, H);
    r.SetOutputTarget(*target);
    const glm::mat4 view = glm::lookAt(glm::vec3(0.f, 0.f, 10.f), glm::vec3(0.f), glm::vec3(0.f, 1.f, 0.f));
    const glm::mat4 proj = glm::perspective(glm::radians(60.f), 1.f, 0.1f, 100.f);
    r.BeginFrame(W, H);
    r.SetCamera(view, proj);
    r.DrawGrid(1.f, 40);
    const TextureHandle mid = r.CompositeFaceTint(base, { &red, 1 });
    r.EndFrame();
    const auto px = target->ReadbackRGBA8();
    const Rgba xp = Strongest(px, W, H, Project(proj * view, { 3.5f, 0.f, 0.f }, W, H), 0);
    const glm::ivec2 cell = Project(proj * view, { 2.5f, 2.5f, 0.f }, W, H);
    const Rgba bg = PixelAt(px, W, cell.x, cell.y);
    check("mid-frame composite succeeds", mid != TextureHandle::Invalid);
    check("mid-frame composite leaves the scene intact", xp.r > xp.g + 60 && !Differs(bg, Background(), 2));
}


// ── 7b. Skin tint: soft light + detail (Skyrim's FaceGen / SkinTint shading) ───
// The composite on display values (CPU references below), then the tinted
// diffuse rendered on a lit quad against the same quad drawn with a plain texture
// of the expected colour: the shaded result must match. Plus the external
// specular mask (vanilla skin's `_s.dds`).
double SoftLightRef(int base, int tint) {   // display bytes in, display value (0..1) out
    const double b = base / 255.0, t = tint / 255.0;
    return b * b + 2.0 * t * b * (1.0 - b);
}
int ToByte(double v) { return static_cast<int>(std::lround(std::clamp(v, 0.0, 1.0) * 255.0)); }

void TestSkinTint(RenderDevice& dev) {
    std::printf("-- skin tint (FaceGen / SkinTint) --\n");
    PbrScene s(dev);
    DiligentSceneRenderer& r = s.r;
    // Base: a vanilla head diffuse's mean skin colour, sRGB (76, 63, 57); BGRA byte order.
    const int B[3] = { 76, 63, 57 };
    const TextureHandle base = r.LoadTextureFromMemory(MakeDds4x4(57, 63, 76, 255), TexColorSpace::Srgb);
    check("skin base loads", base != TextureHandle::Invalid);

    auto composite = [&](std::span<const FaceTintLayer> layers, Rgba& out) {
        const TextureHandle t = r.CompositeFaceTint(base, layers);
        std::vector<std::uint8_t> px;
        int w = 0, h = 0;
        const bool ok = t != TextureHandle::Invalid && r.DebugReadbackRGBA8(t, px, w, h) && w == 4 && h == 4;
        out = ok ? PixelAt(px, w, 1, 1) : Rgba{ -1, -1, -1, -1 };
        return t;
    };
    auto expectSoft = [&](const int (&tint)[3]) {
        return Rgba{ ToByte(SoftLightRef(B[0], tint[0])), ToByte(SoftLightRef(B[1], tint[1])), ToByte(SoftLightRef(B[2], tint[2])), 255 };
    };
    auto near3 = [](Rgba a, Rgba b, int tol) { return Near(a.r, b.r, tol) && Near(a.g, b.g, tol) && Near(a.b, b.b, tol); };

    // SkinTint: a constant tone, full coverage. Pale Nord (221, 221, 221), dark Redguard (113, 87, 68).
    const int pale[3] = { 221, 221, 221 }, dark[3] = { 113, 87, 68 };
    FaceTintLayer paleL;
    paleL.blend = FaceTintLayer::Blend::SoftLight;
    paleL.color = { pale[0] / 255.f, pale[1] / 255.f, pale[2] / 255.f, 1.f };
    FaceTintLayer darkL = paleL;
    darkL.color = { dark[0] / 255.f, dark[1] / 255.f, dark[2] / 255.f, 1.f };

    Rgba p{};
    TextureHandle paleTex = composite({ &paleL, 1 }, p);
    const Rgba ePale = expectSoft(pale);
    std::printf("   soft light pale -> (%d,%d,%d), expected (%d,%d,%d)\n", p.r, p.g, p.b, ePale.r, ePale.g, ePale.b);
    check("SkinTint soft light (pale tone) matches b^2 + 2tb(1-b)", near3(p, ePale, 2));
    check("a pale tone lightens the skin", p.r > B[0] + 20 && p.g > B[1] + 20 && p.b > B[2] + 20);
    TextureHandle darkTex = composite({ &darkL, 1 }, p);
    const Rgba eDark = expectSoft(dark);
    std::printf("   soft light dark -> (%d,%d,%d), expected (%d,%d,%d)\n", p.r, p.g, p.b, eDark.r, eDark.g, eDark.b);
    check("SkinTint soft light (dark tone) matches", near3(p, eDark, 2));
    check("a dark tone darkens and warms the skin", p.r <= B[0] && p.b < B[2] - 10 && p.r > p.g && p.g > p.b);

    FaceTintLayer grey = paleL;
    grey.color = { 0.5f, 0.5f, 0.5f, 1.f };
    r.FreeTexture(composite({ &grey, 1 }, p));
    check("soft light with mid grey is the identity", near3(p, { B[0], B[1], B[2], 255 }, 1));

    FaceTintLayer half = paleL;
    half.color.a = 0.5f;
    r.FreeTexture(composite({ &half, 1 }, p));
    const Rgba eHalf{ ToByte((B[0] / 255.0 + SoftLightRef(B[0], pale[0])) / 2), ToByte((B[1] / 255.0 + SoftLightRef(B[1], pale[1])) / 2),
                      ToByte((B[2] / 255.0 + SoftLightRef(B[2], pale[2])) / 2), 255 };
    check("strength blends towards the soft-lit colour on display values", near3(p, eHalf, 2));

    // Linearity: soft light over a run of layers = soft light by the layers painted
    // over grey. Dark at full coverage, then pale through a half-coverage mask.
    FaceTintLayer paleMasked = paleL;
    paleMasked.mask              = r.LoadTextureFromMemory(MakeDds4x4(128, 128, 128, 255), TexColorSpace::Linear);
    paleMasked.coverageFromAlpha = false;
    const FaceTintLayer run[2] = { darkL, paleMasked };
    r.FreeTexture(composite(run, p));
    const double k = 128.0 / 255.0;
    const int painted[3] = { static_cast<int>(std::lround(dark[0] + (pale[0] - dark[0]) * k)),
                             static_cast<int>(std::lround(dark[1] + (pale[1] - dark[1]) * k)),
                             static_cast<int>(std::lround(dark[2] + (pale[2] - dark[2]) * k)) };
    const Rgba ePainted = expectSoft(painted);
    std::printf("   layered -> (%d,%d,%d), expected (%d,%d,%d)\n", p.r, p.g, p.b, ePainted.r, ePainted.g, ePainted.b);
    check("soft-light layers compose as one painted tint map", near3(p, ePainted, 3));

    // FaceGen: a facetint map (Hulda's mean (206, 196, 196)) then the detail map
    // (vanilla BlankDetailmap: 64 grey, neutral up to the (1, 0, 1)/255 offset).
    const int ft[3] = { 206, 196, 196 };
    FaceTintLayer faceTint;
    faceTint.blend   = FaceTintLayer::Blend::SoftLight;
    faceTint.tintMap = r.LoadTextureFromMemory(MakeDds4x4(196, 196, 206, 255), TexColorSpace::Srgb);
    r.FreeTexture(composite({ &faceTint, 1 }, p));
    const Rgba eFace = expectSoft(ft);
    check("FaceGen soft light by a tint map matches", near3(p, eFace, 2));

    FaceTintLayer detail;
    detail.blend   = FaceTintLayer::Blend::Detail;
    detail.tintMap = r.LoadTextureFromMemory(MakeDds4x4(64, 64, 64, 255), TexColorSpace::Srgb);
    r.FreeTexture(composite({ &detail, 1 }, p));
    check("a 64-grey detail map is (near) neutral", near3(p, { B[0], B[1], B[2], 255 }, 2));
    FaceTintLayer brightDetail = detail;
    brightDetail.tintMap = r.LoadTextureFromMemory(MakeDds4x4(96, 96, 96, 255), TexColorSpace::Srgb);
    r.FreeTexture(composite({ &brightDetail, 1 }, p));
    check("a brighter detail map scales the colour up (x1.5)", Near(p.g, ToByte(B[1] / 255.0 * 3.984375 * 96 / 255.0), 2));
    FaceTintLayer noMap = detail;
    noMap.tintMap = TextureHandle::Invalid;
    r.FreeTexture(composite({ &noMap, 1 }, p));
    check("a detail layer without a map is skipped", near3(p, { B[0], B[1], B[2], 255 }, 1));

    const FaceTintLayer faceGen[2] = { faceTint, detail };
    TextureHandle faceTex = composite(faceGen, p);
    const Rgba eFaceGen{ ToByte(SoftLightRef(B[0], ft[0]) * 3.984375 * 65 / 255.0), ToByte(SoftLightRef(B[1], ft[1]) * 3.984375 * 64 / 255.0),
                         ToByte(SoftLightRef(B[2], ft[2]) * 3.984375 * 65 / 255.0), 255 };
    std::printf("   FaceGen (tint map + detail) -> (%d,%d,%d), expected (%d,%d,%d)\n", p.r, p.g, p.b, eFaceGen.r, eFaceGen.g, eFaceGen.b);
    check("FaceGen tint map + detail matches", near3(p, eFaceGen, 2));

    // Rendered: each tinted diffuse on a lit quad shades like a plain texture of the
    // expected colour, and the shaded colours keep their order and hue.
    const MeshHandle quad = r.UploadMesh(MakeQuad());
    s.lightDir = glm::normalize(glm::vec3(0.3f, 0.4f, 1.f));
    auto shade = [&](TextureHandle diffuse) {
        DrawSurface surf;
        surf.diffuse          = diffuse;
        surf.specularStrength = 0.f;
        const auto px = s.Render([&](auto& rr) { rr.DrawMesh(quad, glm::mat4(1.f), surf); });
        return s.At(px, glm::vec3(0.f));
    };
    auto plain = [&](Rgba c) {
        return r.LoadTextureFromMemory(MakeDds4x4(static_cast<std::uint8_t>(c.b), static_cast<std::uint8_t>(c.g),
                                                  static_cast<std::uint8_t>(c.r), 255), TexColorSpace::Srgb);
    };
    struct Case { const char* name; TextureHandle tinted; Rgba expected; };
    const Case cases[] = { { "SkinTint pale", paleTex, ePale }, { "SkinTint dark", darkTex, eDark }, { "FaceGen", faceTex, eFaceGen } };
    Rgba shaded[3]{};
    for (int i = 0; i < 3; ++i) {
        const Rgba got = shade(cases[i].tinted), want = shade(plain(cases[i].expected));
        shaded[i] = got;
        std::printf("   %-13s rendered (%d,%d,%d), plain reference (%d,%d,%d)\n", cases[i].name, got.r, got.g, got.b, want.r, want.g, want.b);
        char label[96];
        std::snprintf(label, sizeof(label), "%s renders as its expected colour", cases[i].name);
        check(label, near3(got, want, 3));
        std::snprintf(label, sizeof(label), "%s renders as skin (r >= g >= b, not grey-purple)", cases[i].name);
        check(label, got.r >= got.g && got.g >= got.b && got.r > got.b + 4);
    }
    check("the pale tone renders lighter than the dark one", Lum(shaded[0]) > Lum(shaded[1]) + 60);

    // External specular mask: strength 3, a mask of 0 kills the highlight, 255 keeps it.
    if (dev.supportsSwizzle()) {
        s.lightDir = glm::vec3(0.f, 0.f, 1.f);   // head-on: the highlight sits at the centre
        auto lit = [&](TextureHandle mask) {
            DrawSurface surf;
            surf.diffuse          = plain({ 60, 50, 45, 255 });
            surf.specularMap      = mask;
            surf.specularStrength = 3.f;
            surf.roughness        = 0.3f;
            const auto px = s.Render([&](auto& rr) { rr.DrawMesh(quad, glm::mat4(1.f), surf); });
            return Lum(s.At(px, glm::vec3(0.f)));
        };
        const int none = lit(TextureHandle::Invalid);
        const int black = lit(r.LoadTextureFromMemory(MakeDds4x4(0, 0, 0, 255), TexColorSpace::Linear));
        const int white = lit(r.LoadTextureFromMemory(MakeDds4x4(255, 255, 255, 0), TexColorSpace::Linear));   // alpha 0: the mask is .r
        std::printf("   specular: no mask %d, black mask %d, white mask %d\n", none, black, white);
        check("a black specular mask removes the highlight", black < none - 30);
        check("a white specular mask (red channel, alpha 0) keeps it", Near(white, none, 6));
    } else {
        std::printf("   (texture swizzle not supported; specular-mask checks skipped)\n");
    }
}


// ── 8. Legacy DDS formats (decoded on the CPU) ────────────────────────────────
// Diligent's DDS loader rejects 24-bit RGB (vanilla's face-tint masks), and reads
// X8 / 16-bit packed / luminance / alpha layouts differently per backend; the
// renderer decodes those to RGBA8. Each case loads a 4×4 checker of two pixels
// as an sRGB base, composites it through an identity layer (multiply-2x by grey
// 0.5) and reads every texel back.
constexpr std::uint32_t kRGB = 0x40, kALPHAPIXELS = 0x1, kLUMINANCE = 0x20000, kALPHA = 0x2, kFOURCC = 0x4;

struct LegacyPf {
    std::uint32_t flags, bits, r, g, b, a;
};

std::vector<std::uint8_t> MakeLegacyDds(int w, int h, const LegacyPf& pf, const std::vector<std::uint8_t>& pixels,
                                        std::uint32_t fourCC = 0, std::uint32_t extraFlags = 0, std::uint32_t caps2 = 0) {
    std::vector<std::uint32_t> hd(32, 0);
    hd[0]  = 0x20534444;   // 'DDS '
    hd[1]  = 124;
    hd[2]  = 0x1 | 0x2 | 0x4 | 0x1000 | extraFlags;
    hd[3]  = static_cast<std::uint32_t>(h);
    hd[4]  = static_cast<std::uint32_t>(w);
    hd[19] = 32;
    hd[20] = pf.flags;
    hd[21] = fourCC;
    hd[22] = pf.bits;
    hd[23] = pf.r;
    hd[24] = pf.g;
    hd[25] = pf.b;
    hd[26] = pf.a;
    hd[27] = 0x1000;
    hd[28] = caps2;
    std::vector<std::uint8_t> bytes(hd.size() * 4);
    std::memcpy(bytes.data(), hd.data(), bytes.size());
    bytes.insert(bytes.end(), pixels.begin(), pixels.end());
    return bytes;
}

// A w×h checker: pixel (x, y) is `even` when x + y is even, else `odd`.
std::vector<std::uint8_t> Checker(int w, int h, const std::vector<std::uint8_t>& even, const std::vector<std::uint8_t>& odd) {
    std::vector<std::uint8_t> px;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) px.insert(px.end(), ((x + y) & 1) ? odd.begin() : even.begin(), ((x + y) & 1) ? odd.end() : even.end());
    return px;
}

// Reads `tex` back through an identity composite (sRGB-encoded bytes).
bool ReadThroughComposite(DiligentSceneRenderer& r, TextureHandle tex, std::vector<std::uint8_t>& px, int& w, int& h) {
    FaceTintLayer identity;
    identity.blend = FaceTintLayer::Blend::Multiply;
    identity.color = { 0.5f, 0.5f, 0.5f, 1.f };
    const TextureHandle out = r.CompositeFaceTint(tex, { &identity, 1 });
    const bool ok = out != TextureHandle::Invalid && r.DebugReadbackRGBA8(out, px, w, h);
    r.FreeTexture(out);
    return ok;
}

void TestLegacyDds(RenderDevice& dev) {
    std::printf("-- legacy DDS formats --\n");
    DiligentSceneRenderer r(dev);

    struct Case {
        const char*               name;
        LegacyPf                  pf;
        std::vector<std::uint8_t> even, odd;   // file bytes of the two checker pixels
        Rgba                      wantEven, wantOdd;
    };
    const Case cases[] = {
        { "B8G8R8 24-bit (vanilla tint masks)", { kRGB, 24, 0xff0000, 0xff00, 0xff, 0 }, { 30, 20, 10 }, { 200, 150, 250 },
          { 10, 20, 30, 255 }, { 250, 150, 200, 255 } },
        { "R8G8B8 24-bit", { kRGB, 24, 0xff, 0xff00, 0xff0000, 0 }, { 10, 20, 30 }, { 250, 150, 200 }, { 10, 20, 30, 255 },
          { 250, 150, 200, 255 } },
        { "X8R8G8B8 (pad byte 0 -> alpha 255)", { kRGB, 32, 0xff0000, 0xff00, 0xff, 0 }, { 30, 20, 10, 0 }, { 200, 150, 250, 0 },
          { 10, 20, 30, 255 }, { 250, 150, 200, 255 } },
        { "X8B8G8R8", { kRGB, 32, 0xff, 0xff00, 0xff0000, 0 }, { 10, 20, 30, 0 }, { 250, 150, 200, 0 }, { 10, 20, 30, 255 },
          { 250, 150, 200, 255 } },
        // 0xf81f = (31, 0, 31); 0x07e0 = (0, 63, 0)
        { "R5G6B5", { kRGB, 16, 0xf800, 0x7e0, 0x1f, 0 }, { 0x1f, 0xf8 }, { 0xe0, 0x07 }, { 255, 0, 255, 255 }, { 0, 255, 0, 255 } },
        // 0xfc00 = a1 (31, 0, 0); 0x03ff = a0 (0, 31, 31)
        { "A1R5G5B5", { kRGB | kALPHAPIXELS, 16, 0x7c00, 0x3e0, 0x1f, 0x8000 }, { 0x00, 0xfc }, { 0xff, 0x03 }, { 255, 0, 0, 255 },
          { 0, 255, 255, 0 } },
        // 0x8f40 = a8 (15, 4, 0) -> (255, 68, 0, 136); 0x30ff = a3 (0, 15, 15) -> (0, 255, 255, 51)
        { "A4R4G4B4", { kRGB | kALPHAPIXELS, 16, 0xf00, 0xf0, 0xf, 0xf000 }, { 0x40, 0x8f }, { 0xff, 0x30 }, { 255, 68, 0, 136 },
          { 0, 255, 255, 51 } },
        { "L8", { kLUMINANCE, 8, 0xff, 0, 0, 0 }, { 40 }, { 220 }, { 40, 40, 40, 255 }, { 220, 220, 220, 255 } },
        { "A8L8", { kLUMINANCE | kALPHAPIXELS, 16, 0xff, 0, 0, 0xff00 }, { 40, 90 }, { 220, 10 }, { 40, 40, 40, 90 },
          { 220, 220, 220, 10 } },
        { "A8", { kALPHA, 8, 0, 0, 0, 0xff }, { 90 }, { 250 }, { 0, 0, 0, 90 }, { 0, 0, 0, 250 } },
    };
    auto near = [](Rgba a, Rgba b) { return Near(a.r, b.r, 2) && Near(a.g, b.g, 2) && Near(a.b, b.b, 2) && Near(a.a, b.a, 1); };
    for (const Case& c : cases) {
        const TextureHandle tex = r.LoadTextureFromMemory(MakeLegacyDds(4, 4, c.pf, Checker(4, 4, c.even, c.odd)), TexColorSpace::Srgb);
        std::vector<std::uint8_t> px;
        int  w = 0, h = 0;
        bool ok = tex != TextureHandle::Invalid && ReadThroughComposite(r, tex, px, w, h) && w == 4 && h == 4;
        Rgba bad{ -1, -1, -1, -1 };
        for (int y = 0; ok && y < 4; ++y)
            for (int x = 0; ok && x < 4; ++x) {
                const Rgba got = PixelAt(px, w, x, y);
                ok = near(got, ((x + y) & 1) ? c.wantOdd : c.wantEven);
                if (!ok) bad = got;
            }
        if (!ok && bad.r >= 0) std::printf("   %s: got (%d,%d,%d,%d)\n", c.name, bad.r, bad.g, bad.b, bad.a);
        check((std::string(c.name) + " loads and reads back per texel").c_str(), ok);
        r.FreeTexture(tex);
    }

    // An odd size (no row padding in the file, a 5x3 -> 2x1 -> 1x1 chain) and the
    // linear colour space: a Linear base reads back sRGB-encoded.
    {
        std::vector<std::uint8_t> px5;
        for (int i = 0; i < 15; ++i) px5.insert(px5.end(), { 0, 0, static_cast<std::uint8_t>(i * 17) });
        const TextureHandle tex = r.LoadTextureFromMemory(MakeLegacyDds(5, 3, { kRGB, 24, 0xff0000, 0xff00, 0xff, 0 }, px5), TexColorSpace::Linear);
        std::vector<std::uint8_t> px;
        int  w = 0, h = 0;
        bool ok = tex != TextureHandle::Invalid && ReadThroughComposite(r, tex, px, w, h) && w == 5 && h == 3;
        for (int i = 0; ok && i < 15; ++i) ok = Near(PixelAt(px, w, i % 5, i / 5).r, SrgbByte(i * 17 / 255.0), 2);
        check("a 5x3 24-bit texture in linear space reads back per texel", ok);
        r.FreeTexture(tex);
    }

    // As masks (linear): the composite reads a decoded luminance as luminance and
    // a decoded alpha as alpha.
    const TextureHandle base = r.LoadTextureFromMemory(MakeDds4x4(50, 100, 200, 255), TexColorSpace::Srgb);
    auto overlay = [&](TextureHandle mask, bool fromAlpha, Rgba& out) {
        FaceTintLayer red;
        red.color             = { 1.f, 0.f, 0.f, 1.f };
        red.mask              = mask;
        red.coverageFromAlpha = fromAlpha;
        const TextureHandle t = r.CompositeFaceTint(base, { &red, 1 });
        std::vector<std::uint8_t> px;
        int w = 0, h = 0;
        const bool ok = t != TextureHandle::Invalid && r.DebugReadbackRGBA8(t, px, w, h);
        out = ok ? PixelAt(px, w, 1, 1) : Rgba{ -1, -1, -1, -1 };
        r.FreeTexture(t);
        r.FreeTexture(mask);
        return ok;
    };
    const double k = 128.0 / 255.0;
    const Rgba   half{ SrgbByte(SrgbDecode(200) * (1 - k) + k), SrgbByte(SrgbDecode(100) * (1 - k)), SrgbByte(SrgbDecode(50) * (1 - k)), 255 };
    Rgba p{};
    const std::vector<std::uint8_t> grey24 = Checker(64, 64, { 128, 128, 128 }, { 128, 128, 128 });
    overlay(r.LoadTextureFromMemory(MakeLegacyDds(64, 64, { kRGB, 24, 0xff0000, 0xff00, 0xff, 0 }, grey24), TexColorSpace::Linear), false, p);
    std::printf("   24-bit grey mask -> (%d,%d,%d), expected (%d,%d,%d)\n", p.r, p.g, p.b, half.r, half.g, half.b);
    check("a 24-bit luminance mask gives half coverage", near({ p.r, p.g, p.b, 255 }, half));
    overlay(r.LoadTextureFromMemory(MakeLegacyDds(4, 4, { kLUMINANCE, 8, 0xff, 0, 0, 0 }, Checker(4, 4, { 128 }, { 128 })), TexColorSpace::Linear), false, p);
    check("an L8 mask reads as luminance (half coverage), not red only", near({ p.r, p.g, p.b, 255 }, half));
    overlay(r.LoadTextureFromMemory(MakeLegacyDds(4, 4, { kALPHA, 8, 0, 0, 0, 0xff }, Checker(4, 4, { 0 }, { 0 })), TexColorSpace::Linear), true, p);
    check("an A8 mask of alpha 0 is a no-op", Near(p.r, 200, 2) && Near(p.g, 100, 2) && Near(p.b, 50, 2));
    overlay(r.LoadTextureFromMemory(MakeLegacyDds(4, 4, { kALPHA, 8, 0, 0, 0, 0xff }, Checker(4, 4, { 255 }, { 255 })), TexColorSpace::Linear), true, p);
    check("an A8 mask of alpha 255 is full coverage", Near(p.r, 255, 1) && p.g <= 1 && p.b <= 1);

    // Unsupported or corrupt files fail cleanly (one log line each) and are counted.
    const int before = [&] { r.BeginFrame(8, 8); r.EndFrame(); return r.LastFrameStats().failedTextureLoads; }();
    std::printf("   (expect four 'texture load failed' lines)\n");
    check("an unknown FourCC -> Invalid",
          r.LoadTextureFromMemory(MakeLegacyDds(4, 4, { kFOURCC, 0, 0, 0, 0, 0 }, std::vector<std::uint8_t>(16), 0x44434241), TexColorSpace::Srgb) ==
              TextureHandle::Invalid);
    check("truncated 24-bit pixel data -> Invalid",
          r.LoadTextureFromMemory(MakeLegacyDds(4, 4, { kRGB, 24, 0xff0000, 0xff00, 0xff, 0 }, std::vector<std::uint8_t>(40)), TexColorSpace::Srgb) ==
              TextureHandle::Invalid);
    check("a volume texture -> Invalid",
          r.LoadTextureFromMemory(MakeLegacyDds(4, 4, { kLUMINANCE, 8, 0xff, 0, 0, 0 }, std::vector<std::uint8_t>(64), 0, 0x800000, 0x200000),
                                  TexColorSpace::Linear) == TextureHandle::Invalid);
    std::vector<std::uint8_t> shortBc = MakeLegacyDds(64, 64, { kFOURCC, 0, 0, 0, 0, 0 }, std::vector<std::uint8_t>(64), 0x31545844);   // DXT1
    check("a DXT1 with missing blocks -> Invalid (Diligent's error folded into one line)",
          r.LoadTextureFromMemory(shortBc, TexColorSpace::Srgb) == TextureHandle::Invalid);
    r.BeginFrame(8, 8);
    r.EndFrame();
    check("the failures are counted", r.LastFrameStats().failedTextureLoads == before + 4);
}

// ── 9. Overlays ───────────────────────────────────────────────────────────────
// Display-space colours land exactly (they are drawn after the tone map).
bool Exact(Rgba p, Rgba8 c, int tol = 1) { return Near(p.r, c.r, tol) && Near(p.g, c.g, tol) && Near(p.b, c.b, tol); }

// Rows of column x (in [y0, y1)) that hold exactly colour c.
int ExactInColumn(const std::vector<std::uint8_t>& px, int w, int x, int y0, int y1, Rgba8 c) {
    int n = 0;
    for (int y = y0; y < y1; ++y) n += Exact(PixelAt(px, w, x, y), c) ? 1 : 0;
    return n;
}
int ExactInRow(const std::vector<std::uint8_t>& px, int w, int y, Rgba8 c) {
    int n = 0;
    for (int x = 0; x < w; ++x) n += Exact(PixelAt(px, w, x, y), c) ? 1 : 0;
    return n;
}

std::array<OverlayVertex, 2> Seg(glm::vec3 a, glm::vec3 b, Rgba8 c) { return { OverlayVertex{ a, c }, OverlayVertex{ b, c } }; }
std::array<OverlayVertex, 3> Tri(glm::vec3 centre, float r, Rgba8 c) {
    return { OverlayVertex{ centre + glm::vec3(-r, -r, 0.f), c }, OverlayVertex{ centre + glm::vec3(r, -r, 0.f), c },
             OverlayVertex{ centre + glm::vec3(0.f, r, 0.f), c } };
}

void TestOverlayScreen(RenderDevice& dev) {
    std::printf("-- overlays: screen layer --\n");
    constexpr int W = 96, H = 96;
    auto target = OffscreenTarget::Create(dev, W, H);
    DiligentSceneRenderer r(dev);
    r.SetOutputTarget(*target);
    const Rgba  bg = Background();
    const Rgba8 red{ 200, 60, 30, 255 }, blue{ 30, 90, 220, 255 }, green{ 40, 200, 120, 255 };

    check("Overlays() is the renderer itself", r.Overlays() == static_cast<IOverlayRenderer*>(&r));
    r.BeginFrame(W, H);
    r.SetCamera(glm::mat4(1.f), glm::mat4(1.f));
    const ScreenVertex quad[] = { { { 4, 4 }, red }, { { 12, 4 }, red }, { { 12, 12 }, red },
                                  { { 4, 4 }, red }, { { 12, 12 }, red }, { { 4, 12 }, red } };
    r.DrawScreenTriangles(quad);
    const ScreenVertex wide[] = { { { 10, 40 }, blue }, { { 80, 40 }, blue } };   // centred on a pixel boundary
    r.DrawScreenLines(wide, 4.f);
    const ScreenVertex thin[] = { { { 10, 60.5f }, green }, { { 80, 60.5f }, green } };   // centred on row 60
    r.DrawScreenLines(thin, 1.f);
    r.EndFrame();
    const auto px = target->ReadbackRGBA8();

    const Rgba p = PixelAt(px, W, 5, 5);
    std::printf("   triangle pixel (%d,%d,%d), expected (200,60,30)\n", p.r, p.g, p.b);
    check("screen triangles: exact colour at (5,5) (origin top-left)", Exact(p, red) && Exact(PixelAt(px, W, 11, 11), red));
    check("screen triangles: nothing outside their rect", !Differs(PixelAt(px, W, 12, 12), bg, 1) && !Differs(PixelAt(px, W, 13, 5), bg, 1) &&
                                                            !Differs(PixelAt(px, W, 5, H - 6), bg, 1));
    const int rows4 = ExactInColumn(px, W, 40, 30, 50, blue);
    std::printf("   4 px line: %d exact rows\n", rows4);
    check("a 4 px screen line is 4 rows of exact colour", rows4 == 4 && Exact(PixelAt(px, W, 40, 38), blue) && Exact(PixelAt(px, W, 40, 41), blue));
    check("... with background just outside", !Differs(PixelAt(px, W, 40, 37), bg, 1) && !Differs(PixelAt(px, W, 40, 42), bg, 1));
    check("a 1 px screen line is one row", ExactInColumn(px, W, 40, 50, 70, green) == 1 && Exact(PixelAt(px, W, 40, 60), green));
    check("square caps: the line ends half its width past its end point", Exact(PixelAt(px, W, 81, 39), blue) && !Differs(PixelAt(px, W, 83, 39), bg, 1));

    // Batches queued outside a frame are dropped.
    r.DrawScreenTriangles(quad);
    r.BeginFrame(W, H);
    r.SetCamera(glm::mat4(1.f), glm::mat4(1.f));
    r.EndFrame();
    check("overlay calls outside a frame are ignored", !Differs(PixelAt(target->ReadbackRGBA8(), W, 5, 5), bg, 1));
}

void TestOverlayDepth(RenderDevice& dev) {
    std::printf("-- overlays: depth modes against a mesh --\n");
    PbrScene s(dev);   // camera on +Z at 3; the quad at z = 0 spans ±1
    const int        W = PbrScene::W;
    const Rgba       bg = Background();
    const MeshHandle quad = s.r.UploadMesh(MakeQuad());
    const Rgba8      hidden{ 250, 20, 200, 255 }, onTop{ 20, 230, 40, 255 }, front{ 240, 240, 20, 255 }, outside{ 20, 200, 240, 255 };
    const Rgba8      legacy{ 255, 128, 0, 255 }, triHidden{ 128, 0, 255, 255 }, triTop{ 255, 40, 120, 255 }, screen{ 10, 10, 10, 255 };

    auto px = s.Render([&](auto& r) {
        r.DrawMesh(quad, glm::mat4(1.f));
        r.DrawOverlayLines(Seg({ 0.6f, -0.5f, -1.f }, { 0.6f, 0.5f, -1.f }, hidden), 4.f, OverlayDepth::Tested);
        r.DrawOverlayLines(Seg({ 0.f, -0.5f, -1.f }, { 0.f, 0.5f, -1.f }, onTop), 6.f, OverlayDepth::OnTop);
        r.DrawOverlayLines(Seg({ -0.6f, -0.5f, 0.5f }, { -0.6f, 0.5f, 0.5f }, front), 4.f, OverlayDepth::Tested);
        r.DrawOverlayLines(Seg({ 2.f, -0.5f, -1.f }, { 2.f, 0.5f, -1.f }, outside), 4.f, OverlayDepth::Tested);
        const LineVertex lv[] = { { { -0.5f, -0.8f, 0.2f }, 255, 128, 0, 255 }, { { 0.5f, -0.8f, 0.2f }, 255, 128, 0, 255 } };
        r.DrawLines(lv, 3.f);
        r.DrawOverlayTriangles(Tri({ 0.4f, 0.75f, -1.f }, 0.2f, triHidden), OverlayDepth::Tested);
        r.DrawOverlayTriangles(Tri({ -0.4f, 0.75f, -1.f }, 0.2f, triTop), OverlayDepth::OnTop);
    });
    const Rgba quadPx = s.At(px, { 0.3f, -0.3f, 0.f });
    check("the occluding quad is drawn", Differs(quadPx, bg));
    const Rgba h = s.At(px, { 0.6f, 0.f, -1.f });
    std::printf("   tested line behind the quad -> (%d,%d,%d)  quad (%d,%d,%d)\n", h.r, h.g, h.b, quadPx.r, quadPx.g, quadPx.b);
    check("a depth-tested line behind the mesh is hidden", !Exact(h, hidden, 40) && !Differs(h, quadPx, 15));
    check("a depth-tested line in front of the mesh shows, exact", Exact(s.At(px, { -0.6f, 0.f, 0.5f }), front));
    check("a depth-tested line clear of the mesh shows, exact", Exact(s.At(px, { 2.f, 0.f, -1.f }), outside));
    const int across = ExactInRow(px, W, PbrScene::H / 2, onTop);
    std::printf("   on-top 6 px line: %d exact pixels across\n", across);
    check("an on-top line behind the mesh shows, 6 px wide, exact", across == 6);
    check("DrawLines is a depth-tested batch with exact colour", Exact(s.At(px, { 0.f, -0.8f, 0.2f }), legacy));
    check("a depth-tested triangle behind the mesh is hidden", !Exact(s.At(px, { 0.4f, 0.75f, -1.f }), triHidden, 40));
    check("an on-top triangle behind the mesh shows, exact", Exact(s.At(px, { -0.4f, 0.75f, -1.f }), triTop));

    // Layer order: the screen layer is drawn over on-top overlays.
    const glm::ivec2 at = Project(s.proj * s.view, { -0.4f, 0.75f, -1.f }, W, PbrScene::H);
    px = s.Render([&](auto& r) {
        r.DrawOverlayTriangles(Tri({ -0.4f, 0.75f, -1.f }, 0.2f, triTop), OverlayDepth::OnTop);
        const glm::vec2 c(static_cast<float>(at.x) + 0.5f, static_cast<float>(at.y) + 0.5f);
        const ScreenVertex t[] = { { c + glm::vec2(-3, -3), screen }, { c + glm::vec2(3, -3), screen }, { c + glm::vec2(0, 3), screen } };
        r.DrawScreenTriangles(t);
    });
    check("the screen layer draws over on-top overlays", Exact(PixelAt(px, W, at.x, at.y), screen));
    s.r.FreeMesh(quad);
}

// 16×8 R8 atlas: 'A' is a solid 4×6 block at (0,0), '?' a solid 3×3 at (8,0).
GlyphAtlas TinyAtlas() {
    GlyphAtlas a;
    a.width  = 16;
    a.height = 8;
    a.pixels.assign(16 * 8, 0);
    for (int y = 0; y < 6; ++y)
        for (int x = 0; x < 4; ++x) a.pixels[y * 16 + x] = 255;
    for (int y = 0; y < 3; ++y)
        for (int x = 8; x < 11; ++x) a.pixels[y * 16 + x] = 255;
    a.lineHeight = 10.f;
    a.ascent     = 8.f;
    a.glyphs     = { { 'A', 0, 0, 4, 6, 0.f, 0.f, 1.f, 6.f, 6.f }, { '?', 8, 0, 3, 3, 0.f, 0.f, 0.f, 5.f, 4.f }, { ' ', 0, 0, 0, 0, 0.f, 0.f, 0.f, 0.f, 3.f } };
    return a;
}

// True when exactly the pixels of [x0,x1)×[y0,y1) hold colour c within the 1-px border around it.
bool BlockAt(const std::vector<std::uint8_t>& px, int w, int x0, int y0, int x1, int y1, Rgba8 c) {
    for (int y = y0 - 1; y <= y1; ++y)
        for (int x = x0 - 1; x <= x1; ++x) {
            const bool in = x >= x0 && x < x1 && y >= y0 && y < y1;
            if (Exact(PixelAt(px, w, x, y), c) != in) return false;
        }
    return true;
}

void TestOverlayText(RenderDevice& dev) {
    std::printf("-- overlays: text --\n");
    constexpr int W = 96, H = 96;
    auto target = OffscreenTarget::Create(dev, W, H);
    DiligentSceneRenderer r(dev);
    r.SetOutputTarget(*target);
    const Rgba8 ink{ 40, 200, 120, 255 }, pillCol{ 90, 10, 10, 255 };

    check("no atlas: MeasureText is zero", r.MeasureText("A").width == 0.f && !r.HasGlyphAtlas());
    GlyphAtlas bad = TinyAtlas();
    bad.pixels.resize(10);
    r.SetGlyphAtlas(bad);
    check("an atlas whose pixels don't match its size is refused", !r.HasGlyphAtlas());
    r.SetGlyphAtlas(TinyAtlas());
    check("a valid atlas is accepted", r.HasGlyphAtlas());
    const TextExtent e = r.MeasureText("AA");
    check("MeasureText: advance width and line height", e.width == 12.f && e.height == 10.f && e.ascent == 8.f);

    TextPill pill;
    pill.color  = pillCol;
    pill.padX   = 3.f;
    pill.padY   = 2.f;
    pill.radius = 0.f;
    r.BeginFrame(W, H);
    r.SetCamera(glm::mat4(1.f), glm::mat4(1.f));
    r.DrawScreenText({ 10.f, 20.f }, "A", ink);                                                      // top-left
    r.DrawScreenText({ 50.f, 40.f }, "A", ink, { TextAlignH::Right, TextAlignV::Baseline });
    r.DrawScreenText({ 70.f, 70.f }, "A", ink, { TextAlignH::Center, TextAlignV::Middle });
    r.DrawScreenText({ 10.f, 60.f }, "A", ink, {}, &pill);
    r.DrawScreenText({ 30.f, 80.f }, "\xC3\x84", ink);   // Ä: not in the atlas -> the '?' fallback
    r.EndFrame();
    const auto px = target->ReadbackRGBA8();

    const Rgba g = PixelAt(px, W, 12, 24);
    std::printf("   glyph pixel (%d,%d,%d), expected (40,200,120)\n", g.r, g.g, g.b);
    check("top-left: the glyph covers exactly x 11..14, y 22..27 in the text colour", BlockAt(px, W, 11, 22, 15, 28, ink));
    check("right/baseline: x 45..48, y 34..39", BlockAt(px, W, 45, 34, 49, 40, ink));
    check("centre/middle: x 68..71, y 67..72", BlockAt(px, W, 68, 67, 72, 73, ink));
    check("a pill sits behind the text, padded", Exact(PixelAt(px, W, 8, 59), pillCol) && Exact(PixelAt(px, W, 17, 70), pillCol) &&
                                                     Exact(PixelAt(px, W, 12, 64), ink) && !Exact(PixelAt(px, W, 6, 59), pillCol));
    check("a missing glyph draws the fallback", BlockAt(px, W, 30, 83, 33, 86, ink));

    // An RGBA8 atlas tints: texel rgb × text colour.
    GlyphAtlas rgba = TinyAtlas();
    rgba.format = GlyphAtlas::Format::RGBA8;
    rgba.pixels.assign(16 * 8 * 4, 0);
    for (int y = 0; y < 6; ++y)
        for (int x = 0; x < 4; ++x) {
            std::uint8_t* t = &rgba.pixels[(y * 16 + x) * 4];
            t[0] = 255, t[1] = 0, t[2] = 0, t[3] = 255;
        }
    r.SetGlyphAtlas(rgba);
    r.BeginFrame(W, H);
    r.SetCamera(glm::mat4(1.f), glm::mat4(1.f));
    r.DrawScreenText({ 10.f, 20.f }, "A", Rgba8{ 255, 255, 255, 255 });
    r.EndFrame();
    check("an RGBA8 atlas's colour shows through white text", BlockAt(target->ReadbackRGBA8(), W, 11, 22, 15, 28, Rgba8{ 255, 0, 0, 255 }));
}

// root(0) -> com(1) -> spine(2) -> {arm(3), arm(4)}; the camera on -Y looks at the spine.
const std::vector<glm::vec3> kJoints  = { { 0, 0, 0 }, { 0, 0, 2 }, { 0, 0, 6 }, { 4, 0, 6 }, { -4, 0, 6 } };
const std::vector<int>       kParents = { -1, 0, 1, 2, 2 };

bool Blueish(Rgba p) { return p.b > p.r + 40 && p.b > p.g; }    // the default bone colour (150, 200, 255), shaded
bool Goldish(Rgba p) { return p.r > p.b + 60 && p.g > p.b + 40; }   // the selected colour (255, 210, 90), shaded

void TestOverlaySkeleton(RenderDevice& dev) {
    std::printf("-- overlays: skeleton --\n");
    PbrScene s(dev);
    const Rgba bg = Background();
    s.view     = glm::lookAt(glm::vec3(0.f, -30.f, 4.f), glm::vec3(0.f, 0.f, 4.f), glm::vec3(0.f, 0.f, 1.f));
    s.proj     = glm::perspective(glm::radians(25.f), 1.f, 0.1f, 100.f);
    s.lightDir = { 0.f, -1.f, 0.f };
    // An occluder in front of the spine only (a quad in the XZ plane, facing -Y).
    const MeshHandle quad   = s.r.UploadMesh(MakeQuad());
    const glm::mat4  qWorld = glm::translate(glm::mat4(1.f), glm::vec3(0.f, -5.f, 4.f)) *
                             glm::rotate(glm::mat4(1.f), glm::radians(90.f), glm::vec3(1.f, 0.f, 0.f)) * glm::scale(glm::mat4(1.f), glm::vec3(0.8f));
    const glm::vec3 spine(0.f, 0.f, 3.5f), arm(1.2f, 0.f, 6.f), otherArm(-1.2f, 0.f, 6.f);

    SkeletonStyle style;
    style.rootSize     = 1.5f;
    style.selectedBone = 3;
    style.visible      = { 1, 1, 1, 1, 0 };
    auto draw = [&](auto& r) {
        r.DrawMesh(quad, qWorld);
        r.SetSkeletonStyle(style);
        r.DrawBones(kJoints, kParents);
    };
    auto px = s.Render(draw);
    const Rgba sp = s.At(px, spine), a = s.At(px, arm), o = s.At(px, otherArm);
    std::printf("   on top: spine (%d,%d,%d)  selected arm (%d,%d,%d)  hidden arm (%d,%d,%d)\n", sp.r, sp.g, sp.b, a.r, a.g, a.b, o.r, o.g, o.b);
    check("on-top bones show through the occluder in the bone colour", Blueish(sp));
    check("the selected bone is in the selected colour", Goldish(a));
    check("a bone masked out by `visible` is not drawn", !Differs(o, bg, 2));

    style.depth = OverlayDepth::Tested;
    px = s.Render(draw);
    const Rgba spT = s.At(px, spine);
    check("depth-tested bones hide behind the occluder", !Blueish(spT) && Differs(spT, bg));
    check("... and still show where nothing covers them", Goldish(s.At(px, arm)));

    s.r.FreeMesh(quad);
}

// --load-dds <dir>: load every .dds in a folder (e.g. samples pulled from the
// game's BSAs) in both colour spaces and report each.
int LoadDdsFolder(RenderDevice& dev, const std::string& dir) {
    std::printf("-- loading every .dds in %s --\n", dir.c_str());
    DiligentSceneRenderer r(dev);
    std::error_code       ec;
    int                   files = 0;
    for (const auto& e : std::filesystem::directory_iterator(dir, ec)) {
        std::string ext = e.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (ext != ".dds") continue;
        ++files;
        std::ifstream             f(e.path(), std::ios::binary);
        std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        const TextureHandle s = r.LoadTextureFromMemory(bytes, TexColorSpace::Srgb);
        const TextureHandle l = r.LoadTextureFromMemory(bytes, TexColorSpace::Linear);
        check((e.path().filename().string() + " loads (sRGB and linear)").c_str(), s != TextureHandle::Invalid && l != TextureHandle::Invalid);
        r.FreeTexture(s);
        r.FreeTexture(l);
    }
    check("the folder has .dds files", files > 0);
    return g_failures;
}

} // namespace

int main(int argc, char** argv) {
    DeviceOptions opts;
    opts.allowFallback = false;   // test exactly the backend asked for
    std::string ddsFolder;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--load-dds") == 0 && i + 1 < argc) {
            ddsFolder = argv[++i];
        } else if (std::strcmp(argv[i], "--backend") == 0 && i + 1 < argc) {
            const std::string b = argv[++i];
            opts.preferred = b == "d3d12" ? Backend::D3D12 : Backend::Vulkan;
        } else if (std::strcmp(argv[i], "--warp") == 0) {
            opts.preferred = Backend::D3D12;
            opts.useWarp   = true;
        } else if (std::strcmp(argv[i], "--validation") == 0) {
            opts.validation = true;
        }
    }

    std::string log;
    auto dev = RenderDevice::Create(opts, &log);
    std::printf("%s", log.c_str());
    check("device created", dev != nullptr);
    if (!dev) return g_failures;
    std::printf("== %s on %s ==\n", BackendName(dev->backend()), dev->adapterName().c_str());
    if (!ddsFolder.empty()) return LoadDdsFolder(*dev, ddsFolder);

    TestClearReadback(*dev);
    TestEmptyFrame(*dev);
    TestGridAxes(*dev);
    TestResources(*dev);
    TestPbr(*dev);
    TestNormalMap(*dev);
    TestFaceTint(*dev);
    TestSkinTint(*dev);
    TestLegacyDds(*dev);
    TestOverlayScreen(*dev);
    TestOverlayDepth(*dev);
    TestOverlayText(*dev);
    TestOverlaySkeleton(*dev);

    std::printf(g_failures == 0 ? "\nALL PASSED\n" : "\n%d CHECK(S) FAILED\n", g_failures);
    return g_failures;
}
