#pragma once
// ── RENDER — the Diligent backend (render::diligent) ──────────────────────────
// A Diligent Engine implementation of render::ISceneRenderer, plus the
// device / swap-chain plumbing a host application needs to put it on screen.
//
//   RenderDevice          the graphics device + immediate context (Vulkan or
//                         Direct3D 12, with fallback to the other)
//   ViewportSurface       one swap chain on a native window (HWND)
//   OffscreenTarget       an off-screen colour target with CPU readback (tests,
//                         thumbnails)
//   DiligentSceneRenderer ISceneRenderer + IOverlayRenderer: HDR scene pass
//                         (DiligentFX PBR for meshes), face-tint composite, an
//                         ACES tone-map into whichever output target is bound,
//                         then the grid and the overlays (lines, triangles,
//                         text, bones) over the tone-mapped image
//
// No Diligent type crosses this header (pimpl throughout), so consumers need
// neither Diligent's include paths nor its macros.
//
// Conventions: the ISceneRenderer contract is GL-style — world Z-up, right-handed,
// counter-clockwise front faces, projection matrices with NDC z in -1..1 (as
// glm::perspective builds them). The renderer converts to the backend's
// conventions itself; callers never V-flip or remap depth.
//
// The backend target is `render::diligent`; consumers write `#include <render/Diligent.h>`.
// ------------------------------------------------------------------------------

#include <render/Render.h>

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace render::diligent {

enum class Backend { Vulkan, D3D12 };
const char* BackendName(Backend b);

struct DeviceOptions {
    Backend preferred     = Backend::Vulkan;
    bool    allowFallback = true;    // try the other backend if the preferred one fails
    bool    validation    = false;   // backend validation layers (Vulkan layers / D3D12 debug layer)
    bool    useWarp       = false;   // D3D12 only: the WARP software adapter (GPU-less test runs)
};

// ── RenderDevice ──────────────────────────────────────────────────────────────
// Owns the engine factory, the render device and the immediate context. One per
// process is the expected use; every other object here borrows it and must be
// destroyed before it.
class RenderDevice {
public:
    // Null on failure; `log` (optional) receives a human-readable account of what
    // was tried. With allowFallback, a failed preferred backend falls back to the
    // other one.
    static std::unique_ptr<RenderDevice> Create(const DeviceOptions& opts, std::string* log = nullptr);
    ~RenderDevice();
    RenderDevice(const RenderDevice&)            = delete;
    RenderDevice& operator=(const RenderDevice&) = delete;

    Backend            backend() const;
    const std::string& adapterName() const;
    bool               supportsWireframe() const;   // FILL_MODE_WIREFRAME granted
    bool               supportsSwizzle() const;     // texture-view component swizzle granted

    // Ends a frame that did GPU work but presented nothing (lets the engine release
    // stale resources). A frame that presents a ViewportSurface needs no call.
    void FinishFrame();
    // Blocks until the GPU is idle (shutdown, resizes).
    void WaitIdle();

    struct Impl;
    Impl& impl();

private:
    explicit RenderDevice(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> m_impl;
};

// ── ViewportSurface ───────────────────────────────────────────────────────────
// A swap chain on a native window (an HWND on Windows). Colour is RGBA8 sRGB; it
// has no depth buffer (the renderer's scene pass owns depth). Sizes are physical
// pixels; a 0×0 resize (minimised) is ignored.
class ViewportSurface {
public:
    static std::unique_ptr<ViewportSurface> Create(RenderDevice& device, void* nativeWindow,
                                                   int width, int height, std::string* err = nullptr);
    ~ViewportSurface();
    ViewportSurface(const ViewportSurface&)            = delete;
    ViewportSurface& operator=(const ViewportSurface&) = delete;

    void* nativeWindow() const;
    int   width() const;
    int   height() const;
    void  Resize(int width, int height);
    // Presents the last frame rendered into it. syncInterval 0 = don't wait for vblank.
    void  Present(int syncInterval);

    struct Impl;
    Impl& impl();

private:
    explicit ViewportSurface(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> m_impl;
};

// ── OffscreenTarget ───────────────────────────────────────────────────────────
// An RGBA8 sRGB colour target with CPU readback.
class OffscreenTarget {
public:
    static std::unique_ptr<OffscreenTarget> Create(RenderDevice& device, int width, int height);
    ~OffscreenTarget();
    OffscreenTarget(const OffscreenTarget&)            = delete;
    OffscreenTarget& operator=(const OffscreenTarget&) = delete;

    int width() const;
    int height() const;
    // Fill with a colour given in sRGB-encoded 0..1 components (what ReadbackRGBA8 returns).
    void Clear(float r, float g, float b, float a);
    // The target's pixels: rows top to bottom, tightly packed RGBA8 (sRGB-encoded).
    // Waits for the GPU.
    std::vector<std::uint8_t> ReadbackRGBA8();

    struct Impl;
    Impl& impl();

private:
    explicit OffscreenTarget(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> m_impl;
};

// ── DiligentSceneRenderer ─────────────────────────────────────────────────────

// One vertex of a DrawLines batch: world position + RGBA8 colour (display values).
struct LineVertex {
    glm::vec3    pos;
    std::uint8_t r = 255, g = 255, b = 255, a = 255;
};

struct FrameStats {
    int draws            = 0;   // mesh draws issued last frame
    int skippedCompiling = 0;   // mesh draws skipped because their pipeline was still compiling
    int meshes           = 0;   // live meshes
    int textures         = 0;   // live textures
    int failedTextureLoads = 0; // cumulative LoadTexture* failures
};

// Background and ground-grid look. Colours are display (sRGB-encoded) values, as
// they appear on screen: the background is composited after tone mapping, and the
// grid is drawn over the tone-mapped image.
struct ViewStyle {
    std::array<float, 3> background{ 0.11f, 0.115f, 0.125f };
    std::array<float, 3> gridMinor{ 0.28f, 0.29f, 0.31f };
    std::array<float, 3> gridMajor{ 0.42f, 0.43f, 0.46f };
    std::array<float, 3> axisX{ 0.78f, 0.26f, 0.26f };   // red = east
    std::array<float, 3> axisY{ 0.32f, 0.72f, 0.32f };   // green = north
    int                  gridMajorEvery = 4;             // a major line every N minor cells
};

// The overlay font, baked by the host (the renderer has no font rasteriser).
// Atlas rect and size are in atlas texels / screen pixels, which are 1:1 for a
// font baked at the display's pixel density. Bearings follow FreeType: x from
// the pen to the ink's left edge, y from the baseline UP to the ink's top edge.
struct GlyphMetrics {
    std::uint32_t codepoint = 0;
    int           x = 0, y = 0, w = 0, h = 0;   // atlas rect (texels)
    float         sizeW = 0.f, sizeH = 0.f;     // on-screen quad size (pixels); 0 = the rect's size
    float         bearingX = 0.f, bearingY = 0.f;
    float         advance  = 0.f;               // pen advance (pixels)
};

struct GlyphAtlas {
    enum class Format { R8, RGBA8 };   // R8: coverage; RGBA8: rgb tints the text colour, a = coverage
    int                       width = 0, height = 0;
    Format                    format = Format::R8;
    std::vector<std::uint8_t> pixels;            // rows top to bottom, tightly packed
    std::vector<GlyphMetrics> glyphs;
    float                     lineHeight = 0.f;  // baseline-to-baseline (pixels)
    float                     ascent     = 0.f;  // line top to baseline (pixels)
    std::uint32_t             fallback   = '?';  // drawn for code points the atlas lacks
};

class DiligentSceneRenderer final : public ISceneRenderer, public IOverlayRenderer {
public:
    explicit DiligentSceneRenderer(RenderDevice& device);
    ~DiligentSceneRenderer() override;
    DiligentSceneRenderer(const DiligentSceneRenderer&)            = delete;
    DiligentSceneRenderer& operator=(const DiligentSceneRenderer&) = delete;

    // Where EndFrame writes the tone-mapped frame. Set before BeginFrame; the
    // target must outlive the frame. With no target bound, EndFrame skips the
    // output pass (and says so once).
    void SetOutputTarget(ViewportSurface& surface);
    void SetOutputTarget(OffscreenTarget& target);
    void ClearOutputTarget();

    void             SetExposure(float exposure);   // HDR multiplier before ACES (default 1)
    void             SetViewStyle(const ViewStyle& style);
    const ViewStyle& GetViewStyle() const;
    FrameStats       LastFrameStats() const;

    // Queue world-space line segments (pairs of vertices) for this frame: a
    // depth-tested overlay batch (DrawOverlayLines with OverlayDepth::Tested), so
    // the colours are display values.
    void DrawLines(std::span<const LineVertex> segments, float widthPx = 1.f);

    // The font for DrawScreenText / MeasureText (copied; replaces any earlier one).
    // An atlas with no pixels or no glyphs clears it.
    void SetGlyphAtlas(const GlyphAtlas& atlas);
    bool HasGlyphAtlas() const;

    // ── IOverlayRenderer (see <render/Render.h>) ──
    IOverlayRenderer* Overlays() override { return this; }
    void DrawOverlayLines(std::span<const OverlayVertex> segments, float widthPx, OverlayDepth depth) override;
    void DrawOverlayTriangles(std::span<const OverlayVertex> triangles, OverlayDepth depth) override;
    void DrawScreenLines(std::span<const ScreenVertex> segments, float widthPx) override;
    void DrawScreenTriangles(std::span<const ScreenVertex> triangles) override;
    void DrawScreenText(glm::vec2 pos, std::string_view utf8, Rgba8 color, TextAnchor anchor = {},
                        const TextPill* pill = nullptr) override;
    TextExtent MeasureText(std::string_view utf8) const override;
    void SetSkeletonStyle(const SkeletonStyle& style) override;
    void DrawBones(std::span<const glm::vec3> joints, std::span<const int> parents,
                   const glm::mat4& world = glm::mat4(1.f)) override;

    // CPU copy of a texture this renderer produced on the GPU (CompositeFaceTint
    // results): RGBA8 sRGB-encoded, rows top to bottom. False for other textures.
    bool DebugReadbackRGBA8(TextureHandle texture, std::vector<std::uint8_t>& pixels, int& width, int& height);

    // ── ISceneRenderer ──
    void BeginFrame(int width, int height) override;
    void SetCamera(const glm::mat4& view, const glm::mat4& proj) override;
    void EndFrame() override;

    MeshHandle    UploadMesh(const MeshData& data) override;
    void          FreeMesh(MeshHandle handle) override;
    TextureHandle UploadBlendMap(const float* data33x33) override;
    void          UpdateMeshPositions(MeshHandle handle, std::span<const glm::vec3> positions) override;
    void          UpdateMeshNormals(MeshHandle handle, std::span<const glm::vec3> normals) override;
    TextureHandle LoadTexture(const std::string& path, TexColorSpace space) override;
    TextureHandle LoadTextureFromMemory(const std::vector<uint8_t>& bytes, TexColorSpace space) override;
    void          FreeTexture(TextureHandle handle) override;
    TextureHandle CompositeFaceTint(TextureHandle base, std::span<const FaceTintLayer> layers) override;

    void SetLight(const glm::vec3& dir, const glm::vec3& color, const glm::vec3& ambient) override;
    // The infinite ground grid (Z = 0), drawn at EndFrame over the tone-mapped image
    // and depth-tested against the scene: minor lines every `cellSize`, fading out
    // `cellSize * halfExtent` from the camera.
    void DrawGrid(float cellSize = 1.f, int halfExtent = 10) override;
    void DrawMesh(MeshHandle mesh, const glm::mat4& world, const DrawSurface& surface = {}) override;
    void DrawSkinnedMesh(MeshHandle mesh, const glm::mat4& world,
                         std::span<const glm::mat4> boneTransforms, const DrawSurface& surface = {}) override;
    void DrawTerrainTile(MeshHandle mesh, const TerrainSurface& surface) override;

    struct Impl;

private:
    std::unique_ptr<Impl> m_impl;
};

} // namespace render::diligent
