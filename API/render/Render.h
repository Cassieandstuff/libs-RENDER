#pragma once
// ── RENDER — the backend-neutral renderer interface (render::) ────────────────
// The ISceneRenderer interface, the mesh/surface descriptors and typed handles
// it trades in, the texture colour-space tag, the optional overlay interface
// (IOverlayRenderer: lines, triangles, text, bones), and the deduplicating
// GpuAssetCache. No graphics API and no UI toolkit: asset code and an editor's
// logic layer build against this alone; a backend (render::diligent) implements it.
//
// Split out of SCT's sct-renderer/SctRenderer.h, minus the GL implementation
// class and GetOutputTexture() (which handed an ImTextureID to Dear ImGui). How
// a backend presents its frame is the backend's own API, not this interface's.
//
// The interface target is `render::api`; consumers write `#include <render/Render.h>`.
// ------------------------------------------------------------------------------

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include <glm/glm.hpp>

namespace render {

// Colour space of texel data, decided by how the texture is *used* — the file
// itself usually can't tell us.  Skyrim's textures are overwhelmingly
// legacy-header DXT1/DXT5, and the FourCC carries no colour-space bit; even the
// DX10 *_SRGB DXGI enums appear on only a minority of files.  So the caller
// picks, based on the shader slot:
//
//   Srgb   — colour authored for display: diffuse/albedo (slot 0), glow (slot 2).
//   Linear — numeric data: normals (slot 1), height (slot 3), RMAOS / complex
//            material (slot 5), and anything else sampled as a value.
//
// Uploading a data map as Srgb silently corrupts it, so Linear is the safe
// answer whenever a slot's meaning is unclear.  See
// docs/nif-material-conventions.md for the full slot table.
enum class TexColorSpace { Linear, Srgb };

// ── Typed GPU resource handles ────────────────────────────────────────────────
enum class MeshHandle    : uint32_t { Invalid = 0 };
enum class TextureHandle : uint32_t { Invalid = 0 };

// ── Mesh upload descriptor ─────────────────────────────────────────────────────
// boneIndices / boneWeights are optional (leave empty for static meshes).
// tangents are optional (xyz = tangent direction, w = handedness ±1); required
// for normal-map shading.  indices are required; positions/normals/uvs must be
// the same length.
struct MeshData {
    std::vector<glm::vec3>   positions;
    std::vector<glm::vec3>   normals;
    std::vector<glm::vec2>   uvs;
    std::vector<glm::vec4>   tangents;      // optional: xyz=tangent, w=handedness (±1)
    std::vector<glm::u8vec4> boneIndices;   // 4 indices per vertex (uint8)
    std::vector<glm::vec4>   boneWeights;   // 4 weights per vertex, sum == 1
    std::vector<glm::u8vec3> vertexColors;  // optional RGB per vertex (0-255)
    std::vector<uint16_t>    indices;
};

// ── Per-draw surface parameters ───────────────────────────────────────────────
struct DrawSurface {
    // AlphaTestAndBlend: discard fragments below alphaThreshold AND blend remaining
    // pixels (depth write off) — matches NifSkope's independent glEnable(GL_BLEND) +
    // glEnable(GL_ALPHA_TEST) when both NiAlphaProperty bits 0 and 9 are set.
    enum class BlendMode { Opaque, AlphaTest, AlphaBlend, Additive, AlphaTestAndBlend };

    TextureHandle diffuse         = TextureHandle::Invalid;
    TextureHandle normalMap       = TextureHandle::Invalid;  // tangent-space _n.dds
    // True PBR only (slot 5 under the PBRNifPatcher marker): R=roughness,
    // G=metallic, B=ambient occlusion, A=specular level.  Leave Invalid on
    // vanilla meshes — there slot 5 is an environment mask, not material data.
    TextureHandle rmaosMap        = TextureHandle::Invalid;
    TextureHandle glowMap         = TextureHandle::Invalid;  // slot 2, when hasGlowMap
    // External specular mask (vanilla model-space-normal shapes — skin — keep it in
    // slot 7, `_s.dds`): greyscale, the mask is read from its red channel and
    // scales specularStrength. Ignored when specFromNormalAlpha or rmaosMap applies.
    TextureHandle specularMap     = TextureHandle::Invalid;
    glm::vec4     tint            = { 1.f, 1.f, 1.f, 1.f };
    float         roughness       = 0.8f;   // PBR roughness fallback (0=mirror,1=diffuse)
    float         metalness       = 0.0f;   // PBR metalness (0=dielectric, 1=metal)
    glm::vec3     emissiveColor   = { 0.f, 0.f, 0.f };  // scales glowMap
    float         emissiveMultiple = 1.f;
    float         specularStrength = 1.f;
    // Vanilla meshes carry the specular mask in the normal map's alpha channel.
    // PBR meshes do not — their slot 1 alpha is meaningless, so this stays false.
    bool          specFromNormalAlpha = false;
    bool          wireframe       = false;
    bool          xray            = false;
    bool          twoSided        = false;   // disable backface cull (hair, leaves)
    bool          useVertexColor  = false;   // use per-vertex RGB from location 5 as base color
    bool          terrainTex      = false;   // terrain mode: tiled diffuse × (vtxcol×2)
    float         texTileRate     = 6.f;     // UV tile scale for terrain mode
    BlendMode     blendMode       = BlendMode::Opaque;
    float         alphaThreshold  = 0.5f;   // normalised; only used for AlphaTest mode
};

// ── Terrain draw surface ──────────────────────────────────────────────────────
// Passed to DrawTerrainTile.  layer[0] = base BTXT texture; layers[1-5] = ATXT
// alpha overlays.  blendMaps[i] is the per-vertex opacity map for layers[i+1].
// All handles must be valid if layerCount > 0; unused slots are ignored.
struct TerrainSurface {
    static constexpr int kMaxLayers = 6;
    int           layerCount  = 1;
    TextureHandle layers[kMaxLayers]          = {};
    float         layerRates[kMaxLayers]      = {6,6,6,6,6,6};
    TextureHandle blendMaps[kMaxLayers - 1]   = {};  // one per alpha layer
    bool          useVertexColor = false;             // apply VCLR × 2 modulation
};

// ── Face-tint composite layer ─────────────────────────────────────────────────
// One layer for CompositeFaceTint — the RaceMenu / facegen face-tint model. Each
// layer takes a coverage mask (grayscale/alpha .dds) and a tint colour, and is
// composited in order over the base face diffuse. `mask` and `base` are ordinary
// TextureHandles (load the mask Linear — it is coverage data, not colour).
struct FaceTintLayer {
    // Overlay: paint the tint where the mask covers (warpaint, makeup, dirt).
    // Multiply: skin tone — Skyrim's modulate-2x (base × 2·tint), so a mid-grey
    // tint is neutral and a light tone lightens a dark base (a plain multiply
    // could only darken). Used for the skin-tone layer.
    // SoftLight: Skyrim's FaceGen / SkinTint shading. The tint T is applied to the
    // ORIGINAL base b as b² + 2·T·b·(1 − b), on sRGB-encoded values (the game
    // shades in gamma space): T = 0.5 is neutral, lighter tones lighten, darker
    // darken. Coverage k blends towards that: lerp(current, softlight(b, T), k).
    // Because soft light is linear in T, a run of SoftLight layers is exactly
    // softlight(b, M) where M is the layers' colours painted over neutral grey
    // (how the game builds a face tint texture from its masks). T is `color.rgb`
    // (display values), or the texel of `tintMap` when one is given (a FaceGen
    // facetint .dds, loaded Srgb).
    // Detail: Skyrim's FaceGen detail map — the current colour × 3.984375·(D +
    // (1, 0, 1)/255) on sRGB-encoded values, D = the `tintMap` texel (a 64/255 grey
    // map is neutral); k blends towards it. Without a tintMap the layer is skipped.
    enum class Blend : int { Overlay = 0, Multiply = 1, SoftLight = 2, Detail = 3 };

    TextureHandle mask   = TextureHandle::Invalid;  // Invalid = full coverage (no mask)
    glm::vec4     color  = { 1.f, 1.f, 1.f, 1.f };  // rgb = tint colour; a = strength 0..1
    Blend         blend  = Blend::Overlay;
    bool          coverageFromAlpha = true;         // false = use mask luminance as coverage
    TextureHandle tintMap = TextureHandle::Invalid; // SoftLight: per-texel tint colour (replaces color.rgb); Detail: the detail map
};

// ── Overlays ──────────────────────────────────────────────────────────────────
// Viewport overlays (skeletons, paths, frusta, gizmos, labels) are drawn after the
// scene is tone-mapped, straight into the output, so their colours are exact:
// every colour here is a DISPLAY value (sRGB-encoded, as it appears on screen),
// like a UI colour. Widths and screen positions are PHYSICAL pixels (the host
// scales by its device pixel ratio).

// 8-bit RGBA, display space; a = opacity.
struct Rgba8 {
    std::uint8_t r = 255, g = 255, b = 255, a = 255;
};

struct OverlayVertex {   // world (render) space
    glm::vec3 pos{ 0.f };
    Rgba8     color;
};

struct ScreenVertex {    // physical pixels, origin at the top-left, y down
    glm::vec2 pos{ 0.f };
    Rgba8     color;
};

// How a world-space batch meets the scene:
//   Tested — depth-tested against the scene, writes no depth (hidden behind walls).
//   OnTop  — drawn over the scene. On-top primitives still depth-sort among
//            themselves (they get a depth buffer of their own), so a solid bone
//            shows its front faces.
enum class OverlayDepth { Tested, OnTop };

enum class TextAlignH { Left, Center, Right };
enum class TextAlignV { Top, Middle, Baseline };
// Which point of the text's box `pos` names: e.g. {Right, Baseline} puts the end
// of the first line's baseline at `pos`.
struct TextAnchor {
    TextAlignH h = TextAlignH::Left;
    TextAlignV v = TextAlignV::Top;
};

// A rounded box behind a string (v2's name labels). Padding / radius in pixels;
// radius < 0 = fully round ends.
struct TextPill {
    Rgba8 color{ 20, 22, 28, 200 };
    float padX   = 6.f;
    float padY   = 2.f;
    float radius = -1.f;
};

struct TextExtent {
    float width      = 0.f;   // widest line's advance
    float height     = 0.f;   // lines × lineHeight
    float ascent     = 0.f;   // baseline below the box top
    float lineHeight = 0.f;
};

// Blender-style octahedral bones (DrawBones). A bone runs from a
// joint's parent to the joint and is identified by the joint's index. A root (no
// parent) gets a ground ring with a +Y facing tick; the bones from a root to its
// children are not drawn (root-motion offsets, not limbs).
struct SkeletonStyle {
    Rgba8 bone{ 150, 200, 255, 255 };
    Rgba8 selected{ 255, 210, 90, 255 };
    Rgba8 hovered{ 255, 255, 255, 255 };
    Rgba8 root{ 230, 180, 70, 255 };
    int   selectedBone = -1;
    int   hoveredBone  = -1;
    float widthRatio   = 0.14f;   // octahedron half-width / bone length (world units)
    float minWidth     = 0.9f;    // keeps short spine / finger bones readable
    float maxWidth     = 3.0f;
    float rootSize     = 12.f;    // ground ring radius (world units)
    float rootLinePx   = 2.f;     // ring line width (pixels)
    // Per bone: 0 hides it. Empty, or shorter than the skeleton, shows the rest.
    std::vector<std::uint8_t> visible;
    OverlayDepth depth = OverlayDepth::OnTop;
};

// The overlay half of a renderer, reached through ISceneRenderer::Overlays().
// Every call queues work for the current frame (between BeginFrame and EndFrame);
// it is drawn at EndFrame in this order: the grid, Tested batches, OnTop batches,
// then the screen layer (2D lines, triangles, text), each layer in call order.
struct IOverlayRenderer {
    virtual ~IOverlayRenderer() = default;

    // World-space line segments (vertex pairs), `widthPx` physical pixels wide
    // (screen-aligned quads, anti-aliased edges).
    virtual void DrawOverlayLines(std::span<const OverlayVertex> segments, float widthPx, OverlayDepth depth) = 0;
    // World-space flat triangles (vertex triples), not culled.
    virtual void DrawOverlayTriangles(std::span<const OverlayVertex> triangles, OverlayDepth depth) = 0;

    // Screen-space (physical pixels, top-left origin), always on top, drawn last.
    virtual void DrawScreenLines(std::span<const ScreenVertex> segments, float widthPx) = 0;
    virtual void DrawScreenTriangles(std::span<const ScreenVertex> triangles) = 0;
    // UTF-8 text in the host-supplied font; '\n' starts a new line. Glyphs the font
    // lacks draw as its fallback glyph. A null pill draws none.
    virtual void DrawScreenText(glm::vec2 pos, std::string_view utf8, Rgba8 color, TextAnchor anchor = {},
                                const TextPill* pill = nullptr) = 0;
    // The box DrawScreenText would fill (zero without a font).
    virtual TextExtent MeasureText(std::string_view utf8) const = 0;

    // Style for the following DrawBones calls (selection, hover,
    // visibility mask, depth mode).
    virtual void SetSkeletonStyle(const SkeletonStyle& style) = 0;
    // Bones from model-space joint positions and parent indices (-1 = root),
    // placed by `world`.
    virtual void DrawBones(std::span<const glm::vec3> joints, std::span<const int> parents,
                           const glm::mat4& world = glm::mat4(1.f)) = 0;
};

// ── ISceneRenderer ────────────────────────────────────────────────────────────
// Scene-level rendering abstraction.  Implementations are drop-in: they differ
// only in backend machinery; the viewport only calls this.
//
// Frame sequence:
//   BeginFrame(w, h)          — bind / resize the render target, clear
//   SetCamera(view, proj)     — store matrices for this frame
//   DrawGrid / DrawMesh / DrawSkinnedMesh  (any order)
//   EndFrame()                — flush batched primitives, resolve the frame
struct ISceneRenderer {
    virtual ~ISceneRenderer() = default;

    // The renderer's overlay interface, or null when it has none. Optional, with a
    // default, so renderers written before it (SCT's GL backend) build unchanged.
    virtual IOverlayRenderer* Overlays() { return nullptr; }

    // ── Frame lifecycle ───────────────────────────────────────────────────────
    virtual void BeginFrame(int width, int height)                            = 0;
    virtual void SetCamera(const glm::mat4& view, const glm::mat4& proj)     = 0;
    virtual void EndFrame()                                                   = 0;

    // ── GPU resource management ───────────────────────────────────────────────
    virtual MeshHandle    UploadMesh  (const MeshData& data)                  = 0;
    virtual void          FreeMesh   (MeshHandle handle)                      = 0;
    // Upload a 33×33 single-channel float blend map as an R32F texture.
    // Used for terrain VTXT alpha layers; returns TextureHandle::Invalid on failure.
    virtual TextureHandle UploadBlendMap(const float* data33x33)              = 0;
    // Overwrite the position buffer of an already-uploaded mesh in-place.
    // `positions` must be the same length as the original upload.  Used for
    // CPU-side morph blending (face ARKit blend shapes).
    virtual void          UpdateMeshPositions(MeshHandle handle,
                                              std::span<const glm::vec3> positions) = 0;
    // Overwrite the normal buffer the same way (same length as the upload).  Used
    // beside UpdateMeshPositions when the CPU deforms a mesh (CPU-skinned faces), so
    // shading follows the deformed surface.
    virtual void          UpdateMeshNormals  (MeshHandle handle,
                                              std::span<const glm::vec3> normals)   = 0;
    // `space` is required, not defaulted: getting it wrong is silent (a washed
    // out albedo, or a corrupted normal map), so every call site must state
    // which it means.  See TexColorSpace above.
    virtual TextureHandle LoadTexture          (const std::string& path,
                                                TexColorSpace space)                   = 0;
    virtual TextureHandle LoadTextureFromMemory(const std::vector<uint8_t>& bytes,
                                                TexColorSpace space)                   = 0;
    virtual void          FreeTexture          (TextureHandle handle)                  = 0;

    // Composite face-tint layers over a base diffuse into a NEW texture (GPU
    // render-to-texture, one pass per layer). The result is owned by the renderer
    // and must be released with FreeTexture; `base` is left untouched. Returns
    // Invalid when `base` is invalid or `layers` is empty (the caller then uses
    // `base` as-is). Safe to call outside the BeginFrame/EndFrame scene pass.
    virtual TextureHandle CompositeFaceTint(TextureHandle base,
                                            std::span<const FaceTintLayer> layers)     = 0;

    // ── Draw calls ────────────────────────────────────────────────────────────
    // Set the scene directional light.  dir is expected to be normalised.
    // Call once per frame before draw calls.
    virtual void SetLight(const glm::vec3& dir,
                          const glm::vec3& color,
                          const glm::vec3& ambient)                          = 0;

    virtual void DrawGrid(float cellSize = 1.f, int halfExtent = 10)          = 0;

    virtual void DrawMesh(MeshHandle mesh, const glm::mat4& world,
                          const DrawSurface& surface = {})                    = 0;

    virtual void DrawSkinnedMesh(MeshHandle mesh, const glm::mat4& world,
                                 std::span<const glm::mat4> boneTransforms,
                                 const DrawSurface& surface = {})             = 0;

    // Draw a terrain tile using the dedicated multi-layer terrain shader.
    // Terrain meshes are already in world space — no model matrix needed.
    virtual void DrawTerrainTile(MeshHandle mesh, const TerrainSurface& surface) = 0;
};

// Lowercase-normalise a file path for use as a dedup key.
// Call once on the raw path before passing to GpuAssetCache::GetOrLoad.
inline std::string LowerPath(std::string_view path)
{
    std::string r(path);
    for (char& c : r) if (c >= 'A' && c <= 'Z') c += 32;
    return r;
}

// ── GpuAssetCache ─────────────────────────────────────────────────────────────
// Deduplicating key → TextureHandle cache.
// GetOrLoad() invokes the load function only on a cache miss; the result
// (even TextureHandle::Invalid for a failed load) is stored and returned on
// subsequent calls with the same key.
//
// Caller is responsible for key normalisation (use LowerPath for file paths).
// Call Free() before destroying the renderer that owns the handles.
template<typename Key = std::string>
class GpuAssetCache {
public:
    // Return the cached handle for key, or call loadFn() to produce one.
    template<typename LoadFn>
    TextureHandle GetOrLoad(const Key& key, LoadFn&& loadFn)
    {
        auto it = map_.find(key);
        if (it != map_.end()) return it->second;
        TextureHandle h = std::forward<LoadFn>(loadFn)();
        map_.emplace(key, h);
        return h;
    }

    // Release all non-Invalid handles and clear the map.
    void Free(ISceneRenderer& renderer)
    {
        for (auto& [k, h] : map_)
            if (h != TextureHandle::Invalid) renderer.FreeTexture(h);
        map_.clear();
    }

    bool        Empty() const { return map_.empty(); }
    std::size_t Size()  const { return map_.size(); }

private:
    std::unordered_map<Key, TextureHandle> map_;
};

} // namespace render
