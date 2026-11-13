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
// tangents are optional (xyz = tangent direction along +U, w = handedness ±1:
// the bitangent w·cross(normal, tangent) runs along +V); without them a renderer
// derives the frame from the UVs.  indices (or indices32) are required; positions/normals/uvs must be
// the same length.
struct MeshData {
    std::vector<glm::vec3>   positions;
    std::vector<glm::vec3>   normals;
    std::vector<glm::vec2>   uvs;
    std::vector<glm::vec4>   tangents;      // optional: xyz=tangent, w=handedness (±1)
    std::vector<glm::u8vec4> boneIndices;   // 4 indices per vertex (uint8)
    std::vector<glm::vec4>   boneWeights;   // 4 weights per vertex, sum == 1
    std::vector<glm::u8vec3> vertexColors;  // optional RGB per vertex (0-255)
    // Optional alpha per vertex (0-255), beside vertexColors (white when those are
    // absent). Read wherever vertex colours are (DrawSurface::useVertexColor): it
    // scales the surface's alpha for alpha test and blending.
    std::vector<uint8_t>     vertexAlpha;
    std::vector<uint16_t>    indices;
    // 32-bit indices, for meshes past 65535 vertices: used when `indices` is empty.
    std::vector<uint32_t>    indices32;
};

// 8-bit RGBA, display space (as it appears on screen, like a UI colour); a = opacity.
struct Rgba8 {
    std::uint8_t r = 255, g = 255, b = 255, a = 255;
};

// ── Material models ───────────────────────────────────────────────────────────
// How a surface is shaded:
//   Pbr       metal/roughness PBR with image-based ambient (True-PBR meshes, the
//             editor's own previews). Linear colours; lit by SetLight / SetPointLights.
//   Lighting  the classic game look: Lambert diffuse, a Blinn-Phong specular lobe
//             (glossiness = exponent), directional ambient, rim / back / soft
//             light, environment maps, glow, fog (DrawSurface::lighting). It shades
//             in display (gamma) space as the game does: every colour it reads —
//             textures, vertex colours, tint, specular / emissive colours and
//             SceneEnvironment — is a display value. Lit by SetEnvironment.
//   Effect    unlit, as the game's effect shader: the diffuse (source texture) ×
//             DrawSurface::effect's base colour × vertex colour, with greyscale-to-
//             palette, view-angle falloff and a soft depth fade, and fog. Display
//             values throughout, like Lighting.
enum class MaterialModel : std::uint8_t { Pbr, Lighting, Effect };

// Blend factors and alpha-test functions, for the Lighting and Effect models'
// blended / tested surfaces (Pbr blends premultiplied "over" and tests GREATER).
enum class BlendFactor : std::uint8_t {
    One, Zero, SrcColor, InvSrcColor, DestColor, InvDestColor, SrcAlpha, InvSrcAlpha, DestAlpha, InvDestAlpha, SrcAlphaSaturate
};
enum class CompareFunc : std::uint8_t { Always, Less, Equal, LessEqual, Greater, NotEqual, GreaterEqual, Never };

// The Lighting model's parameters beside DrawSurface's shared ones (diffuse,
// normalMap, glowMap, specularMap, tint, emissive*, specularStrength).
struct LightingParams {
    glm::vec3     specularColor  = { 1.f, 1.f, 1.f };   // display value
    float         glossiness     = 80.f;               // the specular exponent
    // The normal map holds model-space normals (skin, FaceGen heads), not tangent-space ones.
    bool          modelSpaceNormals = false;
    // Emissive without a glow map: emissiveColor·emissiveMultiple everywhere (SLSF1_Own_Emit).
    bool          ownEmit        = false;
    // Environment reflection: a cube map, masked by envMask's red channel (or the
    // normal map's alpha when envMask is Invalid and envMaskFromNormalAlpha), × envScale.
    TextureHandle envMap         = TextureHandle::Invalid;   // a cube texture
    TextureHandle envMask        = TextureHandle::Invalid;
    bool          envMaskFromNormalAlpha = false;
    float         envScale       = 1.f;
    // Rim, back and soft lighting (SLSF2_Rim_Lighting / Back_Lighting / Soft_Lighting).
    // effect1 widens soft lighting's wrap; effect2 is the rim's exponent. lightingMap
    // masks rim / soft light (slot 2 when there is no glow map); backLightMap tints
    // light from behind (slot 7).
    bool          rimLight       = false;
    bool          backLight      = false;
    bool          softLight      = false;
    float         effect1        = 0.3f;
    float         effect2        = 2.f;
    TextureHandle lightingMap    = TextureHandle::Invalid;
    TextureHandle backLightMap   = TextureHandle::Invalid;
    // Parallax: offset the UVs by heightMap (red channel) along the view direction.
    TextureHandle heightMap      = TextureHandle::Invalid;
    float         parallaxScale  = 0.04f;
    // Refraction (SLSF1_Refraction: a fire's heat haze, a spell's shimmer): the surface
    // shows what lies behind it instead of its own colour, displaced in screen space by
    // its normal map's xy (tangent space, −1..1) × refractionStrength × a tenth of the
    // frame's height (an estimate: a fire's haze, strength 0.1, bends by up to 1%). Drawn
    // after every other surface (opaque, tested and blended); it writes no depth, and its
    // blend mode and other lighting are ignored.
    bool          refraction         = false;
    float         refractionStrength = 0.f;
    // (The texture coordinate transform is DrawSurface's: uvOffset / uvScale.)
};

// The Effect model's parameters (DrawSurface::effect).
struct EffectParams {
    glm::vec4     baseColor      = { 1.f, 1.f, 1.f, 1.f };   // display rgb, alpha
    float         baseColorScale = 1.f;                      // × rgb (an effect's brightness)
    // Greyscale to palette, as the game's: a 2D gradient looked up per pixel. Colour:
    // across (u) by the source's grey (green), down (v) by base colour red × vertex
    // red × baseColorScale; it replaces the colour (then × baseColorScale), and vertex
    // and base colour tint no further. Alpha: across by the source's alpha, down by
    // base alpha × vertex alpha; it replaces the alpha likewise.
    TextureHandle palette        = TextureHandle::Invalid;
    bool          paletteColor   = false;
    bool          paletteAlpha   = false;
    // Falloff: opacity by how squarely the surface faces the eye, as cosines (1 =
    // facing): startOpacity at startAngle, stopOpacity at stopAngle, mixed between.
    bool          falloff             = false;
    float         falloffStartAngle   = 1.f, falloffStopAngle = 0.f;
    float         falloffStartOpacity = 1.f, falloffStopOpacity = 0.f;
    // > 0 (blended surfaces): fades out within this distance (world units) of what
    // lies behind it, so it meets the scene softly (soft particles).
    float         softDepth      = 0.f;
    // (The texture coordinate transform is DrawSurface's: uvOffset / uvScale.)
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
    // A blended surface normally leaves depth alone. A material can ask otherwise (SLSF2_ZBuffer_Write, as Skyrim's
    // human eyes do), so nothing drawn after it paints over it.
    bool          blendWritesDepth = false;
    // False: drawn whatever lies in front of it (SLSF2_ZBuffer_Test off).
    bool          depthTest       = true;
    // Steps towards the camera, so a coplanar surface drawn over another wins the
    // depth test: 1 for decals (SLSF1_Decal / Dynamic_Decal); 0 = none.
    int           depthBias       = 0;
    // Blended surfaces are drawn back to front. noSort keeps this one in submission
    // order, right after the blended draw submitted before it (NiAlphaProperty's
    // no-sorter bit).
    bool          noSort          = false;
    // Texture coordinates, in every model: uv·uvScale + uvOffset, for every texture the
    // mesh's UVs address (an effect's palette and an environment cube aren't). Past
    // 0..1 a texture repeats; clampU / clampV hold it at its edge texels along that
    // axis instead (a NIF shader's texture clamp mode).
    glm::vec2     uvOffset        = { 0.f, 0.f };
    glm::vec2     uvScale         = { 1.f, 1.f };
    bool          clampU          = false;
    bool          clampV          = false;

    MaterialModel model           = MaterialModel::Pbr;
    // Lighting / Effect: the blended modes' factors (AlphaBlend, Additive and
    // AlphaTestAndBlend all use them) and the tested modes' function.
    BlendFactor   srcBlend        = BlendFactor::SrcAlpha;
    BlendFactor   dstBlend        = BlendFactor::InvSrcAlpha;
    CompareFunc   alphaTestFunc   = CompareFunc::Greater;
    LightingParams lighting;
    EffectParams   effect;
    // Non-zero: picks (ISceneRenderer::RequestPick) report this surface by this id.
    std::uint32_t pickId          = 0;
    // Wireframe over shaded: alpha > 0 draws this surface's edges again in this colour (a display value, exact like an
    // overlay) over the finished frame, where the surface is visible. Needs the device's wireframe fill.
    Rgba8         wireColor       = { 0, 0, 0, 0 };
    // Outline (selection): alpha > 0 paints a band around the visible part of this surface in this colour (a display
    // value), over the finished frame. The band's width is the renderer's (DiligentSceneRenderer: ViewStyle::outlineWidth).
    Rgba8         outlineColor    = { 0, 0, 0, 0 };
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

// ── Point lights ──────────────────────────────────────────────────────────────
// A light placed in the world. `color` is linear radiance at a unit's distance
// before the falloff; `radius` is where it reaches zero (world units).
// ── Image space ───────────────────────────────────────────────────────────────
// What happens between the lit frame and the screen, as a game's image space
// describes it: eye adaptation (exposure follows the frame's average brightness),
// a white point, bloom, and a cinematic grade. Off: the frame goes out as lit
// (exposure × view transform), exactly as without it.
struct ImageSpace {
    bool  enabled = false;
    // Eye adaptation. The frame's average luminance (linear, geometric mean over the
    // geometry; the background doesn't count) is adapted towards over time; exposure
    // = (adaptKey / adapted)^adaptStrength, clamped to [minExposure, maxExposure].
    // Strength 0 keeps exposure 1; 1 brings the average to adaptKey.
    float adaptStrength = 0.f;
    float adaptKey      = 0.18f;   // linear luminance the average is brought towards
    float adaptSpeed    = 10.f;    // per second: the adapted value closes 1 − e^(−speed·dt) of the gap each frame
    float minExposure   = 0.25f, maxExposure = 4.f;
    float deltaSeconds  = 0.f;     // since the last frame; 0 adapts at once (stills, tests)
    // Tone: c·(1 + c/white²)/(1 + c) per channel; white 1 is the identity (then clipped).
    float white = 1.f;
    // Bloom: what is brighter than the threshold (a display value, after exposure)
    // spreads `radius` pixels (at 1080 lines; scaled to the frame's height) and adds
    // back × scale. Scale 0 = none.
    float bloomThreshold = 1.f;
    float bloomScale     = 0.f;
    float bloomRadius    = 8.f;
    // Cinematic grade, on display values: saturation (0 = grey), brightness (×),
    // contrast (around mid-grey), then a tint towards tintColor × luminance by tintAmount.
    float     saturation = 1.f, brightness = 1.f, contrast = 1.f;
    glm::vec3 tintColor  = { 1.f, 1.f, 1.f };
    float     tintAmount = 0.f;
};

// ── Picking ───────────────────────────────────────────────────────────────────
// One surface a pick found: the nearest pixel of it inside the picked rect.
struct PickHit {
    std::uint32_t id       = 0;              // its DrawSurface::pickId
    float         depth    = 1.f;            // depth of that pixel, 0 (near plane) .. 1 (far plane)
    glm::vec3     position = { 0.f, 0.f, 0.f };   // world position of that pixel
    int           x = 0, y = 0;              // the pixel (physical pixels, top-left origin)
};

struct PointLight {
    glm::vec3 position{ 0.f };
    glm::vec3 color{ 1.f };
    float     radius = 100.f;
};

// ── Scene environment (the Lighting model's light) ───────────────────────────
// Every colour is a display (gamma-encoded) value, as the game's records store
// them; the Lighting model shades in that space.
struct SceneEnvironment {
    glm::vec3 sunDirection = { 0.f, 0.f, 1.f };   // towards the light, normalised
    glm::vec3 sunColor     = { 1.f, 1.f, 1.f };
    // Directional ambient: the colour seen by a normal along +X, -X, +Y, -Y, +Z, -Z;
    // a normal between axes mixes them by its squared components.
    glm::vec3 ambient[6]   = { glm::vec3(0.2f), glm::vec3(0.2f), glm::vec3(0.2f),
                               glm::vec3(0.2f), glm::vec3(0.25f), glm::vec3(0.15f) };
    // Placed lights. `color` is the light's colour × its fade (display value); the
    // light falls to zero at `radius`.
    std::vector<PointLight> lights;
    // Distance fog: amount = max · ((d − near) / (far − near))^power, clamped to
    // 0..max; its colour runs from nearColor to farColor over the same range.
    bool      fog      = false;
    float     fogNear  = 0.f, fogFar = 10000.f, fogPower = 1.f, fogMax = 1.f;
    glm::vec3 fogNearColor = { 0.5f, 0.5f, 0.5f };
    glm::vec3 fogFarColor  = { 0.5f, 0.5f, 0.5f };
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
    // World-space points, each a square `sizePx` physical pixels across, centred on its position. Optional, with a
    // default that draws nothing, for renderers written before it.
    virtual void DrawOverlayPoints(std::span<const OverlayVertex> points, float sizePx, OverlayDepth depth) {
        (void)points; (void)sizePx; (void)depth;
    }

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

// ── Image passes ──────────────────────────────────────────────────────────────
// Fullscreen passes from caller-supplied HLSL, rendered into persistent targets that
// later passes and DrawSurface sample, and read back as floats (a node graph's images:
// procedural textures, thumbnails, export). Reached through ISceneRenderer::Images().
//
// Everything runs at once, in call order, on the renderer's thread, outside
// BeginFrame..EndFrame; frames drawn afterwards see the results. A renderer with no
// output target bound runs them (headless use): image work ends the device's frame
// itself every few dozen steps and after each readback, so a long run of passes
// with no frame drawn never exhausts what a frame hands back. (So image calls
// mustn't fall inside another renderer's BeginFrame..EndFrame on the same device.)

enum class PassHandle : std::uint32_t { Invalid = 0 };

// A target's texel format. Float formats store what a pass writes (beyond 0..1,
// negative); RGBA8 is linear UNORM (0..1, no sRGB encoding either way).
enum class TargetFormat : std::uint8_t { R16F, RG16F, RGBA16F, R32F, RG32F, RGBA32F, RGBA8 };

struct TargetDesc {
    int          width  = 0, height = 0;   // texels, 1..16384
    TargetFormat format = TargetFormat::RGBA16F;
    int          mipLevels = 1;            // 0 = the full chain down to 1×1
};

enum class SamplerFilter  : std::uint8_t { Linear, Point };
enum class SamplerAddress : std::uint8_t { Wrap, Clamp, Mirror };

// The settings of a SamplerState the pass's shader declares, by its name.
struct PassSampler {
    std::string    name;
    SamplerFilter  filter  = SamplerFilter::Linear;
    SamplerAddress address = SamplerAddress::Wrap;
};

// A pass: one pixel shader run over a whole target level.
//
// The shader is HLSL, compiled as given: nothing is prepended to it, so the line
// numbers in compiler messages are the caller's. Its input is RENDER's fullscreen
// triangle:
//   float4 : SV_Position    the pixel
//   float2 : TEXCOORD0      uv, 0..1 across the target, origin top-left (v down):
//                           uv (0, 0) is the first texel of ReadTargets' first row
// declared in that order (Direct3D links them by position): both, or SV_Position alone.
// What it returns (SV_Target) is written to the target as it is, with no blending.
// Texture2D and SamplerState are separate objects (one sampler may serve every
// texture); constants live in cbuffers.
struct PassDesc {
    std::string  name;                     // names the pass in messages and errors
    std::string  pixelShader;              // HLSL source
    std::string  entryPoint = "main";
    TargetFormat format     = TargetFormat::RGBA16F;   // what it renders into (RunPass checks)
    // A SamplerState not listed here is Linear + Wrap; a listed name the shader doesn't
    // declare is ignored.
    std::vector<PassSampler> samplers;
};

struct PassResult {
    PassHandle  pass = PassHandle::Invalid;   // Invalid: the shader or its pipeline failed
    // Empty when the compile was clean. Otherwise it starts "pass '<name>':" on a line of
    // its own, then the compiler's own text (errors, or warnings beside a valid pass).
    // That text is in the backend compiler's format (FXC on Direct3D 12, glslang on
    // Vulkan), its line numbers counting the caller's source from 1.
    std::string messages;
};

// One constant of a pass's shader, a member of one of its cbuffers.
struct PassConstant {
    std::string name;
    bool        isInt      = false;   // int / uint / bool (SetConstant's int form)
    int         components = 1;       // a scalar 1, a vector its size, a matrix rows × columns
    int         arraySize  = 0;       // 0 = not an array
    bool        matrix     = false;   // matrices can't be set yet (SetConstant refuses them)
};

struct PassInfo {
    std::vector<PassConstant> constants;   // every cbuffer's members
    std::vector<std::string>  textures;    // the Texture2Ds it declares
    std::vector<std::string>  samplers;    // the SamplerStates it declares
};

// A target level to read back (ReadTargets).
struct TargetRead {
    TextureHandle target = TextureHandle::Invalid;
    int           mip    = 0;
};

// What ReadTargets read: four floats per texel (RGBA; half-floats widened; RGBA8 as
// 0..1; a format without G, B or A reads them as 0, 0 and 1), rows top to bottom.
struct TargetImage {
    bool               ok = false;
    int                width = 0, height = 0;
    std::vector<float> rgba;
    std::string        error;   // why not, when !ok
};

// The image-pass half of a renderer, reached through ISceneRenderer::Images().
// Runtime errors are one line naming the pass and the binding, e.g.
//   pass 'o12': texture 'o7_tex' is not bound
// and go to `error` when it isn't null.
struct IImageRenderer {
    virtual ~IImageRenderer() = default;

    // ── Targets (R6) ──
    // A target passes render into, that passes and DrawSurface sample, and that
    // ReadTargets reads; an ordinary TextureHandle, released with FreeTexture.
    // Sampled, its texels are used as stored (no sRGB decoding, floats unclamped):
    // in the Lighting and Effect models (display values) its numbers go straight in.
    // Its contents are undefined until cleared or rendered. Invalid on failure.
    virtual TextureHandle CreateTarget(const TargetDesc& desc, std::string* error = nullptr) = 0;
    // Fill level `mip` (-1 = every level) with `value`.
    virtual bool ClearTarget(TextureHandle target, const glm::vec4& value, int mip = -1, std::string* error = nullptr) = 0;
    // Levels 1.. from level 0, each texel the average of its 2×2 parent texels.
    virtual bool GenerateMips(TextureHandle target, std::string* error = nullptr) = 0;

    // ── Passes (R5) ──
    // Compiles the shader and builds its pipeline, at once (blocking). A pass lives
    // until FreePass or the renderer's end. It keeps its constant values and texture
    // bindings (by handle; it holds no texture alive: a freed one fails the next run).
    virtual PassResult CreatePass(const PassDesc& desc) = 0;
    virtual void       FreePass(PassHandle pass) = 0;
    virtual PassInfo   Describe(PassHandle pass) const = 0;
    // Set a constant by name (any of the shader's cbuffers): components × max(arraySize, 1)
    // values, element after element. RENDER packs them by HLSL's rules (an array's
    // elements on 16-byte strides). Kept until set again; unset constants are 0.
    // False for an unknown name, a wrong count, a float / int mismatch or a matrix.
    virtual bool SetConstant(PassHandle pass, std::string_view name, std::span<const float> values,
                             std::string* error = nullptr) = 0;
    virtual bool SetConstant(PassHandle pass, std::string_view name, std::span<const std::int32_t> values,
                             std::string* error = nullptr) = 0;
    // Bind a target or any other texture to the Texture2D `name`.
    virtual bool SetTexture(PassHandle pass, std::string_view name, TextureHandle texture,
                            std::string* error = nullptr) = 0;
    // Render the pass over all of `target`'s level `mip`. False when a handle is
    // invalid, the target's format isn't the pass's, a Texture2D the shader declares is
    // unbound or freed, or the target is also one of the pass's inputs (iterate
    // through two targets instead).
    virtual bool RunPass(PassHandle pass, TextureHandle target, int mip = 0, std::string* error = nullptr) = 0;

    // ── Readback (R9) ──
    // Read target levels back to the CPU, one result per request in order, waiting for
    // the GPU once for them all. A request for a texture that isn't a target, or a
    // level it doesn't have, fails alone.
    virtual std::vector<TargetImage> ReadTargets(std::span<const TargetRead> reads) = 0;
    // One level (ReadTargets with one request).
    TargetImage ReadTarget(TextureHandle target, int mip = 0) {
        const TargetRead read{ target, mip };
        std::vector<TargetImage> images = ReadTargets(std::span<const TargetRead>(&read, 1));
        return images.empty() ? TargetImage{} : std::move(images.front());
    }
};

// ── ISceneRenderer ────────────────────────────────────────────────────────────
// Scene-level rendering abstraction.  Implementations are drop-in: they differ
// only in backend machinery; the viewport only calls this.
//
// Frame sequence:
//   BeginFrame(w, h)          — bind / resize the render target, clear
//   SetCamera(view, proj)     — store matrices for this frame
//   DrawGrid / DrawMesh / DrawSkinnedMesh  (any order)
//   EndFrame()                — draw what was queued, resolve the frame
//
// Draw order. Mesh draws are queued and drawn when the queue is flushed, in an
// order of the renderer's choosing: opaque surfaces, then alpha-tested ones, then
// blended ones back to front by their bounds' centre (see DrawSurface::noSort).
// The queue flushes at EndFrame, or earlier at FlushScene(). EndFrame then runs,
// in this order: the flush, the tone map into the output, the grid, the overlays
// (IOverlayRenderer, against the finished scene depth). So an overlay call may come
// before or after the mesh draws it covers, and a readback of the output (or of
// anything the scene pass writes) sees the whole frame only after EndFrame.
// The camera and lights in effect at a flush light what it draws.
struct ISceneRenderer {
    virtual ~ISceneRenderer() = default;

    // The renderer's overlay interface, or null when it has none. Optional, with a
    // default, so renderers written before it (SCT's GL backend) build unchanged.
    virtual IOverlayRenderer* Overlays() { return nullptr; }
    // The renderer's image passes (targets, passes from caller HLSL, readback), or
    // null when it has none. Optional, with a default, like Overlays().
    virtual IImageRenderer* Images() { return nullptr; }

    // ── Frame lifecycle ───────────────────────────────────────────────────────
    virtual void BeginFrame(int width, int height)                            = 0;
    virtual void SetCamera(const glm::mat4& view, const glm::mat4& proj)     = 0;
    virtual void EndFrame()                                                   = 0;
    // Draw what is queued now, into the scene. Later draws sort among themselves
    // only and land over what was flushed (depth-tested against it): a layer the
    // caller wants drawn after everything so far, whatever it is. Optional, with a
    // default that does nothing, for renderers that draw at once.
    virtual void FlushScene() {}

    // ── Picking ───────────────────────────────────────────────────────────────
    // Ask this frame which pickable surfaces (DrawSurface::pickId) cover the pixels
    // [x0, x1) × [y0, y1) (physical pixels, top-left origin; a point is a 1×1 rect).
    // Call between BeginFrame and EndFrame; the surfaces are found in depth order, as
    // the frame draws them (overlays don't count). TakePicks after EndFrame returns
    // one hit per surface, nearest first, and waits for the GPU; empty when no pick
    // was asked for or nothing pickable covers the rect. Optional, with defaults.
    virtual void RequestPick(int x0, int y0, int x1, int y1) { (void)x0; (void)y0; (void)x1; (void)y1; }
    virtual std::vector<PickHit> TakePicks() { return {}; }

    // ── GPU resource management ───────────────────────────────────────────────
    virtual MeshHandle    UploadMesh  (const MeshData& data)                  = 0;
    virtual void          FreeMesh   (MeshHandle handle)                      = 0;
    // Replace an uploaded mesh's contents under the same handle: any attribute, the
    // indices, the vertex count. False when the handle is unknown or `data` would
    // not upload (UploadMesh's rules); the old contents then stay. Draws queued
    // this frame draw the new contents. Optional, with a default that refuses.
    virtual bool          UpdateMesh (MeshHandle handle, const MeshData& data) { (void)handle; (void)data; return false; }
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

    // The point lights this frame, beside SetLight's directional one. A renderer
    // keeps as many as it can and may ignore the rest; call it before draw calls.
    // Optional, with a default, so renderers written before it build unchanged.
    virtual void SetPointLights(std::span<const PointLight> lights)           { (void)lights; }

    // The Lighting model's light for this frame (see SceneEnvironment). Optional,
    // with a default, so renderers without the model build unchanged.
    virtual void SetEnvironment(const SceneEnvironment& environment)          { (void)environment; }

    // The frame's image space (see ImageSpace); set before EndFrame. Optional, with a
    // default, so renderers without it build unchanged.
    virtual void SetImageSpace(const ImageSpace& imageSpace)                  { (void)imageSpace; }

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
