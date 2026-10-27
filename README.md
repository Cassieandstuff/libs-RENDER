# RENDER

A backend-neutral scene renderer interface, `render::` (`<render/Render.h>`), and its
[Diligent Engine](https://github.com/DiligentGraphics) backend, `render::diligent`
(`<render/Diligent.h>`): `ISceneRenderer` plus the device / swap-chain / off-screen
plumbing a host needs to put it on screen. Vulkan or Direct3D 12. C++23.

| | |
|---|---|
| Link | `render::api` (header-only; glm) or `render::diligent` (the backend; Diligent Core / Tools / FX, private) |
| Headers | `<render/Render.h>`, `<render/Diligent.h>` |

```cmake
FetchContent_Declare(RENDER SOURCE_DIR D:/Projects/Common/internal/RENDER)   # or a git repository + tag
FetchContent_MakeAvailable(RENDER)
target_link_libraries(MyTool PRIVATE render::diligent)
```

## The interface

`<render/Render.h>` holds `ISceneRenderer` (frame lifecycle, mesh / texture upload, face-tint composite, grid /
mesh / skinned-mesh / terrain draws), the optional overlay interface `IOverlayRenderer`, the mesh and surface
descriptors, typed handles, and the deduplicating `GpuAssetCache`. No graphics API: asset code and an editor's logic
layer build against it alone.

**Overlays.** `IOverlayRenderer` draws world-space lines (with a pixel width) and triangles, each batch depth-tested or on
top; screen-space lines, triangles and text (with anchors and an optional pill); `MeasureText`; and octahedral bones
(`DrawBones`, styled by `SetSkeletonStyle`). Colours are display (sRGB) values, sizes physical pixels. A renderer offers it
through `ISceneRenderer::Overlays()`, which defaults to `nullptr` (callers check for null).

## The Diligent backend

No Diligent type crosses the public header (pimpl throughout), so consumers need
neither Diligent's include paths nor its macros.

## API

| Class | What |
|---|---|
| `RenderDevice` | Device + immediate context. `Create({preferred, allowFallback, validation, useWarp})` tries the preferred backend, then the other. `FinishFrame()` for frames that present nothing, `WaitIdle()` for teardown. |
| `ViewportSurface` | A swap chain on a native window (HWND): RGBA8 sRGB, no depth. `Resize`, `Present(sync)`. |
| `OffscreenTarget` | RGBA8 sRGB colour target with `ReadbackRGBA8()` (tests, thumbnails). |
| `DiligentSceneRenderer` | `ISceneRenderer` and `IOverlayRenderer` (see [Overlays](#overlays)), plus `SetOutputTarget(surface or offscreen)`, `SetExposure`, `SetViewStyle` (background, grid and axis colours), `LastFrameStats()`, `DrawLines(segments, widthPx)`, `SetGlyphAtlas` (the host-baked overlay font) and `DebugReadbackRGBA8` for face-tint results. |

Destroy in reverse order: surfaces and renderers before their `RenderDevice`.

## Frame model

Same shape as SCT's GL renderer:

- **`BeginFrame(w, h)`** sizes an RGBA16F colour target plus D32 depth, and clears it to transparent.
- **Draws are immediate**, in the order the caller issues them.
- **Overlays** (`DrawLines`, `IOverlayRenderer`) are batched and drawn at `EndFrame`, after the grid; see below.
- **`EndFrame`** tone-maps into the bound output: exposure, then ACES (Narkowicz fit), composited over the `ViewStyle` background. The background is in display values, so exposure and the tone curve don't touch it. Output to an sRGB target is written linear and the hardware encodes it; any other format gets `pow(1/2.2)`.
- **`DrawGrid`** is an infinite ground grid (after Dance Card Creator): a fullscreen pass at `EndFrame`, over the tone-mapped image, that ray-casts the Z = 0 plane. It draws anti-aliased minor/major lines and red/green axes, fading with distance. It is depth-tested against the scene but writes no depth, so geometry below the plane still shows.

`CompositeFaceTint` may run inside a frame; it re-binds the scene targets after.

## Face / skin tint

`CompositeFaceTint(base, layers)` renders the layers, in order, onto a copy of `base` (RGBA8 sRGB, full mip chain) and returns it as a new texture. Each layer has a coverage (mask luminance or alpha × `color.a`; no mask = full) and a blend:

| Blend | Result (k = coverage) | Space |
|---|---|---|
| `Overlay` | lerp(current, tint, k) | linear (GL) |
| `Multiply` | current × lerp(1, 2·tint, k) — modulate 2x | linear (GL) |
| `SoftLight` | lerp(current, b² + 2·T·b·(1 − b), k), b = the **original base**, T = `color.rgb` or the `tintMap` texel | sRGB-encoded (Skyrim shades in gamma space) |
| `Detail` | lerp(current, current × 3.984375·(D + (1, 0, 1)/255), k), D = the `tintMap` texel; skipped without one | sRGB-encoded |

`SoftLight` + `Detail` are Skyrim's FaceGen (shader type 4: facetint map, then detail map) and SkinTint (type 5: a constant skin tone) shading. Soft light is linear in T, so a run of `SoftLight` layers equals soft light by those layers painted over neutral grey — how the game builds a face tint texture from tint masks.

## Overlays

`IOverlayRenderer` (declared in `<render/Render.h>`, reached through `ISceneRenderer::Overlays()`) is what the Scene Editor's viewport overlays — skeletons, actor placeholders, paths, camera frusta, gizmos, labels — are built from. Its colours are **display values** (sRGB, like UI colours) and its widths and screen positions **physical pixels**. Everything is drawn at `EndFrame` straight into the output, after the tone map and the grid, so colours are exact (an sRGB output gets the colour decoded in the shader and re-encoded by the hardware). Three layers, each in call order:

| Layer | Calls | Depth |
|---|---|---|
| Tested | `DrawOverlayLines` / `DrawOverlayTriangles` with `OverlayDepth::Tested`, `DrawLines` | Tested against the scene's depth, no write: hidden behind geometry. |
| On top | the same with `OverlayDepth::OnTop` | The scene depth is cleared first, then read and written: drawn over the scene, but on-top primitives still occlude each other (a solid bone shows its front faces). |
| Screen | `DrawScreenLines`, `DrawScreenTriangles`, `DrawScreenText` | None; pixel coordinates, origin top-left. Drawn last. |

- **Lines** are screen-aligned quads `widthPx` wide (instanced, expanded in the vertex shader after clipping the segment to the near plane), with square caps and a one-pixel anti-aliased fringe. A line centred on a pixel boundary covers exactly `widthPx` pixels across.
- **Triangles** are flat and unculled.
- **Text.** The renderer has no rasteriser: the host bakes a `GlyphAtlas` (R8 coverage or RGBA8, per-glyph rect / size / bearing / advance, line height, ascent, a fallback code point) and hands it to `SetGlyphAtlas`. `DrawScreenText(pos, utf8, rgba, anchor, pill)` lays out UTF-8 by advances (no kerning, `'\n'` breaks lines), anchored left / centre / right × top / middle / baseline, snapped to whole pixels so an atlas baked at the display's density is sampled 1:1. An optional `TextPill` draws a rounded box behind it. `MeasureText` returns the box. The Scene Editor bakes Roboto Medium with Qt (`src/ui/OverlayFont.cpp`).
- **Bones.** `DrawBones(joints, parents, world)` draws Blender-style octahedral bones after Dance Card Creator's `AppendSkeleton`: a bone from each joint's parent to the joint, flat-shaded from a light at the eye; a root gets a ground ring and a +Y facing tick, and root → child offsets are not drawn. `SetSkeletonStyle` sets the colours (defaults from v2's overlay: bone (150, 200, 255), selected (255, 210, 90), hovered white, root (230, 180, 70)), the selected and hovered bone, the width ratio and clamps, a per-bone visibility mask, and the depth mode — `OnTop` by default, as v2 drew skeletons over the meshes.

The CPU half (UTF-8, layout, pills, bone geometry) is pure C++ in [src/OverlayGeometry.h](src/OverlayGeometry.h).

**Meshes** shade through DiligentFX's `PBR_Renderer`:

- Its shaders, signature and PSO caches; our own constant buffers, render targets and draw calls.
- IBL comes from a procedural Z-up sky/ground cube (GL's probe), scaled by `SetLight`'s ambient.
- One directional light, with intensity = colour.
- Pipelines compile asynchronously. A draw whose pipeline isn't ready yet is skipped and counted in `FrameStats::skippedCompiling`.
- Common permutations are prewarmed.

**Material mapping.** `DrawSurface` → PBR flags and material is a pure function: see the table in [src/MaterialMapping.h](src/MaterialMapping.h), unit-tested by `tests/MappingTest.cpp`.

## Texture formats

`LoadTexture*` inspects a DDS header before Diligent sees it ([src/DdsDecode.h](src/DdsDecode.h), pure C++):

| DDS pixel format | Path |
|---|---|
| FourCC DXT1–5, ATI1/ATI2, BC4/BC5, D3DFMT float codes; any DX10 header (2D); 32-bit RGBA8 / BGRA8; R10G10B10A2, R16G16, R32F, L16 | Diligent's loader, as before |
| 24-bit RGB/BGR, X8R8G8B8 / X8B8G8R8, R5G6B5, A1R5G5B5 / X1R5G5B5, A4R4G4B4 / X4R4G4B4, R3G3B2, L8, A8L8, A4L4, A8 (any contiguous masks of ≤ 8 bits in an 8/16/24/32-bit pixel) | Decoded on the CPU to RGBA8: luminance → (L, L, L, A), alpha-only → (0, 0, 0, A), no alpha mask → A = 255. The file's mips are kept and the rest of the chain is box-filtered (colour in linear light for sRGB). Uploaded as `RGBA8_UNORM[_SRGB]` per `TexColorSpace`. |
| Volume / 1D textures, uncompressed cube maps in a decoded format, palettes, YUV, unknown FourCCs, truncated data | `Invalid` |

Why decode the layouts Diligent accepts: Vulkan has no BGRX format, so Diligent's B8G8R8X8 reads the pad byte as alpha there; the 16-bit packed formats have no sRGB variant and are optional on some devices; L8 / A8L8 would sample as (L, 0, 0) / (L, A, 0).

Vanilla Skyrim SE (all 32,920 `.dds` in the Data folder's BSAs) uses eight formats: DXT1, DXT3, DXT5, BGRA8, RGBA8 (Diligent); 24-bit BGR — the 188 face-tint masks — and X8R8G8B8 — 9,326 terrain LOD tiles (decoded); L8 — one volume texture, `effects\noisevolume.dds` (rejected).

A failed load prints one line, `[render] texture load failed: <why> (<pixel format>)`; Diligent's own messages for it are captured and folded into that line.

## Conventions

The `ISceneRenderer` contract is GL-style: Z-up, right-handed, CCW front faces, projections with NDC z in −1..1. All conversions live in [src/MathConv.h](src/MathConv.h):

- **Depth.** `SetCamera` remaps the projection to 0..1 depth.
- **Matrices.** glm matrices upload as raw bytes: `mul(M, v)` in our shaders, and row-major packing for DiligentFX.
- **Winding and orientation.** `FrontCounterClockwise`, and no V-flips.

## Deviations from the GL renderer

| | GL (SCT) | Diligent (here) |
|---|---|---|
| **Mesh shading** | Own PBR shader | DiligentFX `PBR_Renderer`. The look differs slightly. |
| **Normal maps** | Mesh tangents | Derivative TBN; mesh tangents are ignored. The green channel reads as "up the image" — confirm against Skyrim assets. |
| **RMAOS** | Sampled per channel | One swizzled view, bound to physical-descriptor, occlusion and specular. Without device swizzle support, the scalar factors are used instead. |
| **External specular mask** | Not supported | `DrawSurface::specularMap` (vanilla skin's `_s.dds`): a red → alpha swizzled view bound to specular, scaling `specularStrength`. Without swizzle support it is ignored. |
| **Face tint blends** | Overlay, modulate 2x | Also `SoftLight` and `Detail` (Skyrim's FaceGen / SkinTint shading; see [Face / skin tint](#face--skin-tint)) |
| **Additive blend** | Alpha blend | Premultiplied alpha blend (`BLEND`) |
| **Output gamma** | `pow(1/2.2)` | Exact sRGB on sRGB outputs |
| **Background** | Linear HDR clear (0.10, 0.11, 0.14), tone-mapped | Display-space `ViewStyle` background (0.11, 0.115, 0.125), composited after the tone map |
| **Grid** | 2·halfExtent line segments in the scene pass | Infinite, anti-aliased, distance-faded fullscreen grid; `cellSize·halfExtent` is the fade distance |
| **Bone palette** | Unbounded (SSBO) | 256 joints (uniform buffer); larger palettes are clamped, and a message is logged once |
| **DDS formats** | BC1/3/4/5/7 and 32-bit RGBA/BGRA only (the 24-bit tint masks were rejected there too) | Everything Diligent reads, plus legacy uncompressed layouts CPU-decoded to RGBA8 (see [Texture formats](#texture-formats)) |
| **Lines** | `DrawLines` in the HDR scene pass, 1 px GL lines, tone-mapped | Overlay quads of any pixel width after the tone map: display colours, exact |
| **Skeletons** | `DrawSkeleton`: stick bones + joint points | `IOverlayRenderer::DrawBones`: octahedral bones (see [Overlays](#overlays)) |
| **`DrawTerrainTile`** | Implemented | Not implemented yet; a message is logged once |

## Tests

| Test | GPU | Covers |
|---|---|---|
| `render-mapping-tests` | No | The material mapping |
| `render-dds-tests` | No | DDS header routing; the legacy decoder's channel widening, byte order and mip chain (stored, missing, sRGB-aware box filter) |
| `render-overlay-tests` | No | The overlays' CPU half: UTF-8 decoding, text measure / layout / anchors / fallback / pixel snapping, pills, skeleton geometry (root offsets skipped, visibility mask, selection, placement) |
| `render-gpu-tests --backend vulkan\|d3d12 [--warp] [--validation]` | Yes, CTest label `gpu` | Off-screen rendering with readback checks: clear round trip, empty frame = sRGB(ACES(clear)), grid axis colours and orientation, resources and handle hygiene, PBR (lighting, culling, depth, blend, skinning, morphs, wireframe, async compile), normal maps, face tint (including mid-frame), skin tint (soft light by a tone and by a tint map, strength, layer composition, detail map; the tinted diffuse rendered on a lit quad matches a plain texture of the expected colour; external specular mask), legacy DDS formats (per-texel readback, as tint masks, clean failures), overlays (exact colours, line widths and caps, screen placement, tested vs on-top against a mesh, layer order, `DrawLines`, text glyph placement / anchors / pill / fallback / RGBA8 atlas, bones: selected colour, visibility mask, depth modes) |

Run the GPU tests once per backend; `ctest -LE gpu` skips them. `render-gpu-tests --backend … --load-dds <dir>` skips the suite and loads every `.dds` in a folder instead (in both colour spaces), for checking real game files.

## Build

```bash
cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake
cmake --build build --config Release
ctest --test-dir build -C Release            # -LE gpu skips the GPU tests
```

Standalone, `vcpkg.json` supplies glm, and Diligent is fetched at the pins in `CMakeLists.txt`. Diligent's configure
step runs Python 3 and installs libclang and jinja2 with pip; a venv in the build tree keeps that out of the system
Python. A consumer that already builds Diligent (a `DiligentFX` target) shares its copy. `-DRENDER_DILIGENT=OFF` builds
the interface alone.

Moved out of the Scene Editor (`libs/sct-render-api`, `libs/sct-render-diligent`), itself ported from SCT's
`sct-renderer` GL backend.
