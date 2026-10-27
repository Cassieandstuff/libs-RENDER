#pragma once
// DiligentSceneRenderer's private state, shared by its translation units:
//   SceneRenderer.cpp  frame lifecycle, output binding, post pass, grid
//   Overlay.cpp        overlay batches (lines, triangles, text, bones), drawn after the grid
//   Meshes.cpp         mesh upload / free / morph updates
//   Textures.cpp       texture loading, blend maps, views, readback
//   PbrPass.cpp        DiligentFX PBR_Renderer: IBL, PSO caches, SRBs, mesh draws
//   FaceTintPass.cpp   the face-tint composite
#include "Internal.h"

#include "Buffer.h"
#include "PipelineState.h"
#include "ShaderResourceBinding.h"
#include "TextureView.h"

#include <glm/glm.hpp>

#include <cstdint>
#include <map>
#include <memory>
#include <span>
#include <unordered_map>
#include <vector>

namespace render::diligent {

using Diligent::RefCntAutoPtr;

// ── GPU resources behind the public handles ─────────────────────────────────
struct GpuMesh {
    // One buffer per attribute (PBR input slots 0..5); null = absent.
    RefCntAutoPtr<Diligent::IBuffer> positions, normals, uvs, joints, weights, colors;
    RefCntAutoPtr<Diligent::IBuffer> indices;   // u16
    std::uint32_t vertexCount = 0;
    std::uint32_t indexCount  = 0;
};

struct GpuTexture {
    RefCntAutoPtr<Diligent::ITexture>     texture;
    RefCntAutoPtr<Diligent::ITextureView> arraySrv;     // TEX_2D_ARRAY view (what PBR shaders sample)
    RefCntAutoPtr<Diligent::ITextureView> rmaosSrv;     // swizzled RMAOS view, created on first use
    RefCntAutoPtr<Diligent::ITextureView> specSrv;      // red-to-alpha view of a specular mask, on first use
    int  width = 0, height = 0;
    bool srgb = false;
    bool rendered = false;   // produced on the GPU (face tint) — readable via DebugReadbackRGBA8
};

struct PbrState;       // PbrPass.cpp
struct FaceTintState;  // FaceTintPass.cpp
struct OverlayState;   // Overlay.cpp
void DestroyPbrState(PbrState* p);
void DestroyFaceTintState(FaceTintState* t);
void DestroyOverlayState(OverlayState* o);
// Owning pointers to the pass states, deleted in their own files.
using PbrPtr      = std::unique_ptr<PbrState, decltype(&DestroyPbrState)>;
using FaceTintPtr = std::unique_ptr<FaceTintState, decltype(&DestroyFaceTintState)>;
using OverlayPtr  = std::unique_ptr<OverlayState, decltype(&DestroyOverlayState)>;

// The scene pass's depth format (the grid and the world overlays test against it).
inline constexpr Diligent::TEXTURE_FORMAT kSceneDepthFormat = Diligent::TEX_FORMAT_D32_FLOAT;

struct DiligentSceneRenderer::Impl {
    explicit Impl(RenderDevice& device);
    ~Impl();

    RenderDevice&             owner;
    Diligent::IRenderDevice*  device  = nullptr;
    Diligent::IDeviceContext* context = nullptr;

    // ── Output ──
    ViewportSurface* outSurface   = nullptr;
    OffscreenTarget* outOffscreen = nullptr;
    bool             warnedNoOutput = false;
    Diligent::ITextureView* OutputRTV() const;

    // ── Scene (HDR) target ──
    RefCntAutoPtr<Diligent::ITexture> hdrColor, hdrDepth;
    int  width = 0, height = 0;
    bool inFrame = false;
    void EnsureSceneTargets(int w, int h);
    void BindSceneTargets();   // (re)bind HDR colour + depth and the viewport

    // ── Camera / light ──
    glm::mat4 view{ 1.f }, proj{ 1.f }, projZO{ 1.f }, viewProj{ 1.f };
    glm::vec3 cameraPos{ 0.f };
    glm::vec3 lightDir{ 0.371f, 0.928f, 0.0f };   // towards the light
    glm::vec3 lightColor{ 1.0f, 0.98f, 0.92f };
    glm::vec3 ambient{ 0.12f, 0.13f, 0.18f };
    float     exposure = 1.f;

    // ── Post (tone map over the background into the output) ──
    struct PostPipeline {
        RefCntAutoPtr<Diligent::IPipelineState>         pso;
        RefCntAutoPtr<Diligent::IShaderResourceBinding> srb;
    };
    std::map<int, PostPipeline>      postPipelines;   // keyed by output TEXTURE_FORMAT
    RefCntAutoPtr<Diligent::IBuffer> postCB;
    PostPipeline& PostFor(Diligent::TEXTURE_FORMAT format);
    void RunPost(Diligent::ITextureView* outputRTV);

    // ── Infinite grid (over the output, depth-tested against the scene) ──
    ViewStyle                        style;
    bool                             gridRequested = false;
    float                            gridSpacing = 1.f, gridFade = 10.f;
    bool                             warnedGridSize = false;
    std::map<int, PostPipeline>      gridPipelines;   // keyed by output TEXTURE_FORMAT
    RefCntAutoPtr<Diligent::IBuffer> gridCB;
    PostPipeline& GridFor(Diligent::TEXTURE_FORMAT format);
    void RunGrid(Diligent::ITextureView* outputRTV);

    // ── Resources ──
    std::uint32_t nextMeshId = 1, nextTextureId = 1;
    std::unordered_map<std::uint32_t, GpuMesh>    meshes;
    std::unordered_map<std::uint32_t, GpuTexture> textures;
    int failedTextureLoads = 0;
    const GpuMesh*    FindMesh(MeshHandle h) const;
    GpuTexture*       FindTexture(TextureHandle h);
    TextureHandle     RegisterTexture(GpuTexture tex);
    Diligent::ITextureView* RmaosView(GpuTexture& tex);      // lazily created swizzle view
    Diligent::ITextureView* SpecularView(GpuTexture& tex);   // lazily created: .a = .r

    // ── Passes ──
    PbrPtr      pbr{ nullptr, &DestroyPbrState };
    FaceTintPtr tint{ nullptr, &DestroyFaceTintState };
    OverlayPtr  overlay{ nullptr, &DestroyOverlayState };

    FrameStats stats, lastStats;

    // Shader helper: compile HLSL source with combined texture samplers.
    RefCntAutoPtr<Diligent::IShader> CompileShader(const char* name, Diligent::SHADER_TYPE type, const char* source,
                                                   const Diligent::ShaderMacro* macros = nullptr, Diligent::Uint32 numMacros = 0);
};

// Pass lifecycles / hooks implemented in their own files.
PbrPtr CreatePbrState(DiligentSceneRenderer::Impl& r);
void PbrBeginFrame(DiligentSceneRenderer::Impl& r);   // frame constants (lazily refreshed before the first draw)
void PbrInvalidateTexture(DiligentSceneRenderer::Impl& r, std::uint32_t textureId);
void PbrDraw(DiligentSceneRenderer::Impl& r, const GpuMesh& mesh, const glm::mat4& world,
             std::span<const glm::mat4> bones, const DrawSurface& surface);

FaceTintPtr CreateFaceTintState(DiligentSceneRenderer::Impl& r);
TextureHandle FaceTintComposite(DiligentSceneRenderer::Impl& r, TextureHandle base, std::span<const FaceTintLayer> layers);

OverlayPtr CreateOverlayState(DiligentSceneRenderer::Impl& r);
void OverlayBeginFrame(DiligentSceneRenderer::Impl& r);                                // drop last frame's batches
void OverlayRun(DiligentSceneRenderer::Impl& r, Diligent::ITextureView* outputRTV);   // EndFrame, after the grid

} // namespace render::diligent
