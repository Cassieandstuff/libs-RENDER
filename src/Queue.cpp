// The frame's mesh draws. DrawMesh / DrawSkinnedMesh queue; EndFrame sorts and
// draws the queue before the post pass:
//   opaque          front to back (less overdraw)
//   alpha-tested    front to back
//   blended         back to front by the mesh bounds' centre, the order Skyrim's
//                   alpha accumulator uses; a noSort draw keeps the depth of the
//                   blended draw submitted before it, so it stays right after it
// Ties keep submission order. The bone palettes are copied into the queue, so the
// caller's spans need only live through the Draw call.
#include "SceneRendererImpl.h"

#include <algorithm>

namespace render::diligent {

namespace {

int BucketOf(const DrawSurface& s) {
    using Blend = DrawSurface::BlendMode;
    if (s.xray) return 2;
    switch (s.blendMode) {
        case Blend::Opaque: return 0;
        case Blend::AlphaTest: return 1;
        default: return 2;   // AlphaBlend, Additive, AlphaTestAndBlend
    }
}

} // namespace

void DiligentSceneRenderer::Impl::Enqueue(MeshHandle handle, const glm::mat4& world, std::span<const glm::mat4> bones,
                                          const DrawSurface& surface) {
    const GpuMesh* mesh = FindMesh(handle);
    if (!mesh) return;
    QueuedDraw d;
    d.mesh       = static_cast<std::uint32_t>(handle);
    d.world      = world;
    d.boneOffset = static_cast<std::uint32_t>(queuedBones.size());
    d.boneCount  = static_cast<std::uint32_t>(bones.size());
    queuedBones.insert(queuedBones.end(), bones.begin(), bones.end());
    d.surface = surface;
    d.bucket  = BucketOf(surface);
    d.order   = static_cast<std::uint32_t>(queue.size());
    d.centre  = glm::vec3(world * glm::vec4(mesh->boundsCentre, 1.f));
    queue.push_back(d);
}

void DiligentSceneRenderer::Impl::FlushQueue() {
    if (queue.empty()) return;

    // Distance along the view direction (view space looks down -Z), with the camera in effect now.
    for (QueuedDraw& d : queue) d.depth = -(view * glm::vec4(d.centre, 1.f)).z;

    // A noSort blended draw takes the depth of the blended draw before it.
    float lastBlended = 0.f;
    bool  anyBlended  = false;
    for (QueuedDraw& d : queue) {
        if (d.bucket != 2) continue;
        if (d.surface.noSort && anyBlended) d.depth = lastBlended;
        lastBlended = d.depth;
        anyBlended  = true;
    }

    std::vector<const QueuedDraw*> order;
    order.reserve(queue.size());
    for (const QueuedDraw& d : queue) order.push_back(&d);
    std::stable_sort(order.begin(), order.end(), [](const QueuedDraw* a, const QueuedDraw* b) {
        if (a->bucket != b->bucket) return a->bucket < b->bucket;
        return a->bucket == 2 ? a->depth > b->depth : a->depth < b->depth;
    });

    // A soft effect reads the depth of what lies behind it: copy the scene depth once the opaque and tested draws are in.
    const bool soft = std::ranges::any_of(order, [](const QueuedDraw* d) {
        return d->bucket == 2 && d->surface.model == MaterialModel::Effect && d->surface.effect.softDepth > 0.f;
    });
    if (soft) {
        const Diligent::TextureDesc& dd = hdrDepth->GetDesc();
        if (!sceneDepthCopy || sceneDepthCopy->GetDesc().Width != dd.Width || sceneDepthCopy->GetDesc().Height != dd.Height) {
            Diligent::TextureDesc desc = dd;
            desc.Name      = "render scene depth copy";
            desc.BindFlags = Diligent::BIND_DEPTH_STENCIL | Diligent::BIND_SHADER_RESOURCE;
            sceneDepthCopy.Release();
            device->CreateTexture(desc, nullptr, &sceneDepthCopy);
        }
    }
    bool copied = false;
    BindSceneTargets();
    for (const QueuedDraw* d : order) {
        if (soft && !copied && d->bucket == 2 && sceneDepthCopy) {
            Diligent::CopyTextureAttribs copy{ hdrDepth, Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION, sceneDepthCopy,
                                               Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION };
            context->CopyTexture(copy);
            BindSceneTargets();
            copied = true;
        }
        const GpuMesh* mesh = FindMesh(static_cast<MeshHandle>(d->mesh));
        if (!mesh) continue;   // freed mid-frame
        const std::span<const glm::mat4> bones(queuedBones.data() + d->boneOffset, d->boneCount);
        if (d->surface.model == MaterialModel::Pbr) PbrDraw(*this, *mesh, d->world, bones, d->surface);
        else LightingDraw(*this, *mesh, d->world, bones, d->surface);   // Lighting and Effect share the pass
    }
    // A pick (asked for at any point in the frame) replays the frame's pickable draws at EndFrame, whatever flushes came
    // between, so they are kept.
    // Wireframe over shaded is drawn after the tone map, so those draws are kept too.
    auto keep = [&](const QueuedDraw& d, std::vector<QueuedDraw>& draws, std::vector<glm::mat4>& bones) {
        QueuedDraw k = d;
        k.boneOffset = static_cast<std::uint32_t>(bones.size());
        bones.insert(bones.end(), queuedBones.begin() + d.boneOffset, queuedBones.begin() + d.boneOffset + d.boneCount);
        draws.push_back(k);
    };
    for (const QueuedDraw& d : queue) {
        if (d.surface.pickId) keep(d, pickDraws, pickBones);
        if (d.surface.wireColor.a) keep(d, wireDraws, wireBones);
        if (d.surface.outlineColor.a) keep(d, outlineDraws, outlineBones);
    }
    queue.clear();
    queuedBones.clear();
}

} // namespace render::diligent
