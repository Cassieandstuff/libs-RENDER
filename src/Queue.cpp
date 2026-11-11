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

    BindSceneTargets();
    for (const QueuedDraw* d : order) {
        const GpuMesh* mesh = FindMesh(static_cast<MeshHandle>(d->mesh));
        if (!mesh) continue;   // freed mid-frame
        const std::span<const glm::mat4> bones(queuedBones.data() + d->boneOffset, d->boneCount);
        if (d->surface.model == MaterialModel::Pbr) PbrDraw(*this, *mesh, d->world, bones, d->surface);
        else LightingDraw(*this, *mesh, d->world, bones, d->surface);   // Effect falls back to Lighting for now
    }
    // A pick (asked for at any point in the frame) replays the frame's pickable draws at EndFrame, whatever flushes came
    // between, so they are kept.
    for (const QueuedDraw& d : queue) {
        if (!d.surface.pickId) continue;
        QueuedDraw keep = d;
        keep.boneOffset = static_cast<std::uint32_t>(pickBones.size());
        pickBones.insert(pickBones.end(), queuedBones.begin() + d.boneOffset, queuedBones.begin() + d.boneOffset + d.boneCount);
        pickDraws.push_back(keep);
    }
    queue.clear();
    queuedBones.clear();
}

} // namespace render::diligent
