// Mesh upload / free / CPU morph updates. One GPU buffer per attribute, matching
// the PBR input layout's one-slot-per-attribute layout (PbrPass.cpp).
#include "SceneRendererImpl.h"

#include <cstdio>

namespace render::diligent {

using namespace Diligent;

namespace {

template <typename T>
RefCntAutoPtr<IBuffer> CreateVertexBuffer(IRenderDevice* device, const char* name, const std::vector<T>& data,
                                          USAGE usage = USAGE_IMMUTABLE) {
    BufferDesc desc;
    desc.Name      = name;
    desc.Size      = data.size() * sizeof(T);
    desc.Usage     = usage;
    desc.BindFlags = BIND_VERTEX_BUFFER;
    BufferData init{ data.data(), desc.Size };
    RefCntAutoPtr<IBuffer> buf;
    device->CreateBuffer(desc, &init, &buf);
    return buf;
}

} // namespace

const GpuMesh* DiligentSceneRenderer::Impl::FindMesh(MeshHandle h) const {
    auto it = meshes.find(static_cast<std::uint32_t>(h));
    return it == meshes.end() ? nullptr : &it->second;
}

MeshHandle DiligentSceneRenderer::UploadMesh(const MeshData& data) {
    Impl& r = *m_impl;
    const std::size_t n = data.positions.size();
    if (n == 0 || data.indices.empty() || n > 0xFFFF) return MeshHandle::Invalid;

    GpuMesh m;
    m.vertexCount = static_cast<std::uint32_t>(n);
    m.indexCount  = static_cast<std::uint32_t>(data.indices.size());

    // Positions are DEFAULT usage so UpdateMeshPositions can overwrite them.
    m.positions = CreateVertexBuffer(r.device, "render mesh positions", data.positions, USAGE_DEFAULT);

    // Normals and UVs are always bound (the PBR layout reads both); fill when absent.
    if (data.normals.size() == n) m.normals = CreateVertexBuffer(r.device, "render mesh normals", data.normals);
    else m.normals = CreateVertexBuffer(r.device, "render mesh normals", std::vector<glm::vec3>(n, glm::vec3(0.f, 0.f, 1.f)));
    if (data.uvs.size() == n) m.uvs = CreateVertexBuffer(r.device, "render mesh uvs", data.uvs);
    else m.uvs = CreateVertexBuffer(r.device, "render mesh uvs", std::vector<glm::vec2>(n, glm::vec2(0.f)));

    // PBR_Renderer reads joint indices as float4.
    if (data.boneIndices.size() == n && data.boneWeights.size() == n) {
        std::vector<glm::vec4> joints(n);
        for (std::size_t i = 0; i < n; ++i) joints[i] = glm::vec4(data.boneIndices[i]);
        m.joints  = CreateVertexBuffer(r.device, "render mesh joints", joints);
        m.weights = CreateVertexBuffer(r.device, "render mesh weights", data.boneWeights);
    }

    // RGB → RGBA8 with opaque alpha (PackVertexColors reads 4 normalized bytes).
    if (data.vertexColors.size() == n) {
        std::vector<glm::u8vec4> colors(n);
        for (std::size_t i = 0; i < n; ++i) colors[i] = glm::u8vec4(data.vertexColors[i], 255);
        m.colors = CreateVertexBuffer(r.device, "render mesh colors", colors);
    }

    BufferDesc ib;
    ib.Name      = "render mesh indices";
    ib.Size      = data.indices.size() * sizeof(std::uint16_t);
    ib.Usage     = USAGE_IMMUTABLE;
    ib.BindFlags = BIND_INDEX_BUFFER;
    BufferData ibData{ data.indices.data(), ib.Size };
    r.device->CreateBuffer(ib, &ibData, &m.indices);

    if (!m.positions || !m.normals || !m.uvs || !m.indices) return MeshHandle::Invalid;

    const std::uint32_t id = r.nextMeshId++;
    r.meshes.emplace(id, std::move(m));
    return static_cast<MeshHandle>(id);
}

void DiligentSceneRenderer::FreeMesh(MeshHandle handle) {
    // In-flight GPU work keeps the buffers alive (Diligent defers the release).
    m_impl->meshes.erase(static_cast<std::uint32_t>(handle));
}

void DiligentSceneRenderer::UpdateMeshPositions(MeshHandle handle, std::span<const glm::vec3> positions) {
    Impl& r = *m_impl;
    auto it = r.meshes.find(static_cast<std::uint32_t>(handle));
    if (it == r.meshes.end()) return;
    GpuMesh& m = it->second;
    if (positions.size() != m.vertexCount) {
        std::fprintf(stderr, "[render] UpdateMeshPositions: %zu positions for a %u-vertex mesh; ignored\n",
                     positions.size(), m.vertexCount);
        return;
    }
    r.context->UpdateBuffer(m.positions, 0, positions.size_bytes(), positions.data(),
                            RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
}

} // namespace render::diligent
