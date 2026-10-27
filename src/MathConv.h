#pragma once
// Every convention conversion between the GL-style ISceneRenderer contract and
// Diligent lives here.
//
//   • Depth: callers build projections with glm::perspective (clip z in -w..w, GL);
//     D3D12 and Vulkan clip z to 0..w. DepthZeroToOne() remaps z' = 0.5z + 0.5w.
//   • Matrices: glm stores column-major. Our own HLSL uses the default
//     column_major packing with mul(M, v), so a glm matrix uploads as-is.
//     DiligentFX's PBR shaders use row vectors (mul(v, M)) with row_major packing
//     (PackMatrixRowMajor = true): the same raw bytes read as Mᵀ there, and
//     mul(v, Mᵀ) = (M·v)ᵀ — so a raw copy is correct for both.
//   • Winding: GL-style counter-clockwise front faces → FrontCounterClockwise = True.
//   • Y / UVs: Diligent presents top-left-origin on both backends (Vulkan's
//     viewport is flipped internally), matching DDS row 0 = v 0. No flips.
#include <glm/glm.hpp>

#include <cstring>

namespace render::diligent {

inline glm::mat4 DepthZeroToOne(const glm::mat4& glProj) {
    glm::mat4 remap(1.f);
    remap[2][2] = 0.5f;   // z' = 0.5·z …
    remap[3][2] = 0.5f;   //      … + 0.5·w
    return remap * glProj;
}

// Raw copy of a glm matrix into a 16-float destination (a Diligent float4x4 or a
// constant-buffer slot); see the header comment for why no transpose is needed.
inline void StoreMatrix(void* dst, const glm::mat4& m) { std::memcpy(dst, &m[0][0], sizeof(float) * 16); }

} // namespace render::diligent
