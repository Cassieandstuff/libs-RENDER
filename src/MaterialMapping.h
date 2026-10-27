#pragma once
// DrawSurface → PBR_Renderer material, as a pure function (no Diligent types) so
// the mapping table is unit-tested without a GPU (tests/MappingTest.cpp).
//
//   diffuse              BASE_COLOR (sRGB view)
//   tint                 BaseColorFactor
//   normalMap            NORMAL
//   specFromNormalAlpha  the normal map also bound to SPECULAR (reads .a)
//   specularMap          a red→alpha view bound to SPECULAR, when neither of the
//                        other two sources applies (needs swizzle support)
//   rmaosMap             one swizzled view on PHYS_DESC, OCCLUSION and SPECULAR
//   no rmaosMap          Metallic/RoughnessFactor from the surface
//   specularStrength     Specular.Factor (×2 for RMAOS: its A = 0.5 is neutral, and
//                        PBR's dielectric F0 is 0.04·factor·A — GL's 0.08·A)
//   glowMap              EMISSIVE × emissiveColor·emissiveMultiple (0 without a map)
//   useVertexColor       USE_VERTEX_COLORS (when the mesh has colours)
//   twoSided             cull none
//   AlphaTest(+Blend)    MASK at alphaThreshold
//   AlphaBlend/Additive  BLEND (premultiplied) without depth write; Additive is
//                        approximated by BLEND
//   xray                 BLEND at alpha 0.35, no depth write, no culling
//   wireframe            the wireframe pipeline, when the device supports it
#include <render/Render.h>

#include <glm/glm.hpp>

namespace render::diligent {

struct MaterialInputs {
    bool hasDiffuse   = false;
    bool hasNormal    = false;
    bool hasRmaos     = false;   // valid handle AND a swizzled view is available
    bool hasSpecular  = false;   // specularMap: valid handle AND a swizzled view is available
    bool hasGlow      = false;
    bool meshColors   = false;   // the mesh has a vertex-colour stream
    bool skinned      = false;   // DrawSkinnedMesh with a non-empty palette and joint streams
    bool wireframeOk  = false;   // FILL_MODE_WIREFRAME granted
};

struct MappedMaterial {
    enum class Alpha { Opaque, Mask, Blend };
    enum class SpecSource { None, NormalAlpha, Rmaos, External };

    Alpha      alpha      = Alpha::Opaque;
    bool       cullBack   = true;
    bool       depthWrite = true;
    bool       wireframe  = false;

    bool       colorMap    = false;
    bool       normalMap   = false;
    bool       physDescMap = false;   // RMAOS view
    bool       aoMap       = false;   // RMAOS view
    bool       emissiveMap = false;
    SpecSource specSource  = SpecSource::None;
    bool       vertexColors = false;
    bool       joints       = false;

    glm::vec4 baseColorFactor{ 1.f };
    glm::vec3 emissiveFactor{ 0.f };
    float     metallicFactor  = 0.f;
    float     roughnessFactor = 1.f;
    float     specularFactor  = 1.f;
    float     alphaCutoff     = 0.5f;
};

MappedMaterial MapMaterial(const DrawSurface& surface, const MaterialInputs& in);

} // namespace render::diligent
