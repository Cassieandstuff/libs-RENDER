#include "MaterialMapping.h"

namespace render::diligent {

MappedMaterial MapMaterial(const DrawSurface& s, const MaterialInputs& in) {
    using Blend = DrawSurface::BlendMode;
    MappedMaterial m;

    m.colorMap        = in.hasDiffuse;
    m.normalMap       = in.hasNormal;
    m.baseColorFactor = s.tint;

    if (in.hasRmaos) {
        m.physDescMap     = true;
        m.aoMap           = true;
        m.specSource      = MappedMaterial::SpecSource::Rmaos;
        m.metallicFactor  = 1.f;   // the map carries the values
        m.roughnessFactor = 1.f;
        m.specularFactor  = 2.f * s.specularStrength;
    } else {
        m.metallicFactor  = s.metalness;
        m.roughnessFactor = s.roughness;
        m.specularFactor  = s.specularStrength;
        if (s.specFromNormalAlpha && in.hasNormal) m.specSource = MappedMaterial::SpecSource::NormalAlpha;
        else if (in.hasSpecular)                   m.specSource = MappedMaterial::SpecSource::External;
    }

    if (in.hasGlow) {
        m.emissiveMap    = true;
        m.emissiveFactor = s.emissiveColor * s.emissiveMultiple;
    }

    m.vertexColors = s.useVertexColor && in.meshColors;
    m.joints       = in.skinned;
    m.cullBack     = !s.twoSided;

    switch (s.blendMode) {
        case Blend::Opaque: break;
        case Blend::AlphaTest:
        case Blend::AlphaTestAndBlend:
            m.alpha       = MappedMaterial::Alpha::Mask;
            m.alphaCutoff = s.alphaThreshold;
            break;
        case Blend::AlphaBlend:
        case Blend::Additive:
            m.alpha      = MappedMaterial::Alpha::Blend;
            m.depthWrite = false;
            break;
    }

    if (s.xray) {
        m.alpha              = MappedMaterial::Alpha::Blend;
        m.baseColorFactor.a  = 0.35f;
        m.depthWrite         = false;
        m.cullBack           = false;
    }

    m.wireframe = s.wireframe && in.wireframeOk;
    return m;
}

} // namespace render::diligent
