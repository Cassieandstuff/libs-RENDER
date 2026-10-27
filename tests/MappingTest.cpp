// ============================================================================
// RENDER mapping test — DrawSurface → PBR material, no GPU.
// Covers every row of the table in src/MaterialMapping.h.
// ============================================================================

#include "MaterialMapping.h"

#include <cmath>
#include <cstdio>

using namespace render;
using namespace render::diligent;
using Alpha = MappedMaterial::Alpha;
using Spec  = MappedMaterial::SpecSource;
using Blend = DrawSurface::BlendMode;

namespace {

int g_failures = 0;

void check(const char* label, bool ok) {
    std::printf("[ %s ] %s\n", ok ? "PASS" : "FAIL", label);
    if (!ok) ++g_failures;
}

bool Eq(float a, float b) { return std::fabs(a - b) < 1e-6f; }

MaterialInputs All() {
    MaterialInputs in;
    in.hasDiffuse = in.hasNormal = in.hasRmaos = in.hasGlow = in.meshColors = in.skinned = in.wireframeOk = true;
    return in;
}

} // namespace

int main() {
    DrawSurface s;
    MappedMaterial m = MapMaterial(s, MaterialInputs{});
    check("default: opaque, back-face culled, depth write, no maps",
          m.alpha == Alpha::Opaque && m.cullBack && m.depthWrite && !m.wireframe && !m.colorMap && !m.normalMap &&
              !m.physDescMap && !m.aoMap && !m.emissiveMap && m.specSource == Spec::None && !m.vertexColors && !m.joints);
    check("default: metallic/roughness factors from the surface", Eq(m.metallicFactor, 0.f) && Eq(m.roughnessFactor, 0.8f));

    // Textures present → maps used.
    MaterialInputs tex;
    tex.hasDiffuse = tex.hasNormal = true;
    m = MapMaterial(s, tex);
    check("diffuse -> BASE_COLOR, normalMap -> NORMAL", m.colorMap && m.normalMap);

    s.tint = { 0.5f, 0.25f, 1.f, 0.75f };
    m      = MapMaterial(s, tex);
    check("tint -> BaseColorFactor", Eq(m.baseColorFactor.r, 0.5f) && Eq(m.baseColorFactor.g, 0.25f) && Eq(m.baseColorFactor.a, 0.75f));
    s = {};

    // Specular from the normal map's alpha (vanilla), only with a normal map.
    s.specFromNormalAlpha = true;
    s.specularStrength    = 0.7f;
    m = MapMaterial(s, tex);
    check("specFromNormalAlpha -> SPECULAR from the normal map, factor x1", m.specSource == Spec::NormalAlpha && Eq(m.specularFactor, 0.7f));
    m = MapMaterial(s, MaterialInputs{});
    check("specFromNormalAlpha without a normal map -> no specular map", m.specSource == Spec::None);
    s = {};

    // External specular mask (vanilla skin's _s.dds): only when neither the normal
    // alpha nor RMAOS applies.
    MaterialInputs ext;
    ext.hasSpecular    = true;
    s.specularStrength = 3.f;
    m = MapMaterial(s, ext);
    check("specularMap -> SPECULAR from the external mask, factor x1", m.specSource == Spec::External && Eq(m.specularFactor, 3.f));
    s.specFromNormalAlpha = true;
    ext.hasNormal         = true;
    m = MapMaterial(s, ext);
    check("specFromNormalAlpha wins over specularMap", m.specSource == Spec::NormalAlpha);
    ext.hasRmaos = true;
    m = MapMaterial(s, ext);
    check("rmaosMap wins over specularMap", m.specSource == Spec::Rmaos);
    s = {};

    // RMAOS: PHYS_DESC + OCCLUSION + SPECULAR, factors 1, specular x2.
    MaterialInputs rm;
    rm.hasRmaos          = true;
    s.specularStrength   = 0.5f;
    s.specFromNormalAlpha = true;   // ignored when RMAOS is present
    rm.hasNormal         = true;
    m = MapMaterial(s, rm);
    check("rmaosMap -> PHYS_DESC + OCCLUSION + SPECULAR(RMAOS)", m.physDescMap && m.aoMap && m.specSource == Spec::Rmaos);
    check("rmaosMap -> metallic/roughness factors 1", Eq(m.metallicFactor, 1.f) && Eq(m.roughnessFactor, 1.f));
    check("rmaosMap -> specular factor 2 x strength", Eq(m.specularFactor, 1.f));
    s = {};

    // No RMAOS → explicit factors.
    s.roughness = 0.3f;
    s.metalness = 0.9f;
    m = MapMaterial(s, MaterialInputs{});
    check("no rmaosMap -> roughness/metalness factors", Eq(m.roughnessFactor, 0.3f) && Eq(m.metallicFactor, 0.9f) && !m.physDescMap);
    s = {};

    // Glow.
    s.emissiveColor    = { 1.f, 0.5f, 0.f };
    s.emissiveMultiple = 2.f;
    MaterialInputs glow;
    glow.hasGlow = true;
    m = MapMaterial(s, glow);
    check("glowMap -> EMISSIVE x emissiveColor*multiple", m.emissiveMap && Eq(m.emissiveFactor.r, 2.f) && Eq(m.emissiveFactor.g, 1.f));
    m = MapMaterial(s, MaterialInputs{});
    check("no glowMap -> emissive factor 0", !m.emissiveMap && Eq(m.emissiveFactor.r, 0.f));
    s = {};

    // Vertex colours need both the flag and the stream.
    s.useVertexColor = true;
    MaterialInputs colors;
    colors.meshColors = true;
    check("useVertexColor + colour stream -> USE_VERTEX_COLORS", MapMaterial(s, colors).vertexColors);
    check("useVertexColor without a stream -> off", !MapMaterial(s, MaterialInputs{}).vertexColors);
    s = {};

    MaterialInputs skin;
    skin.skinned = true;
    check("skinned -> USE_JOINTS", MapMaterial(s, skin).joints);

    s.twoSided = true;
    check("twoSided -> cull none", !MapMaterial(s, MaterialInputs{}).cullBack);
    s = {};

    // Blend modes.
    s.blendMode      = Blend::AlphaTest;
    s.alphaThreshold = 0.25f;
    m = MapMaterial(s, MaterialInputs{});
    check("AlphaTest -> MASK at alphaThreshold, depth write on", m.alpha == Alpha::Mask && Eq(m.alphaCutoff, 0.25f) && m.depthWrite);
    s.blendMode = Blend::AlphaTestAndBlend;
    check("AlphaTestAndBlend -> MASK", MapMaterial(s, MaterialInputs{}).alpha == Alpha::Mask);
    s.blendMode = Blend::AlphaBlend;
    m = MapMaterial(s, MaterialInputs{});
    check("AlphaBlend -> BLEND, no depth write", m.alpha == Alpha::Blend && !m.depthWrite);
    s.blendMode = Blend::Additive;
    m = MapMaterial(s, MaterialInputs{});
    check("Additive -> BLEND approximation, no depth write", m.alpha == Alpha::Blend && !m.depthWrite);
    s = {};

    // X-ray.
    s.xray = true;
    m = MapMaterial(s, All());
    check("xray -> BLEND at 0.35, no depth write, no culling",
          m.alpha == Alpha::Blend && Eq(m.baseColorFactor.a, 0.35f) && !m.depthWrite && !m.cullBack);
    s = {};

    // Wireframe depends on device support.
    s.wireframe = true;
    check("wireframe when supported", MapMaterial(s, All()).wireframe);
    check("wireframe falls back to solid when unsupported", !MapMaterial(s, MaterialInputs{}).wireframe);

    std::printf(g_failures == 0 ? "\nALL PASSED\n" : "\n%d CHECK(S) FAILED\n", g_failures);
    return g_failures;
}
