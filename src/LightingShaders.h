#pragma once
// HLSL for the Lighting material model (LightingPass.cpp): the classic game look.
// One source, compiled per feature set (the macros below, 0 or 1). Matrices are
// glm column-major, uploaded raw: mul(M, v).
//
// It shades in display (gamma) space, as the game does: sRGB textures are read
// back to their stored values, vertex colours and every constant are display
// values, and the result is decoded to linear only on output (the HDR target is
// linear; the post pass encodes it again).
//
//   SKINNED                 bone palette (LightingBones), joints / weights streams
//   VERTEX_COLORS           the colour stream multiplies diffuse (rgb) and alpha
//   NORMAL_MAP              g_Normal, tangent space: the mesh's tangents with VERTEX_TANGENTS,
//                           else a cotangent frame from the UVs' screen derivatives
//   MODEL_SPACE_NORMALS     g_Normal holds model-space normals instead
//   SPEC_NORMAL_ALPHA       the specular mask is g_Normal's alpha
//   SPEC_MAP                the specular mask is g_Specular's red
//   GLOW_MAP                emissive = g_Glow · emissive colour
//   OWN_EMIT                emissive = emissive colour (no map)
//   ENV_MAP                 g_Env reflection; masked by ENV_MASK (g_EnvMask.r) or
//                           ENV_MASK_NORMAL_ALPHA (g_Normal.a), else unmasked
//   RIM_LIGHT, SOFT_LIGHT   masked by g_LightingMap when LIGHTING_MAP
//   BACK_LIGHT              tinted by g_BackLightMap when BACK_LIGHT_MAP
//   PARALLAX                g_Height offsets the UVs along the view
//   ALPHA_TEST              discard by g_Alpha (threshold, function)
//   NORMAL_MAP_GREEN_UP     the normal map's green points to -V (OpenGL style); 0 = towards +V
//   EFFECT                  the Effect model instead: unlit, g_Diffuse × base colour × vertex colour, then:
//     PALETTE               g_Palette, a 2D gradient: across by the source's grey (or alpha), down by base × vertex colour
//     FALLOFF               opacity by the view angle
//     SOFT_DEPTH            fades out near what lies behind (g_SceneDepth, the scene before the blended draws)
namespace render::diligent::shaders {

inline constexpr const char* kLightingCommon = R"(
#define MAX_LIGHTS 32
cbuffer LightingFrame {
    float4x4 g_ViewProj;
    float4   g_CameraPos;            // xyz
    float4   g_SunDir;               // xyz, towards the light
    float4   g_SunColor;             // rgb
    float4   g_Ambient[6];           // +X -X +Y -Y +Z -Z
    float4   g_Fog;                  // near, far, power, max
    float4   g_FogNearColor;         // rgb, w = fog on
    float4   g_FogFarColor;          // rgb
    float4   g_LightInfo;            // x = light count
    float4   g_LightPos[MAX_LIGHTS];   // xyz, w = radius
    float4   g_LightColor[MAX_LIGHTS]; // rgb
    float4   g_InvProjZ;             // rows 2 and 3 of the inverse projection (depth 0..1 to view space)
    float4   g_InvProjW;
    float4   g_ExposureA;            // image space: adapt key, strength, min exposure, max exposure
    float4   g_ExposureB;            // x = base exposure, y = 1 when it adapts (g_Adapted)
};
cbuffer LightingDraw {
    float4x4 g_World;
    float4   g_Tint;                 // rgba
    float4   g_SpecColor;            // rgb, w = strength
    float4   g_Emissive;             // rgb (colour · multiple), w = glossiness
    float4   g_Effects;              // effect1, effect2, env scale, parallax scale
    float4   g_UV;                   // offset.xy, scale.zw
    float4   g_Alpha;                // threshold, function, additive (fog fades it out), blended (alpha is coverage)
    float4   g_Srgb;                 // 1 where the texture is sRGB: diffuse, glow, env, back light
    float4   g_EffectColor;          // Effect: base colour rgb, alpha
    float4   g_Falloff;              // Effect: start, stop (cosines), start opacity, stop opacity
    float4   g_EffectFlags;          // Effect: soft depth, palette colour, palette alpha, palette sRGB
    float4   g_EffectScale;          // x = an effect's base colour scale, y = refraction strength
};
)";

inline constexpr const char* kLightingVS = R"(
#if SKINNED
cbuffer LightingBones { float4x4 g_Bones[256]; };
#endif
struct VSIn {
    float3 pos     : ATTRIB0;
    float3 nrm     : ATTRIB1;
    float2 uv      : ATTRIB2;
#if SKINNED
    float4 joints  : ATTRIB3;
    float4 weights : ATTRIB4;
#endif
#if VERTEX_COLORS
    float4 color   : ATTRIB5;
#endif
#if VERTEX_TANGENTS
    float4 tangent : ATTRIB6;
#endif
};
struct PSIn {
    float4 pos   : SV_POSITION;
    float3 world : WORLD_POS;
    float3 nrm   : NORMAL;
    float2 uv    : TEX_COORD;
    float4 color : COLOR;
#if MODEL_SPACE_NORMALS
    float3 m0    : MODEL_ROW0;
    float3 m1    : MODEL_ROW1;
    float3 m2    : MODEL_ROW2;
#endif
#if VERTEX_TANGENTS
    float4 tangent : TANGENT;
#endif
};
void main(in VSIn i, out PSIn o) {
    float4x4 model = g_World;
#if SKINNED
    float4x4 skin = g_Bones[(int)i.joints.x] * i.weights.x + g_Bones[(int)i.joints.y] * i.weights.y +
                    g_Bones[(int)i.joints.z] * i.weights.z + g_Bones[(int)i.joints.w] * i.weights.w;
    model = mul(g_World, skin);
#endif
    float4 wp = mul(model, float4(i.pos, 1.0));
    o.pos   = mul(g_ViewProj, wp);
    o.world = wp.xyz;
    float3x3 m3 = (float3x3)model;
    o.nrm   = mul(m3, i.nrm);
    o.uv    = i.uv * g_UV.zw + g_UV.xy;
#if VERTEX_COLORS
    o.color = i.color;
#else
    o.color = float4(1.0, 1.0, 1.0, 1.0);
#endif
#if MODEL_SPACE_NORMALS
    o.m0 = m3[0];
    o.m1 = m3[1];
    o.m2 = m3[2];
#endif
#if VERTEX_TANGENTS
    o.tangent = float4(mul(m3, i.tangent.xyz), i.tangent.w);
#endif
}
)";

inline constexpr const char* kLightingPS = R"(
Texture2D<float4> g_Diffuse;      SamplerState g_Diffuse_sampler;
#if NORMAL_MAP || MODEL_SPACE_NORMALS
Texture2D<float4> g_Normal;       SamplerState g_Normal_sampler;
#endif
#if SPEC_MAP
Texture2D<float4> g_Specular;     SamplerState g_Specular_sampler;
#endif
#if GLOW_MAP
Texture2D<float4> g_Glow;         SamplerState g_Glow_sampler;
#endif
#if ENV_MAP
TextureCube<float4> g_Env;        SamplerState g_Env_sampler;
#endif
#if ENV_MASK
Texture2D<float4> g_EnvMask;      SamplerState g_EnvMask_sampler;
#endif
#if LIGHTING_MAP
Texture2D<float4> g_LightingMap;  SamplerState g_LightingMap_sampler;
#endif
#if BACK_LIGHT_MAP
Texture2D<float4> g_BackLightMap; SamplerState g_BackLightMap_sampler;
#endif
#if PARALLAX
Texture2D<float4> g_Height;       SamplerState g_Height_sampler;
#endif
#if PALETTE
Texture2D<float4> g_Palette;      SamplerState g_Palette_sampler;
#endif
#if SOFT_DEPTH
Texture2D<float>  g_SceneDepth;
#endif
#if REFRACTION
Texture2D<float4> g_SceneColor;   // the scene as it stood before the refraction surfaces
#endif
#if EFFECT
Texture2D<float>  g_Adapted;      // the image space's adapted luminance, as the last frame left it
#endif

struct PSIn {
    float4 pos   : SV_POSITION;
    float3 world : WORLD_POS;
    float3 nrm   : NORMAL;
    float2 uv    : TEX_COORD;
    float4 color : COLOR;
#if MODEL_SPACE_NORMALS
    float3 m0    : MODEL_ROW0;
    float3 m1    : MODEL_ROW1;
    float3 m2    : MODEL_ROW2;
#endif
#if VERTEX_TANGENTS
    float4 tangent : TANGENT;
#endif
};

float3 LinearToSrgb(float3 c) {
    c = max(c, 0.0);
    return lerp(c * 12.92, 1.055 * pow(c, 1.0 / 2.4) - 0.055, step(0.0031308, c));
}
float3 SrgbToLinear(float3 c) {
    c = max(c, 0.0);
    return lerp(c / 12.92, pow((c + 0.055) / 1.055, 2.4), step(0.04045, c));
}
// A texel as the file stores it (an sRGB view decodes it; undo that).
float3 Stored(float3 c, float srgb) { return srgb > 0.5 ? LinearToSrgb(c) : c; }

float3 DirectionalAmbient(float3 n) {
    float3 n2 = n * n;
    return n2.x * (n.x >= 0.0 ? g_Ambient[0].rgb : g_Ambient[1].rgb) +
           n2.y * (n.y >= 0.0 ? g_Ambient[2].rgb : g_Ambient[3].rgb) +
           n2.z * (n.z >= 0.0 ? g_Ambient[4].rgb : g_Ambient[5].rgb);
}

bool AlphaPasses(float a, float threshold, int func) {
    switch (func) {
        case 0: return true;
        case 1: return a < threshold;
        case 2: return a == threshold;
        case 3: return a <= threshold;
        case 4: return a > threshold;
        case 5: return a != threshold;
        case 6: return a >= threshold;
        default: return false;
    }
}

struct Surface {
    float3 n, v;
    float  specMask, gloss;
    float3 rimMask, softMask, backTint;
};

// One light's diffuse (into diffuse) and specular (into spec), the game's terms.
void AddLight(Surface s, float3 l, float3 color, inout float3 diffuse, inout float3 spec) {
    float ndl = dot(s.n, l);
    diffuse += color * saturate(ndl);
#if SOFT_LIGHT
    // Wrap the terminator by effect1: light reaches past 90 degrees, masked.
    float wrapped = saturate((ndl + g_Effects.x) / (1.0 + g_Effects.x));
    diffuse += color * s.softMask * max(wrapped - saturate(ndl), 0.0);
#endif
#if RIM_LIGHT
    float rim = pow(saturate(1.0 - dot(s.n, s.v)), max(g_Effects.y, 0.001)) * saturate(dot(s.v, -l));
    diffuse += color * s.rimMask * rim;
#endif
#if BACK_LIGHT
    diffuse += color * s.backTint * saturate(-ndl);
#endif
    float3 h = normalize(l + s.v);
    spec += color * pow(saturate(dot(s.n, h)), s.gloss) * saturate(ndl) * s.specMask;
}

float3 ApplyFog(float3 color, float3 world) {
    if (g_FogNearColor.w > 0.5) {
        float dist = length(g_CameraPos.xyz - world);
        float t    = saturate((dist - g_Fog.x) / max(g_Fog.y - g_Fog.x, 1e-3));
        float fog  = min(pow(t, max(g_Fog.z, 1e-3)), 1.0) * g_Fog.w;
        if (g_Alpha.z > 0.5) color *= 1.0 - fog;   // additive: fog fades the light out
        else color = lerp(color, lerp(g_FogNearColor.rgb, g_FogFarColor.rgb, t), fog);
    }
    return color;
}

// Distance along the view (positive) for a depth value 0..1 at this pixel.
float ViewDepth(float depth) {
    float4 p = float4(0.0, 0.0, depth, 1.0);
    return -dot(g_InvProjZ, p) / dot(g_InvProjW, p);
}

#if EFFECT
// The exposure the post pass multiplies the frame by: the base exposure, times the image space's eye adaptation when it
// runs (its Exposure(), from the luminance the last frame adapted to).
float FrameExposure() {
    float e = g_ExposureB.x;
    if (g_ExposureB.y > 0.5) {
        float adapted = g_Adapted.Load(int3(0, 0, 0));
        e *= clamp(pow(g_ExposureA.x / max(adapted, 1e-5), g_ExposureA.y), g_ExposureA.z, g_ExposureA.w);
    }
    return max(e, 1e-4);
}

float4 EffectMain(PSIn i, float3 v, float2 uv) {
    float4 src  = g_Diffuse.Sample(g_Diffuse_sampler, uv);
    float3 c    = Stored(src.rgb, g_Srgb.x);
    float  a    = src.a;
    float3 tint = g_EffectColor.rgb * i.color.rgb;
    float  ta   = g_EffectColor.a * i.color.a;
#if PALETTE
    // The gradient's row is the tint: it picks the colour rather than multiplying it.
    if (g_EffectFlags.y > 0.5) {
        c    = Stored(g_Palette.Sample(g_Palette_sampler, float2(saturate(c.g), saturate(tint.r * g_EffectScale.x))).rgb, g_EffectFlags.w);
        tint = float3(1.0, 1.0, 1.0);
    }
    if (g_EffectFlags.z > 0.5) {
        a  = g_Palette.Sample(g_Palette_sampler, float2(saturate(a), saturate(ta))).a;
        ta = 1.0;
    }
#endif
    c *= tint * g_EffectScale.x;
    a *= ta;
#if FALLOFF
    float ndv = abs(dot(normalize(i.nrm), v));
    float t   = saturate((ndv - g_Falloff.y) / max(g_Falloff.x - g_Falloff.y, 1e-4));
    a *= lerp(g_Falloff.w, g_Falloff.z, t);
#endif
#if SOFT_DEPTH
    float behind = ViewDepth(g_SceneDepth.Load(int3(int2(i.pos.xy), 0)));
    a *= saturate((behind - ViewDepth(i.pos.z)) / max(g_EffectFlags.x, 1e-3));
#endif
#if ALPHA_TEST
    if (!AlphaPasses(a, g_Alpha.x, (int)g_Alpha.y)) discard;
#endif
    c = ApplyFog(c, i.world);
    // An effect keeps its own brightness: the frame's exposure is divided back out, so eye adaptation brightening a dark
    // interior doesn't blow a glow, a flame or a dust beam out to white.
    return float4(SrgbToLinear(c) / FrameExposure(), g_Alpha.w > 0.5 ? a : 1.0);
}
#endif

float4 main(in PSIn i, bool front : SV_IsFrontFace) : SV_Target {
    float3 v  = normalize(g_CameraPos.xyz - i.world);
    float2 uv = i.uv;
#if EFFECT
    return EffectMain(i, v, uv);
#elif REFRACTION
    // What lies behind, displaced by the normal map's tangent-space xy × strength × a tenth of the frame's height.
    uint sw, sh;
    g_SceneColor.GetDimensions(sw, sh);
    float2 offset = float2(0.0, 0.0);
#if NORMAL_MAP
    offset = (g_Normal.Sample(g_Normal_sampler, uv).xy * 2.0 - 1.0) * g_EffectScale.y * 0.1 * float(sh);
#endif
    int2 p = int2(clamp(i.pos.xy + offset, float2(0.0, 0.0), float2(float(sw) - 1.0, float(sh) - 1.0)));
    return g_SceneColor.Load(int3(p, 0));
#else
    float3 gn = normalize(i.nrm) * (front ? 1.0 : -1.0);

    // Cotangent frame from screen derivatives: T along +U, B along +V.
    float3 dp1 = ddx(i.world), dp2 = ddy(i.world);
    float2 duv1 = ddx(i.uv), duv2 = ddy(i.uv);
    float3 dp2perp = cross(dp2, gn), dp1perp = cross(gn, dp1);
    float3 T = dp2perp * duv1.x + dp1perp * duv2.x;
    float3 B = dp2perp * duv1.y + dp1perp * duv2.y;
    // The formula assumes screen y up; HLSL's ddy runs down the screen (on every backend), which mirrors the frame.
    float  invmax = -rsqrt(max(max(dot(T, T), dot(B, B)), 1e-20));
    T *= invmax;
    B *= invmax;
#if VERTEX_TANGENTS
    T = normalize(i.tangent.xyz - gn * dot(gn, i.tangent.xyz));   // the mesh's own frame, kept square to the normal
    B = cross(gn, T) * (i.tangent.w < 0.0 ? -1.0 : 1.0) * (front ? 1.0 : -1.0);
#endif

#if PARALLAX
    float  height = g_Height.Sample(g_Height_sampler, uv).r;
    float2 vts    = float2(dot(v, T), dot(v, B));
    uv += vts * (height * g_Effects.w - g_Effects.w * 0.5);
#endif

    float4 base   = g_Diffuse.Sample(g_Diffuse_sampler, uv);
    float3 albedo = Stored(base.rgb, g_Srgb.x) * g_Tint.rgb * i.color.rgb;
    float  alpha  = base.a * g_Tint.a * i.color.a;
#if ALPHA_TEST
    if (!AlphaPasses(alpha, g_Alpha.x, (int)g_Alpha.y)) discard;
#endif

    Surface s;
    s.v = v;
    s.n = gn;
    s.specMask = 1.0;
#if NORMAL_MAP || MODEL_SPACE_NORMALS
    float4 nt = g_Normal.Sample(g_Normal_sampler, uv);
    float3 nm = nt.xyz * 2.0 - 1.0;
#if MODEL_SPACE_NORMALS
    s.n = normalize(mul(float3x3(i.m0, i.m1, i.m2), nm)) * (front ? 1.0 : -1.0);
#else
#if NORMAL_MAP_GREEN_UP
    nm.y = -nm.y;
#endif
    s.n = normalize(nm.x * T + nm.y * B + nm.z * gn);
#endif
#if SPEC_NORMAL_ALPHA
    s.specMask = nt.a;
#endif
#endif
#if SPEC_MAP
    s.specMask = g_Specular.Sample(g_Specular_sampler, uv).r;
#endif
    s.specMask *= g_SpecColor.w;
    s.gloss = max(g_Emissive.w, 1.0);
    s.rimMask = s.softMask = float3(1.0, 1.0, 1.0);
#if LIGHTING_MAP
    s.rimMask = s.softMask = g_LightingMap.Sample(g_LightingMap_sampler, uv).rgb;
#endif
    s.backTint = float3(1.0, 1.0, 1.0);
#if BACK_LIGHT_MAP
    s.backTint = Stored(g_BackLightMap.Sample(g_BackLightMap_sampler, uv).rgb, g_Srgb.w);
#endif

    float3 diffuse = DirectionalAmbient(s.n);
    float3 spec    = float3(0.0, 0.0, 0.0);
    AddLight(s, normalize(g_SunDir.xyz), g_SunColor.rgb, diffuse, spec);
    int count = (int)g_LightInfo.x;
    for (int k = 0; k < count; ++k) {
        float3 d    = g_LightPos[k].xyz - i.world;
        float  dist = length(d);
        float  f    = saturate(dist / max(g_LightPos[k].w, 1e-3));
        float  att  = 1.0 - f * f;
        if (att <= 0.0) continue;
        AddLight(s, d / max(dist, 1e-4), g_LightColor[k].rgb * att, diffuse, spec);
    }

    float3 color = albedo * diffuse + spec * g_SpecColor.rgb;
#if ENV_MAP
    float envMask = 1.0;
#if ENV_MASK
    envMask = g_EnvMask.Sample(g_EnvMask_sampler, uv).r;
#elif ENV_MASK_NORMAL_ALPHA && (NORMAL_MAP || MODEL_SPACE_NORMALS)
    envMask = nt.a;
#endif
    float3 r = reflect(-v, s.n);
    color += Stored(g_Env.Sample(g_Env_sampler, r).rgb, g_Srgb.z) * envMask * g_Effects.z;
#endif
#if GLOW_MAP
    color += Stored(g_Glow.Sample(g_Glow_sampler, uv).rgb, g_Srgb.y) * g_Emissive.rgb;
#elif OWN_EMIT
    color += g_Emissive.rgb;
#endif

    color = ApplyFog(color, i.world);
    // Unblended surfaces cover the pixel whatever their texture's alpha holds (often a mask).
    return float4(SrgbToLinear(color), g_Alpha.w > 0.5 ? alpha : 1.0);
#endif
}
)";

} // namespace render::diligent::shaders
