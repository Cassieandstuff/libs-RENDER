#pragma once
// HLSL for the renderer's own passes (PBR meshes use DiligentFX's shaders).
// Vertex inputs must use the ATTRIBn semantics (Diligent maps them to Vulkan
// locations). Matrices are glm column-major, uploaded raw: mul(M, v).

namespace render::diligent::shaders {

// ── Overlays (Overlay.cpp) ───────────────────────────────────────────────────
// Drawn over the tone-mapped output. Vertex colours are display (sRGB) values:
// LINEAR_OUT = 1 for an sRGB target (decode here, the hardware re-encodes, so the
// colour lands exactly); 0 writes them as they are. Positions are world space, or
// physical pixels (top-left origin) when g_Mode.x = 1 — the screen layer.
#define SCT_OVERLAY_VS_COMMON R"(
cbuffer OverlayCB {
    float4x4 g_ViewProj;
    float4   g_Viewport;   // w, h, 1/w, 1/h (pixels)
    float4   g_Mode;       // x = 1: positions are screen pixels
}
float4 ToClip(float3 p) {
    if (g_Mode.x > 0.5)
        return float4(p.x * 2.0 * g_Viewport.z - 1.0, 1.0 - p.y * 2.0 * g_Viewport.w, 0.0, 1.0);
    return mul(g_ViewProj, float4(p, 1.0));
}
)"

#define SCT_OVERLAY_PS_COMMON R"(
float3 DisplayToOutput(float3 c) {
#if LINEAR_OUT
    float3 lo = c / 12.92;
    float3 hi = pow((max(c, 0.0) + 0.055) / 1.055, 2.4);
    return lerp(hi, lo, step(c, 0.04045));
#else
    return c;
#endif
}
)"

// Lines: one instance per segment, six vertices expanding it into a screen-aligned
// quad `w` pixels wide (square caps) plus a 1-pixel anti-aliasing fringe. The
// segment is clipped to the near plane first, so the expansion never divides by
// w <= 0. The quad keeps each end's depth.
inline constexpr const char* kOverlayLineVS = SCT_OVERLAY_VS_COMMON R"(
struct VSIn {
    float3 a  : ATTRIB0;
    float4 ca : ATTRIB1;
    float3 b  : ATTRIB2;
    float4 cb : ATTRIB3;
    float  w  : ATTRIB4;
};
struct PSIn { float4 pos : SV_POSITION; float4 col : COLOR; float2 edge : TEX_COORD; };
void main(in VSIn i, in uint vid : SV_VertexID, out PSIn o) {
    float4 pa = ToClip(i.a), pb = ToClip(i.b);
    o.col  = i.ca;
    o.edge = float2(0.0, 0.0);
    if (pa.z < 0.0 && pb.z < 0.0) { o.pos = float4(0.0, 0.0, -1.0, 1.0); return; }   // wholly behind: culled
    if (pa.z < 0.0) pa = lerp(pa, pb, pa.z / (pa.z - pb.z));
    else if (pb.z < 0.0) pb = lerp(pb, pa, pb.z / (pb.z - pa.z));

    float2 sa  = (pa.xy / pa.w * float2(0.5, -0.5) + 0.5) * g_Viewport.xy;
    float2 sb  = (pb.xy / pb.w * float2(0.5, -0.5) + 0.5) * g_Viewport.xy;
    float2 d   = sb - sa;
    float  len = length(d);
    float2 dir = len > 1e-4 ? d / len : float2(1.0, 0.0);
    float2 nrm = float2(-dir.y, dir.x);
    float  halfW = 0.5 * max(i.w, 0.0);
    float  ext   = halfW + 1.0;

    // Corners (end, side): (0,-) (0,+) (1,+)  (0,-) (1,+) (1,-)
    bool   atB  = vid == 2 || vid == 4 || vid == 5;
    float  side = (vid == 1 || vid == 2 || vid == 4) ? 1.0 : -1.0;
    float4 c    = atB ? pb : pa;
    float2 s    = (atB ? sb : sa) + dir * (atB ? halfW : -halfW) + nrm * (side * ext);
    float2 ndc  = float2(s.x * g_Viewport.z * 2.0 - 1.0, 1.0 - s.y * g_Viewport.w * 2.0);
    o.pos  = float4(ndc * c.w, c.z, c.w);
    o.col  = atB ? i.cb : i.ca;
    o.edge = float2(side * ext, halfW);
}
)";

inline constexpr const char* kOverlayLinePS = SCT_OVERLAY_PS_COMMON R"(
struct PSIn { float4 pos : SV_POSITION; float4 col : COLOR; float2 edge : TEX_COORD; };
float4 main(in PSIn i) : SV_Target {
    // edge.x = signed distance from the centre line (pixels), edge.y = half width.
    float cov = saturate(i.edge.y + 0.5 - abs(i.edge.x)) * saturate(i.edge.y * 2.0);
    float a   = i.col.a * cov;
    if (a < 0.5 / 255.0) discard;   // the fringe must not write depth (on-top layer)
    return float4(DisplayToOutput(i.col.rgb), a);
}
)";

inline constexpr const char* kOverlayTriVS = SCT_OVERLAY_VS_COMMON R"(
struct VSIn { float3 pos : ATTRIB0; float4 col : ATTRIB1; };
struct PSIn { float4 pos : SV_POSITION; float4 col : COLOR; };
void main(in VSIn i, out PSIn o) {
    o.pos = ToClip(i.pos);
    o.col = i.col;
}
)";

inline constexpr const char* kOverlayTriPS = SCT_OVERLAY_PS_COMMON R"(
struct PSIn { float4 pos : SV_POSITION; float4 col : COLOR; };
float4 main(in PSIn i) : SV_Target {
    if (i.col.a < 0.5 / 255.0) discard;
    return float4(DisplayToOutput(i.col.rgb), i.col.a);
}
)";

// Text: glyph quads in screen pixels, sampling the host's atlas (uploaded RGBA8:
// rgb tints the colour, a = coverage).
inline constexpr const char* kOverlayTextVS = SCT_OVERLAY_VS_COMMON R"(
struct VSIn { float2 pos : ATTRIB0; float2 uv : ATTRIB1; float4 col : ATTRIB2; };
struct PSIn { float4 pos : SV_POSITION; float2 uv : TEX_COORD; float4 col : COLOR; };
void main(in VSIn i, out PSIn o) {
    o.pos = ToClip(float3(i.pos, 0.0));
    o.uv  = i.uv;
    o.col = i.col;
}
)";

inline constexpr const char* kOverlayTextPS = SCT_OVERLAY_PS_COMMON R"(
Texture2D<float4> g_Atlas;
SamplerState      g_Atlas_sampler;
struct PSIn { float4 pos : SV_POSITION; float2 uv : TEX_COORD; float4 col : COLOR; };
float4 main(in PSIn i) : SV_Target {
    float4 t = g_Atlas.Sample(g_Atlas_sampler, i.uv);
    float  a = i.col.a * t.a;
    if (a < 0.5 / 255.0) discard;
    return float4(DisplayToOutput(i.col.rgb * t.rgb), a);
}
)";

#undef SCT_OVERLAY_VS_COMMON
#undef SCT_OVERLAY_PS_COMMON

// ── Fullscreen triangle (post, face tint) ────────────────────────────────────
// uv (0,0) is the top-left of the target on both backends.
inline constexpr const char* kFullscreenVS = R"(
struct PSIn { float4 pos : SV_POSITION; float2 uv : TEX_COORD; };
void main(in uint id : SV_VertexID, out PSIn o) {
    float2 uv = float2((id << 1) & 2, id & 2);
    o.uv  = uv;
    o.pos = float4(uv * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
}
)";

// ── Post: exposure + ACES (Narkowicz fit), the same curve as SCT's GL renderer,
// composited over the background. The scene target is cleared to transparent and
// every pass writes premultiplied colour + coverage, so the background is exact
// (display space, untouched by exposure or the tone curve).
// OUTPUT_GAMMA = 1 when the output is not an sRGB format (the shader encodes);
// an sRGB output encodes in hardware, so the shader writes linear.
inline constexpr const char* kPostPS = R"(
Texture2D<float4> g_HDR;
cbuffer PostCB {
    float4 g_Params;       // x = exposure
    float4 g_Background;   // linear rgb
}
struct PSIn { float4 pos : SV_POSITION; float2 uv : TEX_COORD; };
float3 Aces(float3 x) {
    const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
}
float4 main(in PSIn i) : SV_Target {
    float4 hdr = g_HDR.Load(int3(int2(i.pos.xy), 0));
    float  a   = saturate(hdr.a);
    float3 ldr = a > 0.0 ? Aces(hdr.rgb / max(hdr.a, 1e-4) * g_Params.x) : float3(0.0, 0.0, 0.0);
    ldr = lerp(g_Background.rgb, ldr, a);
#if OUTPUT_GAMMA
    ldr = pow(ldr, 1.0 / 2.2);
#endif
    return float4(ldr, 1.0);
}
)";

// ── Infinite ground grid (after Dance Card Creator's GridShader) ────────────────
// Drawn over the tone-mapped output: each pixel ray-casts the Z = 0 plane through
// the inverse view-projection; minor/major lines with screen-space AA, the X (red)
// and Y (green) axes, a distance fade. Depth-tested against the scene (SV_Depth)
// without writing it, so geometry below the plane still shows. Writes premultiplied
// coverage (blend ONE, INV_SRC_ALPHA). Colours arrive linear.
inline constexpr const char* kGridPS = R"(
cbuffer GridCB {
    float4x4 g_ViewProj;
    float4x4 g_InvViewProj;
    float4   g_CameraPos;
    float4   g_Grid;        // x minor spacing, y major every, z fade distance
    float4   g_Minor;
    float4   g_Major;
    float4   g_AxisX;
    float4   g_AxisY;
}
struct PSIn { float4 pos : SV_POSITION; float2 uv : TEX_COORD; };
struct PSOut { float4 color : SV_Target; float depth : SV_Depth; };

float GridLine(float2 coord) {
    float2 d = max(fwidth(coord), 1e-5);
    float2 g = abs(frac(coord - 0.5) - 0.5) / d;
    return 1.0 - saturate(min(g.x, g.y));
}

void Over(inout float4 acc, float3 c, float w) {   // premultiplied "c over acc" at weight w
    acc = lerp(acc, float4(c, 1.0), w);
}

void main(in PSIn i, out PSOut o) {
    float2 ndc = float2(i.uv.x * 2.0 - 1.0, 1.0 - i.uv.y * 2.0);
    float4 n = mul(g_InvViewProj, float4(ndc, 0.0, 1.0));
    float4 f = mul(g_InvViewProj, float4(ndc, 1.0, 1.0));
    n.xyz /= n.w;
    f.xyz /= f.w;
    float3 dir = f.xyz - n.xyz;

    // No early-out: the derivatives below must stay in uniform control flow.
    bool  hits = abs(dir.z) > 1e-6;
    float t    = hits ? -n.z / dir.z : 0.0;
    hits       = hits && t > 0.0;
    float3 p   = n.xyz + max(t, 0.0) * dir;

    float  minor = GridLine(p.xy / g_Grid.x);
    float  major = GridLine(p.xy / (g_Grid.x * g_Grid.y));
    float2 ad    = max(fwidth(p.xy), 1e-5);
    float  axisX = 1.0 - saturate(abs(p.y) / ad.y * 0.5);   // the X axis is the line y = 0
    float  axisY = 1.0 - saturate(abs(p.x) / ad.x * 0.5);
    float  fade  = saturate(1.0 - length(p.xy - g_CameraPos.xy) / g_Grid.z);

    float4 acc = float4(0.0, 0.0, 0.0, 0.0);
    Over(acc, g_Minor.rgb, minor * fade);
    Over(acc, g_Major.rgb, major * fade);
    Over(acc, g_AxisX.rgb, axisX * fade);
    Over(acc, g_AxisY.rgb, axisY * fade);

    float4 clip = mul(g_ViewProj, float4(p, 1.0));
    if (!hits || acc.a <= 0.0) discard;
    o.color = acc;
    o.depth = saturate(clip.z / clip.w);
}
)";

// ── Face tint: one layer over the accumulated result (GL kTintCompositeFS) ──
// Overlay / modulate-2x: linear maths (sRGB textures linearise on load and the
// sRGB target re-encodes), as GL. Soft light (Skyrim's FaceGen / SkinTint shading,
// not in GL): on sRGB-encoded values, as the game shades in gamma space, against
// the original base, blended towards in encoded space.
inline constexpr const char* kTintPS = R"(
Texture2D<float4> g_Base;      // the base diffuse (same size as the target)
Texture2D<float4> g_Current;   // accumulated result so far (same size as the target)
Texture2D<float4> g_Mask;      // this layer's coverage mask (any size)
SamplerState      g_Mask_sampler;
Texture2D<float4> g_TintMap;   // soft light: per-texel tint colour (any size)
SamplerState      g_TintMap_sampler;
cbuffer TintCB {
    float4 g_Tint;     // rgb = tint colour (display values for soft light), a = strength
    int4   g_Params;   // x = blend (0 overlay, 1 modulate-2x, 2 soft light, 3 detail), y = coverage from alpha, z = has mask, w = copy
    int4   g_Params2;  // x = has tint map
}
float3 ToSrgb(float3 c) {
    c = saturate(c);
    return lerp(1.055 * pow(c, 1.0 / 2.4) - 0.055, c * 12.92, step(c, 0.0031308));
}
float3 ToLinear(float3 c) {
    c = saturate(c);
    return lerp(pow((c + 0.055) / 1.055, 2.4), c / 12.92, step(c, 0.04045));
}
struct PSIn { float4 pos : SV_POSITION; float2 uv : TEX_COORD; };
float4 main(in PSIn i) : SV_Target {
    float4 cur = g_Current.Load(int3(int2(i.pos.xy), 0));
    if (g_Params.w != 0) return cur;
    float4 m   = g_Mask.SampleLevel(g_Mask_sampler, i.uv, 0.0);
    float  cov = g_Params.z == 0 ? 1.0
               : (g_Params.y != 0 ? m.a : dot(m.rgb, float3(0.299, 0.587, 0.114)));
    float  k   = saturate(cov * g_Tint.a);
    float3 rgb;
    if (g_Params.x == 2) {   // soft light: b² + 2·T·b·(1 − b)
        float3 b = ToSrgb(g_Base.Load(int3(int2(i.pos.xy), 0)).rgb);
        float3 t = g_Params2.x != 0 ? ToSrgb(g_TintMap.SampleLevel(g_TintMap_sampler, i.uv, 0.0).rgb)
                                    : saturate(g_Tint.rgb);
        float3 s = b * b + 2.0 * t * b * (1.0 - b);
        rgb = ToLinear(lerp(ToSrgb(cur.rgb), s, k));
    } else if (g_Params.x == 3) {   // FaceGen detail map: c · 3.984375·(D + (1, 0, 1)/255)
        float3 c = ToSrgb(cur.rgb);
        float3 d = ToSrgb(g_TintMap.SampleLevel(g_TintMap_sampler, i.uv, 0.0).rgb);
        float3 s = c * (3.984375 * (d + float3(1.0 / 255.0, 0.0, 1.0 / 255.0)));
        rgb = ToLinear(lerp(c, s, k));
    } else if (g_Params.x == 1) {
        rgb = cur.rgb * lerp(float3(1.0, 1.0, 1.0), 2.0 * g_Tint.rgb, k);   // skin tone (modulate 2x)
    } else {
        rgb = lerp(cur.rgb, g_Tint.rgb, k);                                // overlay
    }
    return float4(rgb, cur.a);
}
)";

} // namespace render::diligent::shaders
