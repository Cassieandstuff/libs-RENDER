// ============================================================================
// RENDER DDS test — header routing and the legacy uncompressed
// decoder (src/DdsDecode.h), no GPU. The GPU side (upload, sampling) is in
// GpuRenderTest.cpp's "legacy DDS formats" section.
// ============================================================================

#include "DdsDecode.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <vector>

using namespace render::diligent::dds;

namespace {

int g_failures = 0;

void check(const char* label, bool ok) {
    std::printf("[ %s ] %s\n", ok ? "PASS" : "FAIL", label);
    if (!ok) ++g_failures;
}

constexpr std::uint32_t RGB = 0x40, ALPHAPIXELS = 0x1, LUMINANCE = 0x20000, ALPHA = 0x2, FOURCC = 0x4, PAL8 = 0x20;

struct Spec {
    std::uint32_t w = 4, h = 4, mips = 1, flags = RGB, fourCC = 0, bits = 32, r = 0, g = 0, b = 0, a = 0;
    std::uint32_t headerFlags = 0x1 | 0x2 | 0x4 | 0x1000, caps2 = 0;
};

// 'DDS ' + a legacy header, then `pixels` verbatim.
std::vector<std::uint8_t> Dds(const Spec& s, const std::vector<std::uint8_t>& pixels) {
    std::vector<std::uint32_t> hd(32, 0);
    hd[0]  = 0x20534444;
    hd[1]  = 124;
    hd[2]  = s.headerFlags | (s.mips > 1 ? 0x20000u : 0u);
    hd[3]  = s.h;
    hd[4]  = s.w;
    hd[7]  = s.mips;
    hd[19] = 32;
    hd[20] = s.flags;
    hd[21] = s.fourCC;
    hd[22] = s.bits;
    hd[23] = s.r;
    hd[24] = s.g;
    hd[25] = s.b;
    hd[26] = s.a;
    hd[27] = 0x1000;
    hd[28] = s.caps2;
    std::vector<std::uint8_t> bytes(hd.size() * 4);
    std::memcpy(bytes.data(), hd.data(), bytes.size());
    bytes.insert(bytes.end(), pixels.begin(), pixels.end());
    return bytes;
}

// `count` copies of one pixel's bytes.
std::vector<std::uint8_t> Fill(std::initializer_list<std::uint8_t> px, std::size_t count) {
    std::vector<std::uint8_t> v;
    for (std::size_t i = 0; i < count; ++i) v.insert(v.end(), px);
    return v;
}

constexpr std::uint32_t CC(const char (&s)[5]) {
    return std::uint32_t(std::uint8_t(s[0])) | std::uint32_t(std::uint8_t(s[1])) << 8 | std::uint32_t(std::uint8_t(s[2])) << 16 |
           std::uint32_t(std::uint8_t(s[3])) << 24;
}

Route RouteOf(const std::vector<std::uint8_t>& b) { return Inspect(b).route; }

// The first pixel of mip `level` after decoding.
bool First(const std::vector<std::uint8_t>& b, int r, int g, int bl, int a, std::size_t level = 0, bool srgb = false) {
    Rgba8Image  img;
    std::string why;
    if (!DecodeLegacy(b, srgb, img, why) || img.mips.size() <= level) {
        std::printf("   decode failed: %s\n", why.c_str());
        return false;
    }
    const std::uint8_t* p = img.mips[level].data();
    const bool ok = p[0] == r && p[1] == g && p[2] == bl && p[3] == a;
    if (!ok) std::printf("   got (%d,%d,%d,%d), expected (%d,%d,%d,%d)\n", p[0], p[1], p[2], p[3], r, g, bl, a);
    return ok;
}

void TestRouting() {
    std::printf("-- routing --\n");
    Spec dxt;
    dxt.flags  = FOURCC;
    dxt.fourCC = CC("DXT5");
    dxt.bits   = 0;
    check("DXT5 -> Native", RouteOf(Dds(dxt, Fill({ 0 }, 16))) == Route::Native);
    dxt.fourCC = CC("DXT1");
    check("DXT1 -> Native", RouteOf(Dds(dxt, Fill({ 0 }, 8))) == Route::Native);
    dxt.fourCC = CC("ABCD");
    const Inspection unknown = Inspect(Dds(dxt, Fill({ 0 }, 16)));
    check("unknown FourCC -> Reject, and says what it is",
          unknown.route == Route::Reject && unknown.format.find("'ABCD'") != std::string::npos && !unknown.problem.empty());

    Spec bgra;
    bgra.flags = RGB | ALPHAPIXELS;
    bgra.r = 0xff0000, bgra.g = 0xff00, bgra.b = 0xff, bgra.a = 0xff000000;
    check("BGRA8 -> Native", RouteOf(Dds(bgra, Fill({ 0, 0, 0, 0 }, 16))) == Route::Native);
    Spec rgba = bgra;
    rgba.r = 0xff, rgba.b = 0xff0000;
    check("RGBA8 -> Native", RouteOf(Dds(rgba, Fill({ 0, 0, 0, 0 }, 16))) == Route::Native);

    Spec bgrx = bgra;
    bgrx.flags = RGB;
    bgrx.a     = 0;
    check("X8R8G8B8 -> Decode", RouteOf(Dds(bgrx, Fill({ 0, 0, 0, 0 }, 16))) == Route::Decode);
    Spec rgb24 = bgrx;
    rgb24.bits = 24;
    check("B8G8R8 24-bit -> Decode", RouteOf(Dds(rgb24, Fill({ 0, 0, 0 }, 16))) == Route::Decode);

    Spec l8;
    l8.flags = LUMINANCE, l8.bits = 8, l8.r = 0xff;
    check("L8 -> Decode", RouteOf(Dds(l8, Fill({ 0 }, 16))) == Route::Decode);
    Spec l16 = l8;
    l16.bits = 16, l16.r = 0xffff;
    check("L16 -> Native (wider than 8 bits)", RouteOf(Dds(l16, Fill({ 0, 0 }, 16))) == Route::Native);

    Spec vol = l8;
    vol.headerFlags |= 0x800000;
    vol.caps2 = 0x200000;
    check("a volume texture -> Reject", RouteOf(Dds(vol, Fill({ 0 }, 64))) == Route::Reject);
    Spec cube = rgb24;
    cube.caps2 = 0x200 | 0xFC00;
    check("a 24-bit cube map -> Reject", RouteOf(Dds(cube, Fill({ 0, 0, 0 }, 96))) == Route::Reject);
    Spec pal;
    pal.flags = PAL8, pal.bits = 8;
    check("a palettised texture -> Reject", RouteOf(Dds(pal, Fill({ 0 }, 16))) == Route::Reject);
    Spec wide = bgrx;
    wide.r = 0x3ff, wide.g = 0xffc00, wide.b = 0x3ff00000, wide.a = 0;
    check("10-bit masks Diligent doesn't map -> Reject", RouteOf(Dds(wide, Fill({ 0, 0, 0, 0 }, 16))) == Route::Reject);
    Spec gappy = bgrx;
    gappy.r = 0x00f0f000;
    check("a non-contiguous mask -> Reject", RouteOf(Dds(gappy, Fill({ 0, 0, 0, 0 }, 16))) == Route::Reject);

    check("24-bit with too few pixel bytes -> Reject", RouteOf(Dds(rgb24, Fill({ 0, 0, 0 }, 15))) == Route::Reject);
    const auto full = Dds(rgb24, Fill({ 0, 0, 0 }, 16));
    check("a header cut short -> Reject", RouteOf({ full.begin(), full.begin() + 100 }) == Route::Reject);
    std::vector<std::uint8_t> badSize = full;
    badSize[4] = 100;
    check("a wrong header size -> Reject", RouteOf(badSize) == Route::Reject);
    check("no DDS magic -> NotDds", RouteOf(std::vector<std::uint8_t>(300, 0xAB)) == Route::NotDds);

    // DX10: 2D goes to Diligent, 3D doesn't.
    std::vector<std::uint8_t> dx10 = Dds(dxt, {});
    std::memcpy(dx10.data() + 84, "DX10", 4);
    const std::uint32_t ext2d[5] = { 98, 3, 0, 1, 0 };   // BC7, TEXTURE2D
    dx10.insert(dx10.end(), reinterpret_cast<const std::uint8_t*>(ext2d), reinterpret_cast<const std::uint8_t*>(ext2d) + 20);
    dx10.resize(dx10.size() + 16);
    check("DX10 2D -> Native", RouteOf(dx10) == Route::Native);
    dx10[128 + 4] = 4;
    check("DX10 3D -> Reject", RouteOf(dx10) == Route::Reject);
}

void TestDecode() {
    std::printf("-- decode --\n");
    Spec s;
    s.flags = RGB, s.bits = 24, s.r = 0xff0000, s.g = 0xff00, s.b = 0xff;
    check("B8G8R8 (bytes b,g,r) -> (r,g,b,255)", First(Dds(s, Fill({ 30, 20, 10 }, 16)), 10, 20, 30, 255));
    s.r = 0xff, s.b = 0xff0000;
    check("R8G8B8 masks (bytes r,g,b) -> (r,g,b,255)", First(Dds(s, Fill({ 10, 20, 30 }, 16)), 10, 20, 30, 255));

    s.bits = 32, s.r = 0xff0000, s.g = 0xff00, s.b = 0xff, s.a = 0;
    check("X8R8G8B8 ignores the pad byte -> alpha 255", First(Dds(s, Fill({ 30, 20, 10, 0 }, 16)), 10, 20, 30, 255));
    s.r = 0xff, s.b = 0xff0000;
    check("X8B8G8R8 -> (r,g,b,255)", First(Dds(s, Fill({ 10, 20, 30, 7 }, 16)), 10, 20, 30, 255));

    s.bits = 16, s.r = 0xf800, s.g = 0x7e0, s.b = 0x1f, s.a = 0;
    // r=16/31, g=32/63, b=31/31 → 0x8000|0x400|0x1f = 0x841f
    check("R5G6B5 widens each channel", First(Dds(s, Fill({ 0x1f, 0x84 }, 16)), 132, 130, 255, 255));
    s.r = 0x7c00, s.g = 0x3e0, s.b = 0x1f, s.a = 0x8000;
    // a=1, r=31, g=0, b=1 → 0x8000|0x7c00|0x0001 = 0xfc01
    check("A1R5G5B5 -> 1-bit alpha", First(Dds(s, Fill({ 0x01, 0xfc }, 16)), 255, 0, 8, 255));
    s.a = 0;
    check("X1R5G5B5 -> alpha 255", First(Dds(s, Fill({ 0x01, 0x7c }, 16)), 255, 0, 8, 255));
    s.r = 0xf00, s.g = 0xf0, s.b = 0xf, s.a = 0xf000;
    check("A4R4G4B4 -> each nibble x17", First(Dds(s, Fill({ 0x21, 0x43 }, 16)), 3 * 17, 2 * 17, 1 * 17, 4 * 17));

    Spec l;
    l.flags = LUMINANCE, l.bits = 8, l.r = 0xff;
    check("L8 -> (L,L,L,255)", First(Dds(l, Fill({ 77 }, 16)), 77, 77, 77, 255));
    l.flags = LUMINANCE | ALPHAPIXELS, l.bits = 16, l.a = 0xff00;
    check("A8L8 -> (L,L,L,A)", First(Dds(l, Fill({ 77, 99 }, 16)), 77, 77, 77, 99));
    l.bits = 8, l.r = 0x0f, l.a = 0xf0;
    check("A4L4 -> (L,L,L,A)", First(Dds(l, Fill({ 0x52 }, 16)), 2 * 17, 2 * 17, 2 * 17, 5 * 17));
    Spec a8;
    a8.flags = ALPHA, a8.bits = 8;
    check("A8 (no mask) -> (0,0,0,A)", First(Dds(a8, Fill({ 200 }, 16)), 0, 0, 0, 200));
    a8.a = 0xff;
    check("A8 (mask) -> (0,0,0,A)", First(Dds(a8, Fill({ 201 }, 16)), 0, 0, 0, 201));

    // Pixel order: row-major, top row first, no row padding (3 px × 3 bytes = 9).
    Spec odd;
    odd.w = 3, odd.h = 2, odd.flags = RGB, odd.bits = 24, odd.r = 0xff0000, odd.g = 0xff00, odd.b = 0xff;
    std::vector<std::uint8_t> px;
    for (std::uint8_t i = 0; i < 6; ++i) px.insert(px.end(), { 0, 0, std::uint8_t(i * 40) });
    Rgba8Image img;
    std::string why;
    const bool ok = DecodeLegacy(Dds(odd, px), false, img, why);
    bool order = ok && img.mips[0].size() == 6u * 4u;
    for (int i = 0; order && i < 6; ++i) order = img.mips[0][i * 4] == i * 40;
    check("pixels decode row-major with no row padding", order);
    check("a 3x2 image gets mips 3x2, 1x1", ok && img.mips.size() == 2 && img.mips[1].size() == 4);
}

void TestMips() {
    std::printf("-- mips --\n");
    Spec s;
    s.w = 512, s.h = 512, s.flags = RGB, s.bits = 24, s.r = 0xff0000, s.g = 0xff00, s.b = 0xff;
    Rgba8Image  img;
    std::string why;
    check("a single-mip 512x512 gets the full 10-level chain",
          DecodeLegacy(Dds(s, Fill({ 0, 0, 0 }, 512 * 512)), false, img, why) && img.mips.size() == 10 && img.mips[9].size() == 4);

    // The file's own mips win over filtering: a 4x4 with a 2x2 mip of another colour.
    Spec m;
    m.mips = 2, m.flags = RGB, m.bits = 24, m.r = 0xff0000, m.g = 0xff00, m.b = 0xff;
    std::vector<std::uint8_t> two = Fill({ 0, 0, 255 }, 16);
    const auto level1 = Fill({ 255, 0, 0 }, 4);
    two.insert(two.end(), level1.begin(), level1.end());
    const auto withMip = Dds(m, two);
    check("a stored mip is used as-is", First(withMip, 0, 0, 255, 255, 1));
    check("levels past the stored ones are filtered from the last", First(withMip, 0, 0, 255, 255, 2));
    check("a declared mip that isn't there is generated instead", First(Dds(m, Fill({ 0, 0, 255 }, 16)), 255, 0, 0, 255, 1));

    // 2x2 → 1x1: black and white average to linear 0.5 (sRGB byte 188) or byte 128.
    const std::uint8_t checker[16] = { 0, 0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0 };
    const auto lin  = Downsample(checker, 2, 2, false);
    const auto srgb = Downsample(checker, 2, 2, true);
    check("linear box filter averages bytes", lin[0] == 128 && lin[3] == 128);
    check("sRGB box filter averages light (188), alpha stays linear (128)", srgb[0] == 188 && srgb[2] == 188 && srgb[3] == 128);
    const std::uint8_t row[12] = { 0, 0, 0, 0, 100, 100, 100, 100, 200, 200, 200, 200 };
    const auto clamp = Downsample(row, 3, 1, false);
    check("an odd width halves to 1 and stays in bounds", clamp.size() == 4 && clamp[0] == 50);
}

} // namespace

int main() {
    TestRouting();
    TestDecode();
    TestMips();
    std::printf(g_failures == 0 ? "\nALL PASSED\n" : "\n%d CHECK(S) FAILED\n", g_failures);
    return g_failures;
}
