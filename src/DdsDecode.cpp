// DDS inspection + legacy uncompressed decode. See DdsDecode.h.
//
// Routing (legacy headers; anything with a DX10 header goes to Diligent):
//   FourCC DXT1-5 / ATI1-2 / BC4-5 / D3DFMT float codes        → Native
//   RGB 32bpp RGBA8 or BGRA8 (masks exact)                      → Native
//   RGB 32bpp R10G10B10A2 / R16G16 / R32F, LUMINANCE 16bpp L16 → Native (channels wider than 8 bits)
//   any other RGB / LUMINANCE / ALPHA layout with 8-, 16-, 24- or 32-bit pixels
//   and contiguous masks of ≤ 8 bits (B8G8R8 24-bit, X8R8G8B8, X8B8G8R8,
//   R5G6B5, A1R5G5B5, X1R5G5B5, A4R4G4B4, X4R4G4B4, R3G3B2, L8, A8L8, A4L4, A8) → Decode
//   volume / 1D textures, legacy uncompressed cube maps, palettes, YUV, unknown
//   FourCCs, truncated headers or pixel data                     → Reject
//
// Why the X8 and 16-bit layouts that Diligent accepts are decoded anyway: Vulkan
// has no BGRX format, so Diligent's B8G8R8X8 reads the padding byte as alpha
// there (vanilla's terrain LOD tiles store 0 in it); the packed 16-bit formats
// have no sRGB variant and are optional on some devices; and L8 / A8L8 would
// arrive as R8 / RG8, which sample as (L,0,0) / (L,A,0) instead of (L,L,L).
#include "DdsDecode.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace render::diligent::dds {

namespace {

constexpr std::uint32_t kMagic = 0x20534444u;   // "DDS "
constexpr std::size_t   kHeaderEnd = 4 + 124;    // magic + DDS_HEADER
constexpr std::size_t   kDx10End   = kHeaderEnd + 20;

constexpr std::uint32_t DDPF_ALPHAPIXELS = 0x1, DDPF_ALPHA = 0x2, DDPF_FOURCC = 0x4, DDPF_PALETTEINDEXED8 = 0x20,
                        DDPF_RGB = 0x40, DDPF_YUV = 0x200, DDPF_LUMINANCE = 0x20000;
constexpr std::uint32_t DDSD_DEPTH = 0x800000, DDSCAPS2_CUBEMAP = 0x200, DDSCAPS2_VOLUME = 0x200000;
constexpr std::uint32_t kMaxDim = 16384;

constexpr std::uint32_t FourCC(const char (&s)[5]) {
    return std::uint32_t(std::uint8_t(s[0])) | std::uint32_t(std::uint8_t(s[1])) << 8 | std::uint32_t(std::uint8_t(s[2])) << 16 |
           std::uint32_t(std::uint8_t(s[3])) << 24;
}

struct PixelFormat {
    std::uint32_t flags = 0, fourCC = 0, bits = 0, r = 0, g = 0, b = 0, a = 0;
};

struct Header {
    std::uint32_t flags = 0, height = 0, width = 0, depth = 0, mips = 0, caps2 = 0;
    PixelFormat   pf;
    bool          dx10 = false;
    std::uint32_t dxgi = 0, dim = 0;
};

std::uint32_t U32(std::span<const std::uint8_t> b, std::size_t at) {
    std::uint32_t v;
    std::memcpy(&v, b.data() + at, 4);
    return v;
}

// Parses magic + header (+ DX10 extension). Precondition: the "DDS " magic.
bool ParseHeader(std::span<const std::uint8_t> b, Header& h, std::string& problem) {
    if (b.size() < kHeaderEnd) {
        problem = "truncated header (" + std::to_string(b.size()) + " bytes)";
        return false;
    }
    if (U32(b, 4) != 124 || U32(b, 4 + 72) != 32) {
        problem = "invalid header sizes";
        return false;
    }
    h.flags     = U32(b, 4 + 4);
    h.height    = U32(b, 4 + 8);
    h.width     = U32(b, 4 + 12);
    h.depth     = U32(b, 4 + 20);
    h.mips      = U32(b, 4 + 24);
    h.pf.flags  = U32(b, 4 + 76);
    h.pf.fourCC = U32(b, 4 + 80);
    h.pf.bits   = U32(b, 4 + 84);
    h.pf.r      = U32(b, 4 + 88);
    h.pf.g      = U32(b, 4 + 92);
    h.pf.b      = U32(b, 4 + 96);
    h.pf.a      = U32(b, 4 + 100);
    h.caps2     = U32(b, 4 + 108);
    h.dx10      = (h.pf.flags & DDPF_FOURCC) && h.pf.fourCC == FourCC("DX10");
    if (h.dx10) {
        if (b.size() < kDx10End) {
            problem = "truncated DX10 header";
            return false;
        }
        h.dxgi = U32(b, kHeaderEnd);
        h.dim  = U32(b, kHeaderEnd + 4);
    }
    return true;
}

std::string Describe(const Header& h) {
    char s[160];
    if (h.dx10) {
        std::snprintf(s, sizeof s, "DX10 DXGI format %u", h.dxgi);
    } else if (h.pf.flags & DDPF_FOURCC) {
        char cc[5] = {};
        std::memcpy(cc, &h.pf.fourCC, 4);
        if (std::all_of(cc, cc + 4, [](char c) { return c >= 32 && c < 127; }))
            std::snprintf(s, sizeof s, "FourCC '%s'", cc);
        else
            std::snprintf(s, sizeof s, "FourCC %u", h.pf.fourCC);
    } else {
        std::string f;
        if (h.pf.flags & DDPF_RGB) f += "RGB|";
        if (h.pf.flags & DDPF_LUMINANCE) f += "LUMINANCE|";
        if (h.pf.flags & DDPF_ALPHA) f += "ALPHA|";
        if (h.pf.flags & DDPF_ALPHAPIXELS) f += "ALPHAPIXELS|";
        if (h.pf.flags & DDPF_PALETTEINDEXED8) f += "PAL8|";
        if (h.pf.flags & DDPF_YUV) f += "YUV|";
        if (f.empty()) {
            char x[24];
            std::snprintf(x, sizeof x, "flags=0x%x|", h.pf.flags);
            f = x;
        }
        f.pop_back();
        std::snprintf(s, sizeof s, "legacy %s %ubpp R=%08x G=%08x B=%08x A=%08x", f.c_str(), h.pf.bits, h.pf.r, h.pf.g, h.pf.b, h.pf.a);
    }
    return s;
}

bool IsMask(const PixelFormat& p, std::uint32_t r, std::uint32_t g, std::uint32_t b, std::uint32_t a) {
    return p.r == r && p.g == g && p.b == b && p.a == a;
}

// Uncompressed legacy layouts Diligent reads correctly on every backend.
bool NativeUncompressed(const PixelFormat& p) {
    if ((p.flags & DDPF_RGB) && p.bits == 32)
        return IsMask(p, 0xff, 0xff00, 0xff0000, 0xff000000) || IsMask(p, 0xff0000, 0xff00, 0xff, 0xff000000) ||
               IsMask(p, 0x3ff00000, 0xffc00, 0x3ff, 0xc0000000) || IsMask(p, 0xffff, 0xffff0000, 0, 0) ||
               IsMask(p, 0xffffffff, 0, 0, 0);
    if (!(p.flags & DDPF_RGB) && (p.flags & DDPF_LUMINANCE) && p.bits == 16) return IsMask(p, 0xffff, 0, 0, 0);
    return false;
}

// FourCCs Diligent's DDS loader maps to a DXGI format.
bool NativeFourCC(std::uint32_t cc) {
    static constexpr std::uint32_t known[] = { FourCC("DXT1"), FourCC("DXT2"), FourCC("DXT3"), FourCC("DXT4"), FourCC("DXT5"),
                                               FourCC("ATI1"), FourCC("BC4U"), FourCC("BC4S"), FourCC("ATI2"), FourCC("BC5U"),
                                               FourCC("BC5S"), FourCC("RGBG"), FourCC("GRGB"), 36, 110, 111, 112, 113, 114, 115, 116 };
    return std::find(std::begin(known), std::end(known), cc) != std::end(known);
}

// One channel: where it sits in the pixel and how to widen it to 8 bits.
struct Channel {
    std::uint32_t mask = 0, shift = 0, max = 0;   // max = 0: channel absent
    std::uint8_t  Get(std::uint32_t px) const {
        if (!max) return 0;
        const std::uint32_t v = (px & mask) >> shift;
        return static_cast<std::uint8_t>((v * 255u + max / 2) / max);
    }
};

bool MakeChannel(std::uint32_t mask, std::uint32_t bits, Channel& c) {
    c = {};
    if (!mask) return true;
    if (bits < 32 && (mask >> bits) != 0) return false;   // outside the pixel
    const std::uint32_t shift = static_cast<std::uint32_t>(std::countr_zero(mask));
    const std::uint32_t v     = mask >> shift;
    if ((v & (v + 1)) != 0) return false;                 // not contiguous
    if (std::popcount(v) > 8) return false;               // wider than 8 bits: not ours
    c = { mask, shift, v };
    return true;
}

struct Layout {
    Channel r, g, b, a;
    bool    luminance = false, alphaOnly = false;
};

// The decodable layout of a legacy uncompressed format, if it is one.
bool DecodeLayout(const PixelFormat& p, Layout& l) {
    if (p.flags & (DDPF_FOURCC | DDPF_PALETTEINDEXED8 | DDPF_YUV)) return false;
    if (p.bits != 8 && p.bits != 16 && p.bits != 24 && p.bits != 32) return false;
    l = {};
    if (p.flags & DDPF_RGB) {
        if (!MakeChannel(p.r, p.bits, l.r) || !MakeChannel(p.g, p.bits, l.g) || !MakeChannel(p.b, p.bits, l.b) ||
            !MakeChannel(p.a, p.bits, l.a))
            return false;
        return l.r.max || l.g.max || l.b.max;
    }
    if (p.flags & DDPF_LUMINANCE) {
        l.luminance = true;
        if (!MakeChannel(p.r, p.bits, l.r) || !MakeChannel(p.a, p.bits, l.a)) return false;
        return l.r.max != 0;
    }
    if (p.flags & DDPF_ALPHA) {
        l.alphaOnly = true;
        const std::uint32_t mask = p.a ? p.a : (p.bits == 32 ? 0xffffffffu : (1u << p.bits) - 1u);
        return MakeChannel(mask, p.bits, l.a) && l.a.max != 0;
    }
    return false;
}

std::uint64_t LevelBytes(std::uint32_t w, std::uint32_t h, std::uint32_t bits) {
    return std::uint64_t(w) * h * (bits / 8);
}

std::uint32_t FullChain(std::uint32_t w, std::uint32_t h) {
    return static_cast<std::uint32_t>(std::bit_width(std::max(w, h)));
}

const std::array<float, 256>& SrgbToLinear() {
    static const std::array<float, 256> lut = [] {
        std::array<float, 256> t{};
        for (int i = 0; i < 256; ++i) {
            const double v = i / 255.0;
            t[i] = static_cast<float>(v <= 0.04045 ? v / 12.92 : std::pow((v + 0.055) / 1.055, 2.4));
        }
        return t;
    }();
    return lut;
}

std::uint8_t LinearToSrgbByte(float v) {
    v = std::clamp(v, 0.f, 1.f);
    const double e = v <= 0.0031308 ? v * 12.92 : 1.055 * std::pow(static_cast<double>(v), 1.0 / 2.4) - 0.055;
    return static_cast<std::uint8_t>(std::lround(e * 255.0));
}

} // namespace

Inspection Inspect(std::span<const std::uint8_t> bytes) {
    Inspection out;
    if (bytes.size() < 4 || U32(bytes, 0) != kMagic) return out;   // NotDds

    auto reject = [&](std::string why) {
        out.route   = Route::Reject;
        out.problem = std::move(why);
        return out;
    };
    Header h;
    std::string problem;
    if (!ParseHeader(bytes, h, problem)) return reject(problem);
    out.format = Describe(h);
    if (h.width == 0 || h.height == 0) return reject("zero size");
    if (h.width > kMaxDim || h.height > kMaxDim) return reject("dimensions out of range");

    if (h.dx10) {
        if (h.dim != 3) return reject(h.dim == 4 ? "volume textures are not supported" : "only 2D textures are supported");
        out.route = Route::Native;
        return out;
    }
    if ((h.flags & DDSD_DEPTH) || (h.caps2 & DDSCAPS2_VOLUME)) return reject("volume textures are not supported");
    if (h.pf.flags & DDPF_FOURCC) {
        if (!NativeFourCC(h.pf.fourCC)) return reject("unsupported pixel format");
        out.route = Route::Native;
        return out;
    }
    if (NativeUncompressed(h.pf)) {
        out.route = Route::Native;
        return out;
    }
    Layout layout;
    if (!DecodeLayout(h.pf, layout)) return reject("unsupported pixel format");
    if (h.caps2 & DDSCAPS2_CUBEMAP) return reject("cube maps in this pixel format are not supported");
    if (bytes.size() - kHeaderEnd < LevelBytes(h.width, h.height, h.pf.bits)) return reject("truncated pixel data");
    out.route = Route::Decode;
    return out;
}

bool DecodeLegacy(std::span<const std::uint8_t> bytes, bool srgb, Rgba8Image& out, std::string& problem) {
    const Inspection ins = Inspect(bytes);
    if (ins.route != Route::Decode) {
        problem = ins.route == Route::Reject ? ins.problem : "not a legacy uncompressed DDS";
        return false;
    }
    Header h;
    Layout l;
    ParseHeader(bytes, h, problem);
    DecodeLayout(h.pf, l);

    const std::uint32_t chain    = FullChain(h.width, h.height);
    const std::uint32_t declared = std::min(std::max(h.mips, 1u), chain);
    const std::uint32_t bpp      = h.pf.bits / 8;

    out        = {};
    out.width  = h.width;
    out.height = h.height;
    out.mips.reserve(chain);

    // The file's own levels, as many as are declared and present.
    std::size_t   at = kHeaderEnd;
    std::uint32_t w = h.width, ht = h.height;
    for (std::uint32_t level = 0; level < declared; ++level) {
        const std::uint64_t n = LevelBytes(w, ht, h.pf.bits);
        if (bytes.size() - at < n) break;   // level 0 is known to fit (Inspect)
        std::vector<std::uint8_t> rgba(std::size_t(w) * ht * 4);
        const std::uint8_t* src = bytes.data() + at;
        for (std::size_t i = 0, count = std::size_t(w) * ht; i < count; ++i, src += bpp) {
            std::uint32_t px = 0;
            std::memcpy(&px, src, bpp);   // little-endian
            std::uint8_t* d = rgba.data() + i * 4;
            if (l.alphaOnly) {
                d[0] = d[1] = d[2] = 0;
                d[3] = l.a.Get(px);
            } else if (l.luminance) {
                d[0] = d[1] = d[2] = l.r.Get(px);
                d[3] = l.a.max ? l.a.Get(px) : 255;
            } else {
                d[0] = l.r.Get(px);
                d[1] = l.g.Get(px);
                d[2] = l.b.Get(px);
                d[3] = l.a.max ? l.a.Get(px) : 255;
            }
        }
        out.mips.push_back(std::move(rgba));
        at += static_cast<std::size_t>(n);
        w  = std::max(w / 2, 1u);
        ht = std::max(ht / 2, 1u);
    }

    // The rest of the chain, filtered from the last decoded level.
    while (out.mips.size() < chain) {
        const std::uint32_t level = static_cast<std::uint32_t>(out.mips.size()) - 1;
        const std::uint32_t lw = std::max(h.width >> level, 1u), lh = std::max(h.height >> level, 1u);
        out.mips.push_back(Downsample(out.mips.back().data(), lw, lh, srgb));
    }
    return true;
}

std::vector<std::uint8_t> Downsample(const std::uint8_t* src, std::uint32_t w, std::uint32_t h, bool srgb) {
    const std::uint32_t dw = std::max(w / 2, 1u), dh = std::max(h / 2, 1u);
    std::vector<std::uint8_t> dst(std::size_t(dw) * dh * 4);
    const auto& lin = SrgbToLinear();
    for (std::uint32_t y = 0; y < dh; ++y) {
        const std::uint32_t y0 = std::min(2 * y, h - 1), y1 = std::min(2 * y + 1, h - 1);
        for (std::uint32_t x = 0; x < dw; ++x) {
            const std::uint32_t x0 = std::min(2 * x, w - 1), x1 = std::min(2 * x + 1, w - 1);
            const std::uint8_t* p[4] = { src + (std::size_t(y0) * w + x0) * 4, src + (std::size_t(y0) * w + x1) * 4,
                                         src + (std::size_t(y1) * w + x0) * 4, src + (std::size_t(y1) * w + x1) * 4 };
            std::uint8_t* d = dst.data() + (std::size_t(y) * dw + x) * 4;
            for (int c = 0; c < 4; ++c) {
                if (srgb && c < 3) {
                    const float avg = (lin[p[0][c]] + lin[p[1][c]] + lin[p[2][c]] + lin[p[3][c]]) * 0.25f;
                    d[c] = LinearToSrgbByte(avg);
                } else {
                    d[c] = static_cast<std::uint8_t>((p[0][c] + p[1][c] + p[2][c] + p[3][c] + 2) / 4);
                }
            }
        }
    }
    return dst;
}

} // namespace render::diligent::dds
