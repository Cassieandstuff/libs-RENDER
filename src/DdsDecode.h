#pragma once
// DDS header inspection, and a CPU decoder for the legacy uncompressed formats
// Diligent's DDS loader rejects or reads differently per backend (24-bit RGB,
// X8 colour, 16-bit packed, luminance, alpha-only). Pure C++: no Diligent, no
// graphics API — LoadTextureFromMemory (Textures.cpp) routes on Inspect() and
// uploads DecodeLegacy()'s RGBA8 mip chain itself.
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace render::diligent::dds {

enum class Route {
    NotDds,   // no "DDS " magic: Diligent's loader decides (PNG, JPEG, KTX, …)
    Native,   // a DDS that Diligent's loader reads as-is (BC, DX10, RGBA8/BGRA8, …)
    Decode,   // a legacy uncompressed DDS: DecodeLegacy() it to RGBA8
    Reject,   // corrupt, or a format neither path handles; `problem` says why
};

struct Inspection {
    Route       route = Route::NotDds;
    std::string format;    // the pixel format, readable ("legacy RGB 24bpp R=00ff0000 …"); empty for NotDds
    std::string problem;   // Route::Reject only
};

Inspection Inspect(std::span<const std::uint8_t> bytes);

// RGBA8, mip 0 first, every level tightly packed (width·4 bytes per row), the
// full chain down to 1×1.
struct Rgba8Image {
    std::uint32_t width = 0, height = 0;
    std::vector<std::vector<std::uint8_t>> mips;
};

// Decodes a Route::Decode file. Mips the file carries are decoded; the rest of
// the chain is box-filtered from the last one (colour averaged in linear light
// when `srgb`). False, with `problem` set, on anything else.
bool DecodeLegacy(std::span<const std::uint8_t> bytes, bool srgb, Rgba8Image& out, std::string& problem);

// One 2×2 box-filter step of a tightly packed RGBA8 image: max(w/2,1) × max(h/2,1)
// (an odd last row / column is clamped). Colour in linear light when `srgb`;
// alpha always linear.
std::vector<std::uint8_t> Downsample(const std::uint8_t* src, std::uint32_t w, std::uint32_t h, bool srgb);

} // namespace render::diligent::dds
