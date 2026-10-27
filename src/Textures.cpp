// Texture loading (DDS and anything else Diligent's TextureLoader reads, plus the
// legacy uncompressed DDS formats it doesn't — see DdsDecode.h), blend maps,
// shader views, and readback of renderer-produced textures.
#include "SceneRendererImpl.h"
#include "DdsDecode.h"

#include "GraphicsAccessories.hpp"
#include "TextureLoader.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>

namespace render::diligent {

using namespace Diligent;

namespace {

// The PBR shaders sample every material texture as a Texture2DArray.
RefCntAutoPtr<ITextureView> CreateArrayView(ITexture* tex, const TextureComponentMapping& swizzle = TextureComponentMapping::Identity()) {
    TextureViewDesc vd;
    vd.ViewType        = TEXTURE_VIEW_SHADER_RESOURCE;
    vd.TextureDim      = RESOURCE_DIM_TEX_2D_ARRAY;
    vd.FirstArraySlice = 0;
    vd.NumArraySlices  = 1;
    vd.Swizzle         = swizzle;
    RefCntAutoPtr<ITextureView> view;
    tex->CreateView(vd, &view);
    return view;
}

// A CPU-decoded image (its full RGBA8 mip chain) as an immutable texture.
TextureHandle UploadRgba8(DiligentSceneRenderer::Impl& r, const dds::Rgba8Image& img, bool srgb) {
    if (img.mips.empty()) return TextureHandle::Invalid;
    TextureDesc desc;
    desc.Name      = "render texture (decoded)";
    desc.Type      = RESOURCE_DIM_TEX_2D;
    desc.Width     = img.width;
    desc.Height    = img.height;
    desc.MipLevels = static_cast<Uint32>(img.mips.size());
    desc.Format    = srgb ? TEX_FORMAT_RGBA8_UNORM_SRGB : TEX_FORMAT_RGBA8_UNORM;
    desc.Usage     = USAGE_IMMUTABLE;
    desc.BindFlags = BIND_SHADER_RESOURCE;
    std::vector<TextureSubResData> subs(img.mips.size());
    for (std::size_t i = 0; i < img.mips.size(); ++i) {
        subs[i].pData  = img.mips[i].data();
        subs[i].Stride = std::max(img.width >> i, 1u) * 4;
    }
    const TextureData init{ subs.data(), static_cast<Uint32>(subs.size()) };
    GpuTexture        tex;
    r.device->CreateTexture(desc, &init, &tex.texture);
    return r.RegisterTexture(std::move(tex));
}

} // namespace

GpuTexture* DiligentSceneRenderer::Impl::FindTexture(TextureHandle h) {
    auto it = textures.find(static_cast<std::uint32_t>(h));
    return it == textures.end() ? nullptr : &it->second;
}

TextureHandle DiligentSceneRenderer::Impl::RegisterTexture(GpuTexture tex) {
    if (!tex.texture) return TextureHandle::Invalid;
    const TextureDesc& desc = tex.texture->GetDesc();
    tex.width  = static_cast<int>(desc.Width);
    tex.height = static_cast<int>(desc.Height);
    tex.srgb   = GetTextureFormatAttribs(desc.Format).ComponentType == COMPONENT_TYPE_UNORM_SRGB;
    if (!tex.arraySrv) tex.arraySrv = CreateArrayView(tex.texture);
    if (!tex.arraySrv) return TextureHandle::Invalid;
    const std::uint32_t id = nextTextureId++;   // never reused
    textures.emplace(id, std::move(tex));
    return static_cast<TextureHandle>(id);
}

// RMAOS (R=roughness G=metallic B=AO A=specular) re-ordered for the PBR shaders,
// which hard-wire roughness=.g, metallic=.b (physical-descriptor map), AO=.r and
// specular=.a: view.r←B, view.g←R, view.b←G, view.a←A.
ITextureView* DiligentSceneRenderer::Impl::RmaosView(GpuTexture& tex) {
    if (!owner.supportsSwizzle()) return nullptr;
    if (!tex.rmaosSrv) {
        tex.rmaosSrv = CreateArrayView(tex.texture, TextureComponentMapping{ TEXTURE_COMPONENT_SWIZZLE_B, TEXTURE_COMPONENT_SWIZZLE_R,
                                                                             TEXTURE_COMPONENT_SWIZZLE_G, TEXTURE_COMPONENT_SWIZZLE_A });
    }
    return tex.rmaosSrv;
}

// A greyscale specular mask (vanilla `_s.dds`) for the PBR shaders, which read the
// specular weight from .a: every channel ← R.
ITextureView* DiligentSceneRenderer::Impl::SpecularView(GpuTexture& tex) {
    if (!owner.supportsSwizzle()) return nullptr;
    if (!tex.specSrv) {
        tex.specSrv = CreateArrayView(tex.texture, TextureComponentMapping{ TEXTURE_COMPONENT_SWIZZLE_R, TEXTURE_COMPONENT_SWIZZLE_R,
                                                                            TEXTURE_COMPONENT_SWIZZLE_R, TEXTURE_COMPONENT_SWIZZLE_R });
    }
    return tex.specSrv;
}

TextureHandle DiligentSceneRenderer::LoadTexture(const std::string& path, TexColorSpace space) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        ++m_impl->failedTextureLoads;
        return TextureHandle::Invalid;
    }
    std::vector<uint8_t> bytes(static_cast<std::size_t>(f.tellg()));
    f.seekg(0);
    f.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!f) {
        ++m_impl->failedTextureLoads;
        return TextureHandle::Invalid;
    }
    return LoadTextureFromMemory(bytes, space);
}

// DDS files whose pixel format Diligent's loader rejects, or reads differently
// per backend, are decoded on the CPU to RGBA8 (DdsDecode.cpp); everything else
// goes through Diligent's loader. A failure is one log line: Diligent's own
// messages about it are captured and folded into that line.
TextureHandle DiligentSceneRenderer::LoadTextureFromMemory(const std::vector<uint8_t>& bytes, TexColorSpace space) {
    Impl& r = *m_impl;
    TextureHandle result = TextureHandle::Invalid;
    if (!bytes.empty()) {
        const bool            srgb = space == TexColorSpace::Srgb;
        const dds::Inspection ins  = dds::Inspect(bytes);
        DiligentLogCapture    capture;
        std::string           problem;
        switch (ins.route) {
        case dds::Route::Reject:
            problem = ins.problem;
            break;
        case dds::Route::Decode: {
            dds::Rgba8Image img;
            if (dds::DecodeLegacy(bytes, srgb, img, problem)) result = UploadRgba8(r, img, srgb);
            break;
        }
        case dds::Route::NotDds:
        case dds::Route::Native: {
            TextureLoadInfo info;
            info.Name   = "render texture";
            info.IsSRGB = srgb;
            RefCntAutoPtr<ITextureLoader> loader;
            CreateTextureLoaderFromMemory(bytes.data(), bytes.size(), false, info, &loader);
            if (loader) {
                GpuTexture tex;
                loader->CreateTexture(r.device, &tex.texture);
                result = r.RegisterTexture(std::move(tex));
            }
            break;
        }
        }
        if (result == TextureHandle::Invalid) {
            if (problem.empty()) problem = capture.FirstError().empty() ? "texture creation failed" : capture.FirstError();
            std::fprintf(stderr, "[render] texture load failed: %s%s%s%s\n", problem.c_str(),
                         ins.format.empty() ? "" : " (", ins.format.c_str(), ins.format.empty() ? "" : ")");
            capture.Discard();
        }
    }
    if (result == TextureHandle::Invalid) ++r.failedTextureLoads;
    return result;
}

TextureHandle DiligentSceneRenderer::UploadBlendMap(const float* data33x33) {
    if (!data33x33) return TextureHandle::Invalid;
    Impl& r = *m_impl;
    TextureDesc desc;
    desc.Name      = "render blend map";
    desc.Type      = RESOURCE_DIM_TEX_2D;
    desc.Width     = 33;
    desc.Height    = 33;
    desc.MipLevels = 1;
    desc.Format    = TEX_FORMAT_R32_FLOAT;
    desc.Usage     = USAGE_IMMUTABLE;
    desc.BindFlags = BIND_SHADER_RESOURCE;
    TextureSubResData sub{ data33x33, 33 * sizeof(float) };
    TextureData       init{ &sub, 1 };
    GpuTexture tex;
    r.device->CreateTexture(desc, &init, &tex.texture);
    return r.RegisterTexture(std::move(tex));
}

void DiligentSceneRenderer::FreeTexture(TextureHandle handle) {
    Impl& r = *m_impl;
    const auto id = static_cast<std::uint32_t>(handle);
    if (r.textures.erase(id)) PbrInvalidateTexture(r, id);
}

bool DiligentSceneRenderer::DebugReadbackRGBA8(TextureHandle handle, std::vector<std::uint8_t>& pixels, int& width, int& height) {
    Impl& r = *m_impl;
    GpuTexture* tex = r.FindTexture(handle);
    if (!tex || !tex->rendered) return false;

    TextureDesc sd = tex->texture->GetDesc();
    sd.Name           = "render readback staging";
    sd.MipLevels      = 1;
    sd.Usage          = USAGE_STAGING;
    sd.BindFlags      = BIND_NONE;
    sd.CPUAccessFlags = CPU_ACCESS_READ;
    sd.MiscFlags      = MISC_TEXTURE_FLAG_NONE;
    RefCntAutoPtr<ITexture> staging;
    r.device->CreateTexture(sd, nullptr, &staging);
    if (!staging) return false;

    CopyTextureAttribs copy{ tex->texture, RESOURCE_STATE_TRANSITION_MODE_TRANSITION, staging,
                             RESOURCE_STATE_TRANSITION_MODE_TRANSITION };
    r.context->CopyTexture(copy);
    r.context->WaitForIdle();

    MappedTextureSubresource mapped;
    r.context->MapTextureSubresource(staging, 0, 0, MAP_READ, MAP_FLAG_DO_NOT_WAIT, nullptr, mapped);
    if (!mapped.pData) return false;
    width  = static_cast<int>(sd.Width);
    height = static_cast<int>(sd.Height);
    pixels.resize(static_cast<std::size_t>(width) * height * 4);
    for (int y = 0; y < height; ++y)
        std::memcpy(pixels.data() + static_cast<std::size_t>(y) * width * 4,
                    static_cast<const std::uint8_t*>(mapped.pData) + static_cast<std::size_t>(y) * mapped.Stride,
                    static_cast<std::size_t>(width) * 4);
    r.context->UnmapTextureSubresource(staging, 0, 0);
    return true;
}

} // namespace render::diligent
