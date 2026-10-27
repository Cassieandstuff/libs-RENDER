#pragma once
// Private: the Diligent-typed halves of the public pimpl classes, shared by the
// library's translation units (and the white-box tests).
#include <render/Diligent.h>

#include "EngineFactory.h"
#include "RefCntAutoPtr.hpp"
#include "DeviceContext.h"
#include "RenderDevice.h"
#include "SwapChain.h"
#include "Texture.h"

#include <string>

namespace render::diligent {

struct RenderDevice::Impl {
    Backend                                          backend = Backend::Vulkan;
    std::string                                      adapter;
    Diligent::RefCntAutoPtr<Diligent::IEngineFactory> factory;
    Diligent::RefCntAutoPtr<Diligent::IRenderDevice>  device;
    Diligent::RefCntAutoPtr<Diligent::IDeviceContext> context;
    bool wireframe = false;
    bool swizzle   = false;
};

struct ViewportSurface::Impl {
    RenderDevice*                                  owner  = nullptr;
    void*                                          window = nullptr;
    Diligent::RefCntAutoPtr<Diligent::ISwapChain>  swapChain;
};

struct OffscreenTarget::Impl {
    RenderDevice*                                 owner = nullptr;
    int                                           width = 0, height = 0;
    Diligent::RefCntAutoPtr<Diligent::ITexture>   color;     // RGBA8_UNORM_SRGB, render target
    Diligent::RefCntAutoPtr<Diligent::ITexture>   staging;   // CPU-readable copy
};

// While one of these is alive, Diligent's messages on this thread are held
// instead of printed, so an operation that fails can report itself in one line
// (Diligent logs a throw site and then its catch site, two lines per failure).
// Flush() prints what is held; the destructor flushes whatever is still held.
class DiligentLogCapture {
public:
    DiligentLogCapture();
    ~DiligentLogCapture();
    DiligentLogCapture(const DiligentLogCapture&)            = delete;
    DiligentLogCapture& operator=(const DiligentLogCapture&) = delete;

    void Add(bool error, std::string line, std::string message);
    const std::string& FirstError() const { return m_firstError; }   // the bare message text
    void Flush();     // print the held lines as usual
    void Discard();   // drop them

private:
    DiligentLogCapture* m_prev = nullptr;
    std::string         m_held;
    std::string         m_firstError;
};

// The colour format every output target (swap chain, off-screen) uses.
inline constexpr Diligent::TEXTURE_FORMAT kOutputFormat = Diligent::TEX_FORMAT_RGBA8_UNORM_SRGB;

} // namespace render::diligent
