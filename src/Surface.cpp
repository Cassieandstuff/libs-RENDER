#include "Internal.h"

#include "EngineFactoryD3D12.h"
#include "EngineFactoryVk.h"
#include "NativeWindow.h"

namespace render::diligent {

using namespace Diligent;

std::unique_ptr<ViewportSurface> ViewportSurface::Create(RenderDevice& device, void* nativeWindow,
                                                         int width, int height, std::string* err) {
    RenderDevice::Impl& dev = device.impl();
    SwapChainDesc desc;
    desc.Width             = static_cast<Uint32>(width > 0 ? width : 1);
    desc.Height            = static_cast<Uint32>(height > 0 ? height : 1);
    desc.ColorBufferFormat = kOutputFormat;
    desc.DepthBufferFormat = TEX_FORMAT_UNKNOWN;   // the scene pass owns depth
    desc.BufferCount       = 3;
    desc.IsPrimary         = true;

    auto impl    = std::make_unique<Impl>();
    impl->owner  = &device;
    impl->window = nativeWindow;
    const Win32NativeWindow window{ nativeWindow };
    if (dev.backend == Backend::D3D12)
        static_cast<IEngineFactoryD3D12*>(dev.factory.RawPtr())
            ->CreateSwapChainD3D12(dev.device, dev.context, desc, FullScreenModeDesc{}, window, &impl->swapChain);
    else
        static_cast<IEngineFactoryVk*>(dev.factory.RawPtr())
            ->CreateSwapChainVk(dev.device, dev.context, desc, window, &impl->swapChain);
    if (!impl->swapChain) {
        if (err) *err = std::string(BackendName(dev.backend)) + ": swap chain creation failed";
        return nullptr;
    }
    return std::unique_ptr<ViewportSurface>(new ViewportSurface(std::move(impl)));
}

ViewportSurface::ViewportSurface(std::unique_ptr<Impl> impl) : m_impl(std::move(impl)) {}

ViewportSurface::~ViewportSurface() {
    // The swap chain's buffers may still be in flight.
    if (m_impl && m_impl->owner) m_impl->owner->WaitIdle();
}

void* ViewportSurface::nativeWindow() const { return m_impl->window; }
int   ViewportSurface::width() const  { return static_cast<int>(m_impl->swapChain->GetDesc().Width); }
int   ViewportSurface::height() const { return static_cast<int>(m_impl->swapChain->GetDesc().Height); }
ViewportSurface::Impl& ViewportSurface::impl() { return *m_impl; }

void ViewportSurface::Resize(int width, int height) {
    if (width <= 0 || height <= 0) return;   // minimised
    const SwapChainDesc& d = m_impl->swapChain->GetDesc();
    if (static_cast<int>(d.Width) == width && static_cast<int>(d.Height) == height) return;
    m_impl->swapChain->Resize(static_cast<Uint32>(width), static_cast<Uint32>(height));
}

void ViewportSurface::Present(int syncInterval) {
    m_impl->swapChain->Present(static_cast<Uint32>(syncInterval < 0 ? 0 : syncInterval));
}

} // namespace render::diligent
