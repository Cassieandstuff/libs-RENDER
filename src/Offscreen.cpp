#include "Internal.h"

#include <cmath>
#include <cstring>

namespace render::diligent {

using namespace Diligent;

namespace {

float SrgbToLinear(float c) {
    return c <= 0.04045f ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f);
}

} // namespace

std::unique_ptr<OffscreenTarget> OffscreenTarget::Create(RenderDevice& device, int width, int height) {
    if (width <= 0 || height <= 0) return nullptr;
    IRenderDevice* dev = device.impl().device;

    TextureDesc desc;
    desc.Name      = "OffscreenTarget color";
    desc.Type      = RESOURCE_DIM_TEX_2D;
    desc.Width     = static_cast<Uint32>(width);
    desc.Height    = static_cast<Uint32>(height);
    desc.Format    = kOutputFormat;
    desc.BindFlags = BIND_RENDER_TARGET | BIND_SHADER_RESOURCE;

    auto impl    = std::make_unique<Impl>();
    impl->owner  = &device;
    impl->width  = width;
    impl->height = height;
    dev->CreateTexture(desc, nullptr, &impl->color);

    desc.Name           = "OffscreenTarget readback";
    desc.Usage          = USAGE_STAGING;
    desc.BindFlags      = BIND_NONE;
    desc.CPUAccessFlags = CPU_ACCESS_READ;
    dev->CreateTexture(desc, nullptr, &impl->staging);
    if (!impl->color || !impl->staging) return nullptr;
    return std::unique_ptr<OffscreenTarget>(new OffscreenTarget(std::move(impl)));
}

OffscreenTarget::OffscreenTarget(std::unique_ptr<Impl> impl) : m_impl(std::move(impl)) {}
OffscreenTarget::~OffscreenTarget() = default;

int OffscreenTarget::width() const  { return m_impl->width; }
int OffscreenTarget::height() const { return m_impl->height; }
OffscreenTarget::Impl& OffscreenTarget::impl() { return *m_impl; }

void OffscreenTarget::Clear(float r, float g, float b, float a) {
    IDeviceContext* ctx = m_impl->owner->impl().context;
    ITextureView*   rtv = m_impl->color->GetDefaultView(TEXTURE_VIEW_RENDER_TARGET);
    // The target is sRGB: the clear value is linear and the hardware encodes it.
    const float linear[4] = { SrgbToLinear(r), SrgbToLinear(g), SrgbToLinear(b), a };
    ctx->SetRenderTargets(1, &rtv, nullptr, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    ctx->ClearRenderTarget(rtv, linear, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    ctx->SetRenderTargets(0, nullptr, nullptr, RESOURCE_STATE_TRANSITION_MODE_NONE);
}

std::vector<std::uint8_t> OffscreenTarget::ReadbackRGBA8() {
    IDeviceContext* ctx = m_impl->owner->impl().context;
    CopyTextureAttribs copy{ m_impl->color, RESOURCE_STATE_TRANSITION_MODE_TRANSITION,
                             m_impl->staging, RESOURCE_STATE_TRANSITION_MODE_TRANSITION };
    ctx->CopyTexture(copy);
    ctx->WaitForIdle();

    std::vector<std::uint8_t> pixels(static_cast<size_t>(m_impl->width) * m_impl->height * 4);
    MappedTextureSubresource mapped{};
    ctx->MapTextureSubresource(m_impl->staging, 0, 0, MAP_READ, MAP_FLAG_DO_NOT_WAIT, nullptr, mapped);
    if (!mapped.pData) return {};
    const size_t rowBytes = static_cast<size_t>(m_impl->width) * 4;
    for (int y = 0; y < m_impl->height; ++y)
        std::memcpy(pixels.data() + y * rowBytes,
                    static_cast<const std::uint8_t*>(mapped.pData) + static_cast<size_t>(y) * mapped.Stride, rowBytes);
    ctx->UnmapTextureSubresource(m_impl->staging, 0, 0);
    return pixels;
}

} // namespace render::diligent
