#include "Internal.h"

#include "EngineFactoryD3D12.h"
#include "EngineFactoryVk.h"

#include <cstdio>
#include <cstring>
#include <vector>

namespace render::diligent {

using namespace Diligent;

const char* BackendName(Backend b) { return b == Backend::Vulkan ? "Vulkan" : "D3D12"; }

namespace {

thread_local DiligentLogCapture* t_capture = nullptr;

// Diligent's messages go to stderr (the app's log). INFO is chatty — adapter,
// extension and allocator reports on every device creation — so it's dropped.
void DILIGENT_CALL_TYPE OnDiligentMessage(DEBUG_MESSAGE_SEVERITY severity, const Char* message,
                                          const Char* function, const Char* file, int line) {
    if (severity == DEBUG_MESSAGE_SEVERITY_INFO || !message) return;
    // Unread vertex-shader outputs in DiligentFX permutations: harmless perf notes.
    if (std::strstr(message, "OutputNotConsumed")) return;
    const bool  error = severity != DEBUG_MESSAGE_SEVERITY_WARNING;
    const char* tag   = error ? "error" : "warning";
    std::string text = std::string("[Diligent] ") + tag + ": " + message;
    if (function && file) text += std::string(" (") + function + ", " + file + ":" + std::to_string(line) + ")";
    text += '\n';
    if (t_capture)
        t_capture->Add(error, std::move(text), message);
    else
        std::fputs(text.c_str(), stderr);
}

void RequestFeatures(EngineCreateInfo& ci, const DeviceOptions& opts) {
    ci.SetValidationLevel(opts.validation ? VALIDATION_LEVEL_1 : VALIDATION_LEVEL_DISABLED);
    ci.Features.WireframeFill           = DEVICE_FEATURE_STATE_OPTIONAL;
    ci.Features.TextureComponentSwizzle = DEVICE_FEATURE_STATE_OPTIONAL;
    ci.Features.AsyncShaderCompilation  = DEVICE_FEATURE_STATE_OPTIONAL;
}

bool TryCreate(Backend backend, const DeviceOptions& opts, RenderDevice::Impl& out, std::string& log) {
    if (backend == Backend::D3D12) {
        IEngineFactoryD3D12* factory = GetEngineFactoryD3D12();
        if (!factory || !factory->LoadD3D12()) { log += "D3D12: d3d12.dll could not be loaded\n"; return false; }
        factory->SetMessageCallback(OnDiligentMessage);
        EngineD3D12CreateInfo ci;
        RequestFeatures(ci, opts);
        if (opts.useWarp) {
            Uint32 n = 0;
            factory->EnumerateAdapters(Version{ 11, 0 }, n, nullptr);
            std::vector<GraphicsAdapterInfo> adapters(n);
            if (n) factory->EnumerateAdapters(Version{ 11, 0 }, n, adapters.data());
            for (Uint32 i = 0; i < n; ++i)
                if (adapters[i].Type == ADAPTER_TYPE_SOFTWARE) { ci.AdapterId = i; break; }
        }
        factory->CreateDeviceAndContextsD3D12(ci, &out.device, &out.context);
        out.factory = factory;
    } else {
        IEngineFactoryVk* factory = GetEngineFactoryVk();
        if (!factory) { log += "Vulkan: no engine factory\n"; return false; }
        factory->SetMessageCallback(OnDiligentMessage);
        EngineVkCreateInfo ci;
        RequestFeatures(ci, opts);
        factory->CreateDeviceAndContextsVk(ci, &out.device, &out.context);
        out.factory = factory;
    }
    if (!out.device || !out.context) {
        log += std::string(BackendName(backend)) + ": device creation failed\n";
        out = {};
        return false;
    }
    out.backend   = backend;
    out.adapter   = out.device->GetAdapterInfo().Description;
    const DeviceFeatures& f = out.device->GetDeviceInfo().Features;
    out.wireframe = f.WireframeFill == DEVICE_FEATURE_STATE_ENABLED;
    out.swizzle   = f.TextureComponentSwizzle == DEVICE_FEATURE_STATE_ENABLED;
    log += std::string(BackendName(backend)) + ": " + out.adapter + "\n";
    return true;
}

} // namespace

DiligentLogCapture::DiligentLogCapture() : m_prev(t_capture) { t_capture = this; }

DiligentLogCapture::~DiligentLogCapture() {
    t_capture = m_prev;
    Flush();
}

void DiligentLogCapture::Add(bool error, std::string line, std::string message) {
    if (error && m_firstError.empty()) m_firstError = std::move(message);
    m_held += line;
}

void DiligentLogCapture::Flush() {
    if (!m_held.empty()) std::fputs(m_held.c_str(), stderr);
    Discard();
}

void DiligentLogCapture::Discard() {
    m_held.clear();
    m_firstError.clear();
}

std::unique_ptr<RenderDevice> RenderDevice::Create(const DeviceOptions& opts, std::string* log) {
    std::string tried;
    auto impl = std::make_unique<Impl>();
    const Backend other = opts.preferred == Backend::Vulkan ? Backend::D3D12 : Backend::Vulkan;
    bool ok = TryCreate(opts.preferred, opts, *impl, tried);
    if (!ok && opts.allowFallback) ok = TryCreate(other, opts, *impl, tried);
    if (log) *log = tried;
    if (!ok) return nullptr;
    return std::unique_ptr<RenderDevice>(new RenderDevice(std::move(impl)));
}

RenderDevice::RenderDevice(std::unique_ptr<Impl> impl) : m_impl(std::move(impl)) {}

RenderDevice::~RenderDevice() {
    if (m_impl && m_impl->context) {
        m_impl->context->Flush();
        m_impl->context->WaitForIdle();
    }
}

Backend            RenderDevice::backend() const           { return m_impl->backend; }
const std::string& RenderDevice::adapterName() const       { return m_impl->adapter; }
bool               RenderDevice::supportsWireframe() const { return m_impl->wireframe; }
bool               RenderDevice::supportsSwizzle() const   { return m_impl->swizzle; }
RenderDevice::Impl& RenderDevice::impl()                   { return *m_impl; }

void RenderDevice::FinishFrame() {
    m_impl->context->Flush();   // FinishFrame expects no outstanding commands
    m_impl->context->FinishFrame();
}

void RenderDevice::WaitIdle() {
    m_impl->context->Flush();
    m_impl->context->WaitForIdle();
    m_impl->device->IdleGPU();
}

} // namespace render::diligent
