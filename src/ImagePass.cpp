// IImageRenderer: persistent targets (R6), fullscreen passes from caller-supplied HLSL (R5) and float readback with
// mips (R9). A pass is RENDER's fullscreen triangle (its own vertex shader, writing TEXCOORD0) and the caller's pixel
// shader, compiled with separate textures and samplers. Its cbuffers are found by reflection and packed here, so the
// caller sets constants by name. Everything runs at once on the immediate context.
#include "SceneRendererImpl.h"

#include "GraphicsAccessories.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>

namespace render::diligent {

using namespace Diligent;

namespace {

// The fullscreen triangle: uv 0..1 across the target, origin top-left (v down), into TEXCOORD0.
constexpr const char* kPassVS = R"(
struct PassVSOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; };
PassVSOut main(uint id : SV_VertexID) {
    PassVSOut o;
    float2 uv = float2((id << 1) & 2, id & 2);
    o.uv  = uv;
    o.pos = float4(uv * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
    return o;
}
)";

constexpr int kMaxTargetSize = 16384;

TEXTURE_FORMAT ToDiligent(TargetFormat f) {
    switch (f) {
    case TargetFormat::R16F:    return TEX_FORMAT_R16_FLOAT;
    case TargetFormat::RG16F:   return TEX_FORMAT_RG16_FLOAT;
    case TargetFormat::RGBA16F: return TEX_FORMAT_RGBA16_FLOAT;
    case TargetFormat::R32F:    return TEX_FORMAT_R32_FLOAT;
    case TargetFormat::RG32F:   return TEX_FORMAT_RG32_FLOAT;
    case TargetFormat::RGBA32F: return TEX_FORMAT_RGBA32_FLOAT;
    case TargetFormat::RGBA8:   return TEX_FORMAT_RGBA8_UNORM;
    }
    return TEX_FORMAT_UNKNOWN;
}

const char* FormatName(TargetFormat f) {
    switch (f) {
    case TargetFormat::R16F:    return "R16F";
    case TargetFormat::RG16F:   return "RG16F";
    case TargetFormat::RGBA16F: return "RGBA16F";
    case TargetFormat::R32F:    return "R32F";
    case TargetFormat::RG32F:   return "RG32F";
    case TargetFormat::RGBA32F: return "RGBA32F";
    case TargetFormat::RGBA8:   return "RGBA8";
    }
    return "?";
}

int Channels(TargetFormat f) {
    switch (f) {
    case TargetFormat::R16F: case TargetFormat::R32F:   return 1;
    case TargetFormat::RG16F: case TargetFormat::RG32F: return 2;
    default:                                            return 4;
    }
}

// Bytes per channel: 1 (RGBA8), 2 (half) or 4 (float).
int ChannelBytes(TargetFormat f) {
    switch (f) {
    case TargetFormat::RGBA8:                                                       return 1;
    case TargetFormat::R16F: case TargetFormat::RG16F: case TargetFormat::RGBA16F: return 2;
    default:                                                                        return 4;
    }
}

float HalfToFloat(std::uint16_t h) {
    const int   sign = (h >> 15) & 1;
    const int   exp  = (h >> 10) & 0x1f;
    const int   man  = h & 0x3ff;
    float       v;
    if (exp == 0)       v = std::ldexp(static_cast<float>(man), -24);                       // zero, subnormal
    else if (exp == 31) v = man ? std::numeric_limits<float>::quiet_NaN() : std::numeric_limits<float>::infinity();
    else                v = std::ldexp(static_cast<float>(man | 0x400), exp - 25);           // (1 + m/1024)·2^(e-15)
    return sign ? -v : v;
}

int MipCount(int w, int h) {
    int n = 1;
    while ((w | h) > 1) {
        w = std::max(w >> 1, 1);
        h = std::max(h >> 1, 1);
        ++n;
    }
    return n;
}

bool Fail(std::string* error, std::string text) {
    if (error) *error = std::move(text);
    return false;
}

bool IsIntType(SHADER_CODE_BASIC_TYPE t) {
    return t == SHADER_CODE_BASIC_TYPE_INT || t == SHADER_CODE_BASIC_TYPE_UINT || t == SHADER_CODE_BASIC_TYPE_BOOL;
}

// The compiler's text from CreateShader's output blob: its first string (the second is the full source).
std::string CompilerText(IDataBlob* blob) {
    if (!blob || blob->GetSize() == 0) return {};
    const char* p = blob->GetConstDataPtr<char>();
    std::string text(p, strnlen(p, blob->GetSize()));
    while (!text.empty() && (text.back() == '\n' || text.back() == '\r' || text.back() == ' ')) text.pop_back();
    return text;
}

} // namespace

// ── Pass state ────────────────────────────────────────────────────────────────

struct ImagePass {
    std::string  name;
    TargetFormat format = TargetFormat::RGBA16F;
    RefCntAutoPtr<IPipelineState>         pso;
    RefCntAutoPtr<IShaderResourceBinding> srb;

    struct CBuffer {
        std::string                name;
        RefCntAutoPtr<IBuffer>     buffer;
        std::vector<std::uint8_t> data;
        bool                       dirty = true;
    };
    struct Constant {
        PassConstant info;
        std::size_t  cbuffer   = 0;
        Uint32       offset    = 0;
        bool         supported = true;   // a 4-byte float / int / uint / bool scalar or vector (or an array of them)
    };
    struct Texture {
        std::string   name;
        TextureHandle bound = TextureHandle::Invalid;
    };
    std::vector<CBuffer>     cbuffers;
    std::vector<Constant>    constants;
    std::vector<Texture>     textures;
    std::vector<std::string> samplers;

    std::string Prefix() const { return "pass '" + name + "': "; }
};

struct ImagePassState {
    RefCntAutoPtr<IShader>                       vs;
    std::unordered_map<std::uint32_t, ImagePass> passes;
    std::uint32_t                                nextPass = 1;   // never reused
};

void DestroyImagePassState(ImagePassState* p) { delete p; }

ImagePassPtr CreateImagePassState(DiligentSceneRenderer::Impl& r) {
    ImagePassPtr st(new ImagePassState{}, &DestroyImagePassState);
    // Separate textures and samplers, as the caller's pixel shaders are compiled (a pipeline's shaders agree on it).
    ShaderCreateInfo ci;
    ci.SourceLanguage = SHADER_SOURCE_LANGUAGE_HLSL;
    ci.Desc           = { "render image pass VS", SHADER_TYPE_VERTEX, false };
    ci.EntryPoint     = "main";
    ci.Source         = kPassVS;
    r.device->CreateShader(ci, &st->vs);
    if (!st->vs) std::fprintf(stderr, "[render] the image pass vertex shader failed to compile\n");
    return st;
}

namespace {

ImagePass* FindPass(DiligentSceneRenderer::Impl& r, PassHandle h) {
    if (!r.imagePasses) return nullptr;
    auto it = r.imagePasses->passes.find(static_cast<std::uint32_t>(h));
    return it == r.imagePasses->passes.end() ? nullptr : &it->second;
}

// Level `mip`'s render-target view, made on first use.
ITextureView* MipRtv(GpuTexture& t, int mip) {
    if (t.mipRtvs.size() <= static_cast<std::size_t>(mip)) t.mipRtvs.resize(static_cast<std::size_t>(mip) + 1);
    RefCntAutoPtr<ITextureView>& view = t.mipRtvs[static_cast<std::size_t>(mip)];
    if (!view) {
        TextureViewDesc vd;
        vd.ViewType        = TEXTURE_VIEW_RENDER_TARGET;
        vd.TextureDim      = RESOURCE_DIM_TEX_2D;
        vd.MostDetailedMip = static_cast<Uint32>(mip);
        vd.NumMipLevels    = 1;
        t.texture->CreateView(vd, &view);
    }
    return view;
}

int Levels(const GpuTexture& t) { return static_cast<int>(t.texture->GetDesc().MipLevels); }

// After a pass or a clear: nothing stays bound as a render target (a later read or sample transitions it freely),
// and a frame in progress gets its scene targets back.
void Unbind(DiligentSceneRenderer::Impl& r) {
    r.context->SetRenderTargets(0, nullptr, nullptr, RESOURCE_STATE_TRANSITION_MODE_NONE);
    if (r.inFrame) r.BindSceneTargets();
}

SamplerDesc ToSampler(const PassSampler* s) {
    const FILTER_TYPE          filter  = s && s->filter == SamplerFilter::Point ? FILTER_TYPE_POINT : FILTER_TYPE_LINEAR;
    TEXTURE_ADDRESS_MODE       address = TEXTURE_ADDRESS_WRAP;
    if (s && s->address == SamplerAddress::Clamp) address = TEXTURE_ADDRESS_CLAMP;
    if (s && s->address == SamplerAddress::Mirror) address = TEXTURE_ADDRESS_MIRROR;
    SamplerDesc d;
    d.MinFilter = d.MagFilter = d.MipFilter = filter;
    d.AddressU = d.AddressV = d.AddressW = address;
    return d;
}

} // namespace

// ── Targets ───────────────────────────────────────────────────────────────────

TextureHandle DiligentSceneRenderer::CreateTarget(const TargetDesc& desc, std::string* error) {
    Impl& r = *m_impl;
    if (desc.width < 1 || desc.height < 1 || desc.width > kMaxTargetSize || desc.height > kMaxTargetSize) {
        Fail(error, "target: size " + std::to_string(desc.width) + "x" + std::to_string(desc.height) + " is outside 1..16384");
        return TextureHandle::Invalid;
    }
    const int full = MipCount(desc.width, desc.height);
    if (desc.mipLevels < 0 || desc.mipLevels > full) {
        Fail(error, "target: " + std::to_string(desc.mipLevels) + " mip levels asked for; a " + std::to_string(desc.width) + "x" +
                        std::to_string(desc.height) + " target has at most " + std::to_string(full));
        return TextureHandle::Invalid;
    }
    const int levels = desc.mipLevels == 0 ? full : desc.mipLevels;

    TextureDesc td;
    td.Name      = "render image target";
    td.Type      = RESOURCE_DIM_TEX_2D;
    td.Width     = static_cast<Uint32>(desc.width);
    td.Height    = static_cast<Uint32>(desc.height);
    td.MipLevels = static_cast<Uint32>(levels);
    td.Format    = ToDiligent(desc.format);
    td.BindFlags = BIND_RENDER_TARGET | BIND_SHADER_RESOURCE;
    td.MiscFlags = levels > 1 ? MISC_TEXTURE_FLAG_GENERATE_MIPS : MISC_TEXTURE_FLAG_NONE;

    GpuTexture tex;
    {
        DiligentLogCapture capture;
        r.device->CreateTexture(td, nullptr, &tex.texture);
        if (!tex.texture) {
            Fail(error, std::string("target: the ") + FormatName(desc.format) + " texture could not be created" +
                            (capture.FirstError().empty() ? "" : ": " + capture.FirstError()));
            capture.Discard();
            return TextureHandle::Invalid;
        }
    }
    tex.target       = true;
    tex.targetFormat = desc.format;
    const TextureHandle h = r.RegisterTexture(std::move(tex));
    if (h == TextureHandle::Invalid) Fail(error, "target: its shader view could not be created");
    return h;
}

bool DiligentSceneRenderer::ClearTarget(TextureHandle target, const glm::vec4& value, int mip, std::string* error) {
    Impl&       r = *m_impl;
    GpuTexture* t = r.FindTexture(target);
    if (!t || !t->target) return Fail(error, "clear: not a target");
    const int levels = Levels(*t);
    if (mip < -1 || mip >= levels) return Fail(error, "clear: level " + std::to_string(mip) + " of " + std::to_string(levels));
    const float rgba[4] = { value.r, value.g, value.b, value.a };
    for (int m = mip < 0 ? 0 : mip; m < (mip < 0 ? levels : mip + 1); ++m) {
        ITextureView* rtv = MipRtv(*t, m);
        if (!rtv) return Fail(error, "clear: no render-target view for level " + std::to_string(m));
        r.context->SetRenderTargets(1, &rtv, nullptr, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
        r.context->ClearRenderTarget(rtv, rgba, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    }
    Unbind(r);
    return true;
}

bool DiligentSceneRenderer::GenerateMips(TextureHandle target, std::string* error) {
    Impl&       r = *m_impl;
    GpuTexture* t = r.FindTexture(target);
    if (!t || !t->target) return Fail(error, "mips: not a target");
    if (Levels(*t) < 2) return true;   // nothing below level 0
    r.context->GenerateMips(t->texture->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE));
    return true;
}

// ── Passes ────────────────────────────────────────────────────────────────────

PassResult DiligentSceneRenderer::CreatePass(const PassDesc& desc) {
    Impl&            r = *m_impl;
    PassResult       result;
    const std::string name   = desc.name.empty() ? std::string("image pass") : desc.name;
    const std::string header = "pass '" + name + "':\n";
    auto fail = [&](std::string text) {
        result.pass     = PassHandle::Invalid;
        result.messages = header + text;
        return result;
    };
    if (!r.imagePasses || !r.imagePasses->vs) return fail("the image pass vertex shader isn't available");
    if (desc.pixelShader.empty()) return fail("no pixel shader source");

    // Compile the caller's source as it is: Diligent puts its own definitions ahead of a `#line 1` (Direct3D) or in a
    // separate preamble (glslang), so the compiler's line numbers are the caller's.
    ShaderCreateInfo ci;
    ci.SourceLanguage               = SHADER_SOURCE_LANGUAGE_HLSL;
    ci.Desc                         = { name.c_str(), SHADER_TYPE_PIXEL, false };
    ci.EntryPoint                   = desc.entryPoint.empty() ? "main" : desc.entryPoint.c_str();
    ci.Source                       = desc.pixelShader.c_str();
    ci.SourceLength                 = desc.pixelShader.size();
    ci.LoadConstantBufferReflection = true;
    RefCntAutoPtr<IShader>   ps;
    RefCntAutoPtr<IDataBlob> output;
    std::string              compiler;
    {
        DiligentLogCapture capture;   // the compiler's text comes back in `messages`, not on the log
        r.device->CreateShader(ci, &ps, &output);
        compiler = CompilerText(output);
        if (!ps && compiler.empty()) compiler = capture.FirstError().empty() ? "the shader failed to compile" : capture.FirstError();
        capture.Discard();
    }
    if (!ps) return fail(compiler);
    if (!compiler.empty()) result.messages = header + compiler;   // warnings beside a valid pass

    ImagePass pass;
    pass.name   = name;
    pass.format = desc.format;

    // What the shader declares.
    const Uint32 resources = ps->GetResourceCount();
    for (Uint32 i = 0; i < resources; ++i) {
        ShaderResourceDesc rd;
        ps->GetResourceDesc(i, rd);
        const std::string rname = rd.Name ? rd.Name : "";
        switch (rd.Type) {
        case SHADER_RESOURCE_TYPE_CONSTANT_BUFFER: {
            const ShaderCodeBufferDesc* bd = ps->GetConstantBufferDesc(i);
            if (!bd) return fail("no reflection for cbuffer '" + rname + "'");
            ImagePass::CBuffer cb;
            cb.name = rname;
            cb.data.assign(std::max<std::size_t>((bd->Size + 15u) & ~15u, 16), 0);
            for (Uint32 v = 0; v < bd->NumVariables; ++v) {
                const ShaderCodeVariableDesc& var = bd->pVariables[v];
                ImagePass::Constant c;
                c.info.name      = var.Name ? var.Name : "";
                c.info.isInt     = IsIntType(var.BasicType);
                c.info.arraySize = var.ArraySize > 1 ? static_cast<int>(var.ArraySize) : 0;
                c.cbuffer        = pass.cbuffers.size();
                c.offset         = var.Offset;
                switch (var.Class) {
                case SHADER_CODE_VARIABLE_CLASS_SCALAR:
                    c.info.components = 1;
                    break;
                case SHADER_CODE_VARIABLE_CLASS_VECTOR:
                    // A vector's size is NumColumns (NumRows from GLSL-style reflection): take whichever is set.
                    c.info.components = std::max<int>(std::max<int>(var.NumColumns, var.NumRows), 1);
                    break;
                case SHADER_CODE_VARIABLE_CLASS_MATRIX_ROWS:
                case SHADER_CODE_VARIABLE_CLASS_MATRIX_COLUMNS:
                    c.info.components = std::max(var.NumRows * var.NumColumns, 1);
                    c.info.matrix     = true;
                    c.supported       = false;
                    break;
                default:   // a struct, or unknown
                    c.supported = false;
                    break;
                }
                if (!c.info.isInt && var.BasicType != SHADER_CODE_BASIC_TYPE_FLOAT) c.supported = false;   // half, double ...
                pass.constants.push_back(std::move(c));
            }
            pass.cbuffers.push_back(std::move(cb));
            break;
        }
        case SHADER_RESOURCE_TYPE_TEXTURE_SRV:
            if (rd.ArraySize > 1) return fail("texture '" + rname + "' is an array; image passes bind single textures");
            pass.textures.push_back({ rname, TextureHandle::Invalid });
            break;
        case SHADER_RESOURCE_TYPE_SAMPLER:
            if (rd.ArraySize > 1) return fail("sampler '" + rname + "' is an array; image passes bind single samplers");
            pass.samplers.push_back(rname);
            break;
        default:
            return fail("'" + rname + "' is a kind of resource image passes don't bind (buffers and UAVs aren't supported)");
        }
    }

    // The pipeline: the fullscreen triangle into one target of the pass's format, no depth, no blending. Every
    // resource is dynamic (textures re-pointed at each run); each sampler is immutable, from desc.samplers by name.
    std::vector<ImmutableSamplerDesc> immutables;
    immutables.reserve(pass.samplers.size());
    for (const std::string& s : pass.samplers) {
        const PassSampler* asked = nullptr;
        for (const PassSampler& p : desc.samplers)
            if (p.name == s) asked = &p;
        immutables.emplace_back(SHADER_TYPE_PIXEL, s.c_str(), ToSampler(asked));
    }
    GraphicsPipelineStateCreateInfo pci;
    pci.PSODesc.Name = name.c_str();
    GraphicsPipelineDesc& gp = pci.GraphicsPipeline;
    gp.NumRenderTargets             = 1;
    gp.RTVFormats[0]                = ToDiligent(desc.format);
    gp.DSVFormat                    = TEX_FORMAT_UNKNOWN;
    gp.PrimitiveTopology            = PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    gp.RasterizerDesc.CullMode      = CULL_MODE_NONE;
    gp.DepthStencilDesc.DepthEnable = False;
    pci.pVS = r.imagePasses->vs;
    pci.pPS = ps;
    pci.PSODesc.ResourceLayout.DefaultVariableType  = SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC;
    pci.PSODesc.ResourceLayout.ImmutableSamplers    = immutables.empty() ? nullptr : immutables.data();
    pci.PSODesc.ResourceLayout.NumImmutableSamplers = static_cast<Uint32>(immutables.size());
    {
        DiligentLogCapture capture;
        r.device->CreateGraphicsPipelineState(pci, &pass.pso);
        if (pass.pso) pass.pso->CreateShaderResourceBinding(&pass.srb, true);
        if (!pass.pso || !pass.srb) {
            std::string why = capture.FirstError().empty() ? "the pipeline could not be created" : "pipeline: " + capture.FirstError();
            capture.Discard();
            return fail(compiler.empty() ? why : compiler + "\n" + why);
        }
    }

    // The cbuffers: one GPU buffer each, bound once; their contents are uploaded at a run when they changed.
    for (ImagePass::CBuffer& cb : pass.cbuffers) {
        BufferDesc bd;
        bd.Name      = "render image pass CB";
        bd.Size      = cb.data.size();
        bd.Usage     = USAGE_DEFAULT;
        bd.BindFlags = BIND_UNIFORM_BUFFER;
        BufferData init;
        init.pData    = cb.data.data();
        init.DataSize = cb.data.size();
        r.device->CreateBuffer(bd, &init, &cb.buffer);
        if (!cb.buffer) return fail("the buffer for cbuffer '" + cb.name + "' could not be created");
        cb.dirty = false;
        if (IShaderResourceVariable* var = pass.srb->GetVariableByName(SHADER_TYPE_PIXEL, cb.name.c_str())) var->Set(cb.buffer);
        else return fail("cbuffer '" + cb.name + "' has no binding");
    }

    const std::uint32_t id = r.imagePasses->nextPass++;
    r.imagePasses->passes.emplace(id, std::move(pass));
    result.pass = static_cast<PassHandle>(id);
    return result;
}

void DiligentSceneRenderer::FreePass(PassHandle pass) {
    Impl& r = *m_impl;
    if (r.imagePasses) r.imagePasses->passes.erase(static_cast<std::uint32_t>(pass));
}

PassInfo DiligentSceneRenderer::Describe(PassHandle handle) const {
    PassInfo         info;
    const ImagePass* pass = FindPass(*m_impl, handle);
    if (!pass) return info;
    for (const ImagePass::Constant& c : pass->constants) info.constants.push_back(c.info);
    for (const ImagePass::Texture& t : pass->textures) info.textures.push_back(t.name);
    info.samplers = pass->samplers;
    return info;
}

namespace {

// Write `values` (4-byte numbers) into a pass's constant `name`, by HLSL's cbuffer packing: a non-array's components
// one after another from its offset; an array's elements each on its own 16-byte register.
bool WriteConstant(ImagePass* pass, std::string_view name, const void* values, std::size_t count, bool isInt, std::string* error) {
    if (!pass) return Fail(error, "set constant: no such pass");
    ImagePass::Constant* c = nullptr;
    for (ImagePass::Constant& k : pass->constants)
        if (k.info.name == name) c = &k;
    if (!c) return Fail(error, pass->Prefix() + "no constant '" + std::string(name) + "'");
    if (c->info.matrix) return Fail(error, pass->Prefix() + "constant '" + c->info.name + "' is a matrix; matrices can't be set yet");
    if (!c->supported) return Fail(error, pass->Prefix() + "constant '" + c->info.name + "' has a type image passes can't set");
    if (c->info.isInt != isInt)
        return Fail(error, pass->Prefix() + "constant '" + c->info.name + "' is " + (c->info.isInt ? "an int" : "a float") +
                               "; it was given " + (isInt ? "ints" : "floats"));
    const std::size_t elements = static_cast<std::size_t>(std::max(c->info.arraySize, 1));
    const std::size_t comps    = static_cast<std::size_t>(c->info.components);
    if (count != elements * comps)
        return Fail(error, pass->Prefix() + "constant '" + c->info.name + "' takes " + std::to_string(elements * comps) +
                               " values; it was given " + std::to_string(count));
    ImagePass::CBuffer& cb     = pass->cbuffers[c->cbuffer];
    const std::size_t   stride = c->info.arraySize > 0 ? 16 : comps * 4;
    const std::size_t   end    = c->offset + (elements - 1) * stride + comps * 4;
    if (end > cb.data.size())
        return Fail(error, pass->Prefix() + "constant '" + c->info.name + "' doesn't fit its cbuffer '" + cb.name + "'");
    const auto* src = static_cast<const std::uint8_t*>(values);
    for (std::size_t e = 0; e < elements; ++e)
        std::memcpy(cb.data.data() + c->offset + e * stride, src + e * comps * 4, comps * 4);
    cb.dirty = true;
    return true;
}

} // namespace

bool DiligentSceneRenderer::SetConstant(PassHandle pass, std::string_view name, std::span<const float> values, std::string* error) {
    return WriteConstant(FindPass(*m_impl, pass), name, values.data(), values.size(), false, error);
}

bool DiligentSceneRenderer::SetConstant(PassHandle pass, std::string_view name, std::span<const std::int32_t> values,
                                        std::string* error) {
    return WriteConstant(FindPass(*m_impl, pass), name, values.data(), values.size(), true, error);
}

bool DiligentSceneRenderer::SetTexture(PassHandle handle, std::string_view name, TextureHandle texture, std::string* error) {
    ImagePass* pass = FindPass(*m_impl, handle);
    if (!pass) return Fail(error, "set texture: no such pass");
    for (ImagePass::Texture& t : pass->textures) {
        if (t.name != name) continue;
        t.bound = texture;
        return true;
    }
    return Fail(error, pass->Prefix() + "no texture '" + std::string(name) + "'");
}

bool DiligentSceneRenderer::RunPass(PassHandle handle, TextureHandle target, int mip, std::string* error) {
    Impl&      r    = *m_impl;
    ImagePass* pass = FindPass(r, handle);
    if (!pass) return Fail(error, "run: no such pass");
    GpuTexture* out = r.FindTexture(target);
    if (!out || !out->target) return Fail(error, pass->Prefix() + "the output is not a target");
    if (out->targetFormat != pass->format)
        return Fail(error, pass->Prefix() + "renders " + FormatName(pass->format) + "; the target is " + FormatName(out->targetFormat));
    if (mip < 0 || mip >= Levels(*out))
        return Fail(error, pass->Prefix() + "the target has no level " + std::to_string(mip));

    // Point every texture at what is bound to it now (a freed texture, or the output itself, stops the run).
    for (ImagePass::Texture& t : pass->textures) {
        if (t.bound == TextureHandle::Invalid) return Fail(error, pass->Prefix() + "texture '" + t.name + "' is not bound");
        if (t.bound == target)
            return Fail(error, pass->Prefix() + "texture '" + t.name + "' is the output target (render through two targets)");
        GpuTexture* in = r.FindTexture(t.bound);
        if (!in) return Fail(error, pass->Prefix() + "texture '" + t.name + "' was freed");
        IShaderResourceVariable* var = pass->srb->GetVariableByName(SHADER_TYPE_PIXEL, t.name.c_str());
        if (!var) return Fail(error, pass->Prefix() + "texture '" + t.name + "' has no binding");
        var->Set(in->srv.RawPtr());
    }
    for (ImagePass::CBuffer& cb : pass->cbuffers) {
        if (!cb.dirty) continue;
        r.context->UpdateBuffer(cb.buffer, 0, cb.data.size(), cb.data.data(), RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
        cb.dirty = false;
    }

    ITextureView* rtv = MipRtv(*out, mip);
    if (!rtv) return Fail(error, pass->Prefix() + "no render-target view for level " + std::to_string(mip));
    r.context->SetRenderTargets(1, &rtv, nullptr, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    r.context->SetViewports(1, nullptr, 0, 0);   // the bound level's size
    r.context->SetPipelineState(pass->pso);
    r.context->CommitShaderResources(pass->srb, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    DrawAttribs draw{ 3, DRAW_FLAG_VERIFY_ALL };
    r.context->Draw(draw);
    Unbind(r);
    return true;
}

// ── Readback ──────────────────────────────────────────────────────────────────

std::vector<TargetImage> DiligentSceneRenderer::ReadTargets(std::span<const TargetRead> reads) {
    Impl&                    r = *m_impl;
    std::vector<TargetImage> images(reads.size());
    struct Pending {
        std::size_t             index = 0;
        RefCntAutoPtr<ITexture> staging;
        TargetFormat            format = TargetFormat::RGBA16F;
    };
    std::vector<Pending> pending;

    // Copy every level asked for into a CPU-readable texture of its own, then wait for the GPU once.
    for (std::size_t i = 0; i < reads.size(); ++i) {
        TargetImage& img = images[i];
        GpuTexture*  t   = r.FindTexture(reads[i].target);
        if (!t || !t->target) {
            img.error = "read: not a target";
            continue;
        }
        const int levels = Levels(*t);
        if (reads[i].mip < 0 || reads[i].mip >= levels) {
            img.error = "read: the target has no level " + std::to_string(reads[i].mip) + " (it has " + std::to_string(levels) + ")";
            continue;
        }
        img.width  = std::max(t->width >> reads[i].mip, 1);
        img.height = std::max(t->height >> reads[i].mip, 1);

        TextureDesc sd;
        sd.Name           = "render image readback";
        sd.Type           = RESOURCE_DIM_TEX_2D;
        sd.Width          = static_cast<Uint32>(img.width);
        sd.Height         = static_cast<Uint32>(img.height);
        sd.MipLevels      = 1;
        sd.Format         = ToDiligent(t->targetFormat);
        sd.Usage          = USAGE_STAGING;
        sd.BindFlags      = BIND_NONE;
        sd.CPUAccessFlags = CPU_ACCESS_READ;
        Pending p;
        p.index  = i;
        p.format = t->targetFormat;
        r.device->CreateTexture(sd, nullptr, &p.staging);
        if (!p.staging) {
            img.error = "read: no staging texture";
            continue;
        }
        CopyTextureAttribs copy{ t->texture, RESOURCE_STATE_TRANSITION_MODE_TRANSITION, p.staging,
                                 RESOURCE_STATE_TRANSITION_MODE_TRANSITION };
        copy.SrcMipLevel = static_cast<Uint32>(reads[i].mip);
        r.context->CopyTexture(copy);
        pending.push_back(std::move(p));
    }
    if (pending.empty()) return images;
    r.context->WaitForIdle();

    for (Pending& p : pending) {
        TargetImage&             img = images[p.index];
        MappedTextureSubresource mapped;
        r.context->MapTextureSubresource(p.staging, 0, 0, MAP_READ, MAP_FLAG_DO_NOT_WAIT, nullptr, mapped);
        if (!mapped.pData) {
            img.error = "read: the staging texture could not be mapped";
            continue;
        }
        const int channels = Channels(p.format);
        const int bytes    = ChannelBytes(p.format);
        img.rgba.assign(static_cast<std::size_t>(img.width) * img.height * 4, 0.f);
        for (int y = 0; y < img.height; ++y) {
            const auto* row = static_cast<const std::uint8_t*>(mapped.pData) + static_cast<std::size_t>(y) * mapped.Stride;
            for (int x = 0; x < img.width; ++x) {
                float* dst = img.rgba.data() + (static_cast<std::size_t>(y) * img.width + x) * 4;
                dst[3]     = 1.f;   // G, B stay 0 and A 1 for a format without them
                for (int c = 0; c < channels; ++c) {
                    const std::uint8_t* src = row + (static_cast<std::size_t>(x) * channels + c) * bytes;
                    if (bytes == 1) {
                        dst[c] = static_cast<float>(*src) / 255.f;
                    } else if (bytes == 2) {
                        std::uint16_t h;
                        std::memcpy(&h, src, 2);
                        dst[c] = HalfToFloat(h);
                    } else {
                        std::memcpy(&dst[c], src, 4);
                    }
                }
            }
        }
        r.context->UnmapTextureSubresource(p.staging, 0, 0);
        img.ok = true;
    }
    return images;
}

} // namespace render::diligent
