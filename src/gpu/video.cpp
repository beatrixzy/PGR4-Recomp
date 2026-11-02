// PGR4-Recomp native renderer: Plume D3D12 device, swapchain and frame ring.
//
// Frame flow follows re:Blue's renderer (gpu/present.cpp, gpu/frame_ring.cpp):
// kNumFrames command lists, each with its own fence; the swapchain image is
// acquired on an acquire semaphore, the list signals a per-image render
// semaphore that present() waits on, and the reused slot's fence is waited
// before its list, upload memory and parked objects are recycled.
#include "gpu/video.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>
#include <mutex>
#include <vector>

#include <plume_d3d12.h>
#include <plume_render_interface.h>
#include <plume_render_interface_builders.h>

#include <fmt/format.h>

#include <rex/cvar.h>
#include <rex/logging.h>

REXCVAR_DEFINE_STRING(gpu_adapter, "", "PGR4",
                      "D3D12 adapter to use, by name (empty: best GPU; \"Microsoft Basic Render Driver\" is WARP)");
REXCVAR_DEFINE_BOOL(gpu_crash_report, false, "PGR4",
                    "On a GPU crash, log the draw it happened in and, with DRED, the faulting resource (slows the GPU)");
REXCVAR_DECLARE(double, aspect_ratio);
REXCVAR_DEFINE_BOOL(gpu_sync, false, "PGR4",
                    "Debug: wait for the GPU to finish each frame after presenting it (no frame in flight)");

#include "present_ps.dxbc.h"
#include "present_vs.dxbc.h"

namespace plume {
extern std::unique_ptr<RenderInterface> CreateD3D12Interface();
} // namespace plume

namespace pgr4::gpu {
namespace {

// Two frames in flight: the CPU records N+1 while the GPU runs N.
constexpr uint32_t kNumFrames = 2;
// Transient upload memory per frame (vertex/index/constant/texel uploads).
constexpr uint64_t kUploadBytesPerFrame = 96ull << 20;
// Root CBVs have no bounds, and a shader's relative constant indexing can
// read up to c255 past a short upload: the buffer has 4 KB of slack after
// the last allocation so that read stays inside it.
constexpr uint64_t kUploadSlack = 4096;

struct FrameSlot {
  std::unique_ptr<plume::RenderCommandList> command_list;
  std::unique_ptr<plume::RenderCommandFence> fence;
  std::unique_ptr<plume::RenderCommandSemaphore> acquire_semaphore;
  bool submitted = false;

  std::unique_ptr<plume::RenderBuffer> upload_buffer;
  uint8_t *upload_memory = nullptr;
  uint64_t upload_offset = 0;
  bool upload_overflow_logged = false;

  // Released once this slot's fence has signalled.
  std::vector<std::shared_ptr<void>> parked;
  // Each draw recorded into the list, in order (crash reports).
  std::vector<Video::DrawNote> draws;
  // Crash report breadcrumbs: the GPU writes draw index + 1 to dword 0 as a
  // draw starts and to dword 1 once it is done (readback heap, always mapped).
  std::unique_ptr<plume::RenderBuffer> crumbs;
  const volatile uint32_t *crumbs_memory = nullptr;
  std::vector<uint32_t> freed_texture_slots;

  // Texture readbacks (PGR4_D3D12_CAPTURE, resolve dumps), written as BMPs
  // once the fence passes.
  struct PendingCapture {
    std::unique_ptr<plume::RenderBuffer> buffer;
    plume::RenderFormat format = plume::RenderFormat::UNKNOWN;
    uint32_t width = 0, height = 0, pitch = 0;
    std::string name;
  };
  std::vector<PendingCapture> captures;
};

struct VideoState {
  std::unique_ptr<plume::RenderInterface> render_iface;
  std::unique_ptr<plume::RenderDevice> device;
  std::unique_ptr<plume::RenderCommandQueue> queue;
  FrameSlot slots[kNumFrames];
  uint32_t frame = 0;
  bool list_open = false;
  uint64_t list_generation = 0; // command lists begun

  // One per swapchain image: a binary semaphore can't be re-signalled while
  // present still waits on it, and re-acquiring the image proves it is free.
  std::vector<std::unique_ptr<plume::RenderCommandSemaphore>> render_semaphores;
  std::unique_ptr<plume::RenderSwapChain> swap_chain;
  std::vector<std::unique_ptr<plume::RenderFramebuffer>> framebuffers;

  std::unique_ptr<plume::RenderDescriptorSet> texture_set;
  std::vector<bool> texture_slot_used;
  uint32_t texture_slot_hint = 0;
  std::unique_ptr<plume::RenderDescriptorSet> sampler_set;
  std::unique_ptr<plume::RenderSampler> linear_clamp;
  std::unique_ptr<plume::RenderSampler> point_clamp;

  Video::OverlayDrawHook overlay_hook;
  void (*before_submit)() = nullptr;

  // Front buffer blit: set 0 one texture (t0 space0), set 1 one sampler
  // (s0 space1), fxc shaders so every D3D12 GPU can run it.
  std::unique_ptr<plume::RenderPipelineLayout> present_layout;
  std::unique_ptr<plume::RenderShader> present_vs;
  std::unique_ptr<plume::RenderShader> present_ps;
  std::unique_ptr<plume::RenderPipeline> present_pipeline;
  std::unique_ptr<plume::RenderDescriptorSet> present_sampler_set;
  // Gamma ramp lookup (256x1 RGBA16 UNORM) and the texels last uploaded.
  std::unique_ptr<plume::RenderTexture> gamma_texture;
  std::unique_ptr<plume::RenderTextureView> gamma_view;
  std::array<uint16_t, 256 * 4> gamma_texels{};
  bool gamma_uploaded = false;

  // D3D12 debug layer messages (PGR4_D3D12_DEBUG=1), drained into the log.
  ID3D12InfoQueue *info_queue = nullptr;
  bool device_removed_logged = false;

  uint32_t capture_interval = 0; // frames, 0 = off

  std::atomic<bool> resize_requested{false};
  std::atomic<bool> shutting_down{false};
  std::atomic<uint64_t> frames_presented{0};
  // Timing since the last report (see Video::AddRendererTime).
  std::atomic<uint64_t> renderer_ns{0};
  std::atomic<uint64_t> renderer_draws{0};
  uint64_t gpu_wait_ns = 0;
  std::atomic<uint64_t> gpu_wait_total_ns{0}; // never reset (Video::GpuWaitTotal)
  uint64_t present_ns = 0;
  std::chrono::steady_clock::time_point report_start = std::chrono::steady_clock::now();
  std::mutex mutex;
  bool ready = false;
};

VideoState &state() {
  static VideoState s;
  return s;
}

bool BuildSwapChainTargets(VideoState &s) {
  s.framebuffers.clear();
  s.render_semaphores.clear();
  const uint32_t count = s.swap_chain->getTextureCount();
  for (uint32_t i = 0; i < count; ++i) {
    const plume::RenderTexture *color[1] = {s.swap_chain->getTexture(i)};
    auto fb = s.device->createFramebuffer(plume::RenderFramebufferDesc(color, 1));
    auto sem = s.device->createCommandSemaphore();
    if (!fb || !sem) {
      REXGPU_ERROR("PGR4 D3D12: swapchain target {} creation failed", i);
      s.framebuffers.clear();
      s.render_semaphores.clear();
      return false;
    }
    s.framebuffers.push_back(std::move(fb));
    s.render_semaphores.push_back(std::move(sem));
  }
  return true;
}

bool BuildBindlessSets(VideoState &s) {
  plume::RenderDescriptorSetBuilder tex_builder;
  tex_builder.begin();
  tex_builder.addTexture(0, kBindlessTextureCount);
  tex_builder.end(true, kBindlessTextureCount);
  s.texture_set = tex_builder.create(s.device.get());

  plume::RenderDescriptorSetBuilder sampler_builder;
  sampler_builder.begin();
  sampler_builder.addSampler(0, kBindlessSamplerCount);
  sampler_builder.end(true, kBindlessSamplerCount);
  s.sampler_set = sampler_builder.create(s.device.get());
  if (!s.texture_set || !s.sampler_set) {
    REXGPU_ERROR("PGR4 D3D12: bindless descriptor set creation failed");
    return false;
  }
  s.texture_slot_used.assign(kBindlessTextureCount, false);

  plume::RenderSamplerDesc desc;
  desc.minFilter = plume::RenderFilter::LINEAR;
  desc.magFilter = plume::RenderFilter::LINEAR;
  desc.mipmapMode = plume::RenderMipmapMode::LINEAR;
  desc.addressU = desc.addressV = desc.addressW = plume::RenderTextureAddressMode::CLAMP;
  s.linear_clamp = s.device->createSampler(desc);
  desc.minFilter = desc.magFilter = plume::RenderFilter::NEAREST;
  desc.mipmapMode = plume::RenderMipmapMode::NEAREST;
  s.point_clamp = s.device->createSampler(desc);
  if (!s.linear_clamp || !s.point_clamp) {
    REXGPU_ERROR("PGR4 D3D12: sampler creation failed");
    return false;
  }
  s.sampler_set->setSampler(kLinearClampSamplerSlot, s.linear_clamp.get());
  s.sampler_set->setSampler(kPointClampSamplerSlot, s.point_clamp.get());
  return true;
}

float HalfToFloat(uint16_t h) {
  const uint32_t sign = uint32_t(h >> 15) << 31, exp = (h >> 10) & 0x1F, mant = h & 0x3FF;
  uint32_t bits;
  if (exp == 0) {
    float f = std::ldexp(float(mant), -24);
    return sign ? -f : f;
  } else if (exp == 31) {
    bits = sign | 0x7F800000 | (mant << 13);
  } else {
    bits = sign | ((exp + 112) << 23) | (mant << 13);
  }
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}

// Writes the slot's pending readbacks as 24-bit BMPs next to the exe.
void WriteCaptures(FrameSlot &slot) {
  for (auto &c : slot.captures) {
    const uint32_t w = c.width, h = c.height;
    const auto *src = static_cast<plume::D3D12Buffer *>(c.buffer.get())->d3d
                          ? static_cast<const uint8_t *>(c.buffer->map())
                          : nullptr;
    if (!src) {
      REXGPU_WARN("PGR4 D3D12: capture {} could not be read back", c.name);
      continue;
    }
    const uint32_t row = (w * 3 + 3) & ~3u;
    std::vector<uint8_t> bmp(54 + size_t(row) * h);
    auto put32 = [&](size_t o, uint32_t v) { std::memcpy(&bmp[o], &v, 4); };
    bmp[0] = 'B';
    bmp[1] = 'M';
    put32(2, uint32_t(bmp.size()));
    put32(10, 54);
    put32(14, 40);
    put32(18, w);
    put32(22, uint32_t(-int32_t(h)));
    bmp[26] = 1;
    bmp[28] = 24;
    const bool half = c.format == plume::RenderFormat::R16G16B16A16_FLOAT;
    const bool r32 = c.format == plume::RenderFormat::R32_FLOAT;
    // PGR4_DUMP_SCALE, or a dump.scale file next to the exe (read per
    // capture): multiplier for float captures (HDR scenes clip at 1).
    float hdr_scale = 1.0f;
    if (const char *v = std::getenv("PGR4_DUMP_SCALE")) {
      hdr_scale = float(std::atof(v));
    }
    if (std::FILE *sf = std::fopen("dump.scale", "rb")) {
      std::fscanf(sf, "%f", &hdr_scale);
      std::fclose(sf);
    }
    for (uint32_t y = 0; y < h; ++y) {
      const uint8_t *in = src + size_t(y) * c.pitch;
      uint8_t *out = &bmp[54 + size_t(y) * row];
      for (uint32_t x = 0; x < w; ++x) {
        float rgb[3];
        if (r32) {
          float v;
          std::memcpy(&v, in + x * 4, 4);
          rgb[0] = rgb[1] = rgb[2] = v * hdr_scale;
        } else if (half) {
          uint16_t v[3];
          std::memcpy(v, in + x * 8, 6);
          for (int i = 0; i < 3; ++i) {
            rgb[i] = HalfToFloat(v[i]) * hdr_scale;
          }
        } else {
          for (int i = 0; i < 3; ++i) {
            rgb[i] = in[x * 4 + i] / 255.0f;
          }
        }
        for (int i = 0; i < 3; ++i) {
          out[x * 3 + 2 - i] = uint8_t(std::clamp(rgb[i], 0.0f, 1.0f) * 255.0f + 0.5f);
        }
      }
    }
    // Tiny targets (luminance averages, exposure) clip to nothing useful in a
    // BMP; log their raw values, NaN and Inf included.
    if (w * h <= 16) {
      std::string values;
      for (uint32_t y = 0; y < h; ++y) {
        for (uint32_t x = 0; x < w; ++x) {
          const uint8_t *in = src + size_t(y) * c.pitch;
          float v[4] = {};
          if (r32) {
            std::memcpy(&v[0], in + x * 4, 4);
          } else if (half) {
            uint16_t hv[4];
            std::memcpy(hv, in + x * 8, 8);
            for (int i = 0; i < 4; ++i) v[i] = HalfToFloat(hv[i]);
          } else {
            for (int i = 0; i < 4; ++i) v[i] = in[x * 4 + i] / 255.0f;
          }
          values += fmt::format(" ({:g} {:g} {:g} {:g})", v[0], v[1], v[2], v[3]);
        }
      }
      REXGPU_INFO("PGR4 D3D12 dump: {} {}x{} values{}", c.name, w, h, values);
    } else if ((r32 || half) && w * h <= 4096) {
      // Downsample chain steps: per-channel range and mean, unclipped.
      float lo[4] = {INFINITY, INFINITY, INFINITY, INFINITY}, hi[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
      double sum[4] = {};
      for (uint32_t y = 0; y < h; ++y) {
        const uint8_t *in = src + size_t(y) * c.pitch;
        for (uint32_t x = 0; x < w; ++x) {
          float v[4] = {};
          if (r32) {
            std::memcpy(&v[0], in + x * 4, 4);
          } else {
            uint16_t hv[4];
            std::memcpy(hv, in + x * 8, 8);
            for (int i = 0; i < 4; ++i) v[i] = HalfToFloat(hv[i]);
          }
          for (int i = 0; i < 4; ++i) {
            lo[i] = std::min(lo[i], v[i]);
            hi[i] = std::max(hi[i], v[i]);
            sum[i] += v[i];
          }
        }
      }
      std::string stats;
      for (int i = 0; i < (r32 ? 1 : 4); ++i) {
        stats += fmt::format(" ch{} {:g}..{:g} mean {:g}", i, lo[i], hi[i], sum[i] / (w * h));
      }
      REXGPU_INFO("PGR4 D3D12 dump: {} {}x{} range{}", c.name, w, h, stats);
    }
    c.buffer->unmap();
    if (FILE *f = std::fopen((c.name + ".bmp").c_str(), "wb")) {
      std::fwrite(bmp.data(), 1, bmp.size(), f);
      std::fclose(f);
    } else {
      REXGPU_WARN("PGR4 D3D12: could not write {}.bmp", c.name);
    }
  }
  slot.captures.clear();
}

// Records a readback of a SHADER_READ texture into the open list.
void QueueCapture(VideoState &s, plume::RenderCommandList *list, const plume::RenderTexture *source,
                  uint32_t width, uint32_t height, std::string name,
                  plume::RenderTextureLayout after = plume::RenderTextureLayout::SHADER_READ) {
  auto *texture = const_cast<plume::RenderTexture *>(source);
  const plume::RenderFormat format = static_cast<plume::D3D12Texture *>(texture)->desc.format;
  if (format != plume::RenderFormat::R8G8B8A8_UNORM && format != plume::RenderFormat::R16G16B16A16_FLOAT &&
      format != plume::RenderFormat::R32_FLOAT) {
    return;
  }
  const uint32_t bpp = plume::RenderFormatSize(format);
  FrameSlot::PendingCapture c;
  c.pitch = (width * bpp + 255) & ~255u;
  c.buffer = s.device->createBuffer(plume::RenderBufferDesc::ReadbackBuffer(uint64_t(c.pitch) * height));
  if (!c.buffer || !static_cast<plume::D3D12Buffer *>(c.buffer.get())->d3d) {
    REXGPU_WARN("PGR4 D3D12: no readback buffer for capture {}", name);
    return;
  }
  c.format = format;
  c.width = width;
  c.height = height;
  c.name = std::move(name);
  list->barriers(plume::RenderBarrierStage::COPY,
                 plume::RenderTextureBarrier(texture, plume::RenderTextureLayout::COPY_SOURCE));
  // Plume's copyTextureRegion only copies into textures, so the
  // texture-to-buffer readback goes through the native list.
  D3D12_TEXTURE_COPY_LOCATION dst = {};
  dst.pResource = static_cast<plume::D3D12Buffer *>(c.buffer.get())->d3d;
  dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
  dst.PlacedFootprint.Footprint.Format =
      format == plume::RenderFormat::R16G16B16A16_FLOAT ? DXGI_FORMAT_R16G16B16A16_FLOAT
      : format == plume::RenderFormat::R32_FLOAT      ? DXGI_FORMAT_R32_FLOAT
                                                      : DXGI_FORMAT_R8G8B8A8_UNORM;
  dst.PlacedFootprint.Footprint.Width = width;
  dst.PlacedFootprint.Footprint.Height = height;
  dst.PlacedFootprint.Footprint.Depth = 1;
  dst.PlacedFootprint.Footprint.RowPitch = c.pitch;
  D3D12_TEXTURE_COPY_LOCATION src = {};
  src.pResource = static_cast<plume::D3D12Texture *>(texture)->d3d;
  src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
  static_cast<plume::D3D12CommandList *>(list)->d3d->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
  list->barriers(plume::RenderBarrierStage::GRAPHICS,
                 plume::RenderTextureBarrier(texture, after));
  s.slots[s.frame].captures.push_back(std::move(c));
}

// Wait for the slot's previous use, then recycle what it held.
void ReclaimSlot(VideoState &s, FrameSlot &slot) {
  if (slot.submitted) {
    const auto t0 = std::chrono::steady_clock::now();
    s.queue->waitForCommandFence(slot.fence.get());
    const uint64_t waited = uint64_t(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
    s.gpu_wait_ns += waited;
    s.gpu_wait_total_ns.fetch_add(waited, std::memory_order_relaxed);
    slot.submitted = false;
  }
  WriteCaptures(slot);
  slot.upload_offset = 0;
  slot.upload_overflow_logged = false;
  slot.parked.clear();
  for (uint32_t index : slot.freed_texture_slots) {
    s.texture_slot_used[index] = false;
  }
  slot.freed_texture_slots.clear();
}

void WaitAllFrames(VideoState &s) {
  for (auto &slot : s.slots) {
    if (slot.submitted) {
      s.queue->waitForCommandFence(slot.fence.get());
      slot.submitted = false;
    }
  }
}

std::string Narrow(const wchar_t *w) {
  std::string out;
  for (; w && *w; ++w) {
    out += *w < 128 ? char(*w) : '?';
  }
  return out;
}

void LogDraw(const char *source, const Video::DrawNote &d, uint32_t index, bool marked) {
  REXGPU_ERROR("PGR4 D3D12: {} {} draw {}: vs {:016X} ps {:016X} count {} first {} base {}{}", source,
               marked ? ">>" : "  ", index, d.vs, d.ps, d.count, d.first, d.base_vertex,
               d.indexed ? " indexed" : "");
}

// Breadcrumbs (gpu_crash_report): for each list the GPU had, the draws that
// started but never finished. The markers number draws from 1.
void LogBreadcrumbs(VideoState &s) {
  for (const auto &slot : s.slots) {
    if (!slot.crumbs_memory) {
      continue;
    }
    const uint32_t started = slot.crumbs_memory[0], finished = slot.crumbs_memory[1];
    REXGPU_ERROR("PGR4 D3D12: breadcrumbs slot {}: {} draws recorded, {} started, {} finished",
                 &slot - s.slots, slot.draws.size(), started, finished);
    if (started == finished) {
      continue;
    }
    for (uint32_t i = finished > 2 ? finished - 2 : 0; i < slot.draws.size() && i <= started; ++i) {
      LogDraw("breadcrumb", slot.draws[i], i, i >= finished && i < started);
    }
  }
}

// DRED (enabled in CreateHostDevice): the GPU page fault address with the
// allocations around it, live or recently freed, and how far each command
// list got, with the shaders of the draw it stopped in.
void LogDeviceRemovedData(VideoState &s, ID3D12Device *d3d) {
  ID3D12DeviceRemovedExtendedData1 *dred = nullptr;
  if (FAILED(d3d->QueryInterface(IID_PPV_ARGS(&dred)))) {
    REXGPU_ERROR("PGR4 D3D12: no DRED data on this driver");
    return;
  }
  D3D12_DRED_PAGE_FAULT_OUTPUT1 fault = {};
  if (SUCCEEDED(dred->GetPageFaultAllocationOutput1(&fault))) {
    REXGPU_ERROR("PGR4 D3D12: DRED page fault at VA {:016X}", uint64_t(fault.PageFaultVA));
    for (auto *n = fault.pHeadExistingAllocationNode; n; n = n->pNext) {
      REXGPU_ERROR("PGR4 D3D12: DRED   live allocation '{}' type {}", Narrow(n->ObjectNameW), int(n->AllocationType));
    }
    for (auto *n = fault.pHeadRecentFreedAllocationNode; n; n = n->pNext) {
      REXGPU_ERROR("PGR4 D3D12: DRED   freed allocation '{}' type {}", Narrow(n->ObjectNameW), int(n->AllocationType));
    }
  }
  D3D12_DRED_AUTO_BREADCRUMBS_OUTPUT1 crumbs = {};
  if (SUCCEEDED(dred->GetAutoBreadcrumbsOutput1(&crumbs))) {
    for (auto *n = crumbs.pHeadAutoBreadcrumbNode; n; n = n->pNext) {
      const uint32_t done = n->pLastBreadcrumbValue ? *n->pLastBreadcrumbValue : 0;
      if (!n->BreadcrumbCount || done >= n->BreadcrumbCount) {
        continue; // finished
      }
      // Draws before the first unfinished op, to find it in our draw record.
      uint32_t draw_index = 0;
      for (uint32_t i = 0; i < done; ++i) {
        const auto op = n->pCommandHistory[i];
        draw_index += op == D3D12_AUTO_BREADCRUMB_OP_DRAWINSTANCED || op == D3D12_AUTO_BREADCRUMB_OP_DRAWINDEXEDINSTANCED;
      }
      REXGPU_ERROR("PGR4 D3D12: DRED list {} stopped at op {} of {} (op type {}), draw {}",
                   fmt::ptr(n->pCommandList), done, n->BreadcrumbCount, int(n->pCommandHistory[done]), draw_index);
      for (const auto &slot : s.slots) {
        auto *list = static_cast<plume::D3D12CommandList *>(slot.command_list.get());
        if (list && list->d3d == n->pCommandList) {
          for (uint32_t i = draw_index > 2 ? draw_index - 2 : 0; i < slot.draws.size() && i <= draw_index + 1; ++i) {
            LogDraw("DRED", slot.draws[i], i, i == draw_index);
          }
        }
      }
    }
  }
  dred->Release();
}

void DrainDebugMessages(VideoState &s) {
  if (s.info_queue) {
    const UINT64 count = s.info_queue->GetNumStoredMessages();
    for (UINT64 i = 0; i < count && i < 256; ++i) {
      SIZE_T size = 0;
      s.info_queue->GetMessage(i, nullptr, &size);
      std::vector<uint8_t> buffer(size);
      auto *message = reinterpret_cast<D3D12_MESSAGE *>(buffer.data());
      if (SUCCEEDED(s.info_queue->GetMessage(i, message, &size))) {
        REXGPU_WARN("PGR4 D3D12 debug: {}", std::string_view(message->pDescription,
                                                             message->DescriptionByteLength ? message->DescriptionByteLength - 1 : 0));
      }
    }
    s.info_queue->ClearStoredMessages();
  }
  if (!s.device_removed_logged && s.device) {
    auto *d3d = static_cast<plume::D3D12Device *>(s.device.get())->d3d;
    const HRESULT reason = d3d->GetDeviceRemovedReason();
    if (FAILED(reason)) {
      s.device_removed_logged = true;
      REXGPU_ERROR("PGR4 D3D12: device removed, reason {:08X}", uint32_t(reason));
      LogBreadcrumbs(s);
      LogDeviceRemovedData(s, d3d);
    }
  }
}

void BuildPresentSetLayouts(plume::RenderDescriptorSetBuilder &texture,
                            plume::RenderDescriptorSetBuilder &sampler) {
  texture.begin();
  texture.addTexture(0);
  texture.addTexture(1);
  texture.end();
  sampler.begin();
  sampler.addSampler(0);
  sampler.end();
}

bool BuildPresentPipeline(VideoState &s) {
  plume::RenderDescriptorSetBuilder tex_b, samp_b;
  BuildPresentSetLayouts(tex_b, samp_b);
  plume::RenderPipelineLayoutBuilder lb;
  lb.begin(false, false);
  lb.addDescriptorSet(tex_b);
  lb.addDescriptorSet(samp_b);
  lb.end();
  s.present_layout = lb.create(s.device.get());
  s.present_vs = s.device->createShader(g_present_vs_dxbc, sizeof(g_present_vs_dxbc), "main",
                                        plume::RenderShaderFormat::DXBC);
  s.present_ps = s.device->createShader(g_present_ps_dxbc, sizeof(g_present_ps_dxbc), "main",
                                        plume::RenderShaderFormat::DXBC);
  if (!s.present_layout || !s.present_vs || !s.present_ps) {
    return false;
  }
  plume::RenderGraphicsPipelineDesc pd;
  pd.pipelineLayout = s.present_layout.get();
  pd.vertexShader = s.present_vs.get();
  pd.pixelShader = s.present_ps.get();
  pd.primitiveTopology = plume::RenderPrimitiveTopology::TRIANGLE_LIST;
  pd.cullMode = plume::RenderCullMode::NONE;
  pd.renderTargetCount = 1;
  pd.renderTargetFormat[0] = plume::RenderFormat::B8G8R8A8_UNORM;
  pd.renderTargetBlend[0] = plume::RenderBlendDesc::Copy();
  s.present_pipeline = s.device->createGraphicsPipeline(pd);
  if (!s.present_pipeline ||
      !static_cast<plume::D3D12GraphicsPipeline *>(s.present_pipeline.get())->d3d) {
    s.present_pipeline.reset();
    return false;
  }
  s.present_sampler_set = samp_b.create(s.device.get());
  if (!s.present_sampler_set) {
    return false;
  }
  s.present_sampler_set->setSampler(0, s.linear_clamp.get());
  return true;
}

void WriteCrumb(FrameSlot &slot, uint32_t index, uint32_t value) {
  auto *list = static_cast<plume::D3D12CommandList *>(slot.command_list.get());
  ID3D12GraphicsCommandList2 *list2 = nullptr;
  if (FAILED(list->d3d->QueryInterface(IID_PPV_ARGS(&list2)))) {
    return;
  }
  D3D12_WRITEBUFFERIMMEDIATE_PARAMETER param = {
      static_cast<plume::D3D12Buffer *>(slot.crumbs.get())->d3d->GetGPUVirtualAddress() + index * 4, value};
  const D3D12_WRITEBUFFERIMMEDIATE_MODE mode =
      index == 0 ? D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_IN : D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_OUT;
  list2->WriteBufferImmediate(1, &param, &mode);
  list2->Release();
}

plume::RenderCommandList *OpenListLocked(VideoState &s) {
  if (!s.ready) {
    return nullptr;
  }
  auto &slot = s.slots[s.frame];
  if (!s.list_open) {
    if (slot.crumbs) {
      // A lost device signals every fence: report before the draw record goes.
      DrainDebugMessages(s);
    }
    ReclaimSlot(s, slot);
    slot.draws.clear();
    slot.command_list->begin();
    s.list_open = true;
    ++s.list_generation;
    if (slot.crumbs) {
      WriteCrumb(slot, 0, 0);
      WriteCrumb(slot, 1, 0);
    }
  }
  return slot.command_list.get();
}

} // namespace

bool Video::CreateHostDevice(void *hwnd) {
  auto &s = state();
  std::lock_guard lock(s.mutex);
  if (s.ready) {
    return true;
  }
  if (!hwnd) {
    REXGPU_ERROR("PGR4 D3D12: no window handle");
    return false;
  }

  if (const char *debug = std::getenv("PGR4_D3D12_DEBUG"); debug && *debug == '1') {
    ID3D12Debug *debug_controller = nullptr;
    if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug_controller)))) {
      debug_controller->EnableDebugLayer();
      // PGR4_D3D12_GBV=1 adds GPU-based validation, which checks resource
      // states per descriptor access (missing barriers) at a large speed cost.
      if (const char *gbv = std::getenv("PGR4_D3D12_GBV"); gbv && *gbv == '1') {
        ID3D12Debug1 *debug1 = nullptr;
        if (SUCCEEDED(debug_controller->QueryInterface(IID_PPV_ARGS(&debug1)))) {
          debug1->SetEnableGPUBasedValidation(TRUE);
          debug1->Release();
          REXGPU_INFO("PGR4 D3D12: GPU-based validation enabled");
        }
      }
      debug_controller->Release();
      REXGPU_INFO("PGR4 D3D12: debug layer enabled");
    }
  }
  if (const char *capture = std::getenv("PGR4_D3D12_CAPTURE")) {
    s.capture_interval = uint32_t(std::strtoul(capture, nullptr, 10));
  }
  if (REXCVAR_GET(gpu_crash_report)) {
    ID3D12DeviceRemovedExtendedDataSettings *dred = nullptr;
    if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&dred)))) {
      dred->SetAutoBreadcrumbsEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
      dred->SetPageFaultEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
      dred->Release();
    }
  }
  s.render_iface = plume::CreateD3D12Interface();
  if (!s.render_iface) {
    REXGPU_ERROR("PGR4 D3D12: CreateD3D12Interface failed");
    return false;
  }
  s.device = s.render_iface->createDevice(REXCVAR_GET(gpu_adapter));
  if (!s.device) {
    REXGPU_ERROR("PGR4 D3D12: createDevice failed");
    return false;
  }
  s.queue = s.device->createCommandQueue(plume::RenderCommandListType::DIRECT);
  for (auto &slot : s.slots) {
    slot.command_list = s.queue->createCommandList();
    slot.fence = s.device->createCommandFence();
    slot.acquire_semaphore = s.device->createCommandSemaphore();
    slot.upload_buffer = s.device->createBuffer(plume::RenderBufferDesc::UploadBuffer(kUploadBytesPerFrame + kUploadSlack));
    slot.upload_memory = slot.upload_buffer ? static_cast<uint8_t *>(slot.upload_buffer->map()) : nullptr;
    if (slot.upload_buffer) {
      slot.upload_buffer->setName(fmt::format("upload slot {}", &slot - s.slots));
    }
    if (!slot.command_list || !slot.fence || !slot.acquire_semaphore || !slot.upload_memory) {
      REXGPU_ERROR("PGR4 D3D12: frame slot creation failed");
      return false;
    }
    if (REXCVAR_GET(gpu_crash_report)) {
      slot.crumbs = s.device->createBuffer(plume::RenderBufferDesc::ReadbackBuffer(256));
      if (slot.crumbs && static_cast<plume::D3D12Buffer *>(slot.crumbs.get())->d3d) {
        slot.crumbs->setName("breadcrumbs");
        slot.crumbs_memory = static_cast<const uint32_t *>(slot.crumbs->map());
      } else {
        slot.crumbs.reset();
      }
    }
  }

  // kNumFrames + 1 images so acquiring the next never waits on scanout.
  plume::RenderSwapChainDesc desc(static_cast<plume::RenderWindow>(hwnd),
                                  plume::RenderFormat::B8G8R8A8_UNORM,
                                  kNumFrames + 1, false, kNumFrames);
  s.swap_chain = s.queue->createSwapChain(desc);
  if (!s.swap_chain || s.swap_chain->isEmpty()) {
    // Rendering (and frame dumps) still run; frames are submitted unpresented.
    REXGPU_ERROR("PGR4 D3D12: createSwapChain failed; rendering without presenting");
    // Left alive: Plume's destructor faults on a swap chain that failed to
    // initialise.
    (void)s.swap_chain.release();
  }
  if ((s.swap_chain && !BuildSwapChainTargets(s)) || !BuildBindlessSets(s)) {
    return false;
  }
  if (!BuildPresentPipeline(s)) {
    REXGPU_ERROR("PGR4 D3D12: present pipeline creation failed");
    return false;
  }

  {
    auto *d3d = static_cast<plume::D3D12Device *>(s.device.get())->d3d;
    if (FAILED(d3d->QueryInterface(IID_PPV_ARGS(&s.info_queue)))) {
      s.info_queue = nullptr;
    } else {
      // Guest clears use whatever value the game asks for.
      D3D12_MESSAGE_ID deny[] = {D3D12_MESSAGE_ID_CREATEINPUTLAYOUT_TYPE_MISMATCH,
                                 D3D12_MESSAGE_ID_CLEARRENDERTARGETVIEW_MISMATCHINGCLEARVALUE,
                                 D3D12_MESSAGE_ID_CLEARDEPTHSTENCILVIEW_MISMATCHINGCLEARVALUE};
      D3D12_MESSAGE_SEVERITY info[] = {D3D12_MESSAGE_SEVERITY_INFO, D3D12_MESSAGE_SEVERITY_MESSAGE};
      D3D12_INFO_QUEUE_FILTER filter = {};
      filter.DenyList.NumIDs = _countof(deny);
      filter.DenyList.pIDList = deny;
      filter.DenyList.NumSeverities = _countof(info);
      filter.DenyList.pSeverityList = info;
      s.info_queue->PushStorageFilter(&filter);
    }
  }
  {
    // Shader model and resource binding tier decide which shader cache and
    // binding model the game's draws can use (SM 5.1-only drivers, e.g. a
    // GT 710 on 391.35, can't run DXIL or large bindless tables).
    auto *d3d_device = static_cast<plume::D3D12Device *>(s.device.get());
    D3D12_FEATURE_DATA_D3D12_OPTIONS options = {};
    d3d_device->d3d->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &options, sizeof(options));
    REXGPU_INFO("PGR4 D3D12: shader model {}.{}, resource binding tier {}",
                int(d3d_device->shaderModel) >> 4, int(d3d_device->shaderModel) & 0xF,
                int(options.ResourceBindingTier));
  }
  REXGPU_INFO("PGR4 D3D12: device '{}', swapchain {}x{} ({} images)",
              s.device->getDescription().name, s.swap_chain ? s.swap_chain->getWidth() : 0,
              s.swap_chain ? s.swap_chain->getHeight() : 0, s.swap_chain ? s.swap_chain->getTextureCount() : 0);
  s.ready = true;
  return true;
}

void Video::Shutdown() {
  // Atomic only, like re:Blue's BeginShutdown: the guest thread may be parked
  // in Present waiting on the UI thread (overlay marshal) while holding the
  // mutex, so locking here could deadlock. The device is left to process
  // teardown.
  state().shutting_down.store(true, std::memory_order_release);
}

void Video::RequestResize() {
  state().resize_requested.store(true, std::memory_order_release);
}

void Video::AddRendererTime(uint64_t ns, uint32_t draws) {
  auto &s = state();
  s.renderer_ns.fetch_add(ns, std::memory_order_relaxed);
  s.renderer_draws.fetch_add(draws, std::memory_order_relaxed);
}

uint64_t Video::GpuWaitTotal() { return state().gpu_wait_total_ns.load(std::memory_order_relaxed); }

uint64_t Video::FrameCount() {
  return state().frames_presented.load(std::memory_order_relaxed);
}

plume::RenderDevice *Video::HostDevice() { return state().device.get(); }

plume::RenderCommandList *Video::CommandList() {
  auto &s = state();
  return s.list_open ? s.slots[s.frame].command_list.get() : nullptr;
}

plume::RenderCommandList *Video::OpenCommandList() {
  auto &s = state();
  if (s.shutting_down.load(std::memory_order_acquire)) {
    return nullptr;
  }
  return OpenListLocked(s);
}

uint64_t Video::ListGeneration() { return state().list_generation; }

void Video::NoteDraw(const DrawNote &note) {
  auto &s = state();
  auto &slot = s.slots[s.frame];
  slot.draws.push_back(note);
  if (slot.crumbs) {
    WriteCrumb(slot, 0, uint32_t(slot.draws.size()));
  }
}

void Video::NoteDrawDone() {
  auto &s = state();
  auto &slot = s.slots[s.frame];
  if (slot.crumbs) {
    WriteCrumb(slot, 1, uint32_t(slot.draws.size()));
  }
}

plume::RenderDescriptorSet *Video::TextureDescriptorSet() { return state().texture_set.get(); }

plume::RenderDescriptorSet *Video::SamplerDescriptorSet() { return state().sampler_set.get(); }

uint32_t Video::AllocateTextureSlot() {
  auto &s = state();
  const uint32_t n = static_cast<uint32_t>(s.texture_slot_used.size());
  for (uint32_t i = 0; i < n; ++i) {
    const uint32_t index = (s.texture_slot_hint + i) % n;
    if (!s.texture_slot_used[index]) {
      s.texture_slot_used[index] = true;
      s.texture_slot_hint = index + 1;
      return index;
    }
  }
  return UINT32_MAX;
}

void Video::FreeTextureSlot(uint32_t slot) {
  auto &s = state();
  if (slot < s.texture_slot_used.size()) {
    s.slots[s.frame].freed_texture_slots.push_back(slot);
  }
}

double Video::DisplayAspect() {
  static const double aspect = [] {
    if (const double set = REXCVAR_GET(aspect_ratio); set != 0.0) {
      return set;
    }
    auto &s = state();
    if (s.swap_chain && s.swap_chain->getHeight()) {
      const double a = double(s.swap_chain->getWidth()) / double(s.swap_chain->getHeight());
      if (a >= 1.0 && a < 16.0 / 9.0 - 0.01) {
        return a;
      }
    }
    return 0.0;
  }();
  return aspect;
}

UploadAllocation Video::Upload(const void *data, uint32_t size, uint32_t alignment) {
  auto &s = state();
  auto &slot = s.slots[s.frame];
  if (!slot.upload_memory || !size) {
    return {};
  }
  const uint64_t align = alignment ? alignment : 1;
  const uint64_t offset = (slot.upload_offset + align - 1) / align * align;
  if (offset + size > kUploadBytesPerFrame) {
    if (!slot.upload_overflow_logged) {
      REXGPU_WARN("PGR4 D3D12: per-frame upload memory exhausted ({} bytes)", kUploadBytesPerFrame);
      slot.upload_overflow_logged = true;
    }
    return {};
  }
  slot.upload_offset = offset + size;
  UploadAllocation alloc;
  alloc.memory = slot.upload_memory + offset;
  alloc.buffer = slot.upload_buffer.get();
  alloc.offset = offset;
  alloc.size = size;
  if (data) {
    std::memcpy(alloc.memory, data, size);
  }
  return alloc;
}

void Video::CaptureTexture(const plume::RenderTexture *texture, uint32_t width, uint32_t height,
                           const std::string &name, bool render_target) {
  auto &s = state();
  if (plume::RenderCommandList *list = OpenListLocked(s)) {
    QueueCapture(s, list, texture, width, height, name,
                 render_target ? plume::RenderTextureLayout::COLOR_WRITE
                               : plume::RenderTextureLayout::SHADER_READ);
  }
}

void Video::ParkErased(std::shared_ptr<void> object) {
  auto &s = state();
  s.slots[s.frame].parked.push_back(std::move(object));
}

void Video::SetBeforeSubmit(void (*fn)()) { state().before_submit = fn; }

void Video::SetOverlayDrawHook(OverlayDrawHook hook) {
  auto &s = state();
  std::lock_guard lock(s.mutex);
  s.overlay_hook = std::move(hook);
}

namespace {

// Ends and submits the open list without presenting (no swapchain image this
// frame), so recorded guest work and parked objects still retire.
void SubmitWithoutPresent(VideoState &s) {
  if (!s.list_open) {
    return;
  }
  auto &slot = s.slots[s.frame];
  slot.command_list->end();
  s.list_open = false;
  const plume::RenderCommandList *lists[] = {slot.command_list.get()};
  if (s.before_submit) {
    s.before_submit();
  }
  s.queue->executeCommandLists(lists, 1, nullptr, 0, nullptr, 0, slot.fence.get());
  slot.submitted = true;
  s.frame = (s.frame + 1) % kNumFrames;
  DrainDebugMessages(s);
}

// Uploads the guest gamma ramp (or identity) when it changed.
bool UpdateGammaRamp(VideoState &s, plume::RenderCommandList *list, const uint16_t *ramp) {
  if (!s.gamma_texture) {
    plume::RenderTextureDesc desc;
    desc.dimension = plume::RenderTextureDimension::TEXTURE_2D;
    desc.width = 256;
    desc.height = 1;
    desc.depth = 1;
    desc.mipLevels = 1;
    desc.arraySize = 1;
    desc.format = plume::RenderFormat::R16G16B16A16_UNORM;
    desc.flags = plume::RenderTextureFlag::NONE;
    desc.multisampling.sampleCount = plume::RenderSampleCount::COUNT_1;
    s.gamma_texture = s.device->createTexture(desc);
    if (!s.gamma_texture) {
      return false;
    }
    plume::RenderTextureViewDesc view_desc;
    view_desc.format = plume::RenderFormat::R16G16B16A16_UNORM;
    view_desc.dimension = plume::RenderTextureViewDimension::TEXTURE_2D;
    view_desc.mipLevels = 1;
    s.gamma_view = s.gamma_texture->createTextureView(view_desc);
    if (!s.gamma_view) {
      s.gamma_texture.reset();
      return false;
    }
  }
  std::array<uint16_t, 256 * 4> texels;
  for (uint32_t i = 0; i < 256; ++i) {
    for (uint32_t c = 0; c < 3; ++c) {
      texels[i * 4 + c] = ramp ? uint16_t((uint32_t(ramp[c * 256 + i] >> 6) * 65535 + 511) / 1023)
                               : uint16_t(i * 257);
    }
    texels[i * 4 + 3] = 0xFFFF;
  }
  if (s.gamma_uploaded && texels == s.gamma_texels) {
    return true;
  }
  UploadAllocation up = Video::Upload(texels.data(), uint32_t(sizeof(texels)), 0x200);
  if (!up) {
    return s.gamma_uploaded;
  }
  list->barriers(plume::RenderBarrierStage::COPY,
                 plume::RenderTextureBarrier(s.gamma_texture.get(), plume::RenderTextureLayout::COPY_DEST));
  list->copyTextureRegion(plume::RenderTextureCopyLocation::Subresource(s.gamma_texture.get(), 0),
                          plume::RenderTextureCopyLocation::PlacedFootprint(
                              up.buffer, plume::RenderFormat::R16G16B16A16_UNORM, 256, 1, 1, 256, up.offset));
  list->barriers(plume::RenderBarrierStage::GRAPHICS,
                 plume::RenderTextureBarrier(s.gamma_texture.get(), plume::RenderTextureLayout::SHADER_READ));
  s.gamma_texels = texels;
  s.gamma_uploaded = true;
  return true;
}

// Letterboxed blit of the guest front buffer into the bound back buffer.
void DrawPresentSource(VideoState &s, plume::RenderCommandList *list,
                       const Video::PresentSource &source, uint32_t w, uint32_t h) {
  if (!source.texture || !source.view || !source.width || !source.height || !s.present_pipeline ||
      !s.gamma_uploaded) {
    return;
  }
  const uint64_t frame_number = s.frames_presented.load(std::memory_order_relaxed);
  if (s.capture_interval && frame_number % s.capture_interval == 0) {
    char name[32];
    std::snprintf(name, sizeof(name), "capture_%06llu", (unsigned long long)frame_number);
    QueueCapture(s, list, source.texture, source.width, source.height, name);
  }
  plume::RenderDescriptorSetBuilder tex_b, samp_b;
  BuildPresentSetLayouts(tex_b, samp_b);
  auto set = tex_b.create(s.device.get());
  if (!set) {
    return;
  }
  set->setTexture(0, source.texture, plume::RenderTextureLayout::SHADER_READ, source.view);
  set->setTexture(1, s.gamma_texture.get(), plume::RenderTextureLayout::SHADER_READ, s.gamma_view.get());
  const float scale = std::min(float(w) / float(source.width), float(h) / float(source.height));
  float vw = float(source.width) * scale;
  float vh = float(source.height) * scale;
  // Narrower than 16:9: the frame was drawn for the display's aspect (taller
  // screen targets, renderer.cpp), so it fills the window.
  if (const double aspect = Video::DisplayAspect(); aspect >= 1.0 && aspect < 16.0 / 9.0 - 0.01) {
    vw = float(w);
    vh = float(h);
  }
  list->setViewports(plume::RenderViewport((float(w) - vw) * 0.5f, (float(h) - vh) * 0.5f, vw, vh));
  list->setScissors(plume::RenderRect(0, 0, int32_t(w), int32_t(h)));
  list->setPipeline(s.present_pipeline.get());
  list->setGraphicsPipelineLayout(s.present_layout.get());
  list->setGraphicsDescriptorSet(set.get(), 0);
  list->setGraphicsDescriptorSet(s.present_sampler_set.get(), 1);
  s.slots[s.frame].draws.push_back({}); // present
  list->drawInstanced(3, 1, 0, 0);
  s.slots[s.frame].parked.push_back(std::shared_ptr<void>(std::move(set)));
  list->setViewports(plume::RenderViewport(0.0f, 0.0f, float(w), float(h)));
}

} // namespace

void Video::Present(const PresentSource &source) {
  auto &s = state();
  if (s.shutting_down.load(std::memory_order_acquire)) {
    return;
  }
  std::lock_guard lock(s.mutex);
  if (!s.ready) {
    return;
  }
  plume::RenderCommandList *list = OpenListLocked(s);
  auto &slot = s.slots[s.frame];

  if (!s.swap_chain) {
    SubmitWithoutPresent(s);
    s.frames_presented.fetch_add(1, std::memory_order_relaxed);
    return;
  }
  if (s.swap_chain->needsResize() ||
      s.resize_requested.exchange(false, std::memory_order_acq_rel)) {
    SubmitWithoutPresent(s);
    WaitAllFrames(s);
    s.framebuffers.clear();
    if (!s.swap_chain->resize() || !BuildSwapChainTargets(s)) {
      // Minimised windows report a 0x0 swapchain; retry next frame.
      if (s.swap_chain->getWidth() && s.swap_chain->getHeight()) {
        REXGPU_ERROR("PGR4 D3D12: swapchain resize failed");
      }
      s.resize_requested.store(true, std::memory_order_release);
      return;
    }
    REXGPU_INFO("PGR4 D3D12: swapchain resized to {}x{}", s.swap_chain->getWidth(), s.swap_chain->getHeight());
    return;
  }
  if (s.framebuffers.empty()) {
    SubmitWithoutPresent(s);
    return;
  }

  uint32_t image = 0;
  if (!s.swap_chain->acquireTexture(slot.acquire_semaphore.get(), &image)) {
    SubmitWithoutPresent(s);
    s.resize_requested.store(true, std::memory_order_release);
    return;
  }
  UpdateGammaRamp(s, list, source.gamma_ramp);
  plume::RenderTexture *back = s.swap_chain->getTexture(image);
  plume::RenderFramebuffer *back_fb = s.framebuffers[image].get();
  const uint32_t w = s.swap_chain->getWidth();
  const uint32_t h = s.swap_chain->getHeight();

  list->barriers(plume::RenderBarrierStage::GRAPHICS,
                 plume::RenderTextureBarrier(back, plume::RenderTextureLayout::COLOR_WRITE));
  list->setFramebuffer(back_fb);
  list->setViewports(plume::RenderViewport(0.0f, 0.0f, float(w), float(h)));
  list->setScissors(plume::RenderRect(0, 0, int32_t(w), int32_t(h)));
  list->clearColor(0, plume::RenderColor(0.0f, 0.0f, 0.0f, 1.0f));
  DrawPresentSource(s, list, source, w, h);
  if (s.overlay_hook && !s.shutting_down.load(std::memory_order_acquire)) {
    s.overlay_hook(list, back_fb, w, h);
  }
  list->setFramebuffer(nullptr);
  list->barriers(plume::RenderBarrierStage::GRAPHICS,
                 plume::RenderTextureBarrier(back, plume::RenderTextureLayout::PRESENT));
  list->end();
  s.list_open = false;

  const plume::RenderCommandList *lists[] = {list};
  plume::RenderCommandSemaphore *waits[] = {slot.acquire_semaphore.get()};
  plume::RenderCommandSemaphore *signals[] = {s.render_semaphores[image].get()};
  if (s.before_submit) {
    s.before_submit();
  }
  s.queue->executeCommandLists(lists, 1, waits, 1, signals, 1, slot.fence.get());
  slot.submitted = true;
  const auto present_t0 = std::chrono::steady_clock::now();
  if (!s.swap_chain->present(image, signals, 1)) {
    REXGPU_WARN("PGR4 D3D12: present failed");
  }
  s.present_ns += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                               std::chrono::steady_clock::now() - present_t0)
                               .count());
  DrainDebugMessages(s);
  if (REXCVAR_GET(gpu_sync)) {
    WaitAllFrames(s);
  }
  s.frame = (s.frame + 1) % kNumFrames;

  const uint64_t n = s.frames_presented.fetch_add(1, std::memory_order_relaxed) + 1;
  if (n == 1 || n % 600 == 0) {
    REXGPU_INFO("PGR4 D3D12: presented {} frames", n);
  }
  if (n % 600 == 0) {
    // Per-frame averages over the last 600 frames. What is left of the frame
    // after the renderer, the GPU wait and present is the guest's own CPU work.
    const auto now = std::chrono::steady_clock::now();
    const double wall = std::chrono::duration<double, std::milli>(now - s.report_start).count();
    const double per = 1.0 / 600.0;
    const double renderer = s.renderer_ns.exchange(0) * 1e-6 * per;
    const double gpu = s.gpu_wait_ns * 1e-6 * per;
    const double present = s.present_ns * 1e-6 * per;
    const double frame = wall * per;
    REXGPU_INFO("PGR4 D3D12: {:.1f} fps, frame {:.1f} ms = renderer {:.1f} + gpu wait {:.1f} + present {:.1f} + "
                "guest/other {:.1f}, {:.0f} draws/frame",
                1000.0 / frame, frame, renderer, gpu, present, frame - renderer - gpu - present,
                s.renderer_draws.exchange(0) * per);
    s.gpu_wait_ns = 0;
    s.present_ns = 0;
    s.report_start = now;
  }
}

} // namespace pgr4::gpu
