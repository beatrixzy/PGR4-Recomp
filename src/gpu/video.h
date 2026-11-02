// PGR4-Recomp native renderer: the Plume (D3D12) host device.
//
// The guest D3D device is PGR4's own XDK D3D running as a command-buffer
// ("null") device: Direct3D_CreateDevice is called with device type 2, which
// keeps every state setter and register shadow working while all PM4 it
// would write lands in the device's 4800-byte scratch ring and is discarded
// (see gpu/hooks/device.cpp). This file owns the real device, swapchain and
// per-frame command lists that the hooked D3D calls record into.
#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>

namespace plume {
struct RenderBuffer;
struct RenderCommandList;
struct RenderDescriptorSet;
struct RenderDevice;
struct RenderFramebuffer;
struct RenderTexture;
struct RenderTextureView;
} // namespace plume

namespace pgr4::gpu {

// Bindless descriptor heap sizes shared by every host pipeline. They come out
// of plume's fixed shader-visible heaps (262144 views, 2048 samplers), which
// the guest renderer's per-draw tables also need.
constexpr uint32_t kBindlessTextureCount = 1024;
constexpr uint32_t kBindlessSamplerCount = 16;
// Sampler slot 0: linear/clamp (D3D9 reset state). Slot 1: point/clamp.
constexpr uint32_t kLinearClampSamplerSlot = 0;
constexpr uint32_t kPointClampSamplerSlot = 1;

// Per-frame transient upload memory, valid until the frame's fence signals.
struct UploadAllocation {
  void *memory = nullptr;
  const plume::RenderBuffer *buffer = nullptr;
  uint64_t offset = 0;
  uint32_t size = 0;
  explicit operator bool() const { return memory != nullptr; }
};

class Video {
public:
  // UI thread, before the guest module launches. hwnd is the game window.
  static bool CreateHostDevice(void *hwnd);
  static void Shutdown();

  // Window pixel size changed (UI thread). The swapchain is rebuilt on the
  // render thread at the next frame boundary.
  static void RequestResize();
  // The display aspect the game is shaped for: --aspect_ratio, or with it
  // unset, the swapchain's when narrower than 16:9 (the Steam Deck's 16:10).
  // 0 keeps 16:9. Fixed at the first call, after the swapchain exists.
  static double DisplayAspect();

  // D3DDevice_Swap: finish and present the frame being recorded. source is
  // the guest front buffer's host copy (drawn letterboxed into the swapchain,
  // already in SHADER_READ); nullptr presents black.
  struct PresentSource {
    const plume::RenderTexture *texture = nullptr;
    const plume::RenderTextureView *view = nullptr;
    uint32_t width = 0;
    uint32_t height = 0;
    // The guest's 256-entry gamma ramp, applied at scan-out like the 360's
    // display controller: red, green then blue, 256 entries each, 10-bit
    // values in the top bits (as D3D keeps them). nullptr: none.
    const uint16_t *gamma_ramp = nullptr;
  };
  static void Present(const PresentSource &source);

  // Frames presented since boot.
  static uint64_t FrameCount();

  // Frame timing: CPU time the renderer spent turning guest draws and
  // resolves into host commands. Present reports it with the time spent
  // waiting on the GPU every 600 frames.
  static void AddRendererTime(uint64_t ns, uint32_t draws);
  // Nanoseconds spent waiting on frame fences since boot.
  static uint64_t GpuWaitTotal();

  // --- Shared host resources --------------------------------------------
  // The accessors below need exclusive access to the frame being recorded:
  // call them from the render thread, or from the UI thread while Present is
  // parked inside the overlay hook.
  static plume::RenderDevice *HostDevice();
  // The frame's command list while it is open, else nullptr.
  static plume::RenderCommandList *CommandList();
  // Opens the frame's command list if needed (render thread). Guest draws
  // record into it until Present submits it.
  static plume::RenderCommandList *OpenCommandList();
  // Bumped each time a command list begins recording: bindings recorded
  // before are gone.
  static uint64_t ListGeneration();
  static plume::RenderDescriptorSet *TextureDescriptorSet();
  static plume::RenderDescriptorSet *SamplerDescriptorSet();
  // UINT32_MAX when the heap is full.
  static uint32_t AllocateTextureSlot();
  // The slot is reused only after the current frame's fence signals.
  static void FreeTextureSlot(uint32_t slot);
  static UploadAllocation Upload(const void *data, uint32_t size, uint32_t alignment);
  // Keep an object alive until the frame recording now has finished on GPU.
  template <typename T> static void ParkUntilFence(std::unique_ptr<T> object) {
    if (object) {
      ParkErased(std::shared_ptr<void>(std::move(object)));
    }
  }
  static void ParkErased(std::shared_ptr<void> object);
  // Called right before each submission: upload memory filled by other
  // threads (texture conversion workers) must be complete by then.
  static void SetBeforeSubmit(void (*fn)());
  // Records the shaders of the draw about to be recorded (GPU crash reports).
  // A guest draw about to be recorded (crash reports), and its end.
  struct DrawNote {
    uint64_t vs = 0, ps = 0;
    uint32_t count = 0, first = 0;
    int32_t base_vertex = 0;
    bool indexed = false;
  };
  static void NoteDraw(const DrawNote &note);
  static void NoteDrawDone();

  // Debug: reads back a texture (R8G8B8A8_UNORM or
  // R16G16B16A16_FLOAT) and writes <name>.bmp once the frame completes; it
  // goes back to COLOR_WRITE for a render target, else SHADER_READ.
  static void CaptureTexture(const plume::RenderTexture *texture, uint32_t width, uint32_t height,
                             const std::string &name,
                             bool render_target = false);

  // Called from Present with the back buffer bound (COLOR_WRITE), after the
  // frame and before the transition to PRESENT. Used by the ImGui overlay.
  using OverlayDrawHook = std::function<void(plume::RenderCommandList *, plume::RenderFramebuffer *,
                                             uint32_t width, uint32_t height)>;
  static void SetOverlayDrawHook(OverlayDrawHook hook);
};

} // namespace pgr4::gpu
