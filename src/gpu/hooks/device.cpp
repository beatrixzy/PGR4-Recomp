// PGR4-Recomp native renderer: device lifecycle hooks.
//
// Direct3D_CreateDevice (0x826CD270) has a branch for device type 2 that
// builds a GPU-less command-buffer device: it sets device+0x2ABC bit 0x80,
// points the ring (+0x30 write, +0x34 limit, +0x38 guarantee) at the
// device's own 4800-byte scratch buffer (sub_826CD868), and runs
// ResetAllState (sub_826DA5B0), which fills the render/sampler state
// dispatch tables from 0x82A31E90 / 0x82A32350 and applies their defaults.
// KickOffAndGetSpace and the ring wrap (sub_826CE4E0) just rewind that
// scratch buffer in this mode, so nothing waits on a GPU and no Vd* kernel
// call is made. Forcing type 2 keeps PGR4's own D3D state code (setters,
// register shadows, dirty masks) intact while every PM4 packet is discarded;
// the hooks below and in the other hooks/*.cpp files replace the calls that
// would have produced visible output.
#include <windows.h>

#include <rex/cvar.h>
#include <rex/hook.h>
#include <rex/logging.h>
#include <rex/memory/utils.h>
#include <rex/runtime.h>
#include <rex/system/kernel_state.h>
#include <rex/types.h>

#include "core/hooks.h"
#include "gpu/hooks/hooks.h"
#include "gpu/renderer.h"
#include "gpu/video.h"

REX_EXTERN(__imp__Direct3D_CreateDevice);
REX_EXTERN(D3D__CreatePresentationSurfaces);

REXCVAR_DECLARE(double, aspect_ratio);

namespace pgr4::gpu::hooks {

// D3DDEVTYPE value that selects the command-buffer device path.
constexpr uint32_t kCommandBufferDeviceType = 2;

// PGR4's 16:9 camera aspect (.rdata float), read by the camera and view
// setup code (sub_824C4B80 and its neighbours). --aspect_ratio rewrites it so
// the camera sees an ultrawide view; the renderer gives the scene targets the
// matching extra width (TargetScale in renderer.cpp).
constexpr uint32_t kCameraAspect = 0x820010A8;

// PGR4's import thunk slot holding the guest address of xboxkrnl's
// VdGlobalDevice.
constexpr uint32_t kVdGlobalDeviceImport = 0x82000898;

// The device the game created for the GPU (published as VdGlobalDevice).
uint32_t g_primary_device = 0;

// The Xenon ABI reserves a GPR slot for every float argument (Z is in f1 but
// still consumes r8), so the placeholder absorbs r8 and Stencil is read from
// r9. Same signature re:Blue uses for its D3DDevice_Clear hook.
u32 D3DDevice_Clear_hook(u32 device, u32 /*count*/, u32 /*rects*/, u32 flags, u32 color, f64 z,
                         u32 /*z_gpr_slot*/, u32 stencil, u32 /*edram_clear*/) {
  pgr4::gpu::Renderer::Clear(device, flags, color, float(z), stencil);
  return 0;
}

// D3DDevice_Swap(D3DDevice*, D3DBaseTexture* pFrontBuffer, scaler params).
// pFrontBuffer is the texture the frame was resolved into.
u32 D3DDevice_Swap_hook(u32 device, u32 front_buffer, u32 /*params*/) {
  pgr4::gpu::Renderer::Swap(device, front_buffer);
  return 0;
}

// Direct3D_CreateDevice(adapter, deviceType, focusWindow, behaviorFlags,
//                       pPresentationParameters, ppReturnedDeviceInterface)
void CreateDevice(PPCContext &ctx, uint8_t *base, PPCFunc *original, uint32_t vd_global_device_import,
                  uint32_t camera_aspect, PPCFunc *create_surfaces) {
  const uint32_t requested_type = ctx.r4.u32;
  // Narrower than 16:9 (the Deck's 16:10): the camera takes the aspect too,
  // and the 16:9 frame is stretched to fill the window (video.cpp).
  if (const double aspect = pgr4::gpu::Video::DisplayAspect(); camera_aspect && aspect >= 1.0 && aspect <= 6.0) {
    // .rdata is mapped read-only on the host.
    uint8_t *host = REX_KERNEL_MEMORY()->TranslateVirtual(camera_aspect);
    DWORD old = 0;
    if (VirtualProtect(host, 4, PAGE_READWRITE, &old)) {
      rex::memory::store_and_swap<float>(host, float(aspect));
      VirtualProtect(host, 4, old, &old);
    }
  }
  const uint32_t present_params = ctx.r7.u32;
  const uint32_t out_device_ptr = ctx.r8.u32;
  ctx.r4.u64 = kCommandBufferDeviceType;
  original(ctx, base);
  const uint32_t hr = ctx.r3.u32;

  uint32_t device = 0;
  if (hr == 0 && out_device_ptr) {
    auto *memory = REX_KERNEL_MEMORY();
    device = rex::memory::load_and_swap<uint32_t>(memory->TranslateVirtual(out_device_ptr));
    // The GPU init path (sub_826DACE8, 0x826DAD54) publishes the device in
    // the kernel's VdGlobalDevice; the command-buffer path skips it, but the
    // resource code (e.g. sub_826CF2B0's owner-thread check) reads the device
    // back through it, from the VdGlobalDevice import slot. The
    // variable lives in the physical heap, so it must go through
    // TranslateVirtual (recompiled loads add the physical host offset).
    const uint32_t vd_global_device = rex::memory::load_and_swap<uint32_t>(
        memory->TranslateVirtual(vd_global_device_import));
    // PGR4 also creates genuine command-buffer devices of its own (type 2);
    // only the device it asked for as the GPU device is the global one.
    if (vd_global_device && requested_type != kCommandBufferDeviceType) {
      rex::memory::store_and_swap<uint32_t>(memory->TranslateVirtual(vd_global_device), device);
      g_primary_device = device;
    }
    if (create_surfaces && device && present_params) {
      rex::CallFrame frame(ctx);
      frame.ctx.r3.u64 = device;
      frame.ctx.r4.u64 = present_params;
      create_surfaces(frame, base);
    }
  }
  REXGPU_INFO("PGR4 D3D12: Direct3D_CreateDevice(type {} -> {}) = {:08X}, device {:08X}",
              requested_type, kCommandBufferDeviceType, hr, device);
}

} // namespace pgr4::gpu::hooks

using namespace pgr4::gpu::hooks;

REX_HOOK_RAW(Direct3D_CreateDevice) {
  CreateDevice(ctx, base, __imp__Direct3D_CreateDevice, kVdGlobalDeviceImport, kCameraAspect);
}

// D3D's Z pass (sub_826D2590 begins it, sub_826D28D8 ends it): the GPU runs
// the draws in between twice, depth only and then shaded. Its bookkeeping
// needs the GPU ring the command-buffer device does not have (EndZPass reads
// device+0x2A90, and resolves under it carve space from the ring's secondary
// buffer and wait for the GPU to consume it), so the first career race, the
// first to use it, crashed or hung. Without it the draws render once, shaded.
REX_HOOK_RAW(sub_826D2590) {
  pgr4::gpu::Renderer::OnZPass(ctx.r3.u32, true);
  ctx.r3.u64 = 0;
}
REX_HOOK_RAW(sub_826D28D8) {
  pgr4::gpu::Renderer::OnZPass(ctx.r3.u32, false);
  ctx.r3.u64 = 0;
}

// D3DDevice_Reset(device, pPresentationParameters) (0x826CD008). The original
// re-initialises the GPU ring (sub_826CEC18 -> VdInitializeRingBuffer) before
// rebuilding the presentation surfaces. A command-buffer device has no ring
// to rebuild, so only the CPU-side surface setup runs: sub_826DA818 queries
// the video mode and creates the front buffer texture and depth surface with
// PGR4's own CreateTexture/CreateSurface. Returns S_OK or E_OUTOFMEMORY like
// the original.
REX_HOOK_RAW(D3DDevice_Reset) {
  const uint32_t device = ctx.r3.u32;
  const uint32_t params = ctx.r4.u32;
  rex::CallFrame frame(ctx);
  frame.ctx.r3.u64 = device;
  frame.ctx.r4.u64 = params;
  D3D__CreatePresentationSurfaces(frame, base);
  const bool ok = frame.ctx.r3.u32 != 0;
  ctx.r3.u64 = ok ? 0 : 0x8007000Eu;
  REXGPU_INFO("PGR4 D3D12: D3DDevice_Reset({:08X}) surfaces {}", device, ok ? "ok" : "failed");
}

REX_HOOK(D3DDevice_Clear, D3DDevice_Clear_hook);
REX_HOOK(D3DDevice_Swap, D3DDevice_Swap_hook);
