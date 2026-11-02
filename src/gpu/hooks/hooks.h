// PGR4-Recomp native renderer: hook bodies shared by PGR4's D3D hooks and
// Geometry Wars' (gw.cpp), whose XDK D3D is the same code at other addresses.
#pragma once

#include <rex/hook.h>
#include <rex/types.h>

namespace pgr4::gpu::hooks {

// device.cpp. The guest addresses CreateDevice patches differ per title:
// the VdGlobalDevice import slot, and PGR4's camera aspect (0 = none).
// create_surfaces (the title's D3D__CreatePresentationSurfaces), if given,
// builds and binds the default back buffer and depth surface, which only the
// skipped GPU path makes (PGR4 gets them from its own D3DDevice_Reset call).
void CreateDevice(PPCContext &ctx, uint8_t *base, PPCFunc *original, uint32_t vd_global_device_import,
                  uint32_t camera_aspect, PPCFunc *create_surfaces = nullptr);
u32 D3DDevice_Clear_hook(u32 device, u32 count, u32 rects, u32 flags, u32 color, f64 z, u32 z_gpr_slot,
                         u32 stencil, u32 edram_clear);
u32 D3DDevice_Swap_hook(u32 device, u32 front_buffer, u32 params);

// draw.cpp
u32 D3DDevice_BeginVertices_hook(u32 device, u32 prim_type, u32 vertex_count, u32 stride);
void D3DDevice_EndVertices_hook(u32 device);
u32 D3DDevice_Resolve_hook(u32 device, u32 flags, u32 source_rect, u32 dest_texture, u32 dest_point,
                           u32 dest_level, u32 dest_slice, u32 clear_color, f64 clear_z, u32 z_gpr_slot,
                           u32 clear_stencil, u32 parameters);

// Title switch: forget guest memory owned by the old title (the vertex
// scratch block, the primary device).
void ResetForTitleSwitch();

} // namespace pgr4::gpu::hooks
