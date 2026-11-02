// PGR4-Recomp native renderer: draw-call hooks.
//
// DrawVertices / DrawIndexedVertices run PGR4's original first: on the
// command-buffer device it only flushes dirty state into the discarded
// scratch ring, which keeps the D3D bookkeeping (dirty masks, resource
// fences) exactly as the game expects. The draw is then rebuilt on the host
// from the device's state shadows (gpu/renderer.cpp).
//
// D3DDevice_BeginVertices (0x826CFA60) normally carves the vertex data for an
// immediate-mode batch out of the PM4 ring and returns a pointer into it;
// D3DDevice_EndVertices (0x826CFF00) and the tail of DrawVerticesUP
// (0x826CFF10) then restore the ring pointer from device+0x3474. The
// command-buffer device's ring is a 4800-byte scratch buffer, so larger
// batches came back NULL and the game's vertex writer (sub_822BCFA8) stored
// through it. These hooks hand out guest scratch memory instead and draw the
// batch at EndVertices, like re:Blue's.
#include <rex/hook.h>
#include <rex/logging.h>
#include <rex/runtime.h>
#include <rex/system/kernel_state.h>
#include <rex/types.h>

#include "core/hooks.h"
#include "gpu/hooks/hooks.h"
#include "gpu/renderer.h"

REX_EXTERN(__imp__D3DDevice_DrawVertices);
REX_EXTERN(__imp__D3DDevice_DrawIndexedVertices);

namespace pgr4::gpu::hooks {

using pgr4::gpu::Renderer;

// Guest-visible scratch for BeginVertices data. Batches are written by the
// game and consumed at EndVertices, so a wrapping arena is enough.
constexpr uint32_t kVertexScratchBytes = 8u << 20;

struct ImmediateBatch {
  uint32_t scratch_base = 0;
  uint32_t scratch_offset = 0;
  // Pending BeginVertices batch.
  uint32_t device = 0;
  uint32_t prim_type = 0;
  uint32_t vertex_count = 0;
  uint32_t stride = 0;
  uint32_t data = 0;
};

ImmediateBatch &batch() {
  static ImmediateBatch b;
  return b;
}

uint32_t AllocVertexScratch(uint32_t bytes) {
  auto &b = batch();
  if (!b.scratch_base) {
    b.scratch_base = REX_KERNEL_MEMORY()->SystemHeapAlloc(kVertexScratchBytes, 128);
    if (!b.scratch_base) {
      REXGPU_ERROR("PGR4 D3D12: vertex scratch allocation failed");
      return 0;
    }
  }
  bytes = (bytes + 127) & ~127u;
  if (bytes > kVertexScratchBytes) {
    REXGPU_ERROR("PGR4 D3D12: BeginVertices batch of {} bytes exceeds scratch", bytes);
    return 0;
  }
  if (b.scratch_offset + bytes > kVertexScratchBytes) {
    b.scratch_offset = 0;
  }
  const uint32_t va = b.scratch_base + b.scratch_offset;
  b.scratch_offset += bytes;
  return va;
}

// D3DDevice_BeginVertices(device, PrimitiveType, VertexCount, VertexStreamZeroStride)
u32 D3DDevice_BeginVertices_hook(u32 device, u32 prim_type, u32 vertex_count, u32 stride) {
  auto &b = batch();
  const uint32_t va = AllocVertexScratch(vertex_count * stride);
  b.device = device;
  b.prim_type = prim_type;
  b.vertex_count = vertex_count;
  b.stride = stride;
  b.data = va;
  return va;
}

void D3DDevice_EndVertices_hook(u32 /*device*/) {
  auto &b = batch();
  const ImmediateBatch pending = b;
  // Consume: a stray EndVertices must not redraw.
  b.data = 0;
  if (pending.data && pending.vertex_count) {
    Renderer::DrawUserPointer(pending.device, pending.prim_type, pending.vertex_count,
                              pending.data, pending.stride);
  }
}

// D3DDevice_DrawVerticesUP(device, PrimitiveType, VertexCount, pVertexStreamZeroData,
//                          VertexStreamZeroStride)
void D3DDevice_DrawVerticesUP_hook(u32 device, u32 prim_type, u32 vertex_count, u32 data,
                                   u32 stride) {
  Renderer::DrawUserPointer(device, prim_type, vertex_count, data, stride);
}

// D3DDevice_Resolve(device, Flags, pSourceRect, pDestTexture, pDestPoint, DestLevel,
//                   DestSliceOrFace, pClearColor, ClearZ, ClearStencil, pParameters).
// ClearZ is a float in f1 (sub_826CBEE0 saves it with fmr f30, f1); like
// BeginTiling, it still takes a GPR slot. PGR4 uses reversed Z, so the
// depth clear value matters (0 for its GEQUAL passes).
u32 D3DDevice_Resolve_hook(u32 device, u32 flags, u32 source_rect, u32 dest_texture,
                           u32 dest_point, u32 dest_level, u32 dest_slice,
                           u32 clear_color, f64 clear_z, u32 /*z_gpr_slot*/,
                           u32 /*clear_stencil*/, u32 /*parameters*/) {
  Renderer::Resolve(device, flags, source_rect, dest_texture, dest_point, dest_level, dest_slice,
                    clear_color, float(clear_z));
  return 0;
}

// D3DDevice_BeginTiling(device, Flags, Count, pTileRects, pClearColor, ClearZ,
//                       ClearStencil). ClearZ is in f1 but still takes a GPR
// slot, like Clear (re:Blue's signature).
u32 D3DDevice_BeginTiling_hook(u32 device, u32 /*flags*/, u32 count, u32 tile_rects,
                               u32 clear_color, f64 clear_z, u32 /*z_gpr_slot*/,
                               u32 clear_stencil) {
  Renderer::BeginTiling(device, count, tile_rects, clear_color, float(clear_z), clear_stencil);
  return 0;
}

// D3DDevice_EndTiling(device, ResolveFlags, pResolveRects, pDestTexture,
//                     pClearColor, ClearZ, ClearStencil, pParameters).
u32 D3DDevice_EndTiling_hook(u32 device, u32 resolve_flags, u32 /*resolve_rects*/,
                             u32 dest_texture, u32 clear_color, f64 clear_z,
                             u32 /*z_gpr_slot*/, u32 /*clear_stencil*/, u32 /*parameters*/) {
  Renderer::EndTiling(device, resolve_flags, dest_texture, clear_color, float(clear_z));
  return 0;
}

void ResetForTitleSwitch() { batch() = {}; }

} // namespace pgr4::gpu::hooks

using namespace pgr4::gpu::hooks;

// D3DDevice_DrawVertices(device, PrimitiveType, StartVertex, VertexCount)
REX_HOOK_RAW(D3DDevice_DrawVertices) {
  const uint32_t device = ctx.r3.u32, prim = ctx.r4.u32, start = ctx.r5.u32, count = ctx.r6.u32;
  __imp__D3DDevice_DrawVertices(ctx, base);
  Renderer::DrawVertices(device, prim, start, count);
}

// D3DDevice_DrawIndexedVertices(device, PrimitiveType, BaseVertexIndex,
//                               StartIndex, IndexCount)
REX_HOOK_RAW(D3DDevice_DrawIndexedVertices) {
  const uint32_t device = ctx.r3.u32, prim = ctx.r4.u32;
  const int32_t base_vertex = ctx.r5.s32;
  const uint32_t start_index = ctx.r6.u32, index_count = ctx.r7.u32;
  __imp__D3DDevice_DrawIndexedVertices(ctx, base);
  Renderer::DrawIndexed(device, prim, base_vertex, start_index, index_count);
}

REX_HOOK(D3DDevice_BeginVertices, D3DDevice_BeginVertices_hook);
REX_HOOK(D3DDevice_EndVertices, D3DDevice_EndVertices_hook);
REX_HOOK(D3DDevice_DrawVerticesUP, D3DDevice_DrawVerticesUP_hook);
REX_HOOK(D3DDevice_Resolve, D3DDevice_Resolve_hook);
REX_HOOK(D3DDevice_BeginTiling, D3DDevice_BeginTiling_hook);
REX_HOOK(D3DDevice_EndTiling, D3DDevice_EndTiling_hook);
