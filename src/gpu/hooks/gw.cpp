// PGR4-Recomp native renderer: Geometry Wars' D3D hooks.
//
// Geometry Wars: Waves (gw.xex, PGR4's arcade machine) is linked in as a
// second title with every symbol prefixed gw_ (gw/gw_manifest.toml). Its XDK
// D3D is the same code as PGR4's, so these hooks route its calls into the same
// renderer the way device.cpp, draw.cpp, shader.cpp and state.cpp do for PGR4
// (addresses in config/gw_d3d_functions.toml). gw.xex has no tiling, Z pass,
// queries, effect-loader shader headers, indexed or DrawVerticesUP draws.
#include <rex/hook.h>
#include <rex/runtime.h>
#include <rex/types.h>

#include "gpu/hooks/hooks.h"
#include "gpu/renderer.h"

REX_EXTERN(__imp__gw_Direct3D_CreateDevice);
REX_EXTERN(gw_D3D__CreatePresentationSurfaces);
REX_EXTERN(__imp__gw_D3DDevice_DrawVertices);
REX_EXTERN(__imp__gw_D3DDevice_CreateVertexShader);
REX_EXTERN(__imp__gw_D3DDevice_CreatePixelShader);
REX_EXTERN(__imp__gw_D3DDevice_SetViewport);

using pgr4::gpu::Renderer;
using namespace pgr4::gpu::hooks;

namespace {
// gw.xex's import thunk slot for xboxkrnl's VdGlobalDevice.
constexpr uint32_t kGwVdGlobalDeviceImport = 0x82000528;
} // namespace

REX_HOOK_RAW(gw_Direct3D_CreateDevice) {
  CreateDevice(ctx, base, __imp__gw_Direct3D_CreateDevice, kGwVdGlobalDeviceImport, 0,
               gw_D3D__CreatePresentationSurfaces);
}

REX_HOOK(gw_D3DDevice_Clear, D3DDevice_Clear_hook);
REX_HOOK(gw_D3DDevice_Swap, D3DDevice_Swap_hook);
REX_HOOK(gw_D3DDevice_Resolve, D3DDevice_Resolve_hook);

// D3DDevice_DrawVertices(device, PrimitiveType, StartVertex, VertexCount)
REX_HOOK_RAW(gw_D3DDevice_DrawVertices) {
  const uint32_t device = ctx.r3.u32, prim = ctx.r4.u32, start = ctx.r5.u32, count = ctx.r6.u32;
  __imp__gw_D3DDevice_DrawVertices(ctx, base);
  Renderer::DrawVertices(device, prim, start, count);
}

REX_HOOK(gw_D3DDevice_BeginVertices, D3DDevice_BeginVertices_hook);
REX_HOOK(gw_D3DDevice_EndVertices, D3DDevice_EndVertices_hook);

REX_HOOK_RAW(gw_D3DDevice_CreateVertexShader) {
  const uint32_t container = ctx.r3.u32;
  __imp__gw_D3DDevice_CreateVertexShader(ctx, base);
  Renderer::OnShaderCreated(ctx.r3.u32, container, false);
}

REX_HOOK_RAW(gw_D3DDevice_CreatePixelShader) {
  const uint32_t container = ctx.r3.u32;
  __imp__gw_D3DDevice_CreatePixelShader(ctx, base);
  Renderer::OnShaderCreated(ctx.r3.u32, container, true);
}

REX_HOOK_RAW(gw_D3DDevice_SetViewport) {
  const uint32_t device = ctx.r3.u32, viewport = ctx.r4.u32;
  __imp__gw_D3DDevice_SetViewport(ctx, base);
  Renderer::OnViewport(device, viewport);
}
