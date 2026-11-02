// PGR4-Recomp native renderer: draw state setters (the set re:Blue hooks).
//
// PGR4's own setters keep the device shadows current, and the renderer reads
// those at draw time, so these run the original unchanged. SetViewport also
// records the D3DVIEWPORT9, which the device only keeps as derived
// PA_CL_VPORT_* register values.
#include <rex/hook.h>
#include <rex/runtime.h>
#include <rex/types.h>

#include "gpu/renderer.h"

REX_EXTERN(__imp__D3DDevice_SetViewport);
REX_EXTERN(__imp__D3DDevice_SetRenderTarget);
REX_EXTERN(__imp__D3DDevice_SetDepthStencilSurface);
REX_EXTERN(__imp__D3DDevice_SetScissorRect);
REX_EXTERN(__imp__D3DDevice_SetVertexShader);
REX_EXTERN(__imp__D3DDevice_SetPixelShader);
REX_EXTERN(__imp__D3DDevice_SetVertexDeclaration);
REX_EXTERN(__imp__D3DDevice_SetTexture);
REX_EXTERN(__imp__D3DDevice_SetStreamSource);
REX_EXTERN(__imp__D3DDevice_SetIndices);

// D3DDevice_SetViewport(device, const D3DVIEWPORT9 *)
REX_HOOK_RAW(D3DDevice_SetViewport) {
  const uint32_t device = ctx.r3.u32, viewport = ctx.r4.u32;
  __imp__D3DDevice_SetViewport(ctx, base);
  pgr4::gpu::Renderer::OnViewport(device, viewport);
}

// Shadowed at device+0x3090 (render targets), +0x30A0 (depth), +0x317C
// (scissor), +0x3190/+0x318C (shaders), +0x2E24 (declaration), fetch
// constants at +0x480 (textures, streams) and +0x308C (index buffer).
REX_HOOK_RAW(D3DDevice_SetRenderTarget) { __imp__D3DDevice_SetRenderTarget(ctx, base); }
REX_HOOK_RAW(D3DDevice_SetDepthStencilSurface) { __imp__D3DDevice_SetDepthStencilSurface(ctx, base); }
REX_HOOK_RAW(D3DDevice_SetScissorRect) { __imp__D3DDevice_SetScissorRect(ctx, base); }
REX_HOOK_RAW(D3DDevice_SetVertexShader) { __imp__D3DDevice_SetVertexShader(ctx, base); }
REX_HOOK_RAW(D3DDevice_SetPixelShader) { __imp__D3DDevice_SetPixelShader(ctx, base); }
REX_HOOK_RAW(D3DDevice_SetVertexDeclaration) { __imp__D3DDevice_SetVertexDeclaration(ctx, base); }
// D3DDevice_SetTexture(device, Sampler, pTexture)
REX_HOOK_RAW(D3DDevice_SetTexture) {
  const uint32_t device = ctx.r3.u32, sampler = ctx.r4.u32, texture = ctx.r5.u32;
  __imp__D3DDevice_SetTexture(ctx, base);
  pgr4::gpu::Renderer::OnSetTexture(device, sampler, texture);
}
REX_HOOK_RAW(D3DDevice_SetStreamSource) { __imp__D3DDevice_SetStreamSource(ctx, base); }
REX_HOOK_RAW(D3DDevice_SetIndices) { __imp__D3DDevice_SetIndices(ctx, base); }
