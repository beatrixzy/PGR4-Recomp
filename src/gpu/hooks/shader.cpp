// PGR4-Recomp native renderer: shader and vertex declaration creation.
//
// The originals run unchanged (PGR4's D3D keeps using its own shader and
// declaration objects); the shader hooks then key the guest object to the
// cached host shader built from the same microcode container.
#include <rex/hook.h>
#include <rex/memory/utils.h>
#include <rex/runtime.h>
#include <rex/system/kernel_state.h>
#include <rex/types.h>

#include "gpu/renderer.h"

REX_EXTERN(__imp__D3DDevice_CreateVertexShader);
REX_EXTERN(__imp__D3DDevice_CreatePixelShader);
REX_EXTERN(__imp__D3DDevice_CreateVertexDeclaration);

// D3DDevice_CreateVertexShader(const DWORD *pFunction) -> D3DVertexShader*
REX_HOOK_RAW(D3DDevice_CreateVertexShader) {
  const uint32_t container = ctx.r3.u32;
  __imp__D3DDevice_CreateVertexShader(ctx, base);
  pgr4::gpu::Renderer::OnShaderCreated(ctx.r3.u32, container, false);
}

// D3DDevice_CreatePixelShader(const DWORD *pFunction) -> D3DPixelShader*
REX_HOOK_RAW(D3DDevice_CreatePixelShader) {
  const uint32_t container = ctx.r3.u32;
  __imp__D3DDevice_CreatePixelShader(ctx, base);
  pgr4::gpu::Renderer::OnShaderCreated(ctx.r3.u32, container, true);
}

// Most of PGR4's shaders come from its effect loader, which builds the objects
// in place instead of calling Create*Shader: sub_822B34A0 / sub_822B35F0
// parse the container (info[0] is the container), then
// XGSet{Vertex,Pixel}ShaderHeader(shader, size, info) copy its header into
// the object. The container is still whole at that point.
namespace {
uint32_t InfoContainer(uint32_t info) {
  return rex::memory::load_and_swap<uint32_t>(REX_KERNEL_MEMORY()->TranslateVirtual(info));
}
} // namespace

REX_EXTERN(__imp__XGSetVertexShaderHeader);
REX_EXTERN(__imp__XGSetPixelShaderHeader);

REX_HOOK_RAW(XGSetVertexShaderHeader) {
  const uint32_t shader = ctx.r3.u32, info = ctx.r5.u32;
  __imp__XGSetVertexShaderHeader(ctx, base);
  if (info) {
    pgr4::gpu::Renderer::OnShaderCreated(shader, InfoContainer(info), false);
  }
}

REX_HOOK_RAW(XGSetPixelShaderHeader) {
  const uint32_t shader = ctx.r3.u32, info = ctx.r5.u32;
  __imp__XGSetPixelShaderHeader(ctx, base);
  if (info) {
    pgr4::gpu::Renderer::OnShaderCreated(shader, InfoContainer(info), true);
  }
}

// D3DDevice_CreateVertexDeclaration(const D3DVERTEXELEMENT9 *, D3DVertexDeclaration *).
// The declaration's element copy (+0x34) is read at draw time, so the
// original is all that runs.
REX_HOOK_RAW(D3DDevice_CreateVertexDeclaration) {
  __imp__D3DDevice_CreateVertexDeclaration(ctx, base);
}
