// PGR4-Recomp native renderer: guest draws to Plume (D3D12).
//
// PGR4's D3D runs as a command-buffer device (gpu/hooks/device.cpp), so its
// setters keep the device's state shadows current: shader constants, fetch
// constants, register shadows, bound shaders, declaration, streams, index
// buffer and render targets. The draw hooks read those shadows at draw time
// and record the equivalent host draw. Guest shaders come from the shader
// model 5.1 cache built from the XenosRecomp HLSL (gpu/shader_cache.h).
#pragma once

#include <cstdint>

namespace pgr4::gpu {

class Renderer {
public:
  // D3DDevice_CreateVertexShader / CreatePixelShader: container is the
  // function the game passed in, shader_object the guest object returned.
  static void OnShaderCreated(uint32_t shader_object, uint32_t container, bool pixel);
  // D3DDevice_SetTexture (debug logging during dumps).
  static void OnSetTexture(uint32_t device, uint32_t sampler, uint32_t texture);
  // D3D's Z pass begins or ends (the device runs without it; logged in dumps).
  static void OnZPass(uint32_t device, bool begin);
  // D3DDevice_SetViewport(device, const D3DVIEWPORT9 *).
  static void OnViewport(uint32_t device, uint32_t viewport);

  static void DrawVertices(uint32_t device, uint32_t primitive, uint32_t start_vertex,
                           uint32_t vertex_count);
  static void DrawIndexed(uint32_t device, uint32_t primitive, int32_t base_vertex,
                          uint32_t start_index, uint32_t index_count);
  // Stream 0 data in guest memory (DrawVerticesUP, Begin/EndVertices).
  static void DrawUserPointer(uint32_t device, uint32_t primitive, uint32_t vertex_count,
                              uint32_t data, uint32_t stride);

  // D3DDevice_Clear flags: 0x1 targets, 0x10 depth, 0x20 stencil. Color is
  // D3DCOLOR (ARGB).
  static void Clear(uint32_t device, uint32_t flags, uint32_t argb, float z, uint32_t stencil);
  // BeginTiling: the tile rects' union sizes the targets until EndTiling;
  // clears color (if clear_color is non-null), depth and stencil.
  static void BeginTiling(uint32_t device, uint32_t rect_count, uint32_t rects, uint32_t clear_color,
                          float z, uint32_t stencil);
  // EndTiling: resolves into dest_texture and closes the tiled pass.
  static void EndTiling(uint32_t device, uint32_t flags, uint32_t dest_texture, uint32_t clear_color,
                        float clear_z);
  // D3DDevice_Resolve / EndTiling: copies a render target into the host copy
  // of dest_texture's memory.
  // dest_level is the mip level and dest_slice the cube face (or array
  // slice) of dest_texture.
  static void Resolve(uint32_t device, uint32_t flags, uint32_t source_rect, uint32_t dest_texture,
                      uint32_t dest_point, uint32_t dest_level, uint32_t dest_slice, uint32_t clear_color,
                      float clear_z = 1.0f);
  // D3DDevice_Swap: presents the host copy of the front buffer texture.
  static void Swap(uint32_t device, uint32_t front_buffer);

  // Title switch (Geometry Wars, launched from PGR4's arcade). SetTitle says
  // whether PGR4 runs, for the guest writes only PGR4 wants;
  // ResetForTitleSwitch drops everything tied to the old title's guest memory.
  static void SetTitle(bool pgr4);
  static void ResetForTitleSwitch();
};

// Debug: logs every thread's host call stack (module+offset).
void DumpThreadStacks(const char *reason);

} // namespace pgr4::gpu
