// PGR4-Recomp native renderer: the shader cache built by
// tools/shaders/build_shader_cache.py. Two builds of the same HLSL and binding
// layout: shader_cache_sm51.bin (fxc, DXBC SM 5.1) for drivers without SM 6,
// and shader_cache_dxil.bin (dxc, DXIL SM 6.0) for everything newer.
#pragma once

#include <cstdint>
#include <vector>

namespace plume {
struct RenderDevice;
}

namespace pgr4::gpu {

struct CachedShaderInput {
  uint8_t usage;       // D3DDECLUSAGE
  uint8_t usage_index;
  uint8_t kind;        // 0 float4, 1 uint4 float bits, 2 uint4 integers
};

struct CachedShader {
  const uint8_t *dxbc = nullptr;  // DXBC or DXIL, see ShaderCache::IsDxil()
  uint32_t dxbc_size = 0;
  bool pixel = false;
  // VS reads the vertex index Xenos preloads into r0.x (stage bit 2).
  bool reads_vertex_index = false;
  // VS fetches vertices by a computed index (stage bit 4): the streams are
  // bound as raw buffers and described in SharedConstants::fetch_layouts.
  bool indexed_fetch = false;
  // VS places vertices in screen space with PosScale at c45 (stage bit 8):
  // PGR4's 2D UI, which ultrawide keeps at 16:9 shape.
  bool screen_2d = false;
  // Sampler registers the shader declares (bit N = register N).
  uint32_t sampler_mask = 0;
  // Float constant registers the shader reads (c0 up to this, at most 256).
  uint32_t constant_regs = 256;
  std::vector<CachedShaderInput> inputs;
};

class ShaderCache {
public:
  // Loads the cache that suits the device from the executable's directory:
  // DXIL when the device supports shader model 6.0, else DXBC. The
  // --shader_format=dxbc|dxil cvar overrides the choice.
  static bool Load(plume::RenderDevice *device);
  // Whether the loaded cache holds DXIL (else DXBC).
  static bool IsDxil();
  // key: XXH64 over the guest container (virtualSize + physicalSize bytes).
  static const CachedShader *Find(uint64_t key);
  // Same hash the cache builder uses.
  static uint64_t HashContainer(const uint8_t *container, uint32_t size);
};

} // namespace pgr4::gpu
