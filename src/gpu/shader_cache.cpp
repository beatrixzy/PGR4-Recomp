#include "gpu/shader_cache.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <unordered_map>

#define XXH_INLINE_ALL
#include <xxhash.h>

#include <windows.h>

#include <plume_d3d12.h>

#include <rex/cvar.h>
#include <rex/logging.h>

REXCVAR_DEFINE_STRING(shader_format, "auto", "PGR4",
                      "Shader cache format: auto (DXIL on shader model 6 GPUs), dxbc or dxil");

namespace pgr4::gpu {
namespace {

constexpr char kMagicDxbc[8] = {'P', 'G', 'R', '4', 'S', 'C', '5', '1'};
constexpr char kMagicDxil[8] = {'P', 'G', 'R', '4', 'S', 'C', '6', '0'};
constexpr uint32_t kVersion = 1;

#pragma pack(push, 1)
struct PackHeader {
  char magic[8];
  uint32_t version;
  uint32_t count;
};
struct PackEntry {
  uint64_t key;
  uint32_t blob_offset;
  uint32_t blob_size;
  uint32_t stage;
  uint32_t input_offset;
  uint32_t input_count;
  uint32_t sampler_mask;
};
#pragma pack(pop)
static_assert(sizeof(PackEntry) == 32);

struct CacheState {
  std::vector<uint8_t> data;
  std::unordered_map<uint64_t, CachedShader> shaders;
  bool loaded = false;
  bool dxil = false;
};

CacheState &cache() {
  static CacheState c;
  return c;
}

std::filesystem::path ExecutableDirectory() {
  wchar_t path[MAX_PATH] = {};
  GetModuleFileNameW(nullptr, path, MAX_PATH);
  return std::filesystem::path(path).parent_path();
}

} // namespace

bool ShaderCache::Load(plume::RenderDevice *device) {
  auto &c = cache();
  if (c.loaded) {
    return true;
  }
  const std::string format = REXCVAR_GET(shader_format);
  const bool sm6 = device && static_cast<plume::D3D12Device *>(device)->shaderModel >= D3D_SHADER_MODEL_6_0;
  c.dxil = format == "dxil" || (format != "dxbc" && sm6);
  const auto path = ExecutableDirectory() / (c.dxil ? "shader_cache_dxil.bin" : "shader_cache_sm51.bin");
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    REXGPU_ERROR("PGR4 D3D12: shader cache {} not found", path.string());
    return false;
  }
  c.data.resize(size_t(file.tellg()));
  file.seekg(0);
  file.read(reinterpret_cast<char *>(c.data.data()), std::streamsize(c.data.size()));

  PackHeader header;
  if (c.data.size() < sizeof(header)) {
    REXGPU_ERROR("PGR4 D3D12: shader cache is truncated");
    return false;
  }
  std::memcpy(&header, c.data.data(), sizeof(header));
  if (std::memcmp(header.magic, c.dxil ? kMagicDxil : kMagicDxbc, sizeof(header.magic)) ||
      header.version != kVersion ||
      sizeof(header) + size_t(header.count) * sizeof(PackEntry) > c.data.size()) {
    REXGPU_ERROR("PGR4 D3D12: shader cache header is invalid");
    return false;
  }
  for (uint32_t i = 0; i < header.count; ++i) {
    PackEntry e;
    std::memcpy(&e, c.data.data() + sizeof(header) + size_t(i) * sizeof(PackEntry), sizeof(e));
    if (size_t(e.blob_offset) + e.blob_size > c.data.size() ||
        size_t(e.input_offset) + size_t(e.input_count) * 4 > c.data.size()) {
      continue;
    }
    CachedShader shader;
    shader.dxbc = c.data.data() + e.blob_offset;
    shader.dxbc_size = e.blob_size;
    // stage bits 0-7: 1 PS, VS flags; bits 8-16: constant registers + 1.
    const uint32_t regs = (e.stage >> 8) & 0x1FF;
    e.stage &= 0xFF;
    shader.constant_regs = regs ? std::min(regs - 1, 256u) : 256u;
    shader.pixel = e.stage == 1;
    shader.reads_vertex_index = !shader.pixel && (e.stage & 2) != 0;
    shader.indexed_fetch = !shader.pixel && (e.stage & 4) != 0;
    shader.screen_2d = !shader.pixel && (e.stage & 8) != 0;
    shader.sampler_mask = e.sampler_mask;
    for (uint32_t j = 0; j < e.input_count; ++j) {
      const uint8_t *in = c.data.data() + e.input_offset + j * 4;
      shader.inputs.push_back({in[0], in[1], in[2]});
    }
    c.shaders.emplace(e.key, std::move(shader));
  }
  c.loaded = true;
  REXGPU_INFO("PGR4 D3D12: shader cache loaded, {} shaders ({}, {} bytes)", c.shaders.size(),
              c.dxil ? "DXIL" : "DXBC", c.data.size());
  return true;
}

bool ShaderCache::IsDxil() { return cache().dxil; }

const CachedShader *ShaderCache::Find(uint64_t key) {
  auto &c = cache();
  auto it = c.shaders.find(key);
  return it != c.shaders.end() ? &it->second : nullptr;
}

uint64_t ShaderCache::HashContainer(const uint8_t *container, uint32_t size) {
  return XXH64(container, size, 0);
}

} // namespace pgr4::gpu
