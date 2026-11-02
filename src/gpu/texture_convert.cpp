#include "gpu/texture_convert.h"

#include <algorithm>
#include <bit>
#include <cstring>

namespace pgr4::gpu {
namespace {

uint32_t AlignUp(uint32_t v, uint32_t a) { return (v + a - 1) / a * a; }

uint32_t Log2Ceil(uint32_t v) { return v <= 1 ? 0 : 32 - std::countl_zero(v - 1); }

// Xbox 360 tiled surface addressing (XGAddress2DTiledOffset /
// XGAddress3DTiledOffset); pitch and height in blocks.
uint32_t TiledOffset2D(uint32_t x, uint32_t y, uint32_t pitch, uint32_t log2_bpb) {
  pitch = AlignUp(pitch, 32);
  const uint32_t macro = ((x >> 5) + (y >> 5) * (pitch >> 5)) << (log2_bpb + 7);
  const uint32_t micro = ((x & 7) + ((y & 0xE) << 2)) << log2_bpb;
  const uint32_t offset = macro + ((micro & ~0xFu) << 1) + (micro & 0xF) + ((y & 1) << 4);
  return ((offset & ~0x1FFu) << 3) + ((y & 16) << 7) + ((offset & 0x1C0) << 2) +
         (((((y & 8) >> 2) + (x >> 3)) & 3) << 6) + (offset & 0x3F);
}

uint32_t TiledOffset3D(uint32_t x, uint32_t y, uint32_t z, uint32_t pitch, uint32_t height,
                       uint32_t log2_bpb) {
  pitch = AlignUp(pitch, 32);
  height = AlignUp(height, 32);
  const uint32_t macro_outer = ((y >> 4) + (z >> 2) * (height >> 4)) * (pitch >> 5);
  const uint32_t macro = ((((x >> 5) + macro_outer) << (log2_bpb + 6)) & 0xFFFFFFF) << 1;
  const uint32_t micro = (((x & 7) + ((y & 6) << 2)) << (log2_bpb + 6)) >> 6;
  const uint32_t offset_outer = ((y >> 3) + (z >> 2)) & 1;
  const uint32_t offset1 = offset_outer + ((((x >> 3) + (offset_outer << 1)) & 3) << 1);
  const uint32_t offset2 =
      ((macro + (micro & ~15u)) << 1) + (micro & 15) + ((z & 3) << (log2_bpb + 6)) + ((y & 1) << 4);
  uint32_t address = (offset1 & 1) << 3;
  address += (offset2 >> 6) & 7;
  address <<= 3;
  address += offset1 & ~1u;
  address <<= 2;
  address += offset2 & ~511u;
  address <<= 3;
  address += offset2 & 63;
  return address;
}

void SwapBlock(uint8_t *dst, const uint8_t *src, uint32_t size, uint32_t endian) {
  switch (endian) {
  case 1:
    for (uint32_t i = 0; i + 1 < size; i += 2) {
      dst[i] = src[i + 1];
      dst[i + 1] = src[i];
    }
    break;
  case 2:
    for (uint32_t i = 0; i + 3 < size; i += 4) {
      dst[i] = src[i + 3];
      dst[i + 1] = src[i + 2];
      dst[i + 2] = src[i + 1];
      dst[i + 3] = src[i];
    }
    break;
  case 3:
    for (uint32_t i = 0; i + 3 < size; i += 4) {
      dst[i] = src[i + 2];
      dst[i + 1] = src[i + 3];
      dst[i + 2] = src[i];
      dst[i + 3] = src[i + 1];
    }
    break;
  default:
    std::memcpy(dst, src, size);
    break;
  }
}

uint8_t Expand(uint32_t v, uint32_t bits) { return uint8_t(v * 255 / ((1u << bits) - 1)); }

// One guest block (already endian-swapped) to host texels.
void ConvertBlock(TexelConversion c, const uint8_t *in, uint8_t *out, uint32_t out_row_pitch) {
  switch (c) {
  case TexelConversion::k565: {
    const uint16_t v = uint16_t(in[0] | (in[1] << 8));
    out[0] = Expand(v & 0x1F, 5);
    out[1] = Expand((v >> 5) & 0x3F, 6);
    out[2] = Expand(v >> 11, 5);
    out[3] = 255;
    break;
  }
  case TexelConversion::k655: {
    const uint16_t v = uint16_t(in[0] | (in[1] << 8));
    out[0] = Expand(v & 0x1F, 5);
    out[1] = Expand((v >> 5) & 0x1F, 5);
    out[2] = Expand(v >> 10, 6);
    out[3] = 255;
    break;
  }
  case TexelConversion::k1555: {
    const uint16_t v = uint16_t(in[0] | (in[1] << 8));
    out[0] = Expand(v & 0x1F, 5);
    out[1] = Expand((v >> 5) & 0x1F, 5);
    out[2] = Expand((v >> 10) & 0x1F, 5);
    out[3] = (v >> 15) ? 255 : 0;
    break;
  }
  case TexelConversion::k4444: {
    const uint16_t v = uint16_t(in[0] | (in[1] << 8));
    out[0] = Expand(v & 0xF, 4);
    out[1] = Expand((v >> 4) & 0xF, 4);
    out[2] = Expand((v >> 8) & 0xF, 4);
    out[3] = Expand(v >> 12, 4);
    break;
  }
  case TexelConversion::k2101010: {
    uint32_t v;
    std::memcpy(&v, in, 4);
    const uint16_t c16[4] = {uint16_t((v & 0x3FF) * 65535 / 1023),
                             uint16_t(((v >> 10) & 0x3FF) * 65535 / 1023),
                             uint16_t(((v >> 20) & 0x3FF) * 65535 / 1023),
                             uint16_t((v >> 30) * 65535 / 3)};
    std::memcpy(out, c16, 8);
    break;
  }
  case TexelConversion::kDXT3A: {
    // 16 x 4-bit alpha, row-major, low nibble first.
    for (uint32_t y = 0; y < 4; ++y) {
      for (uint32_t x = 0; x < 4; ++x) {
        const uint32_t i = y * 4 + x;
        const uint32_t nibble = (in[i >> 1] >> ((i & 1) * 4)) & 0xF;
        out[y * out_row_pitch + x] = uint8_t(nibble * 17);
      }
    }
    break;
  }
  case TexelConversion::kCTX1: {
    // Two 8:8 endpoints, then 2-bit indices; colors 2/3 are 2:1 and 1:2 mixes.
    uint8_t palette[4][2];
    palette[0][0] = in[0];
    palette[0][1] = in[1];
    palette[1][0] = in[2];
    palette[1][1] = in[3];
    for (int ch = 0; ch < 2; ++ch) {
      palette[2][ch] = uint8_t((2 * palette[0][ch] + palette[1][ch]) / 3);
      palette[3][ch] = uint8_t((palette[0][ch] + 2 * palette[1][ch]) / 3);
    }
    uint32_t bits;
    std::memcpy(&bits, in + 4, 4);
    for (uint32_t y = 0; y < 4; ++y) {
      for (uint32_t x = 0; x < 4; ++x) {
        const uint32_t idx = (bits >> ((y * 4 + x) * 2)) & 3;
        out[y * out_row_pitch + x * 2] = palette[idx][0];
        out[y * out_row_pitch + x * 2 + 1] = palette[idx][1];
      }
    }
    break;
  }
  default:
    break;
  }
}

} // namespace

bool PackedMipOffset(uint32_t width, uint32_t height, uint32_t mip, uint32_t block, uint32_t &x_blocks,
                     uint32_t &y_blocks) {
  x_blocks = y_blocks = 0;
  const uint32_t log2_width = Log2Ceil(width), log2_height = Log2Ceil(height);
  const uint32_t log2_size = std::min(log2_width, log2_height);
  if (log2_size > 4 + mip) {
    return false;
  }
  const uint32_t packed_base = log2_size > 4 ? log2_size - 4 : 0;
  const uint32_t packed_mip = mip - packed_base;
  uint32_t x = 0, y = 0;
  if (packed_mip < 3) {
    // The first three tail levels sit 16, 8 and 4 texels along the shorter axis.
    if (log2_width > log2_height) {
      y = 16u >> packed_mip;
    } else {
      x = 16u >> packed_mip;
    }
  } else {
    // Smaller ones run along the longer axis.
    const uint32_t offset = log2_width > log2_height ? (1u << (log2_width - packed_base)) >> (packed_mip - 2)
                                                     : (1u << (log2_height - packed_base)) >> (packed_mip - 2);
    if (log2_width > log2_height) {
      x = offset;
    } else {
      y = offset;
    }
  }
  x_blocks = x / block;
  y_blocks = y / block;
  return true;
}

void SwapEndian(uint8_t *dst, const uint8_t *src, uint32_t size, uint32_t endian) {
  SwapBlock(dst, src, size, endian);
}

TextureLayout DescribeTexture(uint32_t format, bool is_signed, uint32_t dimension, uint32_t width,
                              uint32_t height, uint32_t depth, uint32_t pitch_texels, bool tiled,
                              bool packed_mips) {
  using F = plume::RenderFormat;
  TextureLayout l;
  l.supported = true;
  uint32_t host_texel_bytes = 0; // for conversions that change the size
  switch (format) {
  case 2: // k_8
  case 8:
  case 9:
    l.host_format = is_signed ? F::R8_SNORM : F::R8_UNORM;
    l.guest_block_bytes = 1;
    break;
  case 3: // k_1_5_5_5
    l.conversion = TexelConversion::k1555;
    l.host_format = F::R8G8B8A8_UNORM;
    l.guest_block_bytes = 2;
    host_texel_bytes = 4;
    break;
  case 4: // k_5_6_5
    l.conversion = TexelConversion::k565;
    l.host_format = F::R8G8B8A8_UNORM;
    l.guest_block_bytes = 2;
    host_texel_bytes = 4;
    break;
  case 5: // k_6_5_5
    l.conversion = TexelConversion::k655;
    l.host_format = F::R8G8B8A8_UNORM;
    l.guest_block_bytes = 2;
    host_texel_bytes = 4;
    break;
  case 6: // k_8_8_8_8
  case 14:
  case 50:
    l.host_format = is_signed ? F::R8G8B8A8_SNORM : F::R8G8B8A8_UNORM;
    l.guest_block_bytes = 4;
    break;
  case 7: // k_2_10_10_10
  case 54:
    l.conversion = TexelConversion::k2101010;
    l.host_format = F::R16G16B16A16_UNORM;
    l.guest_block_bytes = 4;
    host_texel_bytes = 8;
    break;
  case 10: // k_8_8
    l.host_format = is_signed ? F::R8G8_SNORM : F::R8G8_UNORM;
    l.guest_block_bytes = 2;
    break;
  case 15: // k_4_4_4_4
    l.conversion = TexelConversion::k4444;
    l.host_format = F::R8G8B8A8_UNORM;
    l.guest_block_bytes = 2;
    host_texel_bytes = 4;
    break;
  case 18: // k_DXT1
  case 51:
    l.host_format = F::BC1_UNORM;
    l.block = 4;
    l.guest_block_bytes = 8;
    break;
  case 19: // k_DXT2_3
  case 52:
    l.host_format = F::BC2_UNORM;
    l.block = 4;
    l.guest_block_bytes = 16;
    break;
  case 20: // k_DXT4_5
  case 53:
    l.host_format = F::BC3_UNORM;
    l.block = 4;
    l.guest_block_bytes = 16;
    break;
  case 24: // k_16
  case 27:
    l.host_format = is_signed ? F::R16_SNORM : F::R16_UNORM;
    l.guest_block_bytes = 2;
    break;
  case 25: // k_16_16
  case 28:
    l.host_format = is_signed ? F::R16G16_SNORM : F::R16G16_UNORM;
    l.guest_block_bytes = 4;
    break;
  case 26: // k_16_16_16_16
  case 29:
    l.host_format = is_signed ? F::R16G16B16A16_SNORM : F::R16G16B16A16_UNORM;
    l.guest_block_bytes = 8;
    break;
  case 30:
    l.host_format = F::R16_FLOAT;
    l.guest_block_bytes = 2;
    break;
  case 31:
    l.host_format = F::R16G16_FLOAT;
    l.guest_block_bytes = 4;
    break;
  case 32:
    l.host_format = F::R16G16B16A16_FLOAT;
    l.guest_block_bytes = 8;
    break;
  case 33:
    l.host_format = F::R32_UINT;
    l.guest_block_bytes = 4;
    break;
  case 36:
    l.host_format = F::R32_FLOAT;
    l.guest_block_bytes = 4;
    break;
  case 37:
    l.host_format = F::R32G32_FLOAT;
    l.guest_block_bytes = 8;
    break;
  case 38:
    l.host_format = F::R32G32B32A32_FLOAT;
    l.guest_block_bytes = 16;
    break;
  case 49: // k_DXN
    l.host_format = F::BC5_UNORM;
    l.block = 4;
    l.guest_block_bytes = 16;
    break;
  case 58: // k_DXT3A
  case 61:
    l.conversion = TexelConversion::kDXT3A;
    l.host_format = F::R8_UNORM;
    l.block = 4;
    l.guest_block_bytes = 8;
    host_texel_bytes = 1;
    break;
  case 59: // k_DXT5A
    l.host_format = F::BC4_UNORM;
    l.block = 4;
    l.guest_block_bytes = 8;
    break;
  case 60: // k_CTX1
    l.conversion = TexelConversion::kCTX1;
    l.host_format = F::R8G8_UNORM;
    l.block = 4;
    l.guest_block_bytes = 8;
    host_texel_bytes = 2;
    break;
  default:
    l.supported = false;
    return l;
  }

  l.width = width;
  l.height = height;
  l.volume = dimension == 2;
  l.depth = l.volume ? depth : 1;
  l.tiled = tiled;
  l.blocks_x = (width + l.block - 1) / l.block;
  l.blocks_y = (height + l.block - 1) / l.block;

  // Shortest side <= 16 with packed mips: the base level is in the mip tail
  // tile, 16 texels along the shorter axis (GetPackedMipOffset, mip 0). Linear
  // textures too: the car damage lattice (8x6 k_8_8_8_8) starts 16 texels in.
  if (packed_mips && std::min(Log2Ceil(width), Log2Ceil(height)) <= 4) {
    if (Log2Ceil(width) > Log2Ceil(height)) {
      l.offset_y = 16 / l.block;
    } else {
      l.offset_x = 16 / l.block;
    }
  }

  const uint32_t log2_bpb = uint32_t(std::countr_zero(l.guest_block_bytes));
  (void)log2_bpb;
  uint32_t pitch_blocks = pitch_texels ? (pitch_texels + l.block - 1) / l.block : l.blocks_x;
  pitch_blocks = std::max(pitch_blocks, l.blocks_x + l.offset_x);
  if (tiled) {
    l.pitch_blocks = AlignUp(pitch_blocks, 32);
    l.height_blocks = AlignUp(l.blocks_y + l.offset_y, 32);
    const uint32_t slices = l.volume ? AlignUp(l.depth, 4) : 1;
    l.face_bytes = AlignUp(l.pitch_blocks * l.height_blocks * slices * l.guest_block_bytes, 4096);
  } else {
    l.pitch_blocks = AlignUp(pitch_blocks * l.guest_block_bytes, 256) / l.guest_block_bytes;
    l.height_blocks = l.blocks_y + l.offset_y;
    l.face_bytes = AlignUp(l.pitch_blocks * l.guest_block_bytes * l.height_blocks * l.depth, 4096);
  }
  const uint32_t faces = dimension == 3 ? 6 : 1;
  l.guest_size = l.face_bytes * faces;

  l.host_block_bytes = host_texel_bytes && l.conversion != TexelConversion::kDXT3A &&
                               l.conversion != TexelConversion::kCTX1
                           ? host_texel_bytes
                           : l.guest_block_bytes;
  // Decoded block formats write texels, not blocks.
  if (l.conversion == TexelConversion::kDXT3A || l.conversion == TexelConversion::kCTX1) {
    l.host_block_bytes = host_texel_bytes;
    l.host_width = l.blocks_x * 4;
    l.host_height = l.blocks_y * 4;
    l.host_rows = l.host_height;
    l.host_row_pitch = AlignUp(l.host_width * host_texel_bytes, 256);
    l.block = 1; // the host format is uncompressed
    return l;
  }
  l.host_width = l.blocks_x * l.block;
  l.host_height = l.blocks_y * l.block;
  l.host_rows = l.blocks_y;
  l.host_row_pitch = AlignUp(l.blocks_x * l.host_block_bytes, 256);
  return l;
}

void ConvertTexture(const TextureLayout &l, const uint8_t *src, uint32_t endian, uint8_t *dst,
                    uint32_t layers) {
  const bool decoded = l.conversion == TexelConversion::kDXT3A || l.conversion == TexelConversion::kCTX1;
  const uint32_t log2_bpb = uint32_t(std::countr_zero(l.guest_block_bytes));
  const uint32_t slice_bytes = l.host_row_pitch * l.host_rows;
  const uint32_t blocks_x = decoded ? l.host_width / 4 : l.blocks_x;
  const uint32_t blocks_y = decoded ? l.host_height / 4 : l.blocks_y;
  uint8_t block[16];
  for (uint32_t layer = 0; layer < layers; ++layer) {
    const uint8_t *face = src + size_t(layer) * l.face_bytes;
    for (uint32_t z = 0; z < l.depth; ++z) {
      uint8_t *slice = dst + size_t(layer * l.depth + z) * slice_bytes;
      for (uint32_t by = 0; by < blocks_y; ++by) {
        for (uint32_t bx = 0; bx < blocks_x; ++bx) {
          const uint32_t gx = bx + l.offset_x, gy = by + l.offset_y;
          uint32_t offset;
          if (l.tiled) {
            offset = l.volume ? TiledOffset3D(gx, gy, z, l.pitch_blocks, l.height_blocks, log2_bpb)
                              : TiledOffset2D(gx, gy, l.pitch_blocks, log2_bpb);
          } else {
            offset = ((z * l.height_blocks + gy) * l.pitch_blocks + gx) * l.guest_block_bytes;
          }
          if (offset + l.guest_block_bytes > l.face_bytes) {
            continue;
          }
          SwapBlock(block, face + offset, l.guest_block_bytes, endian);
          if (decoded) {
            ConvertBlock(l.conversion, block,
                         slice + size_t(by * 4) * l.host_row_pitch + bx * 4 * l.host_block_bytes,
                         l.host_row_pitch);
          } else {
            uint8_t *out = slice + size_t(by) * l.host_row_pitch + bx * l.host_block_bytes;
            if (l.conversion == TexelConversion::kCopy) {
              std::memcpy(out, block, l.guest_block_bytes);
            } else {
              ConvertBlock(l.conversion, block, out, 0);
            }
          }
        }
      }
    }
  }
}

} // namespace pgr4::gpu
