// PGR4-Recomp native renderer: Xenos texture memory to host texel data.
// Mip levels are converted one at a time (DescribeTexture per level).
#pragma once

#include <cstdint>

#include <plume_render_interface_types.h>

namespace pgr4::gpu {

enum class TexelConversion {
  kCopy,        // endian swap only
  k565,         // -> R8G8B8A8
  k655,         // k_6_5_5 -> R8G8B8A8
  k1555,        // -> R8G8B8A8
  k4444,        // -> R8G8B8A8
  k2101010,     // -> R16G16B16A16_UNORM
  kDXT3A,       // 4-bit explicit alpha blocks -> R8
  kCTX1,        // two-channel blocks -> R8G8
};

struct TextureLayout {
  bool supported = false;
  plume::RenderFormat host_format = plume::RenderFormat::UNKNOWN;
  TexelConversion conversion = TexelConversion::kCopy;
  uint32_t block = 1;            // texel block width/height (4 for compressed)
  uint32_t guest_block_bytes = 0;
  uint32_t host_block_bytes = 0;
  uint32_t width = 0, height = 0, depth = 1; // texels
  uint32_t blocks_x = 0, blocks_y = 0;
  bool tiled = false;
  bool volume = false;
  uint32_t pitch_blocks = 0;     // guest row pitch in blocks (tiled: aligned to 32)
  uint32_t height_blocks = 0;    // guest rows of blocks per slice (tiled: aligned to 32)
  uint32_t offset_x = 0, offset_y = 0; // packed-mip position of the base level, blocks
  uint32_t face_bytes = 0;       // guest bytes per cube face / stack layer
  uint32_t guest_size = 0;       // guest bytes covering every face
  uint32_t host_width = 0, host_height = 0; // aligned to the block size
  uint32_t host_row_pitch = 0;   // bytes, 256-aligned
  uint32_t host_rows = 0;        // rows of blocks per slice
};

// dimension: 0 1D, 1 2D, 2 3D, 3 cube.
TextureLayout DescribeTexture(uint32_t format, bool is_signed, uint32_t dimension, uint32_t width,
                              uint32_t height, uint32_t depth, uint32_t pitch_texels, bool tiled,
                              bool packed_mips);

// Position in the packed mip tail tile of `mip` (in blocks), for a texture
// whose base level is width x height texels. False when the level is not in
// the tail (its shortest side is above 16 texels). GetPackedMipOffset.
bool PackedMipOffset(uint32_t width, uint32_t height, uint32_t mip, uint32_t block, uint32_t &x_blocks,
                     uint32_t &y_blocks);

// Writes `layers` images (each host_row_pitch * host_rows * depth bytes) from
// guest memory at src.
void ConvertTexture(const TextureLayout &layout, const uint8_t *src, uint32_t endian, uint8_t *dst,
                    uint32_t layers);

// Xenos endian modes: 0 none, 1 8-in-16, 2 8-in-32, 3 16-in-32.
void SwapEndian(uint8_t *dst, const uint8_t *src, uint32_t size, uint32_t endian);

} // namespace pgr4::gpu
