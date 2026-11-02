#include "gpu/renderer.h"

#include <algorithm>
#include <array>
#include <immintrin.h>
#include <cfloat>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cstring>
#include <limits>
#include <filesystem>
#include <memory>
#include <atomic>
#include <thread>
#include <deque>
#include <condition_variable>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#define XXH_INLINE_ALL
#include <xxhash.h>

#include <plume_d3d12.h>
#include <psapi.h>
#include <timeapi.h>
#include <tlhelp32.h>
#pragma comment(lib, "winmm.lib")
#include <plume_render_interface.h>
#include <plume_render_interface_builders.h>

#include <rex/cvar.h>
#include <rex/logging.h>
#include <rex/runtime.h>
#include <rex/system/kernel_state.h>

#include "gpu/shader_cache.h"
#include "gpu/texture_convert.h"
#include "gpu/video.h"
#include "solid_ps.dxbc.h"

REXCVAR_DEFINE_BOOL(renderer_profile, false, "PGR4",
                    "Log per-stage renderer timings every 600 frames (costs some CPU per draw)");
REXCVAR_DEFINE_INT32(dump_after, 0, "PGR4", "Debug: dump one frame this many seconds after startup (0: off)");
REXCVAR_DEFINE_INT32(dump_every, 100, "PGR4", "Debug: during a dump, capture the main scene every this many draws");
REXCVAR_DEFINE_INT32(dump_from, 0, "PGR4", "Debug: during a dump, first draw of the main scene to capture");
REXCVAR_DEFINE_STRING(tweaks, "", "PGR4",
                      "Set game tweakables at startup: Name=value;Name=value (bools 0/1)");
REXCVAR_DEFINE_INT32(gpu_safe, 0, "PGR4",
                     "Debug bisect of GPU faults, bits: 1 no game textures, 2 no render-to-texture copies, "
                     "4 skip indexed vertex fetch draws, 8 skip vertex-index draws, 16 no mip levels, "
                     "32 no cube/volume textures, 64 indexed fetch reads a null stream, "
                     "128 indexed fetch reads nothing, 256 viewport from SetViewport's struct");
REXCVAR_DEFINE_BOOL(debug_solid_ps, false, "PGR4",
                    "Debug: draw every guest draw with a solid magenta pixel shader");
REXCVAR_DEFINE_BOOL(debug_no_blend, false, "PGR4", "Debug: draw every guest draw with blending off");
REXCVAR_DEFINE_BOOL(quad_index_buffer, true, "PGR4",
                    "Non-indexed quad lists draw from one fixed index buffer (false: indices uploaded per draw)");
REXCVAR_DEFINE_BOOL(poison_targets, false, "PGR4",
                    "Debug: fill new render targets with NaN instead of zero");
REXCVAR_DEFINE_DOUBLE(aspect_ratio, 0.0, "PGR4",
                      "Display aspect ratio for ultrawide (e.g. 2.3333 for 21:9, 3.5556 for 32:9) or narrower "
                      "screens (1.6 for 16:10, the Steam Deck); 0 keeps the game's 16:9 unless the window is "
                      "narrower (then it follows the window). Wider: the camera sees "
                      "wider and scene targets get the extra width. Narrower: the frame is stretched to fill");
REXCVAR_DEFINE_BOOL(edram_clear_sharing, true, "PGR4",
                    "Apply depth clears to every target at the same EDRAM address (the rear-view mirror needs it)");
REXCVAR_DEFINE_BOOL(buffer_recheck, true, "PGR4",
                    "Check big vertex/index buffers for changes every frame (sampled) instead of every 30 frames");
REXCVAR_DEFINE_BOOL(gamma_ramp, true, "PGR4",
                    "Apply the game's display gamma ramp at present, as the 360 does (off: linear, paler)");
REXCVAR_DEFINE_BOOL(render_thread, true, "PGR4",
                    "Record the frame on a render thread; the game thread only copies what each call reads");
REXCVAR_DEFINE_BOOL(render_ahead, true, "PGR4",
                    "With render_thread: the game starts its next frame while the last one is still recorded "
                    "(it waits at Swap only for the frame before)");
REXCVAR_DEFINE_BOOL(buffer_watch, true, "PGR4",
                    "Write-protect the guest pages of cached vertex and index data and refresh a buffer only after a "
                    "write, instead of hashing it every frame");
REXCVAR_DEFINE_BOOL(dynamic_stream_copy, true, "PGR4",
                    "With render_thread: copy vertex streams the game rewrites every frame when a draw is queued");
// Off: PGR4 refills some vertex buffers between draws of one frame, so a copy
// taken at submission gave earlier draws the last contents (black cable
// wedges and crowd patches on the Steam Deck).
REXCVAR_DEFINE_BOOL(late_buffer_fill, false, "PGR4",
                    "Copy vertex/index buffers the game rewrites at frame submission instead of at the draw");
REXCVAR_DEFINE_INT32(texture_workers, -1, "PGR4",
                     "Threads that untile and convert textures while the frame records (-1: auto, 0: inline)");
REXCVAR_DEFINE_BOOL(profile_render_thread, false, "PGR4",
                    "Debug: sample the render thread's call stacks 1000 times a second into render_samples.txt");
namespace pgr4 {
bool WalkCameraActive(); // walk_camera_pace.cpp
} // namespace pgr4

REXCVAR_DEFINE_INT32(frame_rate_limit, 60, "PGR4",
                     "Frames per second the game may run at (0: unlimited; the 360 runs PGR4 at 30)");
REXCVAR_DEFINE_INT32(internal_resolution, 720, "PGR4",
                     "Internal rendering height: 720 (native), 1080, 1440 or 2160; every render target is scaled "
                     "by this over 720");

namespace pgr4::gpu {
namespace {

// ---------------------------------------------------------------------------
// Guest memory

// With the render thread, guest memory the game may change before a queued
// command runs (the device's state, arguments on the guest stack) is copied
// when the command is queued, and while it runs GuestVirtual reads the copies.
// Physical copies hold vertex streams the game rewrites every frame
// (DynamicStreams), read through GuestPhysical.
struct GuestCopy {
  uint32_t address = 0, size = 0, offset = 0;
  bool physical = false;
  const uint8_t *external = nullptr; // a shared snapshot instead of the batch's bytes
};
thread_local const GuestCopy *t_copies = nullptr;
thread_local uint32_t t_copy_count = 0;
thread_local const uint8_t *t_copy_bytes = nullptr;
thread_local bool t_physical_copies = false;

thread_local uint32_t t_copy_hint = 0;

uint8_t *GuestVirtual(uint32_t va) {
  if (t_copy_count) {
    // Reads come in runs from one range (the device's state), so the last
    // range hit is tried first.
    if (t_copy_hint < t_copy_count) {
      const GuestCopy &c = t_copies[t_copy_hint];
      if (va - c.address < c.size && !c.physical) {
        return const_cast<uint8_t *>(t_copy_bytes + c.offset + (va - c.address));
      }
    }
    for (uint32_t i = 0; i < t_copy_count; ++i) {
      const GuestCopy &c = t_copies[i];
      if (va - c.address < c.size && !c.physical) {
        t_copy_hint = i;
        return const_cast<uint8_t *>(t_copy_bytes + c.offset + (va - c.address));
      }
    }
  }
  return REX_KERNEL_MEMORY()->TranslateVirtual(va);
}
uint8_t *GuestPhysical(uint32_t pa) {
  if (t_physical_copies) {
    for (uint32_t i = 0; i < t_copy_count; ++i) {
      const GuestCopy &c = t_copies[i];
      if (c.physical && pa - c.address < c.size) {
        return const_cast<uint8_t *>((c.external ? c.external : t_copy_bytes + c.offset) + (pa - c.address));
      }
    }
  }
  return REX_KERNEL_MEMORY()->TranslatePhysical(pa);
}

uint32_t LoadBE32(const uint8_t *p) {
  uint32_t v;
  std::memcpy(&v, p, 4);
  return _byteswap_ulong(v);
}
uint16_t LoadBE16(const uint8_t *p) {
  uint16_t v;
  std::memcpy(&v, p, 2);
  return _byteswap_ushort(v);
}
uint32_t ReadU32(uint32_t va) { return LoadBE32(GuestVirtual(va)); }
float AsFloat(uint32_t u) {
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

// Internal resolution: scene-sized render targets, and the copies resolved
// from them, are this many times their guest size. Guest sizes stay in guest
// pixels throughout; only what reaches D3D12 (texture sizes, viewports,
// scissors, copy boxes) scales, by the target's own scale.
float ResolutionScale() {
  static const float scale = float(std::clamp(REXCVAR_GET(internal_resolution), 360, 4320)) / 720.0f;
  return scale;
}
// Small targets keep their size: the exposure chain ends in 1x1 targets, which
// at 1.5x would be 2x2 with a viewport covering only part of them, and the
// averaged exposure then mixed in the uncovered texels.
// Host pixels per guest pixel, per axis.
struct Scale2 {
  float x = 1.0f, y = 1.0f;
  bool operator==(const Scale2 &o) const { return x == o.x && y == o.y; }
  bool operator!=(const Scale2 &o) const { return !(*this == o); }
};
// Ultrawide: how much wider than 16:9 the display is (1 for 16:9 or less).
float AspectWidening() {
  static const float widen = [] {
    const double aspect = Video::DisplayAspect();
    return aspect > 16.0 / 9.0 && aspect <= 6.0 ? float(aspect / (16.0 / 9.0)) : 1.0f;
  }();
  return widen;
}
// Narrower than 16:9 (the Deck's 16:10): how much taller the display is. Its
// screen targets get the extra height, mirroring the ultrawide widening.
float AspectNarrowing() {
  static const float narrow = [] {
    const double aspect = Video::DisplayAspect();
    return aspect >= 1.0 && aspect < 16.0 / 9.0 - 0.01 ? float((16.0 / 9.0) / aspect) : 1.0f;
  }();
  return narrow;
}
// Screen-shaped (16:9) targets also take the ultrawide widening: the camera
// draws the wider view squeezed into the guest's 1280x720, and the host
// target's extra width gives it back its pixels. Other targets (shadow maps,
// cube faces, which must stay square) scale evenly.
// PGR4's 2D shaders (HostShader::screen_2d, from the shader cache) are
// squeezed to 16:9 shape on an ultrawide target (ExecuteDraw).
// HUD dials (25E763FFD2C62BC7): placed by constants, and drawn only in races.
constexpr uint64_t kUiDialVs = 0x969A201324F7EFF7ull;


Scale2 TargetScale(uint32_t width, uint32_t height) {
  if (width < 256 || height < 128) {
    return {};
  }
  const float s = ResolutionScale();
  const bool screen = width * 9 == height * 16;
  return {screen ? s * AspectWidening() : s, screen ? s * AspectNarrowing() : s};
}
uint32_t ScaledSize(uint32_t guest, float scale) {
  return std::max(1u, uint32_t(std::ceil(float(guest) * scale)));
}
int32_t ScaledEdge(int32_t guest, float scale) { return int32_t(std::lround(float(guest) * scale)); }

// ---------------------------------------------------------------------------
// PGR4's D3DDevice layout, from the generated D3DDevice_* setters.

namespace dev {
constexpr uint32_t kFetchConstants = 0x480;     // 32 x 6 dwords (SetTexture, SetStreamSource)
constexpr uint32_t kVsConstants = 0x780;        // 256 float4 (ALU constants 0x4000)
constexpr uint32_t kPsConstants = 0x1780;       // 256 float4 (ALU constants 0x4400)
constexpr uint32_t kBoolConstants = 0x2780;     // 8 dwords (0x4900)
constexpr uint32_t kVertexDeclaration = 0x2E24; // SetVertexDeclaration
constexpr uint32_t kIndexBuffer = 0x308C;       // SetIndices
constexpr uint32_t kRenderTargets = 0x3090;     // SetRenderTarget, 4 surfaces
constexpr uint32_t kDepthStencil = 0x30A0;      // SetDepthStencilSurface
constexpr uint32_t kStreamStrides = 0x30E8;     // SetStreamSource, u8 stride / 4
constexpr uint32_t kPixelShader = 0x318C;       // SetPixelShader
// SetScissorRect's RECT (left, top, right, bottom) and the scissor test
// enable it checks. The window scissor registers it also writes hold the
// current tile's bounds under predicated tiling, which this renderer does not
// split draws by.
constexpr uint32_t kScissorRect = 0x317C;
constexpr uint32_t kScissorEnable = 0x2E48;
constexpr uint32_t kVertexShader = 0x3190;      // SetVertexShader
} // namespace dev

// Register shadow blocks the draw-time flush writes from (DrawIndexedVertices
// passes each block's first register and shadow address to sub_826DDC90).
struct RegisterBlock {
  uint32_t first;
  uint32_t count;
  uint32_t shadow;
};
constexpr RegisterBlock kRegisterBlocks[] = {
    {0x2000, 16, 0x2880}, {0x2100, 21, 0x28CC}, {0x2200, 12, 0x2934},
    {0x2280, 21, 0x2964}, {0x2300, 26, 0x29B8}, {0x2380, 8, 0x2A50},
};

uint32_t ReadRegister(uint32_t device, uint32_t reg) {
  for (const auto &b : kRegisterBlocks) {
    if (reg >= b.first && reg < b.first + b.count) {
      return ReadU32(device + b.shadow + (reg - b.first) * 4);
    }
  }
  return 0;
}

constexpr uint32_t kRegColorMask = 0x2104;
constexpr uint32_t kRegBlendRed = 0x2105;
constexpr uint32_t kRegStencilRefMaskBack = 0x210C;
constexpr uint32_t kRegStencilRefMask = 0x210D;
constexpr uint32_t kRegAlphaRef = 0x210E;
constexpr uint32_t kRegDepthControl = 0x2200;
constexpr uint32_t kRegBlendControl0 = 0x2201;
constexpr uint32_t kRegColorControl = 0x2202;
constexpr uint32_t kRegModeControl = 0x2205;   // PA_SU_SC_MODE_CNTL
constexpr uint32_t kRegVteControl = 0x2206;    // PA_CL_VTE_CNTL
constexpr uint32_t kRegViewportXScale = 0x210F; // PA_CL_VPORT_XSCALE, then XOFFSET, YSCALE, YOFFSET
constexpr uint32_t kRegPolyOffsetFrontScale = 0x2380;
constexpr uint32_t kRegPolyOffsetFrontOffset = 0x2381;
constexpr uint32_t kRegBlendControl1 = 0x2209;

// ---------------------------------------------------------------------------
// Shared constants (mirrors the SharedConstants tail in
// tools/shaders/shader_common_sm51.h; the VS copy is bound at b3).

struct SharedConstants {
  uint32_t texture_2d[16];   // c0-c3
  uint32_t texture_3d[16];   // c4-c7
  uint32_t texture_cube[16]; // c8-c11
  uint32_t sampler[16];      // c12-c15
  uint32_t booleans;         // c16.x
  uint32_t swapped_texcoords;
  float half_pixel_offset[2];
  float alpha_threshold; // c17.x
  uint32_t spec_constants;
  uint32_t input_modes;
  int32_t base_vertex; // c17.w, added to SV_VertexID for r0.x
  uint32_t bool_words[8]; // c18-c19
  // c20-c27: per VS input, valid | stream << 1 | byte offset << 5 |
  // byte stride << 16, then the D3DDECLTYPE (indexed vertex fetch).
  uint32_t fetch_layouts[32];
  // c28.xy: clip-space xy scale applied before g_HalfPixelOffset. Oversized
  // guest viewports are shrunk by this factor to fit host limits (see
  // ExecuteDraw), and the vertex shader scales positions back up to match.
  float ndc_scale[2];
  // c28.zw: guest pixels per host pixel, per axis (1 / the target's scale),
  // for pixel shaders that read their screen position.
  float pixel_scale[2];
  // c29.x: ultrawide 2D squeeze for the vertex shader (1 = none). Vertices
  // inside the screen move towards its centre by this factor, so UI keeps its
  // shape; vertices on the left or right edge stay there, so full-screen
  // backdrops, tints and edge-to-edge banners still fill the width.
  float ui_squeeze;
  // c29.yzw: the pixel shader clamps its colour output to [low, high], alpha
  // to [low, alpha high], as the guest render target format would store it
  // (OutputRange; high 0 = no clamp).
  float output_range[3];
  // c30.x: the draw's constant polygon offset, added to the vertex shader's
  // depth (z += offset * w). On the host's float depth buffer the pipeline's
  // constant bias scales with the depth's exponent, so far decals and
  // banners (near 0 in reversed Z) lost it and z-fought ("the void").
  float depth_offset;
  // c30.y: the 16:10 counterpart of ui_squeeze, on y (top and bottom edges
  // stay, so full-screen quads still fill the height).
  float ui_squeeze_y;
  // c30.z: race HUD batches spanning the top and bottom thirds (the shadow
  // plates under every HUD text): the vertex shader moves each vertex in the
  // top third up by this and each in the bottom third down.
  float ui_edge_shift_y;
  // c30.w: per VS input (2 bits, declaration order), how many components its
  // vertex format lacks (0 = four). The shader's decode fills them as the
  // Xenos does (0, 0, 1) instead of trusting the input assembler: AMD's
  // Vulkan drivers under vkd3d-proton fill a float format's missing w of a
  // uint4 input with integer 1, not 1.0f, so float2/float3 positions read as
  // .xyzw got w = 0 (Geometry Wars, PGR4's medals, speedometer, minimap).
  uint32_t input_missing;
};
static_assert(sizeof(SharedConstants) == 31 * 16);

constexpr uint32_t kSpecR11G11B10Normal = 1u << 0;
constexpr uint32_t kSpecAlphaTest = 1u << 1;

// Per-draw table sizes (shader_common_sm51.h).
constexpr uint32_t kTable2DCount = 32;
constexpr uint32_t kTable3DCount = 8;
constexpr uint32_t kTableCubeCount = 8;
// Resource binding tiers 1 and 2 allow 16 samplers per stage.
constexpr uint32_t kTableSamplerCount = 16;
constexpr uint32_t kTableStreamCount = 16; // ByteAddressBuffer[16] t0 space4

// Root descriptor indices (b0 VS, b1 PS, b2 PS shared, b3 VS shared).
constexpr uint32_t kRootVsConstants = 0;
constexpr uint32_t kRootPsConstants = 1;
constexpr uint32_t kRootPsShared = 2;
constexpr uint32_t kRootVsShared = 3;

// Vertex input slot that reads zeros for inputs the declaration lacks.
constexpr uint32_t kNullStream = 15;

// ---------------------------------------------------------------------------
// Host objects

struct HostShader {
  const CachedShader *cached = nullptr;
  std::unique_ptr<plume::RenderShader> shader;
  uint64_t key = 0;
};

struct DeclElement {
  uint16_t stream;
  uint16_t offset;
  uint32_t type;
  uint8_t usage;
  uint8_t usage_index;
};

struct VertexDeclaration {
  std::vector<DeclElement> elements;
  uint64_t hash = 0;
};

// Input layout for one (vertex shader, declaration) pair.
struct InputBinding {
  std::vector<plume::RenderInputElement> elements;
  uint32_t input_modes = 0;
  uint32_t input_missing = 0; // SharedConstants::input_missing
  uint32_t stream_mask = 0;
  bool r11g11b10 = false;
};

// A guest physical range whose write protection is armed (buffer_watch),
// and whether a guest write has hit it since.
struct PageWatch {
  uint32_t start = 0, end = 0;
  std::atomic<bool> written{true};
  // The game's frame (g_game_frame) at the first write since armed.
  std::atomic<uint64_t> written_frame{0};
};

// Swaps the game has issued, and the render thread has recorded. They differ
// only with render_ahead, by at most one.
std::atomic<uint64_t> g_game_frame{0};
std::atomic<uint64_t> g_recorded_frame{0};

// PGR4 is running (not Geometry Wars, launched from its arcade).
std::atomic<bool> g_pgr4_title{true};

struct HostTexture {
  std::unique_ptr<plume::RenderTexture> texture;
  std::unique_ptr<plume::RenderTextureView> view;
  plume::RenderTextureViewDimension dimension = plume::RenderTextureViewDimension::TEXTURE_2D;
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t id = 0;
  uint32_t guest_size = 0;
  uint64_t content_hash = 0;
  uint64_t sample_hash = 0; // TextureSampleHash of the memory at the last full hash
  uint64_t validated_frame = 0;
  uint64_t sampled_frame = 0;
  uint64_t used_frame = 0;
  uint64_t changed_frame = ~0ull; // last re-upload over changed contents
  PageWatch watch; // the base level's guest memory
};

// Host copy of a guest texture's memory written by Resolve/EndTiling.
struct ResolvedTexture {
  std::unique_ptr<plume::RenderTexture> texture;
  plume::RenderFormat format = plume::RenderFormat::UNKNOWN;
  uint32_t width = 0; // guest size; the host texture is ScaledSize() of it
  uint32_t height = 0;
  Scale2 scale; // the source target's
  uint32_t levels = 1; // mip levels, each written by its own Resolve
  uint32_t written_levels = 0; // bit per level resolved so far; views cover the written run from 0
  uint32_t id = 0;
  bool depth = false;
  bool cube = false; // 6 faces, written one Resolve per face
  // Guest memory under the copy when it was resolved. The host copy stands in
  // for that memory only until the game puts other data there (a streamed
  // texture reusing a freed render-to-texture buffer).
  uint32_t guest_base = 0;
  uint64_t guest_hash = 0;
  uint64_t checked_frame = 0;
  std::unordered_map<uint32_t, std::unique_ptr<plume::RenderTextureView>> views; // by swizzle
};

struct RenderTarget {
  std::unique_ptr<plume::RenderTexture> texture;
  plume::RenderFormat format = plume::RenderFormat::UNKNOWN;
  uint32_t guest_format = 0; // RB_COLOR_INFO format (color targets)
  uint32_t width = 0; // guest size; the host texture is ScaledSize() of it
  uint32_t height = 0;
  Scale2 scale; // TargetScale()
  uint32_t id = 0;
  bool depth = false;
  // A new host target's memory is undefined until its first clear (D3D12
  // placed resources). NVIDIA and WARP hand back zeros; AMD hands back stale
  // memory, which the game then reads (e.g. last frame's exposure).
  bool needs_init = true;
  // EDRAM: every surface at the same EDRAM base is a view of the same memory
  // on the 360, so a clear through one is seen through the others. Each view
  // here is its own host texture; a clear of one is queued on the others and
  // applied when they are next bound (BindFramebuffer). PGR4's rear-view
  // mirror is drawn into surfaces it never clears itself, relying on the
  // previous frame's resolve-and-clear of the main scene's (its depth test
  // fails against stale depth otherwise). Depth clears only (ClearTargets).
  uint32_t edram_base = ~0u;
  bool clear_pending = false;
  bool clear_depth = false, clear_stencil = false;
  float clear_z = 0.0f;
  uint32_t clear_stencil_value = 0;
  // Ultrawide: whether the 3D view has drawn across this screen target since
  // its last clear or the last frame, and whether the side bars outside the middle
  // 16:9 were blacked out for 2D-only screens (ExecuteDraw).
  bool wide_scene_drawn = false;
  bool wide_bars_clean = false;
};

struct HostBuffer {
  std::unique_ptr<plume::RenderBuffer> buffer;
  uint64_t id = 0; // new for every host buffer created
  uint32_t size = 0;
  uint32_t capacity = 0; // pooled size class of buffer (see TakePooledBuffer)
  uint64_t content_hash = 0;
  uint64_t sample_hash = 0; // TextureSampleHash at the last full hash
  uint64_t validated_frame = 0;
  uint64_t sampled_frame = 0;
  // Seen changing: filled at submission each frame it is used (LateFill).
  bool dynamic = false;
  uint64_t late_frame = ~0ull;
  uint64_t used_frame = 0;
  PageWatch watch;
};

struct CachedSet {
  std::unique_ptr<plume::RenderDescriptorSet> set;
  uint64_t used_frame = 0;
  uint32_t views = 0;    // shader-visible heap descriptors the set holds
  uint32_t samplers = 0;
};

struct Viewport {
  float x = 0, y = 0, width = 0, height = 0, min_z = 0, max_z = 1;
  bool valid = false;
};

// One bound texture for a draw: what the shader table slot points at.
struct BoundTexture {
  const plume::RenderTexture *texture = nullptr;
  const plume::RenderTextureView *view = nullptr;
  uint64_t id = 0;
  plume::RenderTextureViewDimension dimension = plume::RenderTextureViewDimension::TEXTURE_2D;
};

struct RendererState {
  std::mutex mutex;
  bool init_attempted = false;
  bool ready = false;
  plume::RenderDevice *device = nullptr;

  std::unique_ptr<plume::RenderPipelineLayout> layout;
  std::unordered_map<uint64_t, std::unique_ptr<HostShader>> shaders_by_key;
  std::unordered_map<uint32_t, HostShader *> shaders_by_object;
  std::unordered_set<uint64_t> logged_missing_shaders;

  std::unordered_map<uint32_t, VertexDeclaration> declarations;
  std::unordered_map<uint64_t, InputBinding> input_bindings;
  std::unordered_map<uint64_t, std::unique_ptr<plume::RenderPipeline>> pipelines;

  std::unordered_map<uint64_t, std::unique_ptr<HostTexture>> textures;
  // Host copies of resolved memory, one per (base, size, format); `latest`
  // is the copy last written at each base address, which sampling uses.
  std::unordered_map<uint64_t, std::unique_ptr<ResolvedTexture>> resolved;
  std::unordered_map<uint32_t, ResolvedTexture *> latest_resolved;
  struct SamplerSlot {
    std::unique_ptr<plume::RenderSampler> sampler;
    uint32_t index = 0;
  };
  std::unordered_map<uint32_t, SamplerSlot> samplers; // by guest sampler key
  std::unordered_map<uint64_t, CachedSet> sets;
  // Raw stream tables (indexed vertex fetch), keyed by HostBuffer ids. Their
  // buffers are recreated whenever the guest rewrites them (crowd poses every
  // frame), so these live only a few frames.
  std::unordered_map<uint64_t, CachedSet> stream_sets;
  uint32_t set_views = 0;    // descriptors held by sets and stream_sets
  uint32_t set_samplers = 0;
  uint64_t sets_trimmed_frame = ~0ull;
  uint32_t hide_texture = 0; // debug: hide.tex
  uint64_t skip_ps = 0;       // debug: skip.ps
  uint64_t next_buffer_id = 1;
  std::unordered_map<uint64_t, std::unique_ptr<HostBuffer>> buffers;
  std::unordered_map<uint64_t, std::unique_ptr<RenderTarget>> render_targets; // surface + shape
  // Union of the BeginTiling rects while a tiled pass is open.
  bool tiling_active = false;
  uint32_t tiling_width = 0;
  uint32_t tiling_height = 0;
  std::unordered_map<uint64_t, std::unique_ptr<plume::RenderFramebuffer>> framebuffers;
  std::unordered_map<uint32_t, Viewport> viewports; // by device

  // Placeholders for table slots no texture uses.
  std::unique_ptr<plume::RenderTexture> dummy_2d, dummy_3d, dummy_cube;
  std::unique_ptr<plume::RenderTextureView> dummy_2d_view, dummy_3d_view, dummy_cube_view;
  std::unique_ptr<plume::RenderSampler> default_sampler;
  std::vector<plume::RenderSampler *> sampler_list; // by SamplerFor index
  std::unique_ptr<plume::RenderBuffer> null_stream;
  std::unique_ptr<plume::RenderShader> solid_ps; // --debug_solid_ps
  // Quad list pattern (0 1 2 0 2 3, +4 a quad) for non-indexed quad lists,
  // drawn with the first vertex as base vertex (--quad_index_buffer).
  std::unique_ptr<plume::RenderBuffer> quad_indices;
  uint32_t quad_index_quads = 0;

  uint32_t next_id = 1;
  uint64_t frame = 1;
  bool frame_prepared = false;
  plume::RenderFramebuffer *bound_framebuffer = nullptr;
  // Texture setups already built this frame, by the shaders' sampler fetch
  // constants: a draw with the same ones reuses the tables and slot
  // assignments. Bumping binding_generation (a resolve changed what an
  // address samples, or a descriptor set was freed) drops them.
  struct TextureSetup {
    uint64_t generation = 0;
    plume::RenderDescriptorSet *sets[4] = {};
    uint32_t vs_slots[64], ps_slots[64]; // SharedConstants texture_* and sampler
    bool samples_wide = false;            // see RendererState::draw_samples_wide
  };
  std::unordered_map<uint64_t, TextureSetup> texture_setups;
  // The current draw samples a copy resolved from an ultrawide screen target
  // (a 3D view composited by a 2D quad), so it fills the width (ExecuteDraw).
  bool draw_samples_wide = false;
  // Ultrawide: the race HUD (its speedo dial) drew this / the last frame, so
  // 2D draws are anchored to the window's sides (ExecuteDraw).
  bool dial_drawn = false;
  bool race_hud = false;
  // BufferFor results already validated this frame (direct-mapped, by key).
  struct BufferHit {
    uint64_t key = 0, frame = 0;
    const HostBuffer *buffer = nullptr;
  };
  std::array<BufferHit, 512> buffer_hits;
  // Guest buffers the game rewrites, copied into their host buffers when the
  // frame is submitted (Swap) instead of at the draw. The 360's GPU reads a
  // frame's vertices after the CPU has finished it, and PGR4 fills some of
  // them on other threads after issuing the draw; read at the draw, those
  // came out half-written (one-frame black beams over the track).
  struct LateFill {
    plume::RenderBuffer *buffer;
    uint8_t *mapped;
    const uint8_t *src;
    uint32_t size, endian, half_stride;
    uint64_t half_mask;
  };
  std::vector<LateFill> late_fills;
  // Depth targets holding a clear shared from another view of their EDRAM
  // (RenderTarget::clear_pending); render targets are never freed.
  std::vector<RenderTarget *> shared_clears;
  // Texture + sampler bound for a fetch constant this frame (direct-mapped).
  struct FetchHit {
    uint64_t frame = 0, generation = 0;
    uint32_t fetch[6] = {};
    BoundTexture bound;
    uint32_t sampler = 0;
  };
  std::array<FetchHit, 1024> fetch_hits;
  uint64_t binding_generation = 1;
  // State last recorded into the command list by draws, so unchanged state is
  // not set again (each set is a driver call, thousands per frame). Valid for
  // one list generation; reset wherever bound_framebuffer is.
  struct ListBindings {
    uint64_t generation = ~0ull;
    const plume::RenderPipeline *pipeline = nullptr;
    bool layout = false;
    const plume::RenderDescriptorSet *sets[5] = {};
    plume::RenderBufferReference roots[4];
    bool viewport_valid = false;
    plume::RenderViewport viewport;
    bool scissor_valid = false;
    plume::RenderRect scissor;
    struct VertexStream {
      plume::RenderBufferReference buffer;
      uint32_t size = 0;
      uint32_t stride = ~0u;
    } streams[16];
  } bound;

  // Last uploaded constants, reused while unchanged.
  struct ConstantSlot {
    std::vector<uint8_t> data;
    UploadAllocation alloc;
  };
  ConstantSlot vs_constants, ps_constants, vs_shared, ps_shared;

  uint64_t draws = 0;
  uint64_t skipped_draws = 0;
  // Debug: `dump.now` in the working directory dumps every resolve of the
  // next frame (dump_<seq>_<base>.bmp) and logs the draws feeding each.
  bool dumping = false;
  uint32_t dump_seq = 0;
  uint64_t draws_since_resolve = 0;
  std::unordered_set<uint64_t> logged_once;
};

RendererState &state() {
  static RendererState s;
  return s;
}

bool LogOnce(RendererState &s, uint64_t key) { return s.logged_once.insert(key).second; }

// ---------------------------------------------------------------------------
// Setup

bool CreateDummyTexture(RendererState &s, plume::RenderTextureDimension dimension,
                        plume::RenderTextureViewDimension view_dimension, uint32_t array_size,
                        std::unique_ptr<plume::RenderTexture> &texture,
                        std::unique_ptr<plume::RenderTextureView> &view,
                        plume::RenderCommandList *cmd) {
  plume::RenderTextureDesc desc = plume::RenderTextureDesc::Texture(
      dimension, 1, 1, 1, 1, array_size, plume::RenderFormat::R8G8B8A8_UNORM,
      array_size == 6 ? plume::RenderTextureFlag::CUBE : plume::RenderTextureFlag::NONE);
  texture = s.device->createTexture(desc);
  if (!texture) {
    return false;
  }
  texture->setName(fmt::format("dummy {}", int(view_dimension)));
  plume::RenderTextureViewDesc vd;
  vd.format = plume::RenderFormat::R8G8B8A8_UNORM;
  vd.dimension = view_dimension;
  view = texture->createTextureView(vd);
  if (!view) {
    return false;
  }
  // Zero texels, like an unbound Xenos fetch. Each layer's footprint starts
  // on a D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT boundary.
  const uint32_t pitch = 512;
  UploadAllocation up = Video::Upload(nullptr, pitch * array_size, 512);
  if (!up) {
    return false;
  }
  std::memset(up.memory, 0, pitch * array_size);
  cmd->barriers(plume::RenderBarrierStage::COPY,
                plume::RenderTextureBarrier(texture.get(), plume::RenderTextureLayout::COPY_DEST));
  for (uint32_t layer = 0; layer < array_size; ++layer) {
    cmd->copyTextureRegion(plume::RenderTextureCopyLocation::Subresource(texture.get(), 0, layer),
                           plume::RenderTextureCopyLocation::PlacedFootprint(
                               up.buffer, plume::RenderFormat::R8G8B8A8_UNORM, 1, 1, 1, 64,
                               up.offset + layer * pitch));
  }
  cmd->barriers(plume::RenderBarrierStage::GRAPHICS,
                plume::RenderTextureBarrier(texture.get(), plume::RenderTextureLayout::SHADER_READ));
  return true;
}

void BuildTableLayouts(plume::RenderDescriptorSetBuilder &t2d, plume::RenderDescriptorSetBuilder &t3d,
                       plume::RenderDescriptorSetBuilder &tcube,
                       plume::RenderDescriptorSetBuilder &samplers) {
  t2d.begin();
  t2d.addTexture(0, kTable2DCount);
  t2d.end();
  t3d.begin();
  t3d.addTexture(0, kTable3DCount);
  t3d.end();
  tcube.begin();
  tcube.addTexture(0, kTableCubeCount);
  tcube.end();
  samplers.begin();
  samplers.addSampler(0, kTableSamplerCount);
  samplers.end();
}

bool EnsureReady(RendererState &s) {
  if (s.ready) {
    return true;
  }
  if (s.init_attempted) {
    return false;
  }
  plume::RenderCommandList *cmd = Video::OpenCommandList();
  s.device = Video::HostDevice();
  if (!cmd || !s.device) {
    return false;
  }
  s.init_attempted = true;
  if (!ShaderCache::Load(s.device)) {
    return false;
  }

  plume::RenderDescriptorSetBuilder t2d, t3d, tcube, samp;
  BuildTableLayouts(t2d, t3d, tcube, samp);
  plume::RenderPipelineLayoutBuilder lb;
  lb.begin(false, true);
  lb.addDescriptorSet(t2d);   // Texture2D[32]   t0 space0
  lb.addDescriptorSet(t3d);   // Texture3D[8]    t0 space1
  lb.addDescriptorSet(tcube); // TextureCube[8]  t0 space2
  lb.addDescriptorSet(samp);  // Sampler[16]     s0 space3
  plume::RenderDescriptorSetBuilder streams;
  streams.begin();
  streams.addByteAddressBuffer(0, kTableStreamCount);
  streams.end();
  lb.addDescriptorSet(streams); // ByteAddressBuffer[16] t0 space4 (indexed vertex fetch)
  lb.addRootDescriptor(0, 4, plume::RenderRootDescriptorType::CONSTANT_BUFFER);
  lb.addRootDescriptor(1, 4, plume::RenderRootDescriptorType::CONSTANT_BUFFER);
  lb.addRootDescriptor(2, 4, plume::RenderRootDescriptorType::CONSTANT_BUFFER);
  lb.addRootDescriptor(3, 4, plume::RenderRootDescriptorType::CONSTANT_BUFFER);
  lb.end();
  s.layout = lb.create(s.device);
  if (!s.layout) {
    REXGPU_ERROR("PGR4 D3D12: guest pipeline layout creation failed");
    return false;
  }

  if (!CreateDummyTexture(s, plume::RenderTextureDimension::TEXTURE_2D,
                          plume::RenderTextureViewDimension::TEXTURE_2D, 1, s.dummy_2d,
                          s.dummy_2d_view, cmd) ||
      !CreateDummyTexture(s, plume::RenderTextureDimension::TEXTURE_3D,
                          plume::RenderTextureViewDimension::TEXTURE_3D, 1, s.dummy_3d,
                          s.dummy_3d_view, cmd) ||
      !CreateDummyTexture(s, plume::RenderTextureDimension::TEXTURE_2D,
                          plume::RenderTextureViewDimension::TEXTURE_CUBE, 6, s.dummy_cube,
                          s.dummy_cube_view, cmd)) {
    REXGPU_ERROR("PGR4 D3D12: placeholder texture creation failed");
    return false;
  }

  plume::RenderSamplerDesc sd;
  sd.addressU = sd.addressV = sd.addressW = plume::RenderTextureAddressMode::CLAMP;
  s.default_sampler = s.device->createSampler(sd);
  s.null_stream = s.device->createBuffer(
      plume::RenderBufferDesc::VertexBuffer(256, plume::RenderHeapType::UPLOAD));
  if (!s.default_sampler || !s.null_stream) {
    return false;
  }
  s.null_stream->setName("null stream");
  s.sampler_list.assign(1, s.default_sampler.get());
  std::memset(s.null_stream->map(), 0, 256);
  s.null_stream->unmap();

  s.ready = true;
  REXGPU_INFO("PGR4 D3D12: guest renderer ready");
  return true;
}

// ---------------------------------------------------------------------------
// Shaders

HostShader *ShaderFor(RendererState &s, uint32_t object) {
  auto it = s.shaders_by_object.find(object);
  return it != s.shaders_by_object.end() ? it->second : nullptr;
}

void RegisterShader(RendererState &s, uint32_t object, uint32_t container, bool pixel) {
  const uint8_t *header = GuestVirtual(container);
  const uint32_t flags = LoadBE32(header);
  const uint32_t size = LoadBE32(header + 4) + LoadBE32(header + 8);
  if ((flags & 0xFFFFFF00u) != 0x102A1100u || !size || size > (16u << 20)) {
    s.shaders_by_object.erase(object);
    return;
  }
  const uint64_t key = ShaderCache::HashContainer(header, size);
  auto &entry = s.shaders_by_key[key];
  if (!entry) {
    entry = std::make_unique<HostShader>();
    entry->key = key;
    entry->cached = ShaderCache::Find(key);
    if (entry->cached && s.device) {
      entry->shader = s.device->createShader(entry->cached->dxbc, entry->cached->dxbc_size, "main",
                                             ShaderCache::IsDxil() ? plume::RenderShaderFormat::DXIL
                                                                   : plume::RenderShaderFormat::DXBC);
    }
    if (!entry->shader && s.logged_missing_shaders.insert(key).second) {
      REXGPU_WARN("PGR4 D3D12: no cached {} shader {:016X} (container {:08X}, {} bytes)",
                  pixel ? "pixel" : "vertex", key, container, size);
      // Shaders the game builds at runtime are not in the XEX: keep the
      // container so tools/shaders can recompile it (shaders/bin/runtime).
      std::filesystem::create_directories("shader_dump");
      char name[64];
      std::snprintf(name, sizeof(name), "shader_dump/%016llX.%s.bin", (unsigned long long)key,
                    pixel ? "ps" : "vs");
      if (std::FILE *out = std::fopen(name, "wb")) {
        std::fwrite(header, 1, size, out);
        std::fclose(out);
      }
    }
  }
  s.shaders_by_object[object] = entry.get();
}

// ---------------------------------------------------------------------------
// Vertex declarations and input layouts

const char *const kUsageNames[] = {"POSITION", "BLENDWEIGHT", "BLENDINDICES", "NORMAL",
                                   "PSIZE",    "TEXCOORD",    "TANGENT",      "BINORMAL",
                                   "TESSFACTOR", "POSITIONT", "COLOR",        "FOG",
                                   "DEPTH",    "SAMPLE"};

const VertexDeclaration *DeclarationFor(RendererState &s, uint32_t decl) {
  if (!decl) {
    return nullptr;
  }
  // PGR4's D3DVertexDeclaration: element count at +0x18, D3DVERTEXELEMENT9
  // copies (12 bytes each) from +0x34 (D3DDevice_CreateVertexDeclaration).
  const uint32_t count = ReadU32(decl + 0x18);
  if (count > 64) {
    return nullptr;
  }
  const uint8_t *elements = GuestVirtual(decl + 0x34);
  const uint64_t hash = XXH3_64bits(elements, count * 12);
  auto &d = s.declarations[decl];
  if (d.hash != hash || d.elements.size() != count) {
    d.hash = hash;
    d.elements.clear();
    for (uint32_t i = 0; i < count; ++i) {
      const uint8_t *e = elements + i * 12;
      DeclElement el;
      el.stream = LoadBE16(e);
      el.offset = LoadBE16(e + 2);
      el.type = LoadBE32(e + 4);
      el.usage = e[9];
      el.usage_index = e[10];
      if (el.stream == 0xFF) {
        break;
      }
      d.elements.push_back(el);
    }
  }
  return &d;
}

// Xbox 360 D3DDECLTYPE: bits 0-5 are the GPU vertex format, bit 8 signed,
// bit 9 integer (not normalized). Returns the IA format and the shader-side
// Components a vertex element format supplies.
uint32_t FormatComponents(plume::RenderFormat format) {
  using F = plume::RenderFormat;
  switch (format) {
  case F::R32_FLOAT: case F::R32_UINT: case F::R32_SINT:
  case F::R16_FLOAT: case F::R16_UNORM: case F::R16_SNORM: case F::R16_UINT: case F::R16_SINT:
  case F::R8_UNORM: case F::R8_SNORM: case F::R8_UINT: case F::R8_SINT:
    return 1;
  case F::R32G32_FLOAT: case F::R32G32_UINT: case F::R32G32_SINT:
  case F::R16G16_FLOAT: case F::R16G16_UNORM: case F::R16G16_SNORM: case F::R16G16_UINT: case F::R16G16_SINT:
  case F::R8G8_UNORM: case F::R8G8_SNORM: case F::R8G8_UINT: case F::R8G8_SINT:
    return 2;
  case F::R32G32B32_FLOAT: case F::R32G32B32_UINT: case F::R32G32B32_SINT:
    return 3;
  default:
    return 4;
  }
}

// decode mode (shader_common_sm51.h: 0 float bits, 1 signed, 2 unsigned).
plume::RenderFormat DeclTypeFormat(uint32_t type, uint32_t &mode, bool &packed_normal) {
  using F = plume::RenderFormat;
  mode = 0;
  packed_normal = false;
  if (type == 0x182886) { // D3DCOLOR
    return F::B8G8R8A8_UNORM;
  }
  const uint32_t fmt = type & 0x3F;
  const bool is_signed = (type & 0x100) != 0;
  const bool integer = (type & 0x200) != 0;
  auto pick = [&](F unorm, F snorm, F uint_f, F sint_f) {
    if (integer) {
      mode = is_signed ? 1 : 2;
      return is_signed ? sint_f : uint_f;
    }
    return is_signed ? snorm : unorm;
  };
  switch (fmt) {
  case 0x06: // 8_8_8_8
    return pick(F::R8G8B8A8_UNORM, F::R8G8B8A8_SNORM, F::R8G8B8A8_UINT, F::R8G8B8A8_SINT);
  case 0x19: // 16_16
    return pick(F::R16G16_UNORM, F::R16G16_SNORM, F::R16G16_UINT, F::R16G16_SINT);
  case 0x1A: // 16_16_16_16
    return pick(F::R16G16B16A16_UNORM, F::R16G16B16A16_SNORM, F::R16G16B16A16_UINT,
                F::R16G16B16A16_SINT);
  case 0x1F:
    return F::R16G16_FLOAT;
  case 0x20:
    return F::R16G16B16A16_FLOAT;
  case 0x21:
    mode = is_signed ? 1 : 2;
    return is_signed ? F::R32_SINT : F::R32_UINT;
  case 0x22:
    mode = is_signed ? 1 : 2;
    return is_signed ? F::R32G32_SINT : F::R32G32_UINT;
  case 0x23:
    mode = is_signed ? 1 : 2;
    return is_signed ? F::R32G32B32A32_SINT : F::R32G32B32A32_UINT;
  case 0x24:
    return F::R32_FLOAT;
  case 0x25:
    return F::R32G32_FLOAT;
  case 0x26:
    return F::R32G32B32A32_FLOAT;
  case 0x39:
    return F::R32G32B32_FLOAT;
  case 0x07: // 2_10_10_10 (DEC3N / DEC4N): decoded in the shader
    mode = 3;
    return F::R32_UINT;
  case 0x10: // 10_11_11
  case 0x11: // 11_11_10
    // Raw bits for tfetchR11G11B10's packed-normal decode.
    packed_normal = true;
    return F::R32_UINT;
  default:
    return F::UNKNOWN;
  }
}

const InputBinding *InputBindingFor(RendererState &s, const HostShader *vs,
                                    const VertexDeclaration *decl) {
  const uint64_t key = vs->key ^ (decl->hash * 0x9E3779B97F4A7C15ull);
  auto it = s.input_bindings.find(key);
  if (it != s.input_bindings.end()) {
    return &it->second;
  }
  InputBinding b;
  const auto &inputs = vs->cached->inputs;
  for (uint32_t i = 0; i < inputs.size(); ++i) {
    const auto &in = inputs[i];
    const char *name = in.usage < std::size(kUsageNames) ? kUsageNames[in.usage] : "TEXCOORD";
    const DeclElement *match = nullptr;
    for (const auto &e : decl->elements) {
      if (e.usage == in.usage && e.usage_index == in.usage_index) {
        match = &e;
        break;
      }
    }
    plume::RenderFormat format = plume::RenderFormat::UNKNOWN;
    uint32_t mode = 0;
    bool packed = false;
    if (match) {
      format = DeclTypeFormat(match->type, mode, packed);
      if (format == plume::RenderFormat::UNKNOWN && LogOnce(s, 0x1000000000ull | match->type)) {
        REXGPU_WARN("PGR4 D3D12: unsupported vertex element type {:06X} ({}{})", match->type,
                    name, in.usage_index);
      }
    }
    if (format == plume::RenderFormat::UNKNOWN || !match || match->stream >= kNullStream) {
      b.elements.emplace_back(name, in.usage_index, i, plume::RenderFormat::R32G32B32A32_FLOAT,
                              kNullStream, 0);
      continue;
    }
    b.elements.emplace_back(name, in.usage_index, i, format, match->stream, match->offset);
    b.input_modes |= mode << (i * 2);
    // DEC3N (mode 3) and packed normals unpack four components from one.
    const uint32_t components = mode == 3 || packed ? 4u : FormatComponents(format);
    if (i < 16) {
      b.input_missing |= (4u - components) << (i * 2);
    }
    b.stream_mask |= 1u << match->stream;
    b.r11g11b10 |= packed;
  }
  return &s.input_bindings.emplace(key, std::move(b)).first->second;
}

// ---------------------------------------------------------------------------
// Buffers (vertex/index data in guest memory, endian-swapped once and
// re-validated by content hash)

// Profiling: vertex/index data work per 600 frames (logged from Swap).
// Per-stage timings only when --renderer_profile is set: reading the clock
// dozens of times per draw is itself a measurable cost.
bool g_profile = false;
std::chrono::steady_clock::time_point ProfNow() {
  return g_profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
}

struct StreamStats {
  uint64_t bind_texture_ns = 0, table_ns = 0, setup_textures_ns = 0, rehash_bytes = 0;
  uint64_t buffer_for_ns = 0, hash_ns = 0, special_ns = 0, loop_ns = 0, lookup_ns = 0, fill_ns = 0;
  uint64_t hashed = 0, hashed_bytes = 0, created = 0, created_bytes = 0, wrapped = 0, wrapped_bytes = 0;
  uint64_t indexed_ns = 0, indexed_draws = 0, new_stream_sets = 0;
};
StreamStats g_stream_stats;

// Eight 256-byte strips spread over a texture's or buffer's memory: cheap
// enough for everything bound each frame, and new contents differ in them.
uint64_t TextureSampleHash(const uint8_t *src, uint32_t size) {
  constexpr uint32_t kStrip = 256, kStrips = 8;
  if (size <= kStrip * kStrips) {
    return XXH3_64bits(src, size);
  }
  uint64_t h = 0;
  for (uint32_t i = 0; i < kStrips; ++i) {
    const uint32_t offset = uint32_t(uint64_t(size - kStrip) * i / (kStrips - 1)) & ~(kStrip - 1);
    h = XXH3_64bits_withSeed(src + offset, kStrip, h);
  }
  return h;
}

uint64_t BufferRevalidateInterval(uint32_t size) { return size <= (256u << 10) ? 1 : 30; }

// Xenos vertex fetch takes component X of a 16-bit pair from the high half
// of the (endian-swapped) dword, the opposite of a host R16G16 read. These
// dwords of each vertex get their halves exchanged after the endian swap.
struct HalfSwap {
  uint32_t stride = 0;
  uint64_t dword_mask = 0;
  explicit operator bool() const { return stride && dword_mask; }
};

HalfSwap HalfSwapFor(const VertexDeclaration &decl, uint32_t stream, uint32_t stride) {
  HalfSwap h;
  for (const auto &e : decl.elements) {
    if ((e.stream & 15) != stream) {
      continue;
    }
    uint32_t dwords = 0;
    switch (e.type & 0x3F) {
    case 0x19: // 16_16
    case 0x1F: // 16_16_FLOAT
      dwords = 1;
      break;
    case 0x1A: // 16_16_16_16
    case 0x20: // 16_16_16_16_FLOAT
      dwords = 2;
      break;
    default:
      continue;
    }
    for (uint32_t d = 0; d < dwords; ++d) {
      const uint32_t index = e.offset / 4 + d;
      if (index < 64) {
        h.dword_mask |= 1ull << index;
      }
    }
  }
  h.stride = h.dword_mask ? stride : 0;
  return h;
}

void SwapHalves(uint8_t *data, uint32_t size, const HalfSwap &h) {
  if (!h || h.stride & 3) {
    return;
  }
  for (uint32_t v = 0; v + h.stride <= size; v += h.stride) {
    uint64_t mask = h.dword_mask;
    while (mask) {
      const uint32_t i = uint32_t(std::countr_zero(mask));
      mask &= mask - 1;
      if (i * 4 + 4 > h.stride) {
        break;
      }
      uint32_t w;
      std::memcpy(&w, data + v + i * 4, 4);
      w = (w >> 16) | (w << 16);
      std::memcpy(data + v + i * 4, &w, 4);
    }
  }
}

// Upload buffers recycled by power-of-two size class. Creating a D3D12
// resource costs far more than filling one, and vertex data that changes
// every frame (crowds, particles, UI) would otherwise create dozens a frame.
// A replaced buffer comes back here only once its frame's fence has
// signalled (it is parked until then), so reuse never races the GPU.
struct BufferPool {
  std::mutex mutex;
  std::unordered_map<uint32_t, std::vector<std::unique_ptr<plume::RenderBuffer>>> free;
  uint64_t bytes = 0; // held in `free`
};
// Idle buffers kept for reuse at most (upload heaps live in system memory).
constexpr uint64_t kBufferPoolBytes = 128ull << 20;

BufferPool &buffer_pool() {
  static BufferPool pool;
  return pool;
}

uint32_t BufferSizeClass(uint32_t size) {
  uint32_t c = 256;
  while (c < size) {
    c <<= 1;
  }
  return c;
}

std::unique_ptr<plume::RenderBuffer> TakePooledBuffer(RendererState &s, uint32_t capacity) {
  {
    auto &pool = buffer_pool();
    std::lock_guard lock(pool.mutex);
    auto &list = pool.free[capacity];
    if (!list.empty()) {
      auto buffer = std::move(list.back());
      list.pop_back();
      pool.bytes -= capacity;
      return buffer;
    }
  }
  ++g_stream_stats.created;
  auto buffer = s.device->createBuffer(
      plume::RenderBufferDesc::VertexBuffer(capacity, plume::RenderHeapType::UPLOAD, plume::RenderBufferFlag::INDEX));
  if (buffer) {
    buffer->setName(fmt::format("stream buffer {} KB", capacity >> 10));
  }
  return buffer;
}

// Parks the buffer until the current frame completes, then returns it to the
// pool (or frees it when that size class already holds enough).
void RecycleBuffer(std::unique_ptr<plume::RenderBuffer> buffer, uint32_t capacity) {
  if (!buffer) {
    return;
  }
  Video::ParkErased(std::shared_ptr<void>(buffer.release(), [capacity](void *p) {
    std::unique_ptr<plume::RenderBuffer> owned(static_cast<plume::RenderBuffer *>(p));
    auto &pool = buffer_pool();
    std::lock_guard lock(pool.mutex);
    auto &list = pool.free[capacity];
    if (list.size() < 64 && pool.bytes + capacity <= kBufferPoolBytes) {
      list.push_back(std::move(owned));
      pool.bytes += capacity;
    }
  }));
}

// Endian swap plus 16-bit half exchange into upload memory. Upload heaps are
// write-combined: writing them is fast, reading them back is very slow, so
// the halves are exchanged in a CPU scratch copy (never in place in dst) and
// dst is written once, sequentially.
void SwapToUpload(uint8_t *dst, const uint8_t *src, uint32_t size, uint32_t endian, const HalfSwap &halves) {
  if (!halves) {
    SwapEndian(dst, src, size, endian);
    return;
  }
  thread_local std::vector<uint8_t> scratch;
  scratch.resize(size);
  SwapEndian(scratch.data(), src, size, endian);
  SwapHalves(scratch.data(), size, halves);
  std::memcpy(dst, scratch.data(), size);
}

// Guest writes to cached vertex, index and texture data, after pgr4-native's
// page watches: an object's guest pages are write-protected when it is read,
// the first write faults into GuestWritten, which marks every watch on those
// pages, and an unmarked object is used as it is instead of being re-hashed.
// The callback runs on the writing thread, so the list has its own lock,
// never held across calls into the memory system.
struct WatchList {
  std::mutex mutex;
  std::unordered_map<uint32_t, std::vector<PageWatch *>> buckets; // by 64 KB of guest physical memory
};
constexpr uint32_t kWatchBucketShift = 16;

WatchList &watch_list() {
  static WatchList list;
  return list;
}

// Debug: a watch.addr file next to the exe (hex guest physical address) logs
// the host stack of every write to that page (re-armed twice a second).
std::atomic<uint32_t> g_debug_watch_page{0};
std::atomic<int> g_debug_watch_logs{0};

void LogDebugWatchWrite(uint32_t start, uint32_t length) {
  if (g_debug_watch_logs.fetch_add(1, std::memory_order_relaxed) >= 200) {
    return;
  }
  void *frames[48];
  const USHORT n = RtlCaptureStackBackTrace(0, 48, frames, nullptr);
  std::string line;
  for (USHORT i = 0; i < n; ++i) {
    HMODULE module = nullptr;
    char name[MAX_PATH] = "?";
    if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           static_cast<LPCSTR>(frames[i]), &module)) {
      GetModuleFileNameA(module, name, MAX_PATH);
    }
    const char *base = std::strrchr(name, '\\');
    line += fmt::format(" {}+0x{:X}", base ? base + 1 : name, uint64_t(frames[i]) - uint64_t(module));
  }
  REXGPU_INFO("PGR4 watch.addr: write {:08X}+{:X} thread {} frame {}:{}", start, length, GetCurrentThreadId(),
              g_game_frame.load(std::memory_order_relaxed), line);
}

// Write callbacks taken (debug: shown in the 600-frame stats line).
std::atomic<uint64_t> g_watch_writes{0};

std::pair<uint32_t, uint32_t> GuestWritten(void *, uint32_t start, uint32_t length, bool) {
  g_watch_writes.fetch_add(1, std::memory_order_relaxed);
  auto &list = watch_list();
  const uint32_t end = start + std::max(length, 1u);
  if (const uint32_t page = g_debug_watch_page.load(std::memory_order_relaxed);
      page && start < page + 0x1000 && page < end) {
    LogDebugWatchWrite(start, length);
  }
  std::lock_guard lock(list.mutex);
  for (uint32_t bucket = start >> kWatchBucketShift; bucket <= (end - 1) >> kWatchBucketShift; ++bucket) {
    const auto it = list.buckets.find(bucket);
    if (it == list.buckets.end()) {
      continue;
    }
    for (PageWatch *w : it->second) {
      if (w->start < end && start < w->end && !w->written.load(std::memory_order_relaxed)) {
        w->written_frame.store(g_game_frame.load(std::memory_order_relaxed), std::memory_order_relaxed);
        w->written.store(true, std::memory_order_release);
      }
    }
  }
  return {start, length};
}

// Whether the write callback is registered with the current guest memory. A
// title switch builds a new memory (often at the old one's heap address, so
// its pointer cannot tell them apart) and clears this.
std::atomic<bool> g_watch_registered{false};

// Registers the write callback once per guest memory; false with buffer_watch
// off.
bool WatchesEnabled() {
  static const bool enabled = REXCVAR_GET(buffer_watch);
  if (!enabled) {
    return false;
  }
  if (!g_watch_registered.load(std::memory_order_acquire)) {
    static std::mutex mutex;
    std::lock_guard lock(mutex);
    if (!g_watch_registered.load(std::memory_order_relaxed)) {
      auto *memory = REX_KERNEL_MEMORY();
      memory->RegisterPhysicalMemoryInvalidationCallback(GuestWritten, nullptr);
      g_watch_registered.store(true, std::memory_order_release);
      REXGPU_INFO("PGR4 D3D12: write watches registered (guest memory {})", static_cast<void *>(memory));
    }
  }
  return true;
}

void Unwatch(PageWatch &w) {
  if (!w.end) {
    return;
  }
  auto &list = watch_list();
  std::lock_guard lock(list.mutex);
  for (uint32_t bucket = w.start >> kWatchBucketShift; bucket <= (w.end - 1) >> kWatchBucketShift; ++bucket) {
    auto &v = list.buckets[bucket];
    v.erase(std::remove(v.begin(), v.end(), &w), v.end());
  }
  w.start = w.end = 0;
}

// Arms the watch before the data is read, so a write while it is copied
// marks it again rather than being missed.
void Watch(PageWatch &w, uint32_t start, uint32_t size) {
  if (w.start != start || w.end != start + size) {
    Unwatch(w);
    auto &list = watch_list();
    std::lock_guard lock(list.mutex);
    for (uint32_t bucket = start >> kWatchBucketShift; bucket <= (start + size - 1) >> kWatchBucketShift; ++bucket) {
      list.buckets[bucket].push_back(&w);
    }
    w.start = start;
    w.end = start + size;
  }
  w.written.store(false, std::memory_order_release);
  REX_KERNEL_MEMORY()->EnablePhysicalMemoryAccessCallbacks(start, size, true, false);
}

// Also true when the first write came from a frame the game started after
// the one being recorded (render_ahead): that frame's data is still what was
// copied. The watch stays marked, so the next frame reads the new data.
bool Unwritten(const PageWatch &w) {
  if (!w.end) {
    return false;
  }
  return !w.written.load(std::memory_order_acquire) ||
         w.written_frame.load(std::memory_order_relaxed) > g_recorded_frame.load(std::memory_order_relaxed);
}

// Guest physical address of host guest memory, UINT32_MAX outside the
// physical heaps (which cannot be watched).
uint32_t PhysicalOf(const uint8_t *p) {
  auto *memory = REX_KERNEL_MEMORY();
  const uint8_t *base = memory->physical_membase();
  if (p >= base && p < base + 0x20000000) {
    return uint32_t(p - base);
  }
  return memory->GetPhysicalAddress(memory->HostToGuestVirtual(p));
}

bool Threaded();

// Vertex streams seen changing between frames, by guest physical address.
// With the render thread the game rewrites them while queued draws still
// need the old contents (black cable wedges, crowd slabs on the Steam Deck),
// so draws copy them when queued (CopyDynamicStreams). Entries lapse after
// a while unchanged.
struct DynamicStreams {
  std::shared_mutex mutex;
  std::unordered_map<uint32_t, std::pair<uint32_t, uint64_t>> streams; // address -> size, frame last changed
};
constexpr uint64_t kDynamicStreamFrames = 120;

DynamicStreams &dynamic_streams() {
  static DynamicStreams d;
  return d;
}

void NoteDynamicStream(uint32_t address, uint32_t size, uint64_t frame) {
  auto &d = dynamic_streams();
  std::unique_lock lock(d.mutex);
  d.streams[address] = {size, frame};
}

// The size to copy for a stream at address, 0 when it is not dynamic.
uint32_t DynamicStreamSize(uint32_t address, uint64_t frame) {
  auto &d = dynamic_streams();
  std::shared_lock lock(d.mutex);
  const auto it = d.streams.find(address);
  return it != d.streams.end() && frame - it->second.second < kDynamicStreamFrames ? it->second.first : 0;
}

// identity: where the data lives in guest memory when data is a queued copy
// of it, so the cache entry stays the same across frames.
const HostBuffer *BufferFor(RendererState &s, const uint8_t *data, uint32_t size, uint32_t endian,
                            const HalfSwap &halves = {}, const uint8_t *identity = nullptr) {
  if (!data || !size) {
    return nullptr;
  }
  const bool copied = identity && identity != data;
  if (!identity) {
    identity = data;
  }
  const uint64_t key = ((uint64_t(uintptr_t(identity)) << 2) ^ (uint64_t(size) << 40) ^ endian) +
                       halves.dword_mask * 0x9E3779B97F4A7C15ull + (uint64_t(halves.stride) << 50);
  auto &hit = s.buffer_hits[(key ^ (key >> 17) ^ (key >> 37)) & (s.buffer_hits.size() - 1)];
  if (hit.frame == s.frame && hit.key == key && hit.buffer) {
    return hit.buffer;
  }
  const auto lookup_t0 = ProfNow();
  auto &b = s.buffers[key];
  g_stream_stats.lookup_ns += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                           ProfNow() - lookup_t0)
                                           .count());
  // Between full hashes of a big buffer, a sample of it each frame catches
  // the game rewriting it in place (stale vertices drew stray black shapes).
  static const bool late_fill = REXCVAR_GET(late_buffer_fill);
  if (late_fill && !copied && b && b->dynamic && b->size == size) {
    if (b->late_frame != s.frame) {
      auto buffer = TakePooledBuffer(s, b->capacity);
      if (!buffer || !static_cast<plume::D3D12Buffer *>(buffer.get())->d3d) {
        return nullptr;
      }
      void *mapped = buffer->map();
      if (!mapped) {
        return nullptr;
      }
      s.late_fills.push_back({buffer.get(), static_cast<uint8_t *>(mapped), data, size, endian, halves.stride,
                              halves.dword_mask});
      RecycleBuffer(std::move(b->buffer), b->capacity);
      b->buffer = std::move(buffer);
      b->id = s.next_buffer_id++;
      b->late_frame = s.frame;
    }
    hit = {key, s.frame, b.get()};
    return b.get();
  }
  static const bool recheck = REXCVAR_GET(buffer_recheck);
  const uint32_t physical = WatchesEnabled() || Threaded() ? PhysicalOf(identity) : UINT32_MAX;
  // A copy's contents are compared, not trusted to the page watch.
  const bool watched = WatchesEnabled() && !copied && physical != UINT32_MAX && physical + uint64_t(size) <= 0x20000000;
  if (watched) {
    if (b && b->buffer && b->size == size && Unwritten(b->watch)) {
      b->used_frame = s.frame;
      hit = {key, s.frame, b.get()};
      return b.get();
    }
    if (!b) {
      b = std::make_unique<HostBuffer>();
    }
    Watch(b->watch, physical, size);
  } else
  if (!copied && b && b->buffer && b->size == size && s.frame - b->validated_frame < BufferRevalidateInterval(size) &&
      (!recheck || b->sampled_frame == s.frame || TextureSampleHash(data, size) == b->sample_hash)) {
    b->sampled_frame = s.frame;
    hit = {key, s.frame, b.get()};
    return b.get();
  }
  const auto hash_t0 = ProfNow();
  const uint64_t hash = XXH3_64bits(data, size);
  g_stream_stats.hash_ns += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                         ProfNow() - hash_t0)
                                         .count());
  ++g_stream_stats.hashed;
  g_stream_stats.hashed_bytes += size;
  if (b && b->buffer && b->size == size && b->content_hash == hash) {
    b->validated_frame = s.frame;
    b->used_frame = s.frame;
    b->sample_hash = TextureSampleHash(data, size);
    hit = {key, s.frame, b.get()};
    return b.get();
  }
  const auto fill_t0 = ProfNow();
  const uint32_t capacity = BufferSizeClass(size);
  auto buffer = TakePooledBuffer(s, capacity);
  // Plume returns a buffer object even when the allocation itself failed
  // (out of video memory); its resource is then null and map() would crash.
  if (!buffer || !static_cast<plume::D3D12Buffer *>(buffer.get())->d3d) {
    return nullptr;
  }
  void *mapped = buffer->map();
  if (!mapped) {
    return nullptr;
  }
  g_stream_stats.created_bytes += size;
  SwapToUpload(static_cast<uint8_t *>(mapped), data, size, endian, halves);
  buffer->unmap();
  if (b && b->buffer) {
    b->dynamic = recheck;
    if (Threaded() && physical != UINT32_MAX && REXCVAR_GET(dynamic_stream_copy)) {
      NoteDynamicStream(physical, size, g_game_frame.load(std::memory_order_relaxed));
    }
    RecycleBuffer(std::move(b->buffer), b->capacity);
  } else if (!b) {
    b = std::make_unique<HostBuffer>();
  }
  b->buffer = std::move(buffer);
  g_stream_stats.fill_ns += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                         ProfNow() - fill_t0)
                                         .count());
  b->capacity = capacity;
  b->id = s.next_buffer_id++;
  b->size = size;
  b->content_hash = hash;
  b->sample_hash = TextureSampleHash(data, size);
  b->validated_frame = s.frame;
  hit = {key, s.frame, b.get()};
  return b.get();
}

// Fills the frame's late buffers from guest memory (RendererState::late_fills).
void FlushLateFills(RendererState &s) {
  for (const auto &f : s.late_fills) {
    HalfSwap h;
    h.stride = f.half_stride;
    h.dword_mask = f.half_mask;
    SwapToUpload(f.mapped, f.src, f.size, f.endian, h);
    f.buffer->unmap();
  }
  s.late_fills.clear();
}

// ---------------------------------------------------------------------------
// Textures

plume::RenderSwizzle ToSwizzle(uint32_t component) {
  switch (component) {
  case 0:
    return plume::RenderSwizzle::R;
  case 1:
    return plume::RenderSwizzle::G;
  case 2:
    return plume::RenderSwizzle::B;
  case 3:
    return plume::RenderSwizzle::A;
  case 4:
    return plume::RenderSwizzle::ZERO;
  default:
    return plume::RenderSwizzle::ONE;
  }
}

// Components a guest texture format stores. Xenos repeats the last stored
// component into the missing ones (k_8 reads RRRR, k_8_8 RGGG, k_5_6_5
// RGBB), so the swizzle is folded onto them before it reaches the view.
uint32_t GuestComponentCount(uint32_t format) {
  switch (format) {
  case 0: case 1: case 2: case 8: case 9: case 22: case 23: case 24: case 30: case 36: case 39:
  case 41: case 43: case 44: case 46: case 47: case 58: case 59:
    return 1;
  case 10: case 25: case 31: case 37: case 40: case 42: case 45: case 48: case 49: case 60:
    return 2;
  case 4: case 5: case 16: case 17: case 55: case 56: case 57:
    return 3;
  default:
    return 4;
  }
}

plume::RenderComponentMapping SwizzleMapping(uint32_t swizzle, uint32_t format) {
  const uint32_t last = GuestComponentCount(format) - 1;
  auto component = [&](uint32_t c) { return ToSwizzle(c < 4 ? std::min(c, last) : c); };
  return plume::RenderComponentMapping(component(swizzle & 7), component((swizzle >> 3) & 7),
                                       component((swizzle >> 6) & 7), component((swizzle >> 9) & 7));
}

struct TextureFetch {
  uint32_t dword[6];
  uint32_t Type() const { return dword[0] & 3; }
  bool Tiled() const { return (dword[0] >> 31) != 0; }
  uint32_t PitchTexels() const { return ((dword[0] >> 22) & 0x1FF) * 32; }
  bool Signed() const { return ((dword[0] >> 2) & 3) == 1; }
  uint32_t Format() const { return dword[1] & 0x3F; }
  uint32_t Endian() const { return (dword[1] >> 6) & 3; }
  uint32_t BaseAddress() const { return dword[1] & 0xFFFFF000u; }
  uint32_t Dimension() const { return (dword[5] >> 9) & 3; } // 0 1D, 1 2D, 2 3D, 3 cube
  bool PackedMips() const { return ((dword[5] >> 11) & 1) != 0; }
  uint32_t Swizzle() const { return (dword[3] >> 1) & 0xFFF; }
  uint32_t Width() const {
    return Dimension() == 0 ? (dword[2] & 0xFFFFFF) + 1
                            : Dimension() == 2 ? (dword[2] & 0x7FF) + 1 : (dword[2] & 0x1FFF) + 1;
  }
  uint32_t Height() const {
    return Dimension() == 0 ? 1
                            : Dimension() == 2 ? ((dword[2] >> 11) & 0x7FF) + 1
                                               : ((dword[2] >> 13) & 0x1FFF) + 1;
  }
  uint32_t Depth() const { return Dimension() == 2 ? ((dword[2] >> 22) & 0x3FF) + 1 : 1; }
  // Stacked 2D textures (2D fetch, sampled by tfetch3D): slices of 2D images.
  uint32_t StackDepth() const { return Dimension() == 1 ? ((dword[2] >> 26) & 0x3F) + 1 : 1; }
  // Sampler state bits (clamp, filters, aniso, border). Dword 4's mip
  // range and LOD bias stay out: every texture size has its own mip count,
  // and the extra variants filled the 2048-sampler heap (draws dropped).
  uint32_t SamplerKey() const {
    return ((dword[0] >> 10) & 0x1FF) | (((dword[3] >> 19) & 0x1FF) << 9) | ((dword[5] & 3) << 18);
  }
};

TextureFetch ReadTextureFetch(const uint8_t *p) {
  TextureFetch f;
  for (int i = 0; i < 6; ++i) {
    f.dword[i] = LoadBE32(p + i * 4);
  }
  return f;
}

plume::RenderTextureAddressMode ToAddressMode(uint32_t clamp) {
  switch (clamp) {
  case 0:
    return plume::RenderTextureAddressMode::WRAP;
  case 1:
    return plume::RenderTextureAddressMode::MIRROR;
  case 2:
  case 4:
    return plume::RenderTextureAddressMode::CLAMP;
  case 3:
  case 5:
    return plume::RenderTextureAddressMode::MIRROR_ONCE;
  default:
    return plume::RenderTextureAddressMode::BORDER;
  }
}

// Stable index (into sampler_list) of a fetch's sampler state; 0 is the default.
uint32_t SamplerFor(RendererState &s, const TextureFetch &f) {
  const uint32_t key = f.SamplerKey();
  auto &slot = s.samplers[key];
  if (!slot.sampler) {
    plume::RenderSamplerDesc d;
    d.addressU = ToAddressMode(key & 7);
    d.addressV = ToAddressMode((key >> 3) & 7);
    d.addressW = ToAddressMode((key >> 6) & 7);
    const uint32_t mag = (key >> 9) & 3, min = (key >> 11) & 3, mip = (key >> 13) & 3;
    d.magFilter = mag == 0 ? plume::RenderFilter::NEAREST : plume::RenderFilter::LINEAR;
    d.minFilter = min == 0 ? plume::RenderFilter::NEAREST : plume::RenderFilter::LINEAR;
    d.mipmapMode = mip == 0 ? plume::RenderMipmapMode::NEAREST : plume::RenderMipmapMode::LINEAR;
    // Mip filter 2 (basemap) reads only the base level.
    if (mip == 2) {
      d.maxLOD = 0.0f;
    }
    const uint32_t aniso = (key >> 15) & 7;
    if (aniso > 1 && aniso < 7) {
      d.anisotropyEnabled = true;
      d.maxAnisotropy = 1u << (aniso - 1);
    }
    // Fetch dword 5 bits 0-1: 0 ABGR black, 1 ABGR white, 2/3 the YCbCr
    // blacks (opaque). PGR4's ground shadows sample their maps with a white
    // border so everything outside the shadow frustum stays lit.
    switch ((key >> 18) & 3) {
    case 0:
      d.borderColor = plume::RenderBorderColor::TRANSPARENT_BLACK;
      break;
    case 1:
      d.borderColor = plume::RenderBorderColor::OPAQUE_WHITE;
      break;
    default:
      d.borderColor = plume::RenderBorderColor::OPAQUE_BLACK;
      break;
    }
    slot.sampler = s.device->createSampler(d);
    if (!slot.sampler) {
      s.samplers.erase(key);
      return 0;
    }
    slot.index = uint32_t(s.sampler_list.size());
    s.sampler_list.push_back(slot.sampler.get());
  }
  return slot.index;
}

// Host copy of a resolved texture, viewed with the fetch's swizzle. Resolves
// to ARGB-ordered formats swap red and blue in memory, which the fetch
// swizzle (ZYXW) undoes; the host copy is never swapped, so that pair cancels.
BoundTexture BindResolved(RendererState &s, ResolvedTexture &r, const TextureFetch &f) {
  uint32_t swizzle = f.Swizzle();
  if (!r.depth && (swizzle & 7) == 2 && ((swizzle >> 6) & 7) == 0) {
    swizzle = (swizzle & ~0x1C7u) | 0 | (2u << 6);
  }
  const uint32_t view_key = swizzle | (f.Format() << 12);
  auto &view = r.views[view_key];
  if (!view) {
    plume::RenderTextureViewDesc vd;
    vd.format = r.format;
    vd.dimension = r.cube ? plume::RenderTextureViewDimension::TEXTURE_CUBE
                          : plume::RenderTextureViewDimension::TEXTURE_2D;
    vd.mipLevels = std::max(1u, uint32_t(std::countr_one(r.written_levels)));
    vd.componentMapping = SwizzleMapping(swizzle, f.Format());
    view = r.texture->createTextureView(vd);
  }
  BoundTexture b;
  if (view) {
    b.texture = r.texture.get();
    b.view = view.get();
    // The view's levels are part of its identity (cached descriptor sets).
    b.id = (uint64_t(view_key) << 36) | (uint64_t(std::countr_one(r.written_levels) & 15) << 32) | r.id;
    b.dimension = r.cube ? plume::RenderTextureViewDimension::TEXTURE_CUBE
                         : plume::RenderTextureViewDimension::TEXTURE_2D;
  }
  return b;
}

// Texture conversion workers. An upload's copies are recorded at once, but
// they read their upload memory only when the list runs, so the untile and
// conversion into that memory runs on these threads while the frame goes on
// recording; Video joins them before each submission (pgr4-native records
// the copies on the workers too, which Plume's single list rules out).
class ConvertPool {
public:
  struct Job {
    TextureLayout layout;
    const uint8_t *src = nullptr;
    uint32_t endian = 0;
    uint8_t *dst = nullptr;
    uint32_t stack = 1;
  };

  static ConvertPool &Get() {
    static ConvertPool pool;
    return pool;
  }

  void Run(const Job &job) {
    if (workers_ == 0) {
      ConvertTexture(job.layout, job.src, job.endian, job.dst, job.stack);
      return;
    }
    {
      std::lock_guard lock(mutex_);
      jobs_.push_back(job);
      ++pending_;
    }
    work_.notify_one();
  }

  // Waits for every job queued so far.
  void Join() {
    std::unique_lock lock(mutex_);
    idle_.wait(lock, [&] { return pending_ == 0; });
  }

private:
  ConvertPool() {
    int n = REXCVAR_GET(texture_workers);
    if (n < 0) {
      n = std::clamp(int(std::thread::hardware_concurrency()) / 2 - 2, 1, 4);
    }
    for (int i = 0; i < n; ++i) {
      std::thread([this, i] { Main(i); }).detach();
    }
    workers_ = n;
    if (n > 0) {
      Video::SetBeforeSubmit([] { ConvertPool::Get().Join(); });
    }
    REXGPU_INFO("PGR4 D3D12: {} texture workers", n);
  }

  void Main(int index) {
    const std::wstring name = L"PGR4 texture " + std::to_wstring(index);
    SetThreadDescription(GetCurrentThread(), name.c_str());
    for (;;) {
      Job job;
      {
        std::unique_lock lock(mutex_);
        work_.wait(lock, [&] { return !jobs_.empty(); });
        job = jobs_.front();
        jobs_.pop_front();
      }
      ConvertTexture(job.layout, job.src, job.endian, job.dst, job.stack);
      {
        std::lock_guard lock(mutex_);
        if (--pending_ == 0) {
          idle_.notify_all();
        }
      }
    }
  }

  std::mutex mutex_;
  std::condition_variable work_, idle_;
  std::deque<Job> jobs_;
  uint32_t pending_ = 0;
  int workers_ = 0;
};

HostTexture *UploadTexture(RendererState &s, const TextureFetch &f, uint64_t key,
                           std::unique_ptr<HostTexture> &slot) {
  const TextureLayout layout =
      DescribeTexture(f.Format(), f.Signed(), f.Dimension(), f.Width(), f.Height(), f.Depth(),
                      f.PitchTexels(), f.Tiled(), f.PackedMips());
  if (!layout.supported) {
    if (LogOnce(s, 0x2000000000ull | f.Format())) {
      REXGPU_WARN("PGR4 D3D12: unsupported texture format {} ({}x{})", f.Format(), f.Width(),
                  f.Height());
    }
    return nullptr;
  }
  // A stack of 2D slices (the paint's chip and rain-drop maps) is a 3D texture
  // on the host: tfetch3D samples it, slice by z.
  const uint32_t stack = f.StackDepth();
  const bool stacked = stack > 1;
  const uint32_t guest_bytes = stacked ? layout.face_bytes * stack : layout.guest_size;
  const uint32_t base = f.BaseAddress();
  if (!base || uint64_t(base) + guest_bytes > 0x20000000ull) {
    return nullptr;
  }
  const uint8_t *src = GuestPhysical(base);
  // A texture rewritten in the last 60 frames (a movie's planes) is re-hashed
  // instead: re-arming its watch every frame, and the faults of the writes
  // that follow, cost more than the hash.
  if (WatchesEnabled()) {
    if (!slot) {
      slot = std::make_unique<HostTexture>();
    }
    if (s.frame - slot->changed_frame < 60) {
      Unwatch(slot->watch);
    } else {
      Watch(slot->watch, base, guest_bytes);
    }
  }
  const uint64_t hash = XXH3_64bits(src, guest_bytes);
  if (slot && slot->texture && slot->content_hash == hash) {
    slot->validated_frame = s.frame;
    slot->sample_hash = TextureSampleHash(src, guest_bytes);
    return slot.get();
  }

  plume::RenderCommandList *cmd = Video::OpenCommandList();
  if (!cmd) {
    return nullptr;
  }
  const bool cube = f.Dimension() == 3;
  const bool volume = f.Dimension() == 2;
  const bool host_3d = volume || stacked;

  // Mip levels 1+ of 2D textures, from the fetch's mip address: stored at
  // power-of-two sizes, one after another until the shortest side reaches 16
  // texels, then all in one packed tail tile. Without them detailed paint
  // textures (dirt, flakes) alias into a weave when minified.
  struct MipLevel {
    TextureLayout layout;
    uint32_t address = 0;
    uint32_t face_stride = 0; // cube maps: bytes between a level's faces
  };
  std::vector<MipLevel> mips;
  const uint32_t mip_address = f.dword[5] & 0xFFFFF000u;
  const uint32_t max_level = (f.dword[4] >> 6) & 0xF;
  // Volumes (the paint's rain-drop map) get the levels above the tail. Cube
  // maps store each level's six faces one after another (a packed tail per
  // face): matte surfaces (the tracks' buildings and tunnel walls) sample the
  // blurry low levels, and without them got the sharp base level's lights.
  if (!stacked && mip_address && max_level > 0 && !(REXCVAR_GET(gpu_safe) & 16)) {
    const uint32_t w = f.Width(), h = f.Height(), dp = volume ? layout.depth : 1;
    const uint32_t pw = std::bit_ceil(w), ph = std::bit_ceil(h), pd = std::bit_ceil(dp);
    const uint32_t dim = volume ? 2 : 1;
    const uint32_t host_levels = uint32_t(std::bit_width(std::max({w, h, dp})));
    uint32_t offset = 0;
    for (uint32_t level = 1; level <= max_level && level < host_levels; ++level) {
      const TextureLayout storage =
          DescribeTexture(f.Format(), f.Signed(), dim, std::max(1u, pw >> level), std::max(1u, ph >> level),
                          std::max(1u, pd >> level), 0, f.Tiled(), false);
      MipLevel m;
      m.layout = DescribeTexture(f.Format(), f.Signed(), dim, std::max(1u, w >> level), std::max(1u, h >> level),
                                 std::max(1u, dp >> level), storage.pitch_blocks * storage.block, f.Tiled(), false);
      uint32_t ox = 0, oy = 0;
      const bool packed = f.PackedMips() && PackedMipOffset(pw, ph, level, storage.block, ox, oy);
      if (packed && volume) {
        break; // volume tails also pack along z
      }
      m.layout.offset_x = ox;
      m.layout.offset_y = oy;
      if (packed) {
        // The tail tile is 32x32 blocks.
        m.layout.pitch_blocks = std::max(m.layout.pitch_blocks, 32u);
        m.layout.height_blocks = std::max(m.layout.height_blocks, 32u);
        m.layout.face_bytes = std::max(m.layout.face_bytes, 32u * 32u * m.layout.guest_block_bytes);
      }
      m.address = mip_address + offset;
      m.face_stride = packed ? m.layout.face_bytes : storage.face_bytes;
      if (!m.layout.supported ||
          uint64_t(m.address) + uint64_t(m.face_stride) * (cube ? 6 : 1) > 0x20000000ull) {
        break;
      }
      mips.push_back(m);
      if (!packed) {
        offset += storage.face_bytes * (cube ? 6 : 1);
      }
    }
  }
  const uint32_t levels = 1 + uint32_t(mips.size());

  plume::RenderTextureDesc desc = plume::RenderTextureDesc::Texture(
      host_3d ? plume::RenderTextureDimension::TEXTURE_3D : plume::RenderTextureDimension::TEXTURE_2D,
      layout.host_width, layout.host_height, volume ? layout.depth : stack, levels, cube ? 6 : 1,
      layout.host_format, cube ? plume::RenderTextureFlag::CUBE : plume::RenderTextureFlag::NONE);
  auto texture = s.device->createTexture(desc);
  if (!texture) {
    return nullptr;
  }
  texture->setName(fmt::format("texture {:08X} {}x{} fmt {} levels {}", f.BaseAddress(), f.Width(), f.Height(),
                               f.Format(), levels));
  plume::RenderTextureViewDesc vd;
  vd.format = layout.host_format;
  vd.dimension = cube ? plume::RenderTextureViewDimension::TEXTURE_CUBE
                      : host_3d ? plume::RenderTextureViewDimension::TEXTURE_3D
                                : plume::RenderTextureViewDimension::TEXTURE_2D;
  vd.mipLevels = levels;
  vd.componentMapping = SwizzleMapping(f.Swizzle(), f.Format());
  auto view = texture->createTextureView(vd);
  if (!view) {
    return nullptr;
  }

  const uint32_t layers = cube ? 6 : 1;
  const uint32_t slices = volume ? layout.depth : stack;
  const uint32_t image_bytes = layout.host_row_pitch * layout.host_rows;
  // Each layer's footprint must start on a 512-byte boundary.
  const uint32_t layer_bytes = (image_bytes * slices + 511) & ~511u;
  uint32_t mips_bytes = 0;
  std::vector<uint32_t> mip_offsets; // per level, then per face
  for (const MipLevel &m : mips) {
    for (uint32_t layer = 0; layer < layers; ++layer) {
      mip_offsets.push_back(layer_bytes * layers + mips_bytes);
      mips_bytes += (m.layout.host_row_pitch * m.layout.host_rows * m.layout.depth + 511) & ~511u;
    }
  }
  UploadAllocation up = Video::Upload(nullptr, layer_bytes * layers + mips_bytes, 512);
  if (!up) {
    return nullptr;
  }
  ConvertPool &pool = ConvertPool::Get();
  for (size_t i = 0; i < mips.size(); ++i) {
    for (uint32_t layer = 0; layer < layers; ++layer) {
      pool.Run({mips[i].layout, GuestPhysical(mips[i].address + layer * mips[i].face_stride), f.Endian(),
                static_cast<uint8_t *>(up.memory) + mip_offsets[i * layers + layer], 1});
    }
  }
  for (uint32_t layer = 0; layer < layers; ++layer) {
    pool.Run({layout, src + size_t(layer) * layout.face_bytes, f.Endian(),
              static_cast<uint8_t *>(up.memory) + size_t(layer) * layer_bytes, stack});
  }

  cmd->barriers(plume::RenderBarrierStage::COPY,
                plume::RenderTextureBarrier(texture.get(), plume::RenderTextureLayout::COPY_DEST));
  for (uint32_t layer = 0; layer < layers; ++layer) {
    cmd->copyTextureRegion(
        plume::RenderTextureCopyLocation::Subresource(texture.get(), 0, layer),
        plume::RenderTextureCopyLocation::PlacedFootprint(
            up.buffer, layout.host_format, layout.host_width, layout.host_height, slices,
            layout.host_row_pitch / layout.host_block_bytes * layout.block,
            up.offset + uint64_t(layer) * layer_bytes));
  }
  for (size_t i = 0; i < mips.size(); ++i) {
    const TextureLayout &ml = mips[i].layout;
    for (uint32_t layer = 0; layer < layers; ++layer) {
      cmd->copyTextureRegion(
          plume::RenderTextureCopyLocation::Subresource(texture.get(), uint32_t(i + 1), layer),
          plume::RenderTextureCopyLocation::PlacedFootprint(up.buffer, ml.host_format, ml.host_width, ml.host_height,
                                                            ml.depth, ml.host_row_pitch / ml.host_block_bytes * ml.block,
                                                            up.offset + mip_offsets[i * layers + layer]));
    }
  }
  cmd->barriers(plume::RenderBarrierStage::GRAPHICS,
                plume::RenderTextureBarrier(texture.get(), plume::RenderTextureLayout::SHADER_READ));

  if (slot && slot->texture) {
    Video::ParkUntilFence(std::move(slot->view));
    Video::ParkUntilFence(std::move(slot->texture));
    slot->changed_frame = s.frame;
  } else if (!slot) {
    slot = std::make_unique<HostTexture>();
  }
  slot->texture = std::move(texture);
  slot->view = std::move(view);
  slot->dimension = vd.dimension;
  slot->width = f.Width();
  slot->height = f.Height();
  slot->id = s.next_id++;
  slot->guest_size = guest_bytes;
  slot->content_hash = hash;
  slot->sample_hash = TextureSampleHash(src, guest_bytes);
  slot->validated_frame = s.frame;
  (void)key;
  return slot.get();
}

// Resolved copies are keyed by physical address. Texture objects hold the
// virtual address of their memory, while the fetch constants the device
// samples with hold the physical one; the 0xE0000000 heap maps 4 KB higher
// (0xF1E47000 is physical 0x11E48000).
uint32_t PhysicalKey(uint32_t address) {
  return (address & 0x1FFFFFFFu) + (address >= 0xE0000000u ? 0x1000u : 0u);
}

// Sample of the guest memory a resolve covers: enough to notice the game
// loading other data there, cheap enough to check once per frame. Textures
// start 4 KB aligned and take at least 4 KB, so the sample never leaves the
// texture's own memory.
uint64_t GuestSampleHash(uint32_t physical) {
  constexpr uint32_t kSample = 4 * 1024;
  if (!physical || uint64_t(physical) + kSample > 0x20000000ull) {
    return 0;
  }
  return XXH3_64bits(GuestPhysical(physical), kSample);
}

// The resolved copy for a fetch, or null once the game has reused its memory.
ResolvedTexture *LiveResolved(RendererState &s, uint32_t base) {
  auto it = s.latest_resolved.find(PhysicalKey(base));
  if (it == s.latest_resolved.end()) {
    return nullptr;
  }
  ResolvedTexture *r = it->second;
  if (r->checked_frame != s.frame) {
    r->checked_frame = s.frame;
    if (GuestSampleHash(r->guest_base) != r->guest_hash) {
      s.latest_resolved.erase(it);
      ++s.binding_generation;
      return nullptr;
    }
  }
  return r;
}

BoundTexture BindTexture(RendererState &s, const TextureFetch &f) {
  if (f.Type() != 2) {
    return {};
  }
  const int32_t safe = REXCVAR_GET(gpu_safe);
  if ((s.hide_texture && f.BaseAddress() == s.hide_texture) || (safe & 1) ||
      ((safe & 32) && f.Dimension() >= 2)) {
    switch (f.Dimension()) {
    case 2:
      return {s.dummy_3d.get(), s.dummy_3d_view.get(), 0, plume::RenderTextureViewDimension::TEXTURE_3D};
    case 3:
      return {s.dummy_cube.get(), s.dummy_cube_view.get(), 0, plume::RenderTextureViewDimension::TEXTURE_CUBE};
    default:
      return {s.dummy_2d.get(), s.dummy_2d_view.get(), 0, plume::RenderTextureViewDimension::TEXTURE_2D};
    }
  }
  if (s.dumping) {
    const int32_t exp_adjust = int32_t(f.dword[3] << 13) >> 26;
    if (exp_adjust && LogOnce(s, 0x6000000000ull ^ (uint64_t(f.BaseAddress()) << 6) ^ uint32_t(exp_adjust & 63))) {
      REXGPU_INFO("PGR4 D3D12 dump: texture {:08X} fmt {} exp_adjust {}", f.BaseAddress(), f.Format(), exp_adjust);
    }
  }
  if (ResolvedTexture *resolved = (safe & 2) ? nullptr : LiveResolved(s, f.BaseAddress())) {
    if (s.dumping) {
      REXGPU_INFO("PGR4 D3D12 dump:   tex {:08X} fmt {} {}x{} swz {:03o} -> resolved{} {}x{} sampler {:06X}",
                  f.BaseAddress(), f.Format(), f.Width(), f.Height(), f.Swizzle(), resolved->depth ? " depth" : "",
                  resolved->width, resolved->height, f.SamplerKey());
    }
    if (resolved->scale.x != resolved->scale.y) {
      s.draw_samples_wide = true;
    }
    return BindResolved(s, *resolved, f);
  }
  if (s.dumping) {
    REXGPU_INFO("PGR4 D3D12 dump:   tex {:08X} fmt {} {}x{}x{} dim {} swz {:03o} sampler {:06X}", f.BaseAddress(),
                f.Format(), f.Width(), f.Height(), std::max(f.Depth(), f.StackDepth()), f.Dimension(), f.Swizzle(),
                f.SamplerKey());
  }
  // Debug: PGR4_DUMP_RAW=<w>x<h>:<format> writes the raw guest bytes of every
  // matching texture bound during a dump (dumpraw_<base>.bin, fetch logged).
  // Also from a dump.raw file next to the exe, read during dumps.
  static const char *dump_raw_env = std::getenv("PGR4_DUMP_RAW");
  char dump_raw_file[64] = {};
  if (s.dumping && !dump_raw_env) {
    if (std::FILE *rf = std::fopen("dump.raw", "rb")) {
      std::fgets(dump_raw_file, sizeof(dump_raw_file), rf);
      std::fclose(rf);
    }
  }
  const char *dump_raw_spec = dump_raw_env ? dump_raw_env : dump_raw_file;
  if (s.dumping && dump_raw_spec[0]) {
    unsigned w = 0, h = 0, fmt = 0;
    if (std::sscanf(dump_raw_spec, "%ux%u:%u", &w, &h, &fmt) == 3 && f.Width() == w && f.Height() == h &&
        f.Format() == fmt && LogOnce(s, (0xA000000000ull | f.BaseAddress()) ^ (s.frame << 42))) {
      const TextureLayout l = DescribeTexture(f.Format(), f.Signed(), f.Dimension(), f.Width(), f.Height(),
                                              f.Depth(), f.PitchTexels(), f.Tiled(), f.PackedMips());
      char name[64];
      std::snprintf(name, sizeof(name), "dumpraw_%08X.bin", f.BaseAddress());
      if (FILE *out = std::fopen(name, "wb")) {
        std::fwrite(GuestPhysical(f.BaseAddress()), 1, l.guest_size, out);
        std::fclose(out);
      }
      // The same bytes through each virtual alias of the physical page, to
      // check every view sees the same memory.
      const uint32_t phys = f.BaseAddress();
      const uint32_t aliases[] = {0xA0000000u | phys, 0xC0000000u | phys, 0xE0000000u + phys - 0x1000u};
      for (const uint32_t va : aliases) {
        const uint8_t *alias = REX_KERNEL_MEMORY()->TranslateVirtual(va);
        REXGPU_INFO("PGR4 D3D12 dump: raw {:08X} via {:08X}: {} ({:016X} vs physical {:016X})", phys, va,
                    std::memcmp(alias, GuestPhysical(phys), l.guest_size) == 0 ? "same" : "DIFFERENT",
                    XXH3_64bits(alias, l.guest_size), XXH3_64bits(GuestPhysical(phys), l.guest_size));
      }
      REXGPU_INFO("PGR4 D3D12 dump: raw {} fetch {:08X} {:08X} {:08X} {:08X} {:08X} {:08X} ({} bytes)", name,
                  f.dword[0], f.dword[1], f.dword[2], f.dword[3], f.dword[4], f.dword[5], l.guest_size);
    }
  }
  // Data-describing bits only: sampler state lives in the same dwords.
  const uint64_t key = XXH3_64bits_withSeed(
      &f.dword[0], 8, (uint64_t(f.dword[0] & ~0x7FC00u) << 32) ^ f.dword[2] ^
                          (uint64_t(f.dword[3] & 0x7FFFFu) << 13) ^
                          (uint64_t(f.dword[5] & 0xFFFFFE00u) << 29));
  auto &slot = s.textures[key];
  HostTexture *t = slot.get();
  // Small textures are re-hashed every frame: skinning matrix palettes and
  // the paint selector's colour palette are rewritten by the CPU. Big ones less often; streaming rewrites
  // are rare. A big one that changed in the last second (a Bink movie's planes)
  // is re-hashed every frame too.
  // Between full hashes, a sample of the memory is checked once a frame, so a
  // texture streamed over an old one (same address and size) shows at once
  // instead of the old image lingering until the next full hash.
  const bool hot = t && (t->guest_size <= 16384 || s.frame - t->changed_frame < 60);
  bool stale = !t || !t->texture || s.frame - t->validated_frame >= (hot ? 1u : 30u);
  // With page watches, a texture no write has touched is current. One whose
  // pages were written (often by other data sharing them) gets the checks
  // above, which re-hash a big texture only every 30 frames.
  if (WatchesEnabled() && t && t->texture && Unwritten(t->watch)) {
    stale = false;
  } else if (!stale && t->sampled_frame != s.frame) {
    t->sampled_frame = s.frame;
    stale = TextureSampleHash(GuestPhysical(f.BaseAddress()), t->guest_size) != t->sample_hash;
  }
  if (stale) {
    t = UploadTexture(s, f, key, slot);
  }
  if (!t) {
    return {};
  }
  t->used_frame = s.frame;
  // Debug: during a dump, write the host copy of PGR4_DUMP_TEX=<hex base>,
  // or with PGR4_DUMP_TEX=thin of every 8888 texture at most 4 texels wide
  // (lookup tables such as the car paint palette).
  static const char *dump_tex_env = std::getenv("PGR4_DUMP_TEX");
  static const bool dump_thin = dump_tex_env && std::strcmp(dump_tex_env, "thin") == 0;
  static const uint32_t dump_tex = dump_tex_env && !dump_thin ? uint32_t(std::strtoul(dump_tex_env, nullptr, 16)) : 0u;
  const bool dump_this = dump_thin ? (f.Format() == 6 && std::min(f.Width(), f.Height()) <= 4)
                                   : (dump_tex && f.BaseAddress() == dump_tex);
  if (s.dumping && dump_this && t->dimension == plume::RenderTextureViewDimension::TEXTURE_2D &&
      LogOnce(s, 0x9000000000ull | f.BaseAddress())) {
    REXGPU_INFO("PGR4 D3D12 dump: texture {:08X} fetch {:08X} {:08X} {:08X} {:08X} {:08X} {:08X}", f.BaseAddress(),
                f.dword[0], f.dword[1], f.dword[2], f.dword[3], f.dword[4], f.dword[5]);
    char name[64];
    std::snprintf(name, sizeof(name), "dumptex_%08X", f.BaseAddress());
    Video::CaptureTexture(t->texture.get(), t->width, t->height, name);
  }
  BoundTexture b;
  b.texture = t->texture.get();
  b.view = t->view.get();
  b.id = t->id;
  b.dimension = t->dimension;
  return b;
}

// ---------------------------------------------------------------------------
// Render targets

plume::RenderFormat ColorTargetFormat(uint32_t color_format) {
  switch (color_format) {
  case 0: // 8_8_8_8
  case 1: // 8_8_8_8_GAMMA
  case 2: // 2_10_10_10
  case 10:
    return plume::RenderFormat::R8G8B8A8_UNORM;
  case 4: // 16_16 (fixed -32..32)
  case 6:
    return plume::RenderFormat::R16G16_FLOAT;
  case 14:
    return plume::RenderFormat::R32_FLOAT;
  case 15:
    return plume::RenderFormat::R32G32_FLOAT;
  default: // 2_10_10_10_FLOAT, 16_16_16_16(_FLOAT), ..._AS_16_16_16_16
    return plume::RenderFormat::R16G16B16A16_FLOAT;
  }
}

// What a guest color format can hold, for the formats whose host stand-in
// holds more: {low, high, alpha high}, high 0 for none. 2_10_10_10_FLOAT is
// 7e3 (0..31.875, 2-bit alpha), the 16_16(_16_16) formats are fixed point
// -32..32, and the float16 ones have no infinity on the Xenos.
std::array<float, 3> OutputRange(uint32_t color_format) {
  switch (color_format) {
  case 3:  // 2_10_10_10_FLOAT
  case 12: // 2_10_10_10_FLOAT_AS_16_16_16_16
    return {0.0f, 31.875f, 1.0f};
  case 4: // 16_16
  case 5: // 16_16_16_16
    return {-32.0f, 32.0f, 32.0f};
  case 6: // 16_16_FLOAT
  case 7: // 16_16_16_16_FLOAT
    return {-65504.0f, 65504.0f, 65504.0f};
  default:
    return {0.0f, 0.0f, 0.0f};
  }
}

constexpr plume::RenderFormat kDepthFormat = plume::RenderFormat::D32_FLOAT_S8_UINT;

// PGR4's D3DSurface: +0x1C color/depth info (format in bits 16-19), +0x24
// packed size ((width-1) << 18 | (height-1) << 3), from sub_826C8260.
RenderTarget *RenderTargetFor(RendererState &s, uint32_t surface, bool depth) {
  if (!surface) {
    return nullptr;
  }
  const uint32_t info = ReadU32(surface + 0x1C);
  const uint32_t size = ReadU32(surface + 0x24);
  uint32_t width = (size >> 18) + 1;
  uint32_t height = ((size >> 3) & 0x7FFF) + 1;
  // Predicated tiling: the surface is one EDRAM tile (e.g. 1280x480 for
  // 720p 2xMSAA), but the scene is drawn once over the whole tiled area.
  if (s.tiling_active) {
    width = std::max(width, s.tiling_width);
    height = std::max(height, s.tiling_height);
  }
  const plume::RenderFormat format = depth ? kDepthFormat : ColorTargetFormat((info >> 16) & 0xF);
  if (s.dumping && !depth && ((info >> 20) & 0x3F) && LogOnce(s, 0x7000000000ull ^ (uint64_t(surface) << 8) ^ (info >> 20))) {
    REXGPU_INFO("PGR4 D3D12 dump: surface {:08X} color format {} exp bias {}", surface, (info >> 16) & 0xF,
                int32_t(info << 6) >> 26);
  }
  // PGR4 reuses surface objects for different targets (a color surface is
  // later a depth surface, sizes change between passes), so every variant
  // keeps its own host texture and contents.
  const uint64_t key = XXH3_64bits_withSeed(&surface, 4, (uint64_t(width) << 40) | (uint64_t(height) << 20) |
                                                            (uint64_t(format) << 1) | (depth ? 1 : 0));
  auto &rt = s.render_targets[key];
  if (rt && rt->width == width && rt->height == height && rt->format == format) {
    rt->edram_base = info & 0xFFF;
    rt->guest_format = (info >> 16) & 0xF; // depth: 24_8 or 24_8_FLOAT
    return rt.get();
  }
  plume::RenderCommandList *cmd = Video::OpenCommandList();
  const Scale2 scale = TargetScale(width, height);
  const uint32_t host_width = ScaledSize(width, scale.x), host_height = ScaledSize(height, scale.y);
  if (!cmd || width > 8192 || height > 8192 || host_width > 16384 || host_height > 16384) {
    return nullptr;
  }
  auto texture = s.device->createTexture(
      depth ? plume::RenderTextureDesc::DepthTarget(host_width, host_height, format)
            : plume::RenderTextureDesc::ColorTarget(host_width, host_height, format));
  if (!texture) {
    return nullptr;
  }
  texture->setName(fmt::format("{} target {:08X} {}x{}", depth ? "depth" : "color", surface, width, height));
  cmd->barriers(plume::RenderBarrierStage::GRAPHICS,
                plume::RenderTextureBarrier(texture.get(), depth ? plume::RenderTextureLayout::DEPTH_WRITE
                                                                 : plume::RenderTextureLayout::COLOR_WRITE));
  if (rt) {
    Video::ParkUntilFence(std::move(rt->texture));
    s.bound_framebuffer = nullptr;
    s.bound = {};
  } else {
    rt = std::make_unique<RenderTarget>();
  }
  rt->texture = std::move(texture);
  rt->format = format;
  rt->width = width;
  rt->height = height;
  rt->scale = scale;
  rt->depth = depth;
  rt->needs_init = true;
  rt->edram_base = info & 0xFFF;
  rt->clear_pending = false;
  rt->guest_format = (info >> 16) & 0xF;
  rt->id = s.next_id++;
  REXGPU_INFO("PGR4 D3D12: {} target {:08X} {}x{} format {}", depth ? "depth" : "color", surface,
              width, height, int(format));
  return rt.get();
}

struct Targets {
  RenderTarget *color[4] = {};
  uint32_t color_count = 0;
  RenderTarget *depth = nullptr;
  uint32_t width = 0;
  uint32_t height = 0;
  Scale2 scale; // host pixels per guest pixel (same for all: same size)
};

Targets CurrentTargets(RendererState &s, uint32_t device) {
  Targets t;
  for (uint32_t i = 0; i < 4; ++i) {
    RenderTarget *rt = RenderTargetFor(s, ReadU32(device + dev::kRenderTargets + i * 4), false);
    if (!rt) {
      break;
    }
    t.color[t.color_count++] = rt;
  }
  t.depth = RenderTargetFor(s, ReadU32(device + dev::kDepthStencil), true);
  if (t.color_count) {
    t.width = t.color[0]->width;
    t.height = t.color[0]->height;
    t.scale = t.color[0]->scale;
  } else if (t.depth) {
    t.width = t.depth->width;
    t.height = t.depth->height;
    t.scale = t.depth->scale;
  }
  return t;
}

plume::RenderFramebuffer *BindFramebuffer(RendererState &s, plume::RenderCommandList *cmd,
                                          const Targets &t) {
  uint64_t key = t.depth ? t.depth->id : 0;
  for (uint32_t i = 0; i < t.color_count; ++i) {
    key = key * 0x100000001B3ull + t.color[i]->id;
  }
  key = key * 31 + t.color_count;
  auto &fb = s.framebuffers[key];
  if (!fb) {
    const plume::RenderTexture *colors[4] = {};
    for (uint32_t i = 0; i < t.color_count; ++i) {
      colors[i] = t.color[i]->texture.get();
    }
    fb = s.device->createFramebuffer(plume::RenderFramebufferDesc(
        colors, t.color_count, t.depth ? t.depth->texture.get() : nullptr));
    if (!fb) {
      return nullptr;
    }
  }
  if (s.bound_framebuffer != fb.get()) {
    cmd->setFramebuffer(fb.get());
    s.bound_framebuffer = fb.get();
  }
  // First use of a new target: give it defined contents (zero color, far
  // depth), as EDRAM would have. --poison_targets fills color with NaN
  // instead, to show on any GPU what reading uninitialized memory does.
  for (uint32_t i = 0; i < t.color_count; ++i) {
    if (t.color[i]->needs_init) {
      const float v = REXCVAR_GET(poison_targets) ? std::numeric_limits<float>::quiet_NaN() : 0.0f;
      cmd->clearColor(i, plume::RenderColor(v, v, v, v));
      t.color[i]->needs_init = false;
    }
  }
  if (t.depth && t.depth->needs_init) {
    cmd->clearDepthStencil(true, true, 1.0f, 0);
    t.depth->needs_init = false;
  }
  // Depth clears made through another view of the same EDRAM.
  if (t.depth && t.depth->clear_pending) {
    cmd->clearDepthStencil(t.depth->clear_depth, t.depth->clear_stencil, t.depth->clear_z,
                           t.depth->clear_stencil_value);
    t.depth->clear_pending = false;
  }
  return fb.get();
}

// A draw through a depth view overwrites its EDRAM, so a clear shared onto
// the other views there no longer stands: the shadow pass clears to 1 and
// then draws, and its clear reaching the reflection cube's reversed-Z depth
// left a cube face empty (red car windows in the menus).
void NoteDepthDraw(RendererState &s, const RenderTarget *depth) {
  if (s.shared_clears.empty() || !depth) {
    return;
  }
  std::erase_if(s.shared_clears, [&](RenderTarget *rt) {
    if (!rt->clear_pending) {
      return true;
    }
    if (rt != depth && rt->edram_base == depth->edram_base) {
      rt->clear_pending = false;
      return true;
    }
    return false;
  });
}

// ---------------------------------------------------------------------------
// Pipeline state

plume::RenderBlend ToBlend(uint32_t factor) {
  using B = plume::RenderBlend;
  switch (factor) {
  case 0:
    return B::ZERO;
  case 1:
    return B::ONE;
  case 4:
    return B::SRC_COLOR;
  case 5:
    return B::INV_SRC_COLOR;
  case 6:
    return B::SRC_ALPHA;
  case 7:
    return B::INV_SRC_ALPHA;
  case 8:
    return B::DEST_COLOR;
  case 9:
    return B::INV_DEST_COLOR;
  case 10:
    return B::DEST_ALPHA;
  case 11:
    return B::INV_DEST_ALPHA;
  case 12: // constant color
  case 14: // constant alpha
    return B::BLEND_FACTOR;
  case 13:
  case 15:
    return B::INV_BLEND_FACTOR;
  case 16:
    return B::SRC_ALPHA_SAT;
  default:
    return B::ONE;
  }
}

plume::RenderBlend ToAlphaBlend(uint32_t factor) {
  using B = plume::RenderBlend;
  switch (ToBlend(factor)) {
  case B::SRC_COLOR:
    return B::SRC_ALPHA;
  case B::INV_SRC_COLOR:
    return B::INV_SRC_ALPHA;
  case B::DEST_COLOR:
    return B::DEST_ALPHA;
  case B::INV_DEST_COLOR:
    return B::INV_DEST_ALPHA;
  default:
    return ToBlend(factor);
  }
}

plume::RenderBlendOperation ToBlendOp(uint32_t op) {
  using O = plume::RenderBlendOperation;
  switch (op) {
  case 1:
    return O::SUBTRACT;
  case 2:
    return O::MIN;
  case 3:
    return O::MAX;
  case 4:
    return O::REV_SUBTRACT;
  default:
    return O::ADD;
  }
}

plume::RenderComparisonFunction ToCompare(uint32_t func) {
  using C = plume::RenderComparisonFunction;
  constexpr C kMap[] = {C::NEVER,   C::LESS,      C::EQUAL,         C::LESS_EQUAL,
                        C::GREATER, C::NOT_EQUAL, C::GREATER_EQUAL, C::ALWAYS};
  return kMap[func & 7];
}

plume::RenderStencilOp ToStencilOp(uint32_t op) {
  using S = plume::RenderStencilOp;
  constexpr S kMap[] = {S::KEEP,   S::ZERO,       S::REPLACE,         S::INCREMENT_AND_WRAP,
                        S::DECREMENT_AND_WRAP, S::INVERT, S::INCREMENT_AND_CLAMP,
                        S::DECREMENT_AND_CLAMP};
  return kMap[op & 7];
}

// Xenos primitive types this renderer draws natively, or rewrites to lists.
enum class Topology { kNone, kPoints, kLines, kLineStrip, kTriangles, kTriangleStrip, kFan, kQuads, kRects,
                      // 32-bit indexed strip: no host cut value for 0xFFFFFFFF, so
                      // it is expanded to a list on the CPU.
                      kStrip32 };

Topology ToTopology(uint32_t primitive) {
  switch (primitive) {
  case 1:
    return Topology::kPoints;
  case 2:
    return Topology::kLines;
  case 3:
    return Topology::kLineStrip;
  case 4:
    return Topology::kTriangles;
  case 5:
    return Topology::kFan;
  case 6:
    return Topology::kTriangleStrip;
  case 8:
    return Topology::kRects;
  case 13:
    return Topology::kQuads;
  default:
    return Topology::kNone;
  }
}

plume::RenderPrimitiveTopology HostTopology(Topology t) {
  switch (t) {
  case Topology::kPoints:
    return plume::RenderPrimitiveTopology::POINT_LIST;
  case Topology::kLines:
    return plume::RenderPrimitiveTopology::LINE_LIST;
  case Topology::kLineStrip:
    return plume::RenderPrimitiveTopology::LINE_STRIP;
  case Topology::kTriangleStrip:
    return plume::RenderPrimitiveTopology::TRIANGLE_STRIP;
  default:
    return plume::RenderPrimitiveTopology::TRIANGLE_LIST;
  }
}

struct PipelineKey {
  uint64_t vs = 0;
  uint64_t ps = 0;
  uint64_t decl = 0;
  uint32_t rt_formats[4] = {};
  uint32_t rt_count = 0;
  uint32_t depth_format = 0;
  uint32_t topology = 0;
  uint32_t blend[4] = {};
  uint32_t color_mask = 0;
  uint32_t depth_control = 0;
  uint32_t stencil_front = 0;
  uint32_t stencil_back = 0;
  uint32_t mode_control = 0;
  float depth_bias = 0;       // PA_SU_POLY_OFFSET_FRONT_OFFSET when enabled
  float depth_bias_slope = 0; // PA_SU_POLY_OFFSET_FRONT_SCALE when enabled
};

plume::RenderPipeline *PipelineFor(RendererState &s, const PipelineKey &key, const HostShader *vs,
                                   const HostShader *ps, const InputBinding &input) {
  const uint64_t hash = XXH3_64bits(&key, sizeof(key));
  auto it = s.pipelines.find(hash);
  if (it != s.pipelines.end()) {
    return it->second.get();
  }
  plume::RenderGraphicsPipelineDesc pd;
  pd.pipelineLayout = s.layout.get();
  pd.vertexShader = vs->shader.get();
  pd.pixelShader = ps ? ps->shader.get() : nullptr;
  if (REXCVAR_GET(debug_solid_ps)) {
    if (!s.solid_ps) {
      s.solid_ps = s.device->createShader(g_solid_ps_dxbc, sizeof(g_solid_ps_dxbc), "main",
                                          plume::RenderShaderFormat::DXBC);
    }
    pd.pixelShader = s.solid_ps.get();
  }
  pd.primitiveTopology = HostTopology(Topology(key.topology));

  const uint32_t mode = key.mode_control;
  const bool cull_front = mode & 1, cull_back = (mode >> 1) & 1;
  pd.cullMode = cull_front && cull_back ? plume::RenderCullMode::NONE
                : cull_front            ? plume::RenderCullMode::FRONT
                : cull_back             ? plume::RenderCullMode::BACK
                                        : plume::RenderCullMode::NONE;
  pd.frontFace = (mode >> 2) & 1 ? plume::RenderFrontFace::CLOCKWISE
                                 : plume::RenderFrontFace::COUNTER_CLOCKWISE;
  pd.depthClipEnabled = true;

  const uint32_t dc = key.depth_control;
  if (key.depth_format) {
    pd.depthTargetFormat = kDepthFormat;
    pd.depthEnabled = (dc >> 1) & 1;
    pd.depthWriteEnabled = pd.depthEnabled && ((dc >> 2) & 1);
    pd.depthFunction = ToCompare(dc >> 4);
    // Xenos polygon offset: the slope term is in 1/16 subpixel units, the
    // constant a fraction of the depth range. On a 24_8_FLOAT target it
    // scales with the depth's exponent, as the host float buffer does with
    // 2^-24 steps; for 24_8 it is applied in the vertex shader (depth_offset).
    pd.depthBias = int32_t(std::lround(double(key.depth_bias) * double(1 << 24)));
    pd.slopeScaledDepthBias = key.depth_bias_slope * (1.0f / 16.0f);
    pd.stencilEnabled = dc & 1;
    pd.stencilReadMask = (key.stencil_front >> 8) & 0xFF;
    pd.stencilWriteMask = (key.stencil_front >> 16) & 0xFF;
    pd.stencilReference = key.stencil_front & 0xFF;
    pd.stencilFrontFace.compareFunction = ToCompare(dc >> 8);
    pd.stencilFrontFace.failOp = ToStencilOp(dc >> 11);
    pd.stencilFrontFace.passOp = ToStencilOp(dc >> 14);
    pd.stencilFrontFace.depthFailOp = ToStencilOp(dc >> 17);
    if ((dc >> 7) & 1) {
      pd.stencilBackFace.compareFunction = ToCompare(dc >> 20);
      pd.stencilBackFace.failOp = ToStencilOp(dc >> 23);
      pd.stencilBackFace.passOp = ToStencilOp(dc >> 26);
      pd.stencilBackFace.depthFailOp = ToStencilOp(dc >> 29);
    } else {
      pd.stencilBackFace = pd.stencilFrontFace;
    }
  }

  pd.renderTargetCount = key.rt_count;
  for (uint32_t i = 0; i < key.rt_count; ++i) {
    pd.renderTargetFormat[i] = plume::RenderFormat(key.rt_formats[i]);
    const uint32_t b = key.blend[i];
    plume::RenderBlendDesc bd;
    bd.srcBlend = ToBlend(b & 0x1F);
    bd.blendOp = ToBlendOp((b >> 5) & 7);
    bd.dstBlend = ToBlend((b >> 8) & 0x1F);
    bd.srcBlendAlpha = ToAlphaBlend((b >> 16) & 0x1F);
    bd.blendOpAlpha = ToBlendOp((b >> 21) & 7);
    bd.dstBlendAlpha = ToAlphaBlend((b >> 24) & 0x1F);
    bd.blendEnabled = !(bd.srcBlend == plume::RenderBlend::ONE && bd.dstBlend == plume::RenderBlend::ZERO &&
                        bd.blendOp == plume::RenderBlendOperation::ADD &&
                        bd.srcBlendAlpha == plume::RenderBlend::ONE &&
                        bd.dstBlendAlpha == plume::RenderBlend::ZERO &&
                        bd.blendOpAlpha == plume::RenderBlendOperation::ADD);
    bd.renderTargetWriteMask = uint8_t((key.color_mask >> (i * 4)) & 0xF);
    if (REXCVAR_GET(debug_no_blend)) {
      bd.blendEnabled = false;
      bd.renderTargetWriteMask = 0xF;
    }
    pd.renderTargetBlend[i] = bd;
  }

  plume::RenderInputSlot slots[kNullStream + 1];
  uint32_t slot_count = 0;
  for (uint32_t i = 0; i <= kNullStream; ++i) {
    if ((input.stream_mask >> i) & 1 || i == kNullStream) {
      slots[slot_count++] = plume::RenderInputSlot(i, 0);
    }
  }
  pd.inputSlots = slots;
  pd.inputSlotsCount = slot_count;
  pd.inputElements = input.elements.data();
  pd.inputElementsCount = uint32_t(input.elements.size());

  auto pipeline = s.device->createGraphicsPipeline(pd);
  if (pipeline && !static_cast<plume::D3D12GraphicsPipeline *>(pipeline.get())->d3d) {
    pipeline.reset();
  }
  if (!pipeline) {
    REXGPU_WARN("PGR4 D3D12: pipeline creation failed (vs {:016X} ps {:016X} rt {} depth {})",
                key.vs, key.ps, key.rt_count, key.depth_format);
  }
  return s.pipelines.emplace(hash, std::move(pipeline)).first->second.get();
}

// ---------------------------------------------------------------------------
// Descriptor tables

// Plume's shader-visible heaps are fixed (262144 views, 2048 samplers), and a
// set created once they are full gets no space: writing it faults inside
// D3D12. Released sets keep their space until their frame's fence retires, so
// unused sets are dropped well before the heap fills, and no set is created
// without room for it.
constexpr uint32_t kSetViewLowWater = 65536;
constexpr uint32_t kSetSamplerLowWater = 512;
constexpr uint32_t kSetViewReserve = 1024;   // left for the present pass
constexpr uint32_t kSetSamplerReserve = 32;

void ReleaseSet(RendererState &s, CachedSet &cached) {
  s.set_views -= cached.views;
  s.set_samplers -= cached.samplers;
  cached.views = cached.samplers = 0;
  ++s.binding_generation;
  for (const plume::RenderDescriptorSet *&bound : s.bound.sets) {
    if (bound == cached.set.get()) {
      bound = nullptr;
    }
  }
  Video::ParkUntilFence(std::move(cached.set));
}

uint32_t FreeViews(RendererState &s) {
  return static_cast<plume::D3D12Device *>(s.device)->viewHeapAllocator->freeSize;
}

uint32_t FreeSamplers(RendererState &s) {
  return static_cast<plume::D3D12Device *>(s.device)->samplerHeapAllocator->freeSize;
}

// Room for a new set, dropping every set not used this frame once space runs low.
bool ReserveSet(RendererState &s, uint32_t views, uint32_t samplers) {
  if ((FreeViews(s) < kSetViewLowWater || FreeSamplers(s) < kSetSamplerLowWater) &&
      s.sets_trimmed_frame != s.frame) {
    s.sets_trimmed_frame = s.frame;
    for (auto *sets : {&s.sets, &s.stream_sets}) {
      for (auto it = sets->begin(); it != sets->end();) {
        if (it->second.used_frame != s.frame) {
          ReleaseSet(s, it->second);
          it = sets->erase(it);
        } else {
          ++it;
        }
      }
    }
  }
  const bool fits = FreeViews(s) >= views + kSetViewReserve && FreeSamplers(s) >= samplers + kSetSamplerReserve;
  if (!fits && LogOnce(s, 0xB000000000ull)) {
    REXGPU_WARN("PGR4 D3D12: descriptor heaps full ({} views, {} samplers free, {} views cached)", FreeViews(s),
                FreeSamplers(s), s.set_views);
  }
  return fits;
}

// Whether plume found heap space for a new set.
// ---------------------------------------------------------------------------
// Command list state, set only when it changes

RendererState::ListBindings &Bindings(RendererState &s) {
  const uint64_t generation = Video::ListGeneration();
  if (s.bound.generation != generation) {
    s.bound = {};
    s.bound.generation = generation;
  }
  return s.bound;
}

bool SameBuffer(const plume::RenderBufferReference &a, const plume::RenderBufferReference &b) {
  return a.ref == b.ref && a.offset == b.offset;
}

void BindPipeline(RendererState &s, plume::RenderCommandList *cmd, const plume::RenderPipeline *pipeline) {
  auto &b = Bindings(s);
  if (!b.layout) {
    cmd->setGraphicsPipelineLayout(s.layout.get());
    b.layout = true;
  }
  if (b.pipeline != pipeline) {
    cmd->setPipeline(pipeline);
    b.pipeline = pipeline;
  }
}

void BindSet(RendererState &s, plume::RenderCommandList *cmd, plume::RenderDescriptorSet *set, uint32_t index) {
  auto &b = Bindings(s);
  if (b.sets[index] != set) {
    cmd->setGraphicsDescriptorSet(set, index);
    b.sets[index] = set;
  }
}

void BindRoot(RendererState &s, plume::RenderCommandList *cmd, const plume::RenderBufferReference &ref,
              uint32_t index) {
  auto &b = Bindings(s);
  if (!SameBuffer(b.roots[index], ref)) {
    cmd->setGraphicsRootDescriptor(ref, index);
    b.roots[index] = ref;
  }
}

void BindVertexStream(RendererState &s, plume::RenderCommandList *cmd, uint32_t index,
                      const plume::RenderVertexBufferView *view, const plume::RenderInputSlot *slot) {
  auto &v = Bindings(s).streams[index];
  if (SameBuffer(v.buffer, view->buffer) && v.size == view->size && v.stride == slot->stride) {
    return;
  }
  cmd->setVertexBuffers(index, view, 1, slot);
  v.buffer = view->buffer;
  v.size = view->size;
  v.stride = slot->stride;
}

void BindViewport(RendererState &s, plume::RenderCommandList *cmd, const plume::RenderViewport &vp) {
  auto &b = Bindings(s);
  if (!b.viewport_valid || std::memcmp(&b.viewport, &vp, sizeof(vp))) {
    cmd->setViewports(vp);
    b.viewport = vp;
    b.viewport_valid = true;
  }
}

void BindScissor(RendererState &s, plume::RenderCommandList *cmd, const plume::RenderRect &rect) {
  auto &b = Bindings(s);
  if (!b.scissor_valid || std::memcmp(&b.scissor, &rect, sizeof(rect))) {
    cmd->setScissors(rect);
    b.scissor = rect;
    b.scissor_valid = true;
  }
}

bool SetAllocated(const plume::RenderDescriptorSet *set) {
  const auto *d3d = static_cast<const plume::D3D12DescriptorSet *>(set);
  return (d3d->viewAllocation.count || d3d->samplerAllocation.count) &&
         d3d->viewAllocation.offset != plume::D3D12DescriptorHeapAllocator::INVALID_OFFSET &&
         d3d->samplerAllocation.offset != plume::D3D12DescriptorHeapAllocator::INVALID_OFFSET;
}

plume::RenderDescriptorSet *TableFor(RendererState &s, uint32_t table, const uint64_t *ids,
                                     const BoundTexture *textures, const plume::RenderSampler *const *samplers,
                                     uint32_t count) {
  const uint64_t key = XXH3_64bits_withSeed(ids, count * sizeof(uint64_t), table);
  auto &cached = s.sets[key];
  cached.used_frame = s.frame;
  if (cached.set) {
    return cached.set.get();
  }
  const uint32_t views = table == 3 ? 0 : count, sampler_count = table == 3 ? count : 0;
  if (!ReserveSet(s, views, sampler_count)) {
    return nullptr;
  }
  plume::RenderDescriptorSetBuilder t2d, t3d, tcube, samp;
  BuildTableLayouts(t2d, t3d, tcube, samp);
  plume::RenderDescriptorSetBuilder &builder = table == 0 ? t2d : table == 1 ? t3d : table == 2 ? tcube : samp;
  cached.set = builder.create(s.device);
  if (!cached.set || !SetAllocated(cached.set.get())) {
    cached.set.reset();
    return nullptr;
  }
  cached.views = views;
  cached.samplers = sampler_count;
  s.set_views += views;
  s.set_samplers += sampler_count;
  for (uint32_t i = 0; i < count; ++i) {
    if (table == 3) {
      cached.set->setSampler(i, samplers[i]);
    } else {
      cached.set->setTexture(i, textures[i].texture, plume::RenderTextureLayout::SHADER_READ,
                             textures[i].view);
    }
  }
  return cached.set.get();
}

// Raw views of the draw's vertex streams for indexed vertex fetch.
// ids are the HostBuffer ids, so a recycled buffer address never matches a
// set built for the buffer it replaced.
plume::RenderDescriptorSet *StreamTableFor(RendererState &s, const plume::RenderBuffer *const *buffers,
                                           const uint64_t *ids) {
  const uint64_t key = XXH3_64bits(ids, kTableStreamCount * sizeof(uint64_t));
  auto &cached = s.stream_sets[key];
  cached.used_frame = s.frame;
  if (cached.set) {
    return cached.set.get();
  }
  ++g_stream_stats.new_stream_sets;
  if (!ReserveSet(s, kTableStreamCount, 0)) {
    return nullptr;
  }
  plume::RenderDescriptorSetBuilder builder;
  builder.begin();
  builder.addByteAddressBuffer(0, kTableStreamCount);
  builder.end();
  cached.set = builder.create(s.device);
  if (!cached.set || !SetAllocated(cached.set.get())) {
    cached.set.reset();
    return nullptr;
  }
  cached.views = kTableStreamCount;
  s.set_views += kTableStreamCount;
  for (uint32_t i = 0; i < kTableStreamCount; ++i) {
    cached.set->setBuffer(i, buffers[i] ? buffers[i] : s.null_stream.get());
  }
  return cached.set.get();
}

void EvictUnused(RendererState &s) {
  for (auto it = s.stream_sets.begin(); it != s.stream_sets.end();) {
    if (s.frame - it->second.used_frame > 2) {
      ReleaseSet(s, it->second);
      it = s.stream_sets.erase(it);
    } else {
      ++it;
    }
  }
  for (auto it = s.sets.begin(); it != s.sets.end();) {
    if (s.frame - it->second.used_frame > 120) {
      ReleaseSet(s, it->second);
      it = s.sets.erase(it);
    } else {
      ++it;
    }
  }
  for (auto it = s.textures.begin(); it != s.textures.end();) {
    if (it->second && s.frame - it->second->used_frame > 600) {
      Unwatch(it->second->watch);
      Video::ParkUntilFence(std::move(it->second->view));
      Video::ParkUntilFence(std::move(it->second->texture));
      it = s.textures.erase(it);
    } else {
      ++it;
    }
  }
}

// ---------------------------------------------------------------------------
// Constants

const UploadAllocation &UploadConstants(RendererState::ConstantSlot &slot, const void *data,
                                        uint32_t size) {
  // The last upload also serves a shorter range with the same leading bytes.
  if (slot.alloc && slot.data.size() >= size && !std::memcmp(slot.data.data(), data, size)) {
    return slot.alloc;
  }
  slot.data.assign(static_cast<const uint8_t *>(data), static_cast<const uint8_t *>(data) + size);
  slot.alloc = Video::Upload(data, size, 256);
  return slot.alloc;
}

// Xenos multiplies with D3D9 legacy rules (0 * anything = 0), so stale
// NaN/Inf constants a shader multiplies by zero never show on hardware; on the
// host they spread (the GNX paint's unset scratch colour turned it olive).
// NaN becomes 0 and Inf the largest finite value of its sign.
// Reads registers c0 up to `regs` (the range the shader declares).
void ReadFloatConstants(uint32_t device, uint32_t offset, uint32_t *out, uint32_t regs) {
  const uint8_t *src = GuestVirtual(device + offset);
  const __m128i swap = _mm_setr_epi8(3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12);
  const __m128i exponent = _mm_set1_epi32(0x7F800000);
  const __m128i mantissa = _mm_set1_epi32(0x007FFFFF);
  const __m128i sign = _mm_set1_epi32(int32_t(0x80000000u));
  const __m128i largest = _mm_set1_epi32(0x7F7FFFFF);
  const __m128i zero = _mm_setzero_si128();
  for (uint32_t i = 0; i < regs; ++i) {
    const __m128i v = _mm_shuffle_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i *>(src + i * 16)), swap);
    const __m128i special = _mm_cmpeq_epi32(_mm_and_si128(v, exponent), exponent);
    __m128i r = v;
    if (!_mm_testz_si128(special, special)) {
      const __m128i nan = _mm_andnot_si128(_mm_cmpeq_epi32(_mm_and_si128(v, mantissa), zero), special);
      const __m128i inf = _mm_andnot_si128(nan, special);
      r = _mm_blendv_epi8(r, _mm_or_si128(_mm_and_si128(v, sign), largest), inf);
      r = _mm_andnot_si128(nan, r);
    }
    _mm_storeu_si128(reinterpret_cast<__m128i *>(out + i * 4), r);
  }
}

// ---------------------------------------------------------------------------
// Draw

struct DrawRequest {
  uint32_t device = 0;
  uint32_t primitive = 0;
  bool indexed = false;
  uint32_t first = 0; // start vertex or start index
  uint32_t count = 0;
  int32_t base_vertex = 0;
  // DrawVerticesUP / BeginVertices data.
  uint32_t up_data = 0;
  uint32_t up_stride = 0;
};

void PrepareFrame(RendererState &s) {
  if (s.frame_prepared) {
    return;
  }
  s.frame_prepared = true;
  s.texture_setups.clear();
  s.bound_framebuffer = nullptr;
  s.bound = {};
  // Vertex and index buffers no draw has used for 300 frames (data the game
  // freed or moved) go back to the pool instead of piling up.
  if (s.frame % 120 == 0) {
    for (auto it = s.buffers.begin(); it != s.buffers.end();) {
      if (s.frame - std::max(it->second->validated_frame, it->second->used_frame) > 300) {
        Unwatch(it->second->watch);
        RecycleBuffer(std::move(it->second->buffer), it->second->capacity);
        it = s.buffers.erase(it);
      } else {
        ++it;
      }
    }
  }
  s.vs_constants.alloc = {};
  s.ps_constants.alloc = {};
  s.vs_shared.alloc = {};
  s.ps_shared.alloc = {};
}

float HalfToFloat(uint16_t h) {
  const uint32_t sign = uint32_t(h & 0x8000) << 16;
  uint32_t exp = (h >> 10) & 0x1F, mant = h & 0x3FF;
  uint32_t bits;
  if (exp == 0) {
    if (!mant) {
      bits = sign;
    } else {
      exp = 127 - 15 + 1;
      while (!(mant & 0x400)) {
        mant <<= 1;
        --exp;
      }
      bits = sign | (exp << 23) | ((mant & 0x3FF) << 13);
    }
  } else if (exp == 31) {
    bits = sign | 0x7F800000 | (mant << 13);
  } else {
    bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
  }
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}

uint16_t FloatToHalf(float f) {
  uint32_t bits;
  std::memcpy(&bits, &f, 4);
  const uint16_t sign = uint16_t((bits >> 16) & 0x8000);
  const int32_t exp = int32_t((bits >> 23) & 0xFF) - 127 + 15;
  const uint32_t mant = bits & 0x7FFFFF;
  if (exp <= 0) {
    return sign;
  }
  if (exp >= 31) {
    return uint16_t(sign | 0x7C00);
  }
  return uint16_t(sign | (exp << 10) | (mant >> 13));
}

// Xenos rectangle lists: each rect is three vertices, two edges of an
// axis-aligned rectangle meeting at a right angle; the hardware derives the
// fourth corner. The fourth vertex is built here from the inputs (opposite =
// a + b - corner per float attribute, which matches what the GPU produces
// for the affine vertex shaders these passes use), and every rect becomes
// two triangles. Vertices are written in host order, so the result is
// uploaded unswapped.
struct RectStream {
  std::vector<uint8_t> data;
  uint32_t stride = 0;
};

struct RectExpansion {
  std::array<RectStream, 16> streams;
  std::vector<uint32_t> indices;
};

// Lanes of one declaration element, so the derived corner can be computed
// per component: 32-bit floats, halves, or 8/16/32-bit integers (normalized
// formats are linear in their integer value, so integer math is exact).
enum class LaneKind { kNone, kFloat, kHalf, kInt, kUint };

void ElementLanes(uint32_t type, uint32_t &lane_bytes, uint32_t &lanes, LaneKind &kind) {
  const bool is_signed = (type >> 8) & 1;
  const LaneKind integer = is_signed ? LaneKind::kInt : LaneKind::kUint;
  lane_bytes = 0;
  lanes = 0;
  kind = LaneKind::kNone;
  switch (type & 0x3F) {
  case 36: lane_bytes = 4; lanes = 1; kind = LaneKind::kFloat; return; // 32_FLOAT
  case 37: lane_bytes = 4; lanes = 2; kind = LaneKind::kFloat; return; // 32_32_FLOAT
  case 57: lane_bytes = 4; lanes = 3; kind = LaneKind::kFloat; return; // 32_32_32_FLOAT
  case 38: lane_bytes = 4; lanes = 4; kind = LaneKind::kFloat; return; // 32_32_32_32_FLOAT
  case 31: lane_bytes = 2; lanes = 2; kind = LaneKind::kHalf; return;  // 16_16_FLOAT
  case 32: lane_bytes = 2; lanes = 4; kind = LaneKind::kHalf; return;  // 16_16_16_16_FLOAT
  case 6: lane_bytes = 1; lanes = 4; kind = integer; return;           // 8_8_8_8
  case 25: lane_bytes = 2; lanes = 2; kind = integer; return;          // 16_16
  case 26: lane_bytes = 2; lanes = 4; kind = integer; return;          // 16_16_16_16
  case 33: lane_bytes = 4; lanes = 1; kind = integer; return;          // 32
  case 34: lane_bytes = 4; lanes = 2; kind = integer; return;          // 32_32
  case 35: lane_bytes = 4; lanes = 4; kind = integer; return;          // 32_32_32_32
  default: return;
  }
}

double ReadLane(const uint8_t *p, uint32_t lane_bytes, LaneKind kind) {
  switch (kind) {
  case LaneKind::kFloat: {
    float f;
    std::memcpy(&f, p, 4);
    return f;
  }
  case LaneKind::kHalf: {
    uint16_t h;
    std::memcpy(&h, p, 2);
    return HalfToFloat(h);
  }
  default: {
    uint32_t u = 0;
    std::memcpy(&u, p, lane_bytes);
    if (kind == LaneKind::kInt) {
      const uint32_t shift = 32 - lane_bytes * 8;
      return double(int32_t(u << shift) >> shift);
    }
    return double(u);
  }
  }
}

void WriteLane(uint8_t *p, uint32_t lane_bytes, LaneKind kind, double v) {
  switch (kind) {
  case LaneKind::kFloat: {
    const float f = float(v);
    std::memcpy(p, &f, 4);
    return;
  }
  case LaneKind::kHalf: {
    const uint16_t h = FloatToHalf(float(v));
    std::memcpy(p, &h, 2);
    return;
  }
  default: {
    const double bits = lane_bytes * 8.0;
    const double lo = kind == LaneKind::kInt ? -std::ldexp(1.0, int(bits) - 1) : 0.0;
    const double hi = kind == LaneKind::kInt ? std::ldexp(1.0, int(bits) - 1) - 1 : std::ldexp(1.0, int(bits)) - 1;
    const uint32_t u = uint32_t(int64_t(std::clamp(std::round(v), lo, hi)));
    std::memcpy(p, &u, lane_bytes);
    return;
  }
  }
}

// Index list for fans and quad lists, from guest indices or a sequence.
std::vector<uint32_t> ExpandIndices(Topology t, const uint32_t *indices, uint32_t count) {
  auto at = [&](uint32_t i) { return indices ? indices[i] : i; };
  std::vector<uint32_t> out;
  if (t == Topology::kStrip32) {
    // Strips restart after index 0xFFFFFFFF; odd triangles swap their
    // first two vertices to keep the winding.
    uint32_t start = 0;
    for (uint32_t i = 0; i < count; ++i) {
      if (at(i) == 0xFFFFFFFFu) {
        start = i + 1;
        continue;
      }
      if (i < start + 2) {
        continue;
      }
      const uint32_t a = at(i - 2), b = at(i - 1), c = at(i);
      if ((i - start) & 1) {
        out.insert(out.end(), {b, a, c});
      } else {
        out.insert(out.end(), {a, b, c});
      }
    }
  } else if (t == Topology::kFan) {
    for (uint32_t i = 1; i + 1 < count; ++i) {
      out.insert(out.end(), {at(0), at(i), at(i + 1)});
    }
  } else {
    for (uint32_t q = 0; q + 3 < count; q += 4) {
      out.insert(out.end(), {at(q), at(q + 1), at(q + 2), at(q), at(q + 2), at(q + 3)});
    }
  }
  return out;
}

// False when a table could not be made: the draw would sample the previous
// draw's textures.
bool SetupTexturesUncached(RendererState &s, plume::RenderCommandList *cmd, uint32_t device,
                           const HostShader *vs, const HostShader *ps, SharedConstants &vs_shared,
                           SharedConstants &ps_shared, plume::RenderDescriptorSet **out_sets);

bool SetupTextures(RendererState &s, plume::RenderCommandList *cmd, uint32_t device, const HostShader *vs,
                   const HostShader *ps, SharedConstants &vs_shared, SharedConstants &ps_shared) {
  static_assert(offsetof(SharedConstants, sampler) + sizeof(SharedConstants::sampler) == 64 * 4);
  // Key: the fetch constants of every sampler either shader declares.
  const uint32_t ps_mask = ps ? ps->cached->sampler_mask & 0xFFFF : 0, vs_mask = vs->cached->sampler_mask & 0xFFFF;
  uint8_t key_data[8 + 32 * 24];
  size_t n = 0;
  std::memcpy(key_data, &ps_mask, 4);
  std::memcpy(key_data + 4, &vs_mask, 4);
  n = 8;
  for (uint32_t m = ps_mask | (vs_mask << 16); m;) {
    const uint32_t slot = uint32_t(std::countr_zero(m));
    m &= m - 1;
    std::memcpy(key_data + n, GuestVirtual(device + dev::kFetchConstants + slot * 24), 24);
    n += 24;
  }
  const bool cacheable = !s.dumping && !s.hide_texture;
  const uint64_t key = cacheable ? XXH3_64bits(key_data, n) : 0;
  if (cacheable) {
    auto it = s.texture_setups.find(key);
    if (it != s.texture_setups.end() && it->second.generation == s.binding_generation) {
      const auto &setup = it->second;
      std::memcpy(&vs_shared, setup.vs_slots, sizeof(setup.vs_slots));
      std::memcpy(&ps_shared, setup.ps_slots, sizeof(setup.ps_slots));
      for (uint32_t i = 0; i < 4; ++i) {
        BindSet(s, cmd, setup.sets[i], i);
      }
      s.draw_samples_wide = setup.samples_wide;
      return true;
    }
  }
  s.draw_samples_wide = false;
  plume::RenderDescriptorSet *sets[4] = {};
  if (!SetupTexturesUncached(s, cmd, device, vs, ps, vs_shared, ps_shared, sets)) {
    return false;
  }
  if (cacheable) {
    auto &setup = s.texture_setups[key];
    setup.generation = s.binding_generation;
    std::copy(sets, sets + 4, setup.sets);
    std::memcpy(setup.vs_slots, &vs_shared, sizeof(setup.vs_slots));
    std::memcpy(setup.ps_slots, &ps_shared, sizeof(setup.ps_slots));
    setup.samples_wide = s.draw_samples_wide;
  }
  return true;
}

bool SetupTexturesUncached(RendererState &s, plume::RenderCommandList *cmd, uint32_t device,
                           const HostShader *vs, const HostShader *ps, SharedConstants &vs_shared,
                           SharedConstants &ps_shared, plume::RenderDescriptorSet **out_sets) {
  BoundTexture t2d[kTable2DCount], t3d[kTable3DCount], tcube[kTableCubeCount];
  uint64_t ids2d[kTable2DCount] = {}, ids3d[kTable3DCount] = {}, idscube[kTableCubeCount] = {};
  uint32_t n2d = 0, n3d = 0, ncube = 0;
  // Sampler states by fetch register (VS registers from 16), placed in the
  // table sorted so a combination always maps to the same cached set.
  uint32_t sampler_of[32] = {};
  uint32_t used_mask = 0;

  auto bind_stage = [&](const HostShader *shader, uint32_t first_fetch, SharedConstants &shared) {
    uint32_t mask = shader->cached->sampler_mask;
    while (mask) {
      const uint32_t reg = uint32_t(std::countr_zero(mask));
      mask &= mask - 1;
      const uint32_t fetch_slot = first_fetch + reg;
      if (fetch_slot >= 32) {
        continue;
      }
      const uint8_t *raw = GuestVirtual(device + dev::kFetchConstants + fetch_slot * 24);
      uint32_t words[6];
      std::memcpy(words, raw, sizeof(words));
      auto &hit = s.fetch_hits[XXH3_64bits(words, sizeof(words)) & (s.fetch_hits.size() - 1)];
      const bool reuse = !s.dumping && !s.hide_texture && hit.frame == s.frame &&
                         hit.generation == s.binding_generation && !std::memcmp(hit.fetch, words, sizeof(words));
      const auto bt0 = ProfNow();
      BoundTexture b;
      uint32_t sampler = 0;
      if (reuse) {
        b = hit.bound;
        sampler = hit.sampler;
      } else {
        const TextureFetch f = ReadTextureFetch(raw);
        b = BindTexture(s, f);
        sampler = SamplerFor(s, f);
        hit.frame = s.frame;
        hit.generation = s.binding_generation;
        std::memcpy(hit.fetch, words, sizeof(words));
        hit.bound = b;
        hit.sampler = sampler;
      }
      g_stream_stats.bind_texture_ns += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                     ProfNow() - bt0)
                                                     .count());
      if (b.view) {
        auto place = [&](BoundTexture *table, uint64_t *ids, uint32_t &n, uint32_t limit) -> uint32_t {
          for (uint32_t i = 0; i < n; ++i) {
            if (ids[i] == b.id) {
              return i;
            }
          }
          if (n >= limit) {
            return 0;
          }
          table[n] = b;
          ids[n] = b.id;
          return n++;
        };
        if (b.dimension == plume::RenderTextureViewDimension::TEXTURE_3D) {
          shared.texture_3d[reg] = place(t3d, ids3d, n3d, kTable3DCount);
        } else if (b.dimension == plume::RenderTextureViewDimension::TEXTURE_CUBE) {
          shared.texture_cube[reg] = place(tcube, idscube, ncube, kTableCubeCount);
        } else {
          shared.texture_2d[reg] = place(t2d, ids2d, n2d, kTable2DCount);
        }
      }
      sampler_of[first_fetch + reg] = sampler;
      used_mask |= 1u << (first_fetch + reg);
    }
  };
  if (ps) {
    bind_stage(ps, 0, ps_shared);
  }
  bind_stage(vs, 16, vs_shared);

  // Fill the rest with placeholders so every table slot is a valid view.
  for (uint32_t i = n2d; i < kTable2DCount; ++i) {
    t2d[i] = {s.dummy_2d.get(), s.dummy_2d_view.get(), 0, plume::RenderTextureViewDimension::TEXTURE_2D};
  }
  for (uint32_t i = n3d; i < kTable3DCount; ++i) {
    t3d[i] = {s.dummy_3d.get(), s.dummy_3d_view.get(), 0, plume::RenderTextureViewDimension::TEXTURE_3D};
  }
  for (uint32_t i = ncube; i < kTableCubeCount; ++i) {
    tcube[i] = {s.dummy_cube.get(), s.dummy_cube_view.get(), 0,
                plume::RenderTextureViewDimension::TEXTURE_CUBE};
  }
  const auto table_t0 = ProfNow();
  auto *set2d = TableFor(s, 0, ids2d, t2d, nullptr, kTable2DCount);
  auto *set3d = TableFor(s, 1, ids3d, t3d, nullptr, kTable3DCount);
  auto *setcube = TableFor(s, 2, idscube, tcube, nullptr, kTableCubeCount);
  uint32_t order[32];
  uint32_t nsamp = 0;
  for (uint32_t m = used_mask; m;) {
    const uint32_t i = uint32_t(std::countr_zero(m));
    m &= m - 1;
    order[nsamp++] = sampler_of[i];
  }
  std::sort(order, order + nsamp);
  nsamp = uint32_t(std::unique(order, order + nsamp) - order);
  if (nsamp > kTableSamplerCount) {
    nsamp = kTableSamplerCount;
  }
  const plume::RenderSampler *samplers[kTableSamplerCount];
  uint64_t sampler_ids[kTableSamplerCount] = {};
  for (uint32_t i = 0; i < kTableSamplerCount; ++i) {
    samplers[i] = i < nsamp ? s.sampler_list[order[i]] : s.default_sampler.get();
    sampler_ids[i] = i < nsamp ? order[i] : 0;
  }
  for (uint32_t m = used_mask; m;) {
    const uint32_t i = uint32_t(std::countr_zero(m));
    m &= m - 1;
    const uint32_t pos = uint32_t(std::lower_bound(order, order + nsamp, sampler_of[i]) - order);
    (i < 16 ? ps_shared : vs_shared).sampler[i & 15] = pos < nsamp ? pos : 0;
  }
  auto *setsamp = TableFor(s, 3, sampler_ids, nullptr, samplers, kTableSamplerCount);
  g_stream_stats.table_ns += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                          ProfNow() - table_t0)
                                          .count());
  if (!set2d || !set3d || !setcube || !setsamp) {
    return false;
  }
  BindSet(s, cmd, set2d, 0);
  BindSet(s, cmd, set3d, 1);
  BindSet(s, cmd, setcube, 2);
  BindSet(s, cmd, setsamp, 3);
  out_sets[0] = set2d, out_sets[1] = set3d, out_sets[2] = setcube, out_sets[3] = setsamp;
  return true;
}

// Highest vertex the draw fetches (index + base vertex), UINT32_MAX if the
// index data is unreadable.
uint32_t MaxVertexIndex(const DrawRequest &r) {
  if (!r.indexed) {
    return r.count ? r.first + r.count - 1 : UINT32_MAX;
  }
  const uint32_t ib = ReadU32(r.device + dev::kIndexBuffer);
  if (!ib) {
    return UINT32_MAX;
  }
  const uint32_t common = ReadU32(ib);
  const bool is32 = (common >> 31) != 0;
  const uint32_t data = ReadU32(ib + 0x18);
  const uint32_t length = ReadU32(ib + 0x1C);
  const uint32_t stride = is32 ? 4 : 2;
  if (!data || uint64_t(r.first + r.count) * stride > length) {
    return UINT32_MAX;
  }
  const uint32_t endian = (common >> 29) & 3;
  const uint8_t *src = GuestVirtual(data) + r.first * stride;
  // Scanned in place, swapping each index as it is read: this runs for every
  // draw whose vertex shader reads the vertex index, so no copy of the range.
  // 16-bit indices come as 8in16 (or none), 32-bit as 8in32 (or none).
  if ((!is32 && (endian == 0 || endian == 1)) || (is32 && (endian == 0 || endian == 2))) {
    uint32_t highest_raw = 0;
    bool any = false;
    // Eight (or four) indices at a time, restart indices masked out.
    uint32_t i = 0;
    const __m128i swap16 = _mm_setr_epi8(1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14);
    const __m128i swap32 = _mm_setr_epi8(3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12);
    const __m128i ones = _mm_set1_epi32(-1);
    __m128i top = _mm_setzero_si128(), seen = _mm_setzero_si128();
    if (is32) {
      for (; i + 4 <= r.count; i += 4) {
        __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i *>(src + size_t(i) * 4));
        if (endian == 2) {
          v = _mm_shuffle_epi8(v, swap32);
        }
        const __m128i keep = _mm_andnot_si128(_mm_cmpeq_epi32(v, ones), ones);
        top = _mm_max_epu32(top, _mm_and_si128(v, keep));
        seen = _mm_or_si128(seen, keep);
      }
      top = _mm_max_epu32(top, _mm_shuffle_epi32(top, _MM_SHUFFLE(1, 0, 3, 2)));
      top = _mm_max_epu32(top, _mm_shuffle_epi32(top, _MM_SHUFFLE(2, 3, 0, 1)));
    } else {
      for (; i + 8 <= r.count; i += 8) {
        __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i *>(src + size_t(i) * 2));
        if (endian == 1) {
          v = _mm_shuffle_epi8(v, swap16);
        }
        const __m128i keep = _mm_andnot_si128(_mm_cmpeq_epi16(v, ones), ones);
        top = _mm_max_epu16(top, _mm_and_si128(v, keep));
        seen = _mm_or_si128(seen, keep);
      }
      top = _mm_max_epu16(top, _mm_shuffle_epi32(top, _MM_SHUFFLE(1, 0, 3, 2)));
      top = _mm_max_epu16(top, _mm_shuffle_epi32(top, _MM_SHUFFLE(2, 3, 0, 1)));
      top = _mm_max_epu16(top, _mm_srli_epi32(top, 16));
      top = _mm_and_si128(top, _mm_set1_epi32(0xFFFF));
    }
    if (!_mm_testz_si128(seen, seen)) {
      highest_raw = uint32_t(_mm_cvtsi128_si32(top));
      any = true;
    }
    if (is32) {
      for (; i < r.count; ++i) {
        uint32_t v;
        std::memcpy(&v, src + size_t(i) * 4, 4);
        if (endian == 2) {
          v = _byteswap_ulong(v);
        }
        if (v != 0xFFFFFFFFu) {
          highest_raw = std::max(highest_raw, v);
          any = true;
        }
      }
    } else {
      for (; i < r.count; ++i) {
        uint16_t v;
        std::memcpy(&v, src + size_t(i) * 2, 2);
        if (endian == 1) {
          v = _byteswap_ushort(v);
        }
        if (v != 0xFFFFu) {
          highest_raw = std::max<uint32_t>(highest_raw, v);
          any = true;
        }
      }
    }
    const int64_t highest = any ? int64_t(highest_raw) + r.base_vertex : -1;
    return highest < 0 ? UINT32_MAX : uint32_t(highest);
  }
  std::vector<uint8_t> raw(size_t(r.count) * stride);
  SwapEndian(raw.data(), src, uint32_t(raw.size()), endian);
  const uint32_t restart = is32 ? 0xFFFFFFFFu : 0xFFFFu;
  int64_t highest = -1;
  for (uint32_t i = 0; i < r.count; ++i) {
    uint32_t v = 0;
    std::memcpy(&v, raw.data() + i * stride, stride);
    if (v != restart) {
      highest = std::max<int64_t>(highest, int64_t(v) + r.base_vertex);
    }
  }
  return highest < 0 ? UINT32_MAX : uint32_t(highest);
}

std::unique_ptr<RectExpansion> ExpandRects(const DrawRequest &r, const VertexDeclaration &decl,
                                           uint32_t stream_mask) {
  // Vertex numbers of the draw.
  std::vector<uint32_t> ids(r.count);
  if (r.indexed) {
    const uint32_t ib = ReadU32(r.device + dev::kIndexBuffer);
    if (!ib) {
      return nullptr;
    }
    const uint32_t common = ReadU32(ib);
    const bool is32 = (common >> 31) != 0;
    const uint32_t data = ReadU32(ib + 0x18);
    const uint32_t length = ReadU32(ib + 0x1C);
    const uint32_t stride = is32 ? 4 : 2;
    if (!data || uint64_t(r.first + r.count) * stride > length) {
      return nullptr;
    }
    std::vector<uint8_t> raw(r.count * stride);
    SwapEndian(raw.data(), GuestVirtual(data) + r.first * stride, uint32_t(raw.size()), (common >> 29) & 3);
    for (uint32_t i = 0; i < r.count; ++i) {
      uint32_t v = 0;
      std::memcpy(&v, raw.data() + i * stride, stride);
      ids[i] = uint32_t(int32_t(v) + r.base_vertex);
    }
  } else {
    for (uint32_t i = 0; i < r.count; ++i) {
      ids[i] = r.first + i;
    }
  }
  const uint32_t rect_count = r.count / 3;
  if (!rect_count) {
    return nullptr;
  }

  auto out = std::make_unique<RectExpansion>();
  for (uint32_t i = 0; i < kNullStream; ++i) {
    const uint8_t *base = nullptr;
    uint32_t size = 0, stride = 0, endian = 2;
    if (i == 0 && r.up_data) {
      stride = r.up_stride;
      base = GuestVirtual(r.up_data);
      size = (*std::max_element(ids.begin(), ids.end()) + 1) * stride;
    } else if ((stream_mask >> i) & 1) {
      const uint8_t *fc = GuestVirtual(r.device + dev::kFetchConstants + (95 - i) * 8);
      const uint32_t d0 = LoadBE32(fc), d1 = LoadBE32(fc + 4);
      const uint32_t address = d0 & 0xFFFFFFFCu;
      size = ((d1 >> 2) & 0xFFFFFF) * 4;
      stride = uint32_t(GuestVirtual(r.device + dev::kStreamStrides)[i]) * 4;
      endian = d1 & 3;
      base = address ? GuestPhysical(address) : nullptr;
    }
    if (!base || !stride || !size) {
      continue;
    }
    RectStream &rs = out->streams[i];
    rs.stride = stride;
    rs.data.assign(size_t(rect_count) * 4 * stride, 0);
    for (uint32_t k = 0; k < rect_count; ++k) {
      for (uint32_t j = 0; j < 3; ++j) {
        const uint32_t id = ids[k * 3 + j];
        if (uint64_t(id + 1) * stride <= size) {
          SwapEndian(rs.data.data() + (k * 4 + j) * stride, base + id * stride, stride, endian);
        }
      }
    }
    SwapHalves(rs.data.data(), uint32_t(rs.data.size()), HalfSwapFor(decl, i, stride));
  }

  // The corner is the vertex with the right angle, found from POSITION.
  const DeclElement *position = nullptr;
  for (const auto &e : decl.elements) {
    uint32_t lane_bytes, lanes;
    LaneKind kind;
    ElementLanes(e.type, lane_bytes, lanes, kind);
    if (e.usage == 0 && e.usage_index == 0 && lanes >= 2 && out->streams[e.stream & 15].stride) {
      position = &e;
      break;
    }
  }
  out->indices.reserve(rect_count * 6);
  for (uint32_t k = 0; k < rect_count; ++k) {
    uint32_t corner = 0;
    if (position) {
      const RectStream &ps = out->streams[position->stream & 15];
      uint32_t lane_bytes, lanes;
      LaneKind kind;
      ElementLanes(position->type, lane_bytes, lanes, kind);
      float p[3][2];
      for (uint32_t j = 0; j < 3; ++j) {
        const uint8_t *v = ps.data.data() + (k * 4 + j) * ps.stride + position->offset;
        p[j][0] = float(ReadLane(v, lane_bytes, kind));
        p[j][1] = float(ReadLane(v + lane_bytes, lane_bytes, kind));
      }
      float best = FLT_MAX;
      for (uint32_t j = 0; j < 3; ++j) {
        const float *a = p[(j + 1) % 3], *b = p[(j + 2) % 3];
        const float e1x = a[0] - p[j][0], e1y = a[1] - p[j][1];
        const float e2x = b[0] - p[j][0], e2y = b[1] - p[j][1];
        const float len = std::sqrt((e1x * e1x + e1y * e1y) * (e2x * e2x + e2y * e2y));
        const float cosine = len > 0.0f ? std::fabs(e1x * e2x + e1y * e2y) / len : FLT_MAX;
        if (cosine < best) {
          best = cosine;
          corner = j;
        }
      }
    }
    const uint32_t a = (corner + 1) % 3, b = (corner + 2) % 3;
    for (uint32_t i = 0; i < kNullStream; ++i) {
      RectStream &rs = out->streams[i];
      if (!rs.stride) {
        continue;
      }
      uint8_t *v = rs.data.data() + k * 4 * rs.stride;
      uint8_t *d = v + 3 * rs.stride;
      std::memcpy(d, v + corner * rs.stride, rs.stride);
      for (const auto &e : decl.elements) {
        uint32_t lane_bytes, lanes;
        LaneKind kind;
        ElementLanes(e.type, lane_bytes, lanes, kind);
        if ((e.stream & 15) != i || !lanes || e.offset + lanes * lane_bytes > rs.stride) {
          continue;
        }
        for (uint32_t l = 0; l < lanes; ++l) {
          const uint32_t o = e.offset + l * lane_bytes;
          WriteLane(d + o, lane_bytes, kind,
                    ReadLane(v + a * rs.stride + o, lane_bytes, kind) +
                        ReadLane(v + b * rs.stride + o, lane_bytes, kind) -
                        ReadLane(v + corner * rs.stride + o, lane_bytes, kind));
        }
      }
    }
    // The second triangle is the first turned 180 degrees about the rect's
    // centre: same winding, corner replaced by the derived vertex.
    uint32_t second[3];
    for (uint32_t j = 0; j < 3; ++j) {
      second[j] = j == corner ? 3 : (j == a ? b : a);
    }
    const uint32_t base = k * 4;
    out->indices.insert(out->indices.end(),
                        {base, base + 1, base + 2, base + second[0], base + second[1], base + second[2]});
  }
  return out;
}

// Set on the render thread, which times whole batches instead (a clock read
// per draw is not free, least of all under Wine).
thread_local bool t_timed_by_batch = false;

// Adds the enclosing scope's CPU time to the frame timing report.
struct RendererTimer {
  explicit RendererTimer(uint32_t draws) : draws(draws) {
    if (!t_timed_by_batch) {
      start = std::chrono::steady_clock::now();
    }
  }
  ~RendererTimer() {
    if (t_timed_by_batch) {
      return;
    }
    Video::AddRendererTime(uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                        std::chrono::steady_clock::now() - start)
                                        .count()),
                           draws);
  }
  std::chrono::steady_clock::time_point start;
  uint32_t draws;
};

// Profiling: CPU time per stage of ExecuteDraw, logged from Swap every 600
// frames (stage i ends at Lap(i); kDrawPhases-1 is the draw call and tail).
constexpr int kDrawPhases = 11;
std::array<uint64_t, kDrawPhases> g_draw_phase_ns{};
const char *const kDrawPhaseNames[kDrawPhases] = {"shaders", "decl", "targets", "pipeline", "indices",
                                                  "streams", "framebuffer", "read consts", "textures",
                                                  "upload consts", "draw"};
struct PhaseClock {
  void Lap(int i) {
    const auto now = ProfNow();
    g_draw_phase_ns[i] += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(now - last).count());
    last = now;
  }
  ~PhaseClock() { Lap(kDrawPhases - 1); }
  std::chrono::steady_clock::time_point last = ProfNow();
};

// Horizontal clip-space extent of a 2D draw, for anchoring HUD elements on an
// ultrawide screen. The sprite shaders place a vertex at pos.xy * PosScale.xy
// + PosScale.zw (c45); a dial sits at its g_DialPosition (c88) plus
// g_vGadgetOffset (c44), through g_vPosScale (c45).
bool UiDrawExtent(const DrawRequest &r, const VertexDeclaration &decl, uint64_t vs, const uint32_t *vs_consts,
                  uint32_t vs_regs, float &lo, float &hi, uint32_t axis = 0) {
  if (vs_regs <= 88 && vs == kUiDialVs) {
    return false;
  }
  if (vs_regs <= 45) {
    return false;
  }
  // axis 0: clip-space x extent, 1: y.
  const float scale_x = AsFloat(vs_consts[45 * 4 + axis]), offset_x = AsFloat(vs_consts[45 * 4 + 2 + axis]);
  if (vs == kUiDialVs) {
    const float x = AsFloat(vs_consts[88 * 4 + axis]) + AsFloat(vs_consts[88 * 4 + 2 + axis]) +
                    AsFloat(vs_consts[44 * 4 + axis]);
    lo = hi = x * scale_x + offset_x;
    return std::isfinite(lo);
  }
  const DeclElement *position = nullptr;
  for (const auto &e : decl.elements) {
    if (e.usage == 0 && e.usage_index == 0) {
      position = &e;
      break;
    }
  }
  if (!position || !r.count) {
    return false;
  }
  uint32_t lane_bytes, lanes;
  LaneKind kind;
  ElementLanes(position->type, lane_bytes, lanes, kind);
  if (lane_bytes != 4 || lanes < 2) {
    return false;
  }
  const uint8_t *base = nullptr;
  uint32_t size = 0, stride = 0, endian = 2;
  if (r.up_data) {
    stride = r.up_stride;
    base = GuestVirtual(r.up_data);
    size = ~0u;
  } else {
    const uint32_t stream = position->stream & 15;
    const uint8_t *fc = GuestVirtual(r.device + dev::kFetchConstants + (95 - stream) * 8);
    const uint32_t d0 = LoadBE32(fc), d1 = LoadBE32(fc + 4);
    size = ((d1 >> 2) & 0xFFFFFF) * 4;
    stride = uint32_t(GuestVirtual(r.device + dev::kStreamStrides)[stream]) * 4;
    endian = d1 & 3;
    base = (d0 & 0xFFFFFFFCu) ? GuestPhysical(d0 & 0xFFFFFFFCu) : nullptr;
  }
  if (!base || !stride) {
    return false;
  }
  // Vertex numbers: 2D draws are short, so the index buffer is read directly.
  uint32_t ib_data = 0, ib_common = 0;
  if (r.indexed) {
    const uint32_t ib = ReadU32(r.device + dev::kIndexBuffer);
    if (!ib) {
      return false;
    }
    ib_common = ReadU32(ib);
    ib_data = ReadU32(ib + 0x18);
    if (!ib_data) {
      return false;
    }
  }
  lo = std::numeric_limits<float>::max();
  hi = -lo;
  for (uint32_t i = 0; i < r.count && i < 4096; ++i) {
    uint32_t id = r.first + i;
    if (r.indexed) {
      const bool is32 = (ib_common >> 31) != 0;
      uint8_t raw[4] = {};
      SwapEndian(raw, GuestVirtual(ib_data) + (r.first + i) * (is32 ? 4 : 2), is32 ? 4 : 2, (ib_common >> 29) & 3);
      uint32_t v = 0;
      std::memcpy(&v, raw, is32 ? 4 : 2);
      id = uint32_t(int32_t(v) + r.base_vertex);
    }
    if (size != ~0u && uint64_t(id) * stride + position->offset + axis * 4 + 4 > size) {
      continue;
    }
    uint8_t lane[4];
    SwapEndian(lane, base + uint64_t(id) * stride + position->offset + axis * 4, 4, endian);
    const float x = float(ReadLane(lane, 4, kind)) * scale_x + offset_x;
    if (std::isfinite(x)) {
      lo = std::min(lo, x);
      hi = std::max(hi, x);
    }
  }
  return lo <= hi;
}

void ExecuteDraw(RendererState &s, const DrawRequest &r) {
  RendererTimer timer(1);
  PhaseClock phases;
  if (!EnsureReady(s) || !r.count) {
    return;
  }
  plume::RenderCommandList *cmd = Video::OpenCommandList();
  if (!cmd) {
    return;
  }
  PrepareFrame(s);
  ++s.draws;
  ++s.draws_since_resolve;

  Topology topology = ToTopology(r.primitive);
  if (topology == Topology::kTriangleStrip && r.indexed) {
    const uint32_t ib = ReadU32(r.device + dev::kIndexBuffer);
    if (ib && (ReadU32(ib) >> 31)) {
      topology = Topology::kStrip32;
    }
  }
  if (topology == Topology::kNone) {
    if (LogOnce(s, 0x3000000000ull | r.primitive)) {
      REXGPU_WARN("PGR4 D3D12: primitive type {} not drawn", r.primitive);
    }
    ++s.skipped_draws;
    return;
  }

  const uint32_t vs_object = ReadU32(r.device + dev::kVertexShader);
  const uint32_t ps_object = ReadU32(r.device + dev::kPixelShader);
  HostShader *vs = ShaderFor(s, vs_object);
  HostShader *ps = ShaderFor(s, ps_object);
  // No pixel shader is a depth-only pass; the host pipeline has none either.
  const bool ps_ok = ps_object ? (ps && ps->shader && ps->cached->pixel) : true;
  if (!ps_object) {
    ps = nullptr;
  }
  if (!vs || !vs->shader || vs->cached->pixel || !ps_ok) {
    if (LogOnce(s, 0x5000000000ull ^ (uint64_t(vs_object) << 1) ^ (uint64_t(ps_object) << 33))) {
      REXGPU_WARN("PGR4 D3D12: draw skipped, vs {:08X} ({}) ps {:08X} ({}), {} shader objects known",
                  vs_object, vs ? (vs->shader ? "ok" : "uncached") : "unknown", ps_object,
                  ps ? (ps->shader ? "ok" : "uncached") : "unknown", s.shaders_by_object.size());
    }
    ++s.skipped_draws;
    return;
  }
  if (const int32_t safe = REXCVAR_GET(gpu_safe);
      ((safe & 4) && vs->cached->indexed_fetch) || ((safe & 8) && vs->cached->reads_vertex_index)) {
    ++s.skipped_draws;
    return;
  }
  phases.Lap(0);
  const VertexDeclaration *decl = DeclarationFor(s, ReadU32(r.device + dev::kVertexDeclaration));
  static const VertexDeclaration kEmptyDecl;
  if (!decl) {
    decl = &kEmptyDecl;
  }
  const InputBinding *input = InputBindingFor(s, vs, decl);

  if (s.dumping) {
    const uint32_t color_info = ReadRegister(r.device, 0x2001);
    if (((color_info >> 20) & 0x3F) && LogOnce(s, 0x8000000000ull ^ color_info)) {
      REXGPU_INFO("PGR4 D3D12 dump: RB_COLOR_INFO {:08X} (format {}, exp bias {})", color_info,
                  (color_info >> 16) & 0xF, int32_t(color_info << 6) >> 26);
    }
  }
  phases.Lap(1);
  Targets targets = CurrentTargets(s, r.device);
  if (!targets.width || !targets.height) {
    ++s.skipped_draws;
    return;
  }

  if (s.skip_ps && ps && ps->key == s.skip_ps) {
    return;
  }

  PipelineKey key;
  key.vs = vs->key;
  key.ps = ps ? ps->key : 0;
  key.decl = decl->hash;
  key.rt_count = targets.color_count;
  for (uint32_t i = 0; i < targets.color_count; ++i) {
    key.rt_formats[i] = uint32_t(targets.color[i]->format);
  }
  key.depth_format = targets.depth ? uint32_t(kDepthFormat) : 0;
  key.topology = uint32_t(topology == Topology::kFan || topology == Topology::kQuads ||
                                  topology == Topology::kRects || topology == Topology::kStrip32
                              ? Topology::kTriangles
                              : topology);
  key.blend[0] = ReadRegister(r.device, kRegBlendControl0);
  for (uint32_t i = 1; i < 4; ++i) {
    key.blend[i] = ReadRegister(r.device, kRegBlendControl1 + i - 1);
  }
  key.color_mask = ReadRegister(r.device, kRegColorMask);
  key.depth_control = ReadRegister(r.device, kRegDepthControl);
  key.stencil_front = ReadRegister(r.device, kRegStencilRefMask);
  key.stencil_back = ReadRegister(r.device, kRegStencilRefMaskBack);
  key.mode_control = ReadRegister(r.device, kRegModeControl) & 7;
  // Front and back both culled: no triangle survives on the Xenos. The host
  // has no such mode and drew them unculled, as black shapes over the scene.
  if ((key.mode_control & 3) == 3 && topology != Topology::kPoints && topology != Topology::kLines &&
      topology != Topology::kLineStrip && topology != Topology::kRects) {
    if (s.dumping) {
      REXGPU_INFO("PGR4 D3D12 dump: draw {} culled (front and back), vs {:016X} count {}", s.draws_since_resolve,
                  vs->key, r.count);
    }
    return;
  }
  // Polygon offset (shadow map passes), front values: bits 11-13 of
  // PA_SU_SC_MODE_CNTL enable it for front, back and other primitives.
  if ((ReadRegister(r.device, kRegModeControl) >> 11) & 7) {
    key.depth_bias = AsFloat(ReadRegister(r.device, kRegPolyOffsetFrontOffset));
    key.depth_bias_slope = AsFloat(ReadRegister(r.device, kRegPolyOffsetFrontScale));
  }
  // The car shadow pass draws to 24_8_FLOAT with offset 0.015: a uniform
  // 0.015 there washes the variance shadow out.
  const bool float_depth = targets.depth && targets.depth->guest_format == 1;
  const float depth_offset = float_depth ? 0.0f : key.depth_bias;
  if (!float_depth) {
    key.depth_bias = 0; // not part of the pipeline (depth_offset)
  }
  if (s.dumping) {
    REXGPU_INFO("PGR4 D3D12 dump: draw {} prim {} count {} vs {:016X} ps {:016X} rt {}x{} n{} depth {} "
                "surfaces {:08X}/{:08X} ids {}/{} mode {:X} blend {:08X} "
                "mask {:X} zctl {:08X} cctl {:08X} topo {} dev {:08X} bias {} {} decl{}",
                s.draws_since_resolve, r.primitive, r.count, vs->key, ps ? ps->key : 0, targets.width,
                targets.height, targets.color_count, targets.depth ? 1 : 0,
                ReadU32(r.device + dev::kRenderTargets), ReadU32(r.device + dev::kDepthStencil),
                targets.color_count ? targets.color[0]->id : 0, targets.depth ? targets.depth->id : 0,
                key.mode_control, key.blend[0], key.color_mask,
                key.depth_control, ReadRegister(r.device, kRegColorControl), int(topology), r.device,
                depth_offset, key.depth_bias_slope, [&] {
                  std::string d;
                  for (const auto &e : decl->elements) {
                    d += fmt::format(" {}.{}:{:X}", e.usage, e.usage_index, e.type);
                  }
                  return d;
                }());
  }
  phases.Lap(2);
  plume::RenderPipeline *pipeline = PipelineFor(s, key, vs, ps, *input);
  if (!pipeline) {
    ++s.skipped_draws;
    return;
  }

  phases.Lap(3);
  // Index data first: fans and quads become lists here.
  const plume::RenderBuffer *index_buffer = nullptr;
  uint64_t index_offset = 0;
  uint32_t index_size = 0;
  plume::RenderFormat index_format = plume::RenderFormat::R16_UINT;
  uint32_t draw_count = r.count;
  uint32_t first_index = 0;
  bool indexed = r.indexed;
  int32_t quad_base_vertex = 0; // non-indexed quads from s.quad_indices
  std::unique_ptr<RectExpansion> rects;
  if (topology == Topology::kRects) {
    rects = ExpandRects(r, *decl, input->stream_mask);
    if (!rects || rects->indices.empty()) {
      ++s.skipped_draws;
      return;
    }
    if (s.dumping) {
      for (uint32_t i = 0; i < kNullStream; ++i) {
        const RectStream &rs = rects->streams[i];
        if (rs.stride < 8 || rs.data.size() < 4 * rs.stride) {
          continue;
        }
        std::string v;
        for (uint32_t k = 0; k < 4; ++k) {
          float xy[2];
          std::memcpy(xy, rs.data.data() + k * rs.stride, 8);
          v += fmt::format(" ({} {})", xy[0], xy[1]);
        }
        REXGPU_INFO("PGR4 D3D12 dump:   rect stream {} stride {} first rect xy{} indices {}", i, rs.stride, v,
                    rects->indices.size());
      }
    }
    UploadAllocation up = Video::Upload(rects->indices.data(), uint32_t(rects->indices.size() * 4), 4);
    if (!up) {
      ++s.skipped_draws;
      return;
    }
    index_buffer = up.buffer;
    index_offset = up.offset;
    index_size = up.size;
    index_format = plume::RenderFormat::R32_UINT;
    draw_count = uint32_t(rects->indices.size());
    indexed = true;
  } else if (r.indexed) {
    const uint32_t ib = ReadU32(r.device + dev::kIndexBuffer);
    if (!ib) {
      ++s.skipped_draws;
      return;
    }
    // PGR4's D3DIndexBuffer: +0 Common (bit 31 = 32-bit indices, bits 29-30
    // endian), +0x18 data address, +0x1C length (CreateIndexBuffer).
    const uint32_t common = ReadU32(ib);
    const bool is32 = (common >> 31) != 0;
    const uint32_t endian = (common >> 29) & 3;
    const uint32_t data = ReadU32(ib + 0x18);
    const uint32_t length = ReadU32(ib + 0x1C);
    const uint32_t stride = is32 ? 4 : 2;
    if (!data || uint64_t(r.first + r.count) * stride > length) {
      ++s.skipped_draws;
      return;
    }
    if (topology == Topology::kFan || topology == Topology::kQuads || topology == Topology::kStrip32) {
      std::vector<uint8_t> raw(r.count * stride);
      SwapEndian(raw.data(), GuestVirtual(data) + r.first * stride, uint32_t(raw.size()), endian);
      std::vector<uint32_t> indices(r.count);
      for (uint32_t i = 0; i < r.count; ++i) {
        uint32_t v = 0;
        std::memcpy(&v, raw.data() + i * stride, stride);
        indices[i] = v;
      }
      auto expanded = ExpandIndices(topology, indices.data(), r.count);
      UploadAllocation up = Video::Upload(expanded.data(), uint32_t(expanded.size() * 4), 4);
      if (!up || expanded.empty()) {
        ++s.skipped_draws;
        return;
      }
      index_buffer = up.buffer;
      index_offset = up.offset;
      index_size = up.size;
      index_format = plume::RenderFormat::R32_UINT;
      draw_count = uint32_t(expanded.size());
    } else {
      const HostBuffer *hb = BufferFor(s, GuestVirtual(data), length, endian);
      if (!hb) {
        ++s.skipped_draws;
        return;
      }
      index_buffer = hb->buffer.get();
      index_size = length;
      index_format = is32 ? plume::RenderFormat::R32_UINT : plume::RenderFormat::R16_UINT;
      first_index = r.first;
    }
  } else if (topology == Topology::kQuads && REXCVAR_GET(quad_index_buffer)) {
    const uint32_t quads = r.count / 4;
    if (!quads) {
      ++s.skipped_draws;
      return;
    }
    if (quads > s.quad_index_quads) {
      const uint32_t capacity = std::max(std::bit_ceil(quads), 16384u);
      auto buffer = s.device->createBuffer(plume::RenderBufferDesc::IndexBuffer(
          uint64_t(capacity) * 6 * 4, plume::RenderHeapType::UPLOAD));
      if (!buffer) {
        ++s.skipped_draws;
        return;
      }
      buffer->setName("quad indices");
      auto *out = static_cast<uint32_t *>(buffer->map());
      for (uint32_t q = 0; q < capacity; ++q) {
        const uint32_t v = q * 4;
        const uint32_t pattern[6] = {v, v + 1, v + 2, v, v + 2, v + 3};
        std::memcpy(out + q * 6, pattern, sizeof(pattern));
      }
      buffer->unmap();
      if (s.quad_indices) {
        Video::ParkUntilFence(std::move(s.quad_indices));
      }
      s.quad_indices = std::move(buffer);
      s.quad_index_quads = capacity;
    }
    index_buffer = s.quad_indices.get();
    index_size = quads * 6 * 4;
    index_format = plume::RenderFormat::R32_UINT;
    draw_count = quads * 6;
    quad_base_vertex = int32_t(r.first);
    indexed = true;
  } else if (topology == Topology::kFan || topology == Topology::kQuads) {
    auto expanded = ExpandIndices(topology, nullptr, r.count);
    for (auto &i : expanded) {
      i += r.first;
    }
    UploadAllocation up = Video::Upload(expanded.data(), uint32_t(expanded.size() * 4), 4);
    if (!up || expanded.empty()) {
      ++s.skipped_draws;
      return;
    }
    index_buffer = up.buffer;
    index_offset = up.offset;
    index_size = up.size;
    index_format = plume::RenderFormat::R32_UINT;
    draw_count = uint32_t(expanded.size());
    indexed = true;
  }

  phases.Lap(4);
  // Vertex streams.
  struct StreamBinding {
    plume::RenderVertexBufferView view;
    plume::RenderInputSlot slot;
  };
  std::array<StreamBinding, kNullStream + 1> streams{};
  uint32_t stream_mask = input->stream_mask;
  uint32_t max_index = UINT32_MAX; // computed on demand for wrapped streams
  const auto special_t0 = ProfNow();
  if (rects) {
    for (uint32_t i = 0; i < kNullStream; ++i) {
      const RectStream &rs = rects->streams[i];
      if (!rs.stride) {
        continue;
      }
      UploadAllocation up = Video::Upload(rs.data.data(), uint32_t(rs.data.size()), 4);
      if (!up) {
        ++s.skipped_draws;
        return;
      }
      streams[i].view = plume::RenderVertexBufferView(plume::RenderBufferReference(up.buffer, up.offset),
                                                      uint32_t(rs.data.size()));
      streams[i].slot = plume::RenderInputSlot(i, rs.stride);
      BindVertexStream(s, cmd, i, &streams[i].view, &streams[i].slot);
    }
    stream_mask = 0;
  } else if (r.up_data) {
    const uint32_t size = r.count * r.up_stride;
    if (s.dumping && r.up_stride >= 8 && r.count >= 4) {
      std::string v;
      for (uint32_t k = 0; k < 4; ++k) {
        v += fmt::format(" ({} {})", AsFloat(ReadU32(r.up_data + k * r.up_stride)),
                         AsFloat(ReadU32(r.up_data + k * r.up_stride + 4)));
      }
      REXGPU_INFO("PGR4 D3D12 dump:   up stride {} first xy{} draw_count {} first_index {}", r.up_stride, v, draw_count,
                  first_index);
    }
    UploadAllocation up = Video::Upload(nullptr, size, 4);
    if (!up) {
      ++s.skipped_draws;
      return;
    }
    SwapToUpload(static_cast<uint8_t *>(up.memory), GuestVirtual(r.up_data), size, 2,
                 HalfSwapFor(*decl, 0, r.up_stride));
    streams[0].view = plume::RenderVertexBufferView(plume::RenderBufferReference(up.buffer, up.offset), size);
    streams[0].slot = plume::RenderInputSlot(0, r.up_stride);
    stream_mask &= ~1u;
    BindVertexStream(s, cmd, 0, &streams[0].view, &streams[0].slot);
  }
  const auto loop_t0 = ProfNow();
  g_stream_stats.special_ns += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(loop_t0 - special_t0).count());
  for (uint32_t i = 0; i < kNullStream; ++i) {
    if (!((stream_mask >> i) & 1)) {
      continue;
    }
    // Stream i's vertex fetch constant is slot 95 - i (SetStreamSource):
    // dword 0 address | type, dword 1 size in words << 2 | endian.
    const uint8_t *fc = GuestVirtual(r.device + dev::kFetchConstants + (95 - i) * 8);
    const uint32_t d0 = LoadBE32(fc), d1 = LoadBE32(fc + 4);
    const uint32_t address = d0 & 0xFFFFFFFCu;
    const uint32_t size = ((d1 >> 2) & 0xFFFFFF) * 4;
    const uint32_t stride = uint32_t(GuestVirtual(r.device + dev::kStreamStrides)[i]) * 4;
    if (!address || !size) {
      continue;
    }
    if (vs->cached->reads_vertex_index && stride) {
      // Instancing from the vertex index: the shader fetches vertex
      // (index mod N) of an N-vertex stream, which the host input assembler
      // cannot, so the stream is repeated out to the highest index.
      if (max_index == UINT32_MAX) {
        max_index = MaxVertexIndex(r);
      }
      const uint32_t n = size / stride;
      if (s.dumping) {
        const uint8_t *v0 = GuestPhysical(address);
        REXGPU_INFO("PGR4 D3D12 dump:   stream {} vertices {} stride {} max index {} base {} address {:08X} "
                    "first {:08X} {:08X} {:08X} {:08X} {:08X} {:08X} {:08X}",
                    i, n, stride, max_index, r.base_vertex, address, LoadBE32(v0), LoadBE32(v0 + 4),
                    LoadBE32(v0 + 8), LoadBE32(v0 + 12), LoadBE32(v0 + 16), LoadBE32(v0 + 20), LoadBE32(v0 + 24));
      }
      if (n && max_index != UINT32_MAX && max_index >= n && uint64_t(max_index + 1) * stride <= (32u << 20)) {
        const uint32_t wrapped = (max_index + 1) * stride;
        ++g_stream_stats.wrapped;
        g_stream_stats.wrapped_bytes += wrapped;
        UploadAllocation up = Video::Upload(nullptr, wrapped, 4);
        if (up) {
          // Built in a CPU copy and written to upload memory once (it is
          // write-combined; repeating vertices from it would read it back).
          thread_local std::vector<uint8_t> repeated;
          repeated.resize(wrapped);
          SwapEndian(repeated.data(), GuestPhysical(address), n * stride, d1 & 3);
          SwapHalves(repeated.data(), n * stride, HalfSwapFor(*decl, i, stride));
          for (uint32_t v = n; v <= max_index; ++v) {
            std::memcpy(repeated.data() + size_t(v) * stride, repeated.data() + size_t(v % n) * stride, stride);
          }
          std::memcpy(up.memory, repeated.data(), wrapped);
          streams[i].view = plume::RenderVertexBufferView(plume::RenderBufferReference(up.buffer, up.offset), wrapped);
          streams[i].slot = plume::RenderInputSlot(i, stride);
          BindVertexStream(s, cmd, i, &streams[i].view, &streams[i].slot);
          continue;
        }
      }
    }
    const auto bf_t0 = ProfNow();
    const HostBuffer *hb = BufferFor(s, GuestPhysical(address), size, d1 & 3, HalfSwapFor(*decl, i, stride),
                                     REX_KERNEL_MEMORY()->TranslatePhysical(address));
    g_stream_stats.buffer_for_ns += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                 ProfNow() - bf_t0)
                                                 .count());
    if (!hb) {
      continue;
    }
    streams[i].view = plume::RenderVertexBufferView(plume::RenderBufferReference(hb->buffer.get()), size);
    streams[i].slot = plume::RenderInputSlot(i, stride);
    BindVertexStream(s, cmd, i, &streams[i].view, &streams[i].slot);
  }
  g_stream_stats.loop_ns += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                         ProfNow() - loop_t0)
                                         .count());
  {
    StreamBinding &null_stream = streams[kNullStream];
    null_stream.view = plume::RenderVertexBufferView(plume::RenderBufferReference(s.null_stream.get()), 256);
    null_stream.slot = plume::RenderInputSlot(kNullStream, 0);
    BindVertexStream(s, cmd, kNullStream, &null_stream.view, &null_stream.slot);
  }

  phases.Lap(5);
  NoteDepthDraw(s, targets.depth);
  if (!BindFramebuffer(s, cmd, targets)) {
    ++s.skipped_draws;
    return;
  }
  BindPipeline(s, cmd, pipeline);

  // Constants.
  static uint32_t vs_consts[1024], ps_consts[1024];
  phases.Lap(6);
  // Only the registers each shader declares are read and uploaded (most use
  // far fewer than 256); dumps log fixed registers, so they read all.
  const uint32_t vs_regs = s.dumping ? 256u : std::max(vs->cached->constant_regs, 1u);
  const uint32_t ps_regs = s.dumping || !ps ? 256u : std::max(ps->cached->constant_regs, 1u);
  ReadFloatConstants(r.device, dev::kVsConstants, vs_consts, vs_regs);
  // Crowd sprites (the one-frame "void" over the track): their block constants.
  if (s.dumping && key.vs == 0x6B8928C5281643CEull && vs_regs > 51) {
    REXGPU_INFO("PGR4 D3D12 dump:   crowd c48 {} {} {} {} c49 {} {} {} {} c50 {} {} {} {} c51 {} {} {} {}",
                AsFloat(vs_consts[192]), AsFloat(vs_consts[193]), AsFloat(vs_consts[194]), AsFloat(vs_consts[195]),
                AsFloat(vs_consts[196]), AsFloat(vs_consts[197]), AsFloat(vs_consts[198]), AsFloat(vs_consts[199]),
                AsFloat(vs_consts[200]), AsFloat(vs_consts[201]), AsFloat(vs_consts[202]), AsFloat(vs_consts[203]),
                AsFloat(vs_consts[204]), AsFloat(vs_consts[205]), AsFloat(vs_consts[206]), AsFloat(vs_consts[207]));
  }
  ReadFloatConstants(r.device, dev::kPsConstants, ps_consts, ps_regs);
  SharedConstants vs_shared{}, ps_shared{};
  phases.Lap(7);
  const auto setup_t0 = ProfNow();
  const bool textures_ok = SetupTextures(s, cmd, r.device, vs, ps, vs_shared, ps_shared);
  g_stream_stats.setup_textures_ns += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                   ProfNow() - setup_t0)
                                                   .count());
  if (!textures_ok) {
    ++s.skipped_draws;
    return;
  }
  if (s.dumping && ps) {
    REXGPU_INFO("PGR4 D3D12 dump:   ps c128 {} {} {} {}  c130 {} {} {} {}", AsFloat(ps_consts[512]),
                AsFloat(ps_consts[513]), AsFloat(ps_consts[514]), AsFloat(ps_consts[515]),
                AsFloat(ps_consts[520]), AsFloat(ps_consts[521]), AsFloat(ps_consts[522]),
                AsFloat(ps_consts[523]));
    // PGR4_DUMP_PSC=<reg>,<reg>,...: extra PS float constants (absolute c numbers).
    if (const char *psc = std::getenv("PGR4_DUMP_PSC")) {
      std::string line;
      for (const char *p = psc; *p;) {
        char *end = nullptr;
        const long reg = std::strtol(p, &end, 10);
        if (end == p) {
          break;
        }
        if (reg >= 0 && reg < 256) {
          line += fmt::format(" c{} {} {} {} {}", reg, AsFloat(ps_consts[reg * 4]), AsFloat(ps_consts[reg * 4 + 1]),
                              AsFloat(ps_consts[reg * 4 + 2]), AsFloat(ps_consts[reg * 4 + 3]));
        }
        p = *end == ',' ? end + 1 : end;
      }
      REXGPU_INFO("PGR4 D3D12 dump:   psc{}", line);
    }
    // Debug: a dump.vsc file next to the exe holding a VS key prefix (hex) logs
    // every nonzero VS float constant of the matching draws.
    if (std::FILE *vf = std::fopen("dump.vsc", "rb")) {
      char prefix[24] = {};
      std::fgets(prefix, sizeof(prefix), vf);
      std::fclose(vf);
      const std::string key = fmt::format("{:016X}", vs->key);
      const size_t len = std::strspn(prefix, "0123456789abcdefABCDEF");
      if (len && _strnicmp(key.c_str(), prefix, len) == 0) {
        std::string line;
        for (int reg = 0; reg < 256; ++reg) {
          const uint32_t *c = &vs_consts[reg * 4];
          if (c[0] | c[1] | c[2] | c[3]) {
            line += fmt::format(" c{} {:g} {:g} {:g} {:g};", reg, AsFloat(c[0]), AsFloat(c[1]), AsFloat(c[2]),
                                AsFloat(c[3]));
          }
        }
        REXGPU_INFO("PGR4 D3D12 dump:   vsc d{:04}{}", s.draws, line);
        // The draw's vertex streams too (dumpvs_<key>_<stream>.bin, guest
        // byte order), for evaluating the shader offline.
        if (decl) {
          for (const auto &e : decl->elements) {
            if (e.stream >= kNullStream) {
              continue;
            }
            const uint8_t *fc = GuestVirtual(r.device + dev::kFetchConstants + (95 - e.stream) * 8);
            const uint32_t address = LoadBE32(fc) & 0xFFFFFFFCu, size = ((LoadBE32(fc + 4) >> 2) & 0xFFFFFF) * 4;
            const uint32_t stride = uint32_t(GuestVirtual(r.device + dev::kStreamStrides)[e.stream]) * 4;
            REXGPU_INFO("PGR4 D3D12 dump:   vsc element usage {}.{} stream {} offset {} type {:06X} stride {} "
                        "address {:08X} size {}",
                        e.usage, e.usage_index, e.stream, e.offset, e.type, stride, address, size);
            char name[64];
            // Per draw during a dump (the crowd's draws each start further in).
            if (s.dumping) {
              std::snprintf(name, sizeof(name), "dumpvs_%016llX_%u_d%04llu.bin", (unsigned long long)vs->key,
                            e.stream, (unsigned long long)s.draws);
            } else {
              std::snprintf(name, sizeof(name), "dumpvs_%016llX_%u.bin", (unsigned long long)vs->key, e.stream);
            }
            if (address && size) {
              if (FILE *out = std::fopen(name, "wb")) {
                std::fwrite(GuestPhysical(address), 1, size, out);
                std::fclose(out);
              }
            }
          }
        }
      }
    }
    if (decl) {
      for (const auto &e : decl->elements) {
        if (e.usage != 5 || (e.type & 0x3F) != 0x19 || e.stream >= kNullStream) {
          continue;
        }
        const uint8_t *fc = GuestVirtual(r.device + dev::kFetchConstants + (95 - e.stream) * 8);
        const uint32_t address = LoadBE32(fc) & 0xFFFFFFFCu, size = ((LoadBE32(fc + 4) >> 2) & 0xFFFFFF) * 4;
        const uint32_t stride = uint32_t(GuestVirtual(r.device + dev::kStreamStrides)[e.stream]) * 4;
        if (!address || !stride) {
          continue;
        }
        int lo[2] = {INT32_MAX, INT32_MAX}, hi[2] = {INT32_MIN, INT32_MIN};
        const uint8_t *base = GuestPhysical(address);
        for (uint32_t v = 0; v + 1 <= size / stride; ++v) {
          const uint32_t d = LoadBE32(base + v * stride + e.offset);
          const int lane[2] = {int16_t(d >> 16), int16_t(d & 0xFFFF)};
          for (int k = 0; k < 2; ++k) {
            lo[k] = std::min(lo[k], lane[k]);
            hi[k] = std::max(hi[k], lane[k]);
          }
        }
        REXGPU_INFO("PGR4 D3D12 dump:   texcoord{} off {} type {:06X} x {}..{} y {}..{}  vs c44 {} {} {} {}",
                    e.usage_index, e.offset, e.type, lo[0], hi[0], lo[1], hi[1], AsFloat(vs_consts[176]),
                    AsFloat(vs_consts[177]), AsFloat(vs_consts[178]), AsFloat(vs_consts[179]));
      }
    }
  }
  for (uint32_t i = 0; i < 8; ++i) {
    vs_shared.bool_words[i] = ps_shared.bool_words[i] =
        ReadU32(r.device + dev::kBoolConstants + i * 4);
  }
  if (s.dumping) {
    REXGPU_INFO("PGR4 D3D12 dump:   bools {:08X} {:08X} {:08X} {:08X} {:08X} {:08X} {:08X} {:08X}",
                ps_shared.bool_words[0], ps_shared.bool_words[1], ps_shared.bool_words[2], ps_shared.bool_words[3],
                ps_shared.bool_words[4], ps_shared.bool_words[5], ps_shared.bool_words[6], ps_shared.bool_words[7]);
  }
  const uint32_t color_control = ReadRegister(r.device, kRegColorControl);
  const uint32_t alpha_func = color_control & 7;
  if ((color_control >> 3) & 1 && alpha_func != 7) {
    ps_shared.spec_constants |= kSpecAlphaTest;
    ps_shared.alpha_threshold =
        alpha_func == 0 ? 2.0f : AsFloat(ReadRegister(r.device, kRegAlphaRef));
  }
  // ALPHA_TO_MASK_ENABLE: Xenos turns alpha into MSAA coverage (foliage,
  // crowds, fences). Host targets are single-sampled, so half coverage is
  // the cut-off.
  if ((color_control >> 4) & 1) {
    ps_shared.alpha_threshold = (ps_shared.spec_constants & kSpecAlphaTest)
                                    ? std::max(ps_shared.alpha_threshold, 0.5f)
                                    : 0.5f;
    ps_shared.spec_constants |= kSpecAlphaTest;
  }
  if (input->r11g11b10) {
    vs_shared.spec_constants |= kSpecR11G11B10Normal;
  }
  vs_shared.input_modes = input->input_modes;
  vs_shared.input_missing = input->input_missing;
  const auto indexed_t0 = ProfNow();
  const int32_t fetch_safe = REXCVAR_GET(gpu_safe);
  if (vs->cached->indexed_fetch) {
    ++g_stream_stats.indexed_draws;
    // Layouts of the VS inputs (declaration order) and raw views of their
    // streams, endian-swapped only: the shader reads guest component order.
    const plume::RenderBuffer *raw[kTableStreamCount] = {};
    uint64_t raw_ids[kTableStreamCount] = {};
    const auto &inputs = vs->cached->inputs;
    for (uint32_t k = 0; k < inputs.size() && k < 16; ++k) {
      for (const auto &e : decl->elements) {
        if (e.usage != inputs[k].usage || e.usage_index != inputs[k].usage_index) {
          continue;
        }
        const uint32_t stream = e.stream & 15;
        const uint8_t *fc = GuestVirtual(r.device + dev::kFetchConstants + (95 - stream) * 8);
        const uint32_t d0 = LoadBE32(fc), d1 = LoadBE32(fc + 4);
        const uint32_t address = d0 & 0xFFFFFFFCu;
        const uint32_t size = ((d1 >> 2) & 0xFFFFFF) * 4;
        const uint32_t stride = uint32_t(GuestVirtual(r.device + dev::kStreamStrides)[stream]) * 4;
        if (!address || !size || r.up_data || (fetch_safe & 128)) {
          break;
        }
        if (!raw[stream] && !(fetch_safe & 64)) {
          if (const HostBuffer *hb = BufferFor(s, GuestPhysical(address), size, d1 & 3, {},
                                               REX_KERNEL_MEMORY()->TranslatePhysical(address))) {
            raw[stream] = hb->buffer.get();
            raw_ids[stream] = hb->id;
          }
        }
        if (raw[stream] || (fetch_safe & 64)) {
          vs_shared.fetch_layouts[k * 2] = 1u | (stream << 1) | ((e.offset & 0x7FFu) << 5) | (stride << 16);
          vs_shared.fetch_layouts[k * 2 + 1] = e.type;
        }
        break;
      }
    }
    if (auto *set = StreamTableFor(s, raw, raw_ids)) {
      BindSet(s, cmd, set, 4);
    }
  } else {
    // Every table in the layout must be populated on binding tier 1, used or not.
    const plume::RenderBuffer *none[kTableStreamCount] = {};
    const uint64_t no_ids[kTableStreamCount] = {};
    if (auto *set = StreamTableFor(s, none, no_ids)) {
      BindSet(s, cmd, set, 4);
    }
  }
  g_stream_stats.indexed_ns += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                            ProfNow() - indexed_t0)
                                            .count());
  vs_shared.base_vertex = r.indexed && !rects ? r.base_vertex : 0;

  // Viewport (SetViewport). Screen-space passes (PGR4's luminance and blur
  // downsamples) set a 65535x65535 viewport, which the Xenos takes as given.
  // D3D12 caps viewports at 32767 and drivers differ past that: NVIDIA keeps
  // the requested mapping, AMD clamps it and the pass lands outside its
  // target. So an oversized viewport is shrunk by a power of two k (exact in
  // float) and positions are scaled by k with an offset of (k-1, 1-k) in clip
  // space, which maps every vertex to the same pixel as the guest viewport.
  Viewport vp = s.viewports[r.device];
  if (!vp.valid || vp.width <= 0 || vp.height <= 0) {
    vp = {0, 0, float(targets.width), float(targets.height), 0, 1, true};
  }
  constexpr float kMaxViewportSize = 16384.0f;
  // The luminance and blur downsamples pass SetViewport a 65535x65535
  // D3DVIEWPORT9; for those, take the PA_CL_VPORT registers' mapping when it
  // is a plain D3D one (y down). Other draws keep the struct: the registers'
  // shadow does not track it (the start screen movie lands squashed).
  const bool oversized = std::max(vp.width, vp.height) > kMaxViewportSize;
  const uint32_t vte = ReadRegister(r.device, kRegVteControl);
  const float vx_scale = AsFloat(ReadRegister(r.device, kRegViewportXScale));
  const float vx_offset = AsFloat(ReadRegister(r.device, kRegViewportXScale + 1));
  const float vy_scale = AsFloat(ReadRegister(r.device, kRegViewportXScale + 2));
  const float vy_offset = AsFloat(ReadRegister(r.device, kRegViewportXScale + 3));
  if (s.dumping) {
    REXGPU_INFO("PGR4 D3D12 dump:   vport regs vte {:X} x scale {} offset {} y scale {} offset {}", vte, vx_scale,
                vx_offset, vy_scale, vy_offset);
  }
  if (oversized && (vte & 0xF) == 0xF && vx_scale > 0 && vy_scale < 0 && std::isfinite(vx_offset) &&
      std::isfinite(vy_offset) && !(fetch_safe & 256)) {
    vp.x = vx_offset - vx_scale;
    vp.y = vy_offset + vy_scale;
    vp.width = 2 * vx_scale;
    vp.height = -2 * vy_scale;
  }
  // Host pixels per guest pixel (internal resolution); the shrink applies to
  // the host viewport.
  const Scale2 res_scale = targets.scale;
  float viewport_shrink = 1.0f;
  while (std::max(vp.width * res_scale.x, vp.height * res_scale.y) / viewport_shrink > kMaxViewportSize) {
    viewport_shrink *= 2.0f;
  }
  vs_shared.ndc_scale[0] = vs_shared.ndc_scale[1] = viewport_shrink;
  vs_shared.half_pixel_offset[0] = viewport_shrink - 1.0f;
  vs_shared.half_pixel_offset[1] = 1.0f - viewport_shrink;
  // VTE with the xy scale and offset off (the Bink movie frames): the VS
  // writes window pixels, so map them to clip space over the whole target.
  if (!(vte & 0xF)) {
    vp = {0, 0, float(targets.width), float(targets.height), vp.min_z, vp.max_z, true};
    viewport_shrink = 1.0f;
    vs_shared.ndc_scale[0] = 2.0f / float(targets.width);
    vs_shared.ndc_scale[1] = -2.0f / float(targets.height);
    vs_shared.half_pixel_offset[0] = -1.0f;
    vs_shared.half_pixel_offset[1] = 1.0f;
  }
  vs_shared.ui_squeeze = res_scale.x > res_scale.y && vs->cached->screen_2d ? res_scale.y / res_scale.x : 1.0f;
  vs_shared.ui_squeeze_y = res_scale.y > res_scale.x && vs->cached->screen_2d ? res_scale.x / res_scale.y : 1.0f;
  if (s.dumping && res_scale.x != res_scale.y) {
    REXGPU_INFO("PGR4 D3D12 dump:   screen_2d {} vte {:X}", vs->cached->screen_2d, vte);
  }
  // A 3D draw onto a screen that has only had UI this frame (the menus'
  // spinning medals, in their panels) is part of that UI: it squeezes towards
  // the centre in clip space like the 2D around it. A scene's 3D comes first
  // on its target (wide_scene_drawn), so it keeps the full view.
  const bool ui_3d = res_scale.x != res_scale.y && !vs->cached->screen_2d && (vte & 0xF) && targets.color_count &&
                     !targets.color[0]->wide_scene_drawn && targets.color[0]->wide_bars_clean;
  if (ui_3d) {
    const int axis = res_scale.x > res_scale.y ? 0 : 1;
    const float q = axis == 0 ? res_scale.y / res_scale.x : res_scale.x / res_scale.y;
    vs_shared.ndc_scale[axis] *= q;
    vs_shared.half_pixel_offset[axis] *= q;
  }
  vs_shared.depth_offset = depth_offset;
  if (key.vs == kUiDialVs) {
    s.dial_drawn = true;
  }
  // A small 2D draw spanning the whole 16:9 width (the race intro's banners,
  // whose rounded ends sit off screen) fills the window instead. In a race,
  // HUD elements also move out to the window's sides: a 2D draw in the left
  // or right third of the 16:9 screen shifts by the extra width there, and
  // one in the middle stays centred.
  if (vs_shared.ui_squeeze < 1.0f && (s.race_hud || r.count <= 24)) {
    float lo = 0, hi = 0;
    if (UiDrawExtent(r, *decl, key.vs, vs_consts, vs_regs, lo, hi)) {
      const float q = vs_shared.ui_squeeze;
      const float centre = 0.5f * (lo + hi);
      if (lo <= -0.995f && hi >= 0.995f && r.count <= 24) {
        vs_shared.ui_squeeze = 1.0f;
      } else if (s.race_hud && centre < -1.0f / 3.0f) {
        vs_shared.half_pixel_offset[0] += q - 1.0f;
      } else if (s.race_hud && centre > 1.0f / 3.0f) {
        vs_shared.half_pixel_offset[0] += 1.0f - q;
      }
    }
  }
  // The squeeze itself: PosScale (c45) maps the 2D shaders' positions to clip
  // space, and some derive screen-space texture coordinates from the result
  // (the frosted glass behind menu panels samples the scene there), so
  // scaling its x keeps those in step with the squeezed position. The vertex
  // shader widens full-screen quads back out (build_shader_cache.py).
  // 16:10: the same in y. Race HUD elements in the top or bottom third move
  // out to the taller screen's edges.
  if (vs_shared.ui_squeeze_y < 1.0f && (s.race_hud || r.count <= 24)) {
    // Elements that move: the race HUD, and edge-to-edge bands (the race
    // intro's banners). Menu panels' pieces are narrower and span the
    // thirds, so they are left whole.
    float x_lo = 0, x_hi = 0;
    const bool over_scene = s.race_hud || (UiDrawExtent(r, *decl, key.vs, vs_consts, vs_regs, x_lo, x_hi, 0) &&
                                           x_lo <= -0.995f && x_hi >= 0.995f);
    float lo = 0, hi = 0;
    const bool have_extent = UiDrawExtent(r, *decl, key.vs, vs_consts, vs_regs, lo, hi, 1);
    if (s.dumping) {
      REXGPU_INFO("PGR4 D3D12 dump:   ui y extent {} {}..{} x {}..{} hud {}", have_extent, lo, hi, x_lo, x_hi,
                  s.race_hud);
    }
    if (have_extent) {
      const float q = vs_shared.ui_squeeze_y;
      const float centre = 0.5f * (lo + hi);
      if (lo <= -0.995f && hi >= 0.995f && r.count <= 24) {
        vs_shared.ui_squeeze_y = 1.0f;
      } else if (s.race_hud && lo < -1.0f / 3.0f && hi > 1.0f / 3.0f) {
        vs_shared.ui_edge_shift_y = 1.0f - q;
      } else if (over_scene && (hi >= 0.995f || centre > 1.0f / 3.0f)) {
        // Over a race (HUD, the intro's banners, made of several pieces not
        // all touching the edge): an element in the top or bottom third moves
        // out to that edge, keeping its shape. Menus are left whole: their
        // panels' pieces span the thirds.
        vs_shared.half_pixel_offset[1] += 1.0f - q;
      } else if (over_scene && (lo <= -0.995f || centre < -1.0f / 3.0f)) {
        vs_shared.half_pixel_offset[1] += q - 1.0f;
      }
    }
  }
  if (vs_shared.ui_squeeze < 1.0f && vs_regs > 45) {
    for (const uint32_t lane : {45u * 4 + 0, 45u * 4 + 2}) {
      const float v = AsFloat(vs_consts[lane]) * vs_shared.ui_squeeze;
      std::memcpy(&vs_consts[lane], &v, 4);
    }
  }
  if (vs_shared.ui_squeeze_y < 1.0f && vs_regs > 45) {
    for (const uint32_t lane : {45u * 4 + 1, 45u * 4 + 3}) {
      const float v = AsFloat(vs_consts[lane]) * vs_shared.ui_squeeze_y;
      std::memcpy(&vs_consts[lane], &v, 4);
    }
  }
  if (targets.color_count) {
    const auto range = OutputRange(targets.color[0]->guest_format);
    std::copy(range.begin(), range.end(), ps_shared.output_range);
  }
  vs_shared.pixel_scale[0] = ps_shared.pixel_scale[0] = 1.0f / res_scale.x;
  vs_shared.pixel_scale[1] = ps_shared.pixel_scale[1] = 1.0f / res_scale.y;
  phases.Lap(8);
  if (s.dumping) {
    REXGPU_INFO("PGR4 D3D12 dump:   vs c0 {} {} {} {} c1 {} {} {} {} ndc {} {} half {} {} squeeze {} pixel {} {}",
                AsFloat(vs_consts[0]), AsFloat(vs_consts[1]), AsFloat(vs_consts[2]), AsFloat(vs_consts[3]),
                AsFloat(vs_consts[4]), AsFloat(vs_consts[5]), AsFloat(vs_consts[6]), AsFloat(vs_consts[7]),
                vs_shared.ndc_scale[0], vs_shared.ndc_scale[1], vs_shared.half_pixel_offset[0],
                vs_shared.half_pixel_offset[1], vs_shared.ui_squeeze, vs_shared.pixel_scale[0],
                vs_shared.pixel_scale[1]);
  }
  const auto &vsc = UploadConstants(s.vs_constants, vs_consts, vs_regs * 16);
  const auto &psc = UploadConstants(s.ps_constants, ps_consts, ps_regs * 16);
  const auto &vss = UploadConstants(s.vs_shared, &vs_shared, sizeof(vs_shared));
  const auto &pss = UploadConstants(s.ps_shared, &ps_shared, sizeof(ps_shared));
  if (!vsc || !psc || !vss || !pss) {
    ++s.skipped_draws;
    return;
  }
  BindRoot(s, cmd, plume::RenderBufferReference(vsc.buffer, vsc.offset), kRootVsConstants);
  BindRoot(s, cmd, plume::RenderBufferReference(psc.buffer, psc.offset), kRootPsConstants);
  BindRoot(s, cmd, plume::RenderBufferReference(pss.buffer, pss.offset), kRootPsShared);
  BindRoot(s, cmd, plume::RenderBufferReference(vss.buffer, vss.offset), kRootVsShared);

  // Ultrawide: the screen target is wider than the guest's 16:9, and the 3D
  // view fills it. PGR4's 2D shaders (sprites, text, HUD) are squeezed
  // through their PosScale (above, ui_squeeze): elements keep their shape and
  // full-screen quads (a pasted scene copy, tints) still fill the screen. The
  // Bink movies are squeezed whole, to a centred 16:9 picture unless they
  // paste a widened scene copy. 16:10 (tall) is the same on y.
  const bool wide = res_scale.x > res_scale.y;
  const bool tall = res_scale.y > res_scale.x;
  const bool ui_draw = (wide && vs_shared.ui_squeeze < 1.0f) || (tall && vs_shared.ui_squeeze_y < 1.0f) || ui_3d;
  const bool squeeze_ui = (wide || tall) && !(vte & 0xF) && !s.draw_samples_wide;
  const float host_cx = float(ScaledSize(targets.width, res_scale.x)) * 0.5f;
  const float host_cy = float(ScaledSize(targets.height, res_scale.y)) * 0.5f;
  const float squeeze = squeeze_ui && wide ? res_scale.y / res_scale.x : 1.0f;
  const float squeeze_y = squeeze_ui && tall ? res_scale.x / res_scale.y : 1.0f;
  auto host_x = [&](float guest_x) { return host_cx + (guest_x * res_scale.x - host_cx) * squeeze; };
  auto host_y = [&](float guest_y) { return host_cy + (guest_y * res_scale.y - host_cy) * squeeze_y; };
  // Host viewport (scaled to the internal resolution, shrunk above if
  // oversized), clipped scissor.
  BindViewport(s, cmd, plume::RenderViewport(host_x(vp.x), host_y(vp.y),
                                             vp.width * res_scale.x * squeeze / viewport_shrink,
                                             vp.height * res_scale.y * squeeze_y / viewport_shrink, vp.min_z,
                                             vp.max_z));
  int32_t sx0 = std::max(0, int32_t(vp.x)), sy0 = std::max(0, int32_t(vp.y));
  int32_t sx1 = std::min(int32_t(targets.width), int32_t(vp.x + vp.width));
  int32_t sy1 = std::min(int32_t(targets.height), int32_t(vp.y + vp.height));
  // Guest scissor (UI lists clip their entries to the panel with it).
  if (ReadU32(r.device + dev::kScissorEnable)) {
    sx0 = std::max(sx0, int32_t(ReadU32(r.device + dev::kScissorRect)));
    sy0 = std::max(sy0, int32_t(ReadU32(r.device + dev::kScissorRect + 4)));
    sx1 = std::min(sx1, int32_t(ReadU32(r.device + dev::kScissorRect + 8)));
    sy1 = std::min(sy1, int32_t(ReadU32(r.device + dev::kScissorRect + 12)));
  }
  // A squeezed 2D draw's scissor (list panels, the scrolling radio toast)
  // squeezes with its vertices; an edge on the screen's side stays there,
  // like the shader keeps full-screen quads.
  auto scissor_x = [&](int32_t guest_x) {
    if (ui_draw && wide && guest_x > 0 && guest_x < int32_t(targets.width)) {
      return int32_t(std::lround(host_cx + (float(guest_x) * res_scale.x - host_cx) * (res_scale.y / res_scale.x)));
    }
    return int32_t(std::lround(host_x(float(guest_x))));
  };
  auto scissor_y = [&](int32_t guest_y) {
    if (ui_draw && tall && guest_y > 0 && guest_y < int32_t(targets.height)) {
      return int32_t(std::lround(host_cy + (float(guest_y) * res_scale.y - host_cy) * (res_scale.x / res_scale.y)));
    }
    return int32_t(std::lround(host_y(float(guest_y))));
  };
  BindScissor(s, cmd, plume::RenderRect(scissor_x(sx0), scissor_y(sy0), scissor_x(std::max(sx0, sx1)),
                                        scissor_y(std::max(sy0, sy1))));
  // A 2D-only screen (menus over a backdrop image, the movies) never draws
  // outside the middle 16:9, which would otherwise keep old frames there.
  if ((wide || tall) && targets.color_count) {
    RenderTarget *screen = targets.color[0];
    if (!squeeze_ui && !ui_draw) {
      screen->wide_scene_drawn = true;
    } else if (!screen->wide_scene_drawn && !screen->wide_bars_clean) {
      const int32_t host_w = int32_t(ScaledSize(targets.width, res_scale.x));
      const int32_t host_h = int32_t(ScaledSize(targets.height, res_scale.y));
      const int32_t left = int32_t(std::lround(host_x(0.0f)));
      const int32_t right = int32_t(std::lround(host_x(float(targets.width))));
      const int32_t top = int32_t(std::lround(host_y(0.0f)));
      const int32_t bottom = int32_t(std::lround(host_y(float(targets.height))));
      if (wide && left > 0 && right < host_w) {
        const plume::RenderRect bars[] = {plume::RenderRect(0, 0, left, host_h),
                                          plume::RenderRect(right, 0, host_w, host_h)};
        cmd->clearColor(0, plume::RenderColor(0.0f, 0.0f, 0.0f, 1.0f), bars, 2);
      }
      if (tall && top > 0 && bottom < host_h) {
        const plume::RenderRect bars[] = {plume::RenderRect(0, 0, host_w, top),
                                          plume::RenderRect(0, bottom, host_w, host_h)};
        cmd->clearColor(0, plume::RenderColor(0.0f, 0.0f, 0.0f, 1.0f), bars, 2);
      }
      screen->wide_bars_clean = true;
    }
  }
  if (s.dumping) {
    REXGPU_INFO("PGR4 D3D12 dump:   viewport {} {} {}x{} (host /{}) z {}..{} scissor {} {} {} {}", vp.x, vp.y,
                vp.width, vp.height, viewport_shrink, vp.min_z, vp.max_z, sx0, sy0, std::max(sx0, sx1),
                std::max(sy0, sy1));
  }

  phases.Lap(9);
  const int32_t base_vertex = r.indexed && !rects ? r.base_vertex : quad_base_vertex;
  Video::NoteDraw({key.vs, key.ps, draw_count, indexed ? first_index : r.first, indexed ? base_vertex : 0, indexed});
  if (indexed) {
    plume::RenderIndexBufferView ibv(plume::RenderBufferReference(index_buffer, index_offset), index_size,
                                     index_format);
    cmd->setIndexBuffer(&ibv);
    cmd->drawIndexedInstanced(draw_count, 1, first_index, base_vertex, 0);
  } else {
    cmd->drawInstanced(draw_count, 1, r.first, 0);
  }
  Video::NoteDrawDone();

  // Dump: progressive captures of the main scene, every dump_every draws
  // from dump_from (within a resolve segment), at most 100 of them.
  uint32_t dump_every = uint32_t(std::max(1, REXCVAR_GET(dump_every)));
  uint32_t dump_from = uint32_t(std::max(0, REXCVAR_GET(dump_from)));
  // dump.range ("<from> <every>") overrides both while it exists, so the
  // window can move between dumps without a restart.
  if (s.dumping) {
    if (std::FILE *rf = std::fopen("dump.range", "rb")) {
      unsigned from = 0, every = 0;
      if (std::fscanf(rf, "%u %u", &from, &every) == 2 && every) {
        dump_from = from;
        dump_every = every;
      }
      std::fclose(rf);
    }
  }
  if (s.dumping && targets.color_count && targets.width == 1280 && s.draws_since_resolve >= dump_from &&
      (s.draws_since_resolve - dump_from) % dump_every == 0 &&
      s.draws_since_resolve - dump_from <= 100 * dump_every) {
    char name[64];
    std::snprintf(name, sizeof(name), "dump_%03u_draw%04u", s.dump_seq, s.draws_since_resolve);
    Video::CaptureTexture(targets.color[0]->texture.get(), ScaledSize(targets.width, targets.scale.x),
                          ScaledSize(targets.height, targets.scale.y),
                          name, true);
    s.bound_framebuffer = nullptr;
    s.bound = {};
  }
}

} // namespace

// ---------------------------------------------------------------------------

namespace {
void SyncRenderThread();
} // namespace

void Renderer::OnShaderCreated(uint32_t shader_object, uint32_t container, bool pixel) {
  if (!shader_object || !container) {
    return;
  }
  SyncRenderThread();
  auto &s = state();
  std::lock_guard lock(s.mutex);
  if (!s.device) {
    s.device = Video::HostDevice();
    ShaderCache::Load(s.device);
  }
  RegisterShader(s, shader_object, container, pixel);
}

void ExecResolve(uint32_t device, uint32_t flags, uint32_t source_rect, uint32_t dest_texture,
                 uint32_t dest_point, uint32_t dest_level, uint32_t dest_slice, uint32_t clear_color,
                 float clear_z);

void ExecOnZPass(uint32_t device, bool begin) {
  auto &s = state();
  std::lock_guard lock(s.mutex);
  if (LogOnce(s, 0xB000000000ull | begin)) {
    REXGPU_INFO("PGR4 D3D12: Z pass {} (dev {:08X}), first at frame {}", begin ? "begins" : "ends", device, s.frame);
  }
  if (s.dumping) {
    REXGPU_INFO("PGR4 D3D12 dump: Z pass {} after draw {}", begin ? "begins" : "ends", s.draws);
  }
}

void ExecOnSetTexture(uint32_t device, uint32_t sampler, uint32_t texture) {
  auto &s = state();
  std::lock_guard lock(s.mutex);
  if (s.dumping) {
    const uint8_t *fc = GuestVirtual(device + dev::kFetchConstants + (sampler & 31) * 24);
    REXGPU_INFO("PGR4 D3D12 dump: SetTexture({}, {:08X}) dev {:08X} after draw {}, fetch now {:08X} {:08X}", sampler, texture, device,
                s.draws_since_resolve, LoadBE32(fc), LoadBE32(fc + 4));
  }
}

void ExecOnViewport(uint32_t device, uint32_t viewport) {
  if (!viewport) {
    return;
  }
  auto &s = state();
  std::lock_guard lock(s.mutex);
  // D3DVIEWPORT9: X, Y, Width, Height (DWORD), MinZ, MaxZ (float).
  Viewport &vp = s.viewports[device];
  vp.x = float(ReadU32(viewport));
  vp.y = float(ReadU32(viewport + 4));
  vp.width = float(ReadU32(viewport + 8));
  vp.height = float(ReadU32(viewport + 12));
  vp.min_z = AsFloat(ReadU32(viewport + 16));
  vp.max_z = AsFloat(ReadU32(viewport + 20));
  vp.valid = true;
}

void ExecDrawVertices(uint32_t device, uint32_t primitive, uint32_t start_vertex,
                            uint32_t vertex_count) {
  auto &s = state();
  std::lock_guard lock(s.mutex);
  DrawRequest r;
  r.device = device;
  r.primitive = primitive;
  r.first = start_vertex;
  r.count = vertex_count;
  ExecuteDraw(s, r);
}

void ExecDrawIndexed(uint32_t device, uint32_t primitive, int32_t base_vertex,
                           uint32_t start_index, uint32_t index_count) {
  auto &s = state();
  std::lock_guard lock(s.mutex);
  DrawRequest r;
  r.device = device;
  r.primitive = primitive;
  r.indexed = true;
  r.first = start_index;
  r.count = index_count;
  r.base_vertex = base_vertex;
  ExecuteDraw(s, r);
}

void ExecDrawUserPointer(uint32_t device, uint32_t primitive, uint32_t vertex_count,
                               uint32_t data, uint32_t stride) {
  if (!data || !stride) {
    return;
  }
  auto &s = state();
  std::lock_guard lock(s.mutex);
  DrawRequest r;
  r.device = device;
  r.primitive = primitive;
  r.count = vertex_count;
  r.up_data = data;
  r.up_stride = stride;
  ExecuteDraw(s, r);
}

namespace {

void ClearTargets(RendererState &s, uint32_t device, bool color, const float rgba[4], bool depth,
                  bool stencil, float z, uint32_t stencil_value) {
  if (!EnsureReady(s)) {
    return;
  }
  plume::RenderCommandList *cmd = Video::OpenCommandList();
  if (!cmd) {
    return;
  }
  PrepareFrame(s);
  Targets t = CurrentTargets(s, device);
  if (s.dumping) {
    REXGPU_INFO("PGR4 D3D12 dump: clear color {} ({} {} {} {}) depth {} stencil {} z {} surfaces {:08X} / {:08X} {}x{}",
                color, rgba[0], rgba[1], rgba[2], rgba[3], depth, stencil, z, ReadU32(device + dev::kRenderTargets),
                ReadU32(device + dev::kDepthStencil), t.width, t.height);
  }
  if (!t.width || !BindFramebuffer(s, cmd, t)) {
    return;
  }
  BindScissor(s, cmd, plume::RenderRect(0, 0, int32_t(ScaledSize(t.width, t.scale.x)),
                                        int32_t(ScaledSize(t.height, t.scale.y))));
  if (color) {
    for (uint32_t i = 0; i < t.color_count; ++i) {
      cmd->clearColor(i, plume::RenderColor(rgba[0], rgba[1], rgba[2], rgba[3]));
      t.color[i]->wide_scene_drawn = t.color[i]->wide_bars_clean = false;
    }
  }
  if (t.depth && (depth || stencil)) {
    cmd->clearDepthStencil(depth, stencil, z, stencil_value);
  }
  // The same EDRAM seen through other surfaces (RenderTarget::edram_base).
  for (auto &[_, rt] : s.render_targets) {
    if (!REXCVAR_GET(edram_clear_sharing)) {
      break;
    }
    if (!rt || rt->edram_base == ~0u) {
      continue;
    }
    // Depth only, and only onto views the cleared one covers: a clear wipes
    // just its own area of EDRAM. Sharing a colour clear, or a small pass's
    // depth clear onto the full-screen scene depth, turned the menu car's
    // windows red.
    if (rt->depth && t.depth && (depth || stencil) && rt.get() != t.depth &&
        rt->edram_base == t.depth->edram_base && rt->width <= t.depth->width &&
        rt->height <= t.depth->height) {
      if (s.dumping) {
        REXGPU_INFO("PGR4 D3D12 dump:   depth clear shared from id {} {}x{} to id {} {}x{} (EDRAM base {:X})",
                    t.depth->id, t.depth->width, t.depth->height, rt->id, rt->width, rt->height, rt->edram_base);
      }
      // A pending clear of the other part stays (depth now, stencil before).
      rt->clear_depth = depth || (rt->clear_pending && rt->clear_depth);
      rt->clear_stencil = stencil || (rt->clear_pending && rt->clear_stencil);
      if (depth) {
        rt->clear_z = z;
      }
      if (stencil) {
        rt->clear_stencil_value = stencil_value;
      }
      if (!rt->clear_pending) {
        s.shared_clears.push_back(rt.get());
      }
      rt->clear_pending = true;
    }
  }
}


} // namespace

void ExecClear(uint32_t device, uint32_t flags, uint32_t argb, float z, uint32_t stencil) {
  auto &s = state();
  std::lock_guard lock(s.mutex);
  const float rgba[4] = {float((argb >> 16) & 0xFF) / 255.0f, float((argb >> 8) & 0xFF) / 255.0f,
                         float(argb & 0xFF) / 255.0f, float(argb >> 24) / 255.0f};
  ClearTargets(s, device, (flags & 0xF) != 0, rgba, (flags & 0x10) != 0, (flags & 0x20) != 0, z,
               stencil);
}

void ExecBeginTiling(uint32_t device, uint32_t rect_count, uint32_t rects, uint32_t clear_color,
                           float z, uint32_t stencil) {
  auto &s = state();
  std::lock_guard lock(s.mutex);
  // D3DRECT {x1, y1, x2, y2}: the targets cover the union of the tiles.
  uint32_t width = 0, height = 0;
  for (uint32_t i = 0; rects && i < rect_count && i < 16; ++i) {
    width = std::max(width, ReadU32(rects + i * 16 + 8));
    height = std::max(height, ReadU32(rects + i * 16 + 12));
  }
  s.tiling_active = width && height && width <= 8192 && height <= 8192;
  s.tiling_width = width;
  s.tiling_height = height;
  s.bound_framebuffer = nullptr;
  s.bound = {};
  float rgba[4] = {0, 0, 0, 0};
  if (clear_color) {
    for (int i = 0; i < 4; ++i) {
      rgba[i] = AsFloat(ReadU32(clear_color + i * 4));
    }
  }
  ClearTargets(s, device, clear_color != 0, rgba, true, true, z, stencil);
}

void ExecEndTiling(uint32_t device, uint32_t flags, uint32_t dest_texture, uint32_t clear_color,
                         float clear_z) {
  ExecResolve(device, flags, 0, dest_texture, 0, 0, 0, clear_color, clear_z);
  auto &s = state();
  std::lock_guard lock(s.mutex);
  s.tiling_active = false;
  s.bound_framebuffer = nullptr;
  s.bound = {};
}

void ExecResolve(uint32_t device, uint32_t flags, uint32_t source_rect, uint32_t dest_texture,
                       uint32_t dest_point, uint32_t dest_level, uint32_t dest_slice, uint32_t clear_color,
                       float clear_z) {
  auto &s = state();
  std::lock_guard lock(s.mutex);
  RendererTimer timer(0);
  if (!EnsureReady(s)) {
    return;
  }
  plume::RenderCommandList *cmd = Video::OpenCommandList();
  if (!cmd) {
    return;
  }
  PrepareFrame(s);
  const uint32_t source = flags & 7;
  const bool from_depth = source == 4;
  ResolvedTexture *dumped_resolve = nullptr;
  Targets t = CurrentTargets(s, device);
  RenderTarget *src = from_depth ? t.depth : (source < t.color_count ? t.color[source] : nullptr);
  if (src && (src->clear_pending || src->needs_init)) {
    BindFramebuffer(s, cmd, t); // applies the pending clear before the copy
  }

  if (dest_texture && src) {
    // The destination texture's fetch constant lives at +0x1C (SetTexture
    // copies it from there).
    const TextureFetch f = ReadTextureFetch(GuestVirtual(dest_texture + 0x1C));
    const uint32_t base = f.BaseAddress();
    const uint32_t dw = std::max(1u, f.Width()), dh = std::max(1u, f.Height());
    // Cube destinations (dynamic environment maps) get one Resolve per face.
    const bool cube = !from_depth && f.Dimension() == 3;
    // A 2D texture with mips gets one Resolve per level (the car's soft
    // ground shadow halves a 128x128 image down its chain).
    const uint32_t hw = ScaledSize(dw, src->scale.x), hh = ScaledSize(dh, src->scale.y);
    const uint32_t levels = from_depth || cube
                                ? 1u
                                : std::min((f.dword[4] >> 6) & 0xF, uint32_t(std::bit_width(std::max(hw, hh))) - 1) + 1;
    const uint32_t level = std::min(dest_level, levels - 1);
    const uint64_t rkey = (uint64_t(base) << 32) ^ (uint64_t(dw) << 18) ^ (uint64_t(dh) << 4) ^
                          (uint64_t(src->format) << 50) ^ (cube ? 1ull << 63 : 0);
    auto &r = s.resolved[rkey];
    if (!r || r->width != dw || r->height != dh || r->format != src->format || r->cube != cube ||
        r->scale != src->scale || r->levels != levels) {
      // The copy is at the internal resolution, like its source target.
      plume::RenderTextureDesc desc =
          from_depth ? plume::RenderTextureDesc::DepthTarget(hw, hh, src->format)
          : cube     ? plume::RenderTextureDesc::Texture(plume::RenderTextureDimension::TEXTURE_2D, hw, hh, 1, 1, 6,
                                                         src->format, plume::RenderTextureFlag::CUBE)
                     : plume::RenderTextureDesc::Texture2D(hw, hh, levels, src->format);
      auto texture = s.device->createTexture(desc);
      if (texture) {
        texture->setName(fmt::format("resolve {:08X} {}x{}", base, dw, dh));
        if (r) {
          Video::ParkUntilFence(std::move(r->texture));
          for (auto &[_, v] : r->views) {
            Video::ParkUntilFence(std::move(v));
          }
        } else {
          r = std::make_unique<ResolvedTexture>();
        }
        r->texture = std::move(texture);
        r->views.clear();
        r->format = src->format;
        r->width = dw;
        r->height = dh;
        r->scale = src->scale;
        r->levels = levels;
        r->written_levels = 0;
        r->depth = from_depth;
        r->cube = cube;
        r->id = s.next_id++;
        cmd->barriers(plume::RenderBarrierStage::GRAPHICS,
                      plume::RenderTextureBarrier(r->texture.get(), plume::RenderTextureLayout::SHADER_READ));
        REXGPU_INFO("PGR4 D3D12: resolve target {:08X} {}x{}{}", base, dw, dh, from_depth ? " (depth)" : "");
      }
    }
    if (r && r->texture) {
      // Cached texture sets stay valid unless this changes what a fetch of
      // the address binds (another copy, or a view with more levels); a
      // resolve into the same copy only changes its contents on the GPU.
      bool rebind = false;
      if (!(r->written_levels & (1u << level))) {
        r->written_levels |= 1u << level;
        for (auto &[_, v] : r->views) {
          Video::ParkUntilFence(std::move(v));
        }
        r->views.clear();
        rebind = true;
      }
      ResolvedTexture *&latest = s.latest_resolved[PhysicalKey(base)];
      if (latest != r.get()) {
        latest = r.get();
        rebind = true;
      }
      if (rebind) {
        ++s.binding_generation;
      }
      r->guest_base = PhysicalKey(base);
      r->guest_hash = GuestSampleHash(r->guest_base);
      r->checked_frame = s.frame;
      dumped_resolve = r.get();
      int32_t x0 = 0, y0 = 0, x1 = int32_t(src->width), y1 = int32_t(src->height);
      if (source_rect) {
        x0 = int32_t(ReadU32(source_rect));
        y0 = int32_t(ReadU32(source_rect + 4));
        x1 = int32_t(ReadU32(source_rect + 8));
        y1 = int32_t(ReadU32(source_rect + 12));
      }
      uint32_t dx = 0, dy = 0;
      if (dest_point) {
        dx = ReadU32(dest_point);
        dy = ReadU32(dest_point + 4);
      }
      const int32_t level_w = int32_t(std::max(1u, r->width >> level));
      const int32_t level_h = int32_t(std::max(1u, r->height >> level));
      x1 = std::min({x1, int32_t(src->width), x0 + level_w - int32_t(dx)});
      y1 = std::min({y1, int32_t(src->height), y0 + level_h - int32_t(dy)});
      const plume::RenderTextureLayout src_layout =
          from_depth ? plume::RenderTextureLayout::DEPTH_WRITE : plume::RenderTextureLayout::COLOR_WRITE;
      if (x1 > x0 && y1 > y0) {
        const plume::RenderTextureBarrier before[] = {
            plume::RenderTextureBarrier(src->texture.get(), plume::RenderTextureLayout::COPY_SOURCE),
            plume::RenderTextureBarrier(r->texture.get(), plume::RenderTextureLayout::COPY_DEST)};
        cmd->barriers(plume::RenderBarrierStage::COPY, before, 2);
        if (from_depth) {
          // Depth copies must cover whole subresources.
          if (r->width == src->width && r->height == src->height) {
            cmd->copyTexture(r->texture.get(), src->texture.get());
          }
        } else {
          // Guest rectangle to host pixels, kept inside both textures.
          const Scale2 k = src->scale;
          const int32_t hx0 = ScaledEdge(x0, k.x), hy0 = ScaledEdge(y0, k.y);
          const int32_t hdx = ScaledEdge(int32_t(dx), k.x), hdy = ScaledEdge(int32_t(dy), k.y);
          const int32_t hx1 = std::min({ScaledEdge(x1, k.x), int32_t(ScaledSize(src->width, k.x)),
                                        hx0 + int32_t(std::max(1u, hw >> level)) - hdx});
          const int32_t hy1 = std::min({ScaledEdge(y1, k.y), int32_t(ScaledSize(src->height, k.y)),
                                        hy0 + int32_t(std::max(1u, hh >> level)) - hdy});
          if (hx1 > hx0 && hy1 > hy0) {
            const plume::RenderBox box(hx0, hy0, hx1, hy1);
            cmd->copyTextureRegion(
                plume::RenderTextureCopyLocation::Subresource(r->texture.get(), level,
                                                              r->cube ? std::min(dest_slice, 5u) : 0),
                plume::RenderTextureCopyLocation::Subresource(src->texture.get()), uint32_t(hdx), uint32_t(hdy), 0, &box);
          }
        }
        const plume::RenderTextureBarrier after[] = {
            plume::RenderTextureBarrier(src->texture.get(), src_layout),
            plume::RenderTextureBarrier(r->texture.get(), plume::RenderTextureLayout::SHADER_READ)};
        cmd->barriers(plume::RenderBarrierStage::GRAPHICS, after, 2);
      }
    }
  }

  if (s.dumping) {
    REXGPU_INFO("PGR4 D3D12 dump {:03}: resolve flags {:X} src id {} {}x{} fmt {} (guest {}) -> {:08X} (fmt {}, level {}, slice {}), {} draws",
                s.dump_seq, flags, src ? src->id : 0, src ? src->width : 0, src ? src->height : 0,
                src ? int(src->format) : -1, src ? int(src->guest_format) : -1, dest_texture,
                dest_texture ? ReadTextureFetch(GuestVirtual(dest_texture + 0x1C)).Format() : 0, dest_level, dest_slice,
                s.draws_since_resolve);
    if (dumped_resolve && !dumped_resolve->depth && !dumped_resolve->cube) {
      char name[64];
      std::snprintf(name, sizeof(name), "dump_%03u_%08X", s.dump_seq,
                    ReadTextureFetch(GuestVirtual(dest_texture + 0x1C)).BaseAddress());
      Video::CaptureTexture(dumped_resolve->texture.get(), ScaledSize(dumped_resolve->width, dumped_resolve->scale.x),
                            ScaledSize(dumped_resolve->height, dumped_resolve->scale.y), name);
    } else if (dumped_resolve && dumped_resolve->cube && src) {
      char name[64];
      std::snprintf(name, sizeof(name), "dump_%03u_cube%u", s.dump_seq, dest_slice);
      Video::CaptureTexture(src->texture.get(), ScaledSize(src->width, src->scale.x),
                            ScaledSize(src->height, src->scale.y), name, true);
      s.bound_framebuffer = nullptr;
      s.bound = {};
    }
    ++s.dump_seq;
  }
  s.draws_since_resolve = 0;

  // D3DRESOLVE_CLEARRENDERTARGET / CLEARDEPTHSTENCIL.
  if (flags & 0x300) {
    float rgba[4] = {0, 0, 0, 0};
    if (clear_color) {
      for (int i = 0; i < 4; ++i) {
        rgba[i] = AsFloat(ReadU32(clear_color + i * 4));
      }
    }
    if (s.dumping) { // TEMP
      REXGPU_INFO("PGR4 D3D12 dump: resolve clear ptr {:08X} words {:08X} {:08X} {:08X} {:08X}", clear_color,
                  clear_color ? ReadU32(clear_color) : 0, clear_color ? ReadU32(clear_color + 4) : 0,
                  clear_color ? ReadU32(clear_color + 8) : 0, clear_color ? ReadU32(clear_color + 12) : 0);
    }
    s.bound_framebuffer = nullptr;
    s.bound = {};
    ClearTargets(s, device, (flags & 0x100) != 0, rgba, (flags & 0x200) != 0,
                 (flags & 0x200) != 0, clear_z, 0);
  }
}

// Holds each Swap to the frame_rate_limit, like the 360's presentation
// interval holds PGR4 to 30 fps. A late frame moves the schedule instead of
// letting the next ones catch up in a burst.
static void PaceFrame() {
  using Clock = std::chrono::steady_clock;
  static Clock::time_point next;
  static HANDLE timer = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,
                                               TIMER_ALL_ACCESS);
  int32_t limit = REXCVAR_GET(frame_rate_limit);
  if (pgr4::WalkCameraActive() && (limit <= 0 || limit > 30)) {
    limit = 30;
  }
  if (limit <= 0) {
    next = {};
    return;
  }
  const auto period = std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(1.0 / limit));
  auto now = Clock::now();
  if (next == Clock::time_point{} || now - next > period) {
    next = now;
  }
  // Sleep to within a millisecond of the deadline, then spin the rest.
  while (next - now > std::chrono::milliseconds(1)) {
    const auto wait = next - now - std::chrono::milliseconds(1);
    if (timer) {
      LARGE_INTEGER due;
      due.QuadPart = -std::max<int64_t>(1, std::chrono::duration_cast<std::chrono::nanoseconds>(wait).count() / 100);
      SetWaitableTimer(timer, &due, 0, nullptr, nullptr, FALSE);
      WaitForSingleObject(timer, INFINITE);
    } else {
      Sleep(DWORD(std::chrono::duration_cast<std::chrono::milliseconds>(wait).count()));
    }
    now = Clock::now();
  }
  while (Clock::now() < next) {
    _mm_pause();
  }
  next += period;
}

// PGR4's post-process object (built by sub_823188D0, pointer at 0x82A93A8C)
// keeps the frame time at +292 for the HDR exposure adaptation (sub_8284E5A8:
// g_vSettings.x = it * ExposureAdaptSpeed). The constructor sets 1/30 and
// nothing updates it, since the 360 holds the game at 30 fps; at a higher
// frame rate the exposure adapts that much faster, and the bright moment at
// each race countdown beep drives the scene to black. Give it the real time.
constexpr uint32_t kPostProcessPtr = 0x82A93A8C;
constexpr uint32_t kPostProcessFrameTime = 292;

static void StoreFrameTime() {
  using Clock = std::chrono::steady_clock;
  static Clock::time_point last;
  const auto now = Clock::now();
  const double dt = last == Clock::time_point{} ? 1.0 / 30.0 : std::chrono::duration<double>(now - last).count();
  last = now;
  if (const uint32_t post = ReadU32(kPostProcessPtr)) {
    rex::memory::store_and_swap<float>(GuestVirtual(post + kPostProcessFrameTime),
                                       float(std::clamp(dt, 1.0 / 1000.0, 0.1)));
  }
}

// Debug: log every tweakable (PGR4's registry is a list from 0x82C91BA4,
// ended by the sentinel at 0x82C91BA0; name at +16, type at +12, value from
// +32), so dumps with and without the title update's data can be compared.
static void DumpTweakables() {
  uint32_t node = ReadU32(0x82C91BA4);
  for (int i = 0; node && node != 0x82C91BA0 && i < 20000; ++i, node = ReadU32(node + 4)) {
    const uint32_t name = ReadU32(node + 16);
    const char *text = name ? reinterpret_cast<const char *>(GuestVirtual(name)) : "";
    REXGPU_INFO("PGR4 tweak {} type {} {:08X} {:08X} {:08X} ({})", text, ReadU32(node + 12), ReadU32(node + 32),
                ReadU32(node + 36), ReadU32(node + 40), AsFloat(ReadU32(node + 32)));
  }
}

// Writes the tweaks cvar into PGR4's tweakable registry: type 1 bool (byte
// at +32), 2 int and 3 float (+32). Retried each frame until every name has
// registered, since some register late.
static void ApplyTweaks() {
  static bool done = false;
  if (done) {
    return;
  }
  const std::string spec = REXCVAR_GET(tweaks);
  if (spec.empty() || ReadU32(0x82C91BA4) == 0) {
    done = spec.empty();
    return;
  }
  size_t applied = 0, wanted = 0;
  for (size_t pos = 0; pos < spec.size();) {
    const size_t end = std::min(spec.find(';', pos), spec.size());
    const std::string item = spec.substr(pos, end - pos);
    pos = end + 1;
    const size_t eq = item.find('=');
    if (eq == std::string::npos) {
      continue;
    }
    ++wanted;
    const std::string name = item.substr(0, eq);
    const double value = std::strtod(item.c_str() + eq + 1, nullptr);
    uint32_t node = ReadU32(0x82C91BA4);
    for (int i = 0; node && node != 0x82C91BA0 && i < 20000; ++i, node = ReadU32(node + 4)) {
      const uint32_t text = ReadU32(node + 16);
      if (!text || name != reinterpret_cast<const char *>(GuestVirtual(text))) {
        continue;
      }
      switch (ReadU32(node + 12)) {
      case 1:
        *GuestVirtual(node + 32) = value != 0.0;
        break;
      case 2:
        rex::memory::store_and_swap<uint32_t>(GuestVirtual(node + 32), uint32_t(int32_t(value)));
        break;
      case 3:
        rex::memory::store_and_swap<float>(GuestVirtual(node + 32), float(value));
        break;
      }
      ++applied;
      break;
    }
  }
  if (applied == wanted || Video::FrameCount() > 600) {
    REXGPU_INFO("PGR4 D3D12: applied {} of {} tweaks", applied, wanted);
    done = true;
  }
}

// The 360 maps every frame through the display controller's gamma ramp at
// scan-out. PGR4's D3D keeps the ramp in the device (D3DDevice_Swap uploads
// it from there) at +0x3A8C: PGR4 uses the PWL form, per channel (red,
// green, blue) 128 BE word pairs {base, delta} over the 10-bit input, 8
// steps a segment, values 10-bit in the top bits. Its curve darkens the low
// end; without it the image looks washed out. The format at +0x3518 has
// low bits 0x36 for PWL; the 256-entry form is plain words.
constexpr uint32_t kDeviceGammaFormat = 0x3518;
constexpr uint32_t kDeviceGammaRamp = 0x3A8C;

static const uint16_t *ReadGammaRamp(uint32_t device) {
  static std::array<uint16_t, 768> ramp;
  if (!device || !REXCVAR_GET(gamma_ramp)) {
    return nullptr;
  }
  const uint8_t *src = GuestVirtual(device + kDeviceGammaRamp);
  auto word = [&](uint32_t i) { return uint32_t(src[i * 2] << 8 | src[i * 2 + 1]); };
  if ((ReadU32(device + kDeviceGammaFormat) & 0x3F) != 0x36) {
    for (uint32_t i = 0; i < 768; ++i) {
      ramp[i] = uint16_t(word(i));
    }
    return ramp.data();
  }
  for (uint32_t c = 0; c < 3; ++c) {
    for (uint32_t i = 0; i < 256; ++i) {
      const uint32_t x = (i * 1023 + 127) / 255; // 8-bit front buffer to the 10-bit input
      const uint32_t base = word(c * 256 + (x >> 3) * 2), delta = word(c * 256 + (x >> 3) * 2 + 1);
      ramp[c * 256 + i] = uint16_t(std::min<uint32_t>(base + delta * (x & 7) / 8, 0xFFC0) & 0xFFC0);
    }
  }
  return ramp.data();
}

void ExecSwap(uint32_t device, uint32_t front_buffer) {
  auto &s = state();
  g_profile = REXCVAR_GET(renderer_profile);
  if (const uint64_t n = Video::FrameCount(); n && n % 600 == 0) {
    std::string line;
    for (int i = 0; i < kDrawPhases; ++i) {
      line += fmt::format(" {} {:.1f}", kDrawPhaseNames[i], g_draw_phase_ns[i] * 1e-6 / 600.0);
      g_draw_phase_ns[i] = 0;
    }
    const auto &st = g_stream_stats;
    if (g_profile) {
      REXGPU_INFO("PGR4 D3D12: draw stages, ms/frame:{}", line);
      REXGPU_INFO("PGR4 D3D12: streams per frame: hashed {:.0f} ({:.2f} MB), new buffers {:.0f} ({:.2f} MB), "
                  "wrapped {:.0f} ({:.2f} MB), BufferFor {:.2f} ms (hash {:.2f} ms), rect/UP {:.2f} ms, loop {:.2f} ms, lookup {:.2f} ms, fill {:.2f} ms, {} buffers",
                  st.hashed / 600.0, st.hashed_bytes / 600.0 / 1048576.0, st.created / 600.0,
                  st.created_bytes / 600.0 / 1048576.0, st.wrapped / 600.0, st.wrapped_bytes / 600.0 / 1048576.0,
                  st.buffer_for_ns * 1e-6 / 600.0, st.hash_ns * 1e-6 / 600.0, st.special_ns * 1e-6 / 600.0,
                  st.loop_ns * 1e-6 / 600.0, st.lookup_ns * 1e-6 / 600.0, st.fill_ns * 1e-6 / 600.0,
                  s.buffers.size());
      REXGPU_INFO("PGR4 D3D12: textures per frame: SetupTextures {:.2f} ms (BindTexture {:.2f}, tables {:.2f}), "
                  "{} textures, {} sets; indexed fetch {:.2f} ms ({} draws, {} new stream sets)",
                  st.setup_textures_ns * 1e-6 / 600.0, st.bind_texture_ns * 1e-6 / 600.0, st.table_ns * 1e-6 / 600.0,
                  s.textures.size(), s.sets.size(), st.indexed_ns * 1e-6 / 600.0, st.indexed_draws / 600,
                  st.new_stream_sets / 600);
    }
    // Memory: what each cache holds, to find what grows.
    uint64_t buffer_bytes = 0, texture_bytes = 0, pool_bytes = 0;
    size_t pooled = 0;
    for (const auto &[key, b] : s.buffers) {
      buffer_bytes += b ? b->capacity : 0;
    }
    for (const auto &[key, t] : s.textures) {
      texture_bytes += t ? t->guest_size : 0;
    }
    {
      auto &pool = buffer_pool();
      std::lock_guard pool_lock(pool.mutex);
      for (const auto &[capacity, list] : pool.free) {
        pooled += list.size();
        pool_bytes += uint64_t(capacity) * list.size();
      }
    }
    PROCESS_MEMORY_COUNTERS_EX pmc{};
    K32GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS *>(&pmc), sizeof(pmc));
    REXGPU_INFO("PGR4 D3D12: memory: process {} MB private, buffers {} ({} MB), pool {} ({} MB), textures {} ({} MB "
                "guest), resolved {}, render targets {}, framebuffers {}, pipelines {}, sets {}, stream sets {}",
                pmc.PrivateUsage >> 20, s.buffers.size(), buffer_bytes >> 20, pooled, pool_bytes >> 20,
                s.textures.size(), texture_bytes >> 20, s.resolved.size(), s.render_targets.size(),
                s.framebuffers.size(), s.pipelines.size(), s.sets.size(), s.stream_sets.size());
    g_stream_stats = {};
  }
  Video::PresentSource source;
  {
    std::lock_guard lock(s.mutex);
    if (front_buffer && s.ready) {
      const TextureFetch f = ReadTextureFetch(GuestVirtual(front_buffer + 0x1C));
      ResolvedTexture *front = LiveResolved(s, f.BaseAddress());
      if (front && !front->depth) {
        BoundTexture b = BindResolved(s, *front, f);
        source.texture = b.texture;
        source.view = b.view;
        source.width = ScaledSize(front->width, front->scale.x);
        source.height = ScaledSize(front->height, front->scale.y);
        source.gamma_ramp = ReadGammaRamp(device);
      } else if (LogOnce(s, 0x4000000000ull | f.BaseAddress())) {
        REXGPU_WARN("PGR4 D3D12: front buffer {:08X} has no resolved copy", f.BaseAddress());
      }
    }
    if (s.ready && (s.frame % 600) == 0) {
      REXGPU_INFO("PGR4 D3D12: frame {}: {} draws, {} skipped, {} pipelines, {} textures, {} set views ({} stream sets), {} views free, {} write callbacks",
                  s.frame, s.draws, s.skipped_draws, s.pipelines.size(), s.textures.size(), s.set_views, s.stream_sets.size(), FreeViews(s),
                  g_watch_writes.exchange(0, std::memory_order_relaxed));
    }
  }
  {
    std::lock_guard lock(s.mutex);
    FlushLateFills(s);
  }
  Video::Present(source);
  std::lock_guard lock(s.mutex);
  if (s.ready) {
    EvictUnused(s);
  }
  ++s.frame;
  s.frame_prepared = false;
  s.race_hud = s.dial_drawn;
  s.dial_drawn = false;
  // Each frame decides again whether its screen is 2D only (ExecuteDraw).
  for (auto &[_, rt] : s.render_targets) {
    if (rt) {
      rt->wide_scene_drawn = rt->wide_bars_clean = false;
    }
  }
  s.dumping = false;
  // Debug: hide.tex (a hex texture base address) binds a placeholder in its
  // place, read every frame so textures can be compared live.
  s.hide_texture = 0;
  if (std::FILE *f = std::fopen("hide.tex", "rb")) {
    unsigned base = 0;
    if (std::fscanf(f, "%x", &base) == 1) {
      s.hide_texture = base;
    }
    std::fclose(f);
  }
  // Debug: skip.ps (a hex pixel shader key, as dumps print it) skips every
  // draw using that shader, to find which draws a part of the scene is.
  s.skip_ps = 0;
  if (std::FILE *f = std::fopen("skip.ps", "rb")) {
    unsigned long long key = 0;
    if (std::fscanf(f, "%llx", &key) == 1) {
      s.skip_ps = key;
    }
    std::fclose(f);
  }
  // --dump_after=<seconds>: dump one frame that long after the first one, for
  // machines where creating dump.now is awkward (Steam Deck).
  static const auto start = std::chrono::steady_clock::now();
  static bool timed_dump_done = false;
  const int dump_after = REXCVAR_GET(dump_after);
  bool timed = false;
  if (dump_after > 0 && !timed_dump_done &&
      std::chrono::steady_clock::now() - start >= std::chrono::seconds(dump_after)) {
    timed_dump_done = timed = true;
  }
  // Two frames in a row: some effects (the rear-view mirror) alternate.
  // dump.now may hold a frame count instead, to catch an intermittent frame.
  static int dump_frames_left = 0;
  if (std::FILE *f = std::fopen("dump.now", "rb"); f || timed) {
    int frames = 2;
    if (f) {
      if (std::fscanf(f, "%d", &frames) != 1 || frames < 1) {
        frames = 2;
      }
      std::fclose(f);
      std::remove("dump.now");
    }
    dump_frames_left = std::min(frames, 64);
    s.dump_seq = 0;
    if (g_pgr4_title.load(std::memory_order_relaxed)) {
      DumpTweakables(); // PGR4's registry; Geometry Wars has none
    }
  }
  if (dump_frames_left > 0) {
    --dump_frames_left;
    s.dumping = true;
    REXGPU_INFO("PGR4 D3D12: dumping frame {}", s.frame);
  }
  s.draws = 0;
  s.skipped_draws = 0;
}

// ---------------------------------------------------------------------------
// Render thread (render_thread)
//
// The game thread queues each call with copies of the guest memory it reads
// that the game may change before the call runs: the device's state (fetch
// constants, the float constants its shaders use, bool constants, register
// shadows and bindings) and arguments on the guest stack. The render thread
// runs the calls in order with GuestVirtual reading those copies. Vertex,
// index and texture data are read in place: the game thread waits at Swap
// until the frame is recorded, so they are at most a few draws later than
// on the 360, which also reads them after the call.

namespace {

enum class CommandKind : uint8_t {
  kDrawVertices,
  kDrawIndexed,
  kDrawUserPointer,
  kClear,
  kBeginTiling,
  kEndTiling,
  kResolve,
  kViewport,
  kSetTexture,
  kZPass,
  kSwap,
};

struct Command {
  CommandKind kind = CommandKind::kDrawVertices;
  uint32_t first_copy = 0, copy_count = 0;
  uint32_t a[8] = {};
  float f = 0;
};

struct CommandBatch {
  std::vector<Command> commands;
  std::vector<GuestCopy> copies;
  std::vector<uint8_t> bytes;
  std::vector<std::shared_ptr<const std::vector<uint8_t>>> snapshots; // kept alive for copies' external
  void Clear() {
    commands.clear();
    copies.clear();
    bytes.clear();
    snapshots.clear();
  }
};

// The game thread's latest copy of each dynamic vertex stream. Draws share
// it until a write to its pages (page watch) makes the next draw copy again;
// copying the crowd's streams for each of its ~400 draws cost more than the
// frame.
struct StreamSnapshot {
  PageWatch watch;
  std::shared_ptr<const std::vector<uint8_t>> data;
};

std::unordered_map<uint32_t, StreamSnapshot> &stream_snapshots() {
  static std::unordered_map<uint32_t, StreamSnapshot> snapshots; // game thread (producer lock)
  return snapshots;
}

struct RenderQueue {
  std::mutex producer; // guest threads that call D3D
  std::unique_ptr<CommandBatch> open;
  std::mutex mutex;
  std::condition_variable work, done;
  std::deque<std::unique_ptr<CommandBatch>> pending;
  std::vector<std::unique_ptr<CommandBatch>> spare;
  uint64_t queued = 0, finished = 0;
  uint64_t last_swap = 0; // queued count once the last Swap was flushed
  bool frame_pending = false; // render_ahead: the last frame may still be recording
  bool started = false;
};

RenderQueue &queue() {
  static RenderQueue q;
  return q;
}

// Set while a dump runs (by the render thread), so captures copy every
// float constant for the dump's logs.
std::atomic<bool> g_capture_all{false};

constexpr uint32_t kCommandsPerBatch = 64;

void RunCommand(const CommandBatch &b, const Command &c) {
  t_copies = b.copies.data() + c.first_copy;
  t_copy_count = c.copy_count;
  t_copy_bytes = b.bytes.data();
  t_physical_copies = std::any_of(t_copies, t_copies + t_copy_count, [](const GuestCopy &g) { return g.physical; });
  const uint32_t *a = c.a;
  switch (c.kind) {
  case CommandKind::kDrawVertices:
    ExecDrawVertices(a[0], a[1], a[2], a[3]);
    break;
  case CommandKind::kDrawIndexed:
    ExecDrawIndexed(a[0], a[1], int32_t(a[2]), a[3], a[4]);
    break;
  case CommandKind::kDrawUserPointer:
    ExecDrawUserPointer(a[0], a[1], a[2], a[3], a[4]);
    break;
  case CommandKind::kClear:
    ExecClear(a[0], a[1], a[2], c.f, a[3]);
    break;
  case CommandKind::kBeginTiling:
    ExecBeginTiling(a[0], a[1], a[2], a[3], c.f, a[4]);
    break;
  case CommandKind::kEndTiling:
    ExecEndTiling(a[0], a[1], a[2], a[3], c.f);
    break;
  case CommandKind::kResolve:
    ExecResolve(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], c.f);
    break;
  case CommandKind::kViewport:
    ExecOnViewport(a[0], a[1]);
    break;
  case CommandKind::kSetTexture:
    ExecOnSetTexture(a[0], a[1], a[2]);
    break;
  case CommandKind::kZPass:
    ExecOnZPass(a[0], a[1] != 0);
    break;
  case CommandKind::kSwap:
    ExecSwap(a[0], a[1]);
    g_recorded_frame.fetch_add(1, std::memory_order_relaxed);
    g_capture_all.store(state().dumping, std::memory_order_relaxed);
    break;
  }
  t_copies = nullptr;
  t_copy_count = 0;
  t_copy_bytes = nullptr;
  t_physical_copies = false;
}

// Debug sampler (profile_render_thread): suspends the render thread every
// millisecond and walks its stack, counting each function once per sample
// (inclusive) and the innermost one (exclusive). Every 10 seconds the counts
// go to render_samples.txt as module, offset, counts, for llvm-symbolizer.
void SampleRenderThread(HANDLE thread, ULONG_PTR stack_low, ULONG_PTR stack_high) {
  // Whole stacks, innermost first, counted by their hash.
  struct Stack {
    std::vector<uint64_t> frames;
    uint32_t count = 0;
  };
  std::unordered_map<uint64_t, Stack> stacks;
  uint64_t samples = 0;
  auto last_write = std::chrono::steady_clock::now();
  timeBeginPeriod(1);
  for (;;) {
    Sleep(1);
    if (SuspendThread(thread) == DWORD(-1)) {
      return;
    }
    CONTEXT ctx{};
    ctx.ContextFlags = CONTEXT_FULL;
    uint64_t frames[32];
    int depth = 0;
    if (GetThreadContext(thread, &ctx)) {
      // Only the render thread's own stack is read: under Proton the thread
      // can be stopped outside it (in Wine's code), where a blind read of
      // [rsp] faulted.
      auto on_stack = [&](DWORD64 sp) { return sp >= stack_low && sp + 8 <= stack_high; };
      for (; depth < 32 && ctx.Rip && on_stack(ctx.Rsp); ++depth) {
        frames[depth] = ctx.Rip;
        DWORD64 image = 0;
        RUNTIME_FUNCTION *fn = RtlLookupFunctionEntry(ctx.Rip, &image, nullptr);
        if (!fn) {
          ctx.Rip = *reinterpret_cast<DWORD64 *>(ctx.Rsp); // leaf function
          ctx.Rsp += 8;
          continue;
        }
        void *handler_data = nullptr;
        DWORD64 establisher = 0;
        RtlVirtualUnwind(UNW_FLAG_NHANDLER, image, ctx.Rip, fn, &ctx, &handler_data, &establisher, nullptr);
      }
    }
    ResumeThread(thread);
    ++samples;
    Stack &st = stacks[XXH3_64bits(frames, depth * sizeof(uint64_t))];
    if (!st.count) {
      st.frames.assign(frames, frames + depth);
    }
    ++st.count;
    if (std::chrono::steady_clock::now() - last_write < std::chrono::seconds(10)) {
      continue;
    }
    last_write = std::chrono::steady_clock::now();
    if (std::FILE *f = std::fopen("render_samples.txt", "w")) {
      std::unordered_map<uint64_t, std::string> where;
      std::fprintf(f, "samples %llu\n", (unsigned long long)samples);
      for (const auto &[hash, st] : stacks) {
        std::fprintf(f, "%u", st.count);
        for (const uint64_t rip : st.frames) {
          std::string &w = where[rip];
          if (w.empty()) {
            HMODULE module = nullptr;
            char name[MAX_PATH] = "?";
            if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                       GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                                   reinterpret_cast<LPCSTR>(rip), &module)) {
              GetModuleFileNameA(module, name, MAX_PATH);
            }
            const char *base = std::strrchr(name, '\\');
            w = fmt::format("{}:{:x}", base ? base + 1 : name, rip - uint64_t(module));
          }
          std::fprintf(f, " %s", w.c_str());
        }
        std::fprintf(f, "\n");
      }
      std::fclose(f);
    }
  }
}

void RenderThreadMain() {
  SetThreadDescription(GetCurrentThread(), L"PGR4 render");
  t_timed_by_batch = true;
  if (REXCVAR_GET(profile_render_thread)) {
    HANDLE self = nullptr;
    DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(), &self, 0, FALSE,
                    DUPLICATE_SAME_ACCESS);
    ULONG_PTR low = 0, high = 0;
    GetCurrentThreadStackLimits(&low, &high);
    std::thread(SampleRenderThread, self, low, high).detach();
  }
  auto &q = queue();
  for (;;) {
    std::unique_ptr<CommandBatch> batch;
    {
      std::unique_lock lock(q.mutex);
      q.work.wait(lock, [&] { return !q.pending.empty(); });
      batch = std::move(q.pending.front());
      q.pending.pop_front();
    }
    // The batch's CPU time, less fence waits (reported apart) and the
    // Swap, which presents.
    const auto start = std::chrono::steady_clock::now();
    const uint64_t waited = Video::GpuWaitTotal();
    uint32_t draws = 0;
    auto report = [&] {
      const uint64_t ns = uint64_t(
          std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count());
      const uint64_t wait = Video::GpuWaitTotal() - waited;
      Video::AddRendererTime(ns > wait ? ns - wait : 0, draws);
    };
    bool reported = false;
    for (const Command &c : batch->commands) {
      if (c.kind == CommandKind::kSwap) {
        report(); // Swap ends its batch
        reported = true;
      }
      draws += c.kind <= CommandKind::kDrawUserPointer;
      RunCommand(*batch, c);
    }
    if (!reported) {
      report();
    }
    batch->Clear();
    {
      std::lock_guard lock(q.mutex);
      q.spare.push_back(std::move(batch));
      ++q.finished;
    }
    q.done.notify_all();
  }
}

// Debug: render_thread.mode (read at Swap, when the queue is empty) picks
// 0 threaded, 1 threaded but waiting for every command, 2 not threaded, or
// 100 + a mask of CommandKind bits to wait after, so the modes can be
// compared on one screen.
std::atomic<int> g_thread_mode{0};

bool Threaded() {
  static const bool threaded = REXCVAR_GET(render_thread);
  return threaded && g_thread_mode.load(std::memory_order_relaxed) != 2;
}

void SyncLocked(RenderQueue &q);

void ReadThreadMode() {
  int mode = 0;
  if (std::FILE *f = std::fopen("render_thread.mode", "rb")) {
    if (std::fscanf(f, "%d", &mode) != 1) {
      mode = 0;
    }
    std::fclose(f);
  }
  if (mode != g_thread_mode.load(std::memory_order_relaxed)) {
    // With render_ahead a frame may still be recording: finish it first.
    if (REXCVAR_GET(render_thread)) {
      auto &q = queue();
      std::lock_guard lock(q.producer);
      SyncLocked(q);
    }
    g_thread_mode.store(mode, std::memory_order_relaxed);
    REXGPU_INFO("PGR4 D3D12: render thread mode {}", mode);
  }
}

// Hands the open batch to the render thread (producer lock held).
void FlushLocked(RenderQueue &q) {
  if (!q.open || q.open->commands.empty()) {
    return;
  }
  {
    std::lock_guard lock(q.mutex);
    q.pending.push_back(std::move(q.open));
    ++q.queued;
    if (!q.started) {
      q.started = true;
      std::thread(RenderThreadMain).detach();
    }
  }
  q.work.notify_one();
}

// Waits until everything queued so far has run (producer lock held).
void SyncLocked(RenderQueue &q) {
  FlushLocked(q);
  std::unique_lock lock(q.mutex);
  const uint64_t target = q.queued;
  q.done.wait(lock, [&] { return q.finished >= target; });
}

// At Swap with render_ahead: hands the frame over and waits only for the
// frame before it, so the game's next frame (its simulation, before any
// draw) overlaps this one's recording (producer lock held). The render
// thread reads vertex, index and texture memory late; the game writes those
// while drawing, by when the frame before is normally recorded.
void SwapAheadLocked(RenderQueue &q) {
  FlushLocked(q);
  std::unique_lock lock(q.mutex);
  const uint64_t target = q.last_swap;
  q.last_swap = q.queued;
  q.frame_pending = true;
  q.done.wait(lock, [&] { return q.finished >= target; });
}

// The next frame's first call waits for the last frame to be recorded: the
// game writes vertex data (and fills textures) while drawing, so only its
// simulation, before any D3D call, runs alongside (producer lock held).
void FinishLastFrameLocked(RenderQueue &q) {
  if (!q.frame_pending) {
    return;
  }
  q.frame_pending = false;
  std::unique_lock lock(q.mutex);
  q.done.wait(lock, [&] { return q.finished >= q.last_swap; });
}

// Builds one command: copies of guest ranges, then the arguments.
class Capture {
public:
  explicit Capture(CommandKind kind) : q_(queue()), lock_(q_.producer) {
    FinishLastFrameLocked(q_);
    if (!q_.open) {
      std::lock_guard lock(q_.mutex);
      if (!q_.spare.empty()) {
        q_.open = std::move(q_.spare.back());
        q_.spare.pop_back();
      } else {
        q_.open = std::make_unique<CommandBatch>();
      }
    }
    b_ = q_.open.get();
    c_.kind = kind;
    c_.first_copy = uint32_t(b_->copies.size());
  }
  Capture(const Capture &) = delete;
  Capture &operator=(const Capture &) = delete;

  void Copy(uint32_t address, uint32_t size) {
    if (!address || !size) {
      return;
    }
    GuestCopy g;
    g.address = address;
    g.size = size;
    g.offset = uint32_t(b_->bytes.size());
    b_->bytes.resize(b_->bytes.size() + size);
    std::memcpy(b_->bytes.data() + g.offset, REX_KERNEL_MEMORY()->TranslateVirtual(address), size);
    b_->copies.push_back(g);
  }
  // The device state a draw, clear or resolve reads. Draws also get the
  // float constants their shaders declare.
  void CopyDevice(uint32_t device, bool constants) {
    Copy(device + dev::kFetchConstants, dev::kVsConstants - dev::kFetchConstants);
    if (constants) {
      const bool all = g_capture_all.load(std::memory_order_relaxed);
      auto regs = [&](uint32_t offset) -> uint32_t {
        if (all) {
          return 256;
        }
        const HostShader *shader = ShaderFor(state(), ReadU32(device + offset));
        return shader && shader->cached ? std::max(shader->cached->constant_regs, 1u) : 256u;
      };
      Copy(device + dev::kVsConstants, regs(dev::kVertexShader) * 16);
      Copy(device + dev::kPsConstants, regs(dev::kPixelShader) * 16);
    }
    Copy(device + dev::kBoolConstants, dev::kVertexShader + 4 - dev::kBoolConstants);
    // The bound surfaces' info and size words (RenderTargetFor): the game
    // reuses surface objects for other EDRAM layouts within a frame.
    for (uint32_t i = 0; i < 5; ++i) {
      const uint32_t at = i < 4 ? dev::kRenderTargets + i * 4 : dev::kDepthStencil;
      if (const uint32_t surface = ReadU32(device + at)) {
        Copy(surface + 0x1C, 12);
      }
    }
  }
  void CopyPhysical(uint32_t address, uint32_t size) {
    GuestCopy g;
    g.address = address;
    g.size = size;
    g.offset = uint32_t(b_->bytes.size());
    g.physical = true;
    b_->bytes.resize(b_->bytes.size() + size);
    std::memcpy(b_->bytes.data() + g.offset, REX_KERNEL_MEMORY()->TranslatePhysical(address), size);
    b_->copies.push_back(g);
  }
  // The bound vertex streams the game rewrites every frame, as they are now.
  void CopyDynamicStreams(uint32_t device) {
    const uint64_t frame = g_game_frame.load(std::memory_order_relaxed);
    for (uint32_t stream = 0; stream < 16; ++stream) {
      const uint8_t *fc = REX_KERNEL_MEMORY()->TranslateVirtual(device + dev::kFetchConstants + (95 - stream) * 8);
      const uint32_t address = LoadBE32(fc) & 0xFFFFFFFCu;
      const uint32_t size = ((LoadBE32(fc + 4) >> 2) & 0xFFFFFF) * 4;
      if (!address || !size || address + uint64_t(size) > 0x20000000 || DynamicStreamSize(address, frame) != size) {
        continue;
      }
      if (!WatchesEnabled()) {
        CopyPhysical(address, size);
        continue;
      }
      StreamSnapshot &snap = stream_snapshots()[address];
      if (!snap.data || snap.data->size() != size || snap.watch.written.load(std::memory_order_acquire)) {
        Watch(snap.watch, address, size); // armed before the copy: a write during it marks it again
        const uint8_t *src = REX_KERNEL_MEMORY()->TranslatePhysical(address);
        snap.data = std::make_shared<const std::vector<uint8_t>>(src, src + size);
      }
      GuestCopy g;
      g.address = address;
      g.size = size;
      g.physical = true;
      g.external = snap.data->data();
      b_->copies.push_back(g);
      b_->snapshots.push_back(snap.data);
    }
  }
  // A resolve's destination header: the game repoints texture objects at
  // other memory between frames (ping-ponged buffers).
  void CopyTexture(uint32_t texture) {
    if (texture) {
      Copy(texture + 0x1C, 24);
    }
  }
  Command &command() { return c_; }

  ~Capture() {
    c_.copy_count = uint32_t(b_->copies.size()) - c_.first_copy;
    b_->commands.push_back(c_);
    const int mode = g_thread_mode.load(std::memory_order_relaxed);
    if (c_.kind == CommandKind::kSwap && mode == 0 && REXCVAR_GET(render_ahead)) {
      SwapAheadLocked(q_);
    } else if (c_.kind == CommandKind::kSwap || mode == 1 ||
        (mode >= 100 && ((mode - 100) >> int(c_.kind) & 1))) {
      SyncLocked(q_);
    } else if (b_->commands.size() >= kCommandsPerBatch) {
      FlushLocked(q_);
    }
  }

private:
  RenderQueue &q_;
  std::lock_guard<std::mutex> lock_;
  CommandBatch *b_ = nullptr;
  Command c_;
};

// Shader creation changes the map captures read constant counts from, so
// it runs with the render thread idle.
void SyncRenderThread() {
  if (!Threaded()) {
    return;
  }
  auto &q = queue();
  std::lock_guard lock(q.producer);
  SyncLocked(q);
}

} // namespace

void Renderer::OnSetTexture(uint32_t device, uint32_t sampler, uint32_t texture) {
  if (!Threaded()) {
    ExecOnSetTexture(device, sampler, texture);
    return;
  }
  if (!g_capture_all.load(std::memory_order_relaxed)) {
    return; // only logged during dumps
  }
  Capture c(CommandKind::kSetTexture);
  c.Copy(device + dev::kFetchConstants + (sampler & 31) * 24, 24);
  c.command().a[0] = device, c.command().a[1] = sampler, c.command().a[2] = texture;
}

void Renderer::OnZPass(uint32_t device, bool begin) {
  if (!Threaded()) {
    ExecOnZPass(device, begin);
    return;
  }
  Capture c(CommandKind::kZPass);
  c.command().a[0] = device, c.command().a[1] = begin;
}

void Renderer::OnViewport(uint32_t device, uint32_t viewport) {
  if (!Threaded()) {
    ExecOnViewport(device, viewport);
    return;
  }
  Capture c(CommandKind::kViewport);
  c.Copy(viewport, 24);
  c.command().a[0] = device, c.command().a[1] = viewport;
}

void Renderer::DrawVertices(uint32_t device, uint32_t primitive, uint32_t start_vertex, uint32_t vertex_count) {
  if (!Threaded()) {
    ExecDrawVertices(device, primitive, start_vertex, vertex_count);
    return;
  }
  Capture c(CommandKind::kDrawVertices);
  c.CopyDevice(device, true);
  c.CopyDynamicStreams(device);
  c.command().a[0] = device, c.command().a[1] = primitive, c.command().a[2] = start_vertex;
  c.command().a[3] = vertex_count;
}

void Renderer::DrawIndexed(uint32_t device, uint32_t primitive, int32_t base_vertex, uint32_t start_index,
                           uint32_t index_count) {
  if (!Threaded()) {
    ExecDrawIndexed(device, primitive, base_vertex, start_index, index_count);
    return;
  }
  Capture c(CommandKind::kDrawIndexed);
  c.CopyDevice(device, true);
  c.CopyDynamicStreams(device);
  c.command().a[0] = device, c.command().a[1] = primitive, c.command().a[2] = uint32_t(base_vertex);
  c.command().a[3] = start_index, c.command().a[4] = index_count;
}

void Renderer::DrawUserPointer(uint32_t device, uint32_t primitive, uint32_t vertex_count, uint32_t data,
                               uint32_t stride) {
  if (!Threaded()) {
    ExecDrawUserPointer(device, primitive, vertex_count, data, stride);
    return;
  }
  if (!data || !stride) {
    return;
  }
  Capture c(CommandKind::kDrawUserPointer);
  c.CopyDevice(device, true);
  c.Copy(data, vertex_count * stride);
  c.command().a[0] = device, c.command().a[1] = primitive, c.command().a[2] = vertex_count;
  c.command().a[3] = data, c.command().a[4] = stride;
}

void Renderer::Clear(uint32_t device, uint32_t flags, uint32_t argb, float z, uint32_t stencil) {
  if (!Threaded()) {
    ExecClear(device, flags, argb, z, stencil);
    return;
  }
  Capture c(CommandKind::kClear);
  c.CopyDevice(device, false);
  c.command().a[0] = device, c.command().a[1] = flags, c.command().a[2] = argb, c.command().a[3] = stencil;
  c.command().f = z;
}

void Renderer::BeginTiling(uint32_t device, uint32_t rect_count, uint32_t rects, uint32_t clear_color, float z,
                           uint32_t stencil) {
  if (!Threaded()) {
    ExecBeginTiling(device, rect_count, rects, clear_color, z, stencil);
    return;
  }
  Capture c(CommandKind::kBeginTiling);
  c.CopyDevice(device, false);
  c.Copy(rects, std::min(rect_count, 16u) * 16);
  c.Copy(clear_color, 16);
  c.command().a[0] = device, c.command().a[1] = rect_count, c.command().a[2] = rects;
  c.command().a[3] = clear_color, c.command().a[4] = stencil;
  c.command().f = z;
}

void Renderer::EndTiling(uint32_t device, uint32_t flags, uint32_t dest_texture, uint32_t clear_color,
                         float clear_z) {
  if (!Threaded()) {
    ExecEndTiling(device, flags, dest_texture, clear_color, clear_z);
    return;
  }
  Capture c(CommandKind::kEndTiling);
  c.CopyDevice(device, false);
  c.CopyTexture(dest_texture);
  c.Copy(clear_color, 16);
  c.command().a[0] = device, c.command().a[1] = flags, c.command().a[2] = dest_texture;
  c.command().a[3] = clear_color;
  c.command().f = clear_z;
}

void Renderer::Resolve(uint32_t device, uint32_t flags, uint32_t source_rect, uint32_t dest_texture,
                       uint32_t dest_point, uint32_t dest_level, uint32_t dest_slice, uint32_t clear_color,
                       float clear_z) {
  if (!Threaded()) {
    ExecResolve(device, flags, source_rect, dest_texture, dest_point, dest_level, dest_slice, clear_color, clear_z);
    return;
  }
  Capture c(CommandKind::kResolve);
  c.CopyDevice(device, false);
  c.CopyTexture(dest_texture);
  c.Copy(source_rect, 16);
  c.Copy(dest_point, 8);
  c.Copy(clear_color, 16);
  c.command().a[0] = device, c.command().a[1] = flags, c.command().a[2] = source_rect;
  c.command().a[3] = dest_texture, c.command().a[4] = dest_point, c.command().a[5] = dest_level;
  c.command().a[6] = dest_slice, c.command().a[7] = clear_color;
  c.command().f = clear_z;
}

namespace {

// Debug: a stacks.now file next to the exe logs every thread's call stack
// (module+offset, innermost first), for finding where the game hangs. The
// recompiled guest functions resolve through the PDB as sub_XXXXXXXX.
std::string ThreadStack(HANDLE thread) {
  CONTEXT ctx{};
  ctx.ContextFlags = CONTEXT_FULL;
  if (!GetThreadContext(thread, &ctx)) {
    return " (no context)";
  }
  std::string line;
  for (int depth = 0; depth < 32 && ctx.Rip; ++depth) {
    HMODULE module = nullptr;
    char name[MAX_PATH] = "?";
    if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           reinterpret_cast<LPCSTR>(ctx.Rip), &module)) {
      GetModuleFileNameA(module, name, MAX_PATH);
    }
    const char *base = std::strrchr(name, '\\');
    line += fmt::format(" {}+0x{:X}", base ? base + 1 : name, ctx.Rip - uint64_t(module));
    DWORD64 image = 0;
    RUNTIME_FUNCTION *fn = RtlLookupFunctionEntry(ctx.Rip, &image, nullptr);
    if (!fn) {
      ctx.Rip = *reinterpret_cast<DWORD64 *>(ctx.Rsp); // leaf function
      ctx.Rsp += 8;
      continue;
    }
    void *handler_data = nullptr;
    DWORD64 establisher = 0;
    RtlVirtualUnwind(UNW_FLAG_NHANDLER, image, ctx.Rip, fn, &ctx, &handler_data, &establisher, nullptr);
  }
  return line;
}

} // namespace

void DumpThreadStacks(const char *reason) {
  {
    REXGPU_INFO("PGR4 stacks: {}", reason);
    const DWORD self = GetCurrentThreadId(), process = GetCurrentProcessId();
    HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
    THREADENTRY32 te{sizeof(te)};
    for (BOOL ok = Thread32First(snap, &te); ok; ok = Thread32Next(snap, &te)) {
      if (te.th32OwnerProcessID != process || te.th32ThreadID == self) {
        continue;
      }
      HANDLE thread = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION, FALSE,
                                 te.th32ThreadID);
      if (!thread) {
        continue;
      }
      std::string stack;
      if (SuspendThread(thread) != DWORD(-1)) {
        stack = ThreadStack(thread);
        ResumeThread(thread);
      }
      CloseHandle(thread);
      REXGPU_INFO("PGR4 stacks: thread {}:{}", te.th32ThreadID, stack);
    }
    CloseHandle(snap);
  }
}

namespace {

void WatchForStackDump() {
  for (;;) {
    Sleep(100);
    if (std::FILE *f = std::fopen("stacks.now", "rb")) {
      std::fclose(f);
      std::remove("stacks.now");
      DumpThreadStacks("stacks.now");
    }
    uint32_t page = 0;
    if (std::FILE *f = std::fopen("watch.addr", "rb")) {
      char text[24] = {};
      std::fgets(text, sizeof(text), f);
      std::fclose(f);
      page = uint32_t(std::strtoul(text, nullptr, 16)) & ~0xFFFu;
    }
    if (page != g_debug_watch_page.exchange(page, std::memory_order_relaxed)) {
      REXGPU_INFO("PGR4 watch.addr: page {:08X}", page);
    }
    if (page && WatchesEnabled()) {
      REX_KERNEL_MEMORY()->EnablePhysicalMemoryAccessCallbacks(page, 0x1000, true, false);
    }
    // Writes that bypass the page protection (host-side copies) show up here.
    static uint64_t last_hash = 0;
    if (page) {
      const uint8_t *bytes = GuestPhysical(page);
      const uint64_t hash = XXH3_64bits(bytes, 0x1000);
      if (hash != last_hash) {
        last_hash = hash;
        REXGPU_INFO("PGR4 watch.addr: page {:08X} now {:08X} {:08X} {:08X} {:08X} frame {}", page, LoadBE32(bytes),
                    LoadBE32(bytes + 4), LoadBE32(bytes + 8), LoadBE32(bytes + 12),
                    g_game_frame.load(std::memory_order_relaxed));
      }
    }
  }
}

} // namespace

void Renderer::Swap(uint32_t device, uint32_t front_buffer) {
  static const bool watching = [] {
    std::thread(WatchForStackDump).detach();
    return true;
  }();
  (void)watching;
  // These write guest memory, so they stay on the game thread. The tweak
  // registry and post-process object are PGR4's (not Geometry Wars').
  const bool pgr4 = g_pgr4_title.load(std::memory_order_relaxed);
  if (pgr4) {
    ApplyTweaks();
  }
  PaceFrame();
  if (pgr4) {
    StoreFrameTime();
  }
  if (!Threaded()) {
    ExecSwap(device, front_buffer);
    g_recorded_frame.fetch_add(1, std::memory_order_relaxed);
    g_game_frame.fetch_add(1, std::memory_order_relaxed);
    ReadThreadMode();
    return;
  }
  {
    Capture c(CommandKind::kSwap);
    if (front_buffer) {
      c.Copy(front_buffer + 0x1C, 24);
    }
    c.Copy(device + kDeviceGammaFormat, 4);
    c.Copy(device + kDeviceGammaRamp, 768 * 2);
    c.command().a[0] = device, c.command().a[1] = front_buffer;
  }
  // After the Swap was queued (Capture's destructor): writes from here on
  // belong to the next frame.
  g_game_frame.fetch_add(1, std::memory_order_relaxed);
  ReadThreadMode();
}

// Runs as each title starts (its memory is up, its code not yet running), so
// the write callback lands on that title's memory.
void Renderer::SetTitle(bool pgr4) {
  g_pgr4_title.store(pgr4, std::memory_order_relaxed);
  g_watch_registered.store(false, std::memory_order_release);
  WatchesEnabled();
}

// The outgoing title's threads have stopped; its guest memory goes next. Every
// object found by guest address (textures, resolve copies, vertex and index
// buffers, shader objects, declarations, render targets) is released the way
// eviction does it, parked until the GPU is done with it. Host shaders and
// pipelines (keyed by content) are kept for the next title.
void Renderer::ResetForTitleSwitch() {
  SyncRenderThread();
  auto &s = state();
  std::lock_guard lock(s.mutex);
  for (auto &[_, set] : s.stream_sets) {
    ReleaseSet(s, set);
  }
  s.stream_sets.clear();
  for (auto &[_, set] : s.sets) {
    ReleaseSet(s, set);
  }
  s.sets.clear();
  for (auto &[_, t] : s.textures) {
    if (t) {
      Unwatch(t->watch);
      Video::ParkUntilFence(std::move(t->view));
      Video::ParkUntilFence(std::move(t->texture));
    }
  }
  s.textures.clear();
  for (auto &[_, r] : s.resolved) {
    if (r) {
      Video::ParkUntilFence(std::move(r->texture));
      for (auto &[_, v] : r->views) {
        Video::ParkUntilFence(std::move(v));
      }
    }
  }
  s.resolved.clear();
  s.latest_resolved.clear();
  for (auto &[_, b] : s.buffers) {
    if (b) {
      Unwatch(b->watch);
      RecycleBuffer(std::move(b->buffer), b->capacity);
    }
  }
  s.buffers.clear();
  // Render targets are keyed by EDRAM base and shape; the next title's
  // surfaces would land on the old title's (its sizes, scale, pending shared
  // clears and contents), so they start over like a fresh boot.
  for (auto &[_, fb] : s.framebuffers) {
    Video::ParkUntilFence(std::move(fb));
  }
  s.framebuffers.clear();
  for (auto &[_, rt] : s.render_targets) {
    if (rt) {
      Video::ParkUntilFence(std::move(rt->texture));
    }
  }
  s.render_targets.clear();
  s.shared_clears.clear();
  s.shaders_by_object.clear();
  s.declarations.clear();
  s.viewports.clear();
  s.texture_setups.clear();
  s.late_fills.clear();
  s.buffer_hits = {};
  s.fetch_hits = {};
  s.bound = {};
  s.bound_framebuffer = nullptr;
  s.frame_prepared = false;
  s.vs_constants = {};
  s.ps_constants = {};
  s.vs_shared = {};
  s.ps_shared = {};
  s.tiling_active = false;
  s.dial_drawn = s.race_hud = false;
  ++s.binding_generation;
  {
    auto &d = dynamic_streams();
    std::unique_lock streams_lock(d.mutex);
    d.streams.clear();
  }
  // Render-thread snapshots of dynamic streams (crowd poses, particles) carry
  // their own watches: left armed on the old title's addresses, they would
  // never see a write again and replay stale vertex data.
  for (auto &[_, snap] : stream_snapshots()) {
    Unwatch(snap.watch);
  }
  stream_snapshots().clear();
  {
    auto &list = watch_list();
    std::lock_guard watch_lock(list.mutex);
    list.buckets.clear();
  }
  // The next title's memory needs the write callback registered again.
  g_watch_registered.store(false, std::memory_order_release);
  REXGPU_INFO("PGR4 D3D12: renderer reset for a title switch");
}

} // namespace pgr4::gpu
