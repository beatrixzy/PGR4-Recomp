// pgr4_recompiled - occlusion query results
//
// PGR4 fades each car lamp's glow (brake/tail lights, the halo in front of
// the headlights) in or out by an occlusion query per lamp: sub_8233F708
// reads D3DQuery_GetData and steps the glow towards 1 while the count is
// non-zero, towards 0 while it is zero. The count is summed from the sample
// counters the GPU's Z-pass events write into the query's buffers; the
// command-buffer device never runs those events, so every lamp read as
// hidden and its glow faded to nothing (dim lights next to a 360 capture).
//
// Occlusion queries (type 9) now report visible, as Xenia does by default:
// lamps glow even when another car hides them, which the 360 would not show.

#include <atomic>
#include <chrono>
#include <filesystem>

#include <rex/hook.h>
#include <rex/memory/utils.h>
#include <rex/runtime.h>
#include <rex/system/kernel_state.h>

REX_EXTERN(__imp__D3DQuery_GetData);

namespace {
// Debug: an `occlusion.off` file next to the exe restores the original
// (checked once a second).
bool Disabled() {
  static std::atomic<int64_t> next{0};
  static std::atomic<bool> off{false};
  const int64_t now = std::chrono::steady_clock::now().time_since_epoch().count();
  if (now >= next.load(std::memory_order_relaxed)) {
    next = now + std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::seconds(1)).count();
    off = std::filesystem::exists("occlusion.off");
  }
  return off;
}

constexpr uint32_t kOcclusionQuery = 9;
// Any non-zero count reads as visible; this is a lamp quad's worth.
constexpr uint32_t kVisibleSamples = 1024;
} // namespace

// D3DQuery_GetData(query, pData, size, flags): S_OK (0) when the result is ready.
REX_HOOK_RAW(D3DQuery_GetData) {
  auto *memory = REX_KERNEL_MEMORY();
  const uint32_t query = ctx.r3.u32, data = ctx.r4.u32, size = ctx.r5.u32;
  if (Disabled() || !query ||
      rex::memory::load_and_swap<uint32_t>(memory->TranslateVirtual(query + 4)) != kOcclusionQuery) {
    __imp__D3DQuery_GetData(ctx, base);
    return;
  }
  if (data && size >= 4) {
    rex::memory::store_and_swap<uint32_t>(memory->TranslateVirtual(data), kVisibleSamples);
    for (uint32_t offset = 4; offset + 4 <= size && offset < 16; offset += 4) {
      rex::memory::store_and_swap<uint32_t>(memory->TranslateVirtual(data + offset), 0);
    }
  }
  ctx.r3.u64 = 0;
}
