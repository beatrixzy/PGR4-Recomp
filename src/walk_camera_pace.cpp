// pgr4_recompiled - garage walk camera frame rate
//
// sub_827DCBA0 updates the first-person camera you walk around the garage
// with (Camera.Walk.* tweakables). The 360 holds PGR4 to 30 fps and the
// camera clamps its step to 1/30 s; above 30 fps its collision breaks and
// only looking around works. While it runs, the frame pacer holds the game
// to 30 fps (WalkCameraActive).

#include <atomic>
#include <chrono>

#include <rex/hook.h>
#include <rex/runtime.h>

REX_EXTERN(__imp__sub_827DCBA0);

namespace {
std::atomic<int64_t> last_update{0};

int64_t Now() { return std::chrono::steady_clock::now().time_since_epoch().count(); }
} // namespace

namespace pgr4 {
// True while the walk camera updated in the last half second.
bool WalkCameraActive() {
  const auto window = std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::milliseconds(500));
  return Now() - last_update.load(std::memory_order_relaxed) < window.count();
}
} // namespace pgr4

REX_HOOK_RAW(sub_827DCBA0) {
  last_update.store(Now(), std::memory_order_relaxed);
  __imp__sub_827DCBA0(ctx, base);
}
