// pgr4_recompiled - ground height query guard
//
// sub_822C55B0(grid, point, radius, ...) returns the highest surface under a
// point: it turns the point's X/Z +- radius into cell bounds with fctiwz
// (grid scale at +24 of the grid header) and walks every cell in them. A
// coordinate of +inf (or one past 2^31 cells) saturates a bound to INT_MAX,
// and the signed `cell <= last` loop then never ends: the game hung on the
// car select screen with the Lotus Esprit Essex Turbo (called from
// sub_8243D948 with the point transformed by the car's matrix).
//
// Such a range has no finite answer on hardware either, so the query returns
// the function's own "nothing found" value (its empty-grid result), and the
// first few are logged with the caller's address.

#include <bit>
#include <cmath>

#include <rex/hook.h>
#include <rex/logging.h>
#include <rex/memory/utils.h>
#include <rex/runtime.h>
#include <rex/system/kernel_state.h>

REX_EXTERN(__imp__sub_822C55B0);

namespace {

constexpr uint32_t kNothingFound = 0x820011B0; // float the original returns for an empty grid
constexpr double kMaxCells = 1 << 16;           // per axis; the grid wraps every 16

uint32_t LoadU32(uint32_t address) {
  return rex::memory::load_and_swap<uint32_t>(REX_KERNEL_MEMORY()->TranslateVirtual(address));
}

float LoadF32(uint32_t address) { return std::bit_cast<float>(LoadU32(address)); }

bool SaneRange(double center, double radius, double scale) {
  const double lo = (center - radius) * scale, hi = (center + radius) * scale;
  return std::isfinite(lo) && std::isfinite(hi) && lo > -2147483648.0 && hi < 2147483647.0 &&
         hi - lo < kMaxCells;
}

} // namespace

REX_HOOK_RAW(sub_822C55B0) {
  const uint32_t grid = ctx.r3.u32, point = ctx.r4.u32;
  const uint32_t header = grid ? LoadU32(grid) : 0;
  if (header && point) {
    const double scale = LoadF32(header + 24), radius = float(ctx.f1.f64);
    const float x = LoadF32(point), y = LoadF32(point + 4), z = LoadF32(point + 8);
    if (!SaneRange(x, radius, scale) || !SaneRange(z, radius, scale)) {
      // The caller's object (r31) is a C++ local under non_volatile_as_local
      // codegen, so only the caller's address is logged.
      static int logged = 0;
      if (logged < 3) {
        ++logged;
        REXLOG_WARN("ground query: point ({}, {}, {}) radius {} scale {} from {:08X}", x, y, z, radius, scale,
                    uint32_t(ctx.lr));
      }
      ctx.f1.f64 = LoadF32(kNothingFound);
      return;
    }
  }
  __imp__sub_822C55B0(ctx, base);
}
