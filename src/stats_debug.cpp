// pgr4_recompiled - leaderboard write diagnostics
//
// PGR4 writes leaderboard rows through sub_828A7940 (-> XSessionWriteStats,
// sub_829333C8), called from the per-mode writers below. sub_828A7940 needs
// its writer object up (+4), idle (+5 clear), holding views (+16), and the
// Live session (r5) started (+12 == 3, set when XSessionStart completes,
// sub_828A3E28 case 8). The writers also check the session system
// (byte_82A42A8C) and the session object at dword_82A42A90 (+12 == 3).
// These hooks log each call so a log shows where a race's upload stops.

#include <cstdint>

#include <rex/hook.h>
#include <rex/logging.h>
#include <rex/memory.h>
#include <rex/system/kernel_state.h>

REX_EXTERN(__imp__sub_828A7940);
REX_EXTERN(__imp__sub_82534C90);
REX_EXTERN(__imp__sub_825359C8);
REX_EXTERN(__imp__sub_825451A0);
REX_EXTERN(__imp__sub_82546030);
REX_EXTERN(__imp__sub_82546BE8);
REX_EXTERN(__imp__sub_825477C8);
REX_EXTERN(__imp__sub_8254CD30);
REX_EXTERN(__imp__sub_8254D0F0);
REX_EXTERN(__imp__sub_8254D758);
REX_EXTERN(__imp__sub_8267B2F8);

namespace {

uint32_t Load32(uint32_t addr) {
  return rex::memory::load_and_swap<uint32_t>(REX_KERNEL_MEMORY()->TranslateVirtual(addr));
}

uint8_t Load8(uint32_t addr) {
  return *REX_KERNEL_MEMORY()->TranslateVirtual<uint8_t*>(addr);
}

void LogWriter(const char* name, const PPCContext& ctx) {
  const uint32_t session = Load32(0x82A42A90);
  REXLOG_INFO("stats dbg: {} (r3 {:08X} r4 {:08X} r5 {:08X}) system {} session {:08X} state {}",
              name, ctx.r3.u32, ctx.r4.u32, ctx.r5.u32, Load8(0x82A42A8C), session,
              session ? Load32(session + 12) : 0);
}

}  // namespace

REX_HOOK_RAW(sub_828A7940) {
  const uint32_t writer = ctx.r3.u32, session = ctx.r5.u32;
  REXLOG_INFO(
      "stats dbg: write views (writer {:08X}: up {} busy {} views {:08X} count {}) "
      "(session {:08X}: state {} handle {:08X})",
      writer, Load8(writer + 4), Load8(writer + 5), Load32(writer + 16), Load32(writer + 24),
      session, session ? Load32(session + 12) : 0, session ? Load32(session + 72) : 0);
  __imp__sub_828A7940(ctx, base);
  REXLOG_INFO("stats dbg: write views -> {}", ctx.r3.u32 & 0xFF);
}

#define STATS_WRITER_HOOK(fn)                                \
  REX_HOOK_RAW(fn) {                                         \
    LogWriter(#fn, ctx);                                     \
    __imp__##fn(ctx, base);                                  \
    REXLOG_INFO("stats dbg: " #fn " -> {}", ctx.r3.u32);     \
  }

STATS_WRITER_HOOK(sub_82534C90)
STATS_WRITER_HOOK(sub_825359C8)
STATS_WRITER_HOOK(sub_825451A0)
STATS_WRITER_HOOK(sub_82546030)
STATS_WRITER_HOOK(sub_82546BE8)
STATS_WRITER_HOOK(sub_825477C8)
STATS_WRITER_HOOK(sub_8254CD30)
STATS_WRITER_HOOK(sub_8254D0F0)
STATS_WRITER_HOOK(sub_8254D758)
STATS_WRITER_HOOK(sub_8267B2F8)
