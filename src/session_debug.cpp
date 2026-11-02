// pgr4_recompiled - session creation diagnostics
//
// PGR4 refuses to create System Link / Live sessions ("unable to make a
// session") before ever calling XSessionCreate. The Lua binding
// Session.StartCreate (sub_825DD5D0) needs, in order:
//   byte_82A42A8C (session system up), sub_82651418 (every active profile
//   Live + multiplayer privilege), sub_82650DC8 (a usable host profile),
//   then sub_828A4360 (session object ready, idle, slots, a free task).
// These hooks log each answer (only when it changes) so a log shows which
// one fails.

#include <cstdint>

#include <rex/hook.h>
#include <rex/logging.h>
#include <rex/memory.h>
#include <rex/system/kernel_state.h>

REX_EXTERN(__imp__sub_82651418);
REX_EXTERN(__imp__sub_82650DC8);
REX_EXTERN(__imp__sub_828A4360);
REX_EXTERN(__imp__sub_825DD5D0);
REX_EXTERN(__imp__sub_825D4AB8);
REX_EXTERN(__imp__sub_828A2850);
REX_EXTERN(__imp__sub_828A28A0);
REX_EXTERN(__imp__sub_8266BFD8);

namespace {

uint32_t Load32(uint32_t addr) {
  return rex::memory::load_and_swap<uint32_t>(REX_KERNEL_MEMORY()->TranslateVirtual(addr));
}

uint8_t Load8(uint32_t addr) {
  return *REX_KERNEL_MEMORY()->TranslateVirtual<uint8_t*>(addr);
}

// Profile slots: base 0x82BFE2B0, stride 141984; +76 signin state,
// +56 flags (0x4000 = multiplayer privilege), +81 guest.
void LogProfiles() {
  for (uint32_t i = 0; i < 4; ++i) {
    uint32_t slot = 0x82BFE2B0 + 141984 * i;
    REXLOG_INFO("session dbg:   profile {}: state {} flags {:04X} guest {}", i, Load32(slot + 76),
                rex::memory::load_and_swap<uint16_t>(
                    REX_KERNEL_MEMORY()->TranslateVirtual(slot + 56)),
                Load8(slot + 81));
  }
}

}  // namespace

REX_HOOK_RAW(sub_825DD5D0) {
  const uint32_t session = Load32(0x82A42A94);
  REXLOG_INFO("session dbg: StartCreate: system up {} session {:08X}", Load8(0x82A42A8C), session);
  if (session) {
    REXLOG_INFO("session dbg:   object ready {} busy {} max slots {} task {:08X}",
                Load8(session + 8), Load32(session + 12), Load32(session + 20),
                Load32(session + 64) ? Load32(Load32(session + 64)) : 0);
  }
  LogProfiles();
  __imp__sub_825DD5D0(ctx, base);
}

REX_HOOK_RAW(sub_82651418) {
  __imp__sub_82651418(ctx, base);
  static uint32_t last = ~0u;
  if (ctx.r3.u32 != last) {
    last = ctx.r3.u32;
    REXLOG_INFO("session dbg: all profiles Live+multiplayer = {}", last);
  }
}

REX_HOOK_RAW(sub_82650DC8) {
  __imp__sub_82650DC8(ctx, base);
  static uint32_t last = ~0u;
  if (ctx.r3.u32 != last) {
    last = ctx.r3.u32;
    REXLOG_INFO("session dbg: host profile = {:08X}", last);
  }
}

// System Link (Custom Match over LAN): Lua CreateLANCustomRaceSession
// (sub_825D4AB8) needs the net object 0x82A7E7D0 up (+8), with an address
// (+9, from XNetGetTitleXnAddr) and a link (+10, XNetGetEthernetLinkStatus),
// then sub_82667400 creates the key (sub_828A2850 / sub_828A28A0) and
// succeeds only if sub_8266BFD8 counts at least one channel that opened.
REX_HOOK_RAW(sub_825D4AB8) {
  REXLOG_INFO("session dbg: CreateLANCustomRaceSession: net up {} address {} link {}",
              Load8(0x82A7E7D8), Load8(0x82A7E7D9), Load8(0x82A7E7DA));
  LogProfiles();
  __imp__sub_825D4AB8(ctx, base);
  REXLOG_INFO("session dbg: CreateLANCustomRaceSession -> {}", ctx.r3.u32);
}

REX_HOOK_RAW(sub_828A2850) {
  __imp__sub_828A2850(ctx, base);
  static uint32_t last = ~0u;
  if ((ctx.r3.u32 & 0xFF) != last) {
    last = ctx.r3.u32 & 0xFF;
    REXLOG_INFO("session dbg: LAN create key -> {}", last);
  }
}

REX_HOOK_RAW(sub_828A28A0) {
  __imp__sub_828A28A0(ctx, base);
  static uint32_t last = ~0u;
  if ((ctx.r3.u32 & 0xFF) != last) {
    last = ctx.r3.u32 & 0xFF;
    REXLOG_INFO("session dbg: LAN register key -> {}", last);
  }
}

// Also polled every frame while in a LAN session: log changes only.
REX_HOOK_RAW(sub_8266BFD8) {
  __imp__sub_8266BFD8(ctx, base);
  static uint32_t last = ~0u;
  if (ctx.r3.u32 != last) {
    last = ctx.r3.u32;
    REXLOG_INFO("session dbg: LAN channels open = {}", last);
  }
}

REX_HOOK_RAW(sub_828A4360) {
  const uint32_t obj = ctx.r3.u32, user = ctx.r4.u32 & 0xFF, flags = ctx.r5.u32;
  const uint32_t pub = ctx.r8.u32, priv = ctx.r9.u32;
  __imp__sub_828A4360(ctx, base);
  REXLOG_INFO(
      "session dbg: create(obj {:08X}, user {}, flags {:X}, slots {}+{}) -> {} "
      "[ready {} busy {} max {} task {:08X}]",
      obj, user, flags, pub, priv, ctx.r3.u32 & 0xFF, Load8(obj + 8), Load32(obj + 12),
      Load32(obj + 20), Load32(obj + 64) ? Load32(Load32(obj + 64)) : 0);
}
