// pgr4_recompiled - track streaming flush guard
//
// The track streamer queues load jobs in ten queues (0x82A9A080, 3112 bytes
// apart; popped by sub_823C8408 under each queue's critical section) for two
// worker threads, and counts queued + running jobs in dword_82A9A060
// (sub_823C9C10 adds, the workers and the per-frame cancel sub_823C63A8
// subtract). Leaving a race, sub_8238D410 calls sub_823C9AA0, which empties
// the queues once and then spins until the count is zero.
//
// A tester's game froze there at the end of an online race (log 031: both
// workers idle, the main thread in that spin, count stuck above zero): a job
// counted after the single emptying pass, or one counted twice, never comes
// back, so the spin never ends.
//
// The replacement keeps emptying the queues while it waits, and after a few
// seconds with jobs still outstanding it logs the count and clears it instead
// of freezing; nothing waits on those jobs once the track is unloading.

#include <atomic>
#include <bit>
#include <chrono>
#include <thread>

#include <rex/hook.h>
#include <rex/logging.h>
#include <rex/memory/utils.h>
#include <rex/runtime.h>
#include <rex/system/kernel_state.h>

REX_EXTERN(__imp__sub_823C8408);

namespace {

constexpr uint32_t kPending = 0x82A9A060;   // jobs queued or running
constexpr uint32_t kQueues = 0x82A9A080;    // first queue
constexpr uint32_t kQueueStride = 3112;
constexpr int kQueueCount = 10;
constexpr auto kGiveUp = std::chrono::seconds(3);

std::atomic_ref<uint32_t> PendingWord() {
  return std::atomic_ref<uint32_t>(*REX_KERNEL_MEMORY()->TranslateVirtual<uint32_t*>(kPending));
}

int32_t Pending() {
  return int32_t(std::byteswap(PendingWord().load()));
}

void AddPending(int32_t delta) {
  auto word = PendingWord();
  uint32_t old = word.load();
  while (!word.compare_exchange_weak(old, std::byteswap(uint32_t(int32_t(std::byteswap(old)) + delta)))) {
  }
}

// Pops and drops every queued job (as the original's first pass does).
void DropQueued(PPCContext& ctx, uint8_t* base) {
  // sub_823C8408(queue, out_job) copies the job (12 bytes) to out_job: give
  // it room below our caller's frame.
  const uint32_t saved_sp = ctx.r1.u32;
  ctx.r1.u64 = saved_sp - 0x80;
  const uint32_t job = ctx.r1.u32 + 0x50;
  for (int q = 0; q < kQueueCount; ++q) {
    for (;;) {
      ctx.r3.u64 = kQueues + q * kQueueStride;
      ctx.r4.u64 = job;
      __imp__sub_823C8408(ctx, base);
      if ((ctx.r3.u32 & 0xFF) == 0) {
        break;
      }
      AddPending(-1);
    }
  }
  ctx.r1.u64 = saved_sp;
}

} // namespace

REX_HOOK_RAW(sub_823C9AA0) {
  const auto start = std::chrono::steady_clock::now();
  for (;;) {
    DropQueued(ctx, base);
    const int32_t pending = Pending();
    if (pending == 0) {
      break;
    }
    const auto waited = std::chrono::steady_clock::now() - start;
    if (waited > kGiveUp) {
      REXLOG_WARN("track streaming: {} job(s) never finished after {} s; continuing the unload", pending,
                  std::chrono::duration_cast<std::chrono::seconds>(waited).count());
      AddPending(-pending);
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}
