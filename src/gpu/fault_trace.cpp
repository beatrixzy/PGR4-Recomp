// TEMPORARY bring-up diagnostic for the D3D12 renderer: logs the host call
// stack (module offsets; resolve with pgr4_recompiled.map) for
//  - guest accesses to the low guest page (null-pointer field reads), and
//  - host crashes: access violations outside the guest arena and other
//    fatal exception codes.
// Guest-arena faults elsewhere are the SDK's MMIO / write-watch traffic and
// are left alone.
// Also a hang watchdog: when no frame is presented for kHangSeconds, every
// thread's stack is logged the same way (once per hang).
#include <windows.h>

#include <tlhelp32.h>

#include <cstdint>
#include <thread>

#include <rex/logging.h>

#include "gpu/video.h"

namespace {

constexpr uintptr_t kGuestVirtualBase = 0x100000000ull;
constexpr uintptr_t kGuestArenaEnd = 0x300000000ull; // virtual + physical views

bool IsInteresting(const EXCEPTION_RECORD *rec) {
  switch (rec->ExceptionCode) {
  case EXCEPTION_ACCESS_VIOLATION: {
    if (rec->NumberParameters < 2) {
      return true;
    }
    const uintptr_t addr = rec->ExceptionInformation[1];
    if (addr >= kGuestVirtualBase && addr < kGuestArenaEnd) {
      return addr < kGuestVirtualBase + 0x10000; // null-ish guest pointer
    }
    return true; // host-side fault
  }
  case EXCEPTION_ILLEGAL_INSTRUCTION:
  case EXCEPTION_INT_DIVIDE_BY_ZERO:
  case EXCEPTION_STACK_OVERFLOW:
  case EXCEPTION_PRIV_INSTRUCTION:
  case 0xC0000409: // STATUS_STACK_BUFFER_OVERRUN (fail-fast)
  case 0xC0000374: // STATUS_HEAP_CORRUPTION
  case 0x80000003: // breakpoint / assert
    return true;
  default:
    return false;
  }
}

LONG CALLBACK TraceFault(EXCEPTION_POINTERS *info) {
  const auto *rec = info->ExceptionRecord;
  if (!IsInteresting(rec)) {
    return EXCEPTION_CONTINUE_SEARCH;
  }
  static volatile LONG reported = 0;
  if (InterlockedIncrement(&reported) > 4) {
    return EXCEPTION_CONTINUE_SEARCH;
  }

  const auto module = reinterpret_cast<uintptr_t>(GetModuleHandleW(nullptr));
  CONTEXT ctx = *info->ContextRecord;
  const uintptr_t addr = rec->NumberParameters >= 2 ? rec->ExceptionInformation[1] : 0;
  REXLOG_ERROR("FAULT TRACE: code {:08X} addr 0x{:X} thread {} rip=exe+0x{:X} (abs 0x{:X})",
               uint32_t(rec->ExceptionCode), addr, GetCurrentThreadId(), ctx.Rip - module,
               ctx.Rip);
  for (int frame = 0; frame < 32 && ctx.Rip; ++frame) {
    DWORD64 image_base = 0;
    auto *fn = RtlLookupFunctionEntry(ctx.Rip, &image_base, nullptr);
    if (!fn) {
      ctx.Rip = *reinterpret_cast<DWORD64 *>(ctx.Rsp);
      ctx.Rsp += 8;
    } else {
      void *handler_data = nullptr;
      DWORD64 establisher = 0;
      RtlVirtualUnwind(UNW_FLAG_NHANDLER, image_base, ctx.Rip, fn, &ctx, &handler_data,
                       &establisher, nullptr);
    }
    if (ctx.Rip >= module && ctx.Rip < module + 0x10000000) {
      REXLOG_ERROR("FAULT TRACE:   #{} exe+0x{:X}", frame, ctx.Rip - module);
    } else if (ctx.Rip) {
      HMODULE owner = nullptr;
      wchar_t name[MAX_PATH] = L"?";
      if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                 GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                             reinterpret_cast<LPCWSTR>(ctx.Rip), &owner)) {
        GetModuleFileNameW(owner, name, MAX_PATH);
      }
      char narrow[MAX_PATH];
      WideCharToMultiByte(CP_UTF8, 0, name, -1, narrow, MAX_PATH, nullptr, nullptr);
      REXLOG_ERROR("FAULT TRACE:   #{} {}+0x{:X}", frame, narrow,
                   ctx.Rip - reinterpret_cast<uintptr_t>(owner));
    }
  }
  return EXCEPTION_CONTINUE_SEARCH;
}

constexpr int kHangSeconds = 4;
constexpr int kHangFrames = 24;
constexpr int kHangThreads = 64;

struct ThreadStack {
  DWORD id = 0;
  int count = 0;
  uintptr_t rip[kHangFrames] = {};
};

// Walks a suspended thread. Nothing here allocates or locks: the thread may
// hold the heap or logger lock.
void WalkThread(HANDLE thread, ThreadStack &out) {
  CONTEXT ctx = {};
  ctx.ContextFlags = CONTEXT_FULL;
  if (!GetThreadContext(thread, &ctx)) {
    return;
  }
  for (int frame = 0; frame < kHangFrames && ctx.Rip; ++frame) {
    out.rip[out.count++] = ctx.Rip;
    DWORD64 image_base = 0;
    auto *fn = RtlLookupFunctionEntry(ctx.Rip, &image_base, nullptr);
    if (!fn) {
      ctx.Rip = *reinterpret_cast<DWORD64 *>(ctx.Rsp);
      ctx.Rsp += 8;
    } else {
      void *handler_data = nullptr;
      DWORD64 establisher = 0;
      RtlVirtualUnwind(UNW_FLAG_NHANDLER, image_base, ctx.Rip, fn, &ctx, &handler_data, &establisher, nullptr);
    }
  }
}

void LogFrame(const char *tag, int frame, uintptr_t rip) {
  const auto module = reinterpret_cast<uintptr_t>(GetModuleHandleW(nullptr));
  if (rip >= module && rip < module + 0x10000000) {
    REXLOG_ERROR("{}:   #{} exe+0x{:X}", tag, frame, rip - module);
    return;
  }
  HMODULE owner = nullptr;
  wchar_t name[MAX_PATH] = L"?";
  if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                         reinterpret_cast<LPCWSTR>(rip), &owner)) {
    GetModuleFileNameW(owner, name, MAX_PATH);
  }
  char narrow[MAX_PATH];
  const wchar_t *base = wcsrchr(name, L'\\');
  WideCharToMultiByte(CP_UTF8, 0, base ? base + 1 : name, -1, narrow, MAX_PATH, nullptr, nullptr);
  REXLOG_ERROR("{}:   #{} {}+0x{:X}", tag, frame, narrow, rip - reinterpret_cast<uintptr_t>(owner));
}

void DumpAllThreads() {
  static ThreadStack stacks[kHangThreads];
  int n = 0;
  const DWORD self = GetCurrentThreadId(), pid = GetCurrentProcessId();
  HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
  if (snap == INVALID_HANDLE_VALUE) {
    return;
  }
  THREADENTRY32 te = {sizeof(te)};
  for (BOOL ok = Thread32First(snap, &te); ok && n < kHangThreads; ok = Thread32Next(snap, &te)) {
    if (te.th32OwnerProcessID != pid || te.th32ThreadID == self) {
      continue;
    }
    HANDLE thread = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_LIMITED_INFORMATION, FALSE,
                               te.th32ThreadID);
    if (!thread) {
      continue;
    }
    stacks[n] = {};
    stacks[n].id = te.th32ThreadID;
    if (SuspendThread(thread) != DWORD(-1)) {
      WalkThread(thread, stacks[n]);
      ResumeThread(thread);
      ++n;
    }
    CloseHandle(thread);
  }
  CloseHandle(snap);
  for (int i = 0; i < n; ++i) {
    REXLOG_ERROR("HANG TRACE: thread {}", stacks[i].id);
    for (int f = 0; f < stacks[i].count; ++f) {
      LogFrame("HANG TRACE", f, stacks[i].rip[f]);
    }
  }
}

void HangWatchdog() {
  uint64_t last = 0;
  int still = 0;
  bool reported = false;
  for (;;) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    const uint64_t frames = pgr4::gpu::Video::FrameCount();
    if (frames != last) {
      last = frames;
      still = 0;
      reported = false;
      continue;
    }
    if (frames && !reported && ++still >= kHangSeconds) {
      reported = true;
      REXLOG_ERROR("HANG TRACE: no frame presented for {} s (frame {}), thread stacks follow", kHangSeconds,
                   frames);
      DumpAllThreads();
    }
  }
}

struct Installer {
  Installer() {
    AddVectoredExceptionHandler(1, TraceFault);
    std::thread(HangWatchdog).detach();
  }
} g_installer;

} // namespace
