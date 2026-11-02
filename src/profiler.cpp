// PGR4-Recomp: sampling profiler for performance work, off unless the
// PGR4_PROFILE environment variable is set. A thread samples every thread of
// the process about once a millisecond (suspend, read RIP, resume) and every
// 10 seconds writes profile.txt beside the executable: one line per
// "thread module+offset count". tools/profile_report.py resolves the offsets
// against pgr4_recompiled.map.
#include "profiler.h"

#include <windows.h>
#include <psapi.h>
#include <timeapi.h>
#include <tlhelp32.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#pragma comment(lib, "winmm.lib")

namespace pgr4 {
namespace {

struct Module {
  uintptr_t base = 0, end = 0;
  std::string name;
};

std::vector<Module> ListModules() {
  std::vector<Module> modules;
  HMODULE handles[1024];
  DWORD needed = 0;
  if (!K32EnumProcessModules(GetCurrentProcess(), handles, sizeof(handles), &needed)) {
    return modules;
  }
  for (DWORD i = 0; i < needed / sizeof(HMODULE) && i < 1024; ++i) {
    MODULEINFO info{};
    char name[MAX_PATH] = {};
    if (K32GetModuleInformation(GetCurrentProcess(), handles[i], &info, sizeof(info)) &&
        K32GetModuleBaseNameA(GetCurrentProcess(), handles[i], name, MAX_PATH)) {
      const auto base = reinterpret_cast<uintptr_t>(info.lpBaseOfDll);
      modules.push_back({base, base + info.SizeOfImage, name});
    }
  }
  return modules;
}

void Run() {
  const DWORD self = GetCurrentThreadId();
  const DWORD pid = GetCurrentProcessId();
  timeBeginPeriod(1);
  std::unordered_map<DWORD, HANDLE> threads;
  // (thread id << 40 | module index << 32 | offset) -> samples
  std::unordered_map<uint64_t, uint32_t> counts;
  std::vector<Module> modules = ListModules();
  auto last_write = std::chrono::steady_clock::now();
  auto last_scan = last_write - std::chrono::seconds(10);
  std::vector<std::pair<DWORD, uintptr_t>> batch;
  for (;;) {
    const auto now = std::chrono::steady_clock::now();
    if (now - last_scan >= std::chrono::seconds(1)) {
      last_scan = now;
      HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
      if (snap != INVALID_HANDLE_VALUE) {
        THREADENTRY32 te{sizeof(te)};
        for (BOOL ok = Thread32First(snap, &te); ok; ok = Thread32Next(snap, &te)) {
          if (te.th32OwnerProcessID == pid && te.th32ThreadID != self && !threads.count(te.th32ThreadID)) {
            if (HANDLE h = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION, FALSE,
                                      te.th32ThreadID)) {
              threads[te.th32ThreadID] = h;
            }
          }
        }
        CloseHandle(snap);
      }
    }
    // Nothing is allocated while a thread is suspended (it may hold the heap
    // lock): RIPs go to a reserved batch first.
    batch.clear();
    batch.reserve(threads.size());
    for (auto it = threads.begin(); it != threads.end();) {
      CONTEXT ctx{};
      ctx.ContextFlags = CONTEXT_CONTROL;
      if (SuspendThread(it->second) == DWORD(-1)) {
        CloseHandle(it->second);
        it = threads.erase(it);
        continue;
      }
      const BOOL got = GetThreadContext(it->second, &ctx);
      ResumeThread(it->second);
      if (got) {
        batch.emplace_back(it->first, uintptr_t(ctx.Rip));
      }
      ++it;
    }
    for (const auto &[tid, rip] : batch) {
      uint32_t index = 0xFF;
      uintptr_t offset = rip & 0xFFFFFFFFu;
      for (uint32_t m = 0; m < modules.size() && m < 0xFF; ++m) {
        if (rip >= modules[m].base && rip < modules[m].end) {
          index = m;
          offset = rip - modules[m].base;
          break;
        }
      }
      if (index == 0xFF && modules.size() < 0xFF) {
        modules = ListModules(); // a DLL loaded since
      }
      ++counts[(uint64_t(tid) << 40) | (uint64_t(index) << 32) | uint32_t(offset)];
    }
    if (now - last_write >= std::chrono::seconds(10)) {
      last_write = now;
      if (std::FILE *f = std::fopen("profile.txt", "w")) {
        for (const auto &[key, n] : counts) {
          const uint32_t index = uint32_t(key >> 32) & 0xFF;
          std::fprintf(f, "%u %s %08X %u\n", uint32_t(key >> 40),
                       index < modules.size() ? modules[index].name.c_str() : "?", uint32_t(key), n);
        }
        std::fclose(f);
      }
      // profile.reset beside the exe starts a fresh window.
      if (std::FILE *r = std::fopen("profile.reset", "rb")) {
        std::fclose(r);
        std::remove("profile.reset");
        counts.clear();
      }
    }
    Sleep(1);
  }
}

} // namespace

void StartProfiler() {
  if (!std::getenv("PGR4_PROFILE")) {
    return;
  }
  std::thread(Run).detach();
}

} // namespace pgr4
