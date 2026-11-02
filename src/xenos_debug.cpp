// pgr4_recompiled - xenos renderer debug: dumps the car's environment cube
// (the D3D texture the title's cube passes resolve into) from guest memory
// when dumpcube.now appears next to the exe, as ground truth for the native
// renderer. Built only without PGR4_D3D12_RENDERER.

#include <cstdio>

#include <rex/hook.h>
#include <rex/logging.h>
#include <rex/memory/utils.h>
#include <rex/runtime.h>
#include <rex/system/kernel_state.h>

REX_EXTERN(__imp__D3DDevice_Swap);

namespace {

constexpr uint32_t kCubeTexture = 0x06C10478; // D3DCubeTexture from the native dump's resolve log

uint32_t LoadU32(uint32_t va) {
  return rex::memory::load_and_swap<uint32_t>(REX_KERNEL_MEMORY()->TranslateVirtual(va));
}

void DumpCube() {
  std::FILE *flag = std::fopen("dumpcube.now", "rb");
  if (!flag) {
    return;
  }
  std::fclose(flag);
  std::remove("dumpcube.now");
  uint32_t fetch[6];
  for (int i = 0; i < 6; ++i) {
    fetch[i] = LoadU32(kCubeTexture + 0x1C + i * 4);
  }
  const uint32_t base = fetch[1] & 0x1FFFF000u; // physical address bits
  REXLOG_INFO("xenos cube dump: fetch {:08X} {:08X} {:08X} {:08X} {:08X} {:08X}", fetch[0], fetch[1], fetch[2],
              fetch[3], fetch[4], fetch[5]);
  if (std::FILE *out = std::fopen("cube_xenos.bin", "wb")) {
    std::fwrite(REX_KERNEL_MEMORY()->TranslatePhysical(base), 1, 4u << 20, out);
    std::fclose(out);
  }
}

} // namespace

REX_HOOK_RAW(D3DDevice_Swap) {
  DumpCube();
  __imp__D3DDevice_Swap(ctx, base);
}
