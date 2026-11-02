// pgr4_recompiled - music volume bridge
//
// PGR4 never calls XMPSetVolume. On hardware it captures the XMP output
// (XMPCaptureOutput) into its own mixer and drives the music slider, fades
// and ducking by setting a volume parameter on that captured voice. The
// runtime plays XMP music on its own host stream instead, so that parameter
// never reached anything audible.
//
// sub_827FFBB8 is the music player's SetVolume(float); its only caller is
// the sound manager's vtable slot 39 (sub_827E3190), which the music manager
// uses for the profile music volume and its fades. Forward the value to the
// host music stream after letting the original run.

#include <rex/hook.h>
#include <rex/runtime.h>
#include <rex/system/kernel_state.h>

REX_EXTERN(__imp__sub_827FFBB8);

REX_HOOK_RAW(sub_827FFBB8) {
  const float volume = static_cast<float>(ctx.f1.f64);
  __imp__sub_827FFBB8(ctx, base);

  if (auto* audio = REX_KERNEL_STATE()->emulator()->audio_system()) {
    audio->SetMusicVolume(volume);
  }
}
