// PGR4-Recomp hook helpers.
//
// The same no-op macros re:Blue's renderer uses (reblue src/core/hooks.h),
// for D3D calls the host renderer does not need to model. REX_STUB warns on
// every call and is kept for unimplemented placeholders.
#pragma once

#include <rex/hook.h>

// Intentional no-op with no log output.
#define BD_NOOP(subroutine)                                                    \
  extern "C" REX_FUNC(subroutine) {                                            \
    (void)ctx;                                                                 \
    (void)base;                                                                \
  }

// BD_NOOP that returns a constant u32 in r3, for callers comparing the result.
#define BD_NOOP_RETURN(subroutine, value)                                      \
  extern "C" REX_FUNC(subroutine) {                                            \
    (void)base;                                                                \
    ctx.r3.u64 = (value);                                                      \
  }
