# PGR4-Recomp
This project intends to port Project Gotham Racing 4 to PC / any x86_64 hardware (providing they can run an exe).
> See the current status of the recompilation [here!](https://github.com/beatrixzy/PGR4-Decomp/blob/main/Status.md)
> This project is now in the works of being compatible with the [Goopie Launcher!](https://goopie.xyz)
## Current **known** issues
* Pre-AVX2 machines can't run this at the moment, no idea why but they can't
* Crashes on Steam Deck (acts as if its loading, then crashes)
* Lotus Espirit Essex Turbo doesn't seem to load and freezes the game
* Game being on a different drive appears to cause issues, for now, keep the EXE and extracted game files on the same drive (preferably the C drive)
## Current Progress
- [x] GOD Files
- [x] C++ Converted files
- [X] Launchable EXE (on AVX2+ hardware, non AVX2 capable hw doesn't seem to like it.)
- [ ] Installer EXE (provide your own files) - Possibly not needed providing instructions are made.
### Optional features
- [ ] Adjustable graphics settings
- [ ] Online (unlikely)
- [ ] Mod Support (equally as unlikely as online) 
## Notes
* This is a ONE person effort at the current moment in time, commits will NOT be frequent.
* ANY help is appreciated greatly, feel free to contribute to this project!
## Credits
* [ReXGlue SDK](https://github.com/rexglue/rexglue-sdk) - For allowing C++ to be generated AHEAD
* Tera in the ReXGlue Discord - For allowing me to pick up from where they left off with ReXGlue
* xextool - For allowing me to extract my XEX from a dumped ISO (seriously, that was a LIFE saver)

