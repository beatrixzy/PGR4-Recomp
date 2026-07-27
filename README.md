# PGR4-Decomp [TESTING BRANCH]
This project intends to decompile (and eventually port to pc) Project Gotham Racing 4.
* a Compiled .EXE has been provided in this branch, however you need to provide your own game files (dump it as an ISO, then use XEX tool to extract and decrypt (if needed))

## How to run:
* Go into the directory containing ``pgr4.exe``
* Go to the top bar in Windows, click, then type "cmd"
* In the newly opened command prompt window, type ``pgr4.exe --game_data_root [directory/where/extracted/pgr4/files/are/located]``
* The game should launch! To fix the black screen bug, press ``F4`` then search for ``execute_unclipped_draw_vs_on_cpu``, turn that on and the UI is now functional!

### Please note:
This project is VERY early in its progress, however updates are quite frequent!
## Current **known** issues
* Pre-AVX2 machines can't run this at the moment, no idea why but they can't
* NVIDIA GPU's (or at least, mine) have some graphical issues related to the UI, where some aspects are just pitch black (appears to be some functions needed to be manually lifted)
* Crashes on Steam Deck (acts as if its loading, then crashes)
## Current Progress
- [x] GOD Files
- [x] C++ Converted files
- [X] Launchable EXE (on AVX2+ hardware, non AVX2 capable hw support unknown)
- [ ] Installer EXE (provide your own files)
* Note, races are functional no matter the graphical issues!
### Optional features
- [ ] Adjustable graphics settings
- [ ] Online (unlikely)
- [ ] Mod Support (equally as unlikely as online) 
## Notes
* This is a ONE person effort at the current moment in time, commits towards actual progress will NOT be frequent.
* ANY help is appreciated greatly, feel free to contribute to this project!
## Credits
* [ReXGlue SDK](https://github.com/rexglue/rexglue-sdk) - For allowing C++ to be generated AHEAD
* Tera in the ReXGlue Discord - For allowing me to pick up from where they left off with ReXGlue
* xextool - For allowing me to extract my XEX from a dumped ISO (seriously, that was a LIFE saver)

