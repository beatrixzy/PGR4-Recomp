# PGR4-Decomp
This project intends to port Project Gotham Racing 4 to PC / any x86_64 hardware (providing they can run an exe).
## How to run:
* Download the latest release
* Download [extract-xiso](https://github.com/xboxdev/extract-xiso) to extract game files from an ISO
* Download [god2iso](https://github.com/raburton/god2iso) (necessary for a dump from an actual disk!!!)
* Run god2iso and convert the Game On Demand (god) to an ISO file
* Run extract-xiso and save the game files to a safe directory
* Open the ``run.bat`` file in the PGR4 Recomp area
* Edit the path to the game data root and save changes.
* Run the ``start.bat``
* The game should launch!
## Current **known** issues
* Pre-AVX2 machines can't run this at the moment, no idea why but they can't
* ~~NVIDIA GPU's (or at least, mine) have some graphical issues related to the UI, where some aspects are just pitch black~~ Fix found for this problem.
* Crashes on Steam Deck (acts as if its loading, then crashes)
## Current status of the game
- [x] UI Renders
- [X] Races work
- [X] AI Works
- [X] Fully stable
- [ ] Graphics Work (mostly, just some minor bugs, leaving unchecked for now)

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

