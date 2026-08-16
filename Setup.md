## How to run on Windows:
* Download the latest release.
* Download [extract-xiso](https://github.com/xboxdev/extract-xiso) to extract game files from an ISO.
* Download [god2iso](https://github.com/raburton/god2iso) (necessary for a dump from an actual disk!!!)
* Run god2iso and convert the Game On Demand (god) to an ISO file.
* Run extract-xiso and save the game files to a safe directory.
* Open the ``run.bat`` file in the PGR4 Recomp area.
* Edit the path to the game data root and save changes.
* Run the ``run.bat``
* The game should launch!

## How to run on SteamOS:
* Go into Desktop Mode
* Follow the same extracting steps as Windows, you may need to use Proton to get them to work.
* Instead of opening the ``run.bat``, Add ``pgr4-recomp.exe`` to your Steam Library.
* Navigate to the section in your library named after the EXE and click the cog / gear icon.
* Edit the compatibility layer and set it to ``Proton Experimental``, this is the recommended Proton version as it is known working.
* Then, navigate to the launch arguments and put ``--game_data_root "[directory/to/game/files]" --execute_unclipped_draw_vs_on_cpu --gpu_plugin="xenos"`` in there, all of these arguments are important as the game will **NOT** launch without them.
* Exit out of the gear menu and test in **Gaming Mode** if you don't have access to an external keyboard, if you do, test in **Desktop Mode** to confirm the game launches.

### Please Note:
Steps are due to change as this project evolves, these are just steps for how to get it up and running on Windows and SteamOS, other operating systems will have steps placed in here once their builds are confirmed working.
