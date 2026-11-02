// Title update and DLC packages from the content folder (--content_root,
// default <game_data_root>/Content).
#pragma once

#include <cstdint>
#include <filesystem>

namespace pgr4::content {

// Extracts the title update package into tu_dir (with a copy of the base
// default.xex beside its default.xexp) unless it is already there. Runs from
// OnConfigurePaths, before the XEX image is loaded. False if no usable TU.
bool InstallTitleUpdate(const std::filesystem::path &game_data_root,
                        const std::filesystem::path &tu_dir);

// Installs every PGR4 marketplace package in the content folder that is not
// installed yet. Needs the executable module loaded (title id), so it runs
// from OnPostLaunchModule, before the guest main thread resumes.
void InstallDLC();

// Debug: --dump_guest_image=<file> writes the loaded (patched) image, the
// first `size` bytes from 0x82000000 (PGR4's by default).
void DumpGuestImage(uint32_t size = 0xE50000);

// --dump_cvars <file>: writes every cvar (name, category, type, default,
// allowed values, description; tab separated) for the launcher, then exits.
void DumpCvarsAndExitIfRequested();

}  // namespace pgr4::content
