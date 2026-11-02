// Title update and DLC packages, both read from one content folder
// (--content_root, default <game_data_root>/Content).
//
// Title update: its STFS package is extracted once into
// <game_data_root>/TitleUpdate together with a copy of the base default.xex,
// so the runtime finds default.xexp beside the image it loads and patches it to
// match the recompiled code (built from the same patched image).
//
// DLC: same scheme as re:Blue's DLC catalog. Packs are extracted into
// <user_data_root>/0000000000000000/<title>/00000002/<name>/ with a
// Headers/00000002/<name>.header beside them. The header's license mask is
// what the guest checks, so it is written from the package's own licenses.
#include "content/dlc.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <queue>
#include <span>
#include <system_error>
#include <vector>

#include <fmt/format.h>

#include <rex/cvar.h>
#include <rex/filesystem.h>
#include <rex/filesystem/devices/stfs_container_device.h>
#include <rex/filesystem/entry.h>
#include <rex/filesystem/file.h>
#include <rex/hook.h>
#include <rex/logging.h>
#include <rex/runtime.h>
#include <rex/string.h>
#include <rex/system/kernel_state.h>
#include <rex/system/xam/content_manager.h>
#include <rex/system/xcontent.h>

REXCVAR_DEFINE_STRING(dump_cvars, "", "PGR4", "Write the cvar list to this file and exit (used by the launcher)");
REXCVAR_DEFINE_STRING(dump_guest_image, "", "PGR4", "Debug: write the loaded guest image to this file");
REXCVAR_DEFINE_STRING(content_root, "", "PGR4",
                      "Folder holding the title update and DLC packages (default: <game_data_root>/Content)");

namespace pgr4::content {

namespace {

namespace fs = std::filesystem;
using rex::filesystem::StfsContainerDevice;
using rex::system::XContentType;

constexpr uint32_t kTitleId = 0x4D5307F9;

fs::path ContentRoot(const fs::path& game_data_root) {
  const std::string root = REXCVAR_GET(content_root);
  return root.empty() ? game_data_root / "Content" : fs::path(root);
}

// Every STFS package in the content folder for this title of the given type.
std::vector<fs::path> FindPackages(const fs::path& root, XContentType type) {
  std::vector<fs::path> out;
  std::error_code ec;
  if (!fs::is_directory(root, ec)) {
    return out;
  }
  for (const auto& entry : fs::directory_iterator(root, ec)) {
    if (!entry.is_regular_file(ec)) {
      continue;
    }
    auto header = StfsContainerDevice::ReadPackageHeader(entry.path());
    if (header && header->header.is_magic_valid() &&
        header->metadata.execution_info.title_id == kTitleId && header->metadata.content_type == type) {
      out.push_back(entry.path());
    }
  }
  return out;
}

bool CopyEntry(rex::filesystem::Entry& entry, const fs::path& dest) {
  rex::filesystem::File* file = nullptr;
  if (entry.Open(rex::filesystem::FileAccess::kFileReadData, &file) != 0 || !file) {
    return false;
  }
  std::ofstream out(dest, std::ios::binary | std::ios::trunc);
  std::vector<uint8_t> buffer(1 << 20);
  size_t offset = 0;
  bool ok = static_cast<bool>(out);
  while (ok && offset < entry.size()) {
    const size_t want = std::min(buffer.size(), size_t(entry.size() - offset));
    size_t got = 0;
    file->ReadSync(std::span<uint8_t>(buffer.data(), want), offset, &got);
    ok = got == want && out.write(reinterpret_cast<const char*>(buffer.data()), std::streamsize(got));
    offset += got;
  }
  file->Destroy();
  return ok && out.good();
}

bool ExtractPackage(const fs::path& package, const fs::path& dest) {
  StfsContainerDevice device("", package);
  if (!device.Initialize()) {
    return false;
  }
  auto* root = device.ResolvePath("");
  if (!root) {
    return false;
  }
  std::queue<rex::filesystem::Entry*> queue;
  queue.push(root);
  while (!queue.empty()) {
    auto* entry = queue.front();
    queue.pop();
    for (auto& child : entry->children()) {
      queue.push(child.get());
    }
    if (entry == root) {
      continue;
    }
    const auto path = dest / rex::to_path(rex::string::utf8_fix_path_separators(entry->path()));
    std::error_code ec;
    if (entry->attributes() & rex::filesystem::kFileAttributeDirectory) {
      fs::create_directories(path, ec);
    } else {
      fs::create_directories(path.parent_path(), ec);
      if (!CopyEntry(*entry, path)) {
        return false;
      }
    }
  }
  return true;
}

// The title update also patches the arcade's Geometry Wars (gw.xexp); its
// base image goes beside the patch too. Done on every boot, as TitleUpdate
// folders installed before Geometry Wars ran have the patch only.
void PlaceGeometryWarsBase(const fs::path& game_data_root, const fs::path& tu_dir) {
  std::error_code ec;
  if (!fs::is_regular_file(tu_dir / "gw.xexp", ec) || fs::is_regular_file(tu_dir / "gw.xex", ec)) {
    return;
  }
  fs::copy_file(game_data_root / "gw.xex", tu_dir / "gw.xex", ec);
  if (ec) {
    REXLOG_ERROR("[tu] could not copy gw.xex: {}", ec.message());
  }
}

}  // namespace

bool InstallTitleUpdate(const fs::path& game_data_root, const fs::path& tu_dir) {
  std::error_code ec;
  if (fs::is_regular_file(tu_dir / "default.xexp", ec) && fs::is_regular_file(tu_dir / "default.xex", ec)) {
    PlaceGeometryWarsBase(game_data_root, tu_dir);
    return true;
  }

  const auto packages = FindPackages(ContentRoot(game_data_root), XContentType::kInstaller);
  if (packages.empty()) {
    REXLOG_ERROR("[tu] no title update package in {}", ContentRoot(game_data_root).string());
    return false;
  }

  // Extract into a temporary folder first so a failed install never leaves a
  // half-written TitleUpdate folder that the next boot would accept.
  const auto staging = tu_dir.parent_path() / (tu_dir.filename().string() + ".tmp");
  fs::remove_all(staging, ec);
  fs::create_directories(staging, ec);
  if (!ExtractPackage(packages.front(), staging) || !fs::is_regular_file(staging / "default.xexp", ec)) {
    REXLOG_ERROR("[tu] could not extract {}", packages.front().string());
    fs::remove_all(staging, ec);
    return false;
  }
  // The runtime applies a .xexp to the .xex beside it, so the base image goes
  // next to the patch. The game root's own default.xex stays untouched.
  fs::copy_file(game_data_root / "default.xex", staging / "default.xex",
                fs::copy_options::overwrite_existing, ec);
  if (ec) {
    REXLOG_ERROR("[tu] could not copy default.xex: {}", ec.message());
    fs::remove_all(staging, ec);
    return false;
  }
  fs::remove_all(tu_dir, ec);
  fs::rename(staging, tu_dir, ec);
  if (ec) {
    REXLOG_ERROR("[tu] could not create {}: {}", tu_dir.string(), ec.message());
    return false;
  }
  REXLOG_INFO("[tu] installed {} into {}", packages.front().filename().string(), tu_dir.string());
  PlaceGeometryWarsBase(game_data_root, tu_dir);
  return true;
}

void InstallDLC() {
  auto* rt = rex::Runtime::instance();
  auto* kernel = rt ? rt->kernel_state() : nullptr;
  if (!kernel || !kernel->content_manager()) {
    return;
  }

  const uint32_t title_id = kernel->title_id();
  const auto content_dir = fs::absolute(rt->user_data_root()) / "0000000000000000" /
                           fmt::format("{:08X}", title_id) /
                           fmt::format("{:08X}", uint32_t(XContentType::kMarketplaceContent));

  int installed = 0, present = 0;
  std::error_code ec;
  for (const auto& package : FindPackages(ContentRoot(rt->game_data_root()), XContentType::kMarketplaceContent)) {
    if (fs::is_directory(content_dir / package.filename(), ec)) {
      ++present;
      continue;
    }
    const auto result = kernel->content_manager()->InstallContent(package);
    if (result != 0) {
      REXLOG_ERROR("[dlc] install failed ({:08X}): {}", result, package.string());
      continue;
    }
    REXLOG_INFO("[dlc] installed {}", package.filename().string());
    ++installed;
  }
  REXLOG_INFO("[dlc] {} pack(s) installed, {} already present", installed, present);
}

void DumpGuestImage(uint32_t size) {
  const std::string path = REXCVAR_GET(dump_guest_image);
  if (path.empty()) {
    return;
  }
  constexpr uint32_t kBase = 0x82000000;
  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char*>(REX_KERNEL_MEMORY()->TranslateVirtual(kBase)), size);
  REXLOG_INFO("[debug] guest image written to {}", path);
}

void DumpCvarsAndExitIfRequested() {
  const std::string path = REXCVAR_GET(dump_cvars);
  if (path.empty()) {
    return;
  }
  static const char *kTypes[] = {"bool", "int32", "int64", "uint32", "uint64", "double", "string", "command"};
  auto clean = [](std::string text) {
    for (char &c : text) {
      if (c == '\t' || c == '\n' || c == '\r') {
        c = ' ';
      }
    }
    return text;
  };
  std::ofstream out(path, std::ios::binary);
  for (const auto &e : rex::cvar::GetRegistry()) {
    if (e.type == rex::cvar::FlagType::Command) {
      continue;
    }
    std::string allowed;
    for (const auto &v : e.constraints.allowed_values) {
      allowed += (allowed.empty() ? "" : "|") + v;
    }
    out << clean(e.name) << '\t' << clean(e.category) << '\t' << kTypes[size_t(e.type)] << '\t'
        << clean(e.default_value) << '\t' << clean(allowed) << '\t' << (e.is_debug_only ? 1 : 0) << '\t'
        << clean(e.description) << '\n';
  }
  out.close();
  std::_Exit(0);
}

}  // namespace pgr4::content
