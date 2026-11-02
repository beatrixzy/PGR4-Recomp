// PGR4-Recomp application bridge.
//
// PGR4_D3D12_RENDERER=1: Plume/D3D12 native renderer (src/gpu).
// PGR4_NATIVE_RENDERER=1: Vulkan native renderer.
// both 0: SDK Xenos GPU plugin.
#pragma once

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <thread>

#include <rex/cvar.h>
#include <rex/rex_app.h>
#include <rex/system.h>
#include <rex/kernel/xam/module.h>
#include <rex/ui/window.h>

#include "content/dlc.h"
#include "profiler.h"

#if PGR4_D3D12_RENDERER
#include <rex/ui/imgui_drawer.h>

#include "gpu/hooks/hooks.h"
#include "gpu/imgui_drawer.h"
#include "gpu/renderer.h"
#include "gpu/video.h"
#elif PGR4_NATIVE_RENDERER
#include "render/graphics_system.h"
#endif

// Geometry Wars: Waves (gw.xex), the arcade machine in PGR4's garage, is
// linked in as a second title (gw/gw_manifest.toml, symbols prefixed gw_).
// Only its image info is named here: its generated headers redefine the same
// configuration macros as PGR4's.
extern const rex::PPCImageInfo gw_PPCImageConfig;

REXCVAR_DEFINE_STRING(boot_title, "", "PGR4",
                      "Start in another title instead of PGR4: gw (Geometry Wars: Waves), or "
                      "gw_arcade (Geometry Wars with the launch data PGR4's arcade passes)")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);

class Pgr4RecompiledApp : public rex::ReXApp {
public:
  using rex::ReXApp::ReXApp;

  static std::unique_ptr<rex::ui::WindowedApp>
  Create(rex::ui::WindowedAppContext &ctx) {
    const std::string boot = REXCVAR_GET(boot_title);
    const bool gw = boot == "gw" || boot == "gw_arcade";
    auto app = std::unique_ptr<Pgr4RecompiledApp>(new Pgr4RecompiledApp(
        ctx, "pgr4_recompiled", gw ? gw_PPCImageConfig : PPCImageConfig));
    app->boot_gw_ = gw;
    app->boot_gw_arcade_ = boot == "gw_arcade";
    return app;
  }

protected:
  // Title Update #2. Its code is recompiled from TitleUpdate/default.xex with
  // default.xexp beside it (the runtime applies a sibling .xexp at load, so the
  // image matches the generated code), and the folder is also update:\ for
  // the TU's own data. Kept out of the game root so the base default.xex stays
  // unpatched for base builds sharing the same game files.
  static constexpr const char *kTitleUpdateDir = "TitleUpdate";
  static constexpr const char *kLeaderboardServer =
      "https://pgr4-leaderboard.pgr4-leaderboard.workers.dev";

  void OnConfigurePaths(rex::PathConfig &paths) override {
    pgr4::content::DumpCvarsAndExitIfRequested();
    // Online leaderboards (PGR4-Leaderboard on Cloudflare) unless the launcher
    // or command line set stats_server to another URL; "off" turns them off.
    if (const auto server = rex::cvar::GetFlagByName("stats_server");
        server.empty() || server == "on") {
      rex::cvar::SetFlagByName("stats_server", kLeaderboardServer);
    }
    const auto tu_dir = paths.game_data_root / kTitleUpdateDir;
    if (!pgr4::content::InstallTitleUpdate(paths.game_data_root, tu_dir)) {
      rex::ShowSimpleMessageBox(
          rex::SimpleMessageBoxType::Error,
          "PGR4 needs Title Update #2.\n\nPut the title update package (TU_16L61VP_...) in the "
          "Content folder inside your game folder, or pass --content_root <folder>, then start "
          "the game again. See README.md for details.");
    }
    if (paths.update_data_root.empty()) {
      paths.update_data_root = tu_dir;
    }
  }

  // Both titles load from TitleUpdate, beside their patches (default.xexp,
  // gw.xexp), when the title update is there.
  void OnLoadXexImage(std::string &xex_image) override {
    if (boot_gw_) {
      boot_gw_ = false; // --boot_title=gw: the first image is Geometry Wars
      xex_image = "game:\\gw.xex";
    }
    const std::string name = XexName(xex_image);
    if (name == "default.xex" ||
        (name == "gw.xex" && std::filesystem::is_regular_file(update_data_root() / "gw.xexp") &&
         std::filesystem::is_regular_file(update_data_root() / "gw.xex"))) {
      xex_image = std::string("game:\\") + kTitleUpdateDir + "\\" + name;
    }
  }

  // PGR4's arcade launches Geometry Wars with XamLoaderLaunchTitle
  // ("GAME:\gw.xex"), and Geometry Wars goes back the same way: the app
  // switches titles in-process, as the 360 does.
  std::optional<rex::PPCImageInfo> OnResolveTitle(const std::string &xex_image) override {
    const std::string name = XexName(xex_image);
    if (name == "gw.xex") {
      return gw_PPCImageConfig;
    }
    if (name == "default.xex") {
      return PPCImageConfig;
    }
    return std::nullopt;
  }

  // Lower-case file name of a guest path ("GAME:\gw.xex" -> "gw.xex").
  static std::string XexName(std::string_view path) {
    const size_t slash = path.find_last_of("\\/:");
    std::string name(slash == std::string_view::npos ? path : path.substr(slash + 1));
    std::transform(name.begin(), name.end(), name.begin(),
                   [](unsigned char c) { return char(std::tolower(c)); });
    return name;
  }

  static constexpr uint32_t kGeometryWarsTitleId = 0x4D5307F9;
  bool boot_gw_ = false;
  bool boot_gw_arcade_ = false;

#if PGR4_D3D12_RENDERER
  void OnPreSetup(rex::RuntimeConfig &config) override {
    // Null graphics makes Runtime::Setup skip SDK GPU setup; the Plume device
    // is brought up in OnPreLaunchModule (same split as re:Blue).
    config.gpu_plugin.clear();
    config.graphics = nullptr;
  }

  void OnPreLaunchModule() override {
    if (window()) {
      pgr4::gpu::Video::CreateHostDevice(window()->GetNativeWindowHandle());
    }
    InstallOverlayDrawHook();
  }

  // Title id is known once the module is loaded; the guest has not run yet.
  // After a title switch this runs again, for Geometry Wars or PGR4.
  void OnPostLaunchModule(rex::system::XThread *) override {
    const bool pgr4 = runtime()->kernel_state()->title_id() != kGeometryWarsTitleId;
    pgr4::gpu::Renderer::SetTitle(pgr4);
    if (first_launch_) {
      first_launch_ = false;
      pgr4::StartProfiler();
      std::thread([this] { WatchForLaunchFile(); }).detach();
    }
    if (pgr4) {
      pgr4::content::InstallDLC();
      pgr4::content::DumpGuestImage();
    } else {
      pgr4::content::DumpGuestImage(uint32_t(gw_PPCImageConfig.image_size));
      if (boot_gw_arcade_) {
        // PGR4's arcade (sub_825CC2B8) sets launch data "2" before launching.
        boot_gw_arcade_ = false;
        if (auto xam = runtime()->kernel_state()->GetKernelModule<rex::kernel::xam::XamModule>(
                "xam.xex")) {
          xam->loader_data().launch_data_present = true;
          xam->loader_data().launch_data = {'2'};
        }
      }
    }
  }

  // The old title's guest threads have stopped; its memory goes next.
  void OnTitleSwitch(const std::string &) override {
    pgr4::gpu::Renderer::ResetForTitleSwitch();
    pgr4::gpu::hooks::ResetForTitleSwitch();
  }

  bool first_launch_ = true;

  // Debug: a launch.now file in the working directory holding a guest path
  // ("GAME:\gw.xex", "GAME:\default.xex") makes the running title launch it,
  // as XamLoaderLaunchTitle does, so title switches can be tested unattended.
  void WatchForLaunchFile() {
    for (;;) {
      std::this_thread::sleep_for(std::chrono::milliseconds(250));
      std::string path;
      if (std::FILE *f = std::fopen("launch.now", "rb")) {
        char text[260] = {};
        std::fgets(text, sizeof(text), f);
        std::fclose(f);
        std::remove("launch.now");
        path = text;
        while (!path.empty() && (path.back() == '\n' || path.back() == '\r' || path.back() == ' ')) {
          path.pop_back();
        }
      }
      if (path.empty()) {
        continue;
      }
      app_context().CallInUIThread([this, path] {
        auto *kernel = runtime() ? runtime()->kernel_state() : nullptr;
        auto xam = kernel ? kernel->GetKernelModule<rex::kernel::xam::XamModule>("xam.xex") : nullptr;
        if (!xam) {
          return;
        }
        REXLOG_INFO("[debug] launch.now: launching {}", path);
        xam->loader_data().launch_path = path;
        kernel->TerminateTitle();
      });
    }
  }

  // The SDK builds its ImGui overlays (Backtick: console log, F3: debug,
  // F4: settings) on this drawer when the app brings its own renderer.
  std::unique_ptr<rex::ui::ImmediateDrawer> OnCreateImmediateDrawer() override {
    return std::make_unique<pgr4::gpu::ImGuiOverlayDrawer>();
  }

  // Present runs on the guest thread and ImGui on the UI thread, so the draw
  // is marshalled synchronously; Present stays parked (holding the frame)
  // while the UI thread records into its command list. Same as re:Blue.
  void InstallOverlayDrawHook() {
    pgr4::gpu::Video::SetOverlayDrawHook(
        [this](plume::RenderCommandList *cmd, plume::RenderFramebuffer *fb, uint32_t w,
               uint32_t h) {
          if (!imgui_drawer() || !imgui_drawer()->HasDialogs()) {
            return;
          }
          app_context().CallInUIThreadSynchronous([this, cmd, fb, w, h] {
            pgr4::gpu::OverlayDrawContext ctx(w, h, cmd, fb);
            imgui_drawer()->Draw(ctx);
          });
        });
  }

  void OnWindowPixelSizeChanged(uint32_t, uint32_t) override {
    pgr4::gpu::Video::RequestResize();
  }

  void OnWindowRestored() override { pgr4::gpu::Video::RequestResize(); }

  void OnShutdown() override { pgr4::gpu::Video::Shutdown(); }

  std::unique_ptr<rex::ui::AchievementNotificationDialog>
  CreateAchievementNotificationDialog() override {
    return nullptr;
  }
#elif PGR4_NATIVE_RENDERER
  void OnPreSetup(rex::RuntimeConfig &config) override {
    auto graphics = std::make_unique<pgr4::render::Pgr4GraphicsSystem>();
    native_graphics_ = graphics.get();
    config.gpu_plugin.clear();
    config.graphics = std::move(graphics);
  }

  void OnPostSetup() override {
    if (!native_graphics_ || !window()) {
      return;
    }
    native_graphics_->AttachWindow(window()->GetNativeWindowHandle());
    native_graphics_->LoadShaderDirectory(
        std::filesystem::current_path() / "pgr4-native-shaders");
  }

  void OnWindowPixelSizeChanged(uint32_t, uint32_t) override {
    if (native_graphics_) {
      native_graphics_->ResizeSwapchain();
    }
  }

  void OnWindowRestored() override {
    if (native_graphics_) {
      native_graphics_->ResizeSwapchain();
    }
  }

  void OnShutdown() override { native_graphics_ = nullptr; }

  std::unique_ptr<rex::ui::AchievementNotificationDialog>
  CreateAchievementNotificationDialog() override {
    return nullptr;
  }

private:
  pgr4::render::Pgr4GraphicsSystem *native_graphics_ = nullptr;
#else
  void OnPreSetup(rex::RuntimeConfig &config) override {
    // Default to Xenos so the build runs without --gpu_plugin.
    if (config.gpu_plugin.empty()) {
      config.gpu_plugin = "xenos";
    }
  }
#endif
};
