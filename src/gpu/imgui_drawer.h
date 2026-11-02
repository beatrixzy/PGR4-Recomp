// PGR4-Recomp native renderer: Plume/D3D12 ImmediateDrawer for the SDK's
// ImGui overlays (console log, debug, settings) in detached mode
// (config.graphics == nullptr). Port of re:Blue's ImGuiOverlayDrawer.
#pragma once

#include <cstdint>
#include <memory>

#include <rex/ui/immediate_drawer.h>
#include <rex/ui/presenter.h>

namespace plume {
struct RenderCommandList;
struct RenderDescriptorSet;
struct RenderSampler;
struct RenderFramebuffer;
struct RenderPipeline;
struct RenderPipelineLayout;
struct RenderShader;
struct RenderTexture;
struct RenderTextureView;
} // namespace plume

namespace pgr4::gpu {

// What Present hands the overlay: the open frame list and the back buffer.
class OverlayDrawContext final : public rex::ui::AppUIDrawContext {
public:
  OverlayDrawContext(uint32_t w, uint32_t h, plume::RenderCommandList *cmd,
                     plume::RenderFramebuffer *fb)
      : rex::ui::AppUIDrawContext(w, h), cmd_(cmd), fb_(fb) {}
  plume::RenderCommandList *command_list() const { return cmd_; }
  plume::RenderFramebuffer *framebuffer() const { return fb_; }

private:
  plume::RenderCommandList *cmd_;
  plume::RenderFramebuffer *fb_;
};

// Each texture owns a one-entry descriptor set, so the overlay never touches
// large bindless tables (which SM 5.1-only drivers can't handle).
class PlumeImmediateTexture final : public rex::ui::ImmediateTexture {
public:
  PlumeImmediateTexture(uint32_t w, uint32_t h, std::unique_ptr<plume::RenderTexture> texture,
                        std::unique_ptr<plume::RenderTextureView> view,
                        std::unique_ptr<plume::RenderDescriptorSet> set);
  ~PlumeImmediateTexture() override;
  plume::RenderDescriptorSet *descriptor_set() const { return set_.get(); }

private:
  std::unique_ptr<plume::RenderTexture> texture_;
  std::unique_ptr<plume::RenderTextureView> view_;
  std::unique_ptr<plume::RenderDescriptorSet> set_;
};

class ImGuiOverlayDrawer final : public rex::ui::ImmediateDrawer {
public:
  ImGuiOverlayDrawer();
  ~ImGuiOverlayDrawer() override;

  std::unique_ptr<rex::ui::ImmediateTexture>
  CreateTexture(uint32_t width, uint32_t height, rex::ui::ImmediateTextureFilter filter,
                bool is_repeated, const uint8_t *data) override;
  void Begin(rex::ui::UIDrawContext &ctx, float coord_w, float coord_h) override;
  void BeginDrawBatch(const rex::ui::ImmediateDrawBatch &batch) override;
  void Draw(const rex::ui::ImmediateDraw &draw) override;
  void EndDrawBatch() override;
  void End() override;

private:
  // Lazy: the drawer is created before the host device exists.
  bool TryInitDeviceResources();
  // Records into the open Present command list.
  std::unique_ptr<PlumeImmediateTexture> UploadRGBA8Texture(uint32_t w, uint32_t h,
                                                            const uint8_t *rgba);

  bool resources_ready_ = false;
  bool batch_open_ = false;
  plume::RenderCommandList *cmd_ = nullptr; // valid between Begin/End
  std::unique_ptr<plume::RenderPipelineLayout> layout_;
  std::unique_ptr<plume::RenderPipeline> pipeline_;
  std::unique_ptr<plume::RenderShader> vs_;
  std::unique_ptr<plume::RenderShader> ps_;
  std::unique_ptr<plume::RenderSampler> sampler_;
  std::unique_ptr<plume::RenderDescriptorSet> sampler_set_;
  std::unique_ptr<PlumeImmediateTexture> white_;
};

} // namespace pgr4::gpu
