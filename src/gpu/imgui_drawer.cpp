// PGR4-Recomp native renderer: ImGui overlay drawer on Plume/D3D12.
// Port of re:Blue's gpu/imgui_overlay_drawer.cpp, changed to run on every
// D3D12 GPU: shader model 5.1 DXBC shaders, and one-entry descriptor sets per
// texture instead of the shared bindless heap.
#include "gpu/imgui_drawer.h"

#include <cstring>

#include <plume_d3d12.h>
#include <plume_render_interface.h>
#include <plume_render_interface_builders.h>

#include <rex/logging.h>

#include "gpu/video.h"

#include "imgui_ps.dxbc.h"
#include "imgui_vs.dxbc.h"

namespace pgr4::gpu {
namespace {

const plume::RenderInputSlot kImGuiInputSlot(0, sizeof(rex::ui::ImmediateVertex),
                                             plume::RenderInputSlotClassification::PER_VERTEX_DATA);

void BuildTextureSetLayout(plume::RenderDescriptorSetBuilder &b) {
  b.begin();
  b.addTexture(0, 1);
  b.end();
}

void BuildSamplerSetLayout(plume::RenderDescriptorSetBuilder &b) {
  b.begin();
  b.addSampler(0, 1);
  b.end();
}

} // namespace

PlumeImmediateTexture::PlumeImmediateTexture(uint32_t w, uint32_t h,
                                             std::unique_ptr<plume::RenderTexture> texture,
                                             std::unique_ptr<plume::RenderTextureView> view,
                                             std::unique_ptr<plume::RenderDescriptorSet> set)
    : rex::ui::ImmediateTexture(w, h), texture_(std::move(texture)), view_(std::move(view)),
      set_(std::move(set)) {}

PlumeImmediateTexture::~PlumeImmediateTexture() {
  // An in-flight frame may still reference these.
  Video::ParkUntilFence(std::move(set_));
  Video::ParkUntilFence(std::move(view_));
  Video::ParkUntilFence(std::move(texture_));
}

ImGuiOverlayDrawer::ImGuiOverlayDrawer() = default;
ImGuiOverlayDrawer::~ImGuiOverlayDrawer() = default;

std::unique_ptr<PlumeImmediateTexture>
ImGuiOverlayDrawer::UploadRGBA8Texture(uint32_t w, uint32_t h, const uint8_t *rgba) {
  plume::RenderDevice *device = Video::HostDevice();
  plume::RenderCommandList *cmd = Video::CommandList();
  if (!device || !cmd) {
    return nullptr;
  }

  plume::RenderTextureDesc desc;
  desc.dimension = plume::RenderTextureDimension::TEXTURE_2D;
  desc.width = w;
  desc.height = h;
  desc.depth = 1;
  desc.mipLevels = 1;
  desc.arraySize = 1;
  desc.format = plume::RenderFormat::R8G8B8A8_UNORM;
  desc.flags = plume::RenderTextureFlag::NONE;
  desc.multisampling.sampleCount = plume::RenderSampleCount::COUNT_1;
  auto tex = device->createTexture(desc);
  if (!tex) {
    REXGPU_WARN("ImGui overlay: createTexture failed ({}x{})", w, h);
    return nullptr;
  }
  plume::RenderTextureViewDesc view_desc;
  view_desc.format = plume::RenderFormat::R8G8B8A8_UNORM;
  view_desc.dimension = plume::RenderTextureViewDimension::TEXTURE_2D;
  view_desc.mipLevels = 1;
  auto view = tex->createTextureView(view_desc);
  if (!view) {
    REXGPU_WARN("ImGui overlay: createTextureView failed ({}x{})", w, h);
    return nullptr;
  }

  // D3D12 placed footprints need 256-byte row pitch, so pad short rows.
  constexpr uint32_t kRowPitchAlign = 256;
  const uint32_t bytes_per_row = w * 4;
  const uint32_t row_pitch = (bytes_per_row + kRowPitchAlign - 1) & ~(kRowPitchAlign - 1);
  UploadAllocation up = Video::Upload(nullptr, row_pitch * h, 0x200);
  if (!up) {
    REXGPU_WARN("ImGui overlay: pixel upload allocation failed ({}x{})", w, h);
    return nullptr;
  }
  for (uint32_t y = 0; y < h; ++y) {
    std::memcpy(static_cast<uint8_t *>(up.memory) + size_t(y) * row_pitch,
                rgba + size_t(y) * bytes_per_row, bytes_per_row);
  }

  plume::RenderDescriptorSetBuilder set_b;
  BuildTextureSetLayout(set_b);
  auto set = set_b.create(device);
  if (!set) {
    REXGPU_WARN("ImGui overlay: texture descriptor set failed");
    return nullptr;
  }
  set->setTexture(0, tex.get(), plume::RenderTextureLayout::SHADER_READ, view.get());

  plume::RenderTextureBarrier pre(tex.get(), plume::RenderTextureLayout::COPY_DEST);
  cmd->barriers(plume::RenderBarrierStage::COPY, &pre, 1);
  // PlacedFootprint row width is in texels.
  cmd->copyTextureRegion(plume::RenderTextureCopyLocation::Subresource(tex.get(), 0),
                         plume::RenderTextureCopyLocation::PlacedFootprint(
                             up.buffer, plume::RenderFormat::R8G8B8A8_UNORM, w, h, 1,
                             row_pitch / 4, up.offset));
  plume::RenderTextureBarrier post(tex.get(), plume::RenderTextureLayout::SHADER_READ);
  cmd->barriers(plume::RenderBarrierStage::GRAPHICS, &post, 1);

  return std::make_unique<PlumeImmediateTexture>(w, h, std::move(tex), std::move(view),
                                                 std::move(set));
}

bool ImGuiOverlayDrawer::TryInitDeviceResources() {
  if (resources_ready_) {
    return true;
  }
  plume::RenderDevice *device = Video::HostDevice();
  if (!device || !Video::CommandList()) {
    return false;
  }

  vs_ = device->createShader(g_imgui_vs_dxbc, sizeof(g_imgui_vs_dxbc), "main",
                             plume::RenderShaderFormat::DXBC);
  ps_ = device->createShader(g_imgui_ps_dxbc, sizeof(g_imgui_ps_dxbc), "main",
                             plume::RenderShaderFormat::DXBC);
  if (!vs_ || !ps_) {
    REXGPU_ERROR("ImGui overlay: createShader failed");
    return false;
  }

  plume::RenderDescriptorSetBuilder tex_b;
  BuildTextureSetLayout(tex_b);
  plume::RenderDescriptorSetBuilder samp_b;
  BuildSamplerSetLayout(samp_b);

  plume::RenderPipelineLayoutBuilder lb;
  lb.begin(false, true);
  lb.addDescriptorSet(tex_b);  // t0, space0
  lb.addDescriptorSet(samp_b); // s0, space1
  lb.addPushConstant(0, 2, sizeof(float) * 4, plume::RenderShaderStageFlag::VERTEX); // Ortho b0,space2
  lb.end();
  layout_ = lb.create(device);
  if (!layout_) {
    REXGPU_ERROR("ImGui overlay: pipeline layout create failed");
    return false;
  }

  plume::RenderGraphicsPipelineDesc pd;
  pd.pipelineLayout = layout_.get();
  pd.vertexShader = vs_.get();
  pd.pixelShader = ps_.get();
  pd.depthEnabled = false;
  pd.depthWriteEnabled = false;
  pd.primitiveTopology = plume::RenderPrimitiveTopology::TRIANGLE_LIST;
  pd.cullMode = plume::RenderCullMode::NONE;
  pd.renderTargetCount = 1;
  pd.renderTargetFormat[0] = plume::RenderFormat::B8G8R8A8_UNORM;
  pd.renderTargetBlend[0] = plume::RenderBlendDesc::AlphaBlend();
  pd.depthTargetFormat = plume::RenderFormat::UNKNOWN;
  // pos.f2@0, uv.f2@8, color RGBA8@16, stride 20.
  const plume::RenderInputElement elems[] = {
      plume::RenderInputElement("POSITION", 0, 0, plume::RenderFormat::R32G32_FLOAT, 0, 0),
      plume::RenderInputElement("TEXCOORD", 0, 1, plume::RenderFormat::R32G32_FLOAT, 0, 8),
      plume::RenderInputElement("COLOR", 0, 2, plume::RenderFormat::R8G8B8A8_UNORM, 0, 16),
  };
  pd.inputSlots = &kImGuiInputSlot;
  pd.inputSlotsCount = 1;
  pd.inputElements = elems;
  pd.inputElementsCount = 3;
  pipeline_ = device->createGraphicsPipeline(pd);
  // plume hands back a wrapper even when the D3D12 PSO failed; binding that
  // wrapper crashes inside setPipeline.
  if (!pipeline_ || !static_cast<plume::D3D12GraphicsPipeline *>(pipeline_.get())->d3d) {
    REXGPU_ERROR("ImGui overlay: graphics pipeline creation failed; overlay disabled");
    pipeline_.reset();
    return false;
  }

  plume::RenderSamplerDesc sdesc;
  sdesc.minFilter = plume::RenderFilter::LINEAR;
  sdesc.magFilter = plume::RenderFilter::LINEAR;
  sdesc.mipmapMode = plume::RenderMipmapMode::LINEAR;
  sdesc.addressU = sdesc.addressV = sdesc.addressW = plume::RenderTextureAddressMode::CLAMP;
  sampler_ = device->createSampler(sdesc);
  sampler_set_ = samp_b.create(device);
  if (!sampler_ || !sampler_set_) {
    REXGPU_ERROR("ImGui overlay: sampler creation failed");
    return false;
  }
  sampler_set_->setSampler(0, sampler_.get());

  // 1x1 white for untextured (solid colour) draws.
  static const uint8_t kWhite[4] = {0xFF, 0xFF, 0xFF, 0xFF};
  white_ = UploadRGBA8Texture(1, 1, kWhite);
  if (!white_) {
    REXGPU_ERROR("ImGui overlay: white texture upload failed");
    return false;
  }
  REXGPU_INFO("ImGui overlay ready (shader model 5.1)");
  resources_ready_ = true;
  return true;
}

std::unique_ptr<rex::ui::ImmediateTexture>
ImGuiOverlayDrawer::CreateTexture(uint32_t width, uint32_t height,
                                  rex::ui::ImmediateTextureFilter /*filter*/,
                                  bool /*is_repeated*/, const uint8_t *data) {
  // SDK contract: nullptr until the device exists; ImGuiDrawer retries the
  // font atlas upload on the next Draw.
  if (!TryInitDeviceResources() || !data || !width || !height) {
    return nullptr;
  }
  return UploadRGBA8Texture(width, height, data);
}

void ImGuiOverlayDrawer::Begin(rex::ui::UIDrawContext &ctx, float coord_w, float coord_h) {
  rex::ui::ImmediateDrawer::Begin(ctx, coord_w, coord_h);
  batch_open_ = false;
  cmd_ = nullptr;
  if (!TryInitDeviceResources()) {
    return;
  }
  auto &octx = static_cast<OverlayDrawContext &>(ctx);
  cmd_ = octx.command_list();
  if (!cmd_) {
    return;
  }

  cmd_->setFramebuffer(octx.framebuffer());
  plume::RenderViewport vp(0.0f, 0.0f, float(ctx.render_target_width()),
                           float(ctx.render_target_height()), 0.0f, 1.0f);
  cmd_->setViewports(&vp, 1);
  // Layout first: descriptor sets bind against the current layout.
  cmd_->setGraphicsPipelineLayout(layout_.get());
  cmd_->setPipeline(pipeline_.get());
  cmd_->setGraphicsDescriptorSet(sampler_set_.get(), 1);

  struct Ortho {
    float scale[2];
    float translate[2];
  } ortho;
  ortho.scale[0] = 2.0f / coordinate_space_width();
  ortho.scale[1] = -2.0f / coordinate_space_height();
  ortho.translate[0] = -1.0f;
  ortho.translate[1] = 1.0f;
  cmd_->setGraphicsPushConstants(0, &ortho, 0, sizeof(ortho));
}

void ImGuiOverlayDrawer::BeginDrawBatch(const rex::ui::ImmediateDrawBatch &b) {
  batch_open_ = false;
  if (!cmd_ || b.vertex_count <= 0) {
    return;
  }
  const uint32_t vbytes = uint32_t(sizeof(rex::ui::ImmediateVertex)) * uint32_t(b.vertex_count);
  UploadAllocation va = Video::Upload(b.vertices, vbytes, 16);
  if (!va) {
    return;
  }
  const plume::RenderVertexBufferView vbv(va.buffer->at(va.offset), va.size);
  cmd_->setVertexBuffers(0, &vbv, 1, &kImGuiInputSlot);

  if (b.indices && b.index_count > 0) {
    const uint32_t ibytes = uint32_t(sizeof(uint16_t)) * uint32_t(b.index_count);
    UploadAllocation ia = Video::Upload(b.indices, ibytes, 4);
    if (!ia) {
      return;
    }
    const plume::RenderIndexBufferView ibv(ia.buffer->at(ia.offset), ia.size,
                                           plume::RenderFormat::R16_UINT);
    cmd_->setIndexBuffer(&ibv);
  }
  batch_open_ = true;
}

void ImGuiOverlayDrawer::Draw(const rex::ui::ImmediateDraw &draw) {
  if (!cmd_ || !batch_open_ || draw.count <= 0) {
    return;
  }
  uint32_t l, t, w, h;
  if (!ScissorToRenderTarget(draw, l, t, w, h)) {
    return;
  }
  plume::RenderRect rc(int32_t(l), int32_t(t), int32_t(l + w), int32_t(t + h));
  cmd_->setScissors(&rc, 1);

  auto *tex = draw.texture ? static_cast<PlumeImmediateTexture *>(draw.texture) : white_.get();
  cmd_->setGraphicsDescriptorSet(tex->descriptor_set(), 0);
  cmd_->drawIndexedInstanced(uint32_t(draw.count), 1, uint32_t(draw.index_offset),
                             draw.base_vertex, 0);
}

void ImGuiOverlayDrawer::EndDrawBatch() { batch_open_ = false; }

void ImGuiOverlayDrawer::End() {
  cmd_ = nullptr;
  rex::ui::ImmediateDrawer::End();
}

} // namespace pgr4::gpu
