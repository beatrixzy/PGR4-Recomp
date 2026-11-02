// ImGui overlay pixel shader. Shader model 5.1 (fxc) so it runs on every
// D3D12 GPU, including SM 5.1-only drivers (e.g. GT 710 on 391.35) that
// can't create DXIL pipelines or survive large bindless tables. Each ImGui
// texture has its own one-entry descriptor set bound per draw.
Texture2D    gTexture : register(t0, space0);
SamplerState gSampler : register(s0, space1);

struct PS_IN {
  float4 pos : SV_Position;
  float4 col : COLOR0;
  float2 uv  : TEXCOORD0;
};

float4 main(PS_IN i) : SV_Target {
  // ImGui textures are single-mip.
  return gTexture.SampleLevel(gSampler, i.uv, 0.0) * i.col;
}
