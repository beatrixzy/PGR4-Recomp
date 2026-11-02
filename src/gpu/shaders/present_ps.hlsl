// Front buffer blit: samples the guest front buffer's host copy, then maps
// each 8-bit channel through the guest's gamma ramp (256x1, identity when
// the game set none), as the 360's display controller does at scan-out.
Texture2D    gSource  : register(t0, space0);
Texture2D    gGamma   : register(t1, space0);
SamplerState gSampler : register(s0, space1);

float4 main(float4 pos : SV_Position, float2 uv : TEXCOORD0) : SV_Target {
  const float3 color = gSource.SampleLevel(gSampler, uv, 0.0).rgb;
  const int3 index = int3(round(saturate(color) * 255.0));
  return float4(gGamma.Load(int3(index.r, 0, 0)).r, gGamma.Load(int3(index.g, 0, 0)).g,
                gGamma.Load(int3(index.b, 0, 0)).b, 1.0);
}
