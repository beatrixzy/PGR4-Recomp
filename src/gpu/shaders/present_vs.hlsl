// Front buffer blit: one full-screen triangle from SV_VertexID. Shader model
// 5.1 (fxc), like the ImGui overlay shaders.
struct VS_OUT {
  float4 pos : SV_Position;
  float2 uv  : TEXCOORD0;
};

VS_OUT main(uint id : SV_VertexID) {
  VS_OUT o;
  const float2 uv = float2((id << 1) & 2, id & 2);
  o.pos = float4(uv * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
  o.uv = uv;
  return o;
}
