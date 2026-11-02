// ImGui overlay vertex shader: pixel space -> NDC via a per-frame ortho root
// constant at b0, space2 (re:Blue's interface). Shader model 5.1 (fxc), see
// imgui_ps.hlsl.
struct VS_IN {
  float2 pos : POSITION;
  float2 uv  : TEXCOORD0;
  float4 col : COLOR0; // IA decodes R8G8B8A8_UNORM (IM_COL32 order)
};

struct VS_OUT {
  float4 pos : SV_Position;
  float4 col : COLOR0;
  float2 uv  : TEXCOORD0;
};

cbuffer Ortho : register(b0, space2) {
  float2 uScale;
  float2 uTranslate;
};

VS_OUT main(VS_IN i) {
  VS_OUT o;
  o.pos = float4(i.pos * uScale + uTranslate, 0.0, 1.0);
  o.col = i.col;
  o.uv  = i.uv;
  return o;
}
