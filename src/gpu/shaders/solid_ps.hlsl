// Debug (--debug_solid_ps): stands in for every guest pixel shader, so a
// draw that rasterises shows as solid magenta whatever its shader computes.
float4 main() : SV_Target0 { return float4(1.0, 0.0, 1.0, 1.0); }
