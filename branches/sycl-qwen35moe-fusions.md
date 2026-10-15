# sycl-qwen35moe-fusions: qwen35moe decode fusions for SYCL (default off)

Branch [`sycl-qwen35moe-fusions`](https://github.com/riceharvest/llama.cpp/tree/sycl-qwen35moe-fusions) at `1c7351082`,
one commit on llama.cpp master `a0e6dc738`.

A port of the public-domain "twelve-feature" SYCL patch from
[steveseguin/b70-optimization-lab](https://github.com/steveseguin/b70-optimization-lab/tree/main/repro/ornith-15-35b-a3b-q4km-b70)
(written for llama.cpp `9fee29e9435f`, measured there at 131.5 tok/s single-stream decode on one B70 with Ornith 1.5
35B-A3B Q4_K_M). Ornith 1.5 uses the `qwen35moe` architecture with the same shapes as Qwen3.6-35B-A3B, so the matchers apply
to the stock model too.

**Clean version:** [`sycl-qwen35moe-fusions-v2`](https://github.com/riceharvest/llama.cpp/tree/sycl-qwen35moe-fusions-v2)
(`7581d6d99`, two commits: the fusions, then test-backend-ops cases). Same code with `ornith` renamed to `qwen35moe`
(flags `GGML_SYCL_FUSED_QWEN35MOE_*`, e.g. `GGML_SYCL_FUSED_ORNITH_MOE_GATE_UP` → `GGML_SYCL_FUSED_QWEN35MOE_MOE_GATE_UP`,
`GGML_SYCL_FUSED_MOE_ADD_REDUCE` → `GGML_SYCL_FUSED_QWEN35MOE_MOE_ADD_REDUCE`) and the fault-injection `*_POISON` knobs
removed. The test cases (`test-backend-ops -o QWEN35MOE_...`) rebuild each fusion's one-token sub-graph with the names
and shapes its matcher expects; run them with the flags set. The table below uses the first branch's names.

Each fusion has its own environment flag and is off by default. Its matcher checks exact tensor names, shapes, strides and
consumers, and falls back to the normal path otherwise. Each prints `fused_hits=N` at exit when enabled:

| flag | fuses |
|---|---|
| `GGML_SYCL_FUSED_MOE_ADD_REDUCE` | the ordered sum of the routed expert outputs |
| `GGML_SYCL_FUSED_RESIDUAL_RMS_NORM` | residual ADD + RMS_NORM + weight MUL |
| `GGML_SYCL_FUSED_ORNITH_CONCAT_STATE`, `..._CONCAT_STATE_DIRECT` | conv-state gather, concat and update |
| `GGML_SYCL_FUSED_ORNITH_MOE_GATE_UP` | routed gate + up Q4_K GEMV + SWIGLU in one kernel |
| `GGML_SYCL_FUSED_ORNITH_MOE_SHARED_RESIDUAL_RMS`, `..._MOE_GATE_RESIDUAL_RMS` | shared-expert gate, routed + shared + residual ADD, RMS_NORM, MUL |
| `GGML_SYCL_FUSED_ORNITH_GDN_RMS_GATE` | gated RMSNorm of the GDN output |
| `GGML_SYCL_FUSED_ORNITH_GDN_STATE_IO` | GDN reads its recurrent state in place (single snapshot slot) |
| `GGML_SYCL_FUSED_ORNITH_QK_NORM_ROPE` | q/k RMSNorm + scale + IMRoPE (+ K cache write) |

Two of the lab's twelve are not ported because master already fuses them unconditionally: SSM_CONV + SILU and
ADD + SOFTPLUS + MUL.

The lab's runtime settings for its result: `UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 GGML_SYCL_ENABLE_GRAPH=0`.

Status: compile-checked; B70 measurement pending (correctness vs master greedy output, tg128). Build like any SYCL build
(`-DGGML_SYCL=ON -DGGML_SYCL_F16=ON -DGGML_SYCL_DEVICE_ARCH=bmg-g31-a0`), then set all flags and run `llama-bench`.

## Result on the B70 (testq 006, 2026-10-15)

Qwen3.6-35B-A3B UD-Q4_K_M, tg128, r=2, all arms back to back:

| arm | tok/s |
|---|---|
| master | 93.39 ± 0.19 |
| master + lab runtime settings (`UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 GGML_SYCL_ENABLE_GRAPH=0`) | 93.48 ± 0.10 |
| branch, flags off | 93.36 ± 0.05 |
| **branch, all flags on** | **107.92 ± 1.47 (+15.5%)** |

Greedy 128 tokens on 2 prompts, master vs branch with all flags on: identical. Hits: moe_gate_up, moe_gate_residual_rms,
residual_rms_norm 10280 each; concat_state, gdn_rms_gate 7710; qk_norm_rope 2570. Zero hits: moe_shared_residual_rms (by
design: moe_gate_residual_rms covers the same nodes when the shared-expert gate is a sigmoid), concat_state_direct and
gdn_state_io (their matchers do not fit master's graph; under investigation). `GGML_SYCL_ENABLE_GRAPH` has no effect here:
it is off by default and graphs are disabled for any graph with MUL_MAT_ID.

In the combined `sycl-b70-stack` (with the MoE small-batch and FA changes; testq 012, llama-server, 3 prompts x 256 tokens)
fusions on vs off are **not** identical: the outputs part at characters 340 / 619 / 476. The fused kernels reorder
floating-point sums; quality was not measured (a perplexity pair would settle it). Speed there: 79.8 → 87.5 tok/s (+9.6%).
