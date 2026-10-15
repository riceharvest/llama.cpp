# sycl-mmid-small-batch: fused MUL_MAT_ID GEMV for 2-8 tokens

Branch [`sycl-mmid-small-batch`](https://github.com/riceharvest/llama.cpp/tree/sycl-mmid-small-batch) at `4f29be7b3`,
one commit on llama.cpp master `a0e6dc738`.

With MTP speculative decoding on a B70 (Qwen3.6-35B-A3B), llama.cpp SYCL ran slower than without it (69-81 vs 86.7 tok/s)
and its greedy output differed from no-spec, while Vulkan gained +53% with identical output. One cause found by reading the
code: for a 2-8 token verify batch, SYCL's MUL_MAT_ID left the fused per-token MMVQ path and fell back to copying the expert
ids to the host, waiting on the stream, sorting rows by expert and running a grouped GEMM with f16 activations.

This branch keeps batches of up to 8 tokens on the fused path: all rows are quantized once and the existing GEMV runs once
per token. That removes the host sync from every MoE projection, and each token gets the same arithmetic as single-token
decode. Dense mat-muls still differ between 1 token (ESIMD DMMV) and 2-8 tokens (MMVQ); `GGML_SYCL_ENABLE_ESIMD=0` makes the
1-token path use MMVQ as well.

Status: measured on the B70 (testq 007, 012); see the results below.
Evidence: the divergence analysis in the 001 test notes (b70-code, 2026-10-15).

## Result on the B70 (testq 007, 2026-10-15)

Qwen3.6-35B-A3B-MTP UD-Q4_K_M, llama-server, greedy, 3 prompts x 256 tokens:

| arm | tok/s (mean of 3 prompts) |
|---|---|
| master, no MTP (two processes) | 88.5 / 85.6 |
| branch, no MTP | 83.9 |
| **branch, MTP n=3** | **131.8** (master MTP n=3 in testq 001: 80.9) |
| branch, MTP n=3, `GGML_SYCL_ENABLE_ESIMD=0` | 121.7 |

Follow-up (testq 010 / 012):
- test-backend-ops `-o MUL_MAT_ID` aborts after 34 cases on **master too** (oneDNN "found misaligned buffer"; the next
  case's per-expert output slices are 32-byte aligned and oneDNN wants 64). Not caused by this branch; fix on
  `sycl-dnnl-align`.
- Output vs master without MTP, context checkpoints off: identical on prose and reasoning, still different on code
  (char 241). So the checkpoint batch explains most of the difference, not all.
- MTP output vs no-MTP output still differs (2 of 3 prompts): dense mat-muls take different paths for 1 and 2-8 tokens.
- In the combined `sycl-b70-stack` (with the qwen35moe fusions), MTP n=3 is +39% over fusions alone (121.7 vs 87.5 tok/s in
  one session that ran ~8% below 007's).
