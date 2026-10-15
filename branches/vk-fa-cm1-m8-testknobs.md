# vk-fa-cm1-m8-testknobs: the same work with A/B knobs (test only)

Branch [`vk-fa-cm1-m8-testknobs`](https://github.com/riceharvest/llama.cpp/tree/vk-fa-cm1-m8-testknobs) at `499668eaa`.
Never for upstream: it adds environment knobs used to measure each change on the B70. It starts from the original
FA m8 commit (`archive/vk-fa-cm1-m8-17114e63a`, same tree as `vk-fa-cm1-m8`).

Knobs (unset = the same behaviour as the clean branches):

| variable | effect |
|---|---|
| `GGML_VK_FA_CM1_FORCE_M16` | use the 16x16x16 FA shader even where the driver only advertises 8x16x16 (out of spec, fails tests on Xe2) |
| `GGML_VK_FA_CM1_SG` | FA coopmat1 subgroup size override |
| `GGML_VK_DMMV_WG`, `GGML_VK_DMMV_WG_ID`, `GGML_VK_DMMV_WG_MAX_M` | force the mat-vec workgroup size (`large` / `subgroup`) |
| `GGML_VK_XE2_Q8_SMALLK_DMMV` | send small-k Q8_0 back to DMMV |
| `GGML_VK_MM_TILE`, `GGML_VK_MM_TILE_MAX_M`, `GGML_VK_MM_SPLITK` | force matmul tile size and split-k |
| `GGML_VK_XE_FA_PH2_SPLIT`, `GGML_VK_XE_FA_PCHUNK`, `GGML_VK_XE_FA_VDIM`, `GGML_VK_XE_FA_SKIP` | xe_fa split count, chunk size, and skipping a phase (timing only, wrong output) |
| `GGML_VK_FA_SPLITK_MUL` | FA split-k multiplier experiment |
| `GGML_VK_BARRIER_COMPUTE_ONLY`, `GGML_VK_NO_BARRIER_TIMING` | barrier experiments (the second gives wrong output; timing bound only) |
