# sycl-fa-xe2-occupancy: size the SYCL flash-attention KV split from Xe2 occupancy

Branch [`sycl-fa-xe2-occupancy`](https://github.com/riceharvest/llama.cpp/tree/sycl-fa-xe2-occupancy), on llama.cpp master
`a0e6dc738`: `01b80b29a` (the change, `fattn-common.hpp` +26/-1) and `f116c0ffa` (environment knobs for testing; not for
upstream). **Compile-checked only; nothing measured yet.**

## What changes

SYCL decode attention (the tile kernel, and the vec kernel) splits the KV range over `parallel_blocks` work-groups and
merges them with `flash_attn_combine_results`. The split is chosen to fill one wave of `nsm * max_wg_per_cu`
work-groups. Both numbers come from the CUDA port:

- `nsm = max_compute_units / 16` counts 16 EUs per core. An Xe2 core has 8 XVEs, so a 32-core Arc Pro B70 reads as 16
  (upstream's `topk-radix.cpp` already notes this and doubles it for its own split).
- `max_wg_per_cu = max_work_group_size / max_compute_units` = 1024 / 256 = 4. It is not an occupancy measure.

So the wave is 64 work-groups on the B70. At 32K context that gives 64 work-groups of 8 sub-groups for Qwen3.6-35B-A3B
(GQA 8) and 192 work-groups of 2 sub-groups for Qwen3.8-27B (GQA 6), against 32 cores x 8 XVEs x 8 threads = 2048
hardware threads.

On Xe2 (BMG-G21, BMG-G31, LNL) the branch uses `compute units / 8` cores and `64 / nwarps` work-groups per core (32 with
the large register file, fewer if the launch's dynamic local memory runs out) for the wave size. The search for the
split still starts from the old value, so launches that already have many tiles (prefill) mostly keep their split.
Other GPUs are unchanged.

Expected split on the B70 at 32K (from the launch arithmetic): 35B-A3B 32 → 128 blocks, 27B 16 → 85 blocks.

## Test knobs (`f116c0ffa`)

| variable | effect |
|---|---|
| `GGML_SYCL_FA_XE2_OCC=0` | old wave size (A/B in one build) |
| `GGML_SYCL_FA_PARALLEL_BLOCKS=N` | force the split (clamped to the number of KV tiles) |
| `GGML_SYCL_FA_LOG_SPLIT=1` | log the first 64 split decisions |
| `GGML_SYCL_FA_TILE_GQA_OPT=0` | tile kernel with one Q head per KV read (ncols2=1) |

## Reproduce

```bash
git checkout sycl-fa-xe2-occupancy
cmake -B build-sycl -DGGML_SYCL=ON -DGGML_SYCL_F16=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx
cmake --build build-sycl --target llama-bench test-backend-ops -j 8
./build-sycl/bin/test-backend-ops -o FLASH_ATTN_EXT -b SYCL0 -p hsk=256
for occ in 0 1; do GGML_SYCL_FA_XE2_OCC=$occ ./build-sycl/bin/llama-bench -m <model> -ngl 99 -sm none -fa 1 -p 0 -n 64 -d 32768 -r 1; done
```
