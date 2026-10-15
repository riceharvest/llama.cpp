# vk-xe-fa-decode-split: split phase 2 of Intel's xe_fa decode kernel over the KV range

Branch [`vk-xe-fa-decode-split`](https://github.com/riceharvest/llama.cpp/tree/vk-xe-fa-decode-split) at `41ed2108a`,
one commit on llama.cpp master `a0e6dc738`.

## What changes

Decode on Intel uses the two-phase `xe_fa` flash-attention kernel. Phase 2 (P·V) ran one workgroup per head, so after
long context it used a small part of the GPU. The branch splits phase 2 over the KV range. Each split writes
unnormalized O plus L (row sum) and M (the global max from phase 1) in the existing split-k buffer layout, and the
existing `fa_split_k_reduce` merges them, including attention sinks. The split count targets 16 x core-count
workgroups with at least 4 chunks of 256 per split. A split count of 1 is the old code path. The unused
`qk_sub_groups` push constant carries the split count.

## Results (Qwen3.8-27B UD-Q4_K_XL, tg128 tok/s)

| context | before (split 1) | split | change |
|---|---|---|---|
| 32768 | 16.64 | **18.79** | +13% |
| 8192 | 20.83 | 21.37 | +2.6% |
| 0 | 22.52 | 22.48 | same |

Measured on top of the mat-vec fixes (`vk-mmvq-u32-loads`), where attention is a bigger share of decode time.
Alone on master: 11.23 → 11.95 at 32K (+6%). Flash attention per token at 32K went from ~14 to ~8.8 ms.

Correctness: `test-backend-ops -o FLASH_ATTN_EXT -b Vulkan0` 5321/5321 at split 1, at the default split and at 7.
A control build with phase 2 skipped fails 212 cases, so the tests do reach this kernel.

## Reproduce

```bash
git checkout vk-xe-fa-decode-split && cmake --build build --target llama-bench test-backend-ops
./build/bin/llama-bench -m Qwen3.8-27B-UD-Q4_K_XL.gguf -ngl 99 -sm none -fa 1 -p 0 -n 128 -d 0,8192,32768 -r 3
./build/bin/test-backend-ops -o FLASH_ATTN_EXT -b Vulkan0
```

Next target: phase 1 (Q·K, ~5 ms per token at 32K, K read at ~200 GB/s).
