# riceharvest/llama.cpp: Arc Pro B70 branches (Vulkan and SYCL)

This branch only holds documentation. The code is in the branches below. Each one is based on llama.cpp master
`a0e6dc738` (2026-10-14) and was measured there on one **Intel Arc Pro B70** (32 GB, Fedora 43, kernel 7.1.8,
**Mesa 25.3.6**). None of them is upstream. Commit messages marked DRAFT are placeholders, not upstream text.

| branch | head | what it does | measured on the B70 | doc |
|---|---|---|---|---|
| [`vk-b70-stack`](https://github.com/riceharvest/llama.cpp/tree/vk-b70-stack) | `54b6e8485` | all fixes below in one branch; what [b70-llama-builds](https://github.com/riceharvest/b70-llama-builds) ships | 27B decode 13.2 → 22.7 t/s, 35B-A3B 39 → 64 t/s | [stack](branches/vk-b70-stack.md) |
| [`vk-mmvq-u32-loads`](https://github.com/riceharvest/llama.cpp/tree/vk-mmvq-u32-loads) | `3db0e960c` | MMVQ reads Q3_K/Q6_K/Q8_0/Q5_0/MXFP4 with 32-bit loads; Xe2 mat-vec workgroup rule | 27B decode 13.2 → 22.7 t/s | [mat-vec loads](branches/vk-mmvq-u32-loads.md) |
| [`vk-xe-fa-decode-split`](https://github.com/riceharvest/llama.cpp/tree/vk-xe-fa-decode-split) | `41ed2108a` | Intel `xe_fa` decode kernel: phase 2 split over the KV range | decode after 32K +13% | [FA split](branches/vk-xe-fa-decode-split.md) |
| [`vk-xe2-matmul-splitk`](https://github.com/riceharvest/llama.cpp/tree/vk-xe2-matmul-splitk) | `4fc6ddd00` | Xe2 split-k rule for few-tile, long-k matmuls | pp512 +3.8% (27B), +11.5% (1.7B) | [split-k](branches/vk-xe2-matmul-splitk.md) |
| [`vk-fa-cm1-m8`](https://github.com/riceharvest/llama.cpp/tree/vk-fa-cm1-m8) | `bc0999fac` | 8x16x16 coopmat1 flash-attention shader for devices without 16x16x16 | 27B pp512 at 8K +47% vs scalar FA | [FA m8](branches/vk-fa-cm1-m8.md) |
| [`vk-matvec-glu-fusion`](https://github.com/riceharvest/llama.cpp/tree/vk-matvec-glu-fusion) | `ca985d7ae` | mat-vec epilogue computes SWIGLU/GEGLU (on top of the stack) | **parked, never run on a GPU** | [GLU fusion](branches/vk-matvec-glu-fusion.md) |
| [`sycl-qwen35moe-fusions`](https://github.com/riceharvest/llama.cpp/tree/sycl-qwen35moe-fusions) | `1c7351082` | **SYCL**: port of the b70-optimization-lab qwen35moe decode fusions (10 env flags, default off) | being tested (lab: 131.5 tok/s on Ornith 1.5) | [qwen35moe fusions](branches/sycl-qwen35moe-fusions.md) |
| [`sycl-mmid-small-batch`](https://github.com/riceharvest/llama.cpp/tree/sycl-mmid-small-batch) | `4f29be7b3` | **SYCL**: fused MUL_MAT_ID GEMV for 2-8 tokens (speculative-decoding verify) | being tested | [MoE verify](branches/sycl-mmid-small-batch.md) |
| [`sycl-fa-xe2-occupancy`](https://github.com/riceharvest/llama.cpp/tree/sycl-fa-xe2-occupancy) | `f116c0ffa` | **SYCL**: flash-attention KV split sized from Xe2 occupancy (decode at long context); last commit = test knobs | not run yet | [FA occupancy](branches/sycl-fa-xe2-occupancy.md) |
| [`vk-fa-cm1-m8-testknobs`](https://github.com/riceharvest/llama.cpp/tree/vk-fa-cm1-m8-testknobs) | `499668eaa` | the same work with environment knobs for A/B tests; never for upstream | (test branch) | [test knobs](branches/vk-fa-cm1-m8-testknobs.md) |

Old heads are kept as tags: `archive/vk-mmvq-u32-loads-5bf369db4` (had the results file in the tree),
`archive/vk-fa-cm1-m8-17114e63a` (same tree, old commit message).

## Build and compare any branch

```bash
git clone https://github.com/riceharvest/llama.cpp && cd llama.cpp
git checkout <branch>
cmake -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 8 --target llama-bench test-backend-ops
# baseline: the same commands on a0e6dc738 in another build dir
git worktree add ../llama.cpp-master a0e6dc738
```

When comparing two builds, run each arm's `llama-bench` from its own `build/bin`. Copied binaries keep their
build-tree RUNPATH and can load the other arm's `libggml-vulkan.so`; check with `ldd`.

Benchmark commands used for every table (`-fa 1`, `-sm none`, one GPU, `GGML_VK_VISIBLE_DEVICES=0`):

```bash
./build/bin/llama-bench -m <model.gguf> -ngl 99 -sm none -fa 1 -p 512 -n 128 -d 0,8192,32768 -r 3
./build/bin/test-backend-ops -o MUL_MAT -b Vulkan0          # correctness vs CPU
./build/bin/test-backend-ops -o FLASH_ATTN_EXT -b Vulkan0
```

Models: Qwen3.8-27B UD-Q4_K_XL (unsloth), Qwen3.6-35B-A3B UD-Q4_K_M (unsloth), a 1.7B dense Q8_0.

## Caveats

- One card, one driver. All numbers are on Mesa 25.3.6. Mesa 26.x changes Vulkan speed on Xe2 a lot, so re-measure
  there before drawing conclusions. Other Xe2 devices (B580, B60, Lunar Lake) and Windows are untested.
- llama.cpp does not accept agent-written PRs. These branches are for people to test and, if they hold up, for a
  human to write up upstream.
