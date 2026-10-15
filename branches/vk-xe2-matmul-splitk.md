# vk-xe2-matmul-splitk: split-k for few-tile, long-k matmuls on Xe2

Branch [`vk-xe2-matmul-splitk`](https://github.com/riceharvest/llama.cpp/tree/vk-xe2-matmul-splitk) at `4fc6ddd00`,
one commit (10 lines in `ggml_vk_guess_split_k`) on llama.cpp master `a0e6dc738`.

## What changes

Prompt-processing matmuls with few output tiles but a long k left most of the B70's 32 cores idle, and the default
split-k only kicks in at 16 tiles or fewer. Rule, Xe2 only: when k >= 6144 and 2/3 x cores < tiles <= 6 x cores, split
k in 4 if k >= 12288, else in 2. Chosen from forced-setting sweeps per shape (medium and small tiles, split 2 and 4).

## Results (pp512, r=5)

| model | before | after |
|---|---|---|
| Qwen3.8-27B UD-Q4_K_XL | 684.3 | **710.6** (+3.8%) |
| 1.7B dense Q8_0 | 5464 | **6091** (+11.5%) |

Correctness: MUL_MAT q5_K 78/78, q4_K 96/96, iq4_xs 49/49, f16 336/336.

The remaining gap to SYCL on the 27B (868 tok/s) is mostly GATED_DELTA_NET (72 ms per pp512, a sequential scan) and
the wide matmuls (~53 TFLOPS).

## Reproduce

```bash
./build/bin/llama-bench -m Qwen3.8-27B-UD-Q4_K_XL.gguf -ngl 99 -sm none -fa 1 -p 512 -n 0 -r 5
./build/bin/test-backend-ops -o MUL_MAT -b Vulkan0 -p "type_a=q5_K"
```
