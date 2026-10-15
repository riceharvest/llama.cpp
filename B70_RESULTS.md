# Vulkan decode on Intel Arc Pro B70 (Xe2, Mesa): mat-vec fixes

Branch `vk-mmvq-u32-loads` on riceharvest/llama.cpp, based on llama.cpp master `a0e6dc738` (2026-10-14).
This file is only here so the numbers can be reproduced; it is not part of the code change.

## What changes

Two small changes in the Vulkan backend, no new shaders or files:

1. `mul_mat_vecq_funcs.glsl`: the MMVQ (int8 dot) mat-vec read Q3_K (110-byte), Q6_K (210-byte), Q8_0 (34-byte),
   Q5_0 (22-byte) and MXFP4 (17-byte) blocks with 16-bit or 8-bit loads, because those blocks aren't 4-byte aligned.
   Mesa's ANV driver runs those loads very slowly on Xe2. The fix reads 32-bit words and shifts when a block starts
   off a word boundary. On a B70, the q6_K kernel got 5.2x faster, q3_K 8.3x, q8_0 3.1x and q5_0 5.1x
   (test-backend-ops perf, 4096x14336).
2. `ggml-vulkan.cpp`, Xe2 only: IQ4_XS/IQ4_NL DMMV use the large workgroup for any m (those shaders fill a
   lookup table per workgroup), and MMVQ uses the large workgroup when m <= 512.

## Results

Arc Pro B70 32 GB, Fedora 43, kernel 7.1.8, Mesa 25.3.6, one GPU. Qwen3.8-27B UD-Q4_K_XL (unsloth), `-fa 1`, r=3,
tokens/s. SYCL is master `a0e6dc738` built with `-DGGML_SYCL_F16=ON -DGGML_SYCL_DEVICE_ARCH=bmg-g31-a0`, oneAPI 2025.3.

| test | Vulkan master | this branch | SYCL master |
|---|---|---|---|
| decode (tg128) | 13.2 | **22.7** | 21.9 |
| decode after 8K context | 12.6 | **21.1** | 20.2 |
| decode after 32K context | 11.1 | **17.5** | 16.3 |
| prompt (pp512) | 684 | 695 | 868 |

The decode gain comes from the q6_K/q3_K/q8_0 mat-vecs: q6_K alone was 30% of decode time on this model, at
85-89 GB/s. Prompt processing is unchanged; SYCL is still faster there, and much faster at long context.

A 1.7B dense model in Q8_0 went from 71.5 to 154.0 t/s decode (prompt 5291 -> 5596).

Correctness on the B70 (test-backend-ops vs CPU): MUL_MAT q3_K 48/48, q6_K 55/55, q8_0 89/89, q5_0 63/63,
mxfp4 71/71, iq4_xs 49/49, iq4_nl 56/56.

## Reproduce

```bash
git clone -b vk-mmvq-u32-loads https://github.com/riceharvest/llama.cpp && cd llama.cpp
cmake -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build -j 8 --target llama-bench test-backend-ops
./build/bin/llama-bench -m Qwen3.8-27B-UD-Q4_K_XL.gguf -ngl 99 -sm none -fa 1 -p 512 -n 128 -d 0,8192 -r 3
./build/bin/test-backend-ops -o MUL_MAT -b Vulkan0 -p "type_a=q6_K"
./build/bin/test-backend-ops perf -o MUL_MAT -b Vulkan0 -p "type_a=q6_K,type_b=f32,m=4096,n=1,k=14336,"
```

Build master the same way to compare. Only tested on one B70 with Mesa 25.3.6. Other Xe2 cards (B580, B60, Lunar
Lake) and the Windows driver are untested; the Windows driver may not have the slow 16-bit loads at all.

Not submitted upstream.
