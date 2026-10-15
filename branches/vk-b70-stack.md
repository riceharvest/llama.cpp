# vk-b70-stack: all B70 Vulkan fixes together

Branch [`vk-b70-stack`](https://github.com/riceharvest/llama.cpp/tree/vk-b70-stack) at `54b6e8485`, on llama.cpp
master `a0e6dc738`. Prebuilt by [b70-llama-builds](https://github.com/riceharvest/b70-llama-builds) (Vulkan tarball).

Commits, oldest first: FA m8 shader, MMVQ 32-bit loads, Xe2 mat-vec workgroup rule, xe_fa phase-2 split, Xe2 matmul
split-k, and one extra rule that is only in the stack: on Xe2 with Mesa, Q8_0/Q5_K/Q6_K use MMVQ for any k
(llama.cpp sends k < 2048 to DMMV on Intel). That rule took 35B-A3B decode 62.70 → 64.27 (+2.5%), 27B and 1.7B
unchanged; MUL_MAT q8_0 89/89, q5_K 78/78, q6_K 55/55, MUL_MAT_ID q8_0 75/75, q5_K 3/3, q6_K 3/3.

## Results, Qwen3.6-35B-A3B UD-Q4_K_M (MoE, 3B active), tok/s

| test | Vulkan master | vk-b70-stack | SYCL master |
|---|---|---|---|
| decode | 39.0 | **64.3** | 97.7 |
| decode after 8K | 37.0 | 59.6 | 89.8 |
| decode after 32K | 32.7 | 52.5 | 74.5 |
| pp512 | 1854 | **1931** | 1301 |
| pp512 after 8K | 837 | **1337** | 1142 |
| pp512 after 32K | 312 | 630 | 1083 |

Qwen3.8-27B UD-Q4_K_XL: decode 13.2 → 22.7 (SYCL 21.9), after 32K 11.1 → 19.0 (SYCL 16.3), pp512 684 → 711 (SYCL 868).

On this MoE, Vulkan decode is limited by about 1425 small dispatches per token (~3 µs each) and by mat-vec efficiency
at small shapes. SYCL fuses more (its fused gate/up/GLU path is worth +21% there). Mesa 26.x may close part of the
gap; not measured yet.

MUL_MAT_ID on the stack: q3_K 3/3, q4_K 84/84, q5_K 3/3, q6_K 3/3, q8_0 75/75, iq4_xs 4/4.
