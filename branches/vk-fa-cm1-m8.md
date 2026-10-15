# vk-fa-cm1-m8: 8x16x16 coopmat1 flash-attention shader

Branch [`vk-fa-cm1-m8`](https://github.com/riceharvest/llama.cpp/tree/vk-fa-cm1-m8) at `bc0999fac`, one commit on
llama.cpp master `a0e6dc738`.

## What changes

The coopmat1 flash-attention shader hard-codes 16x16x16 muladds. Devices that only advertise 8x16x16 (Intel Xe2: B580,
Arc Pro B60/B70, Lunar Lake on Mesa ANV) fail the shape check and fall back to scalar FA. Using a shape the driver
doesn't advertise is out of spec (VUID-RuntimeSpirv-OpTypeCooperativeMatrixKHR-10163).

The branch adds an `FA_CM1_M8` build of the same shader: each 16-row block is computed as two 8-row muladds in both
K·Qᵀ and P·V. Block sizes and shared memory are unchanged. The 8x16x16 variant is used only when 16x16x16 isn't
supported, so other devices keep the 16x16x16 shader.

## Results

- Correctness on the B70: FLASH_ATTN_EXT **5321/5321**. For comparison, forcing the 16x16x16 shader on this device (the
  shape-gate relaxation proposed in llama.cpp#30503) fails 4243 of 5321.
- Qwen3.8-27B UD-Q4_K_XL, pp512 after 8K of context: **491** tok/s vs 335 with scalar FA (+47%). Decode is unchanged
  (decode on Intel uses the separate `xe_fa` kernel).

## Reproduce

```bash
./build/bin/test-backend-ops -o FLASH_ATTN_EXT -b Vulkan0
./build/bin/llama-bench -m Qwen3.8-27B-UD-Q4_K_XL.gguf -ngl 99 -sm none -fa 1 -p 512 -n 0 -d 8192 -r 3
```

`GGML_VK_PERF_LOGGER=1` shows which FA path ran. The `vk-fa-cm1-m8-testknobs` branch can force the 16x16x16 shader
(`GGML_VK_FA_CM1_FORCE_M16=1`) or disable the m8 one.
