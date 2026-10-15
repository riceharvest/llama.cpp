# vk-matvec-glu-fusion: SWIGLU/GEGLU in the mat-vec epilogue (parked)

Branch [`vk-matvec-glu-fusion`](https://github.com/riceharvest/llama.cpp/tree/vk-matvec-glu-fusion) at `ca985d7ae`,
three commits on top of `vk-b70-stack` `54b6e8485`.

**Status: parked, never run on a GPU.** Only checked by compiling: C++ `-fsyntax-only` and glslc on 7 mat-vec shader
variants. Expected gain is about ±1%, so it is not worth a test slot yet.

## What changes

- When a mat-vec result (`MUL_MAT` with one output row, or `MUL_MAT_ID` on the mat-vec path) goes straight into a
  split SWIGLU or GEGLU, the mat-vec writes act(gate)·up itself and reads the other operand through the `fuse0`
  binding. The separate GLU dispatch goes away (about 80 per token on Qwen3.6-35B-A3B, one per layer on dense models).
- `graph_optimize` keeps a GLU right after its mat-vec. Before this, the reorder put up and gate in one set and the GLU
  in the next, so the fusion never matched.

## Why only ±1%

It saves one dispatch (~1.8 µs) per GLU, about 0.15 ms of a ~15.6 ms token. But up and gate used to run side by side
without a barrier; now the fused gate kernel waits for up. The bigger win would be one kernel that computes gate and
up together (one input read, one launch), as the SYCL backend and the
[b70-optimization-lab](https://github.com/steveseguin/b70-optimization-lab) `MOE_GATE_UP` fusion do.

## How to test it (if someone picks it up)

`test-backend-ops -o MUL_MAT_VEC_FUSION -b Vulkan0` covers this path (up-mm, gate-mm, GLU; m=1 dense and m<=8 MoE).
`GGML_VK_PERF_LOGGER=1` should show `MUL_MAT_GLU` / `MUL_MAT_ID_GLU` and no standalone GLU.
