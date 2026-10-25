# tinykv branch

This branch adds support for [Qwen3.6-35B-A3B-TinyKV](https://huggingface.co/darioooooo0o/Qwen3.6-35B-A3B-TinyKV)
(and other TinyKV conversions of Qwen3.5/3.6 models): full-attention layers that share and compress the KV cache.
GGUFs: [darioooooo0o/Qwen3.6-35B-A3B-TinyKV-GGUF](https://huggingface.co/darioooooo0o/Qwen3.6-35B-A3B-TinyKV-GGUF).
Stock llama.cpp refuses these files (missing k/v tensors in the shared layers), so use this branch.

```bash
git clone -b tinykv https://github.com/riceharvest/llama.cpp && cd llama.cpp
cmake -B build -DGGML_CUDA=ON        # or -DGGML_VULKAN=ON, -DGGML_SYCL=ON, CPU only: no flag
cmake --build build -j 8 --target llama-server llama-cli
./build/bin/llama-server -m Qwen3.6-35B-A3B-TinyKV-Q4_K_M.gguf -c 131072 -fa on -ngl 99
```

Use `-fa on`: without flash attention the V cache is padded to the widest layer, which gives back part of the saving.

What changed (all gated on the `<arch>.tinykv.*` keys, stock models are unaffected):

- `src/models/tinykv.h`: the converted attention (`attn_tk_q/k/v/d` maps around stock attention), used by `qwen35` and
  `qwen35moe`; per-layer cache head sizes and KV reuse (`llama-hparams`, `llama-memory-hybrid`), null K/V for layers
  that read another layer's cache (`llama-graph.cpp`).
- `conversion/qwen.py`: `TinyKVQwen3_5[Moe]ForConditionalGeneration` (trained `self_attn.tk.*` tensors replace the
  stock ones; maps kept F32). `llama-quant.cpp` never quantizes `attn_tk_*`.
- `tools/tinykv-logprobs`: per-token log-probs of fixed token sequences, used for the parity checks on the model card.
