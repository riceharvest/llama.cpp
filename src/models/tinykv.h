#pragma once

// tinykv: Qwen3.5/3.6 full-attention layers converted to a smaller KV cache
// (https://huggingface.co/darioooooo0o/Qwen3.6-35B-A3B-TinyKV)
//
// q/gate/k/v projections, q/k RMSNorm and partial RoPE are the stock ones. On top, per layer:
//   owner:   k_c = tk_k @ k (all KV heads flattened),  v_c = tk_v @ v    -> written to the KV cache
//   any:     q_c = tk_q[g] @ q for every query head of KV group g
//   any:     out = tk_d[g] @ attn(q_c, k_c, v_c) -> sigmoid gate -> wo
// A reuser (tinykv_owner[il] != il) has no k/v projections and reads its owner's cache. The cache head size is per
// layer (tinykv_k_dim / tinykv_v_dim); the attention scale stays 1/sqrt(stock head size).

#include "models.h"

static inline void tinykv_load_hparams(llama_model_loader & ml, llama_hparams & hp) {
    std::vector<int32_t>  owners;
    std::vector<uint32_t> k_dims, v_dims;
    if (!ml.get_arr(LLM_KV_TINYKV_OWNERS, owners, false)) {
        return;
    }
    ml.get_arr(LLM_KV_TINYKV_K_DIMS, k_dims, true);
    ml.get_arr(LLM_KV_TINYKV_V_DIMS, v_dims, true);
    GGML_ASSERT(owners.size() == hp.n_layer() && k_dims.size() == owners.size() && v_dims.size() == owners.size());
    hp.tinykv = true;
    for (size_t il = 0; il < owners.size(); ++il) {
        hp.tinykv_owner[il] = owners[il];
        hp.tinykv_k_dim[il] = k_dims[il];       // reusers carry their owner's sizes, so cache views line up
        hp.tinykv_v_dim[il] = v_dims[il];
    }
}

// y[:, h] = w[g(h)] @ x[:, h] for x [d_in, n_head, T], w [d_in, d_out, n_kv] (head h belongs to KV group h / (n_head/n_kv))
static inline ggml_tensor * tinykv_group_map(ggml_context * ctx, ggml_tensor * w, ggml_tensor * x, int64_t n_head, int64_t n_tokens) {
    const int64_t d_in = w->ne[0], d_out = w->ne[1], n_kv = w->ne[2], G = n_head / n_kv;
    x = ggml_reshape_4d(ctx, ggml_cont(ctx, x), d_in, G, n_kv, n_tokens);
    x = ggml_cont(ctx, ggml_permute(ctx, x, 0, 1, 3, 2));                  // [d_in, G, T, n_kv]
    x = ggml_mul_mat(ctx, w, ggml_reshape_3d(ctx, x, d_in, G * n_tokens, n_kv));
    x = ggml_reshape_4d(ctx, x, d_out, G, n_tokens, n_kv);
    x = ggml_cont(ctx, ggml_permute(ctx, x, 0, 1, 3, 2));                  // [d_out, G, n_kv, T]
    return ggml_reshape_3d(ctx, x, d_out, n_head, n_tokens);
}

template <typename G>
static inline ggml_tensor * tinykv_build_attn(G & g, const llama_layer & layer, llm_graph_input_attn_kv * inp,
                                              ggml_tensor * cur, ggml_tensor * inp_pos, int * sections, int il) {
    ggml_context * ctx0 = g.ctx0;
    const auto & hp = g.hparams;
    const int64_t hd       = hp.n_embd_head_k_full;
    const int64_t n_head   = hp.n_head(il);
    const int64_t n_kv     = hp.n_head_kv(il);
    const int64_t n_tokens = cur->ne[1];
    const bool    owner    = hp.tinykv_owner[il] == il;

    ggml_tensor * Qfull;
    ggml_tensor * Kcur = nullptr;
    ggml_tensor * Vcur = nullptr;
    if (owner) {
        auto [q, k, v] = g.build_qkv(layer, cur, hd * 2, n_head, hd, n_kv, hd, n_kv, il, false);
        Qfull = q; Kcur = k; Vcur = v;
    } else {
        Qfull = g.build_lora_mm(layer.wq, cur, layer.wq_s);
    }
    g.cb(Qfull, "Qcur_full", il);

    ggml_tensor * Qcur = ggml_view_3d(ctx0, Qfull, hd, n_head, n_tokens,
        ggml_element_size(Qfull) * hd * 2, ggml_element_size(Qfull) * hd * 2 * n_head, 0);
    Qcur = g.build_norm(Qcur, layer.attn_q_norm, nullptr, LLM_NORM_RMS, il);
    ggml_tensor * gate = ggml_view_3d(ctx0, Qfull, hd, n_head, n_tokens,
        ggml_element_size(Qfull) * hd * 2, ggml_element_size(Qfull) * hd * 2 * n_head, ggml_element_size(Qfull) * hd);
    gate = ggml_cont_2d(ctx0, gate, hd * n_head, n_tokens);

    Qcur = ggml_rope_multi(ctx0, Qcur, inp_pos, nullptr, g.n_rot, sections, g.rope_type, g.n_ctx_orig, g.freq_base,
                           g.freq_scale, g.ext_factor, g.attn_factor, g.beta_fast, g.beta_slow);
    Qcur = tinykv_group_map(ctx0, layer.attn_tk_q, Qcur, n_head, n_tokens);
    g.cb(Qcur, "Qcur_tk", il);

    if (owner) {
        Kcur = ggml_reshape_3d(ctx0, Kcur, hd, n_kv, n_tokens);
        Kcur = g.build_norm(Kcur, layer.attn_k_norm, nullptr, LLM_NORM_RMS, il);
        Kcur = ggml_rope_multi(ctx0, Kcur, inp_pos, nullptr, g.n_rot, sections, g.rope_type, g.n_ctx_orig, g.freq_base,
                               g.freq_scale, g.ext_factor, g.attn_factor, g.beta_fast, g.beta_slow);
        Kcur = ggml_mul_mat(ctx0, layer.attn_tk_k, ggml_reshape_2d(ctx0, ggml_cont(ctx0, Kcur), hd * n_kv, n_tokens));
        Kcur = ggml_reshape_3d(ctx0, Kcur, hp.tinykv_k_dim[il], n_kv, n_tokens);
        Vcur = ggml_mul_mat(ctx0, layer.attn_tk_v, ggml_reshape_2d(ctx0, ggml_cont(ctx0, Vcur), hd * n_kv, n_tokens));
        Vcur = ggml_reshape_3d(ctx0, Vcur, hp.tinykv_v_dim[il], n_kv, n_tokens);
        g.cb(Kcur, "Kcur_tk", il);
        g.cb(Vcur, "Vcur_tk", il);
    }

    cur = g.build_attn(inp, nullptr, nullptr, nullptr, Qcur, Kcur, Vcur, nullptr, nullptr, nullptr,
                       1.0f / sqrtf(float(hd)), il);
    cur = tinykv_group_map(ctx0, layer.attn_tk_d, ggml_reshape_3d(ctx0, cur, hp.tinykv_v_dim[il], n_head, n_tokens),
                           n_head, n_tokens);
    cur = ggml_reshape_2d(ctx0, cur, hd * n_head, n_tokens);
    g.cb(cur, "attn_pregate", il);

    cur = ggml_mul(ctx0, cur, ggml_sigmoid(ctx0, gate));
    cur = g.build_lora_mm(layer.wo, cur, layer.wo_s);
    g.cb(cur, "attn_output", il);
    return cur;
}
