// Per-token log-probabilities of fixed token sequences, for parity checks against a reference implementation.
//
//   llama-tinykv-logprobs <model.gguf> <in.txt> <out.txt> [n_gpu_layers=0] [flash_attn=1] [n_batch=2048] [argmax.txt]
//
// in.txt: one sequence per line, space-separated token ids. out.txt: one line per sequence with log p(t_i | t_<i) for
// i = 1..n-1, space-separated (the first token has no prediction). argmax.txt (optional, needle mode): 1 where t_i is
// the argmax; in needle mode only the last 64 positions are scored (answers sit at the end; full-vocab softmax per
// position on the CPU is otherwise the bottleneck at 100k+ tokens).
// TINYKV_CACHE_TYPE=f16|q8_0|q4_0 (env) sets the K and V cache type (default f16).

#include "llama.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <model.gguf> <in.txt> <out.txt> [n_gpu_layers] [flash_attn 0|1] [n_batch]\n", argv[0]);
        return 1;
    }
    const int n_gpu   = argc > 4 ? std::stoi(argv[4]) : 0;
    const int fa      = argc > 5 ? std::stoi(argv[5]) : 1;
    const int n_batch = argc > 6 ? std::stoi(argv[6]) : 2048;
    const char * f_top = argc > 7 ? argv[7] : nullptr;
    const size_t n_tail = f_top ? 64 : 0;

    std::vector<std::vector<llama_token>> seqs;
    size_t n_max = 0;
    {
        std::ifstream in(argv[2]);
        for (std::string line; std::getline(in, line);) {
            std::istringstream ss(line);
            std::vector<llama_token> s;
            for (llama_token t; ss >> t;) {
                s.push_back(t);
            }
            if (s.size() > 1) {
                n_max = std::max(n_max, s.size());
                seqs.push_back(std::move(s));
            }
        }
    }

    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = n_gpu;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) {
        return 1;
    }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    auto cp = llama_context_default_params();
    cp.n_ctx           = n_max + 16;
    cp.n_batch         = n_batch;
    cp.n_ubatch        = std::min(n_batch, 512);
    cp.flash_attn_type = fa ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    if (const char * ct = std::getenv("TINYKV_CACHE_TYPE")) {
        const ggml_type t = !strcmp(ct, "q8_0") ? GGML_TYPE_Q8_0 : !strcmp(ct, "q4_0") ? GGML_TYPE_Q4_0 : GGML_TYPE_F16;
        cp.type_k = t;
        cp.type_v = t;
        fprintf(stderr, "cache type %s\n", ggml_type_name(t));
    }
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        return 1;
    }

    std::ofstream out(argv[3]);
    std::ofstream top;
    if (f_top) {
        top.open(f_top);
    }
    llama_batch batch = llama_batch_init(n_batch, 0, 1);
    for (const auto & s : seqs) {
        llama_memory_clear(llama_get_memory(ctx), true);
        std::vector<double> lp;
        std::vector<int>    hit;
        for (size_t i0 = 0; i0 + 1 < s.size(); i0 += n_batch) {
            const size_t n = std::min((size_t) n_batch, s.size() - 1 - i0);   // predict up to the last token
            batch.n_tokens = n;
            const size_t first = n_tail && s.size() - 1 > n_tail ? s.size() - 1 - n_tail : 0;   // first scored position
            for (size_t j = 0; j < n; ++j) {
                batch.token[j]     = s[i0 + j];
                batch.pos[j]       = i0 + j;
                batch.n_seq_id[j]  = 1;
                batch.seq_id[j][0] = 0;
                batch.logits[j]    = i0 + j >= first;
            }
            if (llama_decode(ctx, batch) != 0) {
                fprintf(stderr, "decode failed\n");
                return 1;
            }
            for (size_t j = 0; j < n; ++j) {
                if (!batch.logits[j]) {
                    continue;
                }
                const float * lg = llama_get_logits_ith(ctx, j);
                double mx = lg[0];
                int    am = 0;
                for (int v = 1; v < n_vocab; ++v) {
                    if (lg[v] > mx) {
                        mx = lg[v];
                        am = v;
                    }
                }
                hit.push_back(am == s[i0 + j + 1]);
                double z = 0.0;
                for (int v = 0; v < n_vocab; ++v) {
                    z += std::exp(lg[v] - mx);
                }
                lp.push_back(lg[s[i0 + j + 1]] - mx - std::log(z));
            }
        }
        for (size_t i = 0; i < lp.size(); ++i) {
            out << (i ? " " : "") << lp[i];
        }
        out << "\n";
        if (f_top) {
            for (size_t i = 0; i < hit.size(); ++i) {
                top << (i ? " " : "") << hit[i];
            }
            top << "\n";
        }
        fprintf(stderr, "sequence of %zu tokens: mean logp %.4f\n", s.size(), [&] { double a = 0; for (double x : lp) a += x; return a / lp.size(); }());
    }
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
