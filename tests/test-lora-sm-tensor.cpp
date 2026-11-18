// test: lora adapters under SPLIT_MODE_TENSOR (meta device)
//
// runs the same model and prompt through:
//   reference : one CPU device, LLAMA_SPLIT_MODE_LAYER
//   meta      : 2+ non-CPU devices, LLAMA_SPLIT_MODE_TENSOR
// and compares the logits of the two runs, with and without a lora adapter
// on all linear layers of the model:
//   - with lora    : exercises the split states of the lora a and b matrices
//   - without lora : checks that the base path is unchanged
//
// usage:
//   ./build/bin/llama-rpc-server -H 127.0.0.1:50051 &
//   ./build/bin/llama-rpc-server -H 127.0.0.1:50052 &
//   ./build/bin/test-lora-sm-tensor --rpc 127.0.0.1:50051,127.0.0.1:50052

#include "common.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"

#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

static bool silent_model_load_progress(float /*progress*/, void * /*user_data*/) {
    return true;
}

// deterministic per-tensor fill
static std::vector<float> fill_tensor(const char * name, size_t n) {
    uint64_t seed = 0x9e3779b97f4a7c15ULL;
    for (const char * p = name; *p; p++) {
        seed = seed * 131 + (uint8_t) *p;
    }
    std::vector<float>              data(n);
    std::mt19937_64                 gen(seed);
    std::normal_distribution<float> dis(0.0f, 0.05f);
    for (size_t i = 0; i < n; i++) {
        data[i] = dis(gen);
    }
    return data;
}

struct gguf_builder {
    struct gguf_context *                     ctx;
    std::map<std::string, std::vector<float>> tensor_data;
    std::string                               arch;

    gguf_builder(const char * arch) : ctx(gguf_init_empty()), arch(arch) {
        gguf_set_val_str(ctx, "general.architecture", arch);
    }

    ~gguf_builder() { gguf_free(ctx); }

    void kv_u32(const char * key, uint32_t v) { gguf_set_val_u32(ctx, key, v); }

    void kv_f32(const char * key, float v) { gguf_set_val_f32(ctx, key, v); }

    void kv_str(const char * key, const char * v) { gguf_set_val_str(ctx, key, v); }

    void tensor(const char * name, uint32_t ne0, uint32_t ne1) {
        ggml_init_params p    = { /*.mem_size=*/16 * 1024 * 1024, /*.mem_buffer=*/nullptr, /*.no_alloc=*/false };
        ggml_context *   mctx = ggml_init(p);
        ggml_tensor *    t    = ggml_new_tensor_2d(mctx, GGML_TYPE_F32, ne0, ne1);
        ggml_set_name(t, name);
        gguf_add_tensor(ctx, t);
        ggml_free(mctx);
        tensor_data[std::string(name)] = fill_tensor(name, (size_t) ne0 * ne1);
    }

    void tensor_1d(const char * name, uint32_t ne0) {
        ggml_init_params p    = { /*.mem_size=*/16 * 1024 * 1024, /*.mem_buffer=*/nullptr, /*.no_alloc=*/false };
        ggml_context *   mctx = ggml_init(p);
        ggml_tensor *    t    = ggml_new_tensor_1d(mctx, GGML_TYPE_F32, ne0);
        ggml_set_name(t, name);
        gguf_add_tensor(ctx, t);
        ggml_free(mctx);
        tensor_data[std::string(name)] = fill_tensor(name, ne0);
    }

    std::string write(const std::string & path) const {
        for (const auto & kv : tensor_data) {
            gguf_set_tensor_data(const_cast<struct gguf_context *>(ctx), kv.first.c_str(), kv.second.data());
        }
        if (!gguf_write_to_file(ctx, path.c_str(), /*only_meta =*/false)) {
            throw std::runtime_error("failed to write " + path);
        }
        return path;
    }
};

static std::string get_tmp_path(const char * name) {
    static size_t id = 0;
    return (std::filesystem::temp_directory_path() /
            (std::string("test-lora-sm-tensor-") + name + "-" + std::to_string(++id) + ".gguf"))
        .string();
}

struct model_dims {
    uint32_t n_vocab   = 128;
    uint32_t n_embd    = 128;
    uint32_t n_head    = 4;
    uint32_t n_head_kv = 2;
    uint32_t n_ff      = 256;
    uint32_t n_layer   = 3;
    uint32_t lora_rank = 4;
};

// plain llama arch, all linear layers
static std::string build_model(const model_dims & d) {
    gguf_builder g("llama");
    g.kv_u32("llama.block_count", d.n_layer);
    g.kv_u32("llama.context_length", 256);
    g.kv_u32("llama.embedding_length", d.n_embd);
    g.kv_u32("llama.attention.head_count", d.n_head);
    g.kv_u32("llama.attention.head_count_kv", d.n_head_kv);
    g.kv_f32("llama.attention.layer_norm_rms_epsilon", 1e-5f);
    g.kv_f32("llama.attention.relative_attention_factor", 1.0f);
    g.kv_u32("llama.feed_forward_length", d.n_ff);
    g.kv_u32("llama.vocab_size", d.n_vocab);
    g.kv_f32("llama.rope.freq_base", 10000.0f);
    g.kv_u32("llama.attention.sliding_window_pattern", 2);

    // dummy tokenizer
    g.kv_str("tokenizer.ggml.model", "test");
    std::vector<std::string> tok_list(d.n_vocab);
    for (uint32_t i = 0; i < d.n_vocab; i++) {
        tok_list[i] = "tok_" + std::to_string(i);
    }
    std::vector<const char *> tok_ptrs(d.n_vocab);
    for (uint32_t i = 0; i < d.n_vocab; i++) {
        tok_ptrs[i] = tok_list[i].c_str();
    }
    std::vector<float> tok_scores(d.n_vocab, 0.0f);
    gguf_set_arr_str(g.ctx, "tokenizer.ggml.tokens", tok_ptrs.data(), d.n_vocab);
    gguf_set_arr_data(g.ctx, "tokenizer.ggml.scores", GGUF_TYPE_FLOAT32, tok_scores.data(), tok_scores.size());

    g.tensor("token_embd.weight", d.n_embd, d.n_vocab);
    g.tensor_1d("output_norm.weight", d.n_embd);
    g.tensor("output.weight", d.n_embd, d.n_vocab);

    const uint32_t head_dim = d.n_embd / d.n_head;
    char           name[128];
    for (uint32_t il = 0; il < d.n_layer; il++) {
        snprintf(name, sizeof(name), "blk.%u.attn_norm.weight", il);
        g.tensor_1d(name, d.n_embd);
        snprintf(name, sizeof(name), "blk.%u.ffn_norm.weight", il);
        g.tensor_1d(name, d.n_embd);
        snprintf(name, sizeof(name), "blk.%u.attn_q.weight", il);
        g.tensor(name, d.n_embd, d.n_head * head_dim);
        snprintf(name, sizeof(name), "blk.%u.attn_k.weight", il);
        g.tensor(name, d.n_embd, d.n_head_kv * head_dim);
        snprintf(name, sizeof(name), "blk.%u.attn_v.weight", il);
        g.tensor(name, d.n_embd, d.n_head_kv * head_dim);
        snprintf(name, sizeof(name), "blk.%u.attn_output.weight", il);
        g.tensor(name, d.n_embd, d.n_head * head_dim);
        snprintf(name, sizeof(name), "blk.%u.ffn_up.weight", il);
        g.tensor(name, d.n_embd, d.n_ff);
        snprintf(name, sizeof(name), "blk.%u.ffn_gate.weight", il);
        g.tensor(name, d.n_embd, d.n_ff);
        snprintf(name, sizeof(name), "blk.%u.ffn_down.weight", il);
        g.tensor(name, d.n_ff, d.n_embd);
    }
    return g.write(get_tmp_path("model"));
}

// rank-d.lora_rank lora on every linear layer, alpha = rank so the scale is 1.0
static std::string build_lora(const model_dims & d) {
    gguf_builder g("llama");
    g.kv_str("general.type", "adapter");
    g.kv_str("adapter.type", "lora");
    g.kv_f32("adapter.lora.alpha", (float) d.lora_rank);

    const uint32_t r        = d.lora_rank;
    const uint32_t head_dim = d.n_embd / d.n_head;
    char           name[128];
    for (uint32_t il = 0; il < d.n_layer; il++) {
        struct {
            const char * base;
            uint32_t     ne0;
            uint32_t     ne1;
        } layers[] = {
            { "attn_q.weight",      d.n_embd, d.n_head * head_dim    },
            { "attn_k.weight",      d.n_embd, d.n_head_kv * head_dim },
            { "attn_v.weight",      d.n_embd, d.n_head_kv * head_dim },
            { "attn_output.weight", d.n_embd, d.n_head * head_dim    },
            { "ffn_up.weight",      d.n_embd, d.n_ff                 },
            { "ffn_gate.weight",    d.n_embd, d.n_ff                 },
            { "ffn_down.weight",    d.n_ff,   d.n_embd               },
        };

        for (const auto & l : layers) {
            snprintf(name, sizeof(name), "blk.%u.%s", il, l.base);
            std::string base = name;
            g.tensor((base + ".lora_a").c_str(), l.ne0, r);
            g.tensor((base + ".lora_b").c_str(), r, l.ne1);
        }
    }
    return g.write(get_tmp_path("lora"));
}

static double nmse(const std::vector<float> & a, const std::vector<float> & b) {
    double mse_a_b = 0.0;
    double mse_a_0 = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        mse_a_b += (a[i] - b[i]) * (a[i] - b[i]);
        mse_a_0 += a[i] * a[i];
    }
    return mse_a_b / mse_a_0;
}

static std::vector<float> run(const std::string &                     model_path,
                              const std::string *                     lora_path,
                              const std::vector<ggml_backend_dev_t> & devs,
                              llama_split_mode                        split_mode,
                              const std::vector<llama_token> &        tokens) {
    llama_model_params mp                 = llama_model_default_params();
    mp.progress_callback                  = silent_model_load_progress;
    std::vector<ggml_backend_dev_t> devs2 = devs;
    devs2.push_back(nullptr);
    mp.devices    = devs2.data();
    mp.split_mode = split_mode;

    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) {
        throw std::runtime_error("failed to load model");
    }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx                = 256;
    cp.n_threads            = 4;
    cp.n_threads_batch      = 4;
    llama_context * ctx     = llama_init_from_model(model, cp);
    if (!ctx) {
        llama_model_free(model);
        throw std::runtime_error("failed to create context");
    }

    llama_adapter_lora * adapter = nullptr;
    if (lora_path != nullptr) {
        adapter = llama_adapter_lora_init(model, lora_path->c_str());
        if (!adapter) {
            llama_free(ctx);
            llama_model_free(model);
            throw std::runtime_error("failed to init lora adapter");
        }
        float scale = 1.0f;
        llama_set_adapters_lora(ctx, &adapter, 1, &scale);
    }

    const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    common_batch   batch(ctx);
    for (uint32_t i = 0; i < tokens.size(); i++) {
        batch.add(tokens[i], i, 0, true);
    }
    if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
        llama_free(ctx);
        llama_model_free(model);
        throw std::runtime_error("failed to process batch");
    }

    std::vector<float> logits;
    logits.reserve(tokens.size() * n_vocab);
    for (uint32_t i = 0; i < tokens.size(); i++) {
        const float * logits_ith = llama_get_logits_ith(ctx, i);
        logits.insert(logits.end(), logits_ith, logits_ith + n_vocab);
    }

    llama_free(ctx);
    llama_model_free(model);
    return logits;
}

int main(int argc, char ** argv) {
    std::vector<std::string> rpc_endpoints;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--rpc") == 0 && i + 1 < argc) {
            std::string ep    = argv[++i];
            size_t      start = 0;
            while (start <= ep.size()) {
                size_t comma = ep.find(',', start);
                rpc_endpoints.push_back(ep.substr(start, comma == std::string::npos ? comma : comma - start));
                if (comma == std::string::npos) {
                    break;
                }
                start = comma + 1;
            }
        } else {
            fprintf(stderr, "usage: %s --rpc host:port,host:port\n", argv[0]);
            return 1;
        }
    }

    ggml_backend_load_all();
    ggml_backend_reg_t rpc_reg = ggml_backend_reg_by_name("RPC");
    if (!rpc_reg) {
        fprintf(stderr, "%s: failed to find RPC backend\n", __func__);
        return 1;
    }
    typedef ggml_backend_reg_t (*ggml_backend_rpc_add_server_t)(const char * endpoint);
    ggml_backend_rpc_add_server_t add_server =
        (ggml_backend_rpc_add_server_t) ggml_backend_reg_get_proc_address(rpc_reg, "ggml_backend_rpc_add_server");
    if (!add_server) {
        fprintf(stderr, "%s: failed to find ggml_backend_rpc_add_server\n", __func__);
        return 1;
    }
    for (const auto & ep : rpc_endpoints) {
        ggml_backend_reg_t reg = add_server(ep.c_str());
        if (!reg) {
            fprintf(stderr, "%s: failed to connect to RPC server %s\n", __func__, ep.c_str());
            return 1;
        }
        ggml_backend_register(reg);
    }

    ggml_backend_dev_t              cpu_dev = nullptr;
    std::vector<ggml_backend_dev_t> meta_devs;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_buffer_type(dev) == ggml_backend_cpu_buffer_type()) {
            cpu_dev = dev;
        } else {
            meta_devs.push_back(dev);
        }
    }
    if (!cpu_dev || meta_devs.size() < 2) {
        fprintf(stderr, "%s: need a CPU device and 2+ non-CPU devices (found %zu)\n", __func__, meta_devs.size());
        return 1;
    }

    model_dims        dims;
    const std::string model_path = build_model(dims);
    const std::string lora_path  = build_lora(dims);

    std::mt19937                    gen(0x1234);
    std::uniform_int_distribution<> dis(0, dims.n_vocab - 1);
    std::vector<llama_token>        tokens(8);
    for (auto & t : tokens) {
        t = dis(gen);
    }

    std::vector<float> ref_lora  = run(model_path, &lora_path, { cpu_dev }, LLAMA_SPLIT_MODE_LAYER, tokens);
    std::vector<float> meta_lora = run(model_path, &lora_path, meta_devs, LLAMA_SPLIT_MODE_TENSOR, tokens);
    const double       nmse_lora = nmse(ref_lora, meta_lora);
    printf("CHECK lora meta-vs-ref nmse=%.3e\n", nmse_lora);
    if (nmse_lora > 1e-4) {
        fprintf(stderr, "%s: lora logits differ, nmse = %g\n", __func__, nmse_lora);
        return 1;
    }
    printf("CHECK-PASSED lora meta-vs-ref\n");

    std::vector<float> ref_bare  = run(model_path, nullptr, { cpu_dev }, LLAMA_SPLIT_MODE_LAYER, tokens);
    std::vector<float> meta_bare = run(model_path, nullptr, meta_devs, LLAMA_SPLIT_MODE_TENSOR, tokens);
    const double       nmse_bare = nmse(ref_bare, meta_bare);
    printf("CHECK no-lora meta-vs-ref nmse=%.3e\n", nmse_bare);
    if (nmse_bare > 1e-4) {
        fprintf(stderr, "%s: base logits differ, nmse = %g\n", __func__, nmse_bare);
        return 1;
    }
    printf("CHECK-PASSED no-lora meta-vs-ref\n");

    printf("test-lora-sm-tensor: all checks passed\n");
    return 0;
}
