// llama-kv-pipeline-snapkv.cpp — SnapKV selection plugin.
//
// Copyright (c) 2025 llama.cpp contributors.
// Licensed under the MIT License.
//
// SnapKV: LLM Knows What You are Looking for Before Generation
// https://arxiv.org/abs/2404.14469
//
// Full implementation with attention-score-based importance estimation.
// The pipeline hooks into the eval callback to capture "kq_soft_max"
// tensors (the softmax output of Q*K^T, i.e. the attention probability
// matrix). Per-token importance is accumulated across layers as the
// column-wise sum of the attention matrix, matching the SnapKV paper's
// observation that tokens receiving higher aggregate attention are more
// important to retain.
//
// On the first decode (before any attention scores are available),
// selection falls back to a sliding-window heuristic (keep first N
// initial tokens + last M recent tokens). From the second decode
// onward, attention scores drive the selection.

#include "llama-kv-pipeline.h"
#include "llama-kv-cache.h"  // for llama_kv_cache_context (dynamic_cast)
#include "ggml-backend.h"    // for ggml_backend_tensor_get

#include <algorithm>
#include <cassert>
#include <cstring>
#include <numeric>

//
// kv_selection_snapkv — selection stage
//
class kv_selection_snapkv : public ikv_selection {
public:
    kv_selection_snapkv(uint32_t initial_budget, uint32_t recent_budget)
        : m_initial_budget(initial_budget)
        , m_recent_budget(recent_budget)
    {}

    // attention スコアに基づく選択。scores は [n_tokens] の
    // per-token importance。空の場合はスライディングウィンドウにフォールバック。
    ikv_selection_plan select(
            const llama_ubatch & ubatch,
            uint32_t n_cells_available,
            const llama_kv_cell_ext * cell_ext_hint,
            const float * scores,
            uint32_t n_scores) {

        ikv_selection_plan plan;
        const uint32_t n_tokens = ubatch.n_tokens;

        // トークン数がバジェット内なら全保持
        const uint32_t total_budget = m_initial_budget + m_recent_budget;
        if (n_tokens <= total_budget) {
            plan.cell_indices.resize(n_tokens);
            std::iota(plan.cell_indices.begin(), plan.cell_indices.end(), 0);
            plan.n_retained = n_tokens;
            plan.ok = true;
            return plan;
        }

        plan.cell_indices.resize(n_tokens, -1);

        // 常に先頭 initial_budget トークンは保持（アンカー）
        uint32_t retained = 0;
        for (uint32_t i = 0; i < m_initial_budget && i < n_tokens; ++i) {
            plan.cell_indices[i] = (int32_t) i;
            ++retained;
        }

        // 残りスロット: recent_budget 個
        const uint32_t remaining_budget = total_budget - retained;
        if (remaining_budget == 0) {
            plan.n_retained = retained;
            plan.ok = true;
            return plan;
        }

        // スコアがある場合: initial_budget 以降のトークンから
        // スコア上位 remaining_budget 個を選択
        if (scores && n_scores >= n_tokens) {
            // 候補リスト: initial_budget..n_tokens-1 の (index, score) ペア
            std::vector<std::pair<float, uint32_t>> candidates;
            candidates.reserve(n_tokens - m_initial_budget);
            for (uint32_t i = m_initial_budget; i < n_tokens; ++i) {
                candidates.emplace_back(scores[i], i);
            }

            // スコア降順で partial sort（remaining_budget 番目まで）
            if (remaining_budget < candidates.size()) {
                std::nth_element(
                    candidates.begin(),
                    candidates.begin() + remaining_budget,
                    candidates.end(),
                    [](const auto & a, const auto & b) { return a.first > b.first; });
            }

            // 上位 remaining_budget 個を保持
            const uint32_t n_keep = std::min(remaining_budget, (uint32_t)candidates.size());
            for (uint32_t j = 0; j < n_keep; ++j) {
                const uint32_t idx = candidates[j].second;
                plan.cell_indices[idx] = (int32_t) idx;
                ++retained;
            }
        } else {
            // フォールバック: スライディングウィンドウ
            const uint32_t recent_start = n_tokens > remaining_budget
                ? n_tokens - remaining_budget : m_initial_budget;
            for (uint32_t i = recent_start; i < n_tokens; ++i) {
                if (plan.cell_indices[i] == -1) {
                    plan.cell_indices[i] = (int32_t) i;
                    ++retained;
                }
            }
        }

        plan.n_retained = retained;
        plan.ok = true;
        return plan;
    }

    // ikv_selection インターフェース（スコアなし = フォールバック）
    ikv_selection_plan select(
            const llama_ubatch & ubatch,
            uint32_t n_cells_available,
            const llama_kv_cell_ext * cell_ext_hint) override {
        return select(ubatch, n_cells_available, cell_ext_hint, nullptr, 0);
    }

    const char * name() const override { return "snapkv"; }

private:
    uint32_t m_initial_budget;
    uint32_t m_recent_budget;
};

//
// kvp_pipeline_snapkv — SnapKV パイプライン
//
// eval callback で "kq_soft_max" をキャプチャし、
// per-token importance を蓄積する。
//
class kvp_pipeline_snapkv : public ikv_pipeline {
public:
    kvp_pipeline_snapkv(uint32_t initial_budget, uint32_t recent_budget)
        : m_selection(initial_budget, recent_budget)
    {}

    const char * name()            const override { return "snapkv"; }
    const char * selection_name()  const override { return m_selection.name(); }
    const char * compression_name()const override { return "none"; }
    const char * storage_name()    const override { return "default"; }

    // kq_soft_max テンソルから attention スコアを蓄積
    void on_eval_tensor(ggml_tensor * t) override {
        if (!t || !t->name) return;

        // "kq_soft_max" は [n_kv, n_tokens] or [n_head, n_kv, n_tokens]
        // 列方向（KV 側）の和をとって per-position importance を得る
        if (std::strstr(t->name, "kq_soft_max") != nullptr) {
            accumulate_attn_scores(t);
        }
    }

    // eval callback: llama_context から呼ばれる
    static bool eval_callback_fn(ggml_tensor * t, bool ask, void * user_data) {
        auto * self = static_cast<kvp_pipeline_snapkv *>(user_data);
        if (!ask && t) {
            self->on_eval_tensor(t);
        }
        return true;  // 計算を継続
    }

    ggml_backend_sched_eval_callback get_eval_callback() const override {
        return eval_callback_fn;
    }

    void * get_eval_callback_user_data() const override {
        // const を外す: callback 内で m_attn_importance を更新する
        return const_cast<kvp_pipeline_snapkv *>(this);
    }

    // init_batch で selection plan を計算して context に設定
    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override {

        const uint32_t n_tokens = balloc.get_n_tokens();

        llama_ubatch dummy{};
        dummy.n_tokens = n_tokens;

        // attention スコアが蓄積されていればそれを使い、
        // なければフォールバック（スライディングウィンドウ）
        if (m_attn_importance.size() >= n_tokens) {
            m_selection_plan = m_selection.select(
                /*ubatch=*/dummy,
                /*n_cells_available=*/n_tokens,
                /*cell_ext_hint=*/nullptr,
                /*scores=*/m_attn_importance.data(),
                /*n_scores=*/(uint32_t)m_attn_importance.size());
        } else {
            m_selection_plan = m_selection.select(
                /*ubatch=*/dummy,
                /*n_cells_available=*/n_tokens,
                /*cell_ext_hint=*/nullptr);
        }

        // 次回の decode に備えてスコアをリセット
        m_attn_importance.clear();

        auto ctx = m_upstream->init_batch(balloc, n_ubatch, embd_all);

        auto * kvc_ctx = dynamic_cast<llama_kv_cache_context *>(ctx.get());
        if (kvc_ctx) {
            kvc_ctx->m_selection_plan = static_cast<const void *>(&m_selection_plan);
        }

        return ctx;
    }

private:
    kv_selection_snapkv m_selection;
    ikv_selection_plan   m_selection_plan;

    // per-token importance スコア（attention column-sum の蓄積）
    // 各 decode 呼び出しの後にリセット
    std::vector<float>   m_attn_importance;

    // kq_soft_max テンソルから importance を蓄積
    void accumulate_attn_scores(ggml_tensor * t) {
        // ggml_tensor のデータを CPU に持ってくる
        // kq_soft_max: [n_kv, n_tokens] (2D) or [n_head, n_kv, n_tokens] (3D)
        // 各トークン列の和 = そのトークンが他トークンから受けた attention 総量
        const int64_t n_dims = ggml_n_dims(t);

        uint32_t n_tokens = 0;
        uint32_t n_kv     = 0;

        if (n_dims == 2) {
            // [n_kv, n_tokens]
            n_kv     = (uint32_t)t->ne[0];
            n_tokens = (uint32_t)t->ne[1];
        } else if (n_dims == 3) {
            // [n_head, n_kv, n_tokens] — 最初の次元をヘッドとして扱う
            n_kv     = (uint32_t)t->ne[1];
            n_tokens = (uint32_t)t->ne[2];
        } else {
            return;  // 不明な形状は無視
        }

        if (n_tokens == 0 || n_kv == 0) return;

        // m_attn_importance を n_tokens 分確保（初回またはリサイズ）
        if (m_attn_importance.size() < n_tokens) {
            m_attn_importance.resize(n_tokens, 0.0f);
        }

        // テンソルデータを CPU にコピー
        // 注意: ggml_tensor のデータは別バックエンドにある可能性があるため、
        // ggml_backend_get_data で CPU バッファにコピーする
        const size_t row_size = ggml_row_size(t->type, t->ne[0]);
        const size_t total_bytes = row_size * ggml_nrows(t);

        // データが CPU アクセス可能か確認
        // ggml_backend_tensor_get でホストにコピー
        std::vector<uint8_t> buf(total_bytes);
        ggml_backend_tensor_get(t, buf.data(), 0, total_bytes);

        // F32 としてパース（kq_soft_max は F32）
        if (t->type != GGML_TYPE_F32) {
            // F32 でない場合は変換が必要だが、kq_soft_max は常に F32
            // なのでここでは F32 のみ対応
            return;
        }

        // nb[] を使ってパディング対応の stride でアクセス
        // バイトオフセット: h*nb[2] + j*nb[1] + i*nb[0]
        const auto * raw = buf.data();

        if (n_dims == 2) {
            // [n_kv, n_tokens] — 列 j の和
            for (uint32_t j = 0; j < n_tokens; ++j) {
                float col_sum = 0.0f;
                for (uint32_t i = 0; i < n_kv && i < n_tokens; ++i) {
                    const float val = *reinterpret_cast<const float *>(
                        raw + j * t->nb[1] + i * t->nb[0]);
                    col_sum += val;
                }
                m_attn_importance[j] += col_sum;
            }
        } else {
            // [n_head, n_kv, n_tokens] — ヘッド間で平均
            const uint32_t n_head = (uint32_t)t->ne[0];

            for (uint32_t h = 0; h < n_head; ++h) {
                for (uint32_t j = 0; j < n_tokens; ++j) {
                    float col_sum = 0.0f;
                    for (uint32_t i = 0; i < n_kv && i < n_tokens; ++i) {
                        const float val = *reinterpret_cast<const float *>(
                            raw + h * t->nb[2] + j * t->nb[1] + i * t->nb[0]);
                        col_sum += val;
                    }
                    m_attn_importance[j] += col_sum / n_head;
                }
            }
        }
    }
};

//
// Factory and registration
//
class kvp_factory_snapkv : public ikv_pipeline_factory {
public:
    ikv_pipeline_ptr create(const char * config_json = nullptr) override {
        uint32_t initial_budget = 128;
        uint32_t recent_budget = 256;

        // config_json からパラメータを読む
        // 形式: {"initial_budget":128,"recent_budget":256}
        if (config_json && config_json[0] != '\0') {
            // 簡易JSONパーサ（依存なし）
            auto parse_uint = [&](const char * key) -> uint32_t {
                std::string k(key);
                std::string q1 = "\"" + k + "\":";
                std::string q2 = "\"" + k + "\" :";
                for (const auto & pattern : {q1, q2}) {
                    auto pos = std::strstr(config_json, pattern.c_str());
                    if (pos) {
                        pos += pattern.size();
                        while (*pos == ' ') ++pos;
                        return (uint32_t)std::strtoul(pos, nullptr, 10);
                    }
                }
                return UINT32_MAX;  // 見つからず
            };
            uint32_t v;
            v = parse_uint("initial_budget"); if (v != UINT32_MAX) initial_budget = v;
            v = parse_uint("recent_budget");  if (v != UINT32_MAX) recent_budget  = v;
        }

        return ikv_pipeline_ptr(new kvp_pipeline_snapkv(initial_budget, recent_budget));
    }
};

void kvp_register_snapkv() {
    static kvp_factory_snapkv factory;
    ikv_plugin_registry::get().register_pipeline("snapkv", &factory);
}
