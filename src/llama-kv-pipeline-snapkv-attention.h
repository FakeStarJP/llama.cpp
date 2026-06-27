#pragma once

// llama-kv-pipeline-snapkv-attention.h — True SnapKV selection plugin.
//
// Copyright (c) 2025 llama.cpp contributors.
// Licensed under the MIT License.
//
// SnapKV: LLM Knows What You are Looking for Before Generation
// https://arxiv.org/abs/2404.14469
//
// True SnapKV uses accumulated attention scores to estimate token
// importance. In llama.cpp's architecture, the selection stage runs
// inside init_batch(), which does not have access to attention weights.
//
// This header defines the interface for a future attention hook that
// would allow true SnapKV selection. The hook is NOT yet implemented
// in the inference core; this file is a placeholder showing how it would
// integrate.
//
// When the attention hook becomes available, the implementation would:
//   1. Accumulate per-token attention scores during each decode step.
//   2. At eviction time, use the accumulated scores to rank tokens.
//   3. Keep the highest-scoring tokens within the KV budget.

#include "llama-kv-pipeline.h"

#include <cstdint>
#include <vector>

//
// ikv_snapkv_attention — true SnapKV selection using attention scores.
//
// This class implements the ikv_selection interface with attention-score-based
// importance estimation. It requires an attention hook that is not yet
// available in the llama.cpp core.
//
// The class is provided as a reference implementation. It will be activated
// once the attention hook interface is added to llama.cpp.
//

class ikv_snapkv_attention : public ikv_selection {
public:
    ikv_snapkv_attention(uint32_t initial_budget, uint32_t recent_budget)
        : m_initial_budget(initial_budget)
        , m_recent_budget(recent_budget)
    {}

    // Importance scores are updated externally via add_attention_scores().
    // In a real implementation, this would be called by the attention hook.
    void add_attention_scores(const float * scores, uint32_t n_tokens) {
        m_importance_scores.insert(m_importance_scores.end(), scores, scores + n_tokens);
    }

    ikv_selection_plan select(
            const llama_ubatch & ubatch,
            uint32_t n_cells_available,
            const llama_kv_cell_ext * cell_ext_hint) override;

    const char * name() const override { return "snapkv-attention"; }

private:
    uint32_t m_initial_budget;
    uint32_t m_recent_budget;

    // Accumulated importance scores per token position.
    std::vector<float> m_importance_scores;
};
