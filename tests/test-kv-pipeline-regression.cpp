// test-kv-pipeline-regression.cpp — Regression tests for the KV pipeline framework.
//
// Copyright (c) 2025 llama.cpp contributors.
// Licensed under the MIT License.
//
// Regression tests verify that existing behaviour is preserved across changes.
// These tests verify that applying and detaching a pipeline does not change
// the underlying KV cache size.

#include "llama.h"
#include "llama-kv-pipeline.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static int test_pipeline_apply_detach_preserves_kv_size() {
    printf("test_pipeline_apply_detach_preserves_kv_size: pipeline apply/remove preserves KV size\n");

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;

    llama_model * model = llama_model_load_from_file(
        "../models/VibeThinker-3B-heretic_decensored.i1-Q6_K.gguf", mparams);
    if (!model) {
        fprintf(stderr, "SKIP: model not available\n");
        return 0;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx   = 512;
    cparams.n_batch = 512;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "FAIL: could not create context\n");
        llama_model_free(model);
        return 1;
    }

    // Record KV cache size BEFORE applying any pipeline.
    uint32_t size_before = llama_kv_cache_get_size(ctx);
    if (size_before == 0) {
        fprintf(stderr, "FAIL: KV size is 0 before pipeline apply\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    // Apply snapkv.
    int rc = llama_kv_pipeline_apply(ctx, "snapkv", nullptr);
    if (rc != 0) {
        fprintf(stderr, "FAIL: snapkv apply failed (rc=%d)\n", rc);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    // REGRESSION: KV cache size must be queryable while pipeline is attached.
    uint32_t size_with_snapkv = llama_kv_cache_get_size(ctx);
    if (size_with_snapkv != size_before) {
        fprintf(stderr, "FAIL: KV size changed while snapkv attached (%u -> %u)\n",
                size_before, size_with_snapkv);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    // Apply kvtc (replaces snapkv).
    rc = llama_kv_pipeline_apply(ctx, "kvtc", nullptr);
    if (rc != 0) {
        fprintf(stderr, "FAIL: kvtc apply failed (rc=%d)\n", rc);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    // REGRESSION: KV cache size must be queryable while kvtc is attached.
    uint32_t size_with_kvtc = llama_kv_cache_get_size(ctx);
    if (size_with_kvtc != size_before) {
        fprintf(stderr, "FAIL: KV size changed while kvtc attached (%u -> %u)\n",
                size_before, size_with_kvtc);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    // Detach pipeline.
    llama_kv_pipeline_free(ctx);

    // Verify KV cache size is unchanged after detach.
    uint32_t size_after = llama_kv_cache_get_size(ctx);
    if (size_after != size_before) {
        fprintf(stderr, "FAIL: KV size changed after pipeline apply/detach (%u -> %u)\n",
                size_before, size_after);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    printf("  OK: size=%u (unchanged across apply/remove, also queryable mid-pipeline)\n", size_before);

    llama_free(ctx);
    llama_model_free(model);
    return 0;
}

static int test_repeated_apply_detach() {
    printf("test_repeated_apply_detach: repeated pipeline apply/detach does not leak\n");

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;

    llama_model * model = llama_model_load_from_file(
        "../models/VibeThinker-3B-heretic_decensored.i1-Q6_K.gguf", mparams);
    if (!model) {
        fprintf(stderr, "SKIP: model not available\n");
        return 0;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx   = 512;
    cparams.n_batch = 512;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "FAIL: could not create context\n");
        llama_model_free(model);
        return 1;
    }

    uint32_t size_initial = llama_kv_cache_get_size(ctx);

    // Apply and detach 10 times.
    for (int i = 0; i < 10; ++i) {
        int rc = llama_kv_pipeline_apply(ctx, "snapkv", nullptr);
        if (rc != 0) {
            fprintf(stderr, "FAIL: apply failed on iteration %d (rc=%d)\n", i, rc);
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }
        llama_kv_pipeline_free(ctx);
    }

    uint32_t size_final = llama_kv_cache_get_size(ctx);
    if (size_final != size_initial) {
        fprintf(stderr, "FAIL: KV size changed after repeated apply/detach (%u -> %u)\n",
                size_initial, size_final);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    printf("  OK: size=%u (unchanged after 10 apply/detach cycles)\n", size_initial);

    llama_free(ctx);
    llama_model_free(model);
    return 0;
}

// REGRESSION: snapkv selection must actually prune KV usage.
// Feed a prompt larger than initial_budget + recent_budget (128 + 256 = 384)
// and verify that KV usage with snapkv is less than baseline.
static int test_snapkv_actually_prunes() {
    printf("test_snapkv_actually_prunes: snapkv must reduce KV usage vs baseline\n");

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;

    llama_model * model = llama_model_load_from_file(
        "../models/VibeThinker-3B-heretic_decensored.i1-Q6_K.gguf", mparams);
    if (!model) {
        fprintf(stderr, "SKIP: model not available\n");
        return 0;
    }

    const auto * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx   = 512;
    cparams.n_batch = 512;

    const int n_prompt = 400;  // larger than 128 + 256 = 384
    std::vector<llama_token> tokens(n_prompt);
    for (int i = 0; i < n_prompt; ++i) {
        tokens[i] = std::rand() % n_vocab;
    }

    // --- Baseline: no pipeline ---
    llama_context * ctx_baseline = llama_init_from_model(model, cparams);
    if (!ctx_baseline) {
        fprintf(stderr, "FAIL: could not create baseline context\n");
        llama_model_free(model);
        return 1;
    }

    {
        llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
        if (llama_decode(ctx_baseline, batch) != 0) {
            fprintf(stderr, "FAIL: baseline decode failed\n");
            llama_free(ctx_baseline);
            llama_model_free(model);
            return 1;
        }
    }

    uint32_t used_baseline = llama_kv_cache_get_used(ctx_baseline);
    llama_free(ctx_baseline);

    // --- With snapkv pipeline ---
    // Re-load model (cheap for small model; avoids state leakage).
    llama_model * model2 = llama_model_load_from_file(
        "../models/VibeThinker-3B-heretic_decensored.i1-Q6_K.gguf", mparams);
    if (!model2) {
        fprintf(stderr, "FAIL: second model load failed\n");
        llama_model_free(model);
        return 1;
    }
    llama_context * ctx_snapkv = llama_init_from_model(model2, cparams);
    if (!ctx_snapkv) {
        fprintf(stderr, "FAIL: could not create snapkv context\n");
        llama_model_free(model2);
        llama_model_free(model);
        return 1;
    }

    int rc = llama_kv_pipeline_apply(ctx_snapkv, "snapkv", nullptr);
    if (rc != 0) {
        fprintf(stderr, "FAIL: snapkv apply failed (rc=%d)\n", rc);
        llama_free(ctx_snapkv);
        llama_model_free(model2);
        llama_model_free(model);
        return 1;
    }

    {
        llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
        if (llama_decode(ctx_snapkv, batch) != 0) {
            fprintf(stderr, "FAIL: snapkv decode failed\n");
            llama_free(ctx_snapkv);
            llama_model_free(model2);
            llama_model_free(model);
            return 1;
        }
    }

    uint32_t used_snapkv = llama_kv_cache_get_used(ctx_snapkv);

    llama_kv_pipeline_free(ctx_snapkv);
    llama_free(ctx_snapkv);
    llama_model_free(model2);
    llama_model_free(model);

    printf("  baseline used=%u, snapkv used=%u\n", used_baseline, used_snapkv);

    if (used_snapkv >= used_baseline) {
        fprintf(stderr, "FAIL: snapkv did not prune KV usage (%u >= %u)\n",
                used_snapkv, used_baseline);
        return 1;
    }

    printf("  OK: snapkv pruned %u cells (%u -> %u)\n",
           used_baseline - used_snapkv, used_baseline, used_snapkv);
    return 0;
}

// REGRESSION: kvtc compression must actually prune KV usage.
// KVTC merges adjacent token pairs, reducing the number of used cells.
static int test_kvtc_actually_prunes() {
    printf("test_kvtc_actually_prunes: kvtc must reduce KV usage vs baseline\n");

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;

    llama_model * model = llama_model_load_from_file(
        "../models/VibeThinker-3B-heretic_decensored.i1-Q6_K.gguf", mparams);
    if (!model) {
        fprintf(stderr, "SKIP: model not available\n");
        return 0;
    }

    const auto * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx   = 512;
    cparams.n_batch = 512;

    const int n_prompt = 200;
    std::vector<llama_token> tokens(n_prompt);
    for (int i = 0; i < n_prompt; ++i) {
        tokens[i] = std::rand() % n_vocab;
    }

    // --- Baseline ---
    llama_context * ctx_baseline = llama_init_from_model(model, cparams);
    if (!ctx_baseline) {
        fprintf(stderr, "FAIL: could not create baseline context\n");
        llama_model_free(model);
        return 1;
    }

    {
        llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
        if (llama_decode(ctx_baseline, batch) != 0) {
            fprintf(stderr, "FAIL: baseline decode failed\n");
            llama_free(ctx_baseline);
            llama_model_free(model);
            return 1;
        }
    }

    uint32_t used_baseline = llama_kv_cache_get_used(ctx_baseline);
    llama_free(ctx_baseline);

    // --- With kvtc pipeline ---
    llama_model * model2 = llama_model_load_from_file(
        "../models/VibeThinker-3B-heretic_decensored.i1-Q6_K.gguf", mparams);
    if (!model2) {
        fprintf(stderr, "FAIL: second model load failed\n");
        llama_model_free(model);
        return 1;
    }
    llama_context * ctx_kvtc = llama_init_from_model(model2, cparams);
    if (!ctx_kvtc) {
        fprintf(stderr, "FAIL: could not create kvtc context\n");
        llama_model_free(model2);
        llama_model_free(model);
        return 1;
    }

    int rc = llama_kv_pipeline_apply(ctx_kvtc, "kvtc", nullptr);
    if (rc != 0) {
        fprintf(stderr, "FAIL: kvtc apply failed (rc=%d)\n", rc);
        llama_free(ctx_kvtc);
        llama_model_free(model2);
        llama_model_free(model);
        return 1;
    }

    {
        llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
        if (llama_decode(ctx_kvtc, batch) != 0) {
            fprintf(stderr, "FAIL: kvtc decode failed\n");
            llama_free(ctx_kvtc);
            llama_model_free(model2);
            llama_model_free(model);
            return 1;
        }
    }

    uint32_t used_kvtc = llama_kv_cache_get_used(ctx_kvtc);

    llama_kv_pipeline_free(ctx_kvtc);
    llama_free(ctx_kvtc);
    llama_model_free(model2);
    llama_model_free(model);

    printf("  baseline used=%u, kvtc used=%u\n", used_baseline, used_kvtc);

    if (used_kvtc >= used_baseline) {
        fprintf(stderr, "FAIL: kvtc did not prune KV usage (%u >= %u)\n",
                used_kvtc, used_baseline);
        return 1;
    }

    printf("  OK: kvtc pruned %u cells (%u -> %u)\n",
           used_baseline - used_kvtc, used_baseline, used_kvtc);
    return 0;
}

int main() {
    llama_backend_init();

    int failures = 0;
    failures += test_pipeline_apply_detach_preserves_kv_size();
    failures += test_repeated_apply_detach();
    failures += test_snapkv_actually_prunes();
    failures += test_kvtc_actually_prunes();

    llama_backend_free();

    if (failures == 0) {
        printf("\nAll regression tests passed.\n");
        return 0;
    }
    fprintf(stderr, "\n%d test(s) failed.\n", failures);
    return 1;
}
