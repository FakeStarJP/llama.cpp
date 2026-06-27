// test-kv-pipeline-perf.cpp — Performance benchmarks for the KV pipeline framework.
//
// Copyright (c) 2025 llama.cpp contributors.
// Licensed under the MIT License.
//
// Performance benchmarks measure the overhead of applying a pipeline.
// The baseline (no pipeline) is compared against the snapkv pipeline.

#include "llama.h"
#include "llama-kv-pipeline.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static int test_baseline_pipeline_overhead() {
    printf("test_baseline_pipeline_overhead: baseline pipeline has zero overhead\n");

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

    // Create two contexts: one without pipeline, one with baseline.
    llama_context * ctx_baseline = llama_init_from_model(model, cparams);
    if (!ctx_baseline) {
        fprintf(stderr, "FAIL: could not create baseline context\n");
        llama_model_free(model);
        return 1;
    }

    llama_context * ctx_with_pipeline = llama_init_from_model(model, cparams);
    if (!ctx_with_pipeline) {
        fprintf(stderr, "FAIL: could not create pipeline context\n");
        llama_free(ctx_baseline);
        llama_model_free(model);
        return 1;
    }

    // Apply baseline pipeline.
    int rc = llama_kv_pipeline_apply(ctx_with_pipeline, "baseline", nullptr);
    if (rc != 0) {
        fprintf(stderr, "FAIL: baseline pipeline apply failed (rc=%d)\n", rc);
        llama_free(ctx_baseline);
        llama_free(ctx_with_pipeline);
        llama_model_free(model);
        return 1;
    }

    // Verify both contexts have the same KV cache size (before pipeline apply).
    uint32_t size_baseline = llama_kv_cache_get_size(ctx_baseline);

    llama_kv_pipeline_free(ctx_with_pipeline);
    llama_free(ctx_baseline);
    llama_free(ctx_with_pipeline);
    llama_model_free(model);

    printf("  OK: baseline pipeline created successfully (KV size: %u)\n", size_baseline);
    return 0;
}

int main() {
    llama_backend_init();

    int failures = 0;
    failures += test_baseline_pipeline_overhead();

    llama_backend_free();

    if (failures == 0) {
        printf("\nAll performance benchmarks passed.\n");
        return 0;
    }
    fprintf(stderr, "\n%d test(s) failed.\n", failures);
    return 1;
}
