// test-debug-abort.cpp — Minimal test to locate the abort() source.
//
// Copyright (c) 2025 llama.cpp contributors.
// Licensed under the MIT License.

#include "llama.h"
#include "llama-kv-pipeline.h"

#include <cstdio>
#include <cstdlib>

int main() {
    printf("=== test-debug-abort START ===\n");
    fflush(stdout);

    llama_backend_init();
    printf("llama_backend_init OK\n");
    fflush(stdout);

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;

    printf("Loading model...\n");
    fflush(stdout);
    llama_model * model = llama_model_load_from_file(
        "../models/VibeThinker-3B-heretic_decensored.i1-Q6_K.gguf", mparams);
    printf("Model loaded: %p\n", (void *)model);
    fflush(stdout);

    if (!model) {
        fprintf(stderr, "FAIL: could not load model\n");
        llama_backend_free();
        return 1;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx   = 512;
    cparams.n_batch = 512;

    printf("Creating context...\n");
    fflush(stdout);
    llama_context * ctx = llama_init_from_model(model, cparams);
    printf("Context created: %p\n", (void *)ctx);
    fflush(stdout);

    if (!ctx) {
        fprintf(stderr, "FAIL: could not create context\n");
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    printf("Calling llama_kv_cache_get_size (before pipeline)...\n");
    fflush(stdout);
    uint32_t size_before = llama_kv_cache_get_size(ctx);
    printf("KV cache size: %u\n", size_before);
    fflush(stdout);

    printf("Calling llama_kv_pipeline_apply(snapkv)...\n");
    fflush(stdout);
    int rc = llama_kv_pipeline_apply(ctx, "snapkv", nullptr);
    printf("llama_kv_pipeline_apply returned: %d\n", rc);
    fflush(stdout);

    printf("Calling llama_kv_cache_get_size (after snapkv)...\n");
    fflush(stdout);
    uint32_t size_after = llama_kv_cache_get_size(ctx);
    printf("KV cache size after snapkv: %u\n", size_after);
    fflush(stdout);

    printf("Calling llama_kv_pipeline_free...\n");
    fflush(stdout);
    llama_kv_pipeline_free(ctx);
    printf("Pipeline freed\n");
    fflush(stdout);

    printf("=== test-debug-abort PASSED ===\n");
    fflush(stdout);

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return 0;
}
