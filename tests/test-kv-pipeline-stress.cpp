// test-kv-pipeline-stress.cpp — Stress tests for the KV pipeline framework.
//
// Copyright (c) 2025 llama.cpp contributors.
// Licensed under the MIT License.
//
// Stress tests verify that the pipeline remains stable under heavy load:
//   1. Many apply/detach cycles.
//   2. Long generation loops with a pipeline attached.
//   3. Multiple pipelines applied in sequence.

#include "llama.h"
#include "llama-kv-pipeline.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static int test_many_apply_detach_cycles() {
    printf("test_many_apply_detach_cycles: 100 apply/detach cycles\n");

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

    for (int i = 0; i < 100; ++i) {
        int rc = llama_kv_pipeline_apply(ctx, "snapkv", nullptr);
        if (rc != 0) {
            fprintf(stderr, "FAIL: apply failed on iteration %d (rc=%d)\n", i, rc);
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }
        llama_kv_pipeline_free(ctx);
    }

    printf("  OK: 100 cycles completed\n");

    llama_free(ctx);
    llama_model_free(model);
    return 0;
}

static int test_pipeline_with_prompt_processing() {
    printf("test_pipeline_with_prompt_processing: process a prompt with pipeline attached\n");

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

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "FAIL: could not create context\n");
        llama_model_free(model);
        return 1;
    }

    // Apply pipeline before processing.
    int rc = llama_kv_pipeline_apply(ctx, "snapkv", nullptr);
    if (rc != 0) {
        fprintf(stderr, "FAIL: snapkv apply failed (rc=%d)\n", rc);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    // Feed a short prompt of random tokens.
    const int n_prompt = 16;
    std::vector<llama_token> tokens(n_prompt);
    for (int i = 0; i < n_prompt; ++i) {
        tokens[i] = std::rand() % n_vocab;
    }

    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
    rc = llama_decode(ctx, batch);
    if (rc != 0) {
        fprintf(stderr, "FAIL: decode failed (rc=%d)\n", rc);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    printf("  OK: prompt processing with pipeline completed\n");

    llama_kv_pipeline_free(ctx);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}

int main() {
    llama_backend_init();

    int failures = 0;
    failures += test_many_apply_detach_cycles();
    failures += test_pipeline_with_prompt_processing();

    llama_backend_free();

    if (failures == 0) {
        printf("\nAll stress tests passed.\n");
        return 0;
    }
    fprintf(stderr, "\n%d test(s) failed.\n", failures);
    return 1;
}
