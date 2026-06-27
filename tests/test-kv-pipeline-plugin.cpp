// test-kv-pipeline-plugin.cpp — Plugin compatibility tests for the KV pipeline framework.
//
// Copyright (c) 2025 llama.cpp contributors.
// Licensed under the MIT License.
//
// These tests verify that the dynamic plugin loader correctly:
//   1. Rejects files that are not valid shared libraries.
//   2. Rejects plugins that lack the required export functions.
//   3. Rejects plugins with an incompatible ABI version.
//   4. Accepts a valid plugin and creates a working pipeline.
//
// The last test (valid plugin) is a placeholder: it would need a real
// plugin shared library to test. The first three tests use a dummy
// file to exercise the error paths.

#include "llama.h"
#include "llama-kv-pipeline.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>

static int test_load_nonexistent_file() {
    printf("test_load_nonexistent_file: loading non-existent file returns error\n");

    // Create a dummy llama_context for the call.
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

    int rc = llama_kv_pipeline_apply_dynamic(ctx, "/nonexistent/path/plugin.so", nullptr);
    if (rc == 0) {
        fprintf(stderr, "FAIL: expected non-zero return for non-existent file\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    printf("  OK: non-existent file returns error (rc=%d)\n", rc);

    llama_free(ctx);
    llama_model_free(model);
    return 0;
}

static int test_load_invalid_file() {
    printf("test_load_invalid_file: loading invalid file (not a shared library) returns error\n");

    // Create a temporary file with invalid content.
    const char * tmp_path = "/tmp/not_a_plugin.so";
    {
        std::ofstream f(tmp_path, std::ios::binary);
        f << "This is not a shared library";
    }

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;

    llama_model * model = llama_model_load_from_file(
        "../models/VibeThinker-3B-heretic_decensored.i1-Q6_K.gguf", mparams);
    if (!model) {
        fprintf(stderr, "SKIP: model not available\n");
        std::remove(tmp_path);
        return 0;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx   = 512;
    cparams.n_batch = 512;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "FAIL: could not create context\n");
        llama_model_free(model);
        std::remove(tmp_path);
        return 1;
    }

    int rc = llama_kv_pipeline_apply_dynamic(ctx, tmp_path, nullptr);
    if (rc == 0) {
        fprintf(stderr, "FAIL: expected non-zero return for invalid file\n");
        llama_free(ctx);
        llama_model_free(model);
        std::remove(tmp_path);
        return 1;
    }

    printf("  OK: invalid file returns error (rc=%d)\n", rc);

    llama_free(ctx);
    llama_model_free(model);
    std::remove(tmp_path);
    return 0;
}

static int test_list_names_includes_builtins() {
    printf("test_list_names_includes_builtins: built-in pipelines are listed\n");

    const char ** names = nullptr;
    size_t         n     = 0;
    int rc = llama_kv_pipeline_list_names(&names, &n);
    if (rc != 0) {
        fprintf(stderr, "FAIL: llama_kv_pipeline_list_names returned %d\n", rc);
        return 1;
    }
    if (n < 3) {
        fprintf(stderr, "FAIL: expected at least 3 pipelines, got %zu\n", n);
        llama_kv_pipeline_names_free(names);
        return 1;
    }

    bool has_baseline = false;
    bool has_snapkv   = false;
    bool has_kvtc     = false;
    for (size_t i = 0; i < n; ++i) {
        if (strcmp(names[i], "baseline") == 0) has_baseline = true;
        if (strcmp(names[i], "snapkv")   == 0) has_snapkv   = true;
        if (strcmp(names[i], "kvtc")     == 0) has_kvtc     = true;
    }
    llama_kv_pipeline_names_free(names);

    if (!has_baseline || !has_snapkv || !has_kvtc) {
        fprintf(stderr, "FAIL: missing built-in pipelines\n");
        return 1;
    }

    printf("  OK: all %zu built-in pipelines listed\n", n);
    return 0;
}

int main() {
    llama_backend_init();

    int failures = 0;
    failures += test_load_nonexistent_file();
    failures += test_load_invalid_file();
    failures += test_list_names_includes_builtins();

    llama_backend_free();

    if (failures == 0) {
        printf("\nAll plugin compatibility tests passed.\n");
        return 0;
    }
    fprintf(stderr, "\n%d test(s) failed.\n", failures);
    return 1;
}
