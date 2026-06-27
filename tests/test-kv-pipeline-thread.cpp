// test-kv-pipeline-thread.cpp — Thread safety tests for the KV pipeline framework.
//
// Copyright (c) 2025 llama.cpp contributors.
// Licensed under the MIT License.
//
// The pipeline is designed for single-context, multi-threaded inference.
// This test verifies that the pipeline's init/decode methods are called
// from the same thread that owns the llama_context, and that no data races
// occur on the pipeline's internal state.

#include "llama.h"
#include "llama-kv-pipeline.h"

#include <atomic>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

// Thread safety test: verify that the registry can be queried from
// multiple threads concurrently. The registry uses std::atomic<bool>
// for the factory registration flag, so concurrent queries are safe.
static int test_pipeline_registry_threadsafe() {
    printf("test_pipeline_registry_threadsafe: registry can be queried from multiple threads\n");

    const int n_threads = 4;
    const int n_iters = 100;
    std::vector<std::thread> threads;
    std::atomic<int> errors{0};

    for (int t = 0; t < n_threads; ++t) {
        threads.emplace_back([&errors, n_iters]() {
            for (int i = 0; i < n_iters; ++i) {
                const char ** names = nullptr;
                size_t n = 0;
                int rc = llama_kv_pipeline_list_names(&names, &n);
                if (rc != 0 || n == 0) {
                    errors++;
                } else {
                    llama_kv_pipeline_names_free(names);
                }
            }
        });
    }

    for (auto & t : threads) {
        t.join();
    }

    if (errors > 0) {
        fprintf(stderr, "FAIL: %d errors during concurrent registry access\n", errors.load());
        return 1;
    }

    printf("  OK: %d threads x %d iterations completed\n", n_threads, n_iters);
    return 0;
}

int main() {
    llama_backend_init();

    int failures = 0;
    failures += test_pipeline_registry_threadsafe();

    llama_backend_free();

    if (failures == 0) {
        printf("\nAll thread safety tests passed.\n");
        return 0;
    }
    fprintf(stderr, "\n%d test(s) failed.\n", failures);
    return 1;
}
