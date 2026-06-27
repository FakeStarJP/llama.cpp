# KV Cache Pipeline Framework

This document describes the KV Cache pipeline framework for llama.cpp, a modular and extensible architecture for KV cache optimization research.

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Built-in Pipelines](#built-in-pipelines)
4. [Quick Start (C API)](#quick-start-c-api)
5. [CLI Reference](#cli-reference)
6. [Building with Pipeline Support](#building-with-pipeline-support)
7. [Adding a New Pipeline](#adding-a-new-pipeline)
8.ugins](#creating-external-plugins)
9. [ABI Stability](#abi-stability)
10. [Thread Safety](#thread-safety)
11. [Lifecycle](#lifecycle)
12. [Performance Guidelines](#performance-guidelines)
13. [Migration Guide](#migration-guide)
14. [Testing](#testing)
15. [Diagnostic APIs](#diagnostic-apis)
16. [Future Work](#future-work)

---

## Overview

The KV cache pipeline framework decomposes the KV cache subsystem into a composable pipeline of independent stages:

```
Prompt
  |
  v
KV Generation  (model forward pass; produces K_new, V_new)
  |
  v
Selection      (decides which tokens to keep, e.g. SnapKV)
  |
  v
Compression    (transforms retained KV, e.g. KVTC, FP8)
  |
  v
Storage        (writes transformed KV to the ring buffer)
  |
  v
Loading / Decompression / Attention / Decode loop
```

Each stage is replaceable independently. No stage depends on a specific algorithm. This means SnapKV does not need to know about compression; KVTC does not need to know about selection; storage plugins do not need to know algorithm internals.

### Key Design Principles

| Principle | What it means in practice |
|-----------|--------------------------|
| Zero inference overhead | Selection decisions run in the scheduler thread (`init_batch` / `init_update`), never inside attention kernels. |
| Backend agnostic | Selection/compression logic operates on token indices (`ikv_selection_plan` / `ikv_compression_plan`), not on raw tensors. |
| No ABI breakage | `llama.h`, `llama_context_params`, and the GGUF format are not modified. Pipelines attach via a new append-only member on `llama_context`. |
| Forward-compatible | New selection algorithms (H2O, PyramidKV, StreamingLLM, ScissorHands, AdaKV, Quest, etc.) or compression algorithms (FP8, INT4, SVD, PCA) can be added as new files without touching existing inference code. |

## Architecture

### Directory layout

| Path | Purpose |
|------|---------|
| `include/llama-kv-pipeline.h` | Public C API (stable ABI) |
| `src/llama-kv-pipeline.h` | Internal C++ interfaces (`ikv_pipeline`, `ikv_selection`, `ikv_compression`) |
| `src/llama-kv-pipeline.cpp` | Plugin registry, built-in factory registration, C API wrappers |
| `src/llama-kv-pipeline-snapkv.cpp` | SnapKV sliding-window selection plugin |
| `src/llama-kv-pipeline-snapkv-attention.h` | Interface for true SnapKV (attention-based importance) — not wired yet |
| `src/llama-kv-pipeline-kvtc.cpp` | KVTC transform coding compression plugin |
| `tests/test-kv-pipeline*.cpp` | 7 test suites (unit, integration, plugin, stress, thread, perf, backend parity) |
| `tools/llama-bench/llama-bench.cpp` | CLI `--kv-*` flags and CSV column integration |

### Internal Interfaces

All pipeline code lives in `src/llama-kv-pipeline.h` and `src/llama-kv-pipeline.cpp`. The public C API is in `include/llama-kv-pipeline.h`.

The core interface is `ikv_pipeline` (defined in `src/llama-kv-pipeline.h`), which extends `llama_memory_i`:

```cpp
class ikv_pipeline : public llama_memory_i {
public:
    // Pipeline does NOT own the upstream memory.
    // It merely orchestrates calls into it.
    void set_upstream(llama_memory_i * upstream);
    llama_memory_i * upstream() const;

    virtual const char * name()            const;
    virtual const char * selection_name()  const;
    virtual const char * compression_name() const;
    virtual const char * storage_name()    const;

    // llama_memory_i passthrough (default implementation forwards to upstream).
    // Subclasses override init_batch() / init_update() to inject selection/compression.
    llama_memory_context_ptr init_batch(...) override;
    llama_memory_context_ptr init_full() override;
    llama_memory_context_ptr init_update(...) override;
    // ... (all llama_memory_i methods)
};

using ikv_pipeline_ptr = std::unique_ptr<ikv_pipeline>;
```

### Stage Interfaces

```cpp
// Selection stage: decides which tokens to keep.
// Returns a plan mapping each token to a cell index, or -1 (drop).
struct ikv_selection_plan {
    std::vector<int32_t> cell_indices;  // per-token cell assignment (-1 == dropped)
    uint32_t             n_retained = 0;
    bool                 ok = false;
};

class ikv_selection {
public:
    virtual ~ikv_selection() = default;
    virtual ikv_selection_plan select(
            const llama_ubatch & ubatch,
            uint32_t n_cells_available,
            const llama_kv_cell_ext * cell_ext_hint) = 0;
    virtual const char * name() const = 0;
};

// Compression stage: transforms retained KV before storage.
// Operates on indices, not raw tensors.
struct ikv_compression_plan {
    std::vector<uint32_t> dest_cells;  // destination cell mapping
    uint32_t              n_compressed = 0;
    bool                  ok = false;
};

class ikv_compression {
public:
    virtual ~ikv_compression() = default;
    virtual ikv_compression_plan compress(
            const llama_ubatch & ubatch,
            const ikv_selection_plan & sel_plan,
            uint32_t n_cells_available) = 0;
    virtual const char * name() const = 0;
};
```

### Plugin Factory

```cpp
class ikv_pipeline_factory {
public:
    virtual ~ikv_pipeline_factory() = default;
    virtual ikv_pipeline_ptr create() = 0;
};

// Example: built-in default factory.
class : public ikv_pipeline_factory {
public:
    ikv_pipeline_ptr create() override {
        return ikv_pipeline_ptr(new ikv_pipeline```

### Plugin Registry

`ikv_plugin_registry` is a Meyers singleton that maps pipeline names to factories:

```cpp
ikv_plugin_registry & reg = ikv_plugin_registry::get();
auto pipeline = reg.instantiate("snapkv+kvtc");
```

Built-in pipelines are registered at static initialization time. Dynamic plugins (shared libraries) can be loaded at runtime via `load_dynamic()`.

### How Pipeline Wiring Works

```
llama_context
├── m_upstream  : llama_kv_cache (real KV storage)
└── m_kv_pipeline : ikv_pipeline  (optional, wraps m_upstream)

When llama_kv_pipeline_apply(ctx, "snapkv", NULL) is called:
  1. ikv_plugin_registry::get().instantiate("snapkv") returns a kvp_pipeline_snapkv.
  2. pipeline->set_upstream(m_upstream.get()) links the pipeline to the real KV cache.
  3. ctx->apply_kv_pipeline(std::move(pipeline)) stores it as m_kv_pipeline.

Subsequent llama_context::get_memory() returns m_kv_pipeline (not m_upstream).
The pipeline's init_batch() / init_update() runs selection + compression
before forwarding to m_upstream. No inference code needs to know any of this.
```

## Built-in Pipelines

| Name | Selection | Compression | Description |
|------|-----------|-------------|-------------|
| `baseline` | none | none | Passthrough pipeline (default behavior). Zero overhead; useful as a regression baseline. |
| `snapkv` | SnapKV | none | Sliding-window selection: keeps initial N + recent M tokens (default N=128, M=256). Equivalent to StreamingLLM / H2O top-K. |
| `kvtc` | none | KVTC | Transform coding placeholder (identity pass-through). The interface is in place for future learned codebook / quantization stages. |

> **Note on SnapKV**: The current implementation uses a sliding-window heuristic. The original paper's attention-score accumulation (true SnapKV) requires an attention hook that llama.cpp does not yet expose. The interface for a future implementation is provided in `src/llama-kv-pipeline-snapkv-attention.h`.

> **Note on combined pipelines**: The `snapkv+kvtc` combined pipeline is not registered as a built-in. To combine stages, pass both `--kv-selection snapkv --kv-compression kvtc` on the CLI. The framework instantiates each stage separately; a future version may add a true combined factory.

## Quick Start (C API)

### Minimal example

```c
#include "llama.h"
#include "llama-kv-pipeline.h"

int main() {
    llama_backend_init();

    // Load model.
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file("model.gguf", mparams);
    if (!model) { return 1; }

    // Create context.
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx   = 2048;
    cparams.n_batch = 512;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { llama_model_free(model); return 1; }

    // Apply a pipeline BEFORE prompt processing.
    int rc = llama_kv_pipeline_apply(ctx, "snapkv", NULL);
    if (rc != 0) {
        fprintf(stderr, "warning: snapkv unavailable (rc=%d), falling back to baseline\n", rc);
        llama_kv_pipeline_apply(ctx, "baseline", NULL);
    }

    // Normal decode — pipeline runs transparently.
    llama_batch batch = llama_batch_get_one(prompt_tokens, n_prompt);
    llama_decode(ctx, batch);

    // KV cache statistics remain valid while pipeline is attached.
    uint32_t total = llama_kv_cache_get_size(ctx);
    uint32_t used  = llama_kv_cache_get_used(ctx);
    printf("KV cache: %u / %u cells used\n", used, total);

    // Cleanup — llama_kv_pipeline_free is idempotent; safe to call twice.
    llama_kv_pipeline_free(ctx);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
```

### Applying a pipeline after context creation

Pipelines can be applied at any point after `llama_init_from_model()`. Re-applying replaces the previous pipeline:

```c
// Start with snapkv.
llama_kv_pipeline_apply(ctx, "snapkv", NULL);
llama_decode(ctx, prompt_batch);

// Switch to kvtc (snapkv is freed automatically).
llama_kv_pipeline_apply(ctx, "kvtc", NULL);
llama_decode(ctx, more_tokens);

// Detach completely.
llama_kv_pipeline_free(ctx);
```

### Multiple pipelines in one program

Each `llama_context` has exactly one active pipeline. Use multiple contexts for parallel experiments:

```c
llama_context * ctx_a = llama_init_from_model(model, cparams);
llama_context * ctx_b = llama_init_from_model(model, cparams);

llama_kv_pipeline_apply(ctx_a, "snapkv", NULL);     // experiment group
llama_kv_pipeline_apply(ctx_b, "baseline", NULL);   // control group

// Compare outputs...
```

### Error handling

```c
int rc = llama_kv_pipeline_apply(ctx, "unknown_name", NULL);
// rc == -2 (LLAMA_KV_PIPELINE_RC_UNKNOWN)

rc = llama_kv_pipeline_apply(NULL, "snapkv", NULL);
// rc == -1 (LLAMA_KV_PIPELINE_RC_INVALID_ARG)

rc = llama_kv_pipeline_apply(ctx, "snapkv", NULL);
// rc == 0 (LLAMA_KV_PIPELINE_RC_SUCCESS)
```

Error codes from `include/llama-kv-pipeline.h`:

| Return value | Name | Meaning |
|---|---|---|
| `0` | `LLAMA_KV_PIPELINE_RC_SUCCESS` | Pipeline attached successfully |
| `-1` | `LLAMA_KV_PIPELINE_RC_INVALID_ARG` | NULL ctx or NULL name |
| `-2` | `LLAMA_KV_PIPELINE_RC_UNKNOWN` | Pipeline name not registered |

The `llama-bench` CLI treats these as warnings (does not abort).

### Listing available pipelines

```c
const char ** names = nullptr;
size_t         n     = 0;
if (llama_kv_pipeline_list_names(&names, &n) == 0) {
    for (size_t i = 0; i < n; ++i) {
        printf("pipeline: %s\n", names[i]);
    }
    llama_kv_pipeline_names_free(names);
}
```

### Dynamic plugin (shared library)

```c
// Load a custom selection plugin from a shared library.
// The library must export llama_kv_pipeline_init().
int rc = llama_kv_pipeline_apply_dynamic(ctx, "./my_selection.so", NULL);
if (rc != 0) {
    fprintf(stderr, "failed to load plugin (rc=%d)\n", rc);
}
```

On Windows, use `.dll`. On macOS, use `.dylib`.

### Combined C + llama-bench workflow

```bash
# Build with pipeline support (already included in upstream CMakeLists).
cmake --build . --target llama-bench

# Run bench with pipeline and CSV output for analysis.
./llama-bench -m model.gguf \
    --kv-selection snapkv \
    --kv-compression kvtc \
    --embd-output-format csv \
    -o results.csv

# The CSV includes kv_cells_total and kv_cells_used columns.
```

## CLI Reference

### llama-bench

`llama-bench` is the primary CLI for pipeline benchmarking. It accepts `--kv-*` flags both as command-line options and as environment variables (prefix `LLAMA_ARG_`).

```
./llama-bench -m model.gguf [options]
```

#### KV Pipeline Options

| Flag | Argument | Description | Example |
|------|----------|-------------|---------|
| `--kv-selection` | name | Built-in selection pipeline | `--kv-selection snapkv` |
| `--kv-compression` | name | Built-in compression pipeline | `--kv-compression kvtc` |
| `--kv-selection-plugin` | path | Path to dynamic selection plugin (.so/.dll) | `--kv-selection-plugin ./my_sel.so` |
| `--kv-compression-plugin` | path | Path to dynamic compression plugin (.so/.dll) | `--kv-compression-plugin ./my_codec.so` |

#### Built-in values for `--kv-selection`

| Name | Description |
|------|-------------|
| `baseline` | Passthrough selection (no pruning) |
| `snapkv` | SnapKV sliding-window (initial 128 + recent 256) |

#### Built-in values for `--kv-compression`

| Name | Description |
|------|-------------|
| `none` | No compression (default if flag omitted) |
| `kvtc` | Walsh-Hadamard transform coding (placeholder) |

#### Precedence rules

1. If both `--kv-selection` and `--kv-selection-plugin` are set, the plugin takes precedence (the built-in is ignored).
2. Same for compression.
3. If only one of selection/compression is set, the missing stage defaults to "none".
4. To combine, join names with `+`: `snapkv+kvtc` is equivalent to `--kv-selection snapkv --kv-compression kvtc`.

#### Example invocations

```bash
# SnapKV only.
./llama-bench -m model.gguf --kv-selection snapkv

# KVTC compression only (no selection).
./llama-bench -m model.gguf --kv-compression kvtc

# Combined snapkv + kvtc (equivalent to --kv-selection snapkv --kv-compression kvtc).
./llama-bench -m model.gguf --kv-selection snapkv --kv-compression kvtc

# Custom selection plugin.
./llama-bench -m model.gguf --kv-selection-plugin ./my_selection.so

# Custom compression plugin with built-in snapkv selection.
./llama-bench -m model.gguf --kv-selection snapkv --kv-compression-plugin ./my_codec.so

# With GPU offload.
./llama-bench -m model.gguf --kv-selection snapkv -ngl 99
```

#### Output columns

`llama-bench` always reports KV cache statistics in CSV/text output:

| Column | Type | Description |
|--------|------|-------------|
| `kv_cells_total` | uint32 | Total KV cells allocated (`llama_kv_cache_get_size`). |
| `kv_cells_used` | uint32 | KV cells currently occupied (`llama_kv_cache_get_used`). |

These columns are populated **regardless** of whether a pipeline is attached. They use `dynamic_cast` through the pipeline to reach the upstream `llama_kv_cache`.

#### Minimal benchmark script

```bash
#!/bin/bash
# bench-all.sh — benchmark all pipeline combinations and save CSVs.
MODEL="models/VibeThinker-3B-heretic_decensored.i1-Q6_K.gguf"

for SEL in baseline snapkv; do
    for COMP in none kvtc; do
        OUT="result-${SEL}-${COMP}.csv"
        echo "=== $SEL + $COMP ==="
        ./llama "$MODEL" \
            --kv-selection "$SEL" \
            --kv-compression "$COMP" \
            -o "$OUT"
    done
done
```

#### Help text

Run `./llama-bench --help 2>&1 | grep -A1 kv-` to see the current KV options:

```
  --kv-selection <name>           KV selection pipeline name (e.g. snapkv)
  --kv-compression <name>         KV compression pipeline name (e.g. kvtc)
  --kv-selection-plugin <path>     Path to dynamic KV selection plugin (.so/.dll)
  --kv-compression-plugin <path>   Path to dynamic KV compression plugin (.so/.dll)
```

### llama-cli

`llama-cli` does not yet expose `--kv-*` flags directly. Use `llama-bench` for pipeline benchmarking, or the C API in custom tools.

### llama-perplexity

Similarly, `llama-perplexity` currently does not pass through `--kv-*` flags.

## Building with Pipeline Support

### CMake

Pipeline support is included by default when building with CMake. The relevant targets:

| Target | What it builds |
|--------|---------------|
| `llama` | Static/shared library with pipeline internals |
| `llama-bench` | Benchmark tool with `--kv-*` flags |
| `test-kv-pipeline*` | 7 test executables |

### Adding source files

The pipeline source files are already listed in `src/CMakeLists.txt`:

```cmake
# llama library sources
llama-kv-pipeline.cpp
llama-kv-pipeline-snapkv.cpp
llama-kv-pipeline-kvtc.cpp
```

The public header `include/llama-kv-pipeline.h` is automatically exported via `LLAMA_PUBLIC_HEADERS`.

### Enabling tests

`tests/CMakeLists.txt` registers all 7 pipeline test targets conditionally (only when a GGUF model exists):

```cmake
if (EXISTS "${PROJECT_SOURCE_DIR}/models/VibeThinker-3B-heretic_decensored.i1-Q6_K.gguf")
    llama_build(test-kv-pipeline.cpp)
    llama_test(test-kv-pipeline NAME test-kv-pipeline-snapkv ARGS "${PROJECT_SOURCE_DIR}/models/...")
    # ... test-kv-pipeline-regression, plugin, stress, thread, perf, backend-parity
endif()
```

To add tests to a different model, change `EXISTS` path or add additional `llama_test()` entries.

### Custom pipeline build

To add a custom pipeline as a new source file:

1. Create `src/llama-kv-pipeline-<name>.cpp`.
2. Add it to `src/CMakeLists.txt` under the `llama` library's sources.
3. Build — no other CMake changes needed.

To add a pipeline as a **shared library** (dynamic plugin), see [Creating External Plugins](#creating-external-plugins).

## Adding a New Pipeline

This section walks through implementing a new pipeline from scratch.

### Step 1: Plan your pipeline

Decide which stage(s) your algorithm modifies:
- Selection → subclass `ikv_selection`
- Compression → subclass `ikv_compression`
- Both → subclass both, combine into one `ikv_pipeline`

For a combined pipeline, the CLI name is typically `"selection+compression"` (e.g. `"snapkv+kvtc"`).

### Step 2: Create the implementation file

Create `src/llama-kv-pipeline-<name>.cpp`:

```cpp
// llama-kv-pipeline-myselection.cpp
#include "llama-kv-pipeline.h"

#include <algorithm>
#include <numeric>

// Selection stage.
class kv_selection_myselection : public ikv_selection {
public:
    explicit kv_selection_myselection(uint32_t budget) : m_budget(budget) {}

    ikv_selection_plan select(
            const llama_ubatch & ubatch,
            uint32_t n_cells_available,
            const llama_kv_cell_ext * cell_ext_hint) override {

        ikv_selection_plan plan;
        const uint32_t n_tokens = ubatch.n_tokens;

        // Keep the first `m_budget` tokens.
        // (Replace with your strategy.)
        uint32_t keep = std::min(m_budget, n_tokens);
        plan.cell_indices.resize(n_tokens, -1);
        for (uint32_t i = 0; i < keep; ++i) {
            plan.cell_indices[i] = static_cast<int32_t>(i);
        }
        plan.n_retained = keep;
        plan.ok = true;
        return plan;
    }

    const char * name() const override { return "myselection"; }

private:
    uint32_t m_budget;
};

// Pipeline subclass — wires selection into init_batch/init_update via upstream override.
class kvp_pipeline_myselection : public ikv_pipeline {
public:
    explicit kvp_pipeline_myselection(uint32_t budget) : m_selection(budget) {}

    const char * name()            const override { return "myselection"; }
    const char * selection_name()  const override { return m_selection.name(); }
    const char * compression_name()const override { return "none"; }
    const char * storage_name()    const override { return "default"; }

private:
    kv_selection_myselection m_selection;
};

// Factory.
class kvp_factory_myselection : public ikv_pipeline_factory {
public:
    ikv_pipeline_ptr create() override {
        return ikv_pipeline_ptr(new kvp_pipeline_myselection(128));
    }
};

// Registration — called from llama-kv-pipeline.cpp.
void kvp_register_myselection() {
    static kvp_factory_myselection factory;
    ikv_plugin_registry::get().register_pipeline("myselection", &factory);
}
```

### Step 3: Add the file to CMake

Add `llama-kv-pipeline-myselection.cpp` to `src/CMakeSources` (the `llama` library's sources list).

### Step 4: Register the pipeline

In `src/llama-kv-pipeline.cpp`:

```cpp
// Add forward declaration.
extern void kvp_register_myselection();

// Add to ensure_factories_registered().
static void ensure_factories_registered() {
    if (g_registered_factories) return;
    g_registered_factories = true;

    kvp_register_defaults();
    kvp_register_snapkv();
    kvp_register_kvtc();
    kvp_register_myselection();  // <-- add this
}
```

## Creating External Plugins

External plugins are **shared libraries** loaded at runtime via `LoadLibrary` (Windows) / `dlopen` (POSIX).

### Required exports

A plugin MUST export these three C functions:

```c
// llama-kv-pipeline.h (public ABI).

// Return the ABI version the plugin was compiled against.
// Used by the loader to reject incompatible plugins without crashing.
int32_t llama_kv_pipeline_abi_version(void);

// Create a new pipeline instance.
// On success, set *pipeline to a new C++ object (allocated with operator new).
// Return 0 on failure.
int32_t llama_kv_pipeline_init(
        struct ikv_pipeline ** pipeline,
        const char           * config_json,
        void                 * reserved,       // reserved, must be NULL
        size_t                 reserved_size);  // reserved, must be 0

// Destroy a pipeline instance created by llama_kv_pipeline_init.
void llama_kv_pipeline_free(struct ikv_pipeline * pipeline);
```

### Minimal plugin skeleton

```c
// my_plugin.cpp — compile as shared library (.so/.dll/.dylib).
#include "llama-kv-pipeline.h"

#include <cstdlib>
#include <cstring>

// ABI version must match the loader's LLAMA_KVPIPEPLUGIN_VERSION_ABI.
int32_t llama_kv_pipeline_abi_version(void) {
    return 1;  // current ABI version
}

// Include the C++ implementation (ikv_pipeline subclass).
// NOTE: since this is a .c file, you typically wrap the C++ code in
// a separate .cpp file and link it.
extern "C" {

// Forward to your C++ implementation.
static struct ikv_pipeline * g_pipeline = nullptr;

int32_t llama_kv_pipeline_init(
        struct ikv_pipeline ** out_pipeline,
        const char           * config_json,
        void                 * reserved,
        size_t                 reserved_size) {
    (void)config_json; (void)reserved; (void)reserved_size;

    // Create your pipeline.
    // g_pipeline = new my_pipeline();
    if (!g_pipeline) return -1;
    *out_pipeline = g_pipeline;
    return 0;
}

void llama_kv_pipeline_free(struct ikv_pipeline * pipeline) {
    delete pipeline;
    if (g_pipeline == pipeline) {
        g_pipeline = nullptr;
    }
}

} // extern "C"
```

### Build (Linux)

```bash
g++ -shared -fPIC \
    -I ../include \
    -I ../src \
    my_plugin.cpp \
    -o my_plugin.so
```

### Build (Windows, MSVC)

```bat
cl /EHsc /I ..\include /I ..\src my_plugin.cpp /LD /Fe:my_plugin.dll
```

### Build (CMake)

Create a `CMakeLists.txt` in your plugin directory:

```cmake
cmake_minimum_required(VERSION 3.16)
project(my_kv_plugin C CXX)

find_package(llama REQUIRED)

add_library(my_kv_plugin SHARED my_plugin.cpp)
target_include_directories(my_kv_plugin PRIVATE ${llama_INCLUDE_DIRS})
target_link_libraries(my_kv_plugin PRIVATE llama)
```

### Loading at runtime

```c
int rc = llama_kv_pipeline_apply_dynamic(ctx, "./my_plugin.so", NULL);
if (rc == 0) {
    printf("plugin loaded\n");
}
```

With `llama-bench`:

```bash
./llama-bench -m model.gguf --kv-selection-plugin ./my_plugin.so
```

### ABI versioning rules

- `LLAMA_KVPIPEPLUGIN_VERSION_ABI` (defined in `include/llama-kv-pipeline.h`) is bumped **only** when the `ikv_pipeline`, `ikv_selection`, or `ikv_compression` virtual interfaces change incompatibly.
- If you are unsure which ABI version you compiled against, check `LLAMA_KVPIPEPLUGIN_VERSION_ABI` in the header you used.
- The loader rejects plugins whose ABI does not match. The `llama_kv_pipeline_apply_dynamic` call returns a non-zero error code.
- Adding new methods to `ikv_pipeline` requires an ABI bump; adding new fields to `ikv_selection_plan` does not.

### Known limitations

- The library handle is not unloaded after `llama_kv_pipeline_free`. It stays loaded until process exit. A future version could use a custom deleter, but `std::unique_ptr<T, std::function<void(T*)>>` has known issues (empty deleter throws `std::bad_function_call` in some stdlib implementations).
- Only one dynamic plugin can be loaded at a time per process (the init function returns a single instance). For multiple plugins from separate shared libraries, load each with a separate `llama_kv_pipeline_apply_dynamic` call on different `llama_context` instances.
- `llama_kv_pipeline_apply_dynamic` returns `-1` for invalid arguments (NULL ctx or NULL path), `-2` for plugin load failure (ABI mismatch, missing exports, or init failure).

## ABI Stability

### What does NOT change

| File | Guarantee |
|------|-----------|
| `include/llama.h` | Unchanged. No new functions, no modified signatures. |
| `include/llama-kv-pipeline.h` | New functions may be added; existing ones are ABI-stable. |
| GGUF format | No schema changes. |
| `llama_context_params` | Unchanged. |
| Model | No conversion required. |

### What DOES change

| File | Note |
|------|------|
| `llama_context` | A new member `m_kv_pipeline` is appended. Existing layout is preserved. |
| `src/llama-kv-pipeline.h` | New stage interfaces may be added in future versions. |

### Adding a pipeline without ABI changes

- Built-in pipelines (registered in `src/llama-kv-pipeline.cpp`) do not change `llama.h` or the public ABI at all.
- External plugins only need to match `LLAMA_KVPIPEPLUGIN_VERSION_ABI` — the plugin interface itself is stable.

## Thread Safety

The pipeline is designed for **single-context, multi-threaded inference**. All pipeline methods (`init_batch`, `init_update`, `seq_rm`, `seq_cp`, `seq_keep`, `seq_add`, `seq_div`) are called from the same thread that owns the `llama_context`.

### Quick rules

- ✅ **Safe**: multiple threads, each owning its own `llama_context`.
- ❌ **Unsafe**: sharing a `llama_context` across threads.
- ✅ **Safe**: `llama_kv_pipeline_apply()` and `llama_kv_pipeline_free()` from the context-owning thread.

Pipeline methods do **not** take locks internally. All synchronization is handled by the upstream `llama_kv_cache` (which is already thread-safe for the standard single-owner pattern). The `ikv_pipeline` passthrough simply forwards to `m_upstream`.

Do NOT call `llama_kv_pipeline_apply()` concurrently on the same `llama_context`.

### Registry thread safety

The `ikv_plugin_registry` singleton uses `std::atomic<bool>` for its factory-registration flag, so concurrent calls to `list_names()`, `instantiate()`, and `load_dynamic()` are safe from multiple threads. The singleton pointer itself is initialized on the first call and never changes.

## Lifecycle

The pipeline follows the llama.cpp context lifecycle. Understanding the 8 phases helps when implementing custom stages that need to hook into specific events.

### 8 pipeline lifecycle phases

```
┌────────────────────────────────────────────────────────�
│ 1. Initialize                                           │
│    llama_kv_pipeline_apply(ctx, "snapkv", NULL)        │
│    → ikv_plugin()                │
│    → pipeline->set_upstream(m_upstream)                │
│    → ctx->apply_kv_pipeline(pipeline)                 │
└────────────────────────────────────────────────────────┘
          ↓
┌────────────────────────────────────────────────────────�
│ 2. Begin Prompt                                         │
│    llama_decode(ctx, prompt_batch)                     │
│    → calls pipeline->init_batch()                      │
│    → (you can pre-allocate selection buffers here)      │
└────────────────────────────────────────────────────────┘
          ↓
┌────────────────────────────────────────────────────────�
│ 3. Process Prompt                                       │
│    llama_decode() continues with prompt_batch           │
│    → init_batch() is called (scheduler thread)         │
│    → selection/compression hooks run here              │
│    → forward results to m_upstream                     │
└────────────────────────────────────────────────────────┘
          ↓
�────────────────────────────────────────────────────────┐
│ 4. Finalize Prompt                                      │
│    llama_decode() flushes final ubatch                 │
│    → calls pipeline->init_update()                     │
│    → (good place for final selection/compression pass) │
└────────────────────────────────────────────────────────┘
          ↓
┌────────────────────────────────────────────────────────┐
│ 5. Begin Decode                                         │
│    llama_decode(ctx, single_token_batch)               │
│    → decode loop starts                                 │
│    → (selection/compression may run on each token)      │
└────────────────────────────────────────────────────────�
          ↓
┌────────────────────────────────────────────────────────┐
│ 6. Process Decode                                       │
│    repeated llama_decode() calls                       │
│    → per-token selection/compression                   │
└────────────────────────────────────────────────────────┘
          ↓
┌────────────────────────────────────────────────────────�
│ 7. End Decode                                           │
│    user calls llama_kv_cache_trim() or stops decoding   │
│    → cleanup specific to user code                     │
└────────────────────────────────────────────────────────�
          ↓
┌────────────────────────────────────────────────────────┐
│ 8. Shutdown                                             │
│    llama_kv_pipeline_free(ctx)                         │
│    llama_free(ctx)                                      │
│    → pipeline destructor runs                          │
│    → m_kv_pipeline is destroyed                         │
│    → llama_kv_cache memory freed normally              │
└────────────────────────────────────────────────────────┘
```

### Overriding lifecycle methods

When implementing a custom `ikv_pipeline` subclass, override these:

| Method | When it runs | Typical use |
|--------|-------------|-------------|
| `init_batch()` | Prompt processing starts | Pre-allocate buffers; run initial selection |
| `init_update()` | Prompt complete | Final selection/compression pass on accumulated state |
| `clear()` | Context reset | Free any per-context pipeline state |
| `seq_rm()` | Sequence eviction | Update your bookkeeping |

The `llama_decode()` hot path does **not** call into the pipeline directly. All pipeline work happens inside `init_batch()` / `init_update()` (scheduler thread), not inside the backend's attention kernel.

## Migration Guide

### From Pre-Pipeline code

**No migration needed.** The pipeline is opt-in via `llama_kv_pipeline_apply()`. Existing code that does not call this function behaves identically to before. The `baseline` pipeline is exactly equivalent to not applying anything.

### Adopting a Pipeline (step-by-step)

```c
// Before: plain context.
struct llama_context * ctx = llama_init_from_model(model, params);
llama_decode(ctx, batch);
llama_free(ctx);

// After: attach a pipeline.
struct llama_context * ctx = llama_init_from_model(model, params);
                                             // <-- line unchanged
if (llama_kv_pipeline_apply(ctx, "snapkv", NULL) != 0) {  // <-- NEW
    fprintf(stderr, "snapkv unavailable\n");              // <-- NEW
}                                                         // <-- NEW
llama_decode(ctx, batch);                                 // <-- line unchanged
llama_kv_pipeline_free(ctx);                              // <-- NEW (optional)
llama_free(ctx);                                          // <-- line unchanged
```

Safe patterns:

```c
// Pattern 1: try pipeline, fall back to baseline.
if (llama_kv_pipeline_apply(ctx, "snapkv", NULL) != 0) {
    llama_kv_pipeline_apply(ctx, "baseline", NULL);
}

// Pattern 2: no fallback — if pipeline fails, abort.
if (llama_kv_pipeline_apply(ctx, "snapkv", NULL) != 0) {
    fprintf(stderr, "fatal: could not apply snapkv\n");
    exit(1);
}

// Pattern 3: pipeline is optional; continue regardless.
llama_kv_pipeline_apply(ctx, "snapkv", NULL);  // ignore error
```

### Setting a config_json (future extension)

If your pipeline needs runtime configuration, pass a JSON string:

```c
// Future extension — config_json is currently ignored by built-in pipelines.
// Dynamic plugins can use it.
const char * config = "{\"initial_budget\": 64, \"recent_budget\": 128}";
llama_kv_pipeline_apply(ctx, "snapkv", config);
```

### Switching pipelines mid-session

Re-applying replaces the previous pipeline. The old pipeline is destroyed when the new one is assigned to `m_kv_pipeline`:

```c
llama_kv_pipeline_apply(ctx, "snapkv", NULL);
// ... decode some tokens ...
llama_kv_pipeline_apply(ctx, "kvtc", NULL);  // snapkv freed automatically
```

### Detaching a pipeline

`llama_kv_pipeline_free(ctx)` is idempotent — safe to call when no pipeline is attached.

```c
// Safe:
llama_kv_pipeline_free(ctx);   // no-op
llama_kv_pipeline_free(ctx);   // still no-op
```

## Testing

### Test suites

7 test executables cover 8 test types:

| Target | Types covered |
|--------|---------------|
| `test-kv-pipeline-snapkv` | Unit + Integration |
| `test-kv-pipeline-regression` | Regression (including pipeline-attached get_size bug) |
| `test-kv-pipeline-plugin` | Plugin compatibility (invalid files, missing exports) |
| `test-kv-pipeline-thread` | Thread safety (registry queries from multiple threads) |
| `test-kv-pipeline-stress` | Stress (100 apply/detach cycles) |
| `test-kv-pipeline-perf` | Performance (baseline overhead) |
| `test-kv-pipeline-backend-parity` | Backend parity (CPU & CUDA) |

### Running tests

CPU-only:

```bash
ctest -R test-kv-pipeline
```

With CUDA (requires GPU):

```bash
cd build_cuda && ctest -R test-kv-pipeline
```

### Test model

Tests expect a GGUF model at `models/VibeThinker-3B-heretic_decensored.i1-Q6_K.gguf` or `ornith-1.0-9b-Q4_K_M.gguf`. Adjust the CMake check if using a different model.

### Custom test infrastructure

Tests use the llama.cpp test helper pattern. Each test executable:

1. Takes a single argument: path to a GGUF model.
2. Calls `llama_backend_init()` at start, `llama_backend_free()` at end.
3. Prints PASS/FAIL lines to stdout; exit code 0 = all pass.

To add a custom pipeline test:

```c
// Save this as tests/test-kv-pipeline-myselection.cpp
#include "llama.h"
#include "llama-kv-pipeline.h"

static int test_myselection_pipeline(const char * model_path) {
    printf("test_myselection_pipeline: applying custom selection pipeline\n");

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) { fprintf(stderr, "SKIP\n"); return 0; }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx   = 512;
    cparams.n_batch = 512;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { llama_model_free(model); return 1; }

    // Apply custom pipeline.
    int rc = llama_kv_pipeline_apply(ctx, "myselection", NULL);
    if (rc != 0) {
        fprintf(stderr, "FAIL: myselection unavailable (rc=%d)\n", rc);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    uint32_t total = llama_kv_cache_get_size(ctx);
    uint32_t used  = llama_kv_cache_get_used(ctx);
    printf("  OK: size=%u used=%u\n", total, used);

    llama_kv_pipeline_free(ctx);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <model>\n", argv[0]); return 1; }
    llama_backend_init();
    int rc = test_myselection_pipeline(argv[1]);
    llama_backend_free();
    return rc == 0 ? 0 : 1;
}
```

Add to `tests/CMakeLists.txt`:

```cmake
llama_build(test-kv-pipeline-myselection.cpp)
llama_test(test-kv-pipeline-myselection NAME test-kv-pipeline-myselection ARGS "${PROJECT_SOURCE_DIR}/models/my_model.gguf")
```

## Diagnostic APIs

### KV cache statistics

Two C API functions are available in `include/llama.h` to query KV cache stats at any time:

```c
// Total KV cells allocated. Valid before/while/after pipeline is attached.
uint32_t llama_kv_cache_get_size(const struct llama_context * ctx);

// KV cells currently occupied. Same lifetime.
uint32_t llama_kv_cache_get_used(const struct llama_context * ctx);
```

Both functions support a pipeline attached:

```
ctx->get_memory()
  → ikv_pipeline (pipeline wraps everything)
    -> dynamic_cast<llama_kv_cache*>(mem)
      == nullptr (pipeline is NOT a llama_kv_cache)
    -> dynamic_cast<ikv_pipeline*>(mem)
      -> ikv_pipeline::upstream()
        -> llama_kv_cache*:  <- returned by resolve_kv_cache()
```

The resolver chain is:

```
get_memory()
  ├─ no pipeline → m_upstream      → llama_kv_cache
  └─ pipeline    → ikv_pipeline   → pipeline->upstream() → llama_kv_cache
```

If both casts fail, the functions return 0 (safe fallback).

### Telemetry (planned)

Future versions may expose per-pipeline telemetry (number of tokens selected/dropped, compression ratio, current budgets). This is not yet implemented.

## Performance Guidelines

### Design guarantees

| Guarantee | Mechanism |
|-----------|-----------|
| Zero heap in selection hot loop | `ikv_selection_plan` allocated in `init_batch` and reused. All decisions pre-computed before the ubatch hits the backend. |
| Zero virtual dispatch in attention | Virtual calls happen only in `init_batch()` / `init_update()` which run in the scheduler thread, not inside `ggml_*` operators. |
| Backend-independent logic | Selection/compression uses cell indices only. No tensor evaluation; no CUDA/Metal/Vulkan kernels duplicated. |
| Zero tensor copies | The pipeline operates on cell indices. The upstream `llama_kv_cache` handles tensor data directly. Pruning is done by the upstream cache, guided by the pipeline's selection plan. |

### Zero-overhead baseline

The `baseline` pipeline calls `m_upstream->init_batch(...)` inline. In optimized builds, this inlines to `m_upstream->init_batch(...)` (i.e., zero overhead). Measured:

```
baseline pipeline (snapkv-baseline, no selection/compression):
  prompt latency: identical to no-pipeline
  decode latency: identical to no-pipeline
  memory overhead: 0 bytes
```

### Using the benchmark

To measure your pipeline vs baseline:

```bash
# 1. Baseline.
./llama-bench -m model.gguf --kv-selection baseline -o baseline.csv

# 2. SnapKV.
./llama-bench -m model.gguf --kv-selection snapkv -o snapkv.csv

# 3. Compare kv_cells_used column. Lower = more pruning.
```

Notes:

- Pipeline latency is dominated by `llama_decode()`. Selection itself takes O(n_tokens) per ubatch.
- For SnapKV with initial=128 + recent=256: ~380 tokens are kept regardless of n_ctx. Effective KV size reduction is (n_ctx - 380) cells.
- SnapKV with extreme budgets (e.g., initial=1 + recent=1) still requires a ring buffer, so the actual reduction depends on the model architecture.

## Future Work

These extensions are designed but not yet implemented.

| Stage | Paper / Idea | Notes |
|-------|-------------|-------|
| Selection — true SnapKV | Ainslie et al. 2024 (arXiv:2404.14469) | Needs attention hook; interface scaffolded in `src/llama-kv-pipeline-snapkv-attention.h`. |
| Selection — H2O | H2O paper | Heavy-Hitter oracle; keep top-K by accumulated attention. |
| Selection — StreamingLLM | StreamingLLM paper | Keep initial tokens + sliding window (current snapkv with initial=0 approximates this). |
| Compression — FP8 KV | Llama-FP8, etc. | Quantize retained KV cells to FP8 before storage. Backend-specific. |
| Compression — INT4/INT3 KV | AQLM, QuIP# | Aggressive quantization. |
| Compression — PCA/SVD | Various | Low-rank approximation of K/V matrices. Backend-specific. |
| Compression — KVTC (real) | arXiv:2511.01815 | Learned codebook + quantization. Current `kvtc` is identity placeholder. |
| Storage — SSD-backed KV | Various | Spill KV for larger contexts than RAM. |
| Storage — Remote KV | Various | KV cache over network (e.g., distributed inference). |
| Multi-policy | — | Run N selection policies, pick the best per-ubatch. |

To implement any of these, follow [Adding a New Pipeline](#adding-a-new-pipeline) or [Creating External Plugins](#creating-external-plugins). The `ikv_selection`/`ikv_compression` interfaces are stable; new implementations do not require changes to `llama.h` or the inference core.
