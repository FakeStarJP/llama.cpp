// llama-kv-pipeline.cpp — KV pipeline loader and default implementation.
//
// Copyright (c) 2025 llama.cpp contributors.
// Licensed under the MIT License.

#include "llama-kv-pipeline.h"
#include "llama-context.h"

#include <atomic>
#include <cassert>
#include <cstring>
#include <string>
#include <vector>

#ifdef _WIN32
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    include <windows.h>
#else
#    include <dlfcn.h>
#endif

//
// built-in pipeline factories
//
// Each built-in pipeline is registered as an ikv_pipeline_factory subclass.
// The registry is populated at static initialization time (before main).
//
// NOTE: any new built-in pipeline must be registered here so that it is
// available via its CLI name (e.g. "snapkv+kvtc").
//
// Registry insertion happens via static initializer. This is safe because
// std::unordered_map::operator[] is constexpr-friendly in C++17.
//
// We use the "static global" pattern as in llama.cpp. The registry itself
// is a singleton (Meyers) to avoid the static initialization order fiasco.
//
// Built-in factories (implemented in their own .cpp files):
//
//   "baseline"        — kvp_factory_default     (passthrough)
//   "snapkv"          — kvp_factory_snapkv
//   "snapkv+kvtc"     — kvp_factory_snapkv_kvtc
//
// The kvp_factory_* classes are的定义 in their respective files.

// Forward declarations of built-in factory registration helpers.
// These are defined in the corresponding pipeline-*.cpp files.
// Implemented in llama-kv-pipeline-snapkv.cpp and
// llama-kv-pipeline-kvtc.cpp respectively.
extern void kvp_register_snapkv();
extern void kvp_register_kvtc();

static void kvp_register_defaults() {
    static kvp_factory_default factory;
    ikv_plugin_registry::get().register_pipeline("baseline", &factory);
}

//
// ikv_plugin_registry
//
static ikv_plugin_registry * g_registry = nullptr;
static std::atomic<bool> g_registered_factories{false};

static void ensure_factories_registered() {
    if (g_registered_factories.load(std::memory_order_acquire)) {
        return;
    }
    g_registered_factories.store(true, std::memory_order_release);

    // NOTE: if a build configuration disables some pipelines (e.g. CUDA
    // specific ones), the registration macros should be guarded here.
    kvp_register_defaults();
    kvp_register_snapkv();
    kvp_register_kvtc();
}

// Note: `g_registry` is a process-global singleton. The pointer itself is
// immutable after the first call. All lookups are const methods, so they
// can be called without a non-const reference.
ikv_plugin_registry & ikv_plugin_registry::get() {
    if (!g_registry) {
        g_registry = new ikv_plugin_registry();
        ensure_factories_registered();
    }
    return *g_registry;
}

void ikv_plugin_registry::register_pipeline(
        const std::string & name,
        ikv_pipeline_factory * factory) {
    m_factories[name] = { factory };
}

ikv_pipeline_ptr ikv_plugin_registry::instantiate(
        const std::string & name, const char * config_json) const {
    ensure_factories_registered();
    auto it = m_factories.find(name);
    if (it == m_factories.end()) {
        return nullptr;
    }
    return it->second.factory->create(config_json);
}

ikv_pipeline_ptr ikv_plugin_registry::load_dynamic(
        const std::string & path) const {

#ifdef _WIN32
    // Windows: LoadLibrary + GetProcAddress
    HMODULE hmod = LoadLibraryA(path.c_str());
    if (!hmod) {
        fprintf(stderr, "llama-kv-pipeline: failed to load plugin '%s' (error %lu)\n",
                path.c_str(), GetLastError());
        return nullptr;
    }

    // Check ABI version first.
    auto * abi_fn = (int32_t (*)()) GetProcAddress(hmod, "llama_kv_pipeline_abi_version");
    if (!abi_fn) {
        fprintf(stderr, "llama-kv-pipeline: plugin '%s' missing llama_kv_pipeline_abi_version\n",
                path.c_str());
        FreeLibrary(hmod);
        return nullptr;
    }
    int32_t plugin_abi = abi_fn();
    if (plugin_abi != LLAMA_KVPIPEPLUGIN_VERSION_ABI) {
        fprintf(stderr, "llama-kv-pipeline: plugin '%s' ABI mismatch "
                "(plugin=%d, expected %d)\n",
                path.c_str(), plugin_abi, LLAMA_KVPIPEPLUGIN_VERSION_ABI);
        FreeLibrary(hmod);
        return nullptr;
    }

    // Get the init function.
    auto * init_fn = (ikv_pipeline_init_fn *) GetProcAddress(hmod, "llama_kv_pipeline_init");
    if (!init_fn) {
        fprintf(stderr, "llama-kv-pipeline: plugin '%s' missing llama_kv_pipeline_init\n",
                path.c_str());
        FreeLibrary(hmod);
        return nullptr;
    }

    ikv_pipeline * pipeline = nullptr;
    int32_t rc = init_fn(&pipeline, nullptr, nullptr, 0);
    if (rc != 0 || !pipeline) {
        fprintf(stderr, "llama-kv-pipeline: plugin '%s' init failed (rc=%d)\n",
                path.c_str(), rc);
        FreeLibrary(hmod);
        return nullptr;
    }

    // NOTE: We do not unload the library here because ikv_pipeline_ptr
    // uses the default deleter. The library remains loaded until process
    // exit. This is a known limitation; a future version could use a
    // custom deleter if ikv_pipeline_ptr is changed to support one.
    FreeLibrary(hmod);  // Decrement the reference count; library stays loaded.
    return ikv_pipeline_ptr(pipeline);

#else
    // POSIX: dlopen + dlsym
    void * handle = dlopen(path.c_str(), RTLD_NOW);
    if (!handle) {
        fprintf(stderr, "llama-kv-pipeline: failed to load plugin '%s' (%s)\n",
                path.c_str(), dlerror());
        return nullptr;
    }

    // Clear any existing error.
    dlerror();

    // Check ABI version.
    auto * abi_fn = (int32_t (*)()) dlsym(handle, "llama_kv_pipeline_abi_version");
    const char * err = dlerror();
    if (err || !abi_fn) {
        fprintf(stderr, "llama-kv-pipeline: plugin '%s' missing llama_kv_pipeline_abi_version (%s)\n",
                path.c_str(), err ? err : "unknown");
        dlclose(handle);
        return nullptr;
    }
    int32_t plugin_abi = abi_fn();
    if (plugin_abi != LLAMA_KVPIPEPLUGIN_VERSION_ABI) {
        fprintf(stderr, "llama-kv-pipeline: plugin '%s' ABI mismatch "
                "(plugin=%d, expected %d)\n",
                path.c_str(), plugin_abi, LLAMA_KVPIPEPLUGIN_VERSION_ABI);
        dlclose(handle);
        return nullptr;
    }

    // Get the init function.
    auto * init_fn = (ikv_pipeline_init_fn *) dlsym(handle, "llama_kv_pipeline_init");
    err = dlerror();
    if (err || !init_fn) {
        fprintf(stderr, "llama-kv-pipeline: plugin '%s' missing llama_kv_pipeline_init (%s)\n",
                path.c_str(), err ? err : "unknown");
        dlclose(handle);
        return nullptr;
    }

    ikv_pipeline * pipeline = nullptr;
    int32_t rc = init_fn(&pipeline, nullptr, nullptr, 0);
    if (rc != 0 || !pipeline) {
        fprintf(stderr, "llama-kv-pipeline: plugin '%s' init failed (rc=%d)\n",
                path.c_str(), rc);
        dlclose(handle);
        return nullptr;
    }

    // Get the free function.
    auto * free_fn = (void (*)(ikv_pipeline *)) dlsym(handle, "llama_kv_pipeline_free");
    err = dlerror();
    if (err || !free_fn) {
        fprintf(stderr, "llama-kv-pipeline: plugin '%s' missing llama_kv_pipeline_free (%s)\n",
                path.c_str(), err ? err : "unknown");
        delete pipeline;
        dlclose(handle);
        return nullptr;
    }

    dlclose(handle);  // Decrement reference count; library stays loaded.
    return ikv_pipeline_ptr(pipeline);
#endif
}

std::vector<std::string> ikv_plugin_registry::list_names() const {
    ensure_factories_registered();
    std::vector<std::string> names;
    names.reserve(m_factories.size());
    for (const auto & kv : m_factories) {
        names.push_back(kv.first);
    }
    return names;
}

//
// C API (include/llama-kv-pipeline.h)
//
// These functions are thin wrappers around the C++ registry. They form
// the public ABI and must remain stable across releases.
//

extern "C" {

struct ikv_pipeline_c {
    // Opaque C handle. The actual pipeline is held inside llama_context
    // as an ikv_pipeline_ptr. This struct exists solely so that the C
    // API can pass around a pointer-sized handle.
    int32_t dummy;
};

LLAMA_API int32_t llama_kv_pipeline_apply(
        struct llama_context * ctx,
        const char * name,
        const char * config_json) {
    if (!ctx || !name) {
        return -1;
    }
    auto pipeline = ikv_plugin_registry::get().instantiate(name, config_json);
    if (!pipeline) {
        return -2;
    }
    // The context takes ownership and wires the pipeline into its
    // memory subsystem. The actual wiring is implemented in
    // llama_context.
    ctx->apply_kv_pipeline(std::move(pipeline));
    return 0;
}

LLAMA_API int32_t llama_kv_pipeline_apply_dynamic(
        struct llama_context * ctx,
        const char * path,
        const char * config_json) {
    if (!ctx || !path) {
        return -1;
    }
    auto pipeline = ikv_plugin_registry::get().load_dynamic(path);
    if (!pipeline) {
        // load_dynamic already printed a diagnostic to stderr.
        return -2;
    }
    ctx->apply_kv_pipeline(std::move(pipeline));
    return 0;
}

LLAMA_API void llama_kv_pipeline_free(struct llama_context * ctx) {
    if (ctx) {
        ctx->apply_kv_pipeline(nullptr);
    }
}

LLAMA_API int32_t llama_kv_pipeline_list_names(
        const char *** names,
        size_t * n_names) {
    if (!names || !n_names) {
        return -1;
    }
    auto list = ikv_plugin_registry::get().list_names();
    const char ** arr = new const char *[list.size() + 1];
    for (size_t i = 0; i < list.size(); ++i) {
        // Duplicate the string so the caller can free the array without
        // depending on the lifetime of the registry's internal strings.
        arr[i] = strdup(list[i].c_str());
    }
    arr[list.size()] = nullptr;
    *names = arr;
    *n_names = list.size();
    return 0;
}

LLAMA_API void llama_kv_pipeline_names_free(const char ** names) {
    if (!names) return;
    for (size_t i = 0; names[i] != nullptr; ++i) {
        free((void *) names[i]);
    }
    delete[] names;
}

} // extern "C"
