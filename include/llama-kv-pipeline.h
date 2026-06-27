#ifndef LLAMA_KV_PIPELINE_H
#define LLAMA_KV_PIPELINE_H

// llama-kv-pipeline.h — public KV Cache pipeline API.
//
// This header is part of the llama.cpp public API. It is NOT included from
// llama.h; users that want to customize the KV pipeline must include this
// header explicitly.
//
// The API is experimental and may change without notice. All functions are
// guarded by LLAMA_KVPIPEPLUGIN_VERSION_ABI.
//
// Example usage:
//
//   struct llama_context * ctx = llama_init_from_model(...);
//
//   // Attach a pre-built pipeline (e.g. "snapkv+kvtc") to the context.
//   // The pipeline wraps the context's existing KV cache; it does not
//   // replace the underlying storage.
//   llama_kv_pipeline_apply(ctx, "snapkv+kvtc", NULL);
//
//   // ... use llama_decode as usual ...
//
//   llama_kv_pipeline_free(ctx);
//   llama_free(ctx);

#include "llama.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque pipeline handle. The actual pipeline is a C++ object that
// implements the internal ikv_pipeline interface.
typedef struct ikv_pipeline ikv_pipeline;

// Apply a named KV pipeline to the context. The pipeline wraps the
// context's existing KV cache (it does not replace the underlying
// storage). The pipeline is owned by the context and freed with
// llama_kv_pipeline_free() or llama_free().
//
// If config_json is non-NULL, it is a JSON-encoded configuration string
// that is passed to the pipeline factory. Built-in pipelines ignore it.
//
// Returns 0 on success, non-zero on failure (e.g. unknown pipeline name).
LLAMA_API int32_t llama_kv_pipeline_apply(
        struct llama_context * ctx,
        const char * name,
        const char * config_json);

// Apply a pipeline from a dynamic plugin (shared object). The plugin
// must export a function with the signature:
//
//   int32_t llama_kv_pipeline_init(ikv_pipeline **, const char *, void *, size_t);
//
// The path is the filesystem path to the .so/.dll/.dylib. The JSON
// configuration is passed to the plugin's init function.
//
// Returns 0 on success, non-zero on failure.
LLAMA_API int32_t llama_kv_pipeline_apply_dynamic(
        struct llama_context * ctx,
        const char * path,
        const char * config_json);

// Detach and free the pipeline associated with the context. After this
// call, the context reverts to its default (passthrough) KV behavior.
// Safe to call if no pipeline is attached.
LLAMA_API void llama_kv_pipeline_free(
        struct llama_context * ctx);

// List available built-in pipelines. The caller must free the returned
// string list with llama_kv_pipeline_names_free().
//
// On success, *names points to a NULL-terminated array of C strings and
// *n_names is the number of entries (excluding the NULL terminator).
LLAMA_API int32_t llama_kv_pipeline_list_names(
        const char *** names,
        size_t * n_names);

LLAMA_API void llama_kv_pipeline_names_free(
        const char ** names);

//
// Dynamic plugin interface.
//
// External plugins (shared libraries) MUST export the following functions:
//
//   // Return the ABI version the plugin was compiled against.
//   // Must return LLAMA_KVPIPEPLUGIN_VERSION_ABI.
//   int32_t llama_kv_pipeline_abi_version(void);
//
//   // Create a new pipeline instance.
//   // On success, set *pipeline to a new instance (allocated with operator new)
//   // and return 0. On failure, return a non-zero error code.
//   int32_t llama_kv_pipeline_init(
//       ikv_pipeline ** pipeline,
//       const char   * config_json,
//       void         * reserved,
//       size_t         reserved_size);
//
//   // Destroy a pipeline instance created by llama_kv_pipeline_init.
//   // The plugin is responsible for deleting the pipeline and releasing
//   // any resources it owns.
//   void llama_kv_pipeline_free(ikv_pipeline * pipeline);
//
// Plugins that do not export these functions will be rejected by the loader.

#ifdef __cplusplus
}
#endif

#endif // LLAMA_KV_PIPELINE_H
