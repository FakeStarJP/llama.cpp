#pragma once

// llama-kv-pipeline.h — internal KV pipeline interfaces.
// Copyright (c) 2025 llama.cpp contributors.
// Licensed under the MIT License.
//
// This header defines the internal interfaces for the composable KV Cache
// pipeline described in docs/README.md. The public C API lives in
// include/llama-kv-pipeline.h; everything here is C++ only and not part of
// the public ABI.
//
// Pipeline overview (see docs/README.md for the full picture):
//
//   Prompt
//     ↓
//   KV Generation  (model forward pass; produces K_new, V_new)
//     ↓
//   Selection      (decides which tokens to keep, e.g. SnapKV)
//     ↓
//   Compression    (transforms retained KV, e.g. KVTC, FP8)
//     ↓
//   Storage        (writes transformed KV to the ring buffer)
//     ↓
//   Loading / Decompression / Attention / Decode loop

#include "llama-memory.h"
#include "llama-batch.h"
#include "llama-kv-cells.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

//
// ikv_pipeline — composable KV cache pipeline interface.
//
// A pipeline is a llama_memory_i that wraps an upstream llama_memory_i
// (typically a llama_kv_cache). It inserts selection/compression stages
// between KV generation and KV storage. The upstream cache operates as the
// "storage" stage of the pipeline.
//
// The pipeline does NOT own the upstream memory — it merely orchestrates
// calls into it. Ownership remains with llama_context.
//
// Thread safety: all methods must be thread-safe or called from the owning thread.
//
class ikv_pipeline : public llama_memory_i {
public:
    // Default pipeline. Does NOT own upstream — caller (llama_context)
    // keeps the real storage alive.
    ikv_pipeline() = default;

    // Non-copyable, non-movable. The pipeline may hold references to
    // upstream state that should not be relocated mid-inference.
    ikv_pipeline(const ikv_pipeline &)            = delete;
    ikv_pipeline(ikv_pipeline &&)                 = delete;
    ikv_pipeline & operator=(const ikv_pipeline &) = delete;
    ikv_pipeline & operator=(ikv_pipeline &&)      = delete;

    ~ikv_pipeline() override = default;

    // Set the upstream memory (typically llama_kv_cache). Must be called
    // before any init/update/clear/seq_* operation.
    void set_upstream(llama_memory_i * upstream) {
        m_upstream = upstream;
    }

    llama_memory_i * upstream() const { return m_upstream; }

    // Query current pipeline stages for diagnostics / telemetry.
    virtual const char * name()            const { return "baseline"; }
    virtual const char * selection_name()  const { return "none";     }
    virtual const char * compression_name() const { return "none";     }
    virtual const char * storage_name()    const { return "default";  }

    // Attention フック: eval callback から各テンソルが通知される。
    // デフォルトでは何もしない。SnapKV 等はこれをオーバーライドして
    // "kq_soft_max" テンソルから per-token スコアを蓄積する。
    virtual void on_eval_tensor(struct ggml_tensor * t) { (void)t; }

    // パイプラインが独自の eval callback を提供する場合、これを返す。
    // llama_context はユーザー callback とパイプライン callback をチェインする。
    // デフォルトでは nullptr（ callback 不要）。
    virtual ggml_backend_sched_eval_callback get_eval_callback() const { return nullptr; }
    virtual void * get_eval_callback_user_data() const { return nullptr; }

    //
    // llama_memory_i passthrough.
    //
    // The default implementation forwards everything to m_upstream.
    // Subclasses override to inject selection / compression hooks.
    //
    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override {
        return m_upstream->init_batch(balloc, n_ubatch, embd_all);
    }

    llama_memory_context_ptr init_full() override {
        return m_upstream->init_full();
    }

    llama_memory_context_ptr init_update(
            llama_context * lctx, bool optimize) override {
        return m_upstream->init_update(lctx, optimize);
    }

    bool get_can_shift() const override {
        return m_upstream->get_can_shift();
    }

    void clear(bool data) override {
        m_upstream->clear(data);
    }

    bool seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) override {
        return m_upstream->seq_rm(seq_id, p0, p1);
    }

    void seq_cp(
            llama_seq_id seq_id_src,
            llama_seq_id seq_id_dst,
            llama_pos p0,
            llama_pos p1) override {
        m_upstream->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    }

    void seq_keep(llama_seq_id seq_id) override {
        m_upstream->seq_keep(seq_id);
    }

    void seq_add(
            llama_seq_id seq_id,
            llama_pos p0,
            llama_pos p1,
            llama_pos shift) override {
        m_upstream->seq_add(seq_id, p0, p1, shift);
    }

    void seq_div(
            llama_seq_id seq_id,
            llama_pos p0,
            llama_pos p1,
            int d) override {
        m_upstream->seq_div(seq_id, p0, p1, d);
    }

    llama_pos seq_pos_min(llama_seq_id seq_id) const override {
        return m_upstream->seq_pos_min(seq_id);
    }

    llama_pos seq_pos_max(llama_seq_id seq_id) const override {
        return m_upstream->seq_pos_max(seq_id);
    }

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override {
        return m_upstream->memory_breakdown();
    }

    void state_write(
            llama_io_write_i & io,
            llama_seq_id seq_id = -1,
            llama_state_seq_flags flags = 0) const override {
        m_upstream->state_write(io, seq_id, flags);
    }

    void state_read(
            llama_io_read_i & io,
            llama_seq_id seq_id = -1,
            llama_state_seq_flags flags = 0) override {
        m_upstream->state_read(io, seq_id, flags);
    }

protected:
    llama_memory_i * m_upstream = nullptr;
};

using ikv_pipeline_ptr = std::unique_ptr<ikv_pipeline>;

//
// ikv_selection — selection stage interface.
//
// Given the set of tokens scheduled for ingestion, returns a
// selection plan. The plan maps each incoming token index to a KV cell
// index (or marks the token as "dropped"). The actual pruning is performed
// by the pipeline via llama_kv_cache::find_slot/apply_ubatch.
//
struct ikv_selection_plan {
    // for each token i in [0, n_tokens), the cell index assigned, or -1
    // if the token is dropped. size == n_tokens.
    std::vector<int32_t> cell_indices;

    // number of tokens retained after selection.
    uint32_t n_retained = 0;

    bool ok = false;
};

class ikv_selection {
public:
    virtual ~ikv_selection() = default;

    // Compute a selection plan for the given ubatch.
    // Returns an empty plan with ok=false when selection cannot be applied
    // (the pipeline then falls back to identity/keep-all).
    //
    // The returned plan.cell_indices has one entry per input token (size ==
    // ubatch.n_tokens). cell_indices[i] is the cell index to assign to token
    // i, or -1 if the token should be dropped. n_retained is the count of
    // non-dropped tokens.
    //
    // The default implementation keeps every token (identity selection).
    virtual ikv_selection_plan select(
            /*in */ const llama_ubatch & ubatch,
            /*in */ uint32_t n_cells_available,
            /*in */ const llama_kv_cell_ext * cell_ext_hint) {
        (void)ubatch; (void)n_cells_available; (void)cell_ext_hint;
        return {};
    }

    virtual const char * name() const { return "none"; }
};

//
// ikv_compression — compression stage interface.
//
// Given retained token indices, optionally transforms the KV tensors
// *before* storage. The default implementation is identity (no transform).
//
// Compression is backend-agnostic: it operates on the index plan, not on
// raw tensor data. Backend-aware transforms (e.g. quantize-to-FP8 by
// injecting a new staging buffer) are planned for a future phase and are
// not required by the initial baseline.
//
struct ikv_compression_plan {
    // for each retained token i (token index in the ubatch), the destination
    // cell index in the storage. Typically 1:1 with retained tokens; advanced
    // codecs may merge tokens. The pipeline uses this to remap retained
    // tokens to their final storage cells after selection.
    std::vector<uint32_t> dest_cells;

    uint32_t n_compressed = 0;
    bool ok = false;
};

class ikv_compression {
public:
    virtual ~ikv_compression() = default;

    // Produce a compression plan from the selection plan.
    // sel_plan.cell_indices[i] is the cell index assigned to token i
    // (or -1 if the token was dropped by the selection stage).
    // The returned plan's dest_cells[i] is the destination cell index in
    // storage for retained token i. Typically 1:1; advanced codecs may merge.
    virtual ikv_compression_plan compress(
            /*in */ const llama_ubatch & ubatch,
            /*in */ const ikv_selection_plan & sel_plan,
            /*in */ uint32_t n_cells_available) {
        (void)ubatch; (void)sel_plan; (void)n_cells_available;
        return {};
    }

    virtual const char * name() const { return "none"; }
};

//
// ikv_pipeline_factory — ABI-stable entry point for runtime plugin loading.
//
// Dynamic plugins (shared libraries) must export a function with this
// signature:
//
//   extern "C" int32_t llama_kv_pipeline_init(
//       ikv_pipeline ** pipeline,
//       const char   * config_cfg_json,
//       void         * reserved,
//       size_t         reserved_size);
//
// On success, *pipeline must point to an instance allocated with operator new.
// The caller takes ownership and deletes the pointer. On failure, return
// a non-zero error code and leave *pipeline == nullptr.
//
// LLAMA_KVPIPEPLUGIN_VERSION_ABI MUST be bumped whenever the ikv_pipeline or
// stage virtual interfaces change incompatibly.
//
#define LLAMA_KVPIPEPLUGIN_VERSION_ABI 1

//
// ikv_pipeline_factory — factory interface for pipeline creation.
//
// Declared after ikv_pipeline because create() returns ikv_pipeline_ptr.
//
class ikv_pipeline_factory {
public:
    virtual ~ikv_pipeline_factory() = default;
    // config_json が非 NULL の場合、パイプラインのパラメータを JSON から読む。
    // NULL の場合はデフォルトパラメータを使用。
    virtual ikv_pipeline_ptr create(const char * config_json = nullptr) = 0;

    // Convenience: allocate a new pipeline instance on the heap and
    // wrap it in a unique_ptr. Useful for default implementations.
    ikv_pipeline_ptr make() {
        return ikv_pipeline_ptr(new ikv_pipeline());
    }
};

// Provide a fallback factory so built-in "baseline" and placeholder
// pipelines can register without implementing their own class.
class kvp_factory_default : public ikv_pipeline_factory {
public:
    static ikv_pipeline_ptr create_inline() {
        return ikv_pipeline_ptr(new ikv_pipeline());
    }
    ikv_pipeline_ptr create(const char * config_json = nullptr) override {
        (void)config_json;
        return create_inline();
    }
};

using ikv_pipeline_init_fn = int32_t (
        ikv_pipeline ** pipeline,
        const char   * config_cfg_json,
        void         * reserved,
        size_t         reserved_size);

//
// ikv_plugin_registry — static registry for built-in (static) plugins and
// runtime loader for dynamic plugins (via dlopen/LoadLibrary).
//
// This is a singleton accessed via ikv_plugin_registry::get(). The
// registration flag is std::atomic<bool>, so concurrent lookups are safe.
// The singleton pointer itself is initialized once and never changes.
//
class ikv_plugin_registry {
public:
    static ikv_plugin_registry & get();

    // NOTE: const overload removed — callers should acquire a non-const
    // reference via get(). All lookups (instantiate, load_dynamic,
    // list_names) are internally synchronized via std::atomic flags.

    // Register a pipeline factory under a name (e.g. "baseline", "snapkv+kvtc").
    // Overwrites any previous registration with the same name.
    void register_pipeline(const std::string & name, ikv_pipeline_factory * factory);

    // Instantiate a registered pipeline by name. Returns nullptr if unknown.
    // config_json が非 NULL の場合、ファクトリに渡される。
    ikv_pipeline_ptr instantiate(const std::string & name, const char * config_json = nullptr) const;

    // Load a dynamic plugin from a shared object path. Returns nullptr if
    // the plugin is missing, has a mismatched ABI, or fails to initialize.
    // The plugin's ownership is transferred to the caller.
    ikv_pipeline_ptr load_dynamic(const std::string & path) const;

    // List all registered static plugin names (for diagnostics / CLI help).
    std::vector<std::string> list_names() const;

private:
    ikv_plugin_registry() = default;

    struct factory_entry {
        ikv_pipeline_factory * factory;
    };

    std::unordered_map<std::string, factory_entry> m_factories;
};
