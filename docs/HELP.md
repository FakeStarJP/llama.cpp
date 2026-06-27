# KV Cache Pipeline — 実運用ガイド

llama.cpp KV Cache パイプラインフレームワークの**構築・実行・検証**を網羅する。
設計の詳細は [`kv_pipeline.md`](./kv_pipeline.md) を参照。

---

## 目次

1. [前提環境](#前提環境)
2. [ビルド](#ビルド)
3. [クイックスタート](#クイックスタート)
4. [SnapKV — 実際の挙動とパラメータ](#snapkv--実際の挙動とパラメータ)
5. [KVTC — 現状と制約](#kvtc--現状と制約)
6. [C API リファレンス](#c-api-リファレンス)
7. [動的プラグイン（.dll / .so）](#動的プラグインdll--so)
8. [テストの実行](#テストの実行)
9. [新しいパイプラインの追加手順](#新しいパイプラインの追加手順)
10. [トラブルシューティング](#トラブルシューティング)
11. [スレッドセーフティ](#スレッドセーフティ)
12. [FAQ](#faq)

---

## 前提環境

| 項目 | 要件 |
|------|------|
| OS | Windows 10/11（MSVC 2022 推奨）または Linux |
| CMake | ≥ 3.16 |
| C++ コンパイラ | C++17 対応（MSVC 19.3+, GCC 9+, Clang 10+） |
| CUDA（GPU ビルド時） | CUDA Toolkit 12.x（`GGML_CUDA=ON`） |
| モデル（テスト用） | `models/VibeThinker-3B-heretic_decensored.i1-Q6_K.gguf` |
| RTTI | 必須（`dynamic_cast` でパイプライン→KVキャストを解決） |

RTTI が無効化されているビルド（`-fno-rtti` / `/GR-`）では selection が動作しない。

---

## ビルド

### CPU のみ

```bat
cd C:\Users\msn\Scripts\Projects\Exllama
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

### CUDA 付き

```bat
cmake -B build_cuda -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build_cuda --config Release
```

### 生成されるバイナリ

| パス（CUDA ビルド時） | 目的 |
|---|---|
| `build_cuda\bin\Release\llama-bench.exe` | ベンチマーク（`--kv-*` フラグ付き） |
| `build_cuda\bin\Release\llama-cli.exe` | チャット CLI（`--kv-*` フラグ付き） |
| `build_cuda\bin\Release\test-kv-pipeline-regression.exe` | 回帰テスト |
| `build_cuda\bin\Release\test-kv-pipeline-snapkv.exe` | SnapKV ユニットテスト |
| …他 5 つの `test-kv-pipeline-*` |  スレッド・ストレス・パフォーマンス等 |

### ソース構成

```
src/llama-kv-pipeline.cpp         — レジストリ・ファクトリ・C API
src/llama-kv-pipeline-snapkv.cpp  — SnapKV selection
src/llama-kv-pipeline-kvtc.cpp    — KVTC placeholder
include/llama-kv-pipeline.h       — 公開 C API
src/llama-kv-pipeline.h           — 内部 C++ インターフェース
```

---

## クイックスタート

### C API から SnapKV を適用

```c
#include "llama.h"
#include "llama-kv-pipeline.h"

int main() {
    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;  // CPU。GPU なら -1 等
    llama_model * model = llama_model_load_from_file("model.gguf", mparams);
    if (!model) return 1;

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx   = 2048;
    cparams.n_batch = 512;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { llama_model_free(model); return 1; }

    /* ★ パイプラインを適用（decode の前に呼ぶ） */
    int rc = llama_kv_pipeline_apply(ctx, "snapkv", NULL);
    if (rc != 0) {
        fprintf(stderr, "snapkv unavailable (rc=%d)\n", rc);
    }

    /* 通常の推論 — パイプラインは透過的に動作 */
    llama_batch batch = llama_batch_get_one(tokens, n_tokens);
    llama_decode(ctx, batch);

    /* KV 統計を確認 */
    printf("KV used=%u / total=%u\n",
           llama_kv_cache_get_used(ctx),
           llama_kv_cache_get_size(ctx));

    llama_kv_pipeline_free(ctx);   // idempotent
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
```

### パイプラインの切替え

```c
llama_kv_pipeline_apply(ctx, "snapkv", NULL);   // snapkv で開始
llama_decode(ctx, batch1);

llama_kv_pipeline_apply(ctx, "kvtc", NULL);     // kvtc に切替（snapkv は自動解放）
llama_decode(ctx, batch2);

llama_kv_pipeline_free(ctx);                    // パイプライン解除
```

### 並行比較

1 つの `llama_context` にはアクティブなパイプラインは 1 つ。比較には別々のコンテキストを使う:

```c
llama_context * ctx_snap = llama_init_from_model(model, cparams);
llama_context * ctx_base = llama_init_from_model(model, cparams);

llama_kv_pipeline_apply(ctx_snap, "snapkv", NULL);
// ctx_base には適用しない（baseline）

llama_decode(ctx_snap, batch);
llama_decode(ctx_base, batch);

printf("snapkv used=%u, baseline used=%u\n",
       llama_kv_cache_get_used(ctx_snap),
       llama_kv_cache_get_used(ctx_base));
```

---

## SnapKV — 実際の挙動とパラメータ

### アルゴリズム

[SnapKV](https://arxiv.org/abs/2404.14469) に基づく **attention-score-based トークン選択**。

1. **eval callback フック**: グラフ計算中に `kq_soft_max` テンソルをキャプチャ
2. **importance 蓄積**: 各トークンのアテンション確率行列の列和を per-token importance として蓄積
3. **selection**:
   - 常に **先頭 `initial_budget` トークン** を保持（アンカー）
   - 残りスロットを **importance スコア上位** で選択
4. 初回 decode（スコア未蓄積）はスライディングウィンドウにフォールバック

### デフォルト値

| パラメータ | デフォルト | 影響 | CLI 指定 |
|-----------|-----------|------|---------|
| `initial_budget` | 128 | 先頭に保持するトークン数（常に保持） | `--kv-selection-config '{"initial_budget":64}'` |
| `recent_budget` | 256 | 追加で保持するトークン数（スコア上位から選択） | `--kv-selection-config '{"recent_budget":128}'` |
| **合計バジェット** | **384** | これ以下のプロンプトでは prune しない | |

### 引数例

```bat
:: デフォルト（128+256=384）
llama-cli.exe -m model.gguf --kv-selection snapkv -ngl 99

:: 小さいバジェット（64+128=192、より積極的に prune）
llama-cli.exe -m model.gguf --kv-selection snapkv --kv-selection-config "{\"initial_budget\":64,\"recent_budget\":128}" -ngl 99

:: 大きいバジェット（256+512=768、保守的に prune）
llama-cli.exe -m model.gguf --kv-selection snapkv --kv-selection-config "{\"initial_budget\":256,\"recent_budget\":512}" -ngl 99
```

### prune 効果

| プロンプト長 | baseline `used` | snapkv `used` | 削減 |
|-------------|----------------|---------------|------|
| 400 トークン | 400 | 384 | 16 セル（4%） |
| 512 トークン | 512 | 384 | 128 セル（25%） |

> `n_tokens > 384` の場合のみ prune が発生。短いプロンプトでは baseline と同一。

### ホットパスでの動作

1. `llama_decode()` → `llama_context::decode()`
2. グラフ構築時: パイプラインの eval callback が `kq_soft_max` をキャプチャ
3. `get_memory()->init_batch(...)` → パイプラインの `init_batch` が呼ばれる
4. SnapKV の `init_batch` → 蓄積スコアに基づき `select()` で selection plan を計算
5. plan を `llama_kv_cache_context::m_selection_plan` に設定
6. `apply_ubatch()` → `cell_indices[i] == -1` のトークンをスキップし、KV セルに書き込まない

### attention スコアの取得メカニズム

- `ikv_pipeline::on_eval_tensor(ggml_tensor * t)` 仮想メソッド
- `ikv_pipeline::get_eval_callback()` / `get_eval_callback_user_data()` で eval callback を提供
- `llama_context::process_ubatch` でユーザー callback とパイプライン callback をチェイン
- パディング対応: テンソルデータは `nb[]`（byte stride）を使ってアクセス

---

## KVTC — 実際の挙動とパラメータ

### アルゴリズム

[KVTC](https://arxiv.org/abs/2511.01815) に基づく **隣接トークンマージ圧縮**。

1. **全トークンを selection で保持**（KVTC 自身は selection しない）
2. **importance スコアの差が最小の隣接ペア** を特定（スコアがない場合は位置距離で代用）
3. ペアを **加重平均で 1 トークンにマージ**（2:1 圧縮）
4. マージレート `merge_rate` でマージ対数を制御

### デフォルト値

| パラメータ | デフォルト | 影響 | CLI 指定 |
|-----------|-----------|------|---------|
| `merge_rate` | 0.5 | 保持トークンの何割を隣接マージするか | `--kv-compression-config '{"merge_rate":0.3}'` |

### 引数例

```bat
:: デフォルト（merge_rate=0.5）
llama-cli.exe -m model.gguf --kv-compression kvtc -ngl 99

:: マージ率 30%（KV 削減 ~30%）
llama-cli.exe -m model.gguf --kv-compression kvtc --kv-compression-config "{\"merge_rate\":0.3}" -ngl 99

:: マージ率 80%（KV 削減 ~80%）
llama-cli.exe -m model.gguf --kv-compression kvtc --kv-compression-config "{\"merge_rate\":0.8}" -ngl 99
```

### 効果

| プロンプト長 | baseline `used` | kvtc `used` | 削減 |
|-------------|----------------|-------------|------|
| 200 トークン | 200 | ~100 | 約50%（merge_rate=0.5） |

### マージの仕組み

1. `kq_soft_max` をキャプチャして per-token importance を計算
2. importance 差が小さい隣接ペアを優先的にマージ
3. マージ先セルに両トークンの KV を加重平均でパック
4. selection plan でマージ相手（右側）を dropped とマーク

### attention スコアの取得

SnapKV と同じメカニズム（`on_eval_tensor` + eval callback チェイン）。

---

## C API リファレンス

### ヘッダ

```c
#include "llama-kv-pipeline.h"   // llama.h とは別。明示的に include が必要
```

### 関数

#### `llama_kv_pipeline_apply`

```c
int32_t llama_kv_pipeline_apply(
    struct llama_context * ctx,
    const char * name,         // "baseline", "snapkv", "kvtc"
    const char * config_json   // NULL 推奨（将来用）
);
```

| 戻り値 | 意味 |
|--------|------|
| `0` | 成功 |
| `-1` | `ctx` または `name` が NULL |
| `-2` | 未登録のパイプライン名 |

再呼び出しで前のパイプラインを置き換え。`llama_decode` 前に呼ぶこと。

#### `llama_kv_pipeline_apply_dynamic`

```c
int32_t llama_kv_pipeline_apply_dynamic(
    struct llama_context * ctx,
    const char * path,         // .dll / .so / .dylib へのパス
    const char * config_json
);
```

動的プラグインを読み込んで適用。ABI バージョン不一致の場合は失敗。

#### `llama_kv_pipeline_free`

```c
void llama_kv_pipeline_free(struct llama_context * ctx);
```

パイプラインを解除して baseline に復帰。idempotent（未適用時の呼び出しも安全）。

#### `llama_kv_pipeline_list_names`

```c
int32_t llama_kv_pipeline_list_names(
    const char *** names,
    size_t * n_names
);
```

組み込みパイプライン名の一覧。戻り値は `llama_kv_pipeline_names_free` で解放。

---

## 動的プラグイン（.dll / .so）

### プラグインが満たすべき export

| 関数 | 役割 |
|------|------|
| `llama_kv_pipeline_abi_version()` | `LLAMA_KVPIPEPLUGIN_VERSION_ABI`（現在 `1`）を返す |
| `llama_kv_pipeline_init(pipe, config, reserved, reserved_sz)` | `ikv_pipeline` を生成して `*pipe` にセット。成功で `0` |
| `llama_kv_pipeline_free(pipe)` | インスタンスを破棄 |

### ビルド（Windows, MSVC）

```bat
cl /EHsc /GR /I ..\include /I ..\src my_plugin.cpp /LD /Fe:my_plugin.dll
```

`/GR`（RTTI 有効化）が必須。省略すると `dynamic_cast` が失敗し selection が動作しない。

### ビルド（Linux）

```bash
g++ -shared -fPIC -frtti \
    -I ../include -I ../src \
    my_plugin.cpp -o my_plugin.so
```

### 読み込み

```c
int rc = llama_kv_pipeline_apply_dynamic(ctx, "./my_plugin.dll", NULL);
```

### 制約

- ライブラリハンドルはプロセス終了までアンロードされない
- プラグインは `llama_kv_pipeline_init` で 1 インスタンスのみ返す

---

## テストの実行

### テストスイート（7 ターゲット）

| ターゲット | 内容 | モデル必要 |
|-----------|------|-----------|
| `test-kv-pipeline-snapkv` | SnapKV ユニット + 統合 | ○ |
| `test-kv-pipeline-regression` | 回帰（KV size 不変 / snapkv prune 効果） | ○ |
| `test-kv-pipeline-plugin` | 動的プラグイン互換性 | ○ |
| `test-kv-pipeline-thread` | レジストリのスレッドセーフティ | ○ |
| `test-kv-pipeline-stress` | 100 回 apply/detach サイクル | ○ |
| `test-kv-pipeline-perf` | ベースラインとのオーバーヘッド比較 | ○ |
| `test-kv-pipeline-backend-parity` | CPU / CUDA 結果一致 | ○ |

### 実行コマンド（CPU ビルド）

```bat
cd build
ctest -R test-kv-pipeline --output-on-failure
```

### 実行コマンド（CUDA ビルド）

```bat
cd build_cuda
ctest -R test-kv-pipeline -C Debug --output-on-failure
```

MSVC マルチ設定ビルドでは `-C Debug` または `-C Release` が必須。

### モデルパス

テストは `${PROJECT_SOURCE_DIR}/models/VibeThinker-3B-heretic_decensored.i1-Q6_K.gguf` の存在を `CMake` の `EXISTS` チェックで前提。モデルがない場合、テストターゲット自体が生成されない。

### 個別テストの直接実行

```bat
build_cuda\bin\Debug\test-kv-pipeline-regression.exe
```

終了コード `0` = 全パス、`1` = 1 つ以上失敗。

---

## 新しいパイプラインの追加手順

### 1. ステージの実装

`src/llama-kv-pipeline-myalgo.cpp`:

```cpp
#include "llama-kv-pipeline.h"
#include "llama-kv-cache.h"  // dynamic_cast 用

class kv_selection_myalgo : public ikv_selection {
public:
    explicit kv_selection_myalgo(uint32_t budget) : m_budget(budget) {}

    ikv_selection_plan select(
            const llama_ubatch & ubatch,
            uint32_t n_cells_available,
            const llama_kv_cell_ext * cell_ext_hint) override {
        // TODO: ここにアルゴリズムを実装
        ikv_selection_plan plan;
        plan.ok = false;  // 未実装時はフォールバック
        return plan;
    }

    const char * name() const override { return "myalgo"; }
private:
    uint32_t m_budget;
};
```

### 2. パイプラインラッパー

selection plan を `init_batch` で計算し、context の `m_selection_plan` に設定する（SnapKV の実装を参考）。

### 3. ファクトリと登録

```cpp
class kvp_factory_myalgo : public ikv_pipeline_factory {
public:
    ikv_pipeline_ptr create() override {
        return ikv_pipeline_ptr(new kvp_pipeline_myalgo(128));
    }
};

void kvp_register_myalgo() {
    static kvp_factory_myalgo factory;
    ikv_plugin_registry::get().register_pipeline("myalgo", &factory);
}
```

### 4. レジストリに追加

`src/llama-kv-pipeline.cpp` の `ensure_factories_registered()` に:

```cpp
extern void kvp_register_myalgo();
// ... 本体で ...
kvp_register_myalgo();
```

### 5. CMake に追加

`src/CMakeLists.txt` のソースリストに `llama-kv-pipeline-myalgo.cpp` を追加。

---

## トラブルシューティング

### `llama_kv_pipeline_apply` が -2 を返す

未登録の名前。`llama_kv_pipeline_list_names` で一覧を確認。

### SnapKV を適用したが `get_used` が減らない

プロンプト長が `initial_budget + recent_budget`（デフォルト 384）以下の場合、prune は発生しない。400 トークン以上で検証すること。

### `dynamic_cast` が nullptr を返す

RTTI が無効化されているビルド（`/GR-` または `-fno-rtti`）。MSVC では `/GR`、GCC では `-frtti` を有効化してリビルド。

### テストがモデル不在で SKIP になる

`models/VibeThinker-3B-heretic_decensored.i1-Q6_K.gguf` をプロジェクトルートの `models/` に配置。

### CMake でテストターゲットが生成されない

CMake の `if (EXISTS ...)` がモデルファイルを見つけられなかった。`models/` ディレクトリとファイル名を確認し、CMake を再構成。

### CUDA ビルドでリンクエラー

`llama-kv-pipeline.cpp` は `llama` 共有ライブラリに含まれる。`GGML_CUDA=ON` でビルド済みの `llama.dll` / `llama.lib` にリンクされていることを確認。

---

## スレッドセーフティ

| 操作 | 安全？ | 条件 |
|------|--------|------|
| 複数スレッドがそれぞれ独自の `llama_context` を所有 | ✅ | — |
| `llama_kv_pipeline_apply` / `_free` を所有スレッドから呼ぶ | ✅ | — |
| レジストリの同時クエリ（`list_names` / `instantiate`） | ✅ | `std::atomic<bool>` で同期 |
| 1 つの `llama_context` を複数スレッドで共有 | ❌ | — |
| 同じ `llama_context` に並行 `apply` | ❌ | — |

---

## FAQ

### パイプラインを使わないと既存コードは壊れる？

いいえ。`llama_kv_pipeline_apply` を呼ばない限り、全て従来と同じ動作。`baseline` パイプラインは「何もしない」と等価。

### パイプラインは推論速度に影響する？

selection 判断は `init_batch()`（スケジューラ段階）で完結。アテンションカーネルは変更なし。`baseline` ではオーバーヘッド実質ゼロ。

### 1 コンテキストに複数パイプラインを適用できる？

いいえ。常にアクティブなのは 1 つ。再 `apply` で前のものは解放される。

### SnapKV はいつ prune する？

`n_tokens > 384`（デフォルト `initial_budget(128) + recent_budget(256)`）のときのみ。それ以下では baseline と同一。

### SnapKV は論文通りに動く？

はい。attention スコアに基づく重要度推定を実装済み。`kq_soft_max` テンソルを eval callback でキャプチャし、per-token importance（列和）を蓄積して選択に用いる。初回 decode のみスライディングウィンドウにフォールバック。

### KVTC はデータを圧縮する？

はい。隣接トークンペアを importance 差に基づいて加重平均でマージし、KV セル数を実削減する。デフォルト `merge_rate=0.5` で約50%の KV 削減。

### プラグインがクラッシュしたら？

`llama_kv_pipeline_init` が失敗すれば `apply_dynamic` が非零を返し、既存推論は継続。
