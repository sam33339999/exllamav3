# 變更紀錄（繁體中文）

本檔記錄 MiaAI Lab 的 ExLlamaV3 fork，相對上游
[turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3)
**v1.4.2**（`5f3c537`，2026-08-11）新增與改動的項目。

對照方式：以該 tag 的檔案樹為基準。上游原本就有的能力維持原樣，說明見
`UPSTREAM_README.md`。其中包含 EXL3 量化、DFlash / DFlash-Laguna 草稿模型、
Qwen3.5 等 MTP 架構、2–8 bit 整數 KV、連續 batching，以及官方建議的
TabbyAPI。版本號仍是 `exllamav3/version.py` 裡的 `1.4.2`。

下列日期對應本倉庫的 commit。

## 2026-10-01 — 推測步數可從環境變數設定

`NUM_DRAFT_TOKENS` 決定每一輪驗證前要草稿幾個 token。留空則沿用各草稿器
自己的預設。

| `DRAFT` | 預設步數 | 這個值在做什麼 |
|---|---|---|
| `mtp` | 4 | Qwen3.8 的 MTP head 一次 forward 預測 1 個 token。伺服器把這個 head 跑這麼多次（greedy），再由目標模型一次驗證整段草稿。4 是 Qwen3.5 MTP 實測較好的窗口（`default_draft_size`）。 |
| `dflash2` | 7（`block_size - 1`） | diffusion block 長度固定。這個值只在驗證前把 block 截短，不要設得比 7 大。 |
| `none` | — | 忽略這個設定。 |

接線：

- `.env` 的 `NUM_DRAFT_TOKENS` 由 `start.sh` 檢查（正整數，至少 1），再傳給
  `tools/serve_openai.py --num_draft_tokens`。
- 伺服器把它編成 `model_init` 的 `-ndt`，並傳進 `Generator(num_draft_tokens=...)`。
  省略時不覆寫，草稿模型的 `default_draft_size` 繼續生效，並跟 `model_init`
  的 `max_history` 對齊。
- `--mtp` 與外部 `--draft_model_dir` 不能同時用。
- 程式庫呼叫端同樣可以對 `Generator` 傳 `num_draft_tokens=`（見 `examples/chat.py`）。

## 2026-09-24 — 24 GB 卡上的 DFlash2 取樣，以及 262k context

給 3090 這類 24 GB 卡、DFlash2 draft 常駐時的服務配方。NVFP4 與 FP8 在這張卡上
配置不出 262144 context；改走上游已有的整數 KV，`-cq 3`（約 3 bit）。`.env`
預設因此是 `CACHE_QUANT=3`、`CONTEXT_SIZE=262144`、`DRAFT=dflash2`。

同一輪加上重複與思考出口，長循環會被取樣壓下去，請求本身不會被中止：

- **DRY**（`SS_DRY`）：exact repeat 比 `DRY_ALLOWED_LENGTH` 更長時降權。
  預設 multiplier 0.2、base 1.75、allowed length 6、range 4096。
- **思考出口**（`SS_ThinkExit`）：`</think>` 還沒合上時，過了 `THINK_BUDGET`
  （預設 4096 個新 token）之後，`</think>` 的 logit 在 `THINK_RAMP`（1536）
  個 token 內線性升到 `THINK_BIAS`（16）。思考階段另有 frequency penalty
  `THINK_FREQ`（0.2）。
- 單 token repetition penalty 預設 1.0（等於關閉）。客戶端明確傳
  `repetition_penalty: 0` 時維持關閉。
- `LOOP_WINDOW` / `LOOP_REPS` 預設不啟用硬停，重複交給上面的 penalty。

## 2026-09-23 — `<think>` 不能吃光 `max_tokens`

Qwen3.8 從 `<think>` 裡面開始。xhigh 思考可以跑上千 token；若把它算進
`max_tokens`，串流會停在思考中段、答案是空的。

- 客戶端沒給 `max_tokens` 時，預設 `DEFAULT_MAX_TOKENS=32768`。思考與答案共用。
- `ANSWER_RESERVE`（預設 512）留給真正的回答。思考把預算用完時，伺服器會
  強制補上 `</think>`。這是最後手段；平常是靠上一節的 think-exit logit 收束。
- `THINKING_BUDGET=0` 表示不另設強制上限，只留 `ANSWER_RESERVE`。
- `REASONING_EFFORT`（`low` / `medium` / `xhigh`，預設 `xhigh`）寫進模板，
  本身不會硬關掉 `</think>`。
- 單次請求可用 `enable_thinking`、`reasoning_effort`、`thinking_budget`、
  `max_tokens` / `max_completion_tokens` 覆寫。
- 長 prefill 或長思考期間，SSE 會送 comment ping，避免代理把連線掐掉。
- 客戶端把 HTTP 連線斷開後，下一個 decode step 會取消生成。

`start.sh` 與 `.env.example` 帶入這些旋鈕。單元測試在
`tests/test_serve_openai.py`、`tests/test_think_exit.py`、
`tests/live_thinking_budget.py`。

## 2026-08-31 — 請求本體上限，過大時回 413

`aiohttp` 的 `web.Application()` 預設 `client_max_size` 是 1 MiB。較長的
對話或工具定義會被拒，而且 `HTTPRequestEntityTooLarge` 是在 `request.json()`
裡面拋出的，先前被收成 400 `invalid JSON`。

- 新旗標 `--max_body_mb`，預設 64 MiB。
- 超過上限回 **413**，`code` 為 `request_entity_too_large`，訊息裡帶設定的上限。
- 真的壞掉的 JSON 仍回 400 `invalid JSON`。

## 2026-08-26 — fork 主體

第一次進這個倉庫的內容是相對於 v1.4.2 的 squash。下面依功能分組。

### 推測解碼：DFlash2 與 DSpark

上游已有 DFlash、DFlash-Laguna，以及 Qwen3.5 / DeepSeek V4 / Hy3 的 MTP head。
本 fork 補上兩種 diffusion draft，並把 MTP 接成服務預設路徑。

**DFlash2**（`exllamav3/architecture/dflash2.py`，模組在
`exllamav3/modules/arch_specific/dflash2.py`）

- 架構字串 `DFlash2DraftModel`。骨幹沿用 DFlash，每個 attention / MLP 外面包
  grouped dynamic convolution，再用 top-k candidate selector 取代逐列 argmax。
- Block 輸入是 `[anchor, mask × (block_size-1)]`。第 1 到第 7 列預測自己的
  位置（z-lab diffusion）。Qwen3.8 這份 draft 的驗證寬度預設是 7。
- 目標模型的 hidden tap 用 layer output（`tap_shift 0`）。
- Greedy selector 的提案與目標逐 token 比對接受，各溫度下輸出邊際與目標分布
  一致。
- `EXL3_SPEC_RS=1` 且 temperature > 0 時，generator 改走與溫度匹配的
  stochastic walk，再用 Leviathan rejection sampling 驗證
  （以 `min(1, p/q)` 接受，拒絕時從 `norm(max(0, p-q))` 重抽）。
  Greedy（T = 0）不受影響。

**DSpark**（`exllamav3/architecture/dspark.py`）

- 架構字串對應 `DSparkDraftModel`（DFlash 骨幹 + Markov head + confidence head），
  例如 RadixArk 的 Qwen3.8-27B-DSpark。
- 每一列都產生草稿：第 j 列預測 anchor+j+1，驗證寬度是 `block_size + 1`。
- Confidence head 把草稿窗口收到「連續都夠有信心」的前綴
  （`params["draft_confidence_len"]`，門檻 `EXL3_DSPARK_CONF`）。

**MTP 服務路徑**

- `tools/serve_openai.py -dm mtp`（以及 `start.sh` 的 `DRAFT=mtp`）使用目標
  checkpoint 裡的 MTP head，約 50 MB，不必再下載一份 draft。
- Draft 的 KV 維持 fp16。FP8 / NVFP4 只套在目標模型上，draft 的語意與對拍基準
  綁在 fp16。

### KV cache：FP8 與 NVFP4

新層在 `exllamav3/cache/fp8.py`、`exllamav3/cache/nvfp4.py`。`model_init` 的
`-cq` 接受：

| `-cq` | 格式 | 大約位元 |
|---|---|---|
| 省略 / `none` | fp16 | 16 |
| `8`、`8,4` | 上游整數 / 非對稱 Hadamard | ~8.5 |
| `3` | 上游整數 KV | 3 |
| `fp8` | E4M3，Triton paged attention 裡線上反量化 | 8 |
| `nvfp4` | E2M1 + 每 16 個元素一個 E4M3 scale | ~4.5 |

- FP8 / NVFP4 的新 K/V 在 Triton kernel 裡量化並 append，attention 直接讀
  packed page。沒有 Triton 時退回反量化後的 fp16 暫存，再寫回 page。
- 整數量化的 draft cache 補上 `max_history`，與 fp16 draft cache 對齊。
- 實測在生成層級與 fp16 對齊；NVFP4 每個 GB 大約能多放 3.5 倍 context
  （fp16 約 64 KB/token，NVFP4 約 18 KB/token；只有 full-attention 層持有 KV）。
- 24 GB 卡同時駐留 DFlash2 時，262144 context 用 `-cq 3`，見 2026-09-24。

### aarch64 / DGX Spark（GB10，sm_120 / sm_121）

上游擴充套件以 x86 AVX 為前提，GB10 上編不過。本 fork 讓同一份原始碼能在
aarch64 從原始碼建置（上游沒有對應的預編譯 wheel）：

- `avx2_target` / `avx512_target` 的 target attribute 與 CPUID 偵測只在 x86 展開。
- CPU MoE（`cpu/moe_mul1.cpp`、`cpu/moe_handoff.cu`）與 CPU all-reduce
  （`parallel/all_reduce_cpu*.cu` / `.cpp`）在 `__aarch64__` 走 scalar fallback。
- `start.sh` 在 aarch64 上把 `TORCH_CUDA_ARCH_LIST` 設成 `12.0;12.1`
  （`.env` 可覆寫）。x86 仍從 GPU 自動偵測。
- `quantize.sh` 同樣預設鎖 `12.0;12.1`，給這台 GB10 量化用。
- PyTorch wheel 預設索引是 CUDA 13.0（`TORCH_INDEX_URL` 可換）。

### OpenAI 相容伺服器

`tools/serve_openai.py`，給這份 Qwen3.8-27B EXL3 直接提供 API，不經過 TabbyAPI。

端點：

- `GET /v1/models`
- `GET /health`（忙碌旗標，以及給監控用的累計 token）
- `POST /v1/chat/completions`（串流與非串流）

行為：

- 取樣預設對齊 Qwen3.8：temperature 0.6、top-p 0.95、top-k 20。請求可覆寫。
- 思考內容以 `<think>` 出現在本文，並另給 `reasoning_content`。
- **工具呼叫**走 Qwen3.8 的 XML。`tools` 由模型的 HF chat template 渲染；
  `tool_choice` 為 required 或指定函式時，加上明確的 system 指示。歷史裡的
  `tool_calls` 與 `role:"tool"` 都交回模板。模型輸出的
  `<tool_call><function=...><parameter=...>` 解析成 OpenAI `tool_calls`，
  引數依請求自己的 JSON schema 轉型，對不上就留字串。生成停在 `</tool_call>`，
  `finish_reason` 為 `tool_calls`。
- 推測解碼是 **batch-1**。同時來的請求排隊，依序服務。
- 綁定位址預設 `0.0.0.0`。這條路徑沒有認證。

### 一鍵啟動、停止、量化

- `start.sh`：沒有 `.env` 就從 `.env.example` 複製並停下來等你改。第一次建立
  `.venv`、裝 GPU 版 PyTorch、從本機目錄（或 `EXL3_REPO`）編譯引擎、安裝
  `aiohttp` / `huggingface_hub`。缺少的 `MODEL_DIR` / `DRAFT_DIR` 會從 Hub 拉
  （私有庫用 `HF_TOKEN` 或 `hf auth login`）。之後直接起 `serve_openai`。
- `DRAFT`：`dflash2`（預設，搭配 `DRAFT_DIR`）、`mtp`、`none`。
- `GPU_MEM_GB` 留空時自動偵測：獨立 GPU 用 VRAM 減 2 GB；GB10 統一記憶體用
  系統可用 RAM 減 16 GB。
- `CPU_CACHE_GB` 會被伺服器收下，但 spilling 層還沒做，數值目前無效。保持 0。
- `stop.sh`：依 `PORT`（預設 8888）找到在聽的行程，確認命令列是
  `serve_openai` 之後才結束。
- `quantize.sh`：GB10 用的 `convert.py` 外殼。`-i` / `-o` / `-b`，工作目錄
  `-w`，中斷後以 `-w <work> -r` 續跑。

### 量化校正資料

`convert.py -cd / --cal_data` 可以用外部 trace 取代內建語料：

- turboderp 形式的 JSON：`{"rows": [{"input_ids", "response_ids"}, ...]}`
- 或純 token id 的 JSON 陣列

續跑時 `cal_data` 必須與當初的工作目錄一致，避免前後模組用到不同分布。
`tools/gen_cal_trace.py`、`tools/quantize_model.py` 負責產生 trace 與量化工作流程。

### Web console

`tools/webconsole.py` 用既有的 `aiohttp` 包一層網頁，不另加相依，也不再包一份
參數邏輯。表單來自 `tools/webconsole/specs.py`，每個欄位對到一條 CLI 旗標。

涵蓋量化、`serve_openai`、對運行中伺服器的聊天（SSE、思考、工具呼叫）、
`eval/*.py` 的 13 個評測、本倉庫的工具腳本、模型瀏覽、工作登錄與即時 log。
工作以獨立 process group 跑，console 重開仍在，也可以整組殺掉。預設只綁
`127.0.0.1`。`tools/webui.py`（舊的 Gradio 入口）保留不動。

### 對拍、評測與除錯工具

這些腳本是開發時用來釘住語意與量測，不是服務路徑的一部分：

- DFlash2：`tools/dflash2_unit_tests.py`、`dflash2_m1_parity.py`、
  `dflash2_cache_trace.py`、divergence / reference baseline 探針，以及
  `eval/dflash2-fixtures/`。
- DSpark：`tools/dspark_m1_parity.py`、`dspark_m2_integration.py`、
  `dspark_m4_bench.py`、`dspark_workload_ppl.py`。
- KV：`tools/fp8cache_test.py`、`nvfp4cache_test.py`、`fp8_to_bf16.py`、
  `nvfp4_to_bf16.py`。
- 速度與長文：`decode_bench.py`、`kernel_profile.py`、`step_breakdown.py`、
  `cudagraph_probe.py`、`gemm_audit.py`、`accept_probe.py`、
  `longctx_passkey.py`、`longctx_sanity.py`。

### 其他相對 v1.4.2 的小改動

- `requirements.txt` 加上 `aiohttp`、`huggingface_hub`，給伺服器與 Hub 下載。
- `eval/ppl.py` 的 WikiText-2 資料集 id 改為 `Salesforce/wikitext`。
- `.gitignore` 排除 `.venv/`、`.env`、`test_models/`、`reference/`、
  `hf-release/`、`jobs.jsonl`。
- 上游 README 改名保留為 `UPSTREAM_README.md`。倉庫根目錄的 `README.md`
  只寫這個 fork 怎麼服務 Qwen3.8-27B。

## 已知限制

- 伺服器一次只生成一個請求（batch-1 推測解碼）。併發會排隊。
- `HOST=0.0.0.0` 時區網上的人都能打 API，沒有認證。
- `CPU_CACHE_GB` 尚未接到真正的換出層。
- DFlash2 的 `NUM_DRAFT_TOKENS` 不能大於該 draft 的固定 block（這份 Qwen3.8 是 7）。
- FP8 / NVFP4 的 draft KV 維持 fp16。
- 在 24 GB、DFlash2 常駐、context 262144 的組合上，用整數 3-bit KV；同一張卡上
  FP8 與 NVFP4 配置不出來。
