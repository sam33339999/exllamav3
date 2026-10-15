# 服務與 KV 預留池

這份說明給 `start.sh` 和 `tools/serve_openai.py`。池子在程序啟動時配一次，結束前不釋放，也不做定期清掃。

## 啟動

```bash
./start.sh
```

第一次會從 `.env.example` 複製出 `.env`。改完再跑一次。沒有 `.env` 時也可以直接：

```bash
.venv/bin/python -u tools/serve_openai.py \
    -m /path/to/model \
    -dm none \
    --max_seqs 4 \
    --max_rq_tokens 2048 \
    --max_output_tokens 8192 \
    -gs 40 \
    -cs auto \
    -cq nvfp4 \
    --host 0.0.0.0 \
    --port 8888
```

`--draft_model none` 關掉 MTP / DFlash / DSpark。多人同時生成時這些 drafter 通常不划算。

## 並行

`--max_seqs`（環境變數 `MAX_SEQS`，預設 4）是同時解碼的人數，也是混合模型線性注意力的 slot 數。多出來的請求在同一個 generator 裡排隊。

每個回覆只先釘住 `--max_rq_tokens`（預設 2048，一個 checkpoint）的新頁，然後重新排隊。一則很長的回覆不會把整個池子訂走。

客戶端沒帶 `max_tokens` 時，輸出上限是 `MAX_OUTPUT_TOKENS`（預設 8192）。有帶就以客戶端為準。思考 token 算在裡面。`/health` 的 `finish` 若是 `length`，就是碰到這個上限。

## 上下文池

`CONTEXT_SIZE` 或 `--cache_size`：

- 數字：池子就是那麼多 token。現在的服務預設走這條。
- `auto`：載入權重前依剩餘記憶體計算，再一次配好。

`auto` 的算法：

```text
leave  = max(KV_RESERVE_GB, free × (1 − KV_UTIL))
budget = min(free − 權重檔大小 − KV_SCRATCH_GB − leave, KV_MAX_GB)
tokens = budget / 每 token 位元組，對齊到 256
```

| 變數 | 預設 | 意思 |
|---|---|---|
| `KV_UTIL` | 0.85 | 對「目前剩餘」的比例，不是對整台記憶體 |
| `KV_RESERVE_GB` | 16 | 至少留給系統的 GiB |
| `KV_MAX_GB` | 48 | 池子硬頂 |
| `KV_SCRATCH_GB` | 4 | 載入時的暫存 |

每 token 位元組用實際的 cache 格式算。nvfp4 的 draft 仍是 fp16；有開 draft 時兩邊一起算。預算不夠一頁就拒絕啟動。啟動紀錄會印 free、total、權重、leave、budget、token 數。`/health` 的 `context_length` 就是這個數。

單條 prompt 超過模型的 `max_position_embeddings` 會在排入時被拒。MiniCPM-V 4.7 是 262144。池子可以比這個大，多出來的是大家共用的頁，不是一個人的上下文。

## EXL3_QC_STAGING

量化 KV（fp8、nvfp4、int）怎麼送進 attention。解碼一直直接讀壓縮過的頁。這個開關只影響 prefill 要不要先展開。

| 值 | 行為 |
|---|---|
| `0` | 不另配暫存。prefill 和 decode 都在 kernel 裡把量化頁解開再算。記憶體最少。prefill 的 attention kernel 大約慢 5% 到 25%，視位元數和 GPU 而定。 |
| `1` | 預設。query 長度到 256 以上的 prefill，先把用到的窗解成 fp16，再跑 fp16 kernel。decode 仍直接讀量化頁。暫存按整個池子、batch 1 配一份 K 和一份 V，載入時的 autosplit 會先把這塊算進去。 |
| `2` | 舊路徑。進 attention 前把整層解成 fp16。除錯用。 |

`EXL3_QC_STAGING=0` 就是關掉那份 fp16 暫存。池子只有 nvfp4 本身。長 prefill 稍慢，不會在 KV 之外再佔一塊跟池子一樣大的 fp16。

設成 `1` 時，MiniCPM-V 4.7（2 個 KV head、head dim 256）大約再加 **2048 bytes/token**。這塊還沒算進 `auto` 的預算。`KV_MAX_GB=48` 再加這份暫存，在已經跑著 ASR / vLLM 的機器上會太滿。共用機器請把 `KV_MAX_GB` 調低，或設 `EXL3_QC_STAGING=0`。

池子配好之後不會因為閒置而縮小。對話放著不會過期。頁面只在別人需要空間時，從最舊的序列、由尾巴往回剪。剪掉之後，下一次進來要把那段 prefill 再算一次。要把記憶體還給其他程序，停掉這個服務。

驗證腳本是 `tools/verify_kv.py`。`(1)` 量 1、2、4、8 人。`(2)` 看共用前綴，並在小池子（`-cs 8192` 或 `16384`）上把頁面塞滿，觀察淘汰。小池子裝不下 8 個人的快取窗口時，`active_jobs` 會停在比較小的數字，人會排隊。`(3)` 先重送同一段對話，確認閒置不會清掉前綴；換一段沒看過的對話，prefill 會回到冷啟動的成本。大池子上腳本會拒絕把幾百萬 token 塞滿。回覆的 `usage` 帶 `cached_tokens` 和 `time_prefill`。`/health` 的 `cache` 帶頁數和 `evictions_live`。
