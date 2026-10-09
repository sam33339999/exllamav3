# Qwen3.8 coder390 EXL3 量化記錄

日期：2026-10-09。機器：NVIDIA GB10，119 GB 統一記憶體。

這份紀錄放在引擎 repo 的 `doc/`，不放在權重目錄裡。權重目錄不會進 git。

原始 BF16 沒有改過。量化工作在寫這份紀錄時還在跑（文字層大約第 54 層，共 64 層）。`EXL3-3.5bpw` 和 `DFlash2-EXL3-5.0bpw` 要等日誌出現 `=== MAIN DONE ===` 和 `=== DFLASH DONE ===` 之後才算完成，完成前不要拿來開服務。

## 原始檔

目錄：`/home/spark/exllamav3/Qwen3.8-27b-coder390-efficientthinking-opus5.5-gpt6astra-grok4.7-dsv4pro-k3-sft-rloo-mtp-dflash2/BF16/`

裡面是三樣東西：

- 主模型。`Qwen3_5ForConditionalGeneration`，64 層文字塔加視覺塔。權重在 `model-00001-of-00002.safetensors`、`model-00002-of-00002.safetensors`。
- MTP。多猜一個 token 的小頭，1 層，檔案是 `mtp-bf16.safetensors`。`model.safetensors.index.json` 已經把它算進同一份模型。
- DFlash2。另一個 5 層草稿模型，在 `BF16/DFlash2-FP8/`。權重是 FP8 E4M3、128×128 block scale，架構名 `DFlash2DraftModel`。

## 怎麼轉

一支腳本：`/home/spark/exl3-coder390-quant.sh`

它先轉主模型，成功之後才轉 DFlash2。暫存和日誌：

- 主模型暫存：`/home/spark/exl3-work-coder390-3.5`
- 日誌：`/home/spark/exl3-work-coder390-3.5/quantize.log`
- DFlash2 暫存：`/home/spark/exl3-work-coder390-dflash2`

主模型用 `convert.py`，參數是這台機器上次成功轉同架構模型時用的那組：

- 文字塔平均 3.5 bpw（校正 250 列 × 2048 token，codebook `mul1`）
- `lm_head` 6 bit
- MTP 的 8 個矩陣 4 bit。文字塔 64 層全部結束後才壓，日誌會出現 `Quantizing MTP tensors`，然後寫進同一個輸出目錄
- 視覺塔 16 bit，等於保留 BF16

DFlash2 用同支 `convert.py`，位元率 5.0。載入時先把 FP8 還原再壓。卷積的 `kernel_projection` 和 `candidate_selector.hidden_projection` 保持 BF16。草稿長度是 7 個 token（block size 8 減 1）。

## 完成後的目錄

都在權重那一層，和 `BF16/` 並列：

`/home/spark/exllamav3/Qwen3.8-27b-coder390-efficientthinking-opus5.5-gpt6astra-grok4.7-dsv4pro-k3-sft-rloo-mtp-dflash2/`

- `EXL3-3.5bpw`：主模型、視覺塔、MTP 頭在同一份檢查點
- `DFlash2-EXL3-5.0bpw`：DFlash2 草稿，單獨一個目錄

MTP 和 DFlash2 兩種加速一次只用一種。

中斷主模型之後可以續跑：

```bash
cd /home/spark/exllamav3
./quantize.sh -w /home/spark/exl3-work-coder390-3.5 -r
```

## 用這個 repo 的伺服器開

`tools/serve_openai.py` 認得 MTP 和 DFlash2。在量化完成、而且 GPU 沒有被量化程式佔住之後：

只用 MTP：

```bash
cd /home/spark/exllamav3
.venv/bin/python -u tools/serve_openai.py \
  -m /home/spark/exllamav3/Qwen3.8-27b-coder390-efficientthinking-opus5.5-gpt6astra-grok4.7-dsv4pro-k3-sft-rloo-mtp-dflash2/EXL3-3.5bpw \
  -dm mtp \
  -cs 262144 \
  -cq nvfp4 \
  --host 127.0.0.1 \
  --port 5000
```

改用 DFlash2 時，把 `-dm mtp` 換成：

```bash
  -dm /home/spark/exllamav3/Qwen3.8-27b-coder390-efficientthinking-opus5.5-gpt6astra-grok4.7-dsv4pro-k3-sft-rloo-mtp-dflash2/DFlash2-EXL3-5.0bpw
```

`-cs` 是 KV cache 的 token 數。`-cq nvfp4` 是這份 fork 的 KV 壓縮。介面是 OpenAI 相容的 `http://127.0.0.1:5000/v1`。

## 用 TabbyAPI 開

這台機器上還沒有裝 TabbyAPI。TabbyAPI 的 `config.yml` 放在它自己的專案目錄，`model_dir` 是模型資料夾的上一層，`model_name` 是資料夾名稱。

只用 MTP。MTP 頭已經在 `EXL3-3.5bpw` 裡面，不用再指一個草稿目錄：

```yaml
network:
  host: 127.0.0.1
  port: 5000
  disable_auth: true

model:
  model_dir: /home/spark/exllamav3/Qwen3.8-27b-coder390-efficientthinking-opus5.5-gpt6astra-grok4.7-dsv4pro-k3-sft-rloo-mtp-dflash2
  model_name: EXL3-3.5bpw
  backend: exllamav3
  max_seq_len: 262144
  cache_size: 262144
  cache_mode: FP16
  vision: false
  reasoning: true

draft_model:
  draft_mode: mtp
  draft_num_tokens: 4
```

改用 DFlash2 時，`draft_model` 改成：

```yaml
draft_model:
  draft_mode: model
  draft_model_dir: /home/spark/exllamav3/Qwen3.8-27b-coder390-efficientthinking-opus5.5-gpt6astra-grok4.7-dsv4pro-k3-sft-rloo-mtp-dflash2
  draft_model_name: DFlash2-EXL3-5.0bpw
  draft_num_tokens: 7
```

在 TabbyAPI 的專案目錄啟動：

```bash
python main.py
```

或是它自帶的 `./start.sh`。

TabbyAPI 預設會自己裝一份上游 exllamav3。DFlash2 是這個 MiaAI fork 才有的架構，MTP 也是這份 fork 測過的路徑。要讓 TabbyAPI 用這裡已經編好的引擎，在 `/home/spark/exllamav3` 的 `.venv` 裡安裝 TabbyAPI，並且不要讓它的更新腳本覆蓋 `exllamav3`。版本檢查擋下來時，在 `config.yml` 設 `developer.unsafe_launch: true`。

TabbyAPI 的 `cache_mode` 文件列出的是 `FP16`、`Q8`、`Q6`、`Q4`，或 `8,8` 這種寫法。`nvfp4` 是這份 fork 的伺服器參數（上面的 `-cq nvfp4`），不要直接寫進 TabbyAPI 的 `cache_mode`，除非你用的那版 TabbyAPI 明確寫了支援。

`disable_auth: true` 只適合 `127.0.0.1`。改成 `0.0.0.0` 時要把認證開回去。
