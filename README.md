# ExLlamaV3 fork (MiaAI Lab)

Fork of [ExLlamaV3](https://github.com/turboderp-org/exllamav3) — EXL3
inference for local LLMs — adding what we needed to serve **Qwen3.8-27B**
on DGX Spark (GB10, aarch64) and 24 GB GPUs.

## What this fork adds

- **Speculative decoding**: DFlash2 and DSpark diffusion draft models, and
  MTP drafting for checkpoints with a multi-token-prediction head
  (Qwen3.8-27B ships one — `--mtp`). The MTP head predicts one token per
  forward; each verification round runs it `NUM_DRAFT_TOKENS` times
  (default 4).
- **Quantized KV cache**: NVFP4 (E2M1 + block-16 E4M3 scales, ~4.5 bits/elem)
  and FP8 lanes with online dequant in the Triton paged-attention kernels —
  measured lossless at generation level, ~3.5x more context per GB than fp16.
- **aarch64 / GB10 (sm_120 / sm_121) support**: builds and runs from source on
  DGX Spark (no prebuilt wheels exist upstream).
- **`tools/serve_openai.py`** — OpenAI-compatible server: chat completions
  (stream + non-stream), Qwen tool calling with typed arguments, batch-1
  speculative decode.
- **`tools/webconsole.py`** — web UI over the CLI: quantize, serve, chat
  playground, eval harnesses, job registry with live logs.
- **`start.sh` / `stop.sh`** — self-bootstrapping launcher: venv, GPU torch,
  engine build, HF weight download, YaRN long-context config swap.
- **`quantize.sh`** — GB10-friendly launcher for the EXL3 quantizer.

## Quick start (serving)

Easiest: the deployment kit —
[MiaAI-Lab/Qwen3.8-27B-DFlash2-EXL3-5.0bpw](https://github.com/MiaAI-Lab/Qwen3.8-27B-DFlash2-EXL3-5.0bpw)
(one command: downloads the weights, builds everything, serves the OpenAI
API — see its README for the full instructions).

From this repo:

```bash
./start.sh    # creates .env from .env.example on first run; edit and rerun
```

Library/API use follows upstream; the upstream README is preserved as
`UPSTREAM_README.md`.

## Speculative decoding

`./start.sh` reads `DRAFT` from `.env`:

| `DRAFT` | What runs | Default steps |
|---|---|---|
| `mtp` | Qwen3.8 MTP head inside the target checkpoint (~50 MB, no extra download) | 4 |
| `dflash2` | Companion diffusion draft (`DRAFT_DIR`) | 7 (`block_size - 1`) |
| `none` | Target model only | — |

The step count is the speculative loop. This checkpoint has
`mtp_num_hidden_layers: 1`, so one forward drafts one token. The server
then loops that head: each step takes the previous hidden state, samples
one greedy token, and feeds it back. After `NUM_DRAFT_TOKENS` steps the
target model verifies the whole draft at once. Four is the measured best
for Qwen3.5 MTP (`default_draft_size` in `exllamav3/architecture/qwen3_5_mtp.py`).

Set the window in `.env` and restart:

```bash
DRAFT=mtp
NUM_DRAFT_TOKENS=4
```

Leave `NUM_DRAFT_TOKENS` unset to keep the drafter default. The same knob
truncates a DFlash2 block before verification; do not set it above 7 for
this Qwen3.8 draft, because the diffusion block is fixed. `DRAFT=none`
ignores it.

The same length is `--num_draft_tokens` on `tools/serve_openai.py`, which
forwards `-ndt` / `--num_draft_tokens` into the generator. Library callers
pass `num_draft_tokens=` to `Generator` (see `examples/chat.py`). `--mtp`
and an external `--draft_model_dir` cannot be combined.

## License

MIT (as upstream). Fork changes: MIT.
