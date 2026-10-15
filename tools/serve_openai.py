#!/usr/bin/env python3
"""
Minimal OpenAI-compatible server for the EXL3 serving target.

Drafter: MTP by default (`-dm mtp`; the draft head lives inside the target
checkpoint). Alternatives: a DFlash2 draft model directory (`-dm <dir>`),
or no drafting (`-dm none`). The start.sh launcher maps the .env `DRAFT`
knob onto these.

Endpoints:
  GET  /v1/models
  GET  /health
  POST /v1/chat/completions   (stream and non-stream, tool calling)

Defaults match the serving convention: temperature 0.6, top-k 20, top-p 0.95,
thinking enabled (reasoning arrives inline in `<think>`), speculative
drafting active (drafter chosen via -dm, see above).
Concurrency: one AsyncGenerator batches up to --max_seqs requests. Each job
holds only a --max_rq_tokens window of new cache pages, then requeues, so one
long reply cannot reserve the whole KV pool. --max_seqs is also the recurrent
slot count (required for hybrid models such as MiniCPM-V / Qwen3.5).

Tool calling (Qwen3.8 XML format):
  - `tools` (OpenAI function specs) are rendered by the model's HF chat template
    (system "# Tools" section). `tool_choice` is accepted; required/specific
    choices are enforced with an explicit system directive.
  - assistant history with `tool_calls` is re-rendered natively by the template
    (arguments are converted JSON-string -> dict, as the template expects).
  - `role:"tool"` messages render as `<tool_response>` blocks natively.
  - Model output `<tool_call><function=name><parameter=k>v</parameter>
    </function></tool_call>` is parsed back into OpenAI `tool_calls` objects;
    generation stops at `</tool_call>`, finish_reason = "tool_calls".
  - Tool-call arguments are typed per the request's own JSON schemas
    (integer/number/boolean/array/object), strings kept on mismatch.

Launch (from repo root):
  .venv/bin/python tools/serve_openai.py \
      -m models/Qwen3.8-27B-EXL3-3.5bpw -gs 22 -cs 262144 -cq nvfp4 \
      --max_seqs 4 --port 8888
"""
import argparse, asyncio, json, os, re, sys, time, threading, uuid
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aiohttp import web

MODEL_DIR = "test_models/Qwen3.8-27B-exl3-3.5bpw-wm"
DRAFT_DIR = "mtp"   # default drafting method: MTP head (no external draft model)
PORT = 8888
# One recurrent checkpoint on Qwen3.5 / MiniCPM-V. Smaller values are rounded
# up to that interval inside Job.prepare_for_queue.
DEFAULT_MAX_RQ_TOKENS = 2048
DEFAULT_MAX_SEQS = 4
DEFAULT_MAX_CHUNK = 2048
# Used when a request omits max_tokens. Overridden by MAX_OUTPUT_TOKENS / --max_output_tokens.
DEFAULT_MAX_OUTPUT_TOKENS = 8192

stats_lock = threading.Lock()
# Cumulative counters for sparkDash live tok/s (GET /health).
stats = {
    "prompt_tokens_total": 0,
    "completion_tokens_total": 0,
    "context_length": None,
}

def _bump_stats(prompt=0, completion=0):
    if prompt <= 0 and completion <= 0:
        return
    with stats_lock:
        if prompt > 0:
            stats["prompt_tokens_total"] += int(prompt)
        if completion > 0:
            stats["completion_tokens_total"] += int(completion)

def _result_new_tokens(r):
    ids = r.get("token_ids") if isinstance(r, dict) else None
    if ids is None:
        return 0
    try:
        return int(ids.shape[-1])
    except Exception:
        return 0

TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"
HOLD_BACK = 16                       # marker-safe holdback for streamed text


def load_model(argv, use_draft = True):
    """Load weights and the paged cache. The AsyncGenerator is created later,
    inside the aiohttp loop (it starts a task in __init__)."""
    from argparse import ArgumentParser

    # The one-time JIT build of the CUDA extension can look like a hang;
    # say so before the import below blocks on it.
    try:
        import importlib.util, os
        if importlib.util.find_spec("exllamav3_ext") is None:
            _root = os.environ.get("TORCH_EXTENSIONS_DIR",
                                   os.path.expanduser("~/.cache/torch_extensions"))
            if not (os.path.isdir(_root) and
                    any(d == "exllamav3_ext"
                        for _, _dirs, _ in os.walk(_root) for d in _dirs)):
                print(" == compiling the CUDA extension "
                      "(one-time; a few minutes of silence is normal) ...", flush = True)
    except Exception:
        pass

    from exllamav3 import model_init
    parser = ArgumentParser()
    model_init.add_args(parser, add_draft_model_args = use_draft)
    args = parser.parse_args(argv)
    if use_draft:
        model, config, cache, tokenizer, draft_model, draft_config, draft_cache = \
            model_init.init(args, progress = True)
    else:
        model, config, cache, tokenizer = model_init.init(args, progress = True)
        draft_model = draft_cache = None
    return model, cache, tokenizer, draft_model, draft_cache


def normalize_messages(messages):
    """OpenAI history -> template-compatible dicts (tool_calls args str->dict)."""
    out = []
    for m in messages:
        m = dict(m)
        if m.get("role") == "assistant" and m.get("tool_calls"):
            calls = []
            for c in m["tool_calls"]:
                fn = dict(c.get("function") or {})
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except ValueError:
                        args = {}
                fn["arguments"] = args
                calls.append({"function": fn})
            m["tool_calls"] = calls
        out.append(m)
    return out


def split_reasoning(text):
    """Split Qwen reasoning from content. Generation starts inside <think>
    (the chat template ends with it), so text before </think> is reasoning.
    Returns (reasoning, content) with markers stripped."""
    close = text.find("</think>")
    if close >= 0:
        reasoning = text[:close]
        content = text[close + len("</think>"):]
        return reasoning.lstrip().removeprefix("<think>").strip(), content.strip("\n")
    if text.lstrip().startswith("<think>"):
        return text.lstrip()[len("<think>"):].strip(), ""
    return "", text


def build_tool_schemas(tools):
    """OpenAI tools list -> {function_name: {param_name: json-schema type}}."""
    schemas = {}
    for t in tools or []:
        fn = (t or {}).get("function") or {}
        name = fn.get("name")
        props = ((fn.get("parameters") or {}).get("properties")) or {}
        if name and isinstance(props, dict):
            schemas[name] = {k: v.get("type") for k, v in props.items()
                             if isinstance(v, dict)}
    return schemas


def _coerce_value(value, jtype):
    """Coerce one XML string parameter to the schema-declared JSON type.
    Lossless: on any mismatch the original string is returned unchanged."""
    v = value.strip()
    if not v:
        return value
    try:
        if jtype == "integer":
            return int(v)
        if jtype == "number":
            try:
                return int(v)
            except ValueError:
                return float(v)
        if jtype == "boolean":
            if v.lower() == "true": return True
            if v.lower() == "false": return False
        if jtype == "array":
            parsed = json.loads(v)
            if isinstance(parsed, list):
                return parsed
        if jtype == "object":
            parsed = json.loads(v)
            if isinstance(parsed, dict):
                return parsed
    except (ValueError, json.JSONDecodeError):
        pass
    return value


def coerce_tool_args(args, fn_schema):
    """Qwen's XML tool format delivers every parameter value as a string;
    OpenAI tool_calls arguments are typed JSON. Coerce each value using the
    request's own tool schema; undeclared params and failed coercions keep
    the raw string."""
    if not fn_schema:
        return args
    out = {}
    for k, v in args.items():
        t = fn_schema.get(k)
        types = t if isinstance(t, list) else [t]
        for tt in types:
            if isinstance(tt, str) and tt in ("integer", "number", "boolean",
                                              "array", "object"):
                cv = _coerce_value(v, tt)
                if not isinstance(cv, str):
                    v = cv
                    break
        out[k] = v
    return out


def parse_tool_calls(text, tool_schemas = None):
    """Parse Qwen XML tool calls. Returns (content_without_calls, [calls]).
    A <tool_call> block left unterminated is treated as complete: the
    </tool_call> stop-condition strips the closing tag from generated text."""
    calls = []
    content = text

    def parse_block(block):
        fm = re.search(r"<function=([^>]+)>", block)
        if not fm:
            return None
        name = fm.group(1).strip()
        args = {}
        for pm in re.finditer(r"<parameter=([^>]+)>\n?(.*?)\n?</parameter>",
                              block[fm.end():], flags = re.S):
            args[pm.group(1).strip()] = pm.group(2)
        if tool_schemas:
            args = coerce_tool_args(args, tool_schemas.get(name))
        return {
            "id": f"call_{uuid.uuid4().hex[:12]}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        }

    while True:
        i = content.find(TOOL_CALL_OPEN)
        if i < 0:
            break
        j = content.find(TOOL_CALL_CLOSE, i)
        if j < 0:
            # truncated close (stop string consumed): parse the remainder
            call = parse_block(content[i + len(TOOL_CALL_OPEN):])
            if call:
                calls.append(call)
            content = content[:i]
            break
        call = parse_block(content[i + len(TOOL_CALL_OPEN):j])
        if call:
            calls.append(call)
        content = content[:i] + content[j + len(TOOL_CALL_CLOSE):]
    return content, calls


def tool_choice_directive(tool_choice, tools):
    """OpenAI tool_choice -> (tools_to_render, extra system directive or None).
    The Qwen template has no tool_choice support, so required/specific are
    enforced with an explicit instruction appended to the history."""
    if tool_choice in (None, "auto"):
        return tools, None
    if tool_choice == "none":
        return None, None
    names = [t["function"]["name"] for t in (tools or [])
             if isinstance(t, dict) and t.get("type") == "function"]
    if isinstance(tool_choice, dict):
        name = (tool_choice.get("function") or {}).get("name")
        return tools, (f"You must call the function `{name}` now. Reply ONLY with "
                       f"the <tool_call> block for `{name}` and nothing else.")
    if tool_choice == "required":
        one_of = " or ".join(f"`{n}`" for n in names)
        return tools, (f"You must call one of the available functions ({one_of}) "
                       "now. Reply ONLY with the <tool_call> block and nothing else.")
    return tools, None


def render_prompt(tokenizer, messages, tools, tool_choice):
    """Apply the chat template. Returns (input_ids, prompt_toks, schemas, forced)."""
    schemas = build_tool_schemas(tools)
    tools, directive = tool_choice_directive(tool_choice, tools)
    if directive:
        messages = list(messages)
        if messages and messages[0].get("role") == "system":
            # Qwen template allows only ONE leading system message — merge
            first = dict(messages[0])
            first["content"] = (first.get("content") or "").rstrip() + "\n\n" + directive
            messages[0] = first
        else:
            messages = [{"role": "system", "content": directive}] + messages
    input_ids = tokenizer.hf_chat_template(
        messages, add_generation_prompt = True, enable_thinking = True,
        tools = tools)
    prompt_toks = int(input_ids.shape[-1])
    forced = tool_choice not in (None, "auto", "none")
    return input_ids, prompt_toks, schemas, forced


def finish_from(reason, calls):
    if calls:
        return "tool_calls"
    return {"max_new_tokens": "length", "eos": "stop",
            "stop_condition": "stop", "banned": "content_filter"}.get(reason, "stop")


async def _release_job(job):
    """Cancel unless the iterator already finished. Idempotent after EOS,
    and shielded so a client disconnect still frees the cache slot."""
    if job is None or job.cancelled:
        return
    try:
        await asyncio.shield(job.cancel())
    except Exception:
        pass


async def consume_job(agen, input_ids, prompt_toks, max_tokens, temperature,
                      top_p, top_k, seed, stop, eos_token_id, max_rq_tokens,
                      on_text = None):
    """Enqueue one job on the shared generator and read it to completion.
    Other in-flight consume_job calls are batched by AsyncGenerator."""
    from exllamav3.generator.sampler.presets import ComboSampler
    from exllamav3.generator.async_generator import AsyncJob
    sampler = ComboSampler(temperature = temperature, top_k = top_k, top_p = top_p)
    stop_conditions = ["<|im_end|>", eos_token_id] + (stop or [])
    job = AsyncJob(
        agen,
        input_ids = input_ids,
        max_new_tokens = max_tokens,
        max_rq_tokens = max_rq_tokens,
        stop_conditions = stop_conditions,
        sampler = sampler,
        seed = seed,
    )
    text = ""
    reason = "max_new_tokens"
    prefill_seen = 0
    try:
        async for r in job:
            if r.get("stage") == "prefill":
                curr = int(r.get("curr_progress") or 0)
                if curr > prefill_seen:
                    _bump_stats(prompt = curr - prefill_seen)
                    prefill_seen = curr
            elif _result_new_tokens(r):
                _bump_stats(completion = _result_new_tokens(r))
            chunk = r.get("text", "")
            if chunk:
                text += chunk
                if on_text is not None:
                    await on_text(chunk)
            if r.get("eos"):
                reason = r.get("eos_reason", reason)
        if prefill_seen < prompt_toks:
            _bump_stats(prompt = prompt_toks - prefill_seen)
    finally:
        await _release_job(job)
    seq = job.job.sequences[0]
    out_toks = int(seq.sequence_ids.seq_len - prompt_toks)
    return text, reason, out_toks


async def generate_full(agen, tokenizer, messages, max_tokens, temperature,
                        top_p, top_k, seed, tools, tool_choice = None, stop = None,
                        on_text = None, max_rq_tokens = DEFAULT_MAX_RQ_TOKENS):
    """One chat completion on the shared batch. Returns
    (text, tool_calls, finish, p_toks, o_toks, reasoning, content)."""
    input_ids, prompt_toks, schemas, forced = render_prompt(
        tokenizer, messages, tools, tool_choice)
    text, reason, out_toks = await consume_job(
        agen, input_ids, prompt_toks, max_tokens, temperature, top_p, top_k,
        seed, stop, tokenizer.eos_token_id, max_rq_tokens,
        on_text = None if forced else on_text)
    # Forced tool_choice is a prompt nudge; at temperature > 0 the model can
    # occasionally skip the call. One greedy retry makes it deterministic.
    if forced and not parse_tool_calls(text, schemas)[1]:
        text, reason, out_toks = await consume_job(
            agen, input_ids, prompt_toks, max_tokens, 0.0, top_p, top_k,
            seed, stop, tokenizer.eos_token_id, max_rq_tokens, on_text = None)
    content, calls = parse_tool_calls(text, schemas)
    reasoning, content = split_reasoning(content)
    return text, calls, finish_from(reason, calls), prompt_toks, out_toks, reasoning, content


async def models(request):
    ctx = stats.get("context_length")
    return web.json_response({"object": "list", "data": [{
        "id": "qwen3.8-27b-exl3-3.5bpw-wm",
        "object": "model",
        "owned_by": "exl3",
        **({"max_model_len": ctx} if ctx else {}),
    }]})


async def health(request):
    agen = request.app.get("agen")
    active = pending = None
    max_seqs = request.app.get("max_seqs")
    if agen is not None:
        gen = agen.generator
        active = gen.num_active_jobs()
        pending = gen.num_pending_jobs()
        max_seqs = gen.max_batch_size
    with stats_lock:
        body = {
            "ok": agen is not None,
            "busy": bool(active or pending),
            "backend": "exl3",
            "prompt_tokens_total": stats["prompt_tokens_total"],
            "completion_tokens_total": stats["completion_tokens_total"],
            "context_length": stats["context_length"],
            "active_jobs": active,
            "pending_jobs": pending,
            "max_seqs": max_seqs,
            "max_output_tokens": request.app.get("default_max_tokens"),
        }
    return web.json_response(body)


def parse_request(body, default_max_tokens):
    messages = body.get("messages")
    if not messages or not isinstance(messages, list):
        return None, "`messages` (list) is required"
    max_tokens = int(body.get("max_tokens") or
                     body.get("max_completion_tokens") or default_max_tokens)
    temperature = float(body.get("temperature", 0.6))
    top_p = float(body.get("top_p", 0.95))
    top_k = int(body.get("top_k", 20))
    seed = body.get("seed")
    tools = body.get("tools") or None
    stop = body.get("stop")
    if isinstance(stop, str):
        stop = [stop]
    elif not isinstance(stop, list):
        stop = None
    return dict(
        messages = normalize_messages(messages),
        max_tokens = max_tokens, temperature = temperature,
        top_p = top_p, top_k = top_k,
        seed = int(seed) if seed is not None else None,
        tools = tools,
        tool_choice = body.get("tool_choice"),
        stop = stop,
        stream = bool(body.get("stream", False)),
        model_id = body.get("model", "qwen3.8-27b-exl3-3.5bpw-wm"),
    ), None


def _cache_error(exc):
    return web.json_response(
        {"error": {"message": f"context/cache: {exc}", "type": "invalid_request_error"}},
        status = 400)


async def chat_completions(request):
    app = request.app
    agen, tokenizer = app.get("agen"), app["tokenizer"]
    if agen is None:
        return web.json_response(
            {"error": {"message": "model is not ready", "type": "server_error"}},
            status = 503)
    try:
        body = await request.json()
    except web.HTTPRequestEntityTooLarge:
        # aiohttp enforces client_max_size inside request.json(); without this
        # branch it falls into the generic handler below and gets misreported
        # as "invalid JSON" (400) even though the body parsed fine.
        return web.json_response(
            {"error": {"message": f"request body exceeds {request.app['max_body_mb']} MiB limit",
                       "type": "invalid_request_error",
                       "code": "request_entity_too_large"}},
            status = 413)
    except Exception:
        return web.json_response({"error": {"message": "invalid JSON"}}, status = 400)
    req, err = parse_request(body, request.app["default_max_tokens"])
    if err:
        return web.json_response({"error": {"message": err}}, status = 400)

    gen_kwargs = dict(
        agen = agen, tokenizer = tokenizer, messages = req["messages"],
        max_tokens = req["max_tokens"], temperature = req["temperature"],
        top_p = req["top_p"], top_k = req["top_k"], seed = req["seed"],
        tools = req["tools"], tool_choice = req["tool_choice"], stop = req["stop"],
        max_rq_tokens = app["max_rq_tokens"],
    )
    if not req["stream"]:
        try:
            text, calls, finish, ptoks, otoks, reasoning, content = await generate_full(**gen_kwargs)
        except AssertionError as e:
            return _cache_error(e)
        msg = {"role": "assistant", "content": content or None}
        if reasoning:
            msg["reasoning_content"] = reasoning
        if calls:
            msg["tool_calls"] = calls
        return web.json_response({
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion", "created": int(time.time()),
            "model": req["model_id"],
            "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
            "usage": {"prompt_tokens": ptoks, "completion_tokens": otoks,
                      "total_tokens": ptoks + otoks},
        })

    # ---- streaming (SSE) ----
    resp = web.StreamResponse(headers = {
        "Content-Type": "text/event-stream", "Cache-Control": "no-cache",
        "Connection": "keep-alive"})
    await resp.prepare(request)
    cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    model_id = req["model_id"]
    req_schemas = build_tool_schemas(req["tools"])
    forced_choice = req["tool_choice"] not in (None, "auto", "none")

    async def send(delta, finish = None):
        obj = {"id": cid, "object": "chat.completion.chunk",
               "created": int(time.time()), "model": model_id,
               "choices": [{"index": 0, "delta": delta,
                            "finish_reason": finish}]}
        await resp.write(f"data: {json.dumps(obj)}\n\n".encode())

    pending, calls_emitted = "", False
    call_idx = [0]
    in_think = [True]          # generation starts inside <think> (template)
    THINK_CLOSE = "</think>"

    async def send_call(c):
        nonlocal calls_emitted
        calls_emitted = True
        await send({"tool_calls": [dict(c, index = call_idx[0])]})
        call_idx[0] += 1

    async def flush_pending(final = False):
        """Emit everything parseable from pending; keep marker-safe tail."""
        nonlocal pending
        while True:
            if in_think[0]:
                close = pending.find(THINK_CLOSE)
                if close >= 0:
                    head, pending = pending[:close], pending[close + len(THINK_CLOSE):]
                    if head.strip():
                        await send({"reasoning_content": head.lstrip("\n")})
                    in_think[0] = False
                    continue
                cut = len(pending) if final else max(0, len(pending) - HOLD_BACK)
                piece = pending[:cut]
                if piece.strip():
                    await send({"reasoning_content": piece})
                pending = pending[cut:]
                return
            if TOOL_CALL_OPEN in pending:
                head, rest = pending.split(TOOL_CALL_OPEN, 1)
                if head.strip() or (final and head):
                    await send({"content": head})
                if TOOL_CALL_CLOSE in rest:
                    block, pending = rest.split(TOOL_CALL_CLOSE, 1)
                    _, calls = parse_tool_calls(
                        TOOL_CALL_OPEN + block + TOOL_CALL_CLOSE,
                        req_schemas)
                    for c in calls:
                        await send_call(c)
                    continue
                # unterminated call: final -> implicit close, else hold
                if final and "<function=" in rest:
                    _, calls = parse_tool_calls(TOOL_CALL_OPEN + rest,
                                                req_schemas)
                    for c in calls:
                        await send_call(c)
                    pending = ""
                else:
                    pending = TOOL_CALL_OPEN + rest
                return
            cut = len(pending) if final else max(0, len(pending) - HOLD_BACK)
            await send({"content": pending[:cut]})
            pending = pending[cut:]
            return

    async def on_text(chunk):
        nonlocal pending
        pending += chunk
        await flush_pending()

    try:
        text, calls, finish, _ptoks, _otoks, reasoning, content = await generate_full(
            **gen_kwargs, on_text = None if forced_choice else on_text)
        await flush_pending(final = True)
        if forced_choice:
            # Buffered path (no deltas were streamed): emit the
            # authoritative complete result as deltas.
            if reasoning:
                await send({"reasoning_content": reasoning})
            if content:
                await send({"content": content})
        if not calls_emitted and calls:
            for c in calls:
                await send_call(c)
        await send({}, finish = finish)
        await resp.write(b"data: [DONE]\n\n")
        await resp.write_eof()
    except AssertionError as e:
        await resp.write(
            f'data: {json.dumps({"error": {"message": f"context/cache: {e}"}})}\n\n'.encode())
        await resp.write_eof()
    except ConnectionResetError:
        pass
    return resp


def main():
    global MODEL_DIR, DRAFT_DIR, PORT
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", default = MODEL_DIR)
    ap.add_argument("-dm", "--draft_model", default = DRAFT_DIR,
                    help = "Draft model path, 'mtp' for MTP drafting (head inside the "
                           "main checkpoint: no extra weights, much smaller KV footprint) "
                           "or 'none' to disable drafting")
    ap.add_argument("-gs", "--grid_size", type = int, default = 110)
    ap.add_argument("-cs", "--cache_size", type = str,
                    default = os.environ.get("CONTEXT_SIZE", "65536"),
                    help = "KV cache size in tokens, or 'auto' to reserve a pool "
                           "from free memory (default 65536, env CONTEXT_SIZE)")
    ap.add_argument("--kv_util", type = float,
                    default = float(os.environ.get("KV_UTIL", "0.85")))
    ap.add_argument("--kv_reserve_gb", type = float,
                    default = float(os.environ.get("KV_RESERVE_GB", "16")))
    ap.add_argument("--kv_max_gb", type = float,
                    default = float(os.environ.get("KV_MAX_GB", "48")))
    ap.add_argument("--kv_scratch_gb", type = float,
                    default = float(os.environ.get("KV_SCRATCH_GB", "4")))
    ap.add_argument("-cq", "--cache_quant", type = str, default = None,
                    help = "Quantized KV cache bits, e.g. 8 or 8,4 (k_bits[,v_bits])")
    ap.add_argument("-p", "--port", type = int, default = PORT)
    ap.add_argument("--host", type = str, default = "0.0.0.0",
                    help = "Interface to bind (use 127.0.0.1 for local-only)")
    ap.add_argument("-ccs", "--cpu_cache_size", type = float, default = 0.0,
                    help = "CPU second-tier cache size in GB (pages spill from "
                           "GPU when the GPU cache is full)")
    ap.add_argument("--max_body_mb", type = int, default = 64,
                    help = "max request body size in MiB (aiohttp's built-in "
                           "default is 1 MiB, far too small for a full tool "
                           "set + a long transcript)")
    ap.add_argument("--max_seqs", type = int, default = DEFAULT_MAX_SEQS,
                    help = "Max sequences decoded together, and the recurrent-state "
                           "slot count (default 4). The KV pool (-cs) is shared.")
    ap.add_argument("--max_rq_tokens", type = int, default = DEFAULT_MAX_RQ_TOKENS,
                    help = "New tokens a job may pin before it requeues and releases "
                           "its cache window (default 2048, one hybrid checkpoint).")
    ap.add_argument("--max_chunk_size", type = int, default = DEFAULT_MAX_CHUNK,
                    help = "Max prefill tokens per generator step (default 2048).")
    ap.add_argument("--max_output_tokens", type = int,
                    default = int(os.environ.get("MAX_OUTPUT_TOKENS", DEFAULT_MAX_OUTPUT_TOKENS)),
                    help = "Completion cap when a request omits max_tokens "
                           "(env MAX_OUTPUT_TOKENS, default 8192). "
                           "A request that sets max_tokens still wins.")
    args = ap.parse_args()
    if args.max_seqs < 1:
        ap.error("--max_seqs must be >= 1")
    if args.max_rq_tokens < 1:
        ap.error("--max_rq_tokens must be >= 1")
    if args.max_output_tokens < 1:
        ap.error("--max_output_tokens must be >= 1")
    _draft = args.draft_model.lower()
    use_mtp = _draft == "mtp"
    use_draft = _draft not in ("none", "", "-")
    argv = ["-m", args.model,
            "-gs", str(args.grid_size), "-cs", str(args.cache_size),
            "-ambs", str(args.max_seqs),
            "--kv_util", str(args.kv_util),
            "--kv_reserve_gb", str(args.kv_reserve_gb),
            "--kv_max_gb", str(args.kv_max_gb),
            "--kv_scratch_gb", str(args.kv_scratch_gb)]
    if use_mtp:
        argv += ["-mtp"]
    elif use_draft:
        argv += ["-dm", args.draft_model]
    if args.cache_quant:
        argv += ["-cq", args.cache_quant]
    if args.cpu_cache_size:
        argv += ["-ccs", str(args.cpu_cache_size)]

    print(f" == loading {args.model}"
          + (" + MTP head" if use_mtp else
             (f" + draft {args.draft_model}" if use_draft else " (no draft)"))
          + f", max_seqs {args.max_seqs}"
          + " ...", flush = True)
    model, cache, tokenizer, draft_model, draft_cache = load_model(argv, use_draft = use_draft)
    stats["context_length"] = int(cache.max_num_tokens)
    print(" == model ready; accepting requests", flush = True)

    async def start_agen(app):
        from exllamav3.generator.async_generator import AsyncGenerator
        kwargs = dict(
            max_chunk_size = args.max_chunk_size,
            max_batch_size = cache.num_slots,
        )
        if draft_model is not None:
            kwargs["draft_model"] = draft_model
            kwargs["draft_cache"] = draft_cache
        app["agen"] = AsyncGenerator(model, cache, tokenizer, **kwargs)
        # Let the iteration task reach its wait before the first request, so a
        # notify cannot land before the condition is being waited on.
        await asyncio.sleep(0)
        print(f" == batching up to {app['agen'].generator.max_batch_size} sequences, "
              f"rq window {args.max_rq_tokens}, "
              f"default max_tokens {args.max_output_tokens}", flush = True)

    async def stop_agen(app):
        agen = app.get("agen")
        if agen is not None:
            await agen.close()

    app = web.Application(client_max_size = args.max_body_mb * 1024 * 1024)
    app["tokenizer"] = tokenizer
    app["max_body_mb"] = args.max_body_mb
    app["max_seqs"] = args.max_seqs
    app["max_rq_tokens"] = args.max_rq_tokens
    app["default_max_tokens"] = args.max_output_tokens
    app.on_startup.append(start_agen)
    app.on_cleanup.append(stop_agen)
    app.router.add_get("/v1/models", models)
    app.router.add_get("/health", health)
    app.router.add_post("/v1/chat/completions", chat_completions)
    web.run_app(app, host = args.host, port = args.port, print = None)


if __name__ == "__main__":
    main()
