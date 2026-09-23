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
thinking enabled (reasoning arrives inline in `<think>` / `reasoning_content`),
speculative drafting active (drafter chosen via -dm, see above).
Thinking is budgeted: a long `<think>` is force-closed with `</think>` so the
answer still has tokens left (see `--thinking_budget` / `--answer_reserve`).
SSE streams send comment pings so proxies do not drop a long think/prefill.
Concurrency: requests are serialized (batch-1 draft); concurrent callers queue.

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
      -m models/Qwen3.8-27B-EXL3-3.5bpw -gs 22 -cs 262144 -cq nvfp4 --port 8888
"""
import argparse, json, os, re, sys, time, threading, uuid
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aiohttp import web

MODEL_DIR = "test_models/Qwen3.8-27B-exl3-3.5bpw-wm"
DRAFT_DIR = "mtp"   # default drafting method: MTP head (no external draft model)
PORT = 8888

gen_lock = threading.Lock()          # serialize generation (batch-1 draft)
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
THINK_CLOSE = "</think>"
THINK_CLOSE_INJECT = "</think>\n\n"
HOLD_BACK = 16                       # marker-safe holdback for streamed text
DEFAULT_MAX_TOKENS = 16384
DEFAULT_ANSWER_RESERVE = 512
DEFAULT_REASONING_EFFORT = "xhigh"
SSE_KEEPALIVE_S = 15
VALID_REASONING_EFFORTS = ("low", "medium", "xhigh")
# Soft cap on tokens spent inside <think> (0 in --thinking_budget = no cap
# besides --answer_reserve). xhigh is the template default and can otherwise
# eat the entire max_tokens budget, leaving an empty / truncated answer.
EFFORT_THINKING_BUDGET = {
    "low": 1024,
    "medium": 4096,
    "xhigh": 8192,
}
SERVER_DEFAULTS = {
    "max_tokens": DEFAULT_MAX_TOKENS,
    "reasoning_effort": DEFAULT_REASONING_EFFORT,
    "thinking_budget": None,          # None = from reasoning_effort
    "answer_reserve": DEFAULT_ANSWER_RESERVE,
    "enable_thinking": True,
}


def build_model(argv, use_draft = True):
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

    from exllamav3 import model_init, Generator
    parser = ArgumentParser()
    model_init.add_args(parser, add_draft_model_args = use_draft)
    args = parser.parse_args(argv)
    if use_draft:
        model, config, cache, tokenizer, draft_model, draft_config, draft_cache = \
            model_init.init(args, progress = True)
        generator = Generator(
            model, cache, tokenizer,
            draft_model = draft_model, draft_cache = draft_cache,
            # num_draft_tokens defaults to the draft model's arch-declared
            # default_draft_size (DFlash2: block_size - 1 = 7, MTP: 4). Must
            # match model_init's max_history sizing, which reads the same caps.
        )
    else:
        model, config, cache, tokenizer = model_init.init(args, progress = True)
        generator = Generator(model, cache, tokenizer)
    return generator, tokenizer


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


def _as_bool(v, default = True):
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("0", "false", "no", "off", "none", ""):
            return False
        if s in ("1", "true", "yes", "on"):
            return True
    return default


def _lookup_request_field(body, *keys):
    """Prefer a top-level field, then chat_template_kwargs, then reasoning.effort."""
    ctk = body.get("chat_template_kwargs")
    if not isinstance(ctk, dict):
        ctk = {}
    for k in keys:
        if k in body and body[k] is not None:
            return body[k]
        if k in ctk and ctk[k] is not None:
            return ctk[k]
    reasoning = body.get("reasoning")
    if isinstance(reasoning, dict):
        for k in keys:
            if k in reasoning and reasoning[k] is not None:
                return reasoning[k]
            if k == "reasoning_effort" and reasoning.get("effort") is not None:
                return reasoning.get("effort")
    return None


def resolve_think_limits(max_tokens, enable_thinking, reasoning_effort,
                         thinking_budget, answer_reserve):
    """Return (think_cap, reserve). think_cap is None when thinking is off.

    think_cap is the max *new* tokens allowed while still inside <think>
    before we inject </think> and generate the answer. Always leaves
    `reserve` tokens for the answer so a long think cannot consume 100% of
    max_tokens (which is what made streams look 'interrupted').
    """
    if not enable_thinking or int(max_tokens) <= 0:
        return None, 0
    reserve = int(answer_reserve or 0)
    if reserve < 0:
        reserve = 0
    if reserve >= max_tokens:
        reserve = max(1, max_tokens // 4)
    if reserve >= max_tokens:
        reserve = 0
    cap = max_tokens - reserve
    if thinking_budget is None:
        thinking_budget = EFFORT_THINKING_BUDGET.get(
            reasoning_effort, EFFORT_THINKING_BUDGET[DEFAULT_REASONING_EFFORT])
    thinking_budget = int(thinking_budget)
    if thinking_budget > 0:
        cap = min(cap, thinking_budget)
    return max(1, cap), reserve


def should_force_close_think(in_think, new_tokens, think_cap):
    return bool(in_think and think_cap is not None and new_tokens >= think_cap)


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


def generate_full(generator, tokenizer, messages, max_tokens, temperature,
                  top_p, top_k, seed, tools, tool_choice = None, stop = None,
                  on_text = None, enable_thinking = True,
                  reasoning_effort = DEFAULT_REASONING_EFFORT,
                  thinking_budget = None,
                  answer_reserve = DEFAULT_ANSWER_RESERVE):
    """Blocking generation; returns (text, tool_calls, finish, p_toks, o_toks,
    reasoning, content).

    If thinking is enabled and the model is still inside <think> when
    `think_cap` new tokens have been produced, generation is cancelled, 
    `</think>\\n\\n` is appended, and the remaining budget is used for the
    answer. That is what stops a long think from eating max_tokens and
    returning an empty / truncated completion.
    """
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
    tpl_kw = dict(enable_thinking = bool(enable_thinking), tools = tools)
    if enable_thinking:
        tpl_kw["reasoning_effort"] = reasoning_effort
    input_ids = tokenizer.hf_chat_template(
        messages, add_generation_prompt = True, **tpl_kw)
    prompt_toks = int(input_ids.shape[-1])
    from exllamav3.generator.sampler.presets import ComboSampler
    from exllamav3 import Job
    forced_choice = tool_choice not in (None, "auto", "none")
    reason = "max_new_tokens"
    text = ""
    think_cap, _reserve = resolve_think_limits(
        max_tokens, enable_thinking, reasoning_effort,
        thinking_budget, answer_reserve)

    def run_once():
        nonlocal text, reason
        text = ""
        reason = "max_new_tokens"
        sampler = ComboSampler(temperature = temperature, top_k = top_k, top_p = top_p)
        stop_conditions = ["<|im_end|>", tokenizer.eos_token_id] + (stop or [])
        job = Job(input_ids = input_ids, max_new_tokens = max_tokens,
                  stop_conditions = stop_conditions,
                  sampler = sampler, seed = seed)
        prefill_seen = 0
        in_think = bool(enable_thinking)

        def consume_results(results):
            nonlocal text, reason, in_think, prefill_seen
            for r in results:
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
                        on_text(chunk)
                    if in_think and THINK_CLOSE in text:
                        in_think = False
                if r.get("eos"):
                    reason = r.get("eos_reason", reason)

        def maybe_continue_after_think():
            """If we are still inside <think>, inject the closer and keep going."""
            nonlocal text, in_think, job, reason
            if not in_think or THINK_CLOSE in text:
                return
            if job in generator.active_jobs or job in generator.pending_jobs:
                generator.cancel(job)
            held = job.held_text or ""
            if held:
                text += held
                if on_text is not None:
                    on_text(held)
                if THINK_CLOSE in text:
                    in_think = False
                    return
            text += THINK_CLOSE_INJECT
            if on_text is not None:
                on_text(THINK_CLOSE_INJECT)
            in_think = False
            import torch
            close_ids = tokenizer.encode(
                THINK_CLOSE_INJECT, add_bos = False, encode_special_tokens = True)
            ids = torch.cat([
                job.sequences[0].sequence_ids.torch().cpu().contiguous().clone(),
                close_ids.cpu(),
            ], dim = -1)
            remaining = int(max_tokens) - int(job.new_tokens) - int(close_ids.shape[-1])
            print(f" == thinking budget: injected {THINK_CLOSE} after "
                  f"{job.new_tokens} tokens (cap={think_cap}, remaining={remaining})",
                  flush = True)
            if remaining <= 0:
                return
            job = Job(input_ids = ids, max_new_tokens = remaining,
                      stop_conditions = stop_conditions,
                      sampler = sampler, seed = seed)
            generator.enqueue(job)
            while generator.num_remaining_jobs():
                consume_results(generator.iterate())

        with gen_lock:
            generator.enqueue(job)
            while generator.num_remaining_jobs():
                consume_results(generator.iterate())
                if should_force_close_think(in_think, job.new_tokens, think_cap):
                    maybe_continue_after_think()
                    break
            # Race: the job hit max_new_tokens in the same step as the cap.
            if in_think and THINK_CLOSE not in text:
                maybe_continue_after_think()
            if prefill_seen < prompt_toks:
                _bump_stats(prompt = prompt_toks - prefill_seen)
        return job

    job = run_once()
    # Forced tool_choice is a prompt nudge; at temperature > 0 the model can
    # occasionally skip the call. One greedy retry makes it deterministic.
    if forced_choice and not parse_tool_calls(text, schemas)[1]:
        temperature = 0.0
        job = run_once()
    seq = job.sequences[0]
    out_toks = int(seq.sequence_ids.seq_len - prompt_toks)
    content, calls = parse_tool_calls(text, schemas)
    if calls:
        finish = "tool_calls"
    else:
        finish = {"max_new_tokens": "length", "eos": "stop",
                  "stop_condition": "stop", "banned": "content_filter"}.get(
                      reason, "stop")
    reasoning, content = split_reasoning(content)
    return text, calls, finish, prompt_toks, out_toks, reasoning, content


async def models(request):
    ctx = stats.get("context_length")
    return web.json_response({"object": "list", "data": [{
        "id": "qwen3.8-27b-exl3-3.5bpw-wm",
        "object": "model",
        "owned_by": "exl3",
        **({"max_model_len": ctx} if ctx else {}),
    }]})


async def health(request):
    with stats_lock:
        return web.json_response({
            "ok": True,
            "busy": gen_lock.locked(),
            "backend": "exl3",
            "prompt_tokens_total": stats["prompt_tokens_total"],
            "completion_tokens_total": stats["completion_tokens_total"],
            "context_length": stats["context_length"],
            "default_max_tokens": SERVER_DEFAULTS["max_tokens"],
            "reasoning_effort": SERVER_DEFAULTS["reasoning_effort"],
            "thinking_budget": SERVER_DEFAULTS["thinking_budget"],
            "answer_reserve": SERVER_DEFAULTS["answer_reserve"],
            "enable_thinking": SERVER_DEFAULTS["enable_thinking"],
        })


def parse_request(body, defaults = None):
    messages = body.get("messages")
    if not messages or not isinstance(messages, list):
        return None, "`messages` (list) is required"
    defaults = defaults or SERVER_DEFAULTS
    max_tokens = int(body.get("max_tokens") or
                     body.get("max_completion_tokens") or
                     defaults.get("max_tokens", DEFAULT_MAX_TOKENS))
    if max_tokens <= 0:
        return None, "`max_tokens` must be > 0"
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
    enable_thinking = _as_bool(
        _lookup_request_field(body, "enable_thinking"),
        defaults.get("enable_thinking", True))
    effort_raw = _lookup_request_field(body, "reasoning_effort")
    if effort_raw is None:
        reasoning_effort = defaults.get("reasoning_effort", DEFAULT_REASONING_EFFORT)
    else:
        reasoning_effort = str(effort_raw).strip().lower()
        if reasoning_effort not in VALID_REASONING_EFFORTS:
            return None, (
                f"`reasoning_effort` must be one of {list(VALID_REASONING_EFFORTS)} "
                f"(got {effort_raw!r})")
    tb_raw = _lookup_request_field(body, "thinking_budget")
    if tb_raw is None:
        thinking_budget = defaults.get("thinking_budget")
    else:
        try:
            thinking_budget = int(tb_raw)
        except (TypeError, ValueError):
            return None, "`thinking_budget` must be an integer"
        if thinking_budget < 0:
            return None, "`thinking_budget` must be >= 0"
    answer_reserve = int(defaults.get("answer_reserve", DEFAULT_ANSWER_RESERVE))
    ar_raw = _lookup_request_field(body, "answer_reserve")
    if ar_raw is not None:
        try:
            answer_reserve = int(ar_raw)
        except (TypeError, ValueError):
            return None, "`answer_reserve` must be an integer"
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
        enable_thinking = enable_thinking,
        reasoning_effort = reasoning_effort,
        thinking_budget = thinking_budget,
        answer_reserve = answer_reserve,
    ), None


async def chat_completions(request):
    app = request.app
    generator, tokenizer = app["generator"], app["tokenizer"]
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
    req, err = parse_request(body, request.app.get("gen_defaults") or SERVER_DEFAULTS)
    if err:
        return web.json_response({"error": {"message": err}}, status = 400)

    import asyncio
    gen_kwargs = dict(
        enable_thinking = req["enable_thinking"],
        reasoning_effort = req["reasoning_effort"],
        thinking_budget = req["thinking_budget"],
        answer_reserve = req["answer_reserve"],
    )
    if not req["stream"]:
        try:
            text, calls, finish, ptoks, otoks, reasoning, content = await asyncio.to_thread(
                generate_full, generator, tokenizer, req["messages"],
                req["max_tokens"], req["temperature"], req["top_p"], req["top_k"],
                req["seed"], req["tools"], req["tool_choice"], req["stop"],
                None, **gen_kwargs)
        except AssertionError as e:
            return web.json_response(
                {"error": {"message": f"context/cache: {e}", "type": "invalid_request_error"}},
                status = 400)
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
        "Connection": "keep-alive", "X-Accel-Buffering": "no"})
    await resp.prepare(request)
    cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    model_id = req["model_id"]
    req_schemas = build_tool_schemas(req["tools"])

    async def run():
        loop = asyncio.get_event_loop()
        queue = asyncio.Queue()

        def on_text(chunk):
            loop.call_soon_threadsafe(queue.put_nowait, ("delta", chunk))

        forced_choice = req["tool_choice"] not in (None, "auto", "none")

        def worker():
            try:
                text, calls, finish, ptoks, otoks, reasoning, content = generate_full(
                    generator, tokenizer, req["messages"], req["max_tokens"],
                    req["temperature"], req["top_p"], req["top_k"],
                    req["seed"], req["tools"], req["tool_choice"], req["stop"],
                    on_text = None if forced_choice else on_text, **gen_kwargs)
                loop.call_soon_threadsafe(queue.put_nowait,
                                          ("done", (calls, finish, reasoning, content)))
            except Exception as e:
                loop.call_soon_threadsafe(queue.put_nowait, ("error", str(e)))
        loop.run_in_executor(None, worker)

        async def send(delta, finish = None):
            obj = {"id": cid, "object": "chat.completion.chunk",
                   "created": int(time.time()), "model": model_id,
                   "choices": [{"index": 0, "delta": delta,
                                "finish_reason": finish}]}
            await resp.write(f"data: {json.dumps(obj)}\n\n".encode())

        pending, finish, calls_emitted = "", None, False
        call_idx = [0]
        # Template opens an unclosed <think> only when thinking is enabled.
        in_think = [bool(req["enable_thinking"])]

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

        while True:
            try:
                kind, payload = await asyncio.wait_for(
                    queue.get(), timeout = SSE_KEEPALIVE_S)
            except asyncio.TimeoutError:
                # Comment ping: keeps proxies/clients from dropping a long
                # prefill or a long think that has not yet closed.
                try:
                    await resp.write(b": ping\n\n")
                except ConnectionResetError:
                    return
                continue
            if kind == "error":
                await resp.write(
                    f'data: {json.dumps({"error": {"message": payload}})}\n\n'.encode())
                break
            if kind == "delta":
                pending += payload
                await flush_pending()
            elif kind == "done":
                calls, finish, reasoning, content = payload
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
                break
        await resp.write_eof()
    try:
        await run()
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
    ap.add_argument("-cs", "--cache_size", type = int, default = 65536,
                    help = "KV cache size in tokens (default 65536; 8192 default "
                           "of model_init is too small for large tool sets)")
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
    ap.add_argument("--default_max_tokens", type = int, default = DEFAULT_MAX_TOKENS,
                    help = "max_tokens used when the request omits it (thinking "
                           "models need far more than 1024; default 16384)")
    ap.add_argument("--reasoning_effort", type = str, default = DEFAULT_REASONING_EFFORT,
                    choices = list(VALID_REASONING_EFFORTS),
                    help = "Qwen chat-template reasoning_effort when the request "
                           "does not set one (low|medium|xhigh)")
    ap.add_argument("--thinking_budget", type = int, default = None,
                    help = "max new tokens inside <think> before force-closing. "
                           "Default depends on reasoning_effort "
                           f"(low={EFFORT_THINKING_BUDGET['low']}, "
                           f"medium={EFFORT_THINKING_BUDGET['medium']}, "
                           f"xhigh={EFFORT_THINKING_BUDGET['xhigh']}). "
                           "0 = no cap besides --answer_reserve")
    ap.add_argument("--answer_reserve", type = int, default = DEFAULT_ANSWER_RESERVE,
                    help = "tokens reserved for the answer so thinking cannot "
                           "consume the entire max_tokens budget")
    ap.add_argument("--no_thinking", action = "store_true",
                    help = "disable thinking by default (request can still "
                           "override with enable_thinking=true)")
    args = ap.parse_args()
    _draft = args.draft_model.lower()
    use_mtp = _draft == "mtp"
    use_draft = _draft not in ("none", "", "-")
    argv = ["-m", args.model,
            "-gs", str(args.grid_size), "-cs", str(args.cache_size)]
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
          + " ...", flush = True)
    generator, tokenizer = build_model(argv, use_draft = use_draft)
    stats["context_length"] = int(args.cache_size)
    SERVER_DEFAULTS["max_tokens"] = int(args.default_max_tokens)
    SERVER_DEFAULTS["reasoning_effort"] = args.reasoning_effort
    SERVER_DEFAULTS["thinking_budget"] = args.thinking_budget
    SERVER_DEFAULTS["answer_reserve"] = int(args.answer_reserve)
    SERVER_DEFAULTS["enable_thinking"] = not args.no_thinking
    think_cap, reserve = resolve_think_limits(
        SERVER_DEFAULTS["max_tokens"], SERVER_DEFAULTS["enable_thinking"],
        SERVER_DEFAULTS["reasoning_effort"], SERVER_DEFAULTS["thinking_budget"],
        SERVER_DEFAULTS["answer_reserve"])
    print(" == model ready; accepting requests", flush = True)
    print(f" == thinking: enable={SERVER_DEFAULTS['enable_thinking']} "
          f"effort={SERVER_DEFAULTS['reasoning_effort']} "
          f"default_max_tokens={SERVER_DEFAULTS['max_tokens']} "
          f"think_cap={think_cap} answer_reserve={reserve}", flush = True)

    app = web.Application(client_max_size = args.max_body_mb * 1024 * 1024)
    app["generator"] = generator
    app["tokenizer"] = tokenizer
    app["max_body_mb"] = args.max_body_mb
    app["gen_defaults"] = dict(SERVER_DEFAULTS)
    app.router.add_get("/v1/models", models)
    app.router.add_get("/health", health)
    app.router.add_post("/v1/chat/completions", chat_completions)
    web.run_app(app, host = args.host, port = args.port, print = None,
                keepalive_timeout = 3600.0)


if __name__ == "__main__":
    main()
