#!/usr/bin/env python3
"""Check a running serve_openai for concurrency, shared KV, and prefill cost.

There is no time-based expiry. A chat that sits idle keeps its prefix until
some later job needs the pages and the tail-first eviction drops them. After
that, the next visit prefills those tokens again.

Usage:
  .venv/bin/python tools/verify_kv.py --url http://127.0.0.1:8888
  .venv/bin/python tools/verify_kv.py --url http://127.0.0.1:8888 concurrency
  .venv/bin/python tools/verify_kv.py --url http://127.0.0.1:8888 share
  .venv/bin/python tools/verify_kv.py --url http://127.0.0.1:8888 pressure

`pressure` fills the pool. On a multi-million-token cache it stops instead of
generating for hours. Point it at a server started with a small -cs (8192 or
16384) to see eviction.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

PAGE = 256


def health(url: str) -> dict:
    with urllib.request.urlopen(url.rstrip("/") + "/health", timeout = 10) as resp:
        return json.loads(resp.read().decode())


def complete(url: str, content: str, max_tokens: int, tag: str) -> dict:
    body = {
        "model": "verify",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data = data,
        headers = {"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout = 600) as resp:
        payload = json.loads(resp.read().decode())
    wall = time.perf_counter() - t0
    usage = payload["usage"]
    return {
        "tag": tag,
        "wall": wall,
        "finish": payload["choices"][0]["finish_reason"],
        "prompt": usage["prompt_tokens"],
        "completion": usage["completion_tokens"],
        "cached": usage.get("cached_tokens", 0),
        "prefill_s": usage.get("time_prefill", 0.0),
    }


def report(row: dict) -> None:
    print(
        f"  {row['tag']}: prompt={row['prompt']} cached={row['cached']} "
        f"completion={row['completion']} prefill={row['prefill_s']:.3f}s "
        f"wall={row['wall']:.3f}s {row['finish']}"
    )


def blob(label: str, n_chars: int) -> str:
    # Distinct, incompressible-enough text so pages do not hash together.
    unit = f"[{label}] 海面上的燈塔每次只回答這一段編號，不要重複別的對話。"
    out = []
    while sum(len(x) for x in out) < n_chars:
        out.append(unit)
    return "".join(out)[:n_chars]


def concurrency(url: str, levels: list[int], max_tokens: int) -> None:
    print("\n== (1) concurrent users ==")
    h = health(url)
    print(f"  server max_seqs={h.get('max_seqs')} context={h.get('context_length')}")

    def one(i: int, barrier: int) -> tuple:
        while time.perf_counter_ns() < barrier:
            pass
        t0 = time.perf_counter()
        row = complete(url, f"連續寫第{i}段，主題是雨，寫到長度上限。", max_tokens, str(i))
        return t0, time.perf_counter(), row

    for n in levels:
        # One short wave first so a new batch width can compile.
        barrier = time.perf_counter_ns() + 200_000_000
        with ThreadPoolExecutor(max_workers = n) as ex:
            futs = [ex.submit(one, i, barrier) for i in range(n)]
            for fut in as_completed(futs):
                fut.result()
        samples = []
        stop_at = time.perf_counter() + 120
        barrier = time.perf_counter_ns() + 200_000_000
        rows = []
        with ThreadPoolExecutor(max_workers = n) as ex:
            futs = [ex.submit(one, i, barrier) for i in range(n)]
            while any(not f.done() for f in futs) and time.perf_counter() < stop_at:
                try:
                    samples.append(health(url).get("active_jobs"))
                except Exception:
                    samples.append(None)
                time.sleep(0.05)
            for fut in as_completed(futs):
                rows.append(fut.result())
        base = min(r[0] for r in rows)
        span = max(r[1] for r in rows) - base
        toks = sum(r[2]["completion"] for r in rows)
        act = [s for s in samples if isinstance(s, int)]
        print(
            f"  n={n} span={span:.3f}s tokens={toks} "
            f"agg={toks / span:.1f} tok/s per={toks / n / span:.1f} "
            f"active_max={max(act) if act else '-'}"
        )
        if n > (h.get("max_seqs") or 1) and act and max(act) < n:
            print(f"  note: only {max(act)} decoded at once; the rest waited in queue")


def share_and_pressure(url: str) -> None:
    print("\n== (2) shared prefix, then a full pool ==")
    h = health(url)
    cache = h.get("cache") or {}
    pages = int(cache.get("pages") or 0)
    print(
        f"  pages={pages} referenced={cache.get('referenced_pages')} "
        f"unreferenced={cache.get('unreferenced_pages')} "
        f"evictions_live={cache.get('evictions_live')}"
    )
    if pages <= 0:
        print("  no cache stats; restart the server so /health includes cache")
        return
    if pages > 128:
        print(
            f"  pool is {pages} pages ({pages * PAGE} tokens). "
            "Filling it would take a long time. Skip eviction; "
            "restart with -cs 8192 or 16384 to watch pages get pruned."
        )
        return

    prefix = blob("shared-prefix", 900)
    before = health(url)["cache"]["evictions_live"]
    first = complete(url, prefix + blob("user-A", 1800), 8, "A-cold")
    report(first)
    second = complete(url, prefix + blob("user-B", 1800), 8, "B-shares-prefix")
    report(second)
    print(
        f"  shared prefix cached_tokens={second['cached']} "
        f"(0 means the hybrid checkpoint did not cover the shared pages yet)"
    )

    print("  filling with unique prompts until a live page is evicted")
    seen = before
    last = None
    for i in range(pages + 4):
        last = complete(url, blob(f"unique-{i}-{time.time_ns()}", 2200), 8, f"fill-{i}")
        now = health(url)["cache"]
        if now["evictions_live"] > seen:
            print(
                f"  eviction at fill {i}: evictions_live {seen} -> {now['evictions_live']} "
                f"unreferenced={now['unreferenced_pages']}/{now['pages']}"
            )
            seen = now["evictions_live"]
            break
    else:
        now = health(url)["cache"]
        print(
            f"  no live eviction after {pages + 4} prompts. "
            f"evictions_live={now['evictions_live']} "
            f"unreferenced={now['unreferenced_pages']}/{now['pages']}"
        )

    again = complete(url, prefix + blob("user-A", 1800), 8, "A-after-pressure")
    report(again)
    if again["cached"] + PAGE < first["prompt"] and again["prefill_s"] > second["prefill_s"]:
        print("  the shared prefix lost pages; the next prefill did more work")
    elif again["cached"] > 0:
        print("  some shared prefix pages survived tail-first eviction")


def expiry(url: str, pause_s: float) -> None:
    print("\n== (3) idle does not expire; a missed prefix prefills again ==")
    text = blob("session-keep", 3500)
    cold = complete(url, text, 8, "cold")
    report(cold)
    warm = complete(url, text, 8, "immediate-replay")
    report(warm)
    time.sleep(pause_s)
    held = complete(url, text, 8, f"after-{pause_s:.0f}s-idle")
    report(held)
    if held["cached"] >= warm["cached"] and held["prefill_s"] <= max(warm["prefill_s"] * 1.5, warm["prefill_s"] + 0.5):
        print("  idle did not drop the prefix. There is no 30-minute expiry.")
    else:
        print("  replay after idle was more expensive; check evictions from other traffic")
    other = complete(url, blob(f"other-{time.time_ns()}", 3500), 8, "different-conversation")
    report(other)
    print(
        f"  a new conversation prefilled in {other['prefill_s']:.3f}s "
        f"with cached={other['cached']}. "
        "That is the cost of coming back after the old pages were evicted."
    )
    if warm["cached"] > cold["cached"]:
        saved = warm["cached"]
        print(f"  keeping the prefix avoided recomputing about {saved} tokens")


def main() -> None:
    ap = argparse.ArgumentParser(description = "Verify concurrent use and KV reuse.")
    ap.add_argument("--url", default = "http://127.0.0.1:8888")
    ap.add_argument("--pause", type = float, default = 20.0, help = "Idle wait in the expiry check")
    ap.add_argument("--tokens", type = int, default = 64, help = "Completion length for the concurrency check")
    ap.add_argument("mode", nargs = "?", default = "all", choices = ("all", "concurrency", "share", "pressure", "expiry"))
    args = ap.parse_args()
    if args.mode in ("all", "concurrency"):
        concurrency(args.url, [1, 2, 4, 8], args.tokens)
    if args.mode in ("all", "share", "pressure"):
        share_and_pressure(args.url)
    if args.mode in ("all", "expiry"):
        expiry(args.url, args.pause)


if __name__ == "__main__":
    main()
