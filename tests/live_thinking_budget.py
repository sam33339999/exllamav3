#!/usr/bin/env python3
"""Live checks against a running serve_openai on :8888.

Exercises:
  1. /health exposes thinking defaults
  2. enable_thinking=false still returns an answer
  3. a tiny thinking_budget force-closes </think> and still produces content
  4. streaming yields reasoning_content then content, and finishes cleanly
"""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8888"
CHAT = URL.rstrip("/") + "/v1/chat/completions"
HEALTH = URL.rstrip("/") + "/health"


def get(url):
    with urllib.request.urlopen(url, timeout = 30) as r:
        return json.loads(r.read().decode())


def post(body, timeout = 180, stream = False):
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        CHAT, data = data,
        headers = {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout = timeout) as r:
        if not stream:
            return json.loads(r.read().decode()), r.headers
        raw = r.read().decode()
        return raw, r.headers


def fail(msg):
    print("FAIL:", msg)
    sys.exit(1)


def main():
    health = get(HEALTH)
    print("health:", json.dumps(health, sort_keys = True))
    if not health.get("ok"):
        fail("health.ok is not true")
    if health.get("default_max_tokens", 0) < 8192:
        fail(f"default_max_tokens too small: {health.get('default_max_tokens')}")
    if health.get("answer_reserve", 0) < 1:
        fail("answer_reserve missing")

    # --- no-think, short ---
    body = {
        "messages": [{"role": "user", "content": "Reply with exactly the word PONG."}],
        "max_tokens": 32,
        "enable_thinking": False,
        "temperature": 0.0,
        "stream": False,
    }
    resp, _ = post(body, timeout = 60)
    msg = resp["choices"][0]["message"]
    content = (msg.get("content") or "").strip()
    print("no-think content:", repr(content[:200]))
    print("no-think finish:", resp["choices"][0].get("finish_reason"),
          "usage:", resp.get("usage"))
    if "PONG" not in content.upper():
        fail(f"enable_thinking=false did not answer PONG: {content!r}")

    # --- force-close a long think with a tiny budget ---
    body = {
        "messages": [{"role": "user",
                      "content": "Write a careful multi-step proof that there are infinitely many primes. Consider Euclid, Euler, and at least one other approach."}],
        "max_tokens": 96,
        "thinking_budget": 16,
        "answer_reserve": 48,
        "reasoning_effort": "xhigh",
        "enable_thinking": True,
        "temperature": 0.6,
        "stream": False,
    }
    resp, _ = post(body, timeout = 180)
    choice = resp["choices"][0]
    msg = choice["message"]
    reasoning = msg.get("reasoning_content") or ""
    content = msg.get("content") or ""
    finish = choice.get("finish_reason")
    usage = resp.get("usage") or {}
    print("force-close finish:", finish, "usage:", usage)
    print("force-close reasoning chars:", len(reasoning),
          "head:", repr(reasoning[:120]))
    print("force-close content chars:", len(content),
          "head:", repr(content[:200]))
    if not content.strip():
        fail("force-close left content empty (think ate max_tokens)")
    if "</think>" in (content + reasoning):
        fail("raw </think> leaked into OpenAI fields")
    # Completion tokens should exceed the think cap — we continued into the answer.
    if int(usage.get("completion_tokens") or 0) <= 16:
        fail(f"did not continue after think cap: usage={usage}")

    # --- stream: thinking on, tiny budget ---
    body["stream"] = True
    raw, headers = post(body, timeout = 180, stream = True)
    ctype = headers.get("Content-Type", "")
    if "text/event-stream" not in ctype:
        fail(f"stream Content-Type={ctype!r}")
    events = []
    saw_ping = False
    for line in raw.splitlines():
        if line.startswith(": "):
            saw_ping = True
            continue
        if not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload.strip() == "[DONE]":
            events.append({"done": True})
            continue
        events.append(json.loads(payload))
    reasoning_parts, content_parts = [], []
    finish_s = None
    for ev in events:
        if ev.get("done"):
            continue
        if ev.get("error"):
            fail(f"stream error event: {ev['error']}")
        ch = (ev.get("choices") or [{}])[0]
        delta = ch.get("delta") or {}
        if delta.get("reasoning_content"):
            reasoning_parts.append(delta["reasoning_content"])
        if delta.get("content"):
            content_parts.append(delta["content"])
        if ch.get("finish_reason"):
            finish_s = ch["finish_reason"]
    stream_reason = "".join(reasoning_parts)
    stream_content = "".join(content_parts)
    print("stream finish:", finish_s, "ping:", saw_ping,
          "reason chars:", len(stream_reason),
          "content chars:", len(stream_content),
          "events:", len(events))
    if not any(ev.get("done") for ev in events):
        fail("stream missing [DONE]")
    if not stream_content.strip():
        fail("stream force-close left content empty")
    if finish_s is None:
        fail("stream missing finish_reason")

    print("OK")


if __name__ == "__main__":
    main()
