#!/usr/bin/env python3
"""Agent-style long-context decode A/B for adaptive-k.

Streaming request; true decode rate = (tokens-1)/(t_last - t_first) over ANY
delta text (reasoning or content), so prefill time and network are excluded.
Salted filler => cold prefill every run (no prefix-cache reuse across runs).
"""
import json
import random
import string
import sys
import time
import urllib.request

import os
BASE = os.environ.get("BASE", "http://localhost:8093")
PORT = BASE.rsplit(":", 1)[1]
SIZES = [32768, 131072]
RUNS = 2

FILLER = ("The quarterly logistics review covered warehouse throughput, carrier reliability, "
          "and the seasonal demand forecast for the northern region. ")


def run(ctx_tokens):
    salt = "".join(random.choices(string.ascii_lowercase, k=12))
    words = int(ctx_tokens / 1.25)
    reps = max(1, words // len(FILLER.split()))
    p = ("Reference tag " + salt + ". " + FILLER * reps +
         "\n\nYou are reviewing the report above. Think through it carefully: "
         "summarize the key risks, reason about root causes, and propose "
         "concrete next steps. Write your analysis in flowing prose.")
    body = json.dumps({
        "model": "glm-flash",
        "messages": [{"role": "user", "content": p}],
        "max_tokens": 400, "temperature": 0, "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(f"{BASE}/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    t_first = t_last = None
    ntok = 0
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.strip()
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                break
            try:
                d = json.loads(payload)
            except Exception:
                continue
            if d.get("usage"):
                ntok = d["usage"].get("completion_tokens", ntok) or ntok
            delta = (d.get("choices") or [{}])[0].get("delta") or {}
            if delta.get("content") or delta.get("reasoning"):
                now = time.time()
                t_first = now if t_first is None else t_first
                t_last = now
    if not ntok:
        return None
    if t_last is None or t_last - t_first < 0.5:
        return None
    return (ntok - 1) / (t_last - t_first)


for size in SIZES:
    rates = [r for r in (run(size) for _ in range(RUNS)) if r]
    if rates:
        med = sorted(rates)[len(rates) // 2]
        print(f"ctx~{size//1024:>4}k: median decode {med:.1f} tok/s "
              f"(runs: {' '.join(f'{r:.1f}' for r in rates)})", flush=True)
    else:
        print(f"ctx~{size//1024:>4}k: FAILED", flush=True)
print("AB_DONE", flush=True)
