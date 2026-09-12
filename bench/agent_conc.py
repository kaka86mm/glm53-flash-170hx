#!/usr/bin/env python3
"""Concurrent long-context agent decode A/B for adaptive-k.

N threads each run salted ~CTX-token prompts (cold prefill), stream the
response, and measure true decode rate from first to last delta token.
Reports per-stream rates and aggregate.
"""
import json
import random
import string
import sys
import threading
import time
import urllib.request

import os
BASE = os.environ.get("BASE", "http://localhost:8093")
PORT = BASE.rsplit(":", 1)[1]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
CTX = int(sys.argv[2]) if len(sys.argv) > 2 else 49152
RUNS = int(sys.argv[3]) if len(sys.argv) > 3 else 2

FILLER = ("The quarterly logistics review covered warehouse throughput, carrier reliability, "
          "and the seasonal demand forecast for the northern region. ")
TASKS = [
    "summarize the key risks, reason about root causes, and propose concrete next steps",
    "audit the numbers, point out inconsistencies, and draft a follow-up question list",
    "build a rollout plan with milestones, owners, and rollback criteria",
]


def one_round(rates, idx):
    salt = "".join(random.choices(string.ascii_lowercase, k=12))
    words = int(CTX / 1.25)
    reps = max(1, words // len(FILLER.split()))
    task = TASKS[idx % len(TASKS)]
    p = ("Reference tag " + salt + ". " + FILLER * reps +
         f"\n\nYou are reviewing the report above. Think through it carefully: {task}. "
         "Write your analysis in flowing prose.")
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
    if ntok and t_last and t_last - t_first >= 0.5:
        rates.append((ntok - 1) / (t_last - t_first))


for _ in range(RUNS):
    rates = []
    th = [threading.Thread(target=one_round, args=(rates, i)) for i in range(N)]
    [t.start() for t in th]
    [t.join() for t in th]
    print(f"round: per-stream {' '.join(f'{r:.1f}' for r in sorted(rates))} "
          f"| aggregate {sum(rates):.1f} tok/s", flush=True)
print("CONC_DONE", flush=True)
