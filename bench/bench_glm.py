#!/usr/bin/env python3
"""Single-stream spec-decode benchmark via /metrics counter deltas."""
import json
import time
import urllib.request

import os
BASE = os.environ.get("BASE", "http://localhost:9004")

WORKLOADS = [
    ("counting", "Count from 1 to 100, comma separated:", 250),
    ("repetition", "Repeat exactly 30 times the line: hello world foo bar", 200),
    ("json", "Output a JSON array of 20 objects with fields name, age, city. Only JSON:", 300),
    ("code", "用Python写一个快速排序，并解释复杂度。", 300),
    ("math", "Solve step by step: what is 847 times 23?", 250),
    ("prose", "写一段关于秋天的散文", 250),
]


def metrics():
    txt = urllib.request.urlopen(f"{BASE}/metrics", timeout=10).read().decode()
    out = {}
    for line in txt.splitlines():
        if line.startswith("#"):
            continue
        if "spec_decode_num" in line and "created" not in line:
            name = line.split("{")[0]
            val = float(line.rsplit(" ", 1)[1])
            out.setdefault(name, []).append(val)
    return {k: sum(v) for k, v in out.items()}


def one(name, prompt, max_tokens):
    a = metrics()
    body = json.dumps(
        {"model": "glm-flash", "prompt": prompt, "max_tokens": max_tokens,
         "temperature": 0}
    ).encode()
    req = urllib.request.Request(
        f"{BASE}/v1/completions", data=body,
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    resp = json.load(urllib.request.urlopen(req, timeout=400))
    dt = time.time() - t0
    b = metrics()
    n = resp["usage"]["completion_tokens"]
    drafts = b.get("vllm:spec_decode_num_drafts_total", 0) - a.get("vllm:spec_decode_num_drafts_total", 0)
    acc = b.get("vllm:spec_decode_num_accepted_tokens_total", 0) - a.get("vllm:spec_decode_num_accepted_tokens_total", 0)
    per = []
    for i in range(7):
        k = f"vllm:spec_decode_num_accepted_tokens_per_pos_total"
        pass
    al = 1 + acc / max(drafts, 1)
    # Degeneration guard: raw prompts at temperature 0 can fall into a
    # prompt-repeat loop, which inflates accept_len (~7.9). Flag it.
    text = resp["choices"][0]["text"]
    rep = text.count(text[:20]) if len(text) >= 20 else 0
    flag = "  DEGENERATE(repeat x%d)" % rep if rep >= 4 else ""
    print(f"{name:10s} tokens={n} t/s={n/dt:.1f} accept_len={al:.2f} "
          f"(drafts={drafts:.0f} accepted={acc:.0f}){flag}")


for name, prompt, mt in WORKLOADS:
    one(name, prompt, mt)
