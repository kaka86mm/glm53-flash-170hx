#!/usr/bin/env python3
"""Production acceptance test for dsv4-vision (GLM EXL3) via litellm."""
import base64
import io
import json
import os
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request

BASE = os.environ.get("LT_BASE", "http://YOUR_LITELLM_HOST:4000")
KEY = os.environ["LT_KEY"]
MODEL = "dsv4-vision"

PASS, FAIL, SKIP = [], [], []


def report(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)


def post(payload, timeout=1800, path="/v1/chat/completions", key=None):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        BASE + path, data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key or KEY}"})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=timeout))
    return r, time.time() - t0


def chat(msgs, **kw):
    p = {"model": MODEL, "messages": msgs, "temperature": 0}
    p.update(kw)
    return post(p)


def content_of(r):
    return r["choices"][0]["message"].get("content") or ""


# ---------- 1. functional quality ----------
QUALITY = [
    ("arith", "847*23=? 只给数字", "19481", 500),
    ("gsm", "水池进水管6小时注满,出水管9小时放空,同开几小时注满?只给数字", "18", 800),
    ("logic", "所有bloops都是razzies,所有razzies都是lazzies,所有bloops必然是lazzies吗?先答yes或no", "yes", 400),
    ("code", "写python函数is_prime(n)并列出30以下质数,简短", "29", 700),
    ("json", "输出一个JSON对象含name,age,city。只要JSON", "city", 300),
    ("cmp", "9.11和9.9哪个大?只给数字", "9.9", 400),
]
print("=== 1. Functional quality (via litellm) ===", flush=True)
for name, q, expect, mt in QUALITY:
    try:
        r, dt = chat([{"role": "user", "content": q}], max_tokens=mt)
        text = content_of(r) + (r["choices"][0]["message"].get("reasoning_content") or "")
        ok = expect.lower() in text.lower()
        report(f"quality.{name}", ok, f"| ans: {content_of(r)[:40]!r} {dt:.1f}s")
    except Exception as e:
        report(f"quality.{name}", False, f"| EXC {str(e)[:60]}")

# ---------- 2. thinking separation ----------
print("\n=== 2. Thinking separation ===", flush=True)
try:
    r, dt = chat([{"role": "user", "content": "847*23=? 思考后回答"}], max_tokens=600)
    m = r["choices"][0]["message"]
    rs = m.get("reasoning_content") or m.get("reasoning") or ""
    ct = m.get("content") or ""
    report("think.hard_reasoning_present", len(rs) > 20, f"| reasoning {len(rs)} chars")
    report("think.content_clean", len(ct) > 0 and "<think>" not in ct, f"| content: {ct[:40]!r}")
except Exception as e:
    report("think.*", False, f"| EXC {str(e)[:60]}")

# ---------- 3. tool calling ----------
print("\n=== 3. Tool calling ===", flush=True)
try:
    tools = [{"type": "function", "function": {
        "name": "get_weather", "description": "Get weather for a city",
        "parameters": {"type": "object",
                       "properties": {"city": {"type": "string"}},
                       "required": ["city"]}}}]
    r, dt = chat([{"role": "user", "content": "北京今天天气如何?用工具查"}],
                 tools=tools, tool_choice="auto", max_tokens=400)
    tc = r["choices"][0]["message"].get("tool_calls")
    ok = bool(tc) and r["choices"][0].get("finish_reason") == "tool_calls"
    args_ok = tc and "北京" in tc[0]["function"]["arguments"]
    report("tool.call", ok, f"| finish={r['choices'][0].get('finish_reason')}")
    report("tool.args", bool(args_ok), f"| {tc[0]['function']['arguments'][:50] if tc else '-'}")
except Exception as e:
    report("tool.*", False, f"| EXC {str(e)[:60]}")

# ---------- 4. vision ----------
print("\n=== 4. Vision ===", flush=True)
try:
    from PIL import Image, ImageDraw

    def img_b64(draw_fn):
        img = Image.new("RGB", (256, 256), "white")
        d = ImageDraw.Draw(img)
        draw_fn(d)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    def vask(draw_fn, q):
        payload = {"model": MODEL, "temperature": 0, "max_tokens": 300, "messages": [
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64(draw_fn)}"}},
                {"type": "text", "text": q}]}]}
        r, _ = post(payload)
        return content_of(r)

    def red(d): d.rectangle([20, 20, 236, 236], fill=(200, 30, 30))
    report("vision.color", "红" in vask(red, "图片主色?只给颜色名"), "")

    def chart(d):
        d.line([30, 220, 230, 220], fill="black", width=2)
        d.line([30, 100, 230, 100], fill="black", width=1)
        d.line([30, 160, 230, 160], fill="black", width=1)
        d.text((6, 96), "100", fill="black"); d.text((12, 156), "50", fill="black")
        d.rectangle([108, 100, 148, 220], fill="gray")
        d.text((122, 224), "B", fill="black")
    report("vision.chart", "100" in vask(chart, "轴刻度50和100,柱B的值?只给数字"), "")
except ImportError:
    SKIP.append("vision (no PIL)")
    print("[SKIP] vision (no PIL)", flush=True)

# ---------- 5. streaming ----------
print("\n=== 5. Streaming ===", flush=True)
try:
    payload = {"model": MODEL, "temperature": 0, "max_tokens": 300, "stream": True,
               "messages": [{"role": "user", "content": "数到50,逗号分隔"}]}
    body = json.dumps(payload).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {KEY}"})
    t0 = time.time()
    stamps = []
    resp = urllib.request.urlopen(req, timeout=300)
    for line in resp:
        s = line.decode().strip()
        if s.startswith("data:") and "[DONE]" not in s:
            d = json.loads(s[5:])
            delta = d["choices"][0].get("delta", {})
            if delta.get("content") or delta.get("reasoning_content"):
                stamps.append(time.time() - t0)
    itls = [b - a for a, b in zip(stamps, stamps[1:])]
    report("stream.chunks", len(stamps) > 50, f"| {len(stamps)} chunks")
    report("stream.ttft", stamps[0] < 30, f"| TTFT {stamps[0]:.1f}s")
    p95 = sorted(itls)[int(0.95 * len(itls))] if itls else 0
    report("stream.itl_p95", p95 < 1.0, f"| p95 {p95*1000:.0f}ms mean {statistics.mean(itls)*1000:.0f}ms")
except Exception as e:
    report("stream.*", False, f"| EXC {str(e)[:60]}")

# ---------- 6. long context ----------
print("\n=== 6. Long context (needle) ===", flush=True)
FILLER = ("The quarterly logistics review covered warehouse throughput, carrier reliability, "
          "and the seasonal demand forecast for the northern region. ")

def needle(target):
    words = int(target / 1.25)
    reps = max(1, words // len(FILLER.split()))
    k = reps // 2
    p = (FILLER * k + "\n\nIMPORTANT: vault code is 7391-ALPHA-ZEBRA.\n\n"
         + FILLER * (reps - k)
         + "\n\nVault code? 只给代码")
    r, dt = chat([{"role": "user", "content": p}], max_tokens=800)
    text = content_of(r) + (r["choices"][0]["message"].get("reasoning_content") or "")
    return "7391" in text, dt, len(r["usage"]["prompt_tokens"])

for tgt in (65536, 131072, 262144):
    try:
        ok, dt, pt = needle(tgt)
        report(f"needle.{tgt//1024}k", ok, f"| {pt} tok {dt:.0f}s")
    except Exception as e:
        report(f"needle.{tgt//1024}k", False, f"| EXC {str(e)[:60]}")

# long output
try:
    r, dt = chat([{"role": "user", "content": "详细介绍中国四大发明,每项至少800字"}], max_tokens=12000)
    ct = r["usage"]["completion_tokens"]
    report("longout.12k", ct > 4000 and r["choices"][0]["finish_reason"] in ("stop", "length"),
           f"| {ct} tokens finish={r['choices'][0]['finish_reason']} {dt:.0f}s")
except Exception as e:
    report("longout.12k", False, f"| EXC {str(e)[:60]}")

# ---------- 7. concurrency ----------
print("\n=== 7. Concurrency (mixed, 40s each) ===", flush=True)
PROMPTS = ["用Python写快排并解释复杂度。", "TCP和UDP的区别?", "写一段秋天的散文",
           "847*23 等于多少?一步步算", "介绍光合作用。", "写一个bash备份脚本",
           "列举十个中国城市各一句话。", "法国大革命的原因?", "输出20个JSON对象数组。",
           "从1数到100。"]

def conc_run(n, dur):
    tot = [0] * n
    errs = [0] * n
    stop = time.time() + dur
    def w(i):
        j = i
        while time.time() < stop:
            try:
                r, _ = post({"model": MODEL, "temperature": 0, "max_tokens": 200,
                             "messages": [{"role": "user", "content": PROMPTS[j % len(PROMPTS)]}]},
                            timeout=600)
                tot[i] += r["usage"]["completion_tokens"]
                j += 1
            except Exception:
                errs[i] += 1
    th = [threading.Thread(target=w, args=(i,)) for i in range(n)]
    t0 = time.time()
    [t.start() for t in th]; [t.join() for t in th]
    wall = time.time() - t0
    agg = sum(tot) / wall
    etot = sum(errs)
    report(f"conc.N{n}", agg > 30 and etot == 0,
           f"| agg {agg:.0f} tok/s, errs {etot}")

for n in (1, 4, 8, 16):
    conc_run(n, 40)

# ---------- 8. robustness ----------
print("\n=== 8. Robustness ===", flush=True)
try:
    post({"model": "nonexistent-model", "messages": [{"role": "user", "content": "x"}]}, timeout=30)
    report("rob.badmodel_rejected", False, "| accepted!?")
except urllib.error.HTTPError as e:
    report("rob.badmodel_rejected", e.code in (400, 404, 500), f"| HTTP {e.code}")
except Exception as e:
    report("rob.badmodel_rejected", False, f"| EXC {str(e)[:50]}")

try:
    r, _ = chat([{"role": "user", "content": "hi"}], max_tokens=999999)
    report("rob.huge_maxtokens", True, f"| accepted, finish={r['choices'][0]['finish_reason']}")
except urllib.error.HTTPError as e:
    report("rob.huge_maxtokens", e.code == 400, f"| HTTP {e.code} (clean reject)")
except Exception as e:
    report("rob.huge_maxtokens", False, f"| EXC {str(e)[:50]}")

# dsv4-flash alias
try:
    r, _ = post({"model": "dsv4-flash", "max_tokens": 50,
                 "messages": [{"role": "user", "content": "1+1=? 只给数字"}]})
    report("rob.alias_dsv4_flash", "2" in content_of(r), "")
except Exception as e:
    report("rob.alias_dsv4_flash", False, f"| EXC {str(e)[:50]}")

# ---------- 9. soak ----------
print("\n=== 9. Soak (8 workers, 6 min mixed) ===", flush=True)
soak_tot = [0] * 8
soak_err = [0] * 8
soak_lat = []
soak_stop = time.time() + 360

def soak_w(i):
    j = i
    while time.time() < soak_stop:
        try:
            t0 = time.time()
            r, _ = post({"model": MODEL, "temperature": 0, "max_tokens": 300,
                         "messages": [{"role": "user", "content": PROMPTS[j % len(PROMPTS)]}]},
                        timeout=600)
            dt = time.time() - t0
            soak_lat.append(dt)
            soak_tot[i] += r["usage"]["completion_tokens"]
            j += 1
        except Exception:
            soak_err[i] += 1
        time.sleep(0.5)

th = [threading.Thread(target=soak_w, args=(i,)) for i in range(8)]
t0 = time.time()
[t.start() for t in th]; [t.join() for t in th]
wall = time.time() - t0
err_rate = sum(soak_err) / max(1, sum(soak_err) + sum(1 for _ in soak_lat))
report("soak.zero_errors", sum(soak_err) == 0, f"| {sum(soak_err)} errors / {len(soak_lat)} reqs")
report("soak.throughput", sum(soak_tot) / wall > 50, f"| agg {sum(soak_tot)/wall:.0f} tok/s over {wall/60:.0f}min")
if soak_lat:
    p95 = sorted(soak_lat)[int(0.95 * len(soak_lat))]
    report("soak.latency_p95", p95 < 120, f"| p95 {p95:.0f}s mean {statistics.mean(soak_lat):.0f}s")

# ---------- summary ----------
print("\n" + "=" * 50, flush=True)
print(f"ACCEPTANCE: {len(PASS)} PASS / {len(FAIL)} FAIL / {len(SKIP)} SKIP", flush=True)
if FAIL:
    print("FAILED:", ", ".join(FAIL), flush=True)
    sys.exit(1)
print("ALL GREEN", flush=True)
