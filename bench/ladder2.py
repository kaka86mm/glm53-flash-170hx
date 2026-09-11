import json, time, urllib.request

def probe(target):
    filler = ("The quarterly logistics review covered warehouse throughput, carrier reliability, "
              "customs delays, and the seasonal demand forecast for the northern region. ")
    words = int(target / 1.25)
    p = filler * max(1, words // len(filler.split())) + "\n\nQuestion: Summarize in one sentence."
    body = json.dumps({"model": "glm-flash", "messages": [{"role": "user", "content": p}],
                       "max_tokens": 8, "temperature": 0}).encode()
    req = urllib.request.Request("http://localhost:8093/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=900))
    u = r["usage"]
    print(f"PROBE {target}: OK {u['prompt_tokens']} tok in {time.time()-t0:.2f}s", flush=True)

for t in [98304, 114688, 131072, 163840, 196608, 262144]:
    try:
        probe(t)
    except Exception as e:
        print(f"PROBE {t}: FAIL {str(e)[:80]}", flush=True)
print("LADDER_DONE", flush=True)
