import json, sys, time, urllib.request

def probe(target, depth):
    filler = ("The quarterly logistics review covered warehouse throughput, carrier reliability, "
              "customs delays, and the seasonal demand forecast for the northern region. ")
    words = int(target / 1.25)
    reps = max(1, words // len(filler.split()))
    k = int(reps * depth)
    p = filler * k + "\n\nIMPORTANT NOTE: the secret vault code is 7391-ALPHA-ZEBRA. Remember it.\n\n" + filler * (reps - k)
    p += "\n\nQuestion: What is the secret vault code mentioned above? Just the code."
    body = json.dumps({"model": "glm-flash", "messages": [{"role": "user", "content": p}],
                       "max_tokens": 800, "temperature": 0}).encode()
    req = urllib.request.Request("http://localhost:8093/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=1800))
    m = r["choices"][0]["message"]
    text = (m.get("content") or "") + (m.get("reasoning_content") or "")
    ok = "7391" in text
    status = "PASS" if ok else "FAIL"
    print(f"needle {target} depth={depth}: {status} in {time.time()-t0:.1f}s", flush=True)

probe(131072, 0.5)
probe(196608, 0.5)
probe(262144, 0.5)
probe(262144, 0.95)
print("NEEDLE_DONE", flush=True)
