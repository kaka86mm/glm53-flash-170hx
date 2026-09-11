import json, sys, time, threading, urllib.request
import os
BASE=os.environ.get("BASE","http://localhost:8093"); N=int(sys.argv[1]); DUR=float(sys.argv[2]) if len(sys.argv)>2 else 40
PROMPTS=["用Python写一个快速排序，并解释复杂度。","Explain the difference between TCP and UDP in detail.","写一段关于秋天的散文","Solve step by step: what is 847 times 23?","介绍一下光合作用的过程。","Write a bash script that backs up a directory with rotation.","列举十个中国城市并各写一句话介绍。","What are the main causes of the French Revolution?","Output a JSON array of 20 objects with fields name, age, city.","Count from 1 to 100, comma separated."]
tot=[0]*N; reqs=[0]*N; stop=time.time()+DUR
def worker(i):
    j=i
    while time.time()<stop:
        p=PROMPTS[j%len(PROMPTS)]; j+=N
        body=json.dumps({"model":"glm-flash","messages":[{"role":"user","content":p}],"max_tokens":200,"temperature":0}).encode()
        r=json.load(urllib.request.urlopen(urllib.request.Request(f"{BASE}/v1/chat/completions",data=body,headers={"Content-Type":"application/json"}),timeout=600))
        tot[i]+=r["usage"]["completion_tokens"]; reqs[i]+=1
t0=time.time(); th=[threading.Thread(target=worker,args=(i,)) for i in range(N)]
[t.start() for t in th]; [t.join() for t in th]; dt=time.time()-t0
print(f"steady N={N} dur={dt:.1f}s tokens={sum(tot)} reqs={sum(reqs)} aggregate={sum(tot)/dt:.1f} t/s per-stream={sum(tot)/dt/N:.1f}")
