"""Resumable core-measurement driver (robust to 45s sandbox cap; no background needed).
Processes one seed per invocation into core_raw.json, skipping finished seeds.
Re-invoke until it prints ALL_DONE, then aggregate with agg_core.py."""
import numpy as np, json, os, sys, time
from latap_v2 import GMM, run, measure_proxy, energy_distance, FAM, TAYLOR, ONLINE
NSEED=12; T=36; STRIDE=3; B=112; N=40; F="core_raw.json"
raw=json.load(open(F)) if os.path.exists(F) else {"Q1":{}, "Q2":{}, "Q3":{}}
done=set(raw["Q1"].keys())
todo=[s for s in range(NSEED) if str(s) not in done]
if not todo:
    print("ALL_DONE", len(done), "seeds"); sys.exit(0)
budget=time.time()+24
for s in todo:
    if time.time()>budget: break
    t0=time.time(); g=GMM(seed=s)
    # Q1 proxy quality
    mp=measure_proxy(g,T=T,stride=STRIDE,B=B,N=N,seed=s)
    raw["Q1"][str(s)]={m:{"sp":float(np.nanmean(np.concatenate(mp[m]["sp"]))),
                          "top1":float(np.mean(mp[m]["top1"]))} for m in ("cos","l2","l1")}
    van=run(g,selector="vanilla",T=T,stride=STRIDE,B=B,N=N,seed=s)
    # Q2 family
    raw["Q2"][str(s)]={name:float(energy_distance(van,run(g,selector="probe",proxy="l2",
        T=T,stride=STRIDE,B=B,N=N,seed=s,allowed=al))) for name,al in
        (("taylor",TAYLOR),("online",ONLINE),("all",None))}
    # Q3 selectors (naive best-fixed searched over family)
    fixed=[float(energy_distance(van,run(g,selector="naive",fixed_pred=i,T=T,stride=STRIDE,B=B,N=N,seed=s))) for i in range(len(FAM))]
    raw["Q3"][str(s)]={
        "naive_bestfixed":min(fixed), "best_fixed_idx":int(np.argmin(fixed)),
        "probe_cos":float(energy_distance(van,run(g,selector="probe",proxy="cos",T=T,stride=STRIDE,B=B,N=N,seed=s))),
        "probe_l2": float(energy_distance(van,run(g,selector="probe",proxy="l2",T=T,stride=STRIDE,B=B,N=N,seed=s))),
        "oracle":   float(energy_distance(van,run(g,selector="oracle",T=T,stride=STRIDE,B=B,N=N,seed=s))),
        "random":   float(energy_distance(van,run(g,selector="random",T=T,stride=STRIDE,B=B,N=N,seed=s)))}
    json.dump(raw,open(F,"w"),indent=2)
    print(f"seed {s} done in {time.time()-t0:.1f}s  cos {raw['Q1'][str(s)]['cos']['sp']:.3f} l2 {raw['Q1'][str(s)]['l2']['sp']:.3f} | online {raw['Q2'][str(s)]['online']:.4f} taylor {raw['Q2'][str(s)]['taylor']:.4f} | probeL2 {raw['Q3'][str(s)]['probe_l2']:.4f} naive {raw['Q3'][str(s)]['naive_bestfixed']:.4f}",flush=True)
left=[s for s in range(NSEED) if str(s) not in raw["Q1"]]
print("remaining", left if left else "NONE -> run again for ALL_DONE")
