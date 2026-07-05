"""Probe-richness sweep: Spearman(proxy_err,true_err) as the probe goes from hard
top-1 (shallow/first-layer analog) to the full soft mixture (deep/exact). Multi-seed."""
import numpy as np, json
from latap_v2 import GMM, measure_proxy
MS=[1,2,3,6]; SEEDS=5; out={m:[] for m in MS}
for s in range(SEEDS):
    g=GMM(K=6,seed=s)
    for m in MS:
        mp=measure_proxy(g,T=36,stride=3,B=80,N=32,seed=s,probe_m=m,metrics=("l2",))
        out[m].append(float(np.nanmean(np.concatenate(mp["l2"]["sp"]))))
def ci(v):
    v=np.array(v); r=np.random.default_rng(0); bs=[v[r.integers(0,len(v),len(v))].mean() for _ in range(2000)]
    return [float(v.mean()),float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))]
R={str(m):ci(out[m]) for m in MS}
json.dump(R,open("probe_depth.json","w"),indent=2)
for m in MS: print(f"top-{m} probe: Spearman {R[str(m)][0]:.3f} [{R[str(m)][1]:.3f},{R[str(m)][2]:.3f}]")
