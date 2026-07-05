"""Resumable regime-sweep / pressure harness, checkpointed per (cell, seed).
Maps WHEN each lever (L2 proxy, online predictors, probe-then-select) helps across
acceleration / dimension / separation / noise / #components. Re-invoke until ALL_DONE."""
import numpy as np, json, os, sys, time
from latap_v2 import GMM, run, measure_proxy, energy_distance, FAM, TAYLOR, ONLINE
SEEDS=5; B=96; N=32
FIXED_CAND=[0,1,3,5]   # T0h0, T1h1, T2h1, OLS2w5 -- representative naive candidates

def eval_seed(cfg, s):
    K=cfg.get("K",6); d=cfg.get("d",8); sigma=cfg.get("sigma",0.35); spread=cfg.get("spread",2.2)
    stride=cfg.get("stride",3); T=cfg.get("T",36)
    b = 64 if d>=16 else B    # cheaper batch for expensive high-dim lstsq
    g=GMM(K=K,d=d,sigma=sigma,spread=spread,seed=s)
    mp=measure_proxy(g,T=T,stride=stride,B=b,N=N,seed=s)
    van=run(g,selector="vanilla",T=T,stride=stride,B=b,N=N,seed=s)
    def E(**kw): return float(energy_distance(van,run(g,T=T,stride=stride,B=b,N=N,seed=s,**kw)))
    fixed=[E(selector="naive",fixed_pred=i) for i in FIXED_CAND]
    return {"q1_cos":float(np.nanmean(np.concatenate(mp["cos"]["sp"]))),
            "q1_l2":float(np.nanmean(np.concatenate(mp["l2"]["sp"]))),
            "q1_top1_cos":float(np.mean(mp["cos"]["top1"])),
            "q1_top1_l2":float(np.mean(mp["l2"]["top1"])),
            "e_taylor":E(selector="probe",proxy="l2",allowed=TAYLOR),
            "e_online":E(selector="probe",proxy="l2",allowed=ONLINE),
            "e_all":E(selector="probe",proxy="l2"),
            "e_naive":float(min(fixed)), "best_fixed_idx":int(FIXED_CAND[int(np.argmin(fixed))]),
            "e_probe_cos":E(selector="probe",proxy="cos"),
            "e_probe_l2":E(selector="probe",proxy="l2"),
            "e_oracle":E(selector="oracle"), "e_random":E(selector="random")}

def cells():
    C=[]
    for st in (2,3,4,6,8): C.append((f"acc_stride{st}", {"stride":st}))
    for d in (2,4,8,16,32): C.append((f"dim{d}", {"d":d}))
    for sp in (1.0,1.5,2.2,3.5): C.append((f"sep{sp}", {"spread":sp}))
    for sg in (0.15,0.35,0.6,1.0): C.append((f"sig{sg}", {"sigma":sg}))
    for K in (2,4,10): C.append((f"K{K}", {"K":K}))
    return C

F="sweep_raw.json"
raw=json.load(open(F)) if os.path.exists(F) else {}
allc=cells()
atoms=[(n,c,s) for n,c in allc for s in range(SEEDS) if str(s) not in raw.get(n,{}).get("seeds",{})]
if not atoms:
    print("ALL_DONE",len(allc),"cells"); sys.exit(0)
budget=time.time()+33
n_done=0
for name,cfg,s in atoms:
    if time.time()>budget: break
    r=eval_seed(cfg,s)
    raw.setdefault(name,{"_cfg":cfg,"seeds":{}})["seeds"][str(s)]=r
    json.dump(raw,open(F,"w"),indent=2); n_done+=1
print(f"did {n_done} atoms this call")
rem=[(n,s) for n,c in allc for s in range(SEEDS) if str(s) not in raw.get(n,{}).get("seeds",{})]
full=[n for n,c in allc if len(raw.get(n,{}).get("seeds",{}))>=SEEDS]
print(f"cells fully done {len(full)}/{len(allc)}; atoms remaining {len(rem)}")
for n in full[-3:]:
    sd=raw[n]["seeds"]; g=lambda k: np.mean([sd[str(s)][k] for s in range(SEEDS)])
    print(f"  {n:12s} q1cos {g('q1_cos'):.3f} q1l2 {g('q1_l2'):.3f} tay {g('e_taylor'):.4f} onl {g('e_online'):.4f} naive {g('e_naive'):.4f} probeL2 {g('e_probe_l2'):.4f} oracle {g('e_oracle'):.4f}")
