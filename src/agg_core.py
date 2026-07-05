import json, numpy as np
from latap_v2 import boot_ci
raw=json.load(open("core_raw.json")); S=sorted(raw["Q1"],key=int)
def col(q,path):
    out=[]
    for s in S:
        d=raw[q][s]
        for k in path.split("."): d=d[k]
        out.append(d)
    return np.array(out,float)
def fmt(c): return f"{c[0]:.4f} [{c[1]:.4f},{c[2]:.4f}]"
R={}
# Q1
R["Q1"]={m:{"spearman":boot_ci(col("Q1",f"{m}.sp")),"top1":boot_ci(col("Q1",f"{m}.top1"))} for m in ("cos","l2","l1")}
R["Q1_l2_minus_cos_sp"]=boot_ci(col("Q1","l2.sp")-col("Q1","cos.sp"))
R["Q1_l2_minus_cos_top1"]=boot_ci(col("Q1","l2.top1")-col("Q1","cos.top1"))
# Q2
R["Q2"]={k:boot_ci(col("Q2",k)) for k in ("taylor","online","all")}
R["Q2_taylor_minus_online"]=boot_ci(col("Q2","taylor")-col("Q2","online"))  # >0 => taylor better(lower)
# Q3
R["Q3"]={k:boot_ci(col("Q3",k)) for k in ("random","naive_bestfixed","probe_cos","probe_l2","oracle")}
R["Q3_probeL2_minus_naive"]=boot_ci(col("Q3","probe_l2")-col("Q3","naive_bestfixed"))  # <0 => probe better
R["Q3_naive_minus_oracle"]=boot_ci(col("Q3","naive_bestfixed")-col("Q3","oracle"))      # headroom of any selector
R["Q3_probeL2_minus_oracle"]=boot_ci(col("Q3","probe_l2")-col("Q3","oracle"))
json.dump(R,open("core_agg.json","w"),indent=2)
print("== Q1 proxy quality (Spearman proxy->true err; top-1 pick acc), 12 seeds ==")
for m in ("cos","l2","l1"): print(f"  {m:4s} sp {fmt(R['Q1'][m]['spearman'])}  top1 {fmt(R['Q1'][m]['top1'])}")
print(f"  paired L2-cos: Spearman {fmt(R['Q1_l2_minus_cos_sp'])}  top1 {fmt(R['Q1_l2_minus_cos_top1'])}   (>0 => L2 better)")
print("== Q2 predictor family: energy dist to vanilla (lower better) ==")
for k in ("taylor","online","all"): print(f"  {k:8s} {fmt(R['Q2'][k])}")
print(f"  paired taylor-online: {fmt(R['Q2_taylor_minus_online'])}   (<0 => Taylor better i.e. online loses)")
print("== Q3 selectors: energy dist to vanilla (lower better) ==")
for k in ("random","naive_bestfixed","probe_cos","probe_l2","oracle"): print(f"  {k:16s} {fmt(R['Q3'][k])}")
print(f"  paired probeL2-naive:  {fmt(R['Q3_probeL2_minus_naive'])}  (<0 => probe beats naive)")
print(f"  paired naive-oracle:   {fmt(R['Q3_naive_minus_oracle'])}  (>0 => selection headroom exists)")
