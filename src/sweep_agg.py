"""Aggregate sweep_raw.json into the 'when does each lever help' map, with paired
bootstrap CIs. Levers: L2>cos proxy (Q1 spearman & selection energy), Taylor vs online
family, probe-then-select vs naive best-fixed. Also selection headroom (naive vs oracle)."""
import json, numpy as np
raw=json.load(open("sweep_raw.json"))
def arr(cell,key): return np.array([raw[cell]["seeds"][s][key] for s in raw[cell]["seeds"]],float)
def bci(v,nb=4000,seed=0):
    v=np.asarray(v,float); v=v[~np.isnan(v)]
    if len(v)==0: return (np.nan,np.nan,np.nan)
    r=np.random.default_rng(seed); bs=np.array([v[r.integers(0,len(v),len(v))].mean() for _ in range(nb)])
    return float(v.mean()),float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))
def sig(ci):  # significance of a paired difference CI (excludes 0?)
    return "" if (ci[1]<=0<=ci[2]) else ("+" if ci[1]>0 else "-")
order=["acc_stride2","acc_stride3","acc_stride4","acc_stride6","acc_stride8",
       "dim2","dim4","dim8","dim16","dim32","sep1.0","sep1.5","sep2.2","sep3.5",
       "sig0.15","sig0.35","sig0.6","sig1.0","K2","K4","K10"]
rows={}
print(f"{'cell':12s} {'q1cos':>6} {'q1l2':>6} {'L2-cos':>16} | {'tay-onl(>0 tayWins)':>20} | {'probe-naive(<0 win)':>20} | {'naive-oracle(head)':>18}")
for c in order:
    q1c=arr(c,"q1_cos"); q1l=arr(c,"q1_l2")
    dL2=bci(q1l-q1c)                          # >0 => L2 proxy better (rank corr)
    dTO=bci(arr(c,"e_taylor")-arr(c,"e_online"))   # <0 => Taylor better(lower); >0 => online better
    dPN=bci(arr(c,"e_probe_l2")-arr(c,"e_naive"))  # <0 => probe beats naive
    dNO=bci(arr(c,"e_naive")-arr(c,"e_oracle"))    # >0 => selection headroom
    cfg=raw[c]["_cfg"]; st=cfg.get("stride",3); T=cfg.get("T",36)
    accel=T/(T/st+4.0)
    rows[c]={"q1_cos":bci(q1c),"q1_l2":bci(q1l),"L2_minus_cos_sp":dL2,
             "taylor_minus_online":dTO,"probeL2_minus_naive":dPN,"naive_minus_oracle":dNO,
             "e_probe_l2":bci(arr(c,"e_probe_l2")),"e_naive":bci(arr(c,"e_naive")),
             "e_oracle":bci(arr(c,"e_oracle")),"e_random":bci(arr(c,"e_random")),
             "e_probe_cos":bci(arr(c,"e_probe_cos")),"accel":float(accel),"cfg":cfg}
    print(f"{c:12s} {q1c.mean():6.3f} {q1l.mean():6.3f} "
          f"{dL2[0]:+.4f}[{dL2[1]:+.3f},{dL2[2]:+.3f}]{sig(dL2):1s} | "
          f"{dTO[0]:+.4f}[{dTO[1]:+.3f},{dTO[2]:+.3f}]{sig(dTO):1s} | "
          f"{dPN[0]:+.4f}[{dPN[1]:+.3f},{dPN[2]:+.3f}]{sig(dPN):1s} | "
          f"{dNO[0]:+.4f}{sig(dNO):1s}")
json.dump(rows,open("sweep_agg.json","w"),indent=2)

# ---- crossover summary
print("\n== LEVER VERDICTS (sig = paired 95% CI excludes 0) ==")
def verdict(key, better):
    wins=[c for c in order if sig(rows[c][key])==better]
    ties=[c for c in order if sig(rows[c][key])==""]
    return wins,ties
# sign conventions: energy lower=better. taylor_minus_online = e_taylor-e_online (<0 => Taylor better).
# probeL2_minus_naive = e_probe-e_naive (<0 => probe better). naive_minus_oracle (>0 => headroom).
w,_=verdict("L2_minus_cos_sp","+");  print(f"L2 proxy > cosine (rank corr) sig: {len(w)}/21")
w,_=verdict("taylor_minus_online","-"); print(f"TAYLOR beats online (e_tay<e_onl) sig: {len(w)}/21: {w}")
w,_=verdict("taylor_minus_online","+"); print(f"online beats Taylor sig: {len(w)}/21: {w}")
w,_=verdict("probeL2_minus_naive","-"); print(f"probe-then-select BEATS naive sig: {len(w)}/21: {w}")
w,_=verdict("probeL2_minus_naive","+"); print(f"probe LOSES to naive best-fixed sig: {len(w)}/21: {w}")
w,_=verdict("naive_minus_oracle","+");  print(f"selection headroom (oracle<naive) sig: {len(w)}/21: {w}")
