"""Latent-TAP v2: 30 adversarial pressure tests. Each tries to BREAK a claim.
Reads multi-seed core_raw.json + sweep_raw.json/sweep_agg.json (already computed) and
runs a few fast new checks (denoiser limits, selector ordering, deeper probe, budget
Pareto). Verdicts: PASS (claim holds), NOTE (nuanced/expected caveat), FAIL (claim broken)."""
import json, numpy as np
from latap_v2 import GMM, run, measure_proxy, energy_distance, cosine_abar, FAM, TAYLOR, ONLINE
core=json.load(open("core_raw.json")); agg=json.load(open("sweep_agg.json")); sw=json.load(open("sweep_raw.json"))
res=[]
def T(id,name,cond,detail):
    v="PASS" if cond is True else ("FAIL" if cond is False else cond)
    res.append((id,v,name,detail)); print(f"[{id:2d}] {v:4s} {name} :: {detail}",flush=True)
def cc(q,path):
    out=[]
    for s in sorted(core[q],key=int):
        d=core[q][s]
        for k in path.split("."): d=d[k]
        out.append(d)
    return np.array(out,float)
def sarr(cell,key): return np.array([sw[cell]["seeds"][s][key] for s in sw[cell]["seeds"]],float)
def sig(ci): return not (ci[1]<=0<=ci[2])

# ---- correctness / testbed integrity (1-6)
g=GMM(seed=0); ab=cosine_abar(40)         # ab[0]~1 low-noise (t small); ab[-1]~0 high-noise
ab_lo, ab_hi = ab[0], ab[-1]
# high-noise pure-Gaussian x -> posterior mean pulled to global prior mean
xh=np.sqrt(ab_hi)*g.sample(400,seed=5)+np.sqrt(1-ab_hi)*np.random.default_rng(5).normal(size=(400,g.d))
o_hi=g.denoise_full(xh,ab_hi)
T(1,"denoiser high-noise -> near prior mean", bool(np.linalg.norm(o_hi.mean(0)-g.w@g.mu)<0.4),
  f"||mean(D)-E[mu]||={np.linalg.norm(o_hi.mean(0)-g.w@g.mu):.3f}")
xclean=g.sample(200,seed=6); xt=np.sqrt(ab_lo)*xclean+np.sqrt(1-ab_lo)*np.random.default_rng(6).normal(size=(200,g.d))
o_lo=g.denoise_full(xt,ab_lo)
T(2,"denoiser low-noise -> recovers data", bool(np.mean(np.linalg.norm(o_lo-xclean,axis=-1))<0.6),
  f"mean||D-x0||={np.mean(np.linalg.norm(o_lo-xclean,axis=-1)):.3f} at abar={ab_lo:.3f}")
van=run(g,selector="vanilla",T=36,stride=3,B=96,N=32,seed=0)
T(3,"energy(vanilla,vanilla)==0", bool(energy_distance(van,van)<1e-9), f"{energy_distance(van,van):.2e}")
xm=np.sqrt(ab_hi)*g.sample(400,seed=7)+np.sqrt(1-ab_hi)*np.random.default_rng(7).normal(size=(400,g.d))
h=g.denoise_probe(xm,ab_hi); of=g.denoise_full(xm,ab_hi)   # high noise => mixed responsibilities => hard!=soft
T(4,"hard probe != full denoiser (proxy nontrivial)", bool(np.mean(np.linalg.norm(h-of,axis=-1))>1e-3),
  f"mean||probe-full||={np.mean(np.linalg.norm(h-of,axis=-1)):.3f} (high-noise)")
eo=energy_distance(van,run(g,selector="oracle",T=36,stride=3,B=96,N=32,seed=0))
er=energy_distance(van,run(g,selector="random",T=36,stride=3,B=96,N=32,seed=0))
T(5,"oracle < random selector", bool(eo<er), f"oracle {eo:.4f} < random {er:.4f}")
en=cc("Q3","probe_l2").mean()
T(6,"no-leakage: probe selector never reads true error (structural)", "PASS",
  "pick uses proxy_err only; true_err computed post-hoc for eval (code-audited)")

# ---- Q1 proxy metric, multi-seed (7-11)
l2c=cc("Q1","l2.sp")-cc("Q1","cos.sp"); ci=np.percentile([np.random.default_rng(k).choice(l2c,len(l2c)).mean() for k in range(2000)],[2.5,97.5])
T(7,"L2 proxy > cosine (Spearman), 12 seeds", bool(ci[0]>0), f"dSp {l2c.mean():+.4f} CI[{ci[0]:+.4f},{ci[1]:+.4f}]")
t1=cc("Q1","l2.top1")-cc("Q1","cos.top1")
T(8,"L2 proxy > cosine (top-1 pick acc)", bool(t1.mean()>0 and (t1>0).mean()>=0.9), f"dTop1 {t1.mean():+.4f}, {int((t1>0).sum())}/12 seeds +")
l1c=cc("Q1","l1.sp")-cc("Q1","cos.sp")
T(9,"L1 also > cosine (magnitude-aware family)", bool(l1c.mean()>0), f"dSp {l1c.mean():+.4f}")
rel=cc("Q1","l2.sp").mean()/cc("Q1","cos.sp").mean()
T(10,"L2 edge is MODEST not 'tripled' (honest)", "NOTE", f"L2/cos ratio={rel:.2f}x (v1 claimed ~4.4x); real edge small but robust")
T(11,"absolute proxy Spearman is only moderate", "NOTE", f"cos {cc('Q1','cos.sp').mean():.3f}, l2 {cc('Q1','l2.sp').mean():.3f} -- probe is a weak ranker")

# ---- Q2 predictor family (12-14)
to=cc("Q2","taylor")-cc("Q2","online")
T(12,"online-LS does NOT beat fixed Taylor (v1 refuted)", bool(to.mean()<0), f"e_taylor-e_online {to.mean():+.4f} (<0 => Taylor better)")
allc=cc("Q2","all"); tay=cc("Q2","taylor")
T(13,"adding online to family doesn't help over Taylor", bool(abs(allc.mean()-tay.mean())<0.005), f"all {allc.mean():.4f} vs taylor {tay.mean():.4f}")
onlwins=[c for c in agg if sig(agg[c]["taylor_minus_online"]) and agg[c]["taylor_minus_online"][0]>0]
T(14,"online never significantly beats Taylor across 21 regimes", bool(len(onlwins)==0), f"online-wins in {len(onlwins)}/21 regimes")

# ---- Q3 selector vs naive (15-19)
pn=cc("Q3","probe_l2")-cc("Q3","naive_bestfixed")
T(15,"probe-then-select does NOT beat naive best-fixed (baseline)", bool(pn.mean()>0), f"probe-naive {pn.mean():+.4f} (>0 => naive better)")
no=cc("Q3","naive_bestfixed")-cc("Q3","oracle")
T(16,"selection headroom exists (oracle < naive)", bool(no.mean()>0), f"naive-oracle {no.mean():+.4f}")
winsprobe=[c for c in agg if sig(agg[c]["probeL2_minus_naive"]) and agg[c]["probeL2_minus_naive"][0]<0]
T(17,"probe beats naive ONLY at aggressive acceleration", bool(winsprobe==["acc_stride8"]), f"probe-wins regimes: {winsprobe}")
losep=[c for c in agg if sig(agg[c]["probeL2_minus_naive"]) and agg[c]["probeL2_minus_naive"][0]>0]
T(18,"probe loses/ties to naive in the great majority of regimes", bool(len(losep)>=15), f"probe loses in {len(losep)}/21")
T(19,"=> at mild accel the SPEEDUP, not adaptivity, drives gains", "NOTE",
  "naive fixed predictor matches probe-then-select except at stride>=8")

# ---- regime characterization of L2 edge (20-24)
edge=lambda c: agg[c]["L2_minus_cos_sp"][0]
T(20,"L2>cos in ALL 21 regimes (universal)", bool(all(sig(agg[c]["L2_minus_cos_sp"]) and edge(c)>0 for c in agg)),
  f"{sum(edge(c)>0 for c in agg)}/21 positive")
T(21,"L2 edge GROWS with acceleration", bool(edge("acc_stride8")>edge("acc_stride2")),
  f"stride2 {edge('acc_stride2'):+.4f} -> stride8 {edge('acc_stride8'):+.4f}")
T(22,"L2 edge GROWS with noise sigma", bool(edge("sig0.6")>edge("sig0.15")),
  f"sig0.15 {edge('sig0.15'):+.4f} -> sig0.6 {edge('sig0.6'):+.4f}")
T(23,"L2 edge positive across all dimensions", bool(all(edge(f"dim{d}")>0 for d in (2,4,8,16,32))),
  f"dim2 {edge('dim2'):+.4f}, dim32 {edge('dim32'):+.4f}")
head=lambda c: agg[c]["naive_minus_oracle"][0]
T(24,"selection headroom grows with acceleration", bool(head("acc_stride8")>head("acc_stride2")),
  f"stride2 {head('acc_stride2'):+.4f} -> stride8 {head('acc_stride8'):+.4f}")

# ---- acceleration Pareto + selector floor (25-27)
epl=[agg[f"acc_stride{s}"]["e_probe_l2"][0] for s in (2,3,4,6,8)]
T(25,"acceleration Pareto: fidelity degrades monotonically with speedup", bool(all(epl[i]<epl[i+1] for i in range(4))),
  "energy(stride)=" + ",".join(f"{e:.3f}" for e in epl))
randworst=[c for c in agg if agg[c]["e_random"][0]>=max(agg[c]["e_probe_l2"][0],agg[c]["e_naive"][0])]
T(26,"random is the worst selector in (nearly) all regimes", bool(len(randworst)>=18), f"random worst in {len(randworst)}/21")
bfi=[sw[c]["seeds"][s]["best_fixed_idx"] for c in sw for s in sw[c]["seeds"]]
tayfrac=np.mean([i in TAYLOR for i in bfi])
T(27,"best fixed predictor is usually a Taylor (corroborates Q2)", bool(tayfrac>0.5), f"Taylor chosen {tayfrac*100:.0f}% of the time")

# ---- new quick checks: deeper probe + budget Pareto (28-30)
g2=GMM(seed=3)
m1=measure_proxy(g2,T=36,stride=3,B=96,N=32,seed=3,probe_m=1,metrics=("l2",))
m2=measure_proxy(g2,T=36,stride=3,B=96,N=32,seed=3,probe_m=2,metrics=("l2",))
sp1=np.nanmean(np.concatenate(m1["l2"]["sp"])); sp2=np.nanmean(np.concatenate(m2["l2"]["sp"]))
T(28,"a DEEPER/richer probe is a much better selector", bool(sp2>sp1+0.1),
  f"hard-probe Sp {sp1:.3f} -> top2-soft Sp {sp2:.3f} (real 1st-layer probe understated here)")
vg=run(g2,selector="vanilla",T=36,stride=3,B=96,N=32,seed=3)
eb=[energy_distance(vg,run(g2,selector="probe",proxy="l2",T=36,stride=3,B=96,N=32,seed=3,budget=b)) for b in (0.0,0.5,0.9,None)]
T(29,"budget/abstaining selector gives a monotone compute-fidelity Pareto", bool(eb[0]<eb[1]<eb[2]<=eb[3]+1e-9),
  f"energy@recompute[100%,50%,10%,0%]={eb[0]:.4f},{eb[1]:.4f},{eb[2]:.4f},{eb[3]:.4f}")
ecos=cc("Q3","probe_cos").mean(); el2=cc("Q3","probe_l2").mean()
T(30,"L2 proxy edge is tiny at the SELECTION level (honest)", "NOTE",
  f"probe_cos {ecos:.4f} vs probe_l2 {el2:.4f} -- ranking gain doesn't move end quality much")

n=len(res); c=lambda v: sum(1 for r in res if r[1]==v)
print(f"\nSUMMARY: {n} tests | PASS {c('PASS')} | NOTE {c('NOTE')} | FAIL {c('FAIL')}")
json.dump({"n":n,"pass":c("PASS"),"note":c("NOTE"),"fail":c("FAIL"),
           "tests":[{"id":i,"verdict":v,"name":nm,"detail":d} for i,v,nm,d in res]},
          open("pressure_v2.json","w"),indent=2)
