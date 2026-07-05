import json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import os
O=os.environ.get("LATAP_DATA", os.path.dirname(os.path.abspath(__file__)))
agg=json.load(open(f"{O}/sweep_agg.json")); pd=json.load(open(f"{O}/probe_depth.json"))
strides=[2,3,4,6,8]; cells=[f"acc_stride{s}" for s in strides]
def g(c,k): return agg[c][k][0]
def lohi(c,k): return agg[c][k][1],agg[c][k][2]
C={"oracle":"#1a7f3c","naive":"#c8702a","probe":"#123a6b","random":"#999999"}
fig,ax=plt.subplots(1,4,figsize=(18,4.2))
# (a) acceleration Pareto
for key,lab,col in [("e_oracle","oracle (upper bound)",C["oracle"]),("e_naive","naive best-fixed",C["naive"]),
                    ("e_probe_l2","probe-then-select (L2)",C["probe"]),("e_random","random",C["random"])]:
    y=[g(c,key) for c in cells]; lo=[agg[c][key][1] for c in cells]; hi=[agg[c][key][2] for c in cells]
    ax[0].plot(strides,y,'o-',color=col,lw=2.3,ms=6,label=lab); ax[0].fill_between(strides,lo,hi,color=col,alpha=.12)
ax[0].set_xlabel("acceleration (cache stride)"); ax[0].set_ylabel("energy distance to full sampler")
ax[0].set_title("(a) Acceleration–fidelity Pareto"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.2)
# (b) L2-cos edge across regimes
groups=[("acc",["acc_stride2","acc_stride3","acc_stride4","acc_stride6","acc_stride8"]),
        ("dim",["dim2","dim4","dim8","dim16","dim32"]),("sep",["sep1.0","sep1.5","sep2.2","sep3.5"]),
        ("noise",["sig0.15","sig0.35","sig0.6","sig1.0"]),("K",["K2","K4","K10"])]
xs=[]; ys=[]; los=[]; his=[]; labs=[]; xi=0; ticks=[]; tlab=[]
for gm,cl in groups:
    for c in cl:
        e=agg[c]["L2_minus_cos_sp"]; xs.append(xi); ys.append(e[0]); los.append(e[0]-e[1]); his.append(e[2]-e[0]); xi+=1
    ticks.append(xi-len(cl)/2-0.5); tlab.append(gm); xi+=0.6
ax[1].bar(xs,ys,yerr=[los,his],color="#5a3fa0",alpha=.8,capsize=2,width=0.8)
ax[1].axhline(0,color='k',lw=.6); ax[1].set_xticks(ticks); ax[1].set_xticklabels(tlab)
ax[1].set_ylabel(r"$\Delta$ Spearman (L2 $-$ cosine)"); ax[1].set_title("(b) L2 proxy edge: small, but +ve in 21/21 regimes"); ax[1].grid(axis='y',alpha=.2)
# (c) probe-naive and naive-oracle vs stride
pn=[g(c,"probeL2_minus_naive") for c in cells]; no=[g(c,"naive_minus_oracle") for c in cells]
ax[2].axhline(0,color='k',lw=.6)
ax[2].plot(strides,pn,'s-',color=C["probe"],lw=2.3,ms=7,label="probe $-$ naive  (<0 ⇒ probe wins)")
ax[2].plot(strides,no,'^-',color=C["oracle"],lw=2.3,ms=7,label="naive $-$ oracle  (headroom)")
ax[2].set_xlabel("acceleration (cache stride)"); ax[2].set_ylabel("energy-distance difference")
ax[2].set_title("(c) Probe adaptivity only pays at high accel"); ax[2].legend(fontsize=8); ax[2].grid(alpha=.2)
# (d) probe richness curve
ms=[1,2,3,6]; y=[pd[str(m)][0] for m in ms]; lo=[pd[str(m)][0]-pd[str(m)][1] for m in ms]; hi=[pd[str(m)][2]-pd[str(m)][0] for m in ms]
ax[3].errorbar(ms,y,yerr=[lo,hi],fmt='o-',color="#b0202a",lw=2.4,ms=7,capsize=3)
ax[3].set_xticks(ms); ax[3].set_xticklabels(["1\n(hard)","2","3","6\n(full)"])
ax[3].set_ylim(0,1.05); ax[3].set_xlabel("probe richness (top-$m$ components)")
ax[3].set_ylabel("proxy ranking fidelity (Spearman)"); ax[3].set_title("(d) Richer probe ⇒ better selector"); ax[3].grid(alpha=.2)
fig.suptitle("Latent-TAP v2 on the exact-residual Gaussian-mixture testbed: what actually holds up",fontsize=13)
fig.tight_layout(rect=[0,0,1,0.95]); fig.savefig(f"{O}/fig_v2.png",dpi=140)
print("wrote fig_v2.png")
