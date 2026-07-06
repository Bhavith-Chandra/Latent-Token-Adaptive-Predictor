"""Positioning of Latent-TAP among training-free diffusion accelerators.
(a) design-space map: decision granularity x reconstruction sophistication;
(b) reported latency speedups (real, from each paper; backbone annotated -- NOT iso-quality,
    shown only to situate the landscape). Latent-TAP is evaluated on an analytic testbed and
    is therefore excluded from (b)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

cExp="#C0452F"; cChp="#2E6DA4"; cDec="#3C8B57"; cAmb="#B07A10"; cSt="#45566A"; ink="#222222"

# design-space coordinates: x = granularity {0 schedule,1 timestep,2 sample,3 token},
#                           y = sophistication {0 reuse,1 reuse+rescale,2 Taylor forecast,3 selected family}
pts = [  # name, x, y, color, (label dx, dy, ha)
 ("DeepCache", 0.00, 0.00, cSt, (0.10,-0.28,"left")),
 ("FORA",      0.00, 0.22, cSt, (0.10, 0.12,"left")),
 ("TeaCache",  1.00, 1.00, cChp,(0.10, 0.12,"left")),
 ("DiCache",   2.00, 1.00, cChp,(0.10, 0.12,"left")),
 ("ToCa",      3.00, 0.00, cAmb,(-0.10,0.14,"right")),
 ("TaylorSeer",1.00, 2.00, cDec,(0.10, 0.12,"left")),
 ("TAP (no code)", 2.66, 3.10, cExp,(-0.12,0.10,"right")),
]
fig, ax = plt.subplots(1, 2, figsize=(13.6, 4.6), gridspec_kw={"width_ratios":[1.28,1.0]})

# ---- (a) design-space map ----
a = ax[0]
a.set_xlim(-0.55, 3.7); a.set_ylim(-0.6, 3.7)
for x in range(4):
    a.axvline(x, color="0.90", lw=0.8, zorder=0)
    a.axhline(x, color="0.90", lw=0.8, zorder=0)
for name,x,y,c,(dx,dy,ha) in pts:
    a.scatter([x],[y], s=46, color=c, edgecolor="white", lw=0.8, zorder=3)
    a.annotate(name, (x,y), (x+dx,y+dy), fontsize=8.5, color=ink, ha=ha, va="center", zorder=4)
# Latent-TAP star (this work), top-right corner: token + selected family
a.scatter([3.0],[3.0], marker="*", s=340, color=cExp, edgecolor="black", lw=0.7, zorder=5)
a.annotate("Latent-TAP\n(this work)", (3.0,3.0), (2.86,2.62), fontsize=9, fontweight="bold",
           color=cExp, ha="right", va="top", zorder=6)
a.set_xticks(range(4)); a.set_xticklabels(["schedule","timestep","sample","token"])
a.set_yticks(range(4)); a.set_yticklabels(["reuse","reuse\n+rescale","Taylor\nforecast","selected\nfamily"])
a.set_xlabel("decision granularity  (coarse $\\rightarrow$ fine)")
a.set_ylabel("reconstruction  (simple $\\rightarrow$ expressive)")
a.set_title("(a) Design space of training-free accelerators", fontsize=11)
for s in ("top","right"): a.spines[s].set_visible(False)

# ---- (b) reported speedups (real; backbone-labelled) ----
b = ax[1]
bars = [  # method, speedup, backbone, color
 ("ToCa",       1.9, "PixArt-$\\alpha$", cAmb),
 ("TeaCache",   2.0, "FLUX",             cChp),
 ("DeepCache",  2.3, "SD-1.5",           cSt),
 ("TaylorSeer", 3.5, "FLUX (latency)",   cDec),
]
names=[x[0] for x in bars]; vals=[x[1] for x in bars]; cols=[x[3] for x in bars]
ypos=range(len(bars))
b.barh(list(ypos), vals, color=cols, alpha=0.85, height=0.6, zorder=3)
for i,(nm,v,bk,c) in enumerate(bars):
    b.text(v+0.06, i, f"{v:.1f}$\\times$  ({bk})", va="center", ha="left", fontsize=8.4, color=ink)
b.axvline(1.0, color="0.6", lw=0.8, ls="--")
b.set_yticks(list(ypos)); b.set_yticklabels(names)
b.set_xlim(0, 4.9); b.set_xlabel("reported latency speedup ($\\times$)")
b.set_title("(b) Reported speedups", fontsize=11)
b.invert_yaxis()
for s in ("top","right"): b.spines[s].set_visible(False)

fig.suptitle("Positioning of Latent-TAP among training-free diffusion accelerators", fontsize=12.6, y=1.00)
fig.tight_layout(rect=[0,0,1,0.95])
fig.savefig("fig_landscape.png", dpi=150)
print("wrote fig_landscape.png")
