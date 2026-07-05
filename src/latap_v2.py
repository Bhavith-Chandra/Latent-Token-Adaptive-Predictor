"""Latent-TAP v2 : exact-residual Gaussian-mixture diffusion testbed.

Faithful reconstruction of the v1 Lane-A testbed. Everything is exact and pure-numpy
(no network), so the *true* denoiser output at any step is available in closed form
(Tweedie for a Gaussian mixture). This lets us measure, without approximation, how well
a cheap PROBE-based proxy ranks a family of cheap PREDICTORS against their true error.

Faithful analogs of the real training-free accelerator (TAP):
  - true output O_t   = exact posterior mean E[x0|x_t]  (soft mixture over K comps)  -- "expensive"
  - probe h_t         = hard-assignment posterior mean  (argmax component only)      -- "cheap first-layer readout"
  - predictor family  = fixed finite-difference Taylor {order,horizon} + online least-squares
  - probe-then-select = pick, per token, the predictor whose extrapolated PROBE best
                        matches the freshly-computed cheap probe (proxy metric: cosine=TAP, or L2=ours)
  - oracle            = pick per token the predictor with min TRUE error (upper bound)

Metrics: (a) rank-correlation between proxy-error and true-error across the family
(cosine vs L2 proxy);  (b) energy distance of accelerated final samples vs the full
(vanilla) sampler -- distributional fidelity;  (c) selector comparison at matched compute.
"""
import numpy as np

# ---------------------------------------------------------------- schedule
def cosine_abar(T):
    s = 0.008
    t = np.linspace(0, 1, T + 1)
    f = np.cos((t + s) / (1 + s) * np.pi / 2) ** 2
    return (f / f[0])[1:]              # abar_1..abar_T, decreasing in index->0

# ---------------------------------------------------------------- GMM prior
class GMM:
    """K-component isotropic Gaussian mixture in R^d; exact + hard denoisers."""
    def __init__(self, K=6, d=8, sigma=0.35, spread=2.2, seed=0):
        r = np.random.default_rng(seed)
        self.K, self.d, self.sigma = K, d, sigma
        self.mu = r.normal(0, spread, size=(K, d))          # component means
        w = r.uniform(0.5, 1.5, size=K); self.w = w / w.sum()
    def sample(self, n, seed=0):
        r = np.random.default_rng(seed)
        comp = r.choice(self.K, size=n, p=self.w)
        return self.mu[comp] + self.sigma * r.normal(size=(n, self.d))
    def _resp_and_post(self, x, abar):
        # x: (...,d). returns responsibilities r_k (...,K) and per-comp x0 posterior means (...,K,d)
        s2 = self.sigma ** 2
        var = abar * s2 + (1 - abar)                        # marginal var of x_t coord
        mt = np.sqrt(abar) * self.mu                        # (K,d) means of x_t comps
        diff = x[..., None, :] - mt                         # (...,K,d)
        logr = np.log(self.w) - 0.5 * (diff ** 2).sum(-1) / var - 0.5 * self.d * np.log(var)
        logr -= logr.max(-1, keepdims=True)
        r = np.exp(logr); r /= r.sum(-1, keepdims=True)     # (...,K)
        c = np.sqrt(abar) * s2 / (abar * s2 + (1 - abar))
        post = self.mu + c * (x[..., None, :] - np.sqrt(abar) * self.mu)   # (...,K,d)
        return r, post
    def denoise_full(self, x, abar):
        r, post = self._resp_and_post(x, abar)
        return (r[..., None] * post).sum(-2)                # soft mixture posterior mean  (expensive)
    def denoise_probe(self, x, abar):
        r, post = self._resp_and_post(x, abar)
        k = r.argmax(-1)                                    # hard assignment (cheap readout)
        return np.take_along_axis(post, k[..., None, None], axis=-2)[..., 0, :]
    def denoise_probe_topm(self, x, abar, m=2):
        """Soft mix over the top-m components only -- an intermediate 'deeper' probe
        (m=1 == hard/first-layer; m=K == full/exact). Tests the depth analogy."""
        r, post = self._resp_and_post(x, abar)
        if m >= self.K: return (r[..., None] * post).sum(-2)
        idx = np.argsort(-r, axis=-1)[..., :m]              # (...,m)
        rt = np.take_along_axis(r, idx, axis=-1)            # (...,m)
        rt = rt / (rt.sum(-1, keepdims=True) + 1e-12)
        pt = np.take_along_axis(post, idx[..., None], axis=-2)  # (...,m,d)
        return (rt[..., None] * pt).sum(-2)

# ---------------------------------------------------------------- predictor family
# spec = (name, deg, window, stride): fit degree `deg` poly to the last `window` anchors
# taken every `stride`-th (horizon), predict at target time. Taylor = interpolation
# (window=deg+1); OLS = over-determined least squares (window>deg+1).
FAM = [("T0h0",0,1,1),("T1h1",1,2,1),("T1h2",1,2,2),("T2h1",2,3,1),("T2h2",2,3,2),
       ("OLS2w5",2,5,1),("OLS2w5h2",2,5,2),("OLS3w6",3,6,1)]
TAYLOR = [i for i,s in enumerate(FAM) if s[0].startswith("T")]
ONLINE = [i for i,s in enumerate(FAM) if s[0].startswith("OLS")]

def _fit_predict(ts, V, target, deg, window, stride):
    """ts: list of anchor times (increasing sampler-order). V: (A,B,N,d) values at anchors.
    Use last `window` anchors stepping by `stride`; fit degree deg poly; eval at target."""
    idx = list(range(len(ts)-1, -1, -stride))[:window][::-1]
    if len(idx) < deg + 1:
        idx = list(range(len(ts)-1, -1, -1))[:max(deg+1,1)][::-1]
    tt = np.array([ts[i] for i in idx], float)
    Vv = V[idx]                                             # (m,B,N,d)
    tt0 = tt - tt[-1]                                        # center for conditioning
    Vsh = Vv.reshape(len(idx), -1)                          # (m, B*N*d)
    A = np.vander(tt0, deg + 1, increasing=True)            # (m,deg+1)
    coef, *_ = np.linalg.lstsq(A, Vsh, rcond=None)          # (deg+1, B*N*d)
    av = np.vander(np.array([target - tt[-1]]), deg + 1, increasing=True)  # (1,deg+1)
    return (av @ coef).reshape(V.shape[1:])                 # (B,N,d)

def predictor(i, ts, V, target):
    _, deg, window, stride = FAM[i]
    return _fit_predict(ts, V, target, deg, window, stride)

# ---------------------------------------------------------------- distances / stats
def _paired_dist(a, b, metric):
    # a,b: (...,d) -> (...) distance
    if metric == "l2":
        return np.linalg.norm(a - b, axis=-1)
    if metric == "cos":
        num = (a * b).sum(-1)
        den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-12
        return 1 - num / den
    if metric == "l1":
        return np.abs(a - b).sum(-1)
    raise ValueError(metric)

def spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 2 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return np.nan
    rx = np.argsort(np.argsort(x)).astype(float); ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    d = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return (rx * ry).sum() / d if d > 1e-12 else np.nan

def spearman_axis0(P, Q):
    """Vectorized Spearman across axis 0 (predictors) for every column.
    P,Q: (K, M). Returns (M,) rank-correlations (nan where degenerate)."""
    def rank(A):
        r = np.empty_like(A, dtype=float)
        idx = np.argsort(A, axis=0)
        ar = np.arange(A.shape[0])[:, None]
        np.put_along_axis(r, idx, np.broadcast_to(ar, A.shape).astype(float), axis=0)
        return r
    rp = rank(P); rq = rank(Q)
    rp -= rp.mean(0); rq -= rq.mean(0)
    num = (rp * rq).sum(0)
    den = np.sqrt((rp ** 2).sum(0) * (rq ** 2).sum(0))
    out = np.full(P.shape[1], np.nan)
    ok = den > 1e-12
    out[ok] = num[ok] / den[ok]
    return out

def energy_distance(A, B, cap=128, seed=0):
    A = A.reshape(len(A), -1); B = B.reshape(len(B), -1)
    r = np.random.default_rng(seed)
    if len(A) > cap: A = A[r.choice(len(A), cap, replace=False)]
    if len(B) > cap: B = B[r.choice(len(B), cap, replace=False)]
    def pd(X, Y):
        return np.sqrt(((X[:, None, :] - Y[None, :, :]) ** 2).sum(-1) + 1e-12).mean()
    return 2 * pd(A, B) - pd(A, A) - pd(B, B)

def boot_ci(vals, nb=2000, seed=0):
    v = np.asarray(vals, float); v = v[~np.isnan(v)]
    if len(v) == 0: return (np.nan, np.nan, np.nan)
    r = np.random.default_rng(seed); n = len(v)
    bs = np.array([v[r.integers(0, n, n)].mean() for _ in range(nb)])
    return float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

# ---------------------------------------------------------------- sampler / experiment
def run(gmm, T=40, stride=3, B=256, N=48, seed=0, selector="probe", proxy="cos",
        fixed_pred=0, budget=None, record=False, allowed=None):
    """One accelerated sampling run.
    stride: every `stride`-th step is a full anchor; others are predicted (accel ~= stride).
    selector in {vanilla, naive(fixed), probe, oracle, random}; proxy in {cos,l2,l1}.
    allowed: list of predictor indices the selector may choose among (None = all FAM).
    budget: if set, only the `budget` fraction of *least-reliable* tokens (highest min proxy
            err) are recomputed full; rest predicted -- an abstaining/budget selector.
    Returns final samples (B,N,d) and, if record, per-(step,token,predictor) proxy/true errs."""
    abar = cosine_abar(T)
    r = np.random.default_rng(1000 + seed)
    x = r.normal(size=(B, N, gmm.d))                       # start from noise (abar~0 at t=T)
    order = list(range(T - 1, -1, -1))                     # sampler order: high t -> low t
    anch_ts, anch_O, anch_h = [], [], []
    recs = []
    al = list(range(len(FAM))) if allowed is None else list(allowed)
    full_calls = 0
    for step, ti in enumerate(order):
        ab = abar[ti]
        h = gmm.denoise_probe(x, ab)                        # cheap probe (always available)
        is_anchor = (step % stride == 0) or (len(anch_ts) < 4) or selector == "vanilla"
        if is_anchor:
            O = gmm.denoise_full(x, ab); full_calls += 1
            anch_ts.append(ti); anch_O.append(O); anch_h.append(h)
            if len(anch_ts) > 8:
                anch_ts.pop(0); anch_O.pop(0); anch_h.pop(0)
            O_used = O
        else:
            Vh = np.stack(anch_h); VO = np.stack(anch_O)
            Ohat = np.stack([predictor(i, anch_ts, VO, ti) for i in al])          # (|al|,B,N,d)
            hhat = np.stack([predictor(i, anch_ts, Vh, ti) for i in al])          # (|al|,B,N,d)
            proxy_err = _paired_dist(hhat, h[None], proxy)                        # (|al|,B,N)
            O_true = gmm.denoise_full(x, ab)                                      # eval only
            true_err = _paired_dist(Ohat, O_true[None], "l2")                    # (|al|,B,N)
            if selector == "probe":
                pick = proxy_err.argmin(0)                                        # (B,N) index into al
            elif selector == "naive":
                pick = np.full((B, N), al.index(fixed_pred) if fixed_pred in al else 0, int)
            elif selector == "oracle":
                pick = true_err.argmin(0)
            elif selector == "random":
                pick = np.random.default_rng(step + 7 * seed).integers(0, len(al), (B, N))
            else:
                raise ValueError(selector)
            O_used = np.take_along_axis(Ohat, pick[None, ..., None], 0)[0]        # (B,N,d)
            if budget is not None:
                rel = proxy_err.min(0)                                            # (B,N) reliability
                thr = np.quantile(rel, budget)                                    # recompute worst (1-budget)
                mask = rel > thr
                if mask.any():
                    O_used = O_used.copy(); O_used[mask] = O_true[mask]; full_calls += mask.mean()
            if record:
                for b in range(0, B, 8):
                    for n in range(0, N, 6):
                        recs.append((proxy_err[:, b, n].copy(), true_err[:, b, n].copy()))
        # DDIM deterministic reverse step
        eps = (x - np.sqrt(ab) * O_used) / np.sqrt(1 - ab)
        ab_next = abar[order[step + 1]] if step + 1 < len(order) else 1.0
        x = np.sqrt(ab_next) * O_used + np.sqrt(1 - ab_next) * eps
    return (x, recs, full_calls) if record else x

def measure_proxy(gmm, T=40, stride=3, B=256, N=48, seed=0, metrics=("cos","l2","l1"), probe_m=1):
    """Walk the *vanilla* (full) trajectory; at every would-be-skipped step, build the
    predictor bank, and for each metric record Spearman(proxy_err, true_err) across the
    family per token, plus whether argmin(proxy_err)==argmin(true_err) (top-1 pick).
    Isolates proxy quality from the selection feedback loop. Returns dict metric-> (spearmans, top1)."""
    abar = cosine_abar(T)
    r = np.random.default_rng(1000 + seed)
    x = r.normal(size=(B, N, gmm.d))
    order = list(range(T - 1, -1, -1))
    anch_ts, anch_O, anch_h = [], [], []
    out = {m: {"sp": [], "top1": []} for m in metrics}
    Pn = len(FAM)
    for step, ti in enumerate(order):
        ab = abar[ti]
        h = gmm.denoise_probe(x, ab) if probe_m == 1 else gmm.denoise_probe_topm(x, ab, probe_m)
        O = gmm.denoise_full(x, ab)
        is_anchor = (step % stride == 0) or (len(anch_ts) < 4)
        if is_anchor:
            anch_ts.append(ti); anch_O.append(O); anch_h.append(h)
            if len(anch_ts) > 8:
                anch_ts.pop(0); anch_O.pop(0); anch_h.pop(0)
        else:
            Vh = np.stack(anch_h); VO = np.stack(anch_O)
            Ohat = np.stack([predictor(i, anch_ts, VO, ti) for i in range(Pn)])
            hhat = np.stack([predictor(i, anch_ts, Vh, ti) for i in range(Pn)])
            true_err = _paired_dist(Ohat, O[None], "l2")                          # (P,B,N)
            P = true_err.shape[0]
            te = true_err.reshape(P, -1)                                          # (P,B*N)
            tbest = te.argmin(0)
            for m in metrics:
                pe = _paired_dist(hhat, h[None], m).reshape(P, -1)                # (P,B*N)
                sp = spearman_axis0(pe, te)                                       # (B*N,)
                out[m]["sp"].append(sp[~np.isnan(sp)])
                out[m]["top1"].append((pe.argmin(0) == tbest).mean())
        # advance along vanilla trajectory
        eps = (x - np.sqrt(ab) * O) / np.sqrt(1 - ab)
        ab_next = abar[order[step + 1]] if step + 1 < len(order) else 1.0
        x = np.sqrt(ab_next) * O + np.sqrt(1 - ab_next) * eps
    return out

if __name__ == "__main__":
    g = GMM(seed=0)
    van = run(g, selector="vanilla", seed=0)
    acc = run(g, selector="probe", proxy="cos", seed=0)
    print("sanity: vanilla vs probe-cos energy dist =", round(energy_distance(van, acc), 4))
    print("family size", len(FAM), "taylor", TAYLOR, "online", ONLINE)
