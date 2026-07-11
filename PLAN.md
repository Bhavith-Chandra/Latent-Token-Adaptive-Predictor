# Phase 3 plan: improve the probe, not the policy

**Goal.** Turn the analysis result ("proxy fidelity, not the selection rule, is the bottleneck
in training-free diffusion caching") into a positive, adopted method: a *better cheap probe*
that raises the speed–quality frontier of a real, standard model, beating the current best
selector at matched compute.

This is the version of the work with a real shot at high impact. It requires GPU compute we do
not currently have; the plan below is written so it can be executed on rented cloud GPUs and so
that a *negative* outcome is still a clean, honest result.

---

## 1. Thesis and hypothesis

**Thesis.** In training-free feature caching, the caching *decision signal* (the "probe") is a
better lever than the caching *policy* (the schedule/threshold/selection rule). Existing methods
mostly improve the policy on top of a shallow probe; we improve the probe and hold the policy
fixed.

**Central hypothesis (H1).** Replacing a method's shallow decision signal with a richer but still
cheap probe, while keeping its caching policy unchanged, moves the speed–quality Pareto curve
outward on a real model.

**Mechanistic hypothesis (H2).** The Pareto gain is explained by proxy fidelity: the richer probe
has a higher rank correlation between its predicted caching error and the true caching error.
We measure this directly and show it tracks the Pareto gain across settings. This is where the
exact-residual testbed (Phase 2) becomes the explanation.

Both hypotheses are falsifiable. If a richer probe does not raise proxy fidelity, or raises
fidelity but not quality, we report that; the paper then reverts to a stronger analysis paper.

---

## 2. What counts as a win (pre-registered success criteria)

Primary (any one is a publishable positive result):
- **Iso-compute quality.** At a fixed latency speedup (e.g. 2.0x on FLUX, 1.5x on DiT-XL),
  our probe lowers FID by >= 0.3 (DiT-XL ImageNet-256) or improves FID/CLIP meaningfully on
  FLUX/MS-COCO, versus the same method with its original probe.
- **Iso-quality speed.** At matched quality (FID within 0.2 of the full model), our probe gives
  a >= 15% larger latency speedup than the baseline probe.
- **Dominated Pareto.** The full speedup-vs-FID curve of our probe lies on or below the
  baseline's across at least 3 operating points.

Mechanistic (required for the "why", not sufficient alone):
- Our probe's proxy fidelity (Spearman between predicted and true per-step/per-token caching
  error) is higher than the baseline probe's, and the per-setting fidelity gain correlates with
  the per-setting Pareto gain (this is H2).

Kill criteria (report honestly, do not spin):
- If richer probes raise fidelity but never move the Pareto by the thresholds above on any real
  model, the positive claim dies. We then publish the analysis + emergence science and state
  plainly that better probes did not translate to real-model gains.

---

## 3. Baseline to fork and backbones

**Primary fork: TeaCache** (`ali-vilab/TeaCache`, Apache-2.0; has `TeaCache4FLUX` and video
variants). Why: its decision is driven by a single cheap scalar indicator derived from the
timestep embedding, rescaled to predict the L1 change of the model output. The *policy*
(accumulate the indicator; recompute when it crosses a threshold, else reuse a rescaled cache)
is cleanly separable from the *signal*. We swap only the signal. This isolates the probe and
gives a clean causal claim that matches the thesis. TeaCache is also widely adopted (ComfyUI,
Forge), so beating it is meaningful.

**Fair-baseline harness: `cache-dit`** (PyPI) unifies several caching methods under one API;
use it to run FORA / ToCa / TeaCache / TaylorSeer with consistent NFE and timing so comparisons
are apples-to-apples.

**Backbones, in order of cost (start cheap, de-risk, then scale):**
1. **DiT-XL/2, ImageNet-256** (first target). Cheapest, standard, established caching numbers
   (e.g. ~1.49x at FID ~3.06 at 50 NFE). Runs on a single 24 GB GPU (RTX 4090 / A5000).
2. **PixArt-alpha, MS-COCO / MJHQ** (second). Text-to-image, light, has ToCa baselines.
3. **FLUX.1-dev, MS-COCO-30k** (flagship). Needs a 48-80 GB GPU (A100/H100) or offloading;
   this is the headline result if the cheaper backbones show the effect.

Secondary comparison target (most on-thesis): **DiCache** already uses a shallow-layer online
probe. If our richer probe beats DiCache's probe under DiCache's own policy, that is the
strongest single result, because it is a head-to-head "probe vs probe" at fixed policy.

---

## 4. The probe designs (cheap -> rich, in priority order)

All must stay cheap (a small fraction of a full block) and be training-free or need only a tiny
offline calibration (a least-squares fit on <= 50 calibration prompts; no backprop through the
big network).

1. **Multi-feature linear probe (first to try).** TeaCache predicts output change from one scalar
   (timestep-embedding norm). Replace it with a small feature vector, computed cheaply per step:
   timestep embedding, the norm and a low-rank projection of the first block's modulated input,
   and the previous step's cached residual norm. Fit a linear (or degree-2) map from these
   features to the true output change on calibration prompts. Richer input -> higher fidelity.
2. **Shallow multi-layer probe.** Use the modulated inputs of the first K=2-3 blocks (a partial
   forward), not just layer 1, as the decision signal. Directly tests "probe richness" from
   Phase 2 on a real network. Cost = K/depth of a step; still cheap.
3. **Uncertainty / disagreement probe.** Use the disagreement between two cheap extrapolators
   (e.g. hold-last vs linear) as a confidence signal; recompute when they disagree. Ties to the
   "oracle headroom" finding.
4. **Calibrated per-token probe (stretch).** Where the method is token-level (ToCa), predict
   per-token caching error from a per-token cheap feature and select per token. This is the
   direct real-model analogue of the GMM probe-then-select.

Ablation discipline: for each, hold the policy and NFE fixed, change only the signal, and report
the Pareto curve plus the proxy-fidelity measurement.

---

## 5. Protocol

1. **Reproduce baselines.** Reproduce TeaCache (and 1-2 others via `cache-dit`) on DiT-XL-256 and
   confirm published speedup/FID within tolerance. This validates the harness before any change.
2. **Instrument proxy fidelity.** For a held-out set of prompts, run the *full* sampler and log
   the true per-step (and per-token, for token methods) output change. Compute the Spearman
   correlation between each probe's predicted change and the true change. This is the real-model
   version of the Phase-2 fidelity metric and the bridge to the mechanism.
3. **Swap the probe, hold the policy.** For each probe design, keep TeaCache's threshold logic and
   rescaling unchanged; only the decision signal differs. Sweep the threshold to trace the full
   speedup-vs-FID Pareto.
4. **Compare on the Pareto plane, not single points.** Plot speedup vs FID (and vs CLIP) for
   baseline probe vs each new probe; a win is a dominated curve, not one lucky point.
5. **Link mechanism to method (H2).** Across backbones/operating points, regress Pareto gain on
   proxy-fidelity gain; a positive, significant slope is the paper's spine.
6. **Scale up only if it works.** Move DiT-XL -> PixArt -> FLUX only after the effect appears on
   the cheap backbone. Do not start on FLUX.

Metrics: FID (DiT-XL: ADM/OpenAI ImageNet-256 reference stats, 10k-50k samples; FLUX/PixArt:
FID + CLIP-Score on MS-COCO-30k or MJHQ-30k), plus measured wall-clock latency speedup at
identical hardware and NFE. Report seeds and CIs.

---

## 6. Compute and budget (estimate)

- DiT-XL/2 256, 50 NFE: full-model 10k-sample FID run is ~a few GPU-hours on one 4090/A5000.
  A probe sweep (5-8 operating points x 2-3 probes x 3 seeds) ~ 1-2 GPU-days.
- PixArt-alpha: similar order, slightly more per image.
- FLUX.1-dev 30k FID: ~1 GPU-day per full run on an A100/H100; the sweep ~ 3-5 GPU-days.
- **Total: roughly 1-2 GPU-weeks on a single A100/H100 (or a 4090 for the DiT-XL/PixArt stages).**
  On rented cloud (~$1.5-2.5/hr for a 4090, ~$2-4/hr for an A100), the DiT-XL proof-of-effect is
  ~$50-150; the full study including FLUX is ~$300-800. Start with the cheap stage; only spend
  on FLUX after the effect is confirmed.

Note: none of this runs on the 8 GB M3. The M3 stays useful for the exact-residual testbed,
calibration-fit prototyping, and analysis; the real-model runs need a rented GPU.

---

## 7. Timeline (4 focused weeks)

- **Week 1 - harness + baselines.** Fork TeaCache; stand up `cache-dit`; reproduce DiT-XL-256
  baseline numbers; build the proxy-fidelity logger. Deliverable: a validated pipeline + baseline
  Pareto curve.
- **Week 2 - probe #1 and #2.** Implement the multi-feature linear probe and the shallow
  multi-layer probe; calibrate; sweep thresholds; produce Pareto curves and fidelity numbers on
  DiT-XL. Go/no-go on the effect.
- **Week 3 - mechanism + second backbone.** If Week-2 shows the effect, replicate on PixArt;
  run the H2 regression (Pareto gain vs fidelity gain); add the uncertainty probe; head-to-head
  vs DiCache's probe at fixed policy.
- **Week 4 - flagship + write-up.** If the effect holds, run the FLUX headline; assemble the
  paper (method + real Pareto results + the exact-residual testbed as mechanism + emergence as
  the "why"); ablations, seeds, CIs, honest limitations.

Decision gate after Week 2: if no probe moves the DiT-XL Pareto by the thresholds in Section 2,
stop the positive push and pivot to the analysis+science paper.

---

## 8. Risks and honest mitigations

- **The richer probe is not cheap enough.** A probe that costs too much erases the speedup.
  Mitigation: cap probe cost at a small fraction of a block; report probe FLOPs explicitly and
  fold them into the measured speedup.
- **Fidelity rises but quality does not (H2 fails).** Possible if the policy, not the signal, is
  actually binding on this backbone. Mitigation: this is itself a clean, reportable finding that
  refines the thesis; also try the token-level method where the signal has more room.
- **Baseline is already near the ceiling.** If TeaCache's probe is already near oracle fidelity
  on real models, there is little to win. Mitigation: measure baseline fidelity first (Week 1);
  if it is already high, switch backbones or target the aggressive-acceleration regime where our
  Phase-2 result says headroom is largest.
- **Compute unavailable.** Then execute Sections 1-2 conceptually, keep the analysis+emergence
  paper as the deliverable, and pre-register this plan so the method study is ready when a GPU is.

---

## 9. How it builds on what we already have

- The **exact-residual GMM testbed** (Phase 2) becomes the controlled mechanism section: it is
  where "probe richness -> proxy fidelity" is shown with ground truth, and it predicts *where*
  (aggressive acceleration) the real-model gain should be largest.
- The **optimality lemma** (per-token selection is already optimal) is the motivation: it is why
  we improve the probe rather than the rule.
- The **emergence result** (probe reliability is an emergent property of training) is the "why
  the probe works at all" and the reason a real learned probe should beat our analytic one.
- The **positioning table/figure** already frame the field along granularity x reconstruction;
  the new method plants a flag on the "richer probe" axis that no current method optimizes.

**One-line framing of the target paper.** *The field optimizes the caching policy, but the
policy is already optimal; the real lever is the probe. We make the probe richer, hold the policy
fixed, and move the speed-quality frontier on real diffusion models, with an exact-residual
testbed explaining why.*
