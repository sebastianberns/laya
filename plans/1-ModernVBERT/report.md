# laya-vision — report

**Status (2026-09-23): built and measured, not published.** The vision stream works end to end —
package support, training, benchmark, and a trained checkpoint on a Kaggle 2×T4. It does **not** meet
the plan's headline success criterion, and on present evidence should not ship as a public
checkpoint without another training run.

Design: [`vision-stream.md`](vision-stream.md) · full run-by-run detail:
[`experiments.md`](experiments.md).

---

## What was built

A fourth checkpoint type, `laya-vision`: **ModernVBERT** (Ettin-150M text + SigLIP2-base-16-512 + a
pixel-shuffle connector) as the encoder, with laya's existing `DecisionModel` head on top, so typed
questions (`choice` / `score` / `noul`) can be asked about images in one forward pass.

- **Package.** `laya/vision.py`; image support in `common.py`, `agent.py`, `router.py`; a `[vision]`
  extra (transformers ≥ 5.3, pillow). An image joins the *state* as a 67-token block placed after
  every option marker, so markers and the `head_max_len` budget are unchanged. The vision tower runs
  **once per call** and its features are shared across every question in that call. Images on a
  text-only checkpoint raise; `Router` sends image requests to `vision`.
- **Research.** `research/scripts/train_vision.py` (the notebook's RLCD loop, frozen SigLIP, optional
  head warm-start), `bench_vision.py`, `vision_data.py`, and
  `notebooks/laya_vision_train_2xT4_kaggle.ipynb`.
- **Tests.** `tests/test_vision.py`, 48 offline checks on a tiny from-config ModernVBERT, wired into
  CI and release.

Every text-only code path is unchanged, and the seven pre-existing test files still pass.

---

## Findings

### 1. The design works. The training is undercooked.

The image is genuinely being read, on every trained task. Removing the image and re-scoring the same
questions (E4) drops CIFAR-100 from 0.745 to 0.060 (chance 0.050) and RVL-CDIP from 0.305 to 0.000.
So the gap to the baseline is a capability gap, not the model exploiting label priors or a broken
pipeline — which is the more encouraging of the two failure modes, and points at representation
quality rather than plumbing.

### 2. It loses to SigLIP2 zero-shot at recognition, and wins where a decision head should

| trained task | laya-vision | SigLIP2 zero-shot | blind |
|---|---|---|---|
| CIFAR-100 | 0.768 | **0.870** | 0.060 |
| RVL-CDIP | 0.342 | **0.422** | 0.000 |
| VQAv2 yes/no | **0.670** | 0.524 | 0.560 |

Contrastive pretraining wins at what it was built for — matching an image to a noun — using a text
tower aligned on billions of pairs. It cannot answer a yes/no question, and there laya-vision wins.

**State the VQAv2 result honestly:** blind (question text, no image) already scores 0.560 from answer
priors, so the image is worth **+0.100**, not the +0.146 the SigLIP2 margin suggests.

### 3. No zero-shot transfer

Pets, never trained on: **0.128** against SigLIP2's 0.956, barely above the 0.050 chance line. This
checkpoint answers the categories it was trained on. "Typed questions about any image" is not
supported, and must not be claimed.

### 4. Warm-starting the head from laya-multilingual is worse than random

Random init wins on **all six** tasks under identical hyper-parameters (VQAv2 0.670 vs 0.510; typed
0.582 vs 0.516). A head trained to read mmBERT's representation space is a worse prior than noise on
Ettin's, which is a different space. `--init-head random` should become the default. Note that no
other published checkpoint was even eligible: `laya` and `laya-typed-decisions` are 1024-wide against
ModernVBERT's 768.

### 5. Text decisions did not collapse

typed-decisions replay held at **0.582**, between the base checkpoints' 0.36 and the fine-tuned
0.766, clearing majority on all three question types. The replay mix did its job.

### 6. One temperature per (type, option count) cannot calibrate a multi-task checkpoint

Fitting made calibration **worse** where the model was already good: CIFAR-100 test ECE 0.063 →
0.285. The `choice:11+` bucket pools CIFAR-100, RVL-CDIP and typed `choice`, which want different
temperatures; the pooled fit flattens the well-calibrated one. Training now refuses a temperature
that does not improve validation ECE, but the underlying limit stands — inference sees only question
type and option count, so a per-task temperature is not expressible.

### 7. An image costs 2.3× a text-only call, and batching is the mitigation

59.5 ms with one image against 26.2 ms text-only on a T4, against a "< 2×" criterion. The tower runs
once per call, so ten questions about one image cost 70.1 ms total — **7 ms per question**. A
preprocessing fix along the way took the image path from 188 ms to 75 ms (E2); the remainder is
mostly the SigLIP2 forward pass at 512px, which only a smaller vision tower would reduce.

---

## Success criteria — scorecard

| criterion | outcome |
|---|---|
| Beats SigLIP2 zero-shot on the trained tasks | ❌ **partial** — VQAv2 yes; CIFAR-100 and RVL-CDIP no |
| Calibrated after temperature fitting (ECE ≤ 0.1) | ❌ fails on CIFAR-100, RVL-CDIP, Pets — though fitting *caused* the CIFAR failure |
| ≥ majority class on typed-decisions text | ✅ all three question types |
| < 2× text-only latency for 1 image | ❌ 2.3× (but ~7 ms per question at ten) |

---

## Conclusions and recommendation

**The untried lever is the frozen SigLIP tower.** Only the connector and a 150M text encoder adapted,
on 24k images over 3 epochs, while the benchmark's baseline brings a tower aligned on billions of
pairs. Unfreezing at a low LR (~5e-6, below the text side's 2.5e-5) is one ~80-minute run and would
settle whether this is undertraining or a structural limit of a 64-token connector bottleneck. It
diverges from the design, which specified a frozen tower, so `vision-stream.md` would need updating.

**Either way, before anything is published:**

1. default `--init-head` to `random` (finding 4);
2. report VQAv2 as "0.560 blind → 0.660 with image", never as a margin over SigLIP2 (finding 2);
3. say plainly that there is no zero-shot transfer (finding 3), and that the model is English-first
   (ModernVBERT's Ettin text side) — no multilingual image claims;
4. carry the latency limit and its batching mitigation into `BENCHMARKS.md` (already done).

**If the unfreeze run does not move CIFAR-100 and RVL-CDIP**, the honest outcome is that the write-up
is the deliverable: a working, calibrated, fast typed-decision head over images that is beaten by
contrastive zero-shot at recognition, and earns its place only where a question cannot be phrased as
an image-text similarity. That is a narrower claim than the plan set out to make, and worth stating
rather than tuning toward.
