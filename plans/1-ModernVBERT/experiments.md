# Vision stream — experiment log

Every run recorded here, in order, with its setup and what it actually showed. Numbers are as
measured; nothing is re-stated from expectation. The summary and the conclusions drawn from these
runs are in [`report.md`](report.md); the design is in [`vision-stream.md`](vision-stream.md).

Hardware: a Kaggle 2×T4 session unless stated otherwise ("local CPU" is an Apple-silicon laptop).

---

## E1 — Pipeline validation (offline, no training)

**Setup.** A tiny from-config `ModernVBertModel` (32-dim text side, 16-dim SigLIP, 32px images) with
an in-memory word-level tokenizer, in `tests/test_vision.py`; then the real 267M ModernVBERT with a
random head, loaded through `Agent` from an on-disk checkpoint.

**Result.** 48 offline checks pass. The load-bearing ones:

- the image block lands after every option marker, and markers are byte-identical to the text-only
  sequence — the head/option budget is untouched;
- the state budget shrinks by exactly `len(prefix_ids)`, and an oversized block raises;
- **computing image features once and repeating them across question rows matches per-row vision to
  < 1e-5**, which is what makes the encode-once design safe;
- a literal `<image>` in user text is scrubbed and cannot pose as a placeholder;
- `images=` on a text checkpoint raises; `Router.route(images=[…])` picks `vision`.

Real-architecture smoke: 1 image = 67 tokens per row, checkpoint round-trips through
`processor/`, and `Router(models={"vision": …})` routes and answers.

---

## E2 — Image latency, and the preprocessing fix

**Setup.** p50 over 7 calls, an untrained checkpoint (timings only), 800×600 image.

**First measurement (T4).** 197 / 195 / 198 ms for 1 / 5 / 10 questions — flat, confirming the vision
tower runs once per call rather than once per question. But 1 question cost 166 ms against 27 ms
text-only, and decomposition showed **preprocessing, not the tower, dominating: 119 ms of it**.

**Cause.** The Idefics3 processor resizes to `size.longest_edge` — 2048 in the ModernVBERT config —
before squashing the image to a single 512px tile. An 800×600 image was being enlarged to 2048×1536
and then thrown away.

**Fix.** Cap the size in `to_pil` before the processor sees the image. A per-call `size` kwarg alone
proved unreliable (it appeared not to be honoured in the installed transformers), so the downscale
happens in laya's own code.

| same T4, 1 image | before | after |
|---|---|---|
| preprocess (CPU) | 125 ms | **24.7 ms** |
| vision tower + connector (GPU) | 64.7 ms | 62.5 ms |
| `predict`, 1 image, 1 question | 188 ms | **74.7 ms** |
| `predict`, text only | 27 ms | 26.8 ms |

**Also measured and rejected:** PIL's `draft` reduced-scale JPEG decode — no gain below ~2048px
(10.3 vs 10.0 ms at 1024×768), ~20% above it, and no training source is that large.

**Final numbers, from the trained-run benchmark** (random-init arm):

| p50 | 1 question | 5 questions | 10 questions |
|---|---|---|---|
| text only | 26.2 ms | 27.7 ms | 29.8 ms |
| 1 image | 59.5 ms | 61.9 ms | 70.1 ms |
| per question, with the image | 59.5 ms | 12.4 ms | **7.0 ms** |

An image costs **2.3×** a text-only call, against a "< 2×" criterion. Most of the remainder is the
SigLIP2 forward pass at 512px.

---

## E3 — First trained run: warm-start vs random init (2026-09-23)

**Setup.** Kaggle 2×T4 DDP, tasks `cifar100,rvl_cdip,vqav2_yesno,typed` — **`koniq` was not
trained**, so KonIQ and Pets are both zero-shot probes. 29,400 training items (8,000 + 8,000 + 8,000
+ 5,400 typed replay), 3 epochs, effective batch 64, SigLIP frozen, connector + text encoder at
2.5e-5 and the head at 1e-4. ~26 min/epoch, 5.2/15 GiB per T4. Both arms used identical
hyper-parameters; only `--init-head` differed. Benchmark: 500 test examples per task, seed 0.

| task | random init | warm (multilingual) | SigLIP2 zero-shot | majority | random | blind (no image) |
|---|---|---|---|---|---|---|
| CIFAR-100 (trained) | 0.768 | 0.742 | **0.870** | 0.022 | 0.050 | 0.060 |
| RVL-CDIP (trained) | 0.342 | 0.336 | **0.422** | 0.128 | 0.063 | 0.000 |
| VQAv2 yes/no (trained) | **0.670** | 0.510 | 0.524 | 0.522 | 0.500 | 0.560 |
| typed-decisions (text replay) | 0.582 | 0.516 | — | — | 0.312 | 0.580 |
| KonIQ (zero-shot) | 0.206 | 0.158 | 0.102 | 0.544 | 0.200 | — |
| Pets (zero-shot) | 0.128 | 0.080 | **0.956** | 0.064 | 0.050 | — |

Score MAE, KonIQ: random 0.396, warm 0.479, SigLIP2 0.605. typed by type (random init): choice
0.564, noul 0.686, score 0.529 — each above its own majority baseline.

**Warm-start is refuted.** Random init wins on all six tasks under identical hyper-parameters,
clearest on VQAv2 (0.670 vs 0.510) and typed (0.582 vs 0.516). A head trained on mmBERT's
representation space is a worse starting point than noise on Ettin's.

**Temperature fitting hurt.** CIFAR-100 test ECE went **0.063 → 0.285** (Brier 0.365 → 0.459) after
the fitted temperature. A bucket pools every task sharing a (type, option count), so `choice:11+`
holds CIFAR-100, RVL-CDIP and typed `choice` at once; fitting one scalar to the pooled NLL flattens
a task that was already well calibrated. Before-temperature ECE was fine on CIFAR (0.063), VQA
(0.007), KonIQ (0.075) and typed (0.057), and poor on RVL-CDIP (0.388) and Pets (0.254).

---

## E4 — Blind ablation: what is the image worth?

**Setup.** The random-init checkpoint, 200 test examples per task, seed 0, scored twice through the
identical code path: once with the image, once with `images=[]`.

| task | with image | blind | delta | chance |
|---|---|---|---|---|
| CIFAR-100 | 0.745 | 0.060 | **+0.685** | 0.050 |
| RVL-CDIP | 0.305 | 0.000 | **+0.305** | 0.063 |
| VQAv2 yes/no | 0.660 | 0.560 | +0.100 | 0.500 |
| typed-decisions (control, no images) | 0.580 | 0.580 | +0.000 | 0.312 |

(Accuracies differ slightly from E3 because n=200 rather than 500.)

**The control passed** — `typed` has no images, and its delta is exactly 0.000, so the harness is
measuring what it claims to.

**CIFAR-100 and RVL-CDIP read the image**, blind scoring at or below chance. RVL-CDIP's 0.342 is
therefore genuine vision, not label priors — the gap to SigLIP2 is a capability gap, not a shortcut.

**VQAv2's margin is a third of what it looked.** Blind already scores 0.560 from answer priors in
the question text, so the image is worth **+0.100**, not the +0.146 the SigLIP2 comparison implies.

An oddity, unresolved and immaterial to the conclusions: blind RVL-CDIP is exactly 0.000 rather than
the ~0.063 chance rate, i.e. blind the model collapses onto a label that is essentially never
correct, rather than guessing.

---

## Bugs these runs surfaced

Each was found by a run, not by reading the code:

| found by | bug | fix |
|---|---|---|
| E1 (smoke, DDP) | `train_ex[rank::world]` gave ranks unequal lengths — one rank would block forever on an all-reduce | drop the remainder so every rank runs the same number of steps |
| E1 | `total_updates` floored the trailing accumulation cycle, so the cosine schedule ran past `T_max` and rose again | count it with `ceil` |
| E3 prep | every rank built every split independently, decoding and re-encoding each image; a 5000-row shuffle buffer had to fill first | rank 0 builds, others wait, all load a cache; `decode=False`; buffer 1000; progress every 1000 |
| E3 report | benchmark scored `koniq` as a trained task from a static list, turning "both near chance" into a win | trained tasks come from the checkpoint's own `cfg["vision"]["tasks"]` |
| E3 report | typed-decisions mixes three question types, but the task took the first example's type: Brier pooled 2-option with 20-option, score MAE was dropped | per-type metrics and baselines |
| E3 | a fitted temperature could be shipped that calibrates worse than none, and outside the `[0.5, 5]` band inference actually applies | ship only if it improves val ECE, judged at the clamped value |
| E2 | preprocessing dominated image latency | cap the resize in `to_pil` |
| E4 | blind was a one-off cell | a standard baseline in `bench_vision.py` (`--no-blind` to skip) |
