# Unfreezing the SigLIP tower in laya-vision

**Status:** run and measured; the decision rule returned **PARTIAL**. See
[`report.md`](report.md) for the outcome and [`experiments.md`](experiments.md) for the runs. Branch:
`sb/vision` (continues from initiative 1). This file stays the design; it is not updated with
results.

## Context

[`plans/1-ModernVBERT/report.md`](../1-ModernVBERT/report.md) ends with exactly one untried lever.
The first trained `laya-vision` checkpoint reads images genuinely — removing the image and re-scoring
the same questions drops CIFAR-100 from 0.745 to 0.060 (chance 0.050) and RVL-CDIP from 0.305 to
0.000 — yet it loses to SigLIP2 zero-shot on both:

| trained task | laya-vision (frozen tower) | SigLIP2 zero-shot | blind |
|---|---|---|---|
| CIFAR-100 | 0.768 | **0.870** | 0.060 |
| RVL-CDIP | 0.342 | **0.422** | 0.000 |
| VQAv2 yes/no | **0.670** | 0.524 | 0.560 |

Throughout that run the SigLIP tower was frozen by design, so only the 9.4M connector and the 149M
text encoder adapted, on 24k images over 3 epochs — while the baseline brings a text tower aligned on
billions of image–text pairs. The tower is 93.5M parameters in 12 layers.

**The question is binary:** is the checkpoint *undertrained*, or is a 64-token connector bottleneck a
*structural limit* on recognition? Unfreezing the tower is the cheapest experiment that separates
those two, and nothing else should change while it runs.

**Hypothesis.** Letting the tower adapt at a low learning rate closes most of the gap to SigLIP2 on
CIFAR-100 and RVL-CDIP. — *Outcome: half right. CIFAR-100 passed SigLIP2 (0.916 vs 0.870); RVL-CDIP
moved +0.018 and still loses. See [`report.md`](report.md) findings 1 and 2.*

**Falsifier, stated up front.** If the tower demonstrably trains (loss moves, trainable parameter
count is non-zero, per-epoch validation shifts) and CIFAR-100 and RVL-CDIP still do not move, then
the limit is structural, the write-up is the deliverable, and this line of work stops. That is a
result, not a failure.

## Land first

Independent of the experiment, and carried from the report's pre-publication list. Every run here
uses random init anyway, and the documentation claims should be corrected whether or not unfreezing
works:

1. ✅ **Default `--init-head` to `random`** in `research/scripts/train_vision.py`. Warm-starting from
   `laya-multilingual` lost on all six tasks at identical hyper-parameters; a head trained on
   mmBERT's representation space is a worse prior than noise on Ettin's. Keep `multilingual` as an
   option, with the measured outcome in the flag's help text.
2. ✅ **State VQAv2 as "0.560 blind → 0.660 with image"**, never as a margin over SigLIP2: answer
   priors in the question text supply most of the score, and the image is worth +0.100.
   (`BENCHMARKS.md` *Limits*.)
3. ✅ **Say plainly that there is no zero-shot transfer** (Pets 0.128 against SigLIP2's 0.956, chance
   0.050) and that the model is English-first (ModernVBERT's Ettin text side).
4. ✅ **Confirm** the latency limit and its batching mitigation are in `BENCHMARKS.md` — already
   written, needs re-checking only if numbers change.

## Code changes

All in `research/scripts/train_vision.py` unless noted. The default path must stay byte-identical to
today's behaviour, so the frozen baseline remains comparable.

- **`--vision-lr`** (default `0.0` = frozen). Non-zero unfreezes `model.encoder.vision_model` and
  adds a **third AdamW param group** next to the existing encoder/head groups, which currently read:

  ```python
  enc_params  = [p for n, p in model.named_parameters() if n.startswith("encoder.") and p.requires_grad]
  head_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
  ```

  The tower's parameters must be excluded from `enc_params` when they get their own group, or they
  would be optimised twice at two learning rates.
- **`--vision-unfreeze-last N`** (default 0 = the whole tower). A fallback for memory pressure or
  forgetting, *not* an experimental arm. The 12 layers are 7.09M each; the last 6 are 43M.
- **Log trainable parameter counts per group** at startup, so every run states what it actually
  trained rather than what its flags implied.
- **Per-epoch validation accuracy per task.** Today only the end-of-run calibration touches the val
  split, which is too late to see overfitting or forgetting in a 94M-parameter tower fine-tuned on
  24k images. Reuse the existing `Items` dataset, `collate` and the val cache the run already builds;
  report per task, since a mean across tasks would hide exactly the trade-off being watched.
- **Record `vision_lr`** in the config's `vision` block, so a checkpoint states how it was trained.

## Implementation notes

What landed, and where it differs from the design above. All in `research/scripts/train_vision.py`.

- **`--vision-lr` works as specified.** Verified on CPU with a `--smoke` run per arm: the default
  reports `2 optimiser groups … SigLIP tower frozen (0.00M of 93.52M trainable)`, `--vision-lr 5e-6`
  reports `3 optimiser groups … SigLIP tower @ 5.0e-06 (93.52M of 93.52M trainable)`, and the
  encoder group is 158.48M in both — the tower moves into its own group rather than being added to
  the encoder's. After one optimiser step, 160 of the tower's 208 tensors differ between the two
  checkpoints, so the tower demonstrably trains; `laya.Agent` still loads the result and answers.
- **Per-epoch validation is on by default**, with `--no-val-epoch` to skip it, rather than being
  opt-in. It is the only in-run signal of forgetting, and a monitor nobody switches on is not a
  monitor. It runs on rank 0 only (no collectives), followed by a barrier so the other ranks do not
  start the next epoch and sit in an all-reduce while rank 0 validates and saves.
- **`--vision-unfreeze-last N` also unfreezes the tower's `post_layernorm`**, which sits after the
  layers and belongs to the same block. Measured: `--vision-unfreeze-last 6` trains 42.53M of the
  tower's 93.52M, as the 43M estimate above predicted.
- **The checkpoint records more than `vision_lr`.** The `vision` block carries `vision_lr` and
  `vision_unfreeze_last`; `training` carries the learning rates, micro-batch, grad-accum,
  `trainable_vision_params` and the per-epoch validation history. A checkpoint states how it was
  trained, and the E5 comparison reads the checkpoints rather than the scrollback.
- **Notebook:** [`notebooks/laya_vision_siglip_tuning_2xT4_kaggle.ipynb`](../../notebooks/laya_vision_siglip_tuning_2xT4_kaggle.ipynb)
  drives E5–E7 on a Kaggle 2×T4, and evaluates the decision rule below in code.

## Experiments

Data, seed, epochs and effective batch are fixed at initiative 1's values. The tower's learning rate
is the only variable.

*E5 chose 5e-6 and caught a 2e-5 collapse that would have faked a null result.*

**E5 — learning-rate probe.** 1 epoch, `--n-per-task 3000`, at `2e-6`, `5e-6` and `2e-5` (the text
side runs at 2.5e-5), plus a frozen control on the same subset. Decide on per-task validation
accuracy. ~10–12 min each. This exists so a null result in E6 cannot be blamed on a guessed LR.

**E6 — full run.** 3 epochs, full data, the winning LR, random init head,
`--micro-batch 8 --grad-accum 4` (effective batch 64, as in E3 — comparability depends on it).
~80 min.

**E7 — benchmark.** `bench_vision.py` with both checkpoints as arms (`frozen=` from E3, `unfrozen=`
from E6), blind baselines on, `--n 500 --seed 0`, reusing the same data cache.

## Decision rule, pre-registered

Fixed before the run, so the outcome cannot be rationalised afterwards. Baseline is initiative 1's
random-init arm; SigLIP2 references are 0.870 (CIFAR-100), 0.422 (RVL-CDIP), 0.956 (Pets).

| outcome | rule | what follows |
|---|---|---|
| **Undertrained** | CIFAR-100 ≥ 0.80 **and** RVL-CDIP ≥ 0.40 | continue the line: scale data and epochs, revisit KonIQ |
| **Partial** | at least one improves by ≥ +0.03, thresholds not met | one follow-up run before deciding |
| **Structural limit** | neither improves by ≥ +0.03 | stop; write up the narrower claim |

The original wording of the middle row was "exactly one improves by ≥ +0.03". Both improving without
either reaching its threshold is the same situation — real movement, not enough of it — so it scores
`PARTIAL` as well. Fixed here and in the notebook before any number existed, not afterwards.

**Guardrails**, reported whether or not they bind:

- typed-decisions ≥ 0.55 — text decisions must not collapse (baseline 0.582);
- VQAv2 image contribution (with − blind) ≥ +0.08 (baseline +0.100);
- before-temperature ECE no worse than baseline by > 0.05;
- Pets and KonIQ zero-shot reported as a **forgetting monitor**, not a stop criterion — both are near
  chance already, and a drop would still be informative about what tuning the tower costs.

## Risks

- **Memory.** A trainable tower adds ~0.37 GB of gradients and ~0.75 GB of AdamW states. E3 used
  5.2/15 GiB at micro-batch 8, so it should fit. If a T4 runs out, `--micro-batch 4 --grad-accum 8`
  holds the effective batch at 64. Gradient checkpointing is already enabled on the encoder and
  `SiglipVisionModel` supports it.
- **Catastrophic forgetting** of general SigLIP features on only 24k images. Mitigated by the low LR,
  watched by per-epoch validation and the Pets/KonIQ probes, with `--vision-unfreeze-last` as the
  fallback.
- **fp16 instability** with a newly trainable tower: watch for GradScaler step skips and NaNs. A T4
  has no usable bf16.
- **DDP.** A text-only replay batch now leaves the *whole tower* without gradients, so
  `find_unused_parameters=True` becomes load-bearing rather than merely prudent.
- **Comparability.** Same data cache, seed, effective batch and epoch count as E3, or the comparison
  is void.

## Out of scope

- Data scaling, more epochs, and adding KonIQ — all plausible levers, deliberately held fixed so that
  any movement is attributable to the tower. They become follow-ups only if E6 succeeds.
- Partial unfreezing as a first-class arm (the flag exists as a fallback).
- Publishing a checkpoint.
- Anything at inference time: unfreezing changes no inference-time architecture, so the measured 2.3×
  image latency and its batching mitigation stand unchanged.

## Validation

- ✅ `--vision-lr 0` reproduces today's behaviour exactly: frozen tower, two param groups.
- ✅ `tests/test_vision.py` and the other seven suites pass unchanged; `ruff` and `compileall` clean.
- ✅ A CPU `--smoke` run with `--vision-lr 5e-6` reports three param groups and a non-zero trainable
  tower count, and one with `--vision-lr 0` reports two.
- ✅ A 2-GPU `--smoke` run before any real stage, as in initiative 1 — DDP is where the last two
  training bugs appeared. Ran clean on a Kaggle 2×T4 before E5.

## Deliverables

✅ [`report.md`](report.md) and [`experiments.md`](experiments.md), written after E5–E7.
`BENCHMARKS.md` is unchanged: nothing here is published, so no published number changed.
