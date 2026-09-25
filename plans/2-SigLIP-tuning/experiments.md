# SigLIP tuning — experiment log

Every run recorded here, in order, with its setup and what it actually showed. Numbers are as
measured. The summary and the conclusions drawn from these runs are in [`report.md`](report.md); the
design is in [`siglip-tuning.md`](siglip-tuning.md). Initiative 1's runs (E1–E4) are in
[`../1-ModernVBERT/experiments.md`](../1-ModernVBERT/experiments.md), and this log continues its
numbering.

Hardware: a Kaggle 2×T4 session throughout ("local CPU" is an Apple-silicon laptop). Data, seed,
epochs and effective batch are initiative 1's values; the tower's learning rate is the only variable.

---

## E5 — Learning-rate probe (2026-09-25)

**Setup.** 1 epoch, `--n-per-task 3000` (12,000 train items), `--micro-batch 8 --grad-accum 4` on
2 GPUs = effective batch 64, random-init head, tasks `cifar100,rvl_cdip,vqav2_yesno,typed`. Four
arms on the identical subset: a frozen control and three tower learning rates. Decided on per-task
**validation** accuracy, 600 examples per task. ~12.9 min per frozen arm, ~15.1 min per unfrozen one
— a backward pass through the tower costs about 21%.

| arm | CIFAR-100 | RVL-CDIP | VQAv2 | typed | avg loss |
|---|---|---|---|---|---|
| frozen (control) | 0.640 | 0.300 | 0.595 | **0.522** | 1.0451 |
| 2e-6 | 0.760 | 0.342 | 0.580 | 0.507 | **0.9584** |
| **5e-6** | **0.770** | **0.343** | **0.632** | 0.510 | 0.9840 |
| 2e-5 | 0.052 | 0.083 | 0.553 | 0.482 | 1.5393 |

**5e-6 won** and was carried into E6: it beat the frozen control on every image task (+0.130 CIFAR,
+0.043 RVL, +0.037 VQAv2) for −0.012 on typed, and beat 2e-6 on all three.

**2e-5 destroys the tower.** CIFAR-100 fell to 0.052 against a 0.050 chance line and RVL-CDIP to
0.083, with reward flat at about −1.5 for the whole epoch: the run was not slow to learn, it had
stopped learning. 2e-5 is a defensible guess — it sits below the text side's 2.5e-5 — so **without
this probe a null result in E6 would have been attributed to a structural limit rather than to a
learning rate**. That is the whole reason the probe was pre-registered, and it paid for itself.

---

## E6 — Full run at 5e-6 (2026-09-25)

**Setup.** 3 epochs, full data: 29,400 train items (14,700 per rank), 1,380 optimiser updates,
effective batch 64, random-init head, `--vision-lr 5e-6` with the whole 93.52M tower trainable
alongside the 158.48M text+connector group at 2.5e-5 and the 14.97M head at 1e-4. 5,771 s ≈ 96 min.

| epoch | CIFAR-100 | RVL-CDIP | VQAv2 | typed | avg loss | fp16-skipped updates |
|---|---|---|---|---|---|---|
| 1 | 0.850 | 0.340 | 0.573 | 0.540 | 0.8240 | 8 |
| 2 | 0.905 | 0.337 | 0.663 | 0.575 | 0.5238 | 0 |
| 3 | **0.927** | **0.357** | 0.655 | **0.647** | 0.4392 | 0 |

(Validation accuracy, 600 examples per task — not comparable with the test numbers in E7.)

**No forgetting and no fp16 instability.** Every task improved or held across all three epochs, and
mean reward turned positive during epoch 2 and stayed there. The 8 skipped updates are all in
epoch 1, which is GradScaler finding its scale; zero afterwards. Three epochs at 5e-6 accumulate
roughly seven times the probe's tower displacement without any sign of the 2e-5 collapse, so the two
risks the plan listed — catastrophic forgetting and fp16 instability on a T4 — did not materialise
at this learning rate.

**RVL-CDIP was flat from the first epoch** (0.340 → 0.337 → 0.357) while CIFAR-100 gained 0.077 over
the same epochs. The divergence is visible during training, not only in the benchmark.

**Fitted temperatures:** choice 2.453 (val ECE 0.2220 → 0.0658), noul 1.565 (0.0516 → 0.0218), score
refused (0.1353 → 0.1387); buckets `choice:11+` 2.4834 and `noul:2` 1.5648. The refusal guard
behaved as designed on the val split — and E7 shows why that is not sufficient.

---

## E7 — Benchmark (2026-09-25)

**Setup.** `bench_vision.py --model unfrozen=… --n 500 --seed 0`, blind baselines on, held-out test
splits. The frozen arm is initiative 1's recorded E3 random-init row rather than a re-score: that
checkpoint was not reachable from this session. Same harness, same seed, same `--n`, same code path,
so the comparison holds; a same-session re-score would have been tighter.

| task | E3 frozen | **E6 unfrozen** | Δ | SigLIP2 zero-shot | blind | image worth |
|---|---|---|---|---|---|---|
| CIFAR-100 (trained) | 0.768 | **0.916** | **+0.148** | 0.870 | 0.050 | +0.866 |
| RVL-CDIP (trained) | 0.342 | 0.360 | +0.018 | **0.422** | 0.032 | +0.328 |
| VQAv2 yes/no (trained) | 0.670 | 0.658 | −0.012 | 0.524 | 0.520 | +0.138 |
| typed-decisions (text replay) | 0.582 | 0.612 | +0.030 | — | — | — |
| KonIQ (zero-shot) | 0.206 | 0.326 | +0.120 | 0.102 | 0.326 | **+0.000** |
| Pets (zero-shot) | 0.128 | 0.102 | −0.026 | **0.956** | 0.054 | +0.048 |

ECE before → after the fitted temperature: CIFAR-100 0.051 → 0.139, RVL-CDIP 0.509 → 0.205, VQAv2
0.091 → 0.059, KonIQ 0.086 → 0.086, Pets 0.334 → 0.117, typed 0.109 → 0.129. typed clears its
majority baseline on all three question types.

**CIFAR-100 beats SigLIP2 zero-shot**, 0.916 against 0.870 — the first time any laya-vision
checkpoint has beaten the contrastive baseline at recognition. Blind scores 0.050, exactly chance,
so the whole of it is the image being read.

**RVL-CDIP did not move** (+0.018) and still loses to SigLIP2. Its blind score is 0.032, so what it
does score is genuine vision rather than label priors — the same picture as initiative 1, at a
higher parameter count and three more epochs of tower training.

**KonIQ's +0.120 is an artefact.** Its blind delta is **0.000**: the checkpoint answers KonIQ's
image-quality scores entirely from the question text, with the image unread. It was never trained on
KonIQ, and this is the blind control doing its job — without it, +0.120 on a zero-shot task would
have read as transfer.

**Calibration got worse before temperature on four of six tasks**: RVL-CDIP 0.388 → 0.509, VQAv2
0.007 → 0.091, Pets 0.254 → 0.334, typed 0.057 → 0.109. The tuned model is more confident and not
proportionately more right.

**Temperature fitting degraded CIFAR-100 again**, test ECE 0.051 → 0.139, exactly initiative 1's
finding 6 with a second data point. The `_keep_temp` guard did what it was built to do — pooled val
ECE improved 0.2220 → 0.0658 — but the `choice:11+` bucket pools CIFAR-100, RVL-CDIP and typed
`choice`, and a pooled win can hide a per-task loss. The guard cannot fix this: inference sees only
question type and option count, so a per-task temperature is not expressible.

**VQAv2 fell slightly in accuracy while relying less on priors:** 0.670 → 0.658, with blind dropping
0.560 → 0.520 and the image contribution rising +0.100 → +0.138. Note the blind figures are not
strictly comparable — E4 measured blind at n=200, E7 at n=500.

---

## Notes these runs produced

| found by | what | outcome |
|---|---|---|
| DDP smoke | torch's scheduler warning fires on every fp16-skipped update, saying nothing about how many | count skipped updates per epoch and record them in the checkpoint (`4ac76a9`) |
| E5 | a trainable tower can collapse outright, and the rolling checkpoint is overwritten every epoch | `--keep-epoch-checkpoints`, off by default (`25a0a76`) |
| E5 | frozen arm logged "find_unused_parameters=True … did not find any unused parameters" | benign: with four tasks shuffled together, every micro-batch happened to contain an image row. The flag stays — a text-only micro-batch is still possible, and with a trainable tower it would leave the whole tower ungradiented |
| E5 | probe arms cost 21% more per epoch than the frozen control | E6 estimated at ~100 min, measured 96 |
| E7 | KonIQ's zero-shot gain is entirely blind | the blind baseline is what caught it; keep it on for every arm |

Both instrumentation changes landed mid-initiative: E5 ran without the skip counter, E6 and E7 with
it. Neither changes training, so the arms remain comparable.
