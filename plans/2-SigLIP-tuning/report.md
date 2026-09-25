# Unfreezing the SigLIP tower — report

**Status (2026-09-25): run and measured, not published.** The pre-registered decision rule returns
**PARTIAL**. Unfreezing the tower took CIFAR-100 from 0.768 to **0.916**, past SigLIP2 zero-shot's
0.870 — the first time a laya-vision checkpoint has beaten the contrastive baseline at recognition —
while RVL-CDIP moved +0.018 and still loses. Initiative 1's open question therefore has a split
answer: **recognition was undertrained; document classification is not explained by an undertrained
tower.**

Design: [`siglip-tuning.md`](siglip-tuning.md) · full run-by-run detail:
[`experiments.md`](experiments.md) · what came before:
[`../1-ModernVBERT/report.md`](../1-ModernVBERT/report.md).

---

## What was built

`research/scripts/train_vision.py` gained the ability to train the SigLIP tower, and the
instrumentation needed to tell whether it did:

- **`--vision-lr`** (default `0.0` = frozen, byte-identical to initiative 1) gives the tower its own
  AdamW group, excluded from the encoder group so it is not optimised twice.
  **`--vision-unfreeze-last N`** opens only the last N of the 12 layers plus `post_layernorm`
  (42.53M of 93.52M at N=6); it was never needed.
- **Trainable parameters are logged per group** at startup, so every run states what it trained.
- **Validation accuracy per task, every epoch** — previously only end-of-run calibration touched the
  val split, which is too late to see a 94M-parameter tower forgetting.
- **fp16-skipped updates are counted** per epoch, and **`--keep-epoch-checkpoints`** preserves each
  epoch's weights. Both were added mid-initiative in response to what E5 showed.
- The checkpoint records `vision_lr`, `vision_unfreeze_last`, `trainable_vision_params`, the learning
  rates, the batch shape and the per-epoch validation history.
- `--init-head` now defaults to `random`, and `BENCHMARKS.md` carries initiative 1's honest limits.
- `notebooks/laya_vision_siglip_tuning_2xT4_kaggle.ipynb` runs E5–E7 and evaluates the decision rule
  in code.

Data, seed, epochs and effective batch (64) were held at initiative 1's values throughout. The
tower's learning rate was the only variable.

---

## Findings

### 1. The tower was the binding constraint on recognition

CIFAR-100 **0.768 → 0.916**, against SigLIP2 zero-shot's 0.870. Blind — the same questions with the
image removed — scores 0.050, exactly chance, so none of it is label priors. Initiative 1's
recommendation was right for this task: nine epochs' worth of connector and text-encoder training
could not do what three epochs of a trainable tower did.

### 2. It was not the constraint on document classification

RVL-CDIP **0.342 → 0.360**, still below SigLIP2's 0.422, and flat from the first epoch (0.340 →
0.337 → 0.357) while CIFAR-100 gained 0.077 over the same epochs. Its blind score is 0.032, so the
0.360 is genuine vision — this is a capability ceiling, not a shortcut. Unfreezing 93.5M parameters
for three epochs bought +0.018, which is the clearest evidence in this initiative that something
other than training budget limits documents.

### 3. Text decisions improved rather than degrading

typed-decisions **0.582 → 0.612**, clearing the 0.55 guardrail and every per-type majority baseline.
The guardrail existed because a trainable tower might crowd out the text replay; it did the
opposite. Mean reward turned positive during epoch 2 and stayed there — the model was still learning
at the end of training, not trading one task against another.

### 4. The learning rate is narrow, and the probe was load-bearing

5e-6 beat the frozen control on every image task; **2e-5 destroyed the model**, CIFAR-100 0.052
against a 0.050 chance line with reward flat all epoch. 2e-5 is a defensible guess, below the text
side's 2.5e-5. Without the probe, E6 at 2e-5 would have produced a clean null and the
pre-registered rule would have returned STRUCTURAL LIMIT on an artefact.

### 5. Tuning the tower cost calibration

Before-temperature ECE got worse on four of six tasks: RVL-CDIP 0.388 → 0.509, VQAv2 0.007 → 0.091,
Pets 0.254 → 0.334, typed 0.057 → 0.109. The checkpoint is more confident without being
proportionately more right, and the guardrail "no worse than baseline by > 0.05" **binds on all
four**. This is the price of the CIFAR-100 gain and it must be stated alongside it.

### 6. Temperature fitting degraded CIFAR-100 again — the bucket limit is structural

CIFAR-100 test ECE **0.051 → 0.139** after the fitted temperature, reproducing initiative 1's
finding 6. The `_keep_temp` guard worked on its own terms — pooled val ECE improved 0.2220 → 0.0658 —
but `choice:11+` pools CIFAR-100, RVL-CDIP and typed `choice`, and a pooled win hides a per-task
loss. No guard can fix this: inference sees only question type and option count, so a per-task
temperature is not expressible. Two runs now show it.

### 7. Still no zero-shot transfer, and the blind control caught a false one

Pets **0.128 → 0.102** against SigLIP2's 0.956, near the 0.050 chance line. KonIQ appears to gain
+0.120, but its **blind delta is 0.000** — the checkpoint answers KonIQ entirely from the question
text, with the image unread. Without the blind baseline that would have read as transfer. The
forgetting monitor is otherwise clean: nothing was lost that the checkpoint had.

### 8. No forgetting, no fp16 instability

Every task improved or held across all three epochs. All 8 fp16-skipped updates fall in epoch 1
(GradScaler finding its scale), none afterwards, over roughly seven times the probe's tower
displacement. Two of the plan's four risks did not materialise at 5e-6, and `--vision-unfreeze-last`
was never needed. Memory also held: the run completed at `--micro-batch 8` without the fallback.

---

## Scorecard — the pre-registered decision rule

Fixed in [`siglip-tuning.md`](siglip-tuning.md) before the run, evaluated in code by the notebook.
Baseline is initiative 1's E3 random-init arm.

| rule | measured | outcome |
|---|---|---|
| **Undertrained**: CIFAR-100 ≥ 0.80 **and** RVL-CDIP ≥ 0.40 | 0.916 ✅ / 0.360 ❌ | not met |
| **Structural limit**: neither improves by ≥ +0.03 | CIFAR-100 +0.148 | not met |
| **Partial**: at least one improves by ≥ +0.03, thresholds not met | CIFAR-100 only | ✅ **PARTIAL** |

**Guardrails**

| guardrail | measured | holds |
|---|---|---|
| typed-decisions ≥ 0.55 | 0.612 | ✅ |
| VQAv2 image contribution ≥ +0.08 | +0.138 | ✅ |
| before-temp ECE no worse than baseline by > 0.05 | CIFAR 0.051/0.063, KonIQ 0.086/0.075 | ✅ |
| " | RVL 0.509/0.388, VQAv2 0.091/0.007, Pets 0.334/0.254, typed 0.109/0.057 | ❌ **binds** |
| Pets / KonIQ forgetting monitor (reported, not scored) | 0.102 (was 0.128) / 0.326 blind-only | no real loss |

---

## Conclusions and recommendation

**The consequence the rule pre-committed to is one follow-up run before deciding.** What that run
should test is now much better specified than it was when the plan was written, because the result
splits by task rather than landing between two outcomes.

**RVL-CDIP is more likely resolution-limited than budget-limited.** Sixteen document classes are
separated by layout and printed text, and the checkpoint sees one 512px tile compressed into 64
connector tokens, at which the text on a document is not legible. CIFAR-100 — low-resolution object
recognition, exactly what survives that compression — is the task that moved. `--tiles-per-side 2`
(5 tiles, ~335 image tokens, still inside `max_len 1024`) tests this directly at roughly 3× the
vision compute, call it 3 hours for the full four-task run.

**That belongs in a new initiative, not this one.** Initiative 2 pre-registered itself as
unfreeze-only, and retrofitting a resolution arm would weaken the discipline that makes this result
trustworthy. Initiative 2 closes on PARTIAL with a split diagnosis; the resolution hypothesis is
initiative 3.

**On publishing.** The unfrozen checkpoint beats the frozen one on CIFAR-100 (+0.148), typed
(+0.030) and the VQAv2 image contribution (+0.038), and is worse on calibration for four tasks and
on VQAv2 accuracy (−0.012). It is the better checkpoint, and it is still not a good one to publish
unqualified: two of six tasks lose to a zero-shot baseline that ships today, there is no zero-shot
transfer, and the fitted temperature makes the best task's calibration worse. If it is published,
the model card has to carry finding 2, finding 5, finding 6 and finding 7 in the same breath as
0.916, and `--tiles-per-side` should be settled first.

**What would change the verdict.** If initiative 3 moves RVL-CDIP past 0.40 at higher resolution,
the honest reading becomes "laya-vision was input-limited, not structurally limited", and the line
continues to data and epochs. If resolution does not move it either, then the 64-token connector is
the limit for text-bearing images, and that is the narrower claim to write up and stop at — with
CIFAR-100 0.916 standing as evidence that the design works where the input survives the bottleneck.
