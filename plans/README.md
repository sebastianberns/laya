# plans/

Design plans, and the record of what happened when they were built. One numbered directory per
initiative. Start here, then read the initiative's `report.md` if it has one — it is the fastest way
to learn what is true now, rather than what was intended when the plan was written.

## Initiatives

| | initiative | status | start with |
|---|---|---|---|
| 1 | [ModernVBERT vision stream](1-ModernVBERT/) — images as an input to laya, via a fourth checkpoint `laya-vision` | implemented and trained once, **not published**; branch `sb/vision` | [`report.md`](1-ModernVBERT/report.md) |
| 2 | [SigLIP tuning](2-SigLIP-tuning/) — unfreeze the vision tower, to decide whether `laya-vision` is undertrained or structurally limited | planned, nothing run; branch `sb/vision` | [`siglip-tuning.md`](2-SigLIP-tuning/siglip-tuning.md) |

## Layout

```
plans/
  README.md                 this file: the index
  <n>-<topic>/
    <topic>.md              the design: what to build and why. Written first.
    report.md               what was measured, what it means, what to do next. Written after a run.
    experiments.md          the run-by-run log: setup, numbers, what each run showed.
```

Only the design document is required. `report.md` and `experiments.md` appear once an initiative has
produced results worth keeping; a plan that has not been built yet is a single file.

## What goes where

**`<topic>.md` — the design.** Context, the decisions and their rationale, the code changes, the
validation strategy, the success criteria, and what is out of scope. It stays a design document: it
does not accumulate results. When the implementation diverges from it, update it — either fix the
decision in place, or record the divergence in an *Implementation notes* section, so the file never
describes code that does not exist. When a measurement settles a decision the plan made, note the
outcome next to that decision and link to the detail, rather than leaving the plan quietly
contradicting the results.

**`report.md` — the summary.** What was built, the findings, a scorecard against the plan's success
criteria, and the recommendation. This is the document to write for someone who was not there, and
the one to read before resuming work. Findings are stated as what was measured, including the ones
that went the wrong way; a refuted assumption is a finding, not a mistake to quietly drop.

**`experiments.md` — the log.** One entry per run, in order: setup (hardware, data, configuration),
the numbers, and what the run showed. Enough detail to re-run it. This is also where bugs that a run
surfaced get recorded — which run found them and what the fix was — because the same class of bug
tends to come back.

## Conventions

- **Measured numbers only.** Anything in `report.md` or `experiments.md` is something that was run.
  Estimates, expectations and extrapolations are labelled as such. The same rule as `BENCHMARKS.md`:
  do not change a number without re-running what produced it.
- **Record the controls, not just the headline.** A result is only as good as what it was compared
  against — baselines, ablations, and the conditions under which the comparison is fair (identical
  hyper-parameters, disjoint splits, the same code path).
- **Shortfalls get written up, not tuned away.** If a success criterion is missed, say so plainly and
  say why; if the honest outcome is a narrower claim than the plan set out to make, that is the
  outcome.
- **Status in one line**, at the top of the design document and in the table above: what exists, what
  is measured, what is published, and which branch it lives on.
