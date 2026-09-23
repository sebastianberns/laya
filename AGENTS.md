# AGENTS.md

This file provides guidance to coding agents (Claude Code, Codex, etc.) when working with code in this repository.

## What this is

`laya` is a PyPI package: a non-autoregressive "System 1" decision engine. It answers typed questions (`choice`, `score`, `noul`) about an arbitrary state (string, dict, or list) in **one encoder forward pass**, with no text generation. It is torch-only inference code; training happens in `notebooks/` (RLCD: proper-scoring-rule rewards, GRPO-style policy gradient) and weights live on the Hugging Face Hub.

## Commands

Set these env vars before anything that imports `transformers`. When TensorFlow is installed, `transformers` probes it at import, and TF's abseil runtime can deadlock model construction:

```bash
export USE_TF=0 USE_TORCH=1 TOKENIZERS_PARALLELISM=false
```

```bash
pip install -e ".[vision]"            # CI installs CPU torch first: pip install torch --index-url https://download.pytorch.org/whl/cpu

# Tests are plain scripts, not pytest. Each exits non-zero on failure. Run one at a time:
python tests/test_router.py           # routing + language detection (pure, no weights)
python tests/test_criteria.py
python tests/test_download.py         # unittest; mocks snapshot_download
python tests/test_shortlist.py
python tests/test_decision_model.py   # tiny from-config BERT, offline
python tests/test_packaging.py        # pyproject metadata vs. dependency floors
python tests/test_email.py
python tests/test_vision.py           # tiny from-config ModernVBERT, offline; needs .[vision], else it skips

# Needs real checkpoints on disk (not run in CI). Default root is ~/laya_models/{laya,laya-multilingual,laya-typed-decisions}
python tests/test_local_e2e.py [model_root]    # LAYA_DEVICE=cpu|cuda|mps

# Lint, as CI runs it (a narrow error-only rule set, not full style)
ruff check laya/ --select=E9,F63,F7,F82,F401,F811 --line-length=120
python -m compileall -q laya/ tests/

# Package
python -m build && python -m twine check dist/*
```

Most test files use a hand-rolled `check(name, got, want)` helper that collects PASS/FAIL lists and calls `sys.exit(1 if FAIL else 0)` at the end. Follow that pattern when adding tests. A new test file also needs to be added to **both** `.github/workflows/ci.yml` and `release.yml`, because each lists the test scripts explicitly.

## Releasing

The version is duplicated in `pyproject.toml` and `laya/__init__.py` (`__version__`), so bump both. Pushing a `v*` tag triggers `release.yml`, which runs the tests, checks that the pyproject version matches the tag, publishes to PyPI via trusted publishing, and creates a GitHub Release.

## Architecture

**Inference path** (`laya/agent.py` → `laya/common.py`):
- `Agent.__init__` resolves a local dir or a Hub repo (with an optional `subfolder`). Hub downloads use `allow_patterns` so that only the requested checkpoint's files are fetched: `rl_agent_config.json`, `model.safetensors`, `tokenizer/*`, `encoder/*`. The encoder is built **from config** out of `encoder/`, and all weights come from `model.safetensors`, loaded with `strict=True` after `_verify_compatibility`. `_fix_tokenizer_config` rewrites the shipped tokenizer config in place to work around incompatibilities across transformers versions.
- `build_sequence` produces one sequence per question: `[CLS] "<type> question: <instructions>" [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]`. The `[MASK]` positions are the answer markers. Options share a fixed `head_max_len` token budget, and the state gets `max_len - head_max_len`. With many options, each one is truncated to `(head_max_len-16)//n` tokens, which is why high-cardinality `choice` degrades (see `shortlist.py`).
- `DecisionModel.forward`: encoder hidden states + a type embedding → a small `TransformerEncoder` head → gather the hidden state at each marker → a scalar logit per option. It also returns an `act_head` output (the escalate/act probability), fed by entropy/top-2 features.
- `system_one` (aliased as `predict`) batches all questions into one forward pass, then applies a temperature per bucket. The lookup order is `temperature_by_options[temp_bucket(qtype, k)]`, falling back to `temperature[qtype]`. All temperatures pass through `clamp_temperature` first, because some shipped buckets are fitted to sharpen rather than soften. `noul` always renders as the two options `[false, true]`, and its answer is `p[true]`.
- `common.py` also contains training-side helpers that are exported publicly: `proper_reward`, `td_lambda_targets`, `ece_score`.

**Vision** (`laya/vision.py`, the `laya-vision` checkpoint; `pip install "laya[vision]"` = transformers>=5.3 + pillow):
- The encoder is ModernVBERT (`ModernVBertModel`: Ettin-150M text + SigLIP2 + a pixel-shuffle connector). A checkpoint is a vision one when `rl_agent_config.json` has a `vision` block (`tiles_per_side`, default 1). It ships `processor/` in place of `tokenizer/`, and `self.tok` is `processor.tokenizer`. `_verify_compatibility` additionally requires `encoder.vision_model.*` / `encoder.connector.*`.
- `load_processor` builds the `Idefics3Processor` from its parts with the **PIL** image processor, because `AutoProcessor` demands torchvision in recent transformers. Training and inference must use the same resize path.
- `system_one(state, questions, images=[...])`: `image_block` preprocesses once into a placeholder block (`<fake_token_around_image><global-img><image>×64<fake_token_around_image>` per image, 67 tokens without splitting). `build_sequence(prefix_ids=...)` opens the state segment with that block, *after* every marker, so markers and the `head_max_len` budget are unchanged. The block is never truncated and raises if it doesn't fit. The vision tower runs once per call via `encoder.get_image_features` and its features are `repeat`ed per question row (ModernVBERT assigns feature blocks to `<image>` runs in row order). Image tokens are scrubbed from all text on a vision checkpoint (`reserved_tokens=IMAGE_TOKENS`). `images` on a text checkpoint raises `ValueError`.
- ModernVBERT is English-first: don't claim multilingual image+text decisions.
- Training / eval: `research/scripts/train_vision.py` (the notebook's RLCD loop, frozen SigLIP, optional head warm-start from multilingual), `bench_vision.py` → `research/results/vision_benchmark.json`, with the shared splits in `vision_data.py`. `notebooks/laya_vision_train_2xT4_kaggle.ipynb` drives all of it on Kaggle (smoke test first, then the real run); it clones the branch because the vision code is not on PyPI.

**Routing** (`laya/router.py` + `laya/lang.py`):
- `Router.route()` is pure: no weights, no I/O. The precedence order is: explicit `model` > explicit `task` > images present (→ `vision`) > typed-decisions workflow match (only with `auto_task_detection=True`, and only on an exact match of the question-id set) > explicit `lang` > script/language detection > `default`.
- `lang.analyse` is pure-Python script detection plus Latin-script stopword and diacritic heuristics. The rule is that a Latin-script text whose language can't be identified is **never** assumed to be English. The English checkpoint collapses off-English while staying confident, so the routing decision has to happen before the forward pass.
- `Router.load` caches agents in an LRU (`max_loaded`, default 1) under an `RLock`. Inference is deliberately kept outside the lock. `DEFAULT_MODELS` points all four checkpoints at subfolders of the bundle repo `convaiinnovations/laya` (root = english, `multilingual/`, `typed-decisions/`, `vision/`). `standalone_repos=True` uses the per-model repos instead. `laya-vision` is not published yet, so use `Router(models={"vision": "/path"})`. `preload()` with no names builds only the text checkpoints, plus `vision` when it was passed in `models=`.

**Other modules:** `shortlist.py` pre-ranks large `choice` label sets with a caller-supplied `embed_fn` (or `embed_fn_from_agent`, which mean-pools the loaded encoder) and then runs one `predict` on the top `k` labels. `presets.py` and `email.py` contain ready-made question schemas and email-cleaning helpers. Everything public is re-exported from `laya/__init__.py` and listed in `__all__`.

## Repository layout notes

- `research/` contains benchmark harnesses and result JSONs, and nothing in it is imported by `laya`. Edit `research/scripts/build_benchmark_nb.py`, not the generated `.ipynb`.
- The numbers in `README.md` and `BENCHMARKS.md` are measured results. Don't change them without re-running the benchmarks. Jev figures are third-party published numbers and were never measured here.
- `setuptools` is pinned to `packages = ["laya"]` because the root-level `assets/`, `research/`, and `notebooks/` directories break auto-discovery.

## Plans

**[`plans/README.md`](plans/README.md) is the entry point**: the index of initiatives, the folder
layout, and what belongs in a design, a report and an experiment log. One numbered directory per
initiative, `plans/<n>-<topic>/`.

When you pick up work on an initiative, read its `report.md` first if it has one — it says what is
true now — then the design. If the implementation diverges from the design, update the design.

- **`plans/1-ModernVBERT/`: vision stream, `laya-vision` (implemented and trained once, not
  published; branch `sb/vision`).** Architecture is under *Vision* above; measured outcomes are in
  [`report.md`](plans/1-ModernVBERT/report.md). In short: the image path works and text decisions did
  not collapse, but the checkpoint loses to SigLIP2 zero-shot on CIFAR-100 and RVL-CDIP, shows no
  zero-shot transfer, and the head warm-start is refuted (random init wins on every task). Unfreezing
  the frozen SigLIP tower is the untried lever. Flux VAE latents were considered and rejected.
- **`plans/2-SigLIP-tuning/`: unfreeze the SigLIP tower (planned, nothing run; branch `sb/vision`).**
  Follows directly from initiative 1's open question — undertrained, or a structural limit of the
  64-token connector? Adds `--vision-lr` to `research/scripts/train_vision.py` (default 0 = frozen,
  today's behaviour), holds data/seed/epochs/batch fixed so any gain is attributable, and carries a
  **pre-registered decision rule**: CIFAR-100 ≥ 0.80 and RVL-CDIP ≥ 0.40 means undertrained, neither
  moving by ≥ +0.03 means structural and the line stops. It also owns the four pre-publication fixes
  from initiative 1's report, starting with defaulting `--init-head` to `random`.
