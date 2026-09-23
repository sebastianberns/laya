# Laya vision stream on ModernVBERT

**Status:** the inference path, offline tests, training script and benchmark script are implemented, and a CPU smoke run of training and the benchmark completes on real data. Still to do, and needing the user's go-ahead: the GPU training run, the benchmark numbers, and publishing the checkpoint. See *Implementation notes* at the end for where the code diverges from the design below.

## Context

Laya answers typed questions (`choice` / `score` / `noul`) about a text or JSON state in one encoder forward pass. The goal is to let images join the state.

Research found a vision tower for the ModernBERT architecture: **ModernVBERT** (arXiv:2510.01149, MIT, `ModernVBERT/modernvbert`). It combines:
- Ettin-150M, a bidirectional ModernBERT-architecture text encoder (hidden size 768);
- SigLIP2-base-16-512;
- a connector that pixel-shuffles each 512px tile down to 64 tokens and projects them into the text-embedding space;
- MLM modality alignment over 10B tokens.

Transformers ships it natively from **5.3.0** as `ModernVBertModel`. Its forward takes `input_ids` + `pixel_values` (plus `pixel_attention_mask`, or a precomputed `image_hidden_states`), replaces the `<image>` placeholder ids with projected vision features, and returns `last_hidden_state`. That is exactly the interface laya's decision head needs.

Flux VAE latents are out of scope: the user decided the idea is not promising.

The deliverable is a new, fourth checkpoint, `laya-vision`: ModernVBERT as the encoder with laya's `DecisionModel` head on top. It needs inference support in the package, a training script, and a validation benchmark. The existing text checkpoints stay unchanged, and so does every text-only code path.

## Design

**Sequence layout.** The image block goes into the state segment, *after* the options:

```
[CLS] <type> question: ins [SEP] [MASK] opt0 [MASK] opt1 … [SEP] <image block> state-text [SEP]
```

- Marker positions and the head/option budgets (`head_max_len`) are unaffected.
- The image block is the exact id sequence the ModernVBERT processor produces for one image (`<fake_token_around_image>`, the `<image>` × n tokens, any row/col or global tokens). It is never truncated. If it doesn't fit in `max_len - len(head)`, raise a `ValueError` (the same style as the existing "options exceed head_max_len" error).
- The state text takes whatever room is left.

**Encode once, reuse per question.** Laya runs one sequence per question. Images are preprocessed once per `system_one` call, and the vision tower plus connector run once (via the model's image-feature method, passed as `image_hidden_states`). The features are then repeated across all question rows, so N questions don't cost N vision passes.

**Token budget.** The checkpoint config sets `max_len: 1024`, `head_max_len: 256` (the same as multilingual/typed-decisions) and a `vision` block. The default processor setting disables image splitting (1 tile = 64 image tokens plus 3 wrapper tokens = 67), with a configurable `tiles_per_side` for documents (implemented under that name rather than `max_image_tiles`; see the notes). That leaves about 700 tokens for the state text.

**Head initialisation.** The head is new, but its shapes match `laya-multilingual` (mmBERT-base, d=768). The training script can warm-start the `head.*`, `type_emb.*`, `scorer.*` and `act_head.*` weights from it (the flag `--init-head multilingual`) or initialise them randomly, and the benchmark compares the two.

## Code changes

**`laya/common.py`:**
- `DecisionModel.__init__`: take `d` from `encoder.config.text_config.hidden_size` when a `text_config` exists, else from `encoder.config.hidden_size`.
- `DecisionModel.forward(..., **encoder_kwargs)`: pass `pixel_values` / `pixel_attention_mask` / `image_hidden_states` through to `self.encoder(...)`. The text-only call is unchanged.
- `build_sequence(..., prefix_ids=None)`: splice `prefix_ids` (the image block) in front of the state tokens, subtract it from `room`, and raise if it doesn't fit.
- `build_model`: `AutoModel.from_config` / `from_pretrained` already resolve to `ModernVBertModel` for this config, so only the `hidden_size` lookup needs changing. Confirm this during implementation.

**`laya/vision.py` (new, small):**
- `load_processor(model_dir)`: import `AutoProcessor` lazily, with a clear error if transformers is older than 5.3 or pillow is missing.
- `image_block(processor, images) -> (prefix_ids, pixel_values, pixel_attention_mask)`: call the processor on `"<image>" * len(images)` with the images, then strip CLS/SEP. Accept PIL images, paths, or bytes.

**`laya/agent.py`:**
- Loading: if `cfg.get("vision")`, load the processor from `processor/`, fall back to the Hub id, and use `processor.tokenizer` as `self.tok`. Add `processor/*` to `allow_patterns`. `_fix_tokenizer_config` stays tokenizer-only.
- Extend `_verify_compatibility` to require the connector/vision prefixes (`encoder.vision_model.`, `encoder.connector.`) whenever `cfg["vision"]` is set.
- `system_one(state, questions, images=None)`:
  - With `images` on a text-only checkpoint, raise `ValueError`.
  - On a vision checkpoint, build `prefix_ids` once, encode the image features once, and repeat them per row.
  - `images=None` on a vision checkpoint behaves exactly like today.
  - `predict = system_one` is kept.

**`laya/router.py`:**
- Add `"vision": (BUNDLE_REPO, "vision")` to `DEFAULT_MODELS` and `"laya-vision"` to `STANDALONE_MODELS`, plus an alias in `_ALIASES`.
- `route(..., images=None)` / `predict(..., images=None)`: images route to `vision` with reason `"state includes N image(s)"`, placed in the precedence order after explicit `model`/`task` and before `lang`/detection. An explicit `model` combined with images on a text checkpoint surfaces the agent's `ValueError`.
- Until the checkpoint is published, users can pass `models={"vision": "/path"}`.

**`pyproject.toml`:** add `[project.optional-dependencies] vision = ["transformers>=5.3.0", "pillow>=9"]`. The core floor stays as it is, so `tests/test_packaging.py` keeps passing.

**`laya/__init__.py`:** export `load_processor` / `image_block` only if that turns out to be useful. Otherwise leave the package surface alone.

**`AGENTS.md`:** document the vision checkpoint, the image-block layout, and the `[vision]` extra.

## Training: `research/scripts/train_vision.py`

- Port the RLCD loop from the notebook cell `train_ddp.py` (`notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb`) unchanged: the GRPO-style noisy logits, `proper_reward(w_sph=0.75, w_rps=1.0)`, the soft-CE guidance, and the two-LR AdamW.
- Differences from the notebook:
  - the encoder is `ModernVBERT/modernvbert`;
  - the SigLIP tower is frozen;
  - the LR for the connector and text encoder is 2.5e-5 and the head's is 1e-4;
  - items carry pixel tensors;
  - collation extends `collate_items` with stacked `pixel_values`.
- The script writes the standard layout: `rl_agent_config.json` (+ `vision` block, fitted `temperature` / `temperature_by_options` using the notebook's `fit_one_temp`), `model.safetensors`, `encoder/` config, `processor/`.
- **Data mix.** Everything is converted to typed questions with ≤20 options, and choice options are sampled per item so label order varies.
  - Vision:
    - `choice`: a CIFAR-100 / ImageNet-100 subset, RVL-CDIP (document type);
    - `noul`: VQAv2 yes/no;
    - `score`: KonIQ-10k quality in 5 bins.
  - Text-only replay: the `LocalLLaMA/typed-decisions` train split, so text decisions don't collapse.
  - Each dataset gets a small builder function in the script. The splits are fixed-seed and disjoint from the eval splits.

## Validation

1. **Unit tests: new `tests/test_vision.py`** (offline, `check()` style, added to both `ci.yml` and `release.yml`). It uses a tiny from-config `ModernVBertModel` (small SigLIP + small ModernBERT configs), following the pattern of `tests/test_decision_model.py`, plus a dummy image. It asserts that:
   - the image block comes after every marker, and the markers are identical to the text-only sequence;
   - the state budget shrinks by exactly `len(prefix_ids)`;
   - an oversized image block raises an error;
   - rows with the vision features computed once and repeated match rows with per-row vision to within 1e-5;
   - `system_one` without images on the tiny vision model returns valid probabilities;
   - `images=` on a text model raises an error;
   - `Router.route(images=[…])` picks `vision` and `model=` still overrides it.
2. **Regression:** the whole existing suite passes unchanged (`test_router`, `test_criteria`, `test_download`, `test_shortlist`, `test_decision_model`, `test_packaging`, `test_email`), plus `ruff check laya/ --select=E9,F63,F7,F82,F401,F811 --line-length=120` and `compileall`. Run it in a fresh `.venv` with `pip install -e .[vision]`.
3. **Benchmark: `research/scripts/bench_vision.py`** writes `research/results/vision_benchmark.json`. It covers the held-out splits of every training task plus one task never seen in training (a zero-shot probe, e.g. Oxford-IIIT Pets, 20 sampled breeds).
   - Metrics per task: accuracy, Brier, ECE before and after temperature fitting (reusing `ece_score`), and score MAE.
   - Baselines:
     - SigLIP2 zero-shot image–text similarity for `choice`/`noul` (the "no decision head" reference);
     - majority class and random;
     - head warm-start versus random init.
   - Text regression: `laya-vision` on the typed-decisions test split, compared with the published `laya-typed-decisions` 0.766 and the base checkpoints' 0.36.
   - Latency: p50 with 1 image × {1, 5, 10} questions on CPU (and on a T4 if one is available), compared against the text checkpoints' numbers in `README.md`.
4. **End-to-end smoke test:** `Router(models={"vision": out_dir}).predict({"note": "customer photo"}, questions, images=["damaged_parcel.jpg"])` returns a routing reason that mentions images, plus sensible answers.

**Success criteria (reported, not gamed):**
- `laya-vision` beats SigLIP2 zero-shot on the trained tasks;
- it stays calibrated after temperature fitting (ECE ≤ 0.1);
- it stays ≥ majority class on typed-decisions text;
- it adds < 2× the latency of text-only `laya-multilingual` for 1 image.

Any shortfall is written up in `BENCHMARKS.md`'s honest-limits style rather than tuned away.

## Out of scope / notes

- The Flux VAE path is dropped, per the user.
- ModernVBERT is English-first (the Ettin text side). Multilingual image+text decisions are a known limitation to note in docs; don't claim them.
- Training needs a GPU (a Kaggle 2×T4 like the existing notebook). Code, tests, and CPU smoke runs can be done locally. Actually running training and publishing a Hub checkpoint needs the user's go-ahead and credentials.

## Sources
- [ModernVBERT paper](https://arxiv.org/pdf/2510.01149) · [illuin-tech/modernvbert](https://github.com/illuin-tech/modernvbert) · [HF model card](https://huggingface.co/ModernVBERT/modernvbert)
- [transformers ModernVBert source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/modernvbert/modeling_modernvbert.py) · [ModernBERT `inputs_embeds`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/modernbert/modular_modernbert.py)
- [SigLIP 2](https://huggingface.co/blog/siglip2)

## Implementation notes (divergences from the design above)

- **Processor built from parts, PIL backend.** In transformers 5.17, `AutoProcessor`/`AutoImageProcessor` raise without torchvision for this model. `laya.vision.load_processor` therefore builds `Idefics3Processor(Idefics3ImageProcessorPil, tokenizer, image_seq_len)` directly (on 5.3 the class is `Idefics3ImageProcessor`). This adds no torchvision dependency, and training and inference resize identically. `image_seq_len` comes from `encoder.image_seq_len`, so it always matches the connector.
- **`tiles_per_side`, not `max_image_tiles`.** The Idefics3 processor splits by longest edge, so the natural knob is "tiles per side" (`size.longest_edge = 512 × n`, up to n×n tiles plus a global view). A cap on the total isn't expressible. It lives in the config's `vision` block.
- **Image-token scrubbing.** `build_sequence` gained `reserved_tokens`. On a vision checkpoint every Idefics3 image special token is scrubbed from text, like `[MASK]`, so a literal `<image>` in a state can't pose as a placeholder and break the feature merge.
- **Shared features.** `get_image_features(...).pooler_output` is `[tiles, 64, d]`, and `inputs_merger` consumes blocks in row order, so sharing one call's image features across all question rows is `feats.repeat(n_rows, 1, 1)`. Tested against per-row vision (max diff < 1e-5).
- **Router.** The `images` check sits after `model`/`task` and before workflow detection (no text checkpoint can see images). Aliases are `laya-vision`, `image` and `images`. `preload()` with no names skips `vision` unless it was passed in `models=`, because the checkpoint isn't on the Hub yet.
- **`build_model` confirmed.** `AutoModel.from_pretrained("ModernVBERT/modernvbert")` resolves to `ModernVBertModel` (the MLM-head keys are dropped as unexpected), and `from_config` does the same. Only the `hidden_size` lookup changed. Head warm-start from multilingual loads all 35 head tensors with no shape mismatch.
- **Data sources, resolved.** They live in `research/scripts/vision_data.py`:
  - CIFAR-100 `uoft-cs/cifar100`;
  - ImageNet-100 `clane9/imagenet-100` (opt-in via `--tasks`);
  - RVL-CDIP `chainyo/rvl-cdip` (parquet; splits train/val/test);
  - VQAv2 `lmms-lab/VQAv2`. Only `validation` carries answers, so the yes/no items use a crc32(image_id) 80/10/10 split and no image crosses splits. The soft target is the fraction of annotators answering yes;
  - KonIQ-10k from `chaofengc/IQA-PyTorch-Datasets` (a 6.3 GB tgz, extracted once) plus its metainfo CSV. The `score` target is the per-image vote distribution `c1..c5` rather than 5 MOS bins, on the official train/val/test split;
  - Pets `timm/oxford-iiit-pet` (test only, 20 seed-fixed breeds);
  - typed-decisions replay, with val carved as 10% of train by case id.
- **Calibration split.** Temperatures are fitted on each task's `val` split (600 per task), never on test. `temperature_by_options` buckets need ≥ 30 items.
- **Latency, measured on a Kaggle T4** (random/near-random weights, so timing only). 1 image adds 67 tokens per row, and the vision tower runs once per call: 197/195/198 ms for 1/5/10 questions before the preprocessing fix, i.e. flat, which is what the encode-once design is for.

  The first smoke run showed preprocessing, not the tower, dominating: 119 ms of 166 ms. The cause was the Idefics3 processor resizing to `size.longest_edge` (2048 in the ModernVBERT config) before squashing to one 512 tile. Capping the size in `to_pil` (a per-call `size` kwarg alone proved unreliable) gives, on the same T4:

  | | before | after |
  |---|---|---|
  | preprocess (CPU) | 125 ms | 24.7 ms |
  | vision tower + connector (GPU) | 64.7 ms | 62.5 ms |
  | `predict`, 1 image, 1 question | 188 ms | 74.7 ms |
  | `predict`, text only | 27 ms | 26.8 ms |

  **The "< 2× text-only" success criterion is therefore not going to be met for 1 image**: 74.7 ms against a 53.6 ms bar, with ~62 ms of it the irreducible SigLIP forward pass at 512px. Per question it looks much better, since the tower cost is amortised. Report it in `BENCHMARKS.md`'s honest-limits style, and/or restate the criterion per question; do not tune it away.
