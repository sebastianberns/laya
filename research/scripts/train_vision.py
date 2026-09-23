"""Train laya-vision: ModernVBERT encoder + laya's DecisionModel head, with RLCD.

The loop is the notebook's `train_ddp.py` (notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb)
unchanged -- GRPO-style noisy logits, proper_reward(w_sph=0.75, w_rps=1.0), soft-CE guidance,
two-LR AdamW, cosine schedule -- with these differences:

  * encoder: ModernVBERT/modernvbert (Ettin-150M + SigLIP2-base-16-512); the SigLIP tower is frozen
  * LRs: connector + text encoder 2.5e-5, head 1e-4
  * items carry images; the collate stacks their pixel tiles in row order
  * data: vision tasks + text-only typed-decisions replay (see vision_data.py)
  * head: random init, or warm-started from laya-multilingual (--init-head multilingual; same d=768)

Output is a standard laya checkpoint that `laya.Agent(out_dir)` loads:
rl_agent_config.json (+ `vision` block, fitted temperatures), model.safetensors, encoder/, processor/.

    # 2xT4 (Kaggle), as the notebook
    torchrun --standalone --nproc_per_node=2 research/scripts/train_vision.py --out /kaggle/working/laya-vision
    # CPU smoke run: a few examples per task, one short epoch
    python research/scripts/train_vision.py --out /tmp/lv --smoke --tasks cifar100,vqav2_yesno,typed
"""
import argparse
import datetime
import functools
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from laya.common import (  # noqa: E402
    QTYPE_NAMES, QTYPES, build_model, build_sequence, clamp_temperature, collate_items, ece_score,
    proper_reward, temp_bucket,
)
from laya.vision import IMAGE_TOKENS, image_block, load_processor  # noqa: E402
from vision_data import VISION_TASKS, build  # noqa: E402

HEAD_PREFIXES = ("head.", "type_emb.", "scorer.", "act_head.")
MULTILINGUAL = ("convaiinnovations/laya", "multilingual")

# examples per task and split; --smoke shrinks everything
DEFAULT_N = {"cifar100": 8000, "imagenet100": 8000, "rvl_cdip": 8000, "vqav2_yesno": 8000, "koniq": 7058,
             "typed": None}
VAL_N = 600


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--encoder", default="ModernVBERT/modernvbert")
    ap.add_argument("--init-head", choices=["random", "multilingual"], default="multilingual")
    ap.add_argument("--tasks", default=",".join(VISION_TASKS + ["typed"]))
    ap.add_argument("--n-per-task", type=int, default=None, help="cap examples per task (default: DEFAULT_N)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr-encoder", type=float, default=2.5e-5)
    ap.add_argument("--lr-head", type=float, default=1e-4)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--head-max-len", type=int, default=256)
    ap.add_argument("--tiles-per-side", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--data-cache", default=os.path.join(os.path.expanduser("~"), ".cache", "laya_vision", "examples"),
                    help="where built splits are cached; reused across runs")
    ap.add_argument("--prepare-only", action="store_true",
                    help="build and cache the data, then exit (run once before torchrun)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true", help="tiny data, one epoch, a few steps; checks the pipeline")
    return ap.parse_args()


# --------------------------------------------------------------------------- data
class Items(torch.utils.data.Dataset):
    """Examples -> tokenized items with pixel tiles, built lazily so images stay compressed in memory."""

    def __init__(self, examples, proc, max_len, head_max_len, tiles_per_side):
        self.examples, self.proc = examples, proc
        self.max_len, self.head_max_len, self.tiles = max_len, head_max_len, tiles_per_side

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return encode(self.examples[i], self.proc, self.max_len, self.head_max_len, self.tiles)


def encode(ex, proc, max_len, head_max_len, tiles_per_side=1):
    prefix, pv, pam = image_block(proc, ex["images"], tiles_per_side) if ex["images"] else (None, None, None)
    seq, markers = build_sequence(proc.tokenizer, ex["state"], ex["q"], max_len, head_max_len,
                                  prefix_ids=prefix, reserved_tokens=IMAGE_TOKENS)
    if len(markers) != len(ex["target"]):
        return None     # options overflowed head_max_len; the notebook drops these too
    target = ex["target"]
    return {"ids": seq, "markers": markers, "qtype": QTYPES[ex["q"]["t"]], "target": target,
            "label": max(range(len(target)), key=target.__getitem__), "task": ex["task"],
            "pixel_values": None if pv is None else pv[0], "pixel_mask": None if pam is None else pam[0]}


def collate(items, pad_id):
    """collate_items + the batch's image tiles, stacked in row order as [1, tiles, 3, H, W].

    ModernVBERT assigns image-feature blocks to `<image>` runs in row-major order, so rows
    without images simply contribute no tiles.
    """
    items = [it for it in items if it is not None]
    if not items:
        return None
    b = collate_items([[{k: v for k, v in it.items() if not k.startswith("pixel")} for it in items]], pad_id)
    tiles = [it["pixel_values"] for it in items if it["pixel_values"] is not None]
    if tiles:
        b["pixel_values"] = torch.cat(tiles)[None]
        masks = [it["pixel_mask"] for it in items if it["pixel_values"] is not None]
        b["pixel_attention_mask"] = torch.cat(masks)[None] if all(m is not None for m in masks) else None
    return b


def cache_file(cache_dir, task, split, n, seed):
    return os.path.join(cache_dir, "%s-%s-n%s-s%d.pt" % (task, split, n, seed))


def build_cache(tasks, split, n_for, seed, cache_dir, log):
    """Build any split not already cached. One process only -- see `main`."""
    os.makedirs(cache_dir, exist_ok=True)
    for t in tasks:
        path = cache_file(cache_dir, t, split, n_for(t), seed)
        if os.path.exists(path):
            log("  %-12s %-5s cached" % (t, split))
            continue
        t0 = time.time()
        ex = build(t, split, n_for(t), seed)
        tmp = path + ".tmp"                     # never leave a half-written cache behind
        torch.save(ex, tmp)
        os.replace(tmp, path)
        log("  %-12s %-5s %6d examples (%.0fs)" % (t, split, len(ex), time.time() - t0))


def load_examples(tasks, split, n_for, seed, cache_dir, log):
    out = []
    for t in tasks:
        path = cache_file(cache_dir, t, split, n_for(t), seed)
        if not os.path.exists(path):            # single-process run, or a cache that was cleared
            build_cache([t], split, n_for, seed, cache_dir, log)
        ex = torch.load(path, weights_only=False)
        log("  %-12s %-5s %6d examples" % (t, split, len(ex)))
        out.extend(ex)
    return out


# --------------------------------------------------------------------------- model
def warm_start_head(model, log):
    from huggingface_hub import hf_hub_download

    repo, sub = MULTILINGUAL
    sd = load_file(hf_hub_download(repo, "%s/model.safetensors" % sub))
    head = {k: v for k, v in sd.items() if k.startswith(HEAD_PREFIXES)}
    own = model.state_dict()
    bad = [k for k, v in head.items() if k not in own or own[k].shape != v.shape]
    if bad:
        raise ValueError("multilingual head does not fit: %s" % bad[:5])
    missing, _ = model.load_state_dict(head, strict=False)
    left = [k for k in missing if k.startswith(HEAD_PREFIXES)]
    note = " (still random: %s)" % left if left else ""
    log("warm-started %d head tensors from %s/%s%s" % (len(head), repo, sub, note))


def fit_one_temp(sel):
    """The notebook's temperature fit: one scalar per bucket by LBFGS on soft-target NLL."""
    if len(sel) < 10:
        return 1.0
    kmax = max(len(z) for z, _ in sel)
    Z = torch.full((len(sel), kmax), -1e4)
    T = torch.zeros((len(sel), kmax))
    for i, (z, t) in enumerate(sel):
        Z[i, :len(z)] = torch.tensor(z)
        T[i, :len(t)] = torch.tensor(t, dtype=torch.float32)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = -(T * torch.log_softmax(Z / log_t.exp(), -1)).sum(-1).mean()
        loss.backward()
        return loss
    opt.step(closure)
    return float(torch.clamp(log_t.exp(), 0.1, 10.0).item())


def forward(model, b, device):
    return model(b["input_ids"].to(device), b["attention_mask"].to(device), b["marker_pos"].to(device),
                 b["marker_mask"].to(device), b["qtype"].to(device),
                 pixel_values=b["pixel_values"].to(device) if "pixel_values" in b else None,
                 pixel_attention_mask=b["pixel_attention_mask"].to(device)
                 if b.get("pixel_attention_mask") is not None else None)


def _ece_at(sel, t):
    """Val ECE if this group were served at temperature `t`."""
    conf, correct = [], []
    for z, target in sel:
        z = np.asarray(z, dtype=np.float64) / t
        p = np.exp(z - z.max())
        p = p / p.sum()
        conf.append(float(p.max()))
        correct.append(float(np.argmax(p) == np.argmax(target)))
    return ece_score(np.array(conf), np.array(correct))


def _keep_temp(sel, fitted, label, log):
    """Ship a fitted temperature only when it calibrates better than leaving it alone.

    A bucket pools every task with the same (type, option count): laya-vision's `choice:11+`
    holds CIFAR-100, RVL-CDIP and typed-decisions at once. One scalar cannot serve all three --
    fitting to the pooled NLL flattened a task that was already well calibrated and tripled its
    test ECE -- so a temperature that does not improve val ECE is refused.

    The check runs on the *clamped* value, because that is what inference applies: laya confines
    temperatures to [0.5, 5] at load, so judging an unclamped fit would score a setting that never
    actually runs.
    """
    fitted = clamp_temperature(fitted)
    base, tuned = _ece_at(sel, 1.0), _ece_at(sel, fitted)
    if tuned <= base:
        log("    %-12s T=%.3f  val ECE %.4f -> %.4f" % (label, fitted, base, tuned))
        return round(fitted, 4)
    log("    %-12s T=%.3f REFUSED (val ECE %.4f -> %.4f); shipping T=1" % (label, fitted, base, tuned))
    return 1.0


def calibrate(model, loader, device, amp, log):
    """Per-type and per-bucket temperatures on the held-out val split (never the test split)."""
    model.eval()
    preds = []
    with torch.no_grad():
        for b in loader:
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                logits, _ = forward(model, b, device)
            z = logits.float().cpu()
            for r in range(z.size(0)):
                k = int(b["marker_mask"][r].sum())
                preds.append((int(b["qtype"][r]), z[r, :k].tolist(), b["target"][r, :k].tolist()))
    temps = []
    for qt in range(3):
        sel = [(z, t) for q, z, t in preds if q == qt]
        temps.append(_keep_temp(sel, fit_one_temp(sel), QTYPE_NAMES[qt], log) if sel else 1.0)
    by_bucket = {}
    for key in sorted({temp_bucket(q, len(z)) for q, z, _ in preds}):
        sel = [(z, t) for q, z, t in preds if temp_bucket(q, len(z)) == key]
        if len(sel) >= 30:
            kept = _keep_temp(sel, fit_one_temp(sel), key, log)
            if kept != 1.0:
                by_bucket[key] = kept
    log("fitted temperatures (choice, score, noul): %s | by bucket: %s" % ([round(t, 3) for t in temps], by_bucket))
    return [round(t, 4) for t in temps], by_bucket, len(preds)


def save_checkpoint(out, model, proc, cfg):
    os.makedirs(out, exist_ok=True)
    sd = {k: v.half().contiguous().cpu() for k, v in model.state_dict().items()}
    save_file(sd, os.path.join(out, "model.safetensors"))
    model.encoder.config.save_pretrained(os.path.join(out, "encoder"))
    proc.save_pretrained(os.path.join(out, "processor"))
    # also as a standalone preprocessor_config.json, for transformers versions that read that file
    proc.image_processor.save_pretrained(os.path.join(out, "processor"))
    with open(os.path.join(out, "rl_agent_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)


# --------------------------------------------------------------------------- main
def main():
    a = parse_args()
    ddp = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if ddp:
        local = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local)
        device = torch.device("cuda", local)
        # Rank 0 builds the data while the others wait at a barrier; the default 30-minute
        # collective timeout would kill them mid-build on a slow streaming dataset.
        kw = {"timeout": datetime.timedelta(hours=4)}
        try:
            dist.init_process_group("nccl", device_id=device, **kw)
        except TypeError:           # older torch: no device_id, barrier() just warns
            dist.init_process_group("nccl", **kw)
        rank, world = dist.get_rank(), dist.get_world_size()
    else:
        rank, world = 0, 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"

    def log(*msg):
        if rank == 0:
            print(*msg, flush=True)

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    tasks = [t.strip() for t in a.tasks.split(",") if t.strip()]
    epochs = 1 if a.smoke else a.epochs

    def n_for(t):
        if a.smoke:
            return 12
        return a.n_per_task if a.n_per_task is not None else DEFAULT_N.get(t)

    val_n = (lambda t: 12) if a.smoke else (lambda t: VAL_N)
    # Data first, before any weights are downloaded: --prepare-only is then a CPU-only job, and a
    # data problem surfaces before a 1.8 GB download. Building a split streams images, so doing it
    # on every rank duplicates the cost and saturates the CPU while the GPUs idle: rank 0 builds,
    # the others wait, all load from the cache.
    log("preparing data for %s (cache: %s)" % (tasks, a.data_cache))
    if rank == 0:
        build_cache(tasks, "train", n_for, a.seed, a.data_cache, log)
        build_cache(tasks, "val", val_n, a.seed, a.data_cache, log)
    if ddp:
        dist.barrier()
    if a.prepare_only:
        log("data prepared; exiting (--prepare-only)")
        if ddp:
            dist.destroy_process_group()
        return

    cfg = {
        "encoder": a.encoder, "head_layers": 2, "max_len": a.max_len, "head_max_len": a.head_max_len,
        "act_costs": {"escalate": 0.5}, "amp_dtype": "fp16", "model_name": "laya-vision",
        "vision": {"tiles_per_side": a.tiles_per_side, "tasks": tasks},
    }
    model = build_model(cfg)                      # from_pretrained: ModernVBertModel
    proc = load_processor(a.encoder, image_seq_len=model.encoder.image_seq_len)
    cfg["vision"]["image_seq_len"] = model.encoder.image_seq_len
    if a.init_head == "multilingual":
        warm_start_head(model, log)
    for p in model.encoder.vision_model.parameters():
        p.requires_grad_(False)
    try:
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except Exception as e:                       # not fatal: only costs memory
        log("gradient checkpointing unavailable: %s" % e)
    model.to(device).train()
    # find_unused_parameters: a micro-batch of text-only replay rows never touches the connector,
    # so its parameters get no gradient in that step. Without the flag DDP raises on such a batch.
    net = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device.index], find_unused_parameters=True) \
        if ddp else model

    train_ex = load_examples(tasks, "train", n_for, a.seed, a.data_cache, log)
    val_ex = load_examples(tasks, "val", val_n, a.seed, a.data_cache, log)
    random.Random(a.seed).shuffle(train_ex)
    # Every rank must run the same number of batches: an uneven shard leaves one rank waiting
    # for an all-reduce that never comes. Drop the remainder (at most world_size-1 items).
    mine = train_ex[rank::world][:len(train_ex) // world]

    pad = proc.tokenizer.pad_token_id
    loader_kw = {"collate_fn": functools.partial(collate, pad_id=pad), "num_workers": 0 if a.smoke else a.workers}
    train_ds = Items(mine, proc, a.max_len, a.head_max_len, a.tiles_per_side)

    enc_params = [p for n, p in model.named_parameters() if n.startswith("encoder.") and p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    opt = torch.optim.AdamW([{"params": enc_params, "lr": a.lr_encoder}, {"params": head_params, "lr": a.lr_head}],
                            weight_decay=0.01)
    batches_per_epoch = math.ceil(len(mine) / a.micro_batch)          # DataLoader keeps the last short batch
    total_updates = max(1, math.ceil(batches_per_epoch / a.grad_accum) * epochs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_updates, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    GROUP_SIZE, SIGMA_START, SIGMA_END = 4, 0.4, 0.1

    log("training: %d items (%d per rank) | %d epochs | %d updates | init-head=%s | frozen SigLIP"
        % (len(train_ex), len(mine), epochs, total_updates, a.init_head))
    t0 = time.time()
    for epoch in range(epochs):
        g = torch.Generator().manual_seed(a.seed + epoch + rank)
        loader = torch.utils.data.DataLoader(train_ds, batch_size=a.micro_batch, shuffle=True, generator=g,
                                             **loader_kw)
        sigma = SIGMA_START + (SIGMA_END - SIGMA_START) * (epoch / max(1, epochs - 1))
        opt.zero_grad(set_to_none=True)
        run_loss, n_b = 0.0, 0
        for step, b in enumerate(loader):
            if b is None:
                continue
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                logits, act = forward(net, b, device)
            logits = logits.float()
            mask = b["marker_mask"].to(device)
            k = mask.sum(-1, keepdim=True).float()
            target = b["target"].to(device)

            # 1. G noisy logit samples, zero-mean over each row's options
            eps = torch.randn((GROUP_SIZE,) + logits.shape, device=device) * sigma * mask
            eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
            z = logits.detach().unsqueeze(0) + eps
            q = torch.softmax(z.masked_fill(~mask, -1e4), -1)
            # 2. proper-scoring-rule reward, group-normalised advantage
            with torch.no_grad():
                r = proper_reward(q, target.unsqueeze(0), b["qtype"].to(device), mask, w_sph=0.75, w_rps=1.0)
                adv = (r - r.mean(0, keepdim=True))
                adv = adv / (adv.std() + 1e-6)
            # 3. policy gradient + soft cross-entropy guidance
            logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
            loss_rl = -(adv * logp).mean()
            loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
            loss = (loss_rl + loss_ce) / a.grad_accum + 0.0 * act.sum()
            scaler.scale(loss).backward()

            if (step + 1) % a.grad_accum == 0 or step + 1 == len(loader):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad(set_to_none=True)
            run_loss += loss.item() * a.grad_accum
            n_b += 1
            if n_b % 50 == 0 or a.smoke:
                log("  epoch %d/%d step %d | loss %.4f | reward %.3f | lr %.2e | %.0fs"
                    % (epoch + 1, epochs, n_b, loss.item() * a.grad_accum, r.mean().item(), sched.get_last_lr()[0],
                       time.time() - t0))
            if a.smoke and n_b >= 3:
                break
        log("=== epoch %d/%d done | avg loss %.4f | %.0fs"
            % (epoch + 1, epochs, run_loss / max(1, n_b), time.time() - t0))
        if ddp:
            dist.barrier()
        if rank == 0:
            # rolling checkpoint, so a Kaggle timeout does not lose finished epochs
            save_checkpoint(os.path.join(a.out, "checkpoint_latest"), model, proc,
                            dict(cfg, checkpoint={"epoch": epoch + 1, "avg_loss": run_loss / max(1, n_b)}))

    if rank == 0:
        val_ds = Items(val_ex, proc, a.max_len, a.head_max_len, a.tiles_per_side)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=16, **loader_kw)
        temps, by_bucket, n_cal = calibrate(model, val_loader, device, amp, log)
        cfg.update(temperature=temps, temperature_by_options=by_bucket, fine_tuned=True, init_head=a.init_head,
                   training={"epochs": epochs, "items": len(train_ex), "calibration_items": n_cal,
                             "world_size": world, "hours": round((time.time() - t0) / 3600, 3), "smoke": a.smoke})
        save_checkpoint(a.out, model, proc, cfg)
        log("saved laya-vision checkpoint to %s" % a.out)
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
