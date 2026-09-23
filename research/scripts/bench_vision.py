"""Benchmark laya-vision on held-out splits, against SigLIP2 zero-shot and trivial baselines.

Writes research/results/vision_benchmark.json. Per task and model: accuracy, Brier, ECE before and
after the checkpoint's fitted temperatures (laya.ece_score), and MAE for `score`. Tasks are the
test splits of every training task plus `pets`, a 20-breed zero-shot probe never seen in training,
and `typed`, the typed-decisions text test split (the text-regression check).

    python research/scripts/bench_vision.py --model warm=/ckpt/laya-vision --model random=/ckpt/laya-vision-rand
    python research/scripts/bench_vision.py --model warm=/ckpt/laya-vision --n 50 --tasks cifar100,pets --no-siglip

Metrics are reported as measured; the success criteria at the end are checks, not targets to tune.
"""
import argparse
import json
import os
import sys
import time
from collections import Counter

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from laya import Agent, ece_score  # noqa: E402
from laya.common import QTYPES, build_sequence, collate_items, temp_bucket  # noqa: E402
from laya.vision import IMAGE_TOKENS, image_block, to_pil  # noqa: E402
from vision_data import VISION_TASKS, build  # noqa: E402

# Published reference points (README / BENCHMARKS.md), not re-measured here.
TYPED_PUBLISHED = {"laya-typed-decisions": 0.766, "base checkpoints": 0.36}
TRAINED_TASKS = VISION_TASKS + ["imagenet100"]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True, help="name=path of a laya-vision checkpoint")
    ap.add_argument("--tasks", default=",".join(VISION_TASKS + ["pets", "typed"]))
    ap.add_argument("--n", type=int, default=500, help="test examples per task")
    ap.add_argument("--device", default=None)
    ap.add_argument("--siglip", default="google/siglip2-base-patch16-512")
    ap.add_argument("--no-siglip", action="store_true")
    ap.add_argument("--latency-baseline", default="convaiinnovations/laya:multilingual",
                    help="text checkpoint (repo[:subfolder]) timed on the same state; '' to skip")
    ap.add_argument("--out", default=os.path.join(ROOT, "research", "results", "vision_benchmark.json"))
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


# --------------------------------------------------------------------------- metrics
def metrics(rows):
    """rows: dicts with p (after temps), p_raw (T=1), target and qtype. Accuracy is vs argmax(target).

    A task can mix question types (typed-decisions does), and Brier over 2 options is not
    comparable with Brier over 20, so a mixed task also reports each type separately.
    """
    if not rows:
        return {}
    gold = np.array([int(np.argmax(r["target"])) for r in rows])
    out = {"n": len(rows)}
    for key, name in (("p_raw", "before_temp"), ("p", "after_temp")):
        P = [np.asarray(r[key]) for r in rows]
        pred = np.array([int(np.argmax(p)) for p in P])
        conf = np.array([float(p.max()) for p in P])
        correct = (pred == gold).astype(float)
        brier = float(np.mean([((p - np.asarray(r["target"])) ** 2).sum() for p, r in zip(P, rows)]))
        out[name] = {"accuracy": round(float(correct.mean()), 4), "brier": round(brier, 4),
                     "ece": round(ece_score(conf, correct), 4)}
    out["accuracy"] = out["after_temp"]["accuracy"]
    score_rows = [r for r in rows if r["qtype"] == "score"]
    if score_rows:
        exp_p = [float((np.arange(len(r["p"])) * np.asarray(r["p"])).sum()) for r in score_rows]
        exp_t = [float((np.arange(len(r["target"])) * np.asarray(r["target"])).sum()) for r in score_rows]
        out["mae"] = round(float(np.mean(np.abs(np.array(exp_p) - np.array(exp_t)))), 4)
    types = sorted({r["qtype"] for r in rows})
    out["types"] = types
    if len(types) > 1:
        out["by_type"] = {t: metrics([r for r in rows if r["qtype"] == t]) for t in types}
    return out


def trivial_baselines(examples):
    """Majority class (most common gold label in the split) and uniform random, analytically.

    Reported per question type as well when the task mixes them.
    """
    types = sorted({e["q"]["t"] for e in examples})
    if len(types) > 1:
        out = {t: trivial_baselines([e for e in examples if e["q"]["t"] == t]) for t in types}
        gold_all = [int(np.argmax(e["target"])) for e in examples]
        ks_all = [len(e["target"]) for e in examples]
        out["overall"] = {"random": {"accuracy": round(float(np.mean([1.0 / k for k in ks_all])), 4)},
                          "n": len(gold_all)}
        return out
    gold = [int(np.argmax(e["target"])) for e in examples]
    ks = [len(e["target"]) for e in examples]
    # choice options are shuffled per item, so "majority" means the majority *label name*
    names = [list(e["q"]["crit"])[g] if e["q"]["t"] == "choice" else g for e, g in zip(examples, gold)]
    top = Counter(names).most_common(1)[0][0]
    maj_acc = float(np.mean([n == top for n in names]))
    rand_acc = float(np.mean([1.0 / k for k in ks]))
    rand_brier = float(np.mean([((np.full(k, 1.0 / k) - np.asarray(e["target"])) ** 2).sum()
                                for k, e in zip(ks, examples)]))
    return {"majority": {"accuracy": round(maj_acc, 4), "label": str(top)},
            "random": {"accuracy": round(rand_acc, 4), "brier": round(rand_brier, 4)}}


# --------------------------------------------------------------------------- laya
@torch.no_grad()
def laya_logits(agent, ex):
    """Raw logits for one example, exactly as Agent.system_one builds and runs it (no rounding)."""
    prefix = pv = pam = None
    if ex["images"]:
        prefix, pv, pam = image_block(agent.processor, ex["images"], int(agent.vision.get("tiles_per_side", 1)))
    seq, markers = build_sequence(agent.tok, ex["state"], ex["q"], agent.cfg.get("max_len", 512),
                                  agent.cfg.get("head_max_len", 192), prefix_ids=prefix, reserved_tokens=IMAGE_TOKENS)
    b = collate_items([[{"ids": seq, "markers": markers, "qtype": QTYPES[ex["q"]["t"]]}]], agent.tok.pad_token_id)
    kw = {}
    if pv is not None:
        kw["pixel_values"] = pv.to(agent.device)
        kw["pixel_attention_mask"] = pam.to(agent.device) if pam is not None else None
    with torch.autocast(agent.device.type, dtype=agent.dtype, enabled=agent.device.type == "cuda"):
        logits, _ = agent.model(b["input_ids"].to(agent.device), b["attention_mask"].to(agent.device),
                                b["marker_pos"].to(agent.device), b["marker_mask"].to(agent.device),
                                b["qtype"].to(agent.device), **kw)
    return logits[0, :len(markers)].float().cpu().numpy()


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def eval_laya(agent, examples):
    rows = []
    for ex in examples:
        z = laya_logits(agent, ex)
        qt, k = QTYPES[ex["q"]["t"]], len(z)
        t = agent.temperature_by_options.get(temp_bucket(qt, k), agent.temperature[qt])
        rows.append({"p_raw": softmax(z), "p": softmax(z / t), "target": ex["target"], "qtype": ex["q"]["t"]})
    return metrics(rows)


def eval_typed(agent, examples):
    """Text regression on typed-decisions: accuracy by question, as the notebook scores it."""
    m = eval_laya(agent, examples)
    m["published_reference"] = TYPED_PUBLISHED
    return m


# --------------------------------------------------------------------------- SigLIP2 zero-shot
class SigLIP:
    """Image-text similarity as the 'no decision head' reference for choice / noul / score."""

    def __init__(self, repo, device):
        from transformers import AutoModel, AutoTokenizer
        try:
            from transformers import AutoProcessor
            self.proc = AutoProcessor.from_pretrained(repo)
            self.img = self.proc.image_processor
            self.tok = self.proc.tokenizer
        except ImportError:        # no torchvision: build the PIL image processor directly
            from transformers.models.siglip.image_processing_pil_siglip import SiglipImageProcessorPil
            self.img = SiglipImageProcessorPil.from_pretrained(repo)
            self.tok = AutoTokenizer.from_pretrained(repo)
        self.model = AutoModel.from_pretrained(repo).to(device).eval()
        self.device = device

    @torch.no_grad()
    def probs(self, image, texts):
        px = self.img(images=[to_pil(image)], return_tensors="pt")["pixel_values"].to(self.device)
        tx = self.tok(texts, padding="max_length", max_length=64, truncation=True, return_tensors="pt").to(self.device)
        out = self.model(pixel_values=px, input_ids=tx["input_ids"])
        return torch.softmax(out.logits_per_image[0].float(), -1).cpu().numpy()


def siglip_prompts(ex):
    q = ex["q"]
    if q["t"] == "choice":
        return ["a photo of a %s." % k for k in q["crit"]]
    if q["t"] == "noul":
        return ["%s No." % q["ins"], "%s Yes." % q["ins"]]
    return ["a photo of %s quality." % c.split(":")[0] for c in q["crit"]]


def eval_siglip(sig, examples):
    rows = []
    for ex in examples:
        p = sig.probs(ex["images"][0], siglip_prompts(ex))
        rows.append({"p_raw": p, "p": p, "target": ex["target"], "qtype": ex["q"]["t"]})
    m = metrics(rows)
    m.pop("before_temp", None)
    return m


# --------------------------------------------------------------------------- latency
def p50(fn, reps=7):
    fn()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return round(1000 * float(np.median(ts)), 1)


def latency(agent, image, baseline=None):
    q = {"type": "choice", "instructions": "What is shown?", "criteria": ["parcel", "invoice", "person", "vehicle"]}
    state = {"note": "customer photo"}
    out = {}
    for n in (1, 5, 10):
        qs = {"q%d" % i: q for i in range(n)}
        out["%dq" % n] = {"vision_1_image_ms": p50(lambda: agent.predict(state, qs, images=[image])),
                          "vision_text_only_ms": p50(lambda: agent.predict(state, qs))}
        if baseline is not None:
            out["%dq" % n]["baseline_text_ms"] = p50(lambda: baseline.predict(state, qs))
    return out


# --------------------------------------------------------------------------- main
def main():
    a = parse_args()
    tasks = [t.strip() for t in a.tasks.split(",") if t.strip()]
    models = dict(m.split("=", 1) for m in a.model)
    report = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "n_per_task": a.n, "seed": a.seed,
              "device": None, "tasks": {}, "latency": {}, "models": models}

    data = {}
    for t in tasks:
        t0 = time.time()
        data[t] = build(t, "test", a.n, a.seed)
        print("%-12s %5d test examples (%.0fs)" % (t, len(data[t]), time.time() - t0), flush=True)

    for t in tasks:
        types = sorted({e["q"]["t"] for e in data[t]})
        report["tasks"][t] = {"type": types[0] if len(types) == 1 else "mixed: " + "+".join(types),
                              "baselines": trivial_baselines(data[t])}

    agents = {}
    for name, path in models.items():
        agents[name] = agent = Agent(path, device=a.device)
        report["device"] = str(agent.device)
        for t in tasks:
            t0 = time.time()
            m = eval_typed(agent, data[t]) if t == "typed" else eval_laya(agent, data[t])
            report["tasks"][t][name] = m
            print("%-8s %-12s acc %.3f  ece %.3f -> %.3f  (%.0fs)" % (
                name, t, m["accuracy"], m["before_temp"]["ece"], m["after_temp"]["ece"], time.time() - t0), flush=True)

    if not a.no_siglip:
        sig = SigLIP(a.siglip, next(iter(agents.values())).device)
        for t in tasks:
            if t != "typed":
                report["tasks"][t]["siglip2_zero_shot"] = eval_siglip(sig, data[t])
                print("siglip2  %-12s acc %.3f" % (t, report["tasks"][t]["siglip2_zero_shot"]["accuracy"]), flush=True)
        del sig

    baseline = None
    if a.latency_baseline:
        repo, _, sub = a.latency_baseline.partition(":")
        baseline = Agent(repo, device=a.device, subfolder=sub or None)
    img = next((d["images"][0] for t in tasks for d in data[t] if d["images"]), None)
    if img is not None:
        for name, agent in agents.items():
            report["latency"][name] = latency(agent, img, baseline)
    report["latency_baseline"] = a.latency_baseline or None

    report["success_criteria"] = criteria(report, list(models), tasks)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(json.dumps(report["success_criteria"], indent=2))
    print("wrote %s" % a.out)


def criteria(report, names, tasks):
    """The plan's success criteria, evaluated as measured (True/False/None when not measurable)."""
    out = {}
    for name in names:
        c = {}
        trained = [t for t in tasks if t in TRAINED_TASKS and "siglip2_zero_shot" in report["tasks"][t]]
        c["beats_siglip2_on_trained_tasks"] = {
            t: report["tasks"][t][name]["accuracy"] > report["tasks"][t]["siglip2_zero_shot"]["accuracy"]
            for t in trained} or None
        c["ece_after_temp_le_0.1"] = {t: report["tasks"][t][name]["after_temp"]["ece"] <= 0.1
                                      for t in tasks if name in report["tasks"][t]}
        if "typed" in tasks:
            base = report["tasks"]["typed"]["baselines"]
            # mixed tasks nest baselines per type; compare per type, else against the one majority
            if "majority" in base:
                c["typed_ge_majority"] = (report["tasks"]["typed"][name]["accuracy"]
                                          >= base["majority"]["accuracy"])
            else:
                by_type = report["tasks"]["typed"][name].get("by_type", {})
                c["typed_ge_majority"] = {qt: by_type[qt]["accuracy"] >= base[qt]["majority"]["accuracy"]
                                          for qt in by_type if qt in base}
        lat = report["latency"].get(name, {}).get("1q", {})
        if "baseline_text_ms" in lat:
            c["latency_1_image_lt_2x_text_baseline"] = lat["vision_1_image_ms"] < 2 * lat["baseline_text_ms"]
        out[name] = c
    return out


if __name__ == "__main__":
    main()
