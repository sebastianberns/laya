"""Typed-question datasets for laya-vision: shared by train_vision.py and bench_vision.py.

Every builder returns plain examples

    {"task": str, "state": str|dict, "images": [bytes], "q": {"t", "ins", "crit"}, "target": [float]}

with images kept as encoded bytes (a decoded 512px tile is 3 MB; bytes are ~50 KB), so a
DataLoader can hold tens of thousands of them. Splits are fixed-seed and disjoint:

  task          source                                   train / val / test
  cifar100      uoft-cs/cifar100 (choice, <=20 opts)     train / test even idx / test odd idx
  imagenet100   clane9/imagenet-100 (choice, <=20 opts)  train / validation even / odd
  rvl_cdip      chainyo/rvl-cdip (choice, 16 doc types)  train / validation / test
  vqav2_yesno   lmms-lab/VQAv2 validation (noul)         hash(image_id) 80 / 10 / 10
  koniq         KonIQ-10k (score, 5 levels, soft votes)  official train / val / test
  typed         LocalLLaMA/typed-decisions (text replay) train (90%) / train (10%) / test
  pets          timm/oxford-iiit-pet, 20 fixed breeds    test only: the zero-shot probe

VQAv2 publishes answers only for `validation`, hence the image-level hash split: an image
never appears in two splits. The typed-decisions val slice is carved from train by case id.
"""
import csv
import io
import json
import os
import random
import tarfile
import zlib
from typing import Dict, List, Optional

VISION_TASKS = ["cifar100", "rvl_cdip", "vqav2_yesno", "koniq"]
ALL_TASKS = VISION_TASKS + ["imagenet100", "typed", "pets"]
MAX_OPTIONS = 20

CONTEXTS = ["", {"source": "customer upload"}, {"note": "attached image"}, {"channel": "support ticket"},
            "See the attached image.", {"context": "automated intake"}]


def _bucket(key, n: int = 100) -> int:
    return zlib.crc32(str(key).encode()) % n


def _img_bytes(img) -> bytes:
    """The image's encoded bytes, re-encoding only when there is no other choice.

    With `decode=False` (see `_load`) datasets hands back {"bytes", "path"} straight from the
    parquet file, so building a split costs no decode and no re-encode -- it was the dominant
    cost, and it is pure waste when the DataLoader decodes lazily during training anyway.
    """
    if isinstance(img, dict):
        if img.get("bytes"):
            return img["bytes"]
        if img.get("path"):
            with open(img["path"], "rb") as f:
                return f.read()
        raise ValueError("image dict has neither bytes nor path")
    buf = io.BytesIO()
    img = img.convert("RGB")
    img.save(buf, format="JPEG" if max(img.size) > 128 else "PNG", quality=92)
    return buf.getvalue()


def _choice(names: List[str], gold: int, rng: random.Random, ins: str, max_options: int = MAX_OPTIONS) -> Dict:
    """A choice question over at most `max_options` labels, gold included, order shuffled."""
    pool = [i for i in range(len(names)) if i != gold]
    picked = rng.sample(pool, min(len(pool), max_options - 1)) + [gold]
    rng.shuffle(picked)
    crit = {names[i]: None for i in picked}
    target = [1.0 if i == gold else 0.0 for i in picked]
    return {"q": {"t": "choice", "ins": ins, "crit": crit}, "target": target}


def _take(ds, n: Optional[int], keep=lambda row: True, label=""):
    out = []
    for row in ds:
        if keep(row):
            out.append(row)
            _tick(len(out), n, label)
            if n is not None and len(out) >= n:
                break
    return out


def _tick(i, n, label):
    """Progress while a split streams: without it a slow build is indistinguishable from a hang."""
    if label and i and i % PROGRESS_EVERY == 0:
        print("    %s: %d%s" % (label, i, "/%d" % n if n else ""), flush=True)


# A large buffer means nothing is yielded until it fills, which for document scans is minutes of
# downloading before the first example. Shuffling a stream is a nicety here: the builders sample
# and shuffle their own output anyway.
SHUFFLE_BUFFER = 1_000
PROGRESS_EVERY = 1_000


def _load(repo, split, config=None, streaming=True, seed=0, shuffle=True, image_column=None):
    """Load a split. `image_column` is handed back undecoded, so no image is decoded at build time."""
    from datasets import Image as DsImage
    from datasets import load_dataset

    ds = load_dataset(repo, config, split=split, streaming=streaming)
    if image_column:
        try:
            ds = ds.cast_column(image_column, DsImage(decode=False))
        except Exception:       # older datasets, or a column that is not an Image feature
            pass
    if shuffle:
        ds = ds.shuffle(seed=seed, buffer_size=SHUFFLE_BUFFER) if streaming else ds.shuffle(seed=seed)
    return ds


# --------------------------------------------------------------------------- builders
CIFAR_INS = ["What is in this image?", "Which object or animal does the photo show?", "Classify the image.",
             "What is the main subject of the picture?"]


def build_cifar100(split: str, n: Optional[int], seed: int = 0) -> List[Dict]:
    src = "train" if split == "train" else "test"
    ds = _load("uoft-cs/cifar100", src, seed=seed, streaming=False, shuffle=False, image_column="img")
    names = [s.replace("_", " ") for s in ds.features["fine_label"].names]
    idx = list(range(len(ds)))
    if split != "train":
        idx = [i for i in idx if i % 2 == (0 if split == "val" else 1)]
    rng = random.Random("cifar100-%s-%d" % (split, seed))
    rng.shuffle(idx)
    out = []
    for i in idx[:n]:
        row = ds[i]
        ex = _choice(names, row["fine_label"], rng, rng.choice(CIFAR_INS))
        out.append(dict(ex, task="cifar100", state=rng.choice(CONTEXTS), images=[_img_bytes(row["img"])]))
    return out


def build_imagenet100(split: str, n: Optional[int], seed: int = 0) -> List[Dict]:
    # val/test alternate over the unshuffled validation stream, so they never overlap
    src = "train" if split == "train" else "validation"
    ds = _load("clane9/imagenet-100", src, seed=seed, shuffle=split == "train", image_column="image")
    names = [s.split(",")[0].strip() for s in ds.features["label"].names]
    rng = random.Random("imagenet100-%s-%d" % (split, seed))
    parity = {"val": 0, "test": 1}.get(split)
    out = []
    for k, row in enumerate(ds):
        if parity is not None and k % 2 != parity:
            continue
        ex = _choice(names, row["label"], rng, rng.choice(CIFAR_INS))
        out.append(dict(ex, task="imagenet100", state=rng.choice(CONTEXTS), images=[_img_bytes(row["image"])]))
        if n is not None and len(out) >= n:
            break
    return out


RVL_INS = ["What type of document is this?", "Classify the scanned document.", "Which document category fits best?"]


def build_rvl_cdip(split: str, n: Optional[int], seed: int = 0) -> List[Dict]:
    ds = _load("chainyo/rvl-cdip", split, seed=seed, image_column="image")   # splits: train/val/test
    names = ds.features["label"].names
    rng = random.Random("rvl-%s-%d" % (split, seed))
    out = []
    for row in _take(ds, n, label="rvl_cdip %s" % split):
        ex = _choice(names, row["label"], rng, rng.choice(RVL_INS))
        out.append(dict(ex, task="rvl_cdip", state=rng.choice(CONTEXTS), images=[_img_bytes(row["image"])]))
    return out


def build_vqav2_yesno(split: str, n: Optional[int], seed: int = 0) -> List[Dict]:
    lo, hi = {"train": (0, 80), "val": (80, 90), "test": (90, 100)}[split]
    ds = _load("lmms-lab/VQAv2", "validation", seed=seed, image_column="image")

    def keep(r):
        return r["answer_type"] == "yes/no" and lo <= _bucket(r["image_id"]) < hi

    out = []
    rng = random.Random("vqa-%s-%d" % (split, seed))
    for row in _take(ds, n, keep, label="vqav2_yesno %s" % split):
        votes = [a["answer"].strip().lower() for a in row["answers"]]
        yes, no = votes.count("yes"), votes.count("no")
        if yes + no == 0:
            continue
        p_yes = yes / (yes + no)
        out.append({"task": "vqav2_yesno", "state": rng.choice(CONTEXTS), "images": [_img_bytes(row["image"])],
                    "q": {"t": "noul", "ins": row["question"], "crit": None}, "target": [1 - p_yes, p_yes]})
    return out


KONIQ_LEVELS = ["bad: heavily distorted, unusable", "poor: clearly degraded", "fair: acceptable with visible flaws",
                "good: sharp and well exposed", "excellent: flawless technical quality"]
KONIQ_INS = ["Rate the technical quality of this photo.", "How good is the image quality?",
             "Score the photo's sharpness, exposure and noise."]


def _koniq_files(cache_dir: Optional[str] = None):
    """Metadata CSV + extracted image dir (the archive is 6.3 GB; extracted once, then cached)."""
    from huggingface_hub import hf_hub_download

    meta = hf_hub_download("chaofengc/IQA-PyTorch-Datasets-metainfo", "meta_info_KonIQ10kDataset.csv",
                           repo_type="dataset")
    root = cache_dir or os.path.join(os.path.expanduser("~"), ".cache", "laya_vision", "koniq10k")
    done = os.path.join(root, ".extracted")
    if not os.path.exists(done):
        tgz = hf_hub_download("chaofengc/IQA-PyTorch-Datasets", "koniq10k.tgz", repo_type="dataset")
        os.makedirs(root, exist_ok=True)
        with tarfile.open(tgz, "r:gz") as tf:
            for m in tf:
                if m.isfile() and m.name.lower().endswith(".jpg"):
                    m.name = os.path.basename(m.name)
                    try:
                        tf.extract(m, os.path.join(root, "images"), filter="data")
                    except TypeError:       # Python without extraction filters (< 3.10.12)
                        tf.extract(m, os.path.join(root, "images"))
        open(done, "w").close()
    return meta, os.path.join(root, "images")


def build_koniq(split: str, n: Optional[int], seed: int = 0) -> List[Dict]:
    meta, img_dir = _koniq_files()
    with open(meta) as f:
        rows = [r for r in csv.DictReader(f) if r["official_split"] == split]
    rng = random.Random("koniq-%s-%d" % (split, seed))
    rng.shuffle(rows)
    out = []
    for r in rows[:n]:
        votes = [float(r["c%d" % i]) for i in range(1, 6)]
        s = sum(votes) or 1.0
        with open(os.path.join(img_dir, r["img_name"]), "rb") as f:
            raw = f.read()
        out.append({"task": "koniq", "state": rng.choice(CONTEXTS), "images": [raw],
                    "q": {"t": "score", "ins": rng.choice(KONIQ_INS), "crit": list(KONIQ_LEVELS)},
                    "target": [v / s for v in votes], "mos": float(r["mos"])})
    return out


def build_typed(split: str, n: Optional[int], seed: int = 0) -> List[Dict]:
    """Text-only replay of LocalLLaMA/typed-decisions, one example per question (as in the notebook)."""
    src = "test" if split == "test" else "train"
    ds = _load("LocalLLaMA/typed-decisions", src, config="all", streaming=False, shuffle=False)
    out = []
    for row in ds:
        if src == "train" and (_bucket(row["id"]) < 10) != (split == "val"):
            continue
        state, questions, gold = json.loads(row["state"]), json.loads(row["questions"]), json.loads(row["gold"])
        for qid, q in questions.items():
            if qid not in gold:
                continue
            t, crit, probs = q["type"], q.get("criteria"), gold[qid]["probabilities"]
            if t == "choice":
                crit = {c: None for c in crit} if isinstance(crit, list) else crit
                target = [probs.get(k, 0.0) for k in crit]
            elif t == "noul":
                target = [probs.get("false", 0.5), probs.get("true", 0.5)]
            else:
                target = [probs.get(str(i), 0.0) for i in range(len(crit))]
            s = sum(target)
            target = [v / s for v in target] if s > 0 else [1.0 / len(target)] * len(target)
            out.append({"task": "typed", "state": state, "images": [], "case": row["id"], "qid": qid,
                        "workflow": row["workflow"], "q": {"t": t, "ins": q["instructions"], "crit": crit},
                        "target": target})
    random.Random("typed-%s-%d" % (split, seed)).shuffle(out)
    return out[:n] if n is not None else out


PET_INS = ["Which breed is this pet?", "Identify the cat or dog breed in the photo."]


def build_pets(split: str, n: Optional[int], seed: int = 0) -> List[Dict]:
    """Zero-shot probe: never trained on. 20 breeds fixed by seed, evaluated as a 20-way choice."""
    if split != "test":
        raise ValueError("pets is an evaluation-only probe")
    ds = _load("timm/oxford-iiit-pet", "test", streaming=False, shuffle=False, image_column="image")
    names = [s.replace("_", " ") for s in ds.features["label"].names]
    rng = random.Random("pets-%d" % seed)
    breeds = sorted(rng.sample(range(len(names)), MAX_OPTIONS))
    sub = [names[i] for i in breeds]
    idx = [i for i, lab in enumerate(ds["label"]) if lab in breeds]     # label column only: no decode
    rng.shuffle(idx)
    out = []
    for i in idx[:n]:
        row = ds[i]
        order = list(range(len(sub)))
        rng.shuffle(order)
        gold = breeds.index(row["label"])
        crit = {sub[j]: None for j in order}
        out.append({"task": "pets", "state": rng.choice(CONTEXTS), "images": [_img_bytes(row["image"])],
                    "q": {"t": "choice", "ins": rng.choice(PET_INS), "crit": crit},
                    "target": [1.0 if j == gold else 0.0 for j in order]})
    return out


BUILDERS = {
    "cifar100": build_cifar100,
    "imagenet100": build_imagenet100,
    "rvl_cdip": build_rvl_cdip,
    "vqav2_yesno": build_vqav2_yesno,
    "koniq": build_koniq,
    "typed": build_typed,
    "pets": build_pets,
}


def build(task: str, split: str, n: Optional[int], seed: int = 0) -> List[Dict]:
    return BUILDERS[task](split, n, seed)
