"""Offline tests for the vision stream: image block layout, shared image features, routing.

A tiny from-config ModernVBERT (small SigLIP + small ModernBERT), an in-memory word-level
tokenizer and the PIL image processor at 32px stand in for the real checkpoint, so nothing is
downloaded. Needs the vision extra (transformers>=5.3, pillow); without it the test skips.
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from laya.vision import _require_vision_deps
    _require_vision_deps()
except ImportError as e:
    print("SKIP test_vision: %s" % e)
    sys.exit(0)

import torch  # noqa: E402
from PIL import Image  # noqa: E402
from safetensors.torch import save_file  # noqa: E402
from tokenizers import Tokenizer, models, pre_tokenizers  # noqa: E402
from transformers import AutoModel, ModernVBertConfig, PreTrainedTokenizerFast  # noqa: E402
from transformers.models.idefics3.processing_idefics3 import Idefics3Processor  # noqa: E402

from laya.agent import Agent  # noqa: E402
from laya.common import DecisionModel, build_sequence  # noqa: E402
from laya.router import Router  # noqa: E402
from laya.vision import IMAGE_TOKENS, _image_processor_class, image_block, load_processor  # noqa: E402

PASS, FAIL = [], []


def check(name, got, want):
    if got == want:
        PASS.append(name)
    else:
        FAIL.append("%s: got %r, want %r" % (name, got, want))


def check_raises(name, fn, exc=ValueError):
    try:
        fn()
    except exc:
        PASS.append(name)
    except Exception as e:
        FAIL.append("%s: raised %s (%s), want %s" % (name, type(e).__name__, e, exc.__name__))
    else:
        FAIL.append("%s: no exception" % name)


# --------------------------------------------------------------------- tiny processor + model
SPECIALS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + IMAGE_TOKENS
WORDS = ("choice score noul question is the a an of photo image document parcel damaged intact red blue "
         "cat dog level false true no yes statement does not hold holds what colour kind note customer "
         "word : , { } \" level").split()
TILE = 32           # image_size of the tiny SigLIP, and the processor's tile edge
PATCH = 8           # -> 16 patches per tile
SHUFFLE = 2         # -> 4 image tokens per tile


def tiny_processor():
    vocab = {t: i for i, t in enumerate(dict.fromkeys(SPECIALS + WORDS))}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]",
                                  cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]",
                                  additional_special_tokens=IMAGE_TOKENS)
    ip = _image_processor_class()(size={"longest_edge": TILE}, max_image_size={"longest_edge": TILE},
                                  do_image_splitting=False)
    return Idefics3Processor(ip, tok, image_seq_len=(TILE // PATCH) ** 2 // SHUFFLE ** 2)


def tiny_encoder(tok):
    cfg = ModernVBertConfig(
        text_config=dict(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=2, max_position_embeddings=512, pad_token_id=tok.pad_token_id,
                         cls_token_id=tok.cls_token_id, sep_token_id=tok.sep_token_id,
                         bos_token_id=tok.cls_token_id, eos_token_id=tok.sep_token_id,
                         global_attn_every_n_layers=1, local_attention=16),
        vision_config=dict(hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
                           image_size=TILE, patch_size=PATCH),
        image_token_id=tok.convert_tokens_to_ids("<image>"),
        pixel_shuffle_factor=SHUFFLE,
    )
    return AutoModel.from_config(cfg, attn_implementation="sdpa")


torch.manual_seed(0)
proc = tiny_processor()
tok = proc.tokenizer
enc = tiny_encoder(tok)
check("model/auto resolves ModernVBertModel", type(enc).__name__, "ModernVBertModel")
check("model/image_seq_len matches processor", enc.image_seq_len, proc.image_seq_len)
model = DecisionModel(enc, head_layers=1).eval()
check("model/head width from text_config", model.type_emb.embedding_dim, 32)

RED = Image.new("RGB", (40, 30), "red")
BLUE = Image.new("RGB", (20, 60), "blue")
Q = {"t": "choice", "ins": "what colour is the photo", "crit": {"red": None, "blue": "a blue photo"}}
STATE = {"note": " ".join(["word"] * 400)}


# --------------------------------------------------------------------- image block
prefix, pv, pam = image_block(proc, [RED])
img_id = tok.convert_tokens_to_ids("<image>")
check("block/one image = wrapper + seq_len image tokens", len(prefix), proc.image_seq_len + 3)
check("block/image token count", prefix.count(img_id), proc.image_seq_len)
check("block/no CLS or SEP", tok.cls_token_id in prefix or tok.sep_token_id in prefix, False)
check("block/pixel_values shape", tuple(pv.shape), (1, 1, 3, TILE, TILE))
prefix2, pv2, _ = image_block(proc, [RED, BLUE])
check("block/two images", (len(prefix2), pv2.shape[1]), (2 * len(prefix), 2))

# paths and bytes are accepted as well as PIL images
with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, "red.png")
    RED.save(path)
    with open(path, "rb") as f:
        raw = f.read()
    check("block/path input", image_block(proc, [path])[0], prefix)
    check("block/bytes input", image_block(proc, [raw])[0], prefix)
check("block/single image not in a list", image_block(proc, RED)[0], prefix)

# The first resize must be capped at the tile size, not the processor's shipped `size` (2048 on the
# real checkpoint): with splitting off the image is squashed to one tile anyway, so a large
# intermediate is pure cost, and preprocessing dominates image latency.
_saved = proc.image_processor.size
proc.image_processor.size = {"longest_edge": 64 * TILE}
check("block/ignores an oversized configured size", image_block(proc, [RED])[1].shape[-2:], (TILE, TILE))
check("block/same tokens with an oversized configured size", image_block(proc, [RED])[0], prefix)
proc.image_processor.size = _saved
check_raises("block/unsupported type", lambda: image_block(proc, [42]), TypeError)


# --------------------------------------------------------------------- sequence layout
MAX_LEN, HEAD = 256, 64
txt_ids, txt_markers = build_sequence(tok, STATE, Q, MAX_LEN, HEAD, reserved_tokens=IMAGE_TOKENS)
img_ids, img_markers = build_sequence(tok, STATE, Q, MAX_LEN, HEAD, prefix_ids=prefix, reserved_tokens=IMAGE_TOKENS)
check("layout/markers identical to text-only", img_markers, txt_markers)
start = img_ids.index(prefix[0])
check("layout/block is contiguous", img_ids[start:start + len(prefix)], prefix)
check("layout/block after every marker", start > max(img_markers), True)
check("layout/block opens the state segment", img_ids[start - 1], tok.sep_token_id)
check("layout/head unchanged", img_ids[:start], txt_ids[:start])
txt_state = len(txt_ids) - start - 1
img_state = len(img_ids) - start - len(prefix) - 1
check("budget/state shrinks by exactly the block", txt_state - img_state, len(prefix))
check("budget/both fill max_len", (len(txt_ids), len(img_ids)), (MAX_LEN, MAX_LEN))
check("budget/ends with SEP", img_ids[-1], tok.sep_token_id)

check_raises("budget/oversized block raises",
             lambda: build_sequence(tok, STATE, Q, 60, 32, prefix_ids=prefix * 10))

# A literal "<image>" in user text must not become an image placeholder.
sneaky_ids, _ = build_sequence(tok, "a photo <image> <fake_token_around_image>", Q, MAX_LEN, HEAD,
                               reserved_tokens=IMAGE_TOKENS)
check("scrub/no image tokens from text", img_id in sneaky_ids, False)


# --------------------------------------------------------------------- shared image features
qs = [Q, {"t": "noul", "ins": "is the parcel damaged", "crit": None},
      {"t": "score", "ins": "kind of photo", "crit": ["no", "yes", "red"]}]
rows = [build_sequence(tok, {"note": "customer photo"}, q, MAX_LEN, HEAD, prefix_ids=prefix2) for q in qs]
from laya.common import collate_items  # noqa: E402
b = collate_items([[{"ids": s, "markers": m, "qtype": {"choice": 0, "score": 1, "noul": 2}[q["t"]]}
                    for (s, m), q in zip(rows, qs)]], tok.pad_token_id)
args = (b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"])
n = len(qs)
with torch.no_grad():
    feats = enc.get_image_features(pixel_values=pv2).pooler_output
    shared, _ = model(*args, image_hidden_states=feats.repeat(n, 1, 1))
    per_row, _ = model(*args, pixel_values=pv2.repeat(n, 1, 1, 1, 1))
    no_img, _ = model(*args)
diff = (shared - per_row).abs().max().item()
check("shared/features once == per-row vision (1e-5)", diff < 1e-5, True)
check("shared/images change the logits", (shared - no_img).abs().max().item() > 1e-6, True)


# --------------------------------------------------------------------- Agent on a vision checkpoint
def write_checkpoint(d, model, proc, cfg):
    os.makedirs(os.path.join(d, "encoder"))
    model.encoder.config.save_pretrained(os.path.join(d, "encoder"))
    proc.save_pretrained(os.path.join(d, "processor"))
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, os.path.join(d, "model.safetensors"))
    with open(os.path.join(d, "rl_agent_config.json"), "w") as f:
        json.dump(cfg, f)


QUESTIONS = {
    "colour": {"type": "choice", "instructions": "what colour is the photo", "criteria": ["red", "blue"]},
    "damaged": {"type": "noul", "instructions": "is the parcel damaged"},
    "quality": {"type": "score", "instructions": "kind of photo", "criteria": ["no", "yes", "red"]},
}

with tempfile.TemporaryDirectory() as ckpt:
    write_checkpoint(ckpt, model, proc, {
        "encoder": "ModernVBERT/modernvbert", "head_layers": 1, "max_len": MAX_LEN, "head_max_len": HEAD,
        "vision": {"tiles_per_side": 1}, "act_costs": {"escalate": 1.0},
    })
    check("processor/round-trips from disk",
          image_block(load_processor(os.path.join(ckpt, "processor")), [RED])[0], prefix)

    agent = Agent(ckpt, device="cpu")
    check("agent/vision flag", bool(agent.vision), True)
    check("agent/weights loaded", torch.equal(agent.model.scorer[1].weight, model.scorer[1].weight), True)

    def valid(res):
        a = res["answers"]
        p = list(a["colour"]["probabilities"].values())
        s = list(a["quality"]["probabilities"].values())
        return (abs(sum(p) - 1) < 1e-3 and abs(sum(s) - 1) < 1e-3 and 0 <= a["damaged"]["noul"] <= 1
                and set(a) == set(QUESTIONS))

    text_res = agent.predict({"note": "customer photo"}, QUESTIONS)
    check("agent/no images gives valid probabilities", valid(text_res), True)
    check("agent/images=None is the text path",
          agent.predict({"note": "customer photo"}, QUESTIONS, images=None)["answers"], text_res["answers"])
    img_res = agent.predict({"note": "customer photo"}, QUESTIONS, images=[RED])
    check("agent/with image gives valid probabilities", valid(img_res), True)
    check("agent/image tokens counted", img_res["usage"]["input_tokens"] - text_res["usage"]["input_tokens"],
          len(QUESTIONS) * len(prefix))
    check("agent/single image outside a list", agent.predict({"note": "x"}, QUESTIONS, images=RED)["answers"],
          agent.predict({"note": "x"}, QUESTIONS, images=[RED])["answers"])

    # _verify_compatibility requires the vision tower when the config says vision
    del_dir = os.path.join(ckpt, "stripped")
    os.makedirs(del_dir)
    sd = {k: v.contiguous() for k, v in model.state_dict().items() if not k.startswith("encoder.vision_model.")}
    for name in ("encoder", "processor"):
        os.symlink(os.path.join(ckpt, name), os.path.join(del_dir, name))
    save_file(sd, os.path.join(del_dir, "model.safetensors"))
    with open(os.path.join(ckpt, "rl_agent_config.json")) as f, \
            open(os.path.join(del_dir, "rl_agent_config.json"), "w") as g:
        g.write(f.read())
    check_raises("agent/missing vision tower rejected", lambda: Agent(del_dir, device="cpu"))

    # Router end to end with a local vision checkpoint
    r = Router(models={"vision": ckpt})
    routed = r.predict({"note": "customer photo"}, QUESTIONS, images=[RED])
    check("router/predict routes images to vision", routed["routing"]["model"], "vision")
    check("router/reason mentions images", "image" in routed["routing"]["reason"], True)
    check("router/same answers as the agent", routed["answers"], img_res["answers"])


# --------------------------------------------------------------------- text checkpoint rejects images
text_agent = Agent.__new__(Agent)
text_agent.vision = None
check_raises("text/images on a text checkpoint raise", lambda: text_agent.system_one("x", QUESTIONS, images=[RED]))


# --------------------------------------------------------------------- routing (pure)
r = Router()
d = r.route({"note": "customer photo"}, QUESTIONS, images=[RED])
check("route/images pick vision", d.model, "vision")
check("route/reason counts images", d.reason, "state includes 1 image(s)")
check("route/two images", r.route("x", QUESTIONS, images=[RED, BLUE]).reason, "state includes 2 image(s)")
check("route/bare image counts as one", r.route("x", QUESTIONS, images=RED).model, "vision")
check("route/model= overrides images", r.route("x", QUESTIONS, model="english", images=[RED]).model, "english")
check("route/task= overrides images", r.route("x", QUESTIONS, task="typed_decisions", images=[RED]).model,
      "typed-decisions")
check("route/images beat lang", r.route("x", QUESTIONS, lang="de", images=[RED]).model, "vision")
check("route/empty images is text routing", r.route("I was charged twice", QUESTIONS, images=[]).model, "english")
check("route/local vision override", Router(models={"vision": "/tmp/v"}).route("x", QUESTIONS, images=[RED])["repo"],
      "/tmp/v")


print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL", f)
if not FAIL:
    print("all vision tests passed")
sys.exit(1 if FAIL else 0)
