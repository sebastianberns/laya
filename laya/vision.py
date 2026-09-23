"""Image inputs for the laya-vision checkpoint (ModernVBERT encoder).

An image joins the state as a block of placeholder tokens -- `<fake_token_around_image>`,
`<global-img>`, `<image>` x image_seq_len, `<fake_token_around_image>` per image (plus row/col
tiles when splitting) -- that the encoder swaps for projected SigLIP features. `build_sequence`
places that block at the start of the state segment, after every option marker.

Needs the `vision` extra: `pip install "laya[vision]"` (transformers>=5.3, pillow).
"""
import io
import json
import os
import re
from typing import Any, List, Optional, Sequence, Tuple

# Special tokens of the Idefics3-style image block. Scrubbed from all text on a vision
# checkpoint so user text cannot pose as an image placeholder.
IMAGE_TOKENS = ["<fake_token_around_image>", "<image>", "<global-img>", "<end_of_utterance>"] + [
    "<row_%d_col_%d>" % (r, c) for r in range(1, 7) for c in range(1, 7)
]

MIN_TRANSFORMERS = (5, 3)


def _require_vision_deps():
    try:
        import transformers
    except ImportError as e:        # pragma: no cover - transformers is a core dependency
        raise ImportError('laya-vision needs transformers>=5.3: pip install "laya[vision]"') from e
    version = tuple(int(p) for p in re.findall(r"\d+", transformers.__version__)[:2])
    if version < MIN_TRANSFORMERS:
        raise ImportError(
            "laya-vision runs on ModernVBERT, which needs transformers>=%d.%d (found %s): "
            'pip install "laya[vision]"' % (MIN_TRANSFORMERS + (transformers.__version__,)))
    try:
        import PIL  # noqa: F401
    except ImportError as e:
        raise ImportError('laya-vision needs pillow to read images: pip install "laya[vision]"') from e


def _image_processor_class():
    # The PIL backend, always: it needs no torchvision, and training and inference must resize
    # identically. transformers>=5.x names it *Pil; 5.3 ships it under the plain name.
    try:
        from transformers.models.idefics3.image_processing_pil_idefics3 import Idefics3ImageProcessorPil
        return Idefics3ImageProcessorPil
    except ImportError:
        from transformers.models.idefics3.image_processing_idefics3 import Idefics3ImageProcessor
        return Idefics3ImageProcessor


def load_processor(path_or_repo: str, image_seq_len: Optional[int] = None, token: Optional[str] = None):
    """Load the image+text processor that ships in a checkpoint's `processor/` (or a Hub repo).

    Built from its parts rather than via AutoProcessor, which in recent transformers insists on
    torchvision for this image processor. `image_seq_len` should come from the encoder config
    (`encoder.image_seq_len`) so the placeholder count always matches the connector's output.
    """
    _require_vision_deps()
    from transformers import AutoTokenizer
    from transformers.models.idefics3.processing_idefics3 import Idefics3Processor

    kw = {"token": token} if token else {}
    tokenizer = AutoTokenizer.from_pretrained(path_or_repo, **kw)
    image_processor = _image_processor_class().from_pretrained(path_or_repo, **kw)
    if image_seq_len is None:
        cfg_file = os.path.join(path_or_repo, "processor_config.json")
        if os.path.exists(cfg_file):
            with open(cfg_file) as f:
                image_seq_len = json.load(f).get("image_seq_len")
    return Idefics3Processor(image_processor, tokenizer, image_seq_len=int(image_seq_len or 64))


def _as_list(images) -> List[Any]:
    if images is None:
        return []
    if isinstance(images, (list, tuple)):
        return list(images)
    return [images]


def to_pil(image, max_side: Optional[int] = None):
    """A PIL RGB image from a PIL image, a file path, raw bytes, or a binary file object.

    `max_side` shrinks anything larger so its longest edge is at most that, and never enlarges.
    The processor resizes to the same geometry anyway, but capping here is what actually saves the
    work, and it does not depend on the processor honouring a per-call `size`. (PIL's `draft`
    reduced-scale JPEG decode was measured and left out: no gain below ~2048px, ~20% above it, and
    no training source here is that large.)
    """
    from PIL import Image

    if isinstance(image, Image.Image):
        img = image
    elif isinstance(image, (str, os.PathLike)):
        img = Image.open(image)
    elif isinstance(image, (bytes, bytearray, memoryview)):
        img = Image.open(io.BytesIO(bytes(image)))
    elif hasattr(image, "read"):
        img = Image.open(image)
    else:
        raise TypeError("unsupported image of type %s; pass a PIL image, a path, or bytes" % type(image).__name__)
    if max_side and max(img.size) > max_side:
        img = img.copy() if img is image else img       # thumbnail is in-place: never touch the caller's image
        img.thumbnail((max_side, max_side), Image.LANCZOS)
    img.load()
    return img.convert("RGB")


def image_block(processor, images: Sequence[Any], tiles_per_side: int = 1) -> Tuple[List[int], Any, Any]:
    """Preprocess images once: (prefix_ids, pixel_values, pixel_attention_mask).

    `prefix_ids` is the placeholder block for all images in order, without CLS/SEP, ready for
    `build_sequence(prefix_ids=...)`. `pixel_values` is [1, tiles, 3, H, W] for the whole call.
    `tiles_per_side=1` sends each image as one 512px tile (~67 tokens); a larger value splits
    images into up to n x n tiles plus a global view, for documents where detail matters.
    """
    tile = processor.image_processor.max_image_size["longest_edge"]
    imgs = [to_pil(im, max_side=tile * max(1, int(tiles_per_side))) for im in _as_list(images)]
    if not imgs:
        raise ValueError("image_block needs at least one image")
    # `size` caps the processor's first resize. The shipped config sets it to 2048, but with
    # splitting off the image is squashed to one `tile`-square anyway, so that intermediate is
    # discarded work -- and preprocessing dominates image latency. The images are already capped
    # above, which is what guarantees the saving; this kwarg only stops the processor from
    # scaling them back up. With splitting on, the grid genuinely needs tile * n.
    kw = {"do_image_splitting": tiles_per_side > 1,
          "size": {"longest_edge": tile * max(1, int(tiles_per_side))}}
    out = processor(text=[processor.image_token * len(imgs)], images=[imgs],
                    add_special_tokens=False, return_tensors="pt", **kw)
    tok = processor.tokenizer
    ids = [i for i in out["input_ids"][0].tolist() if i not in (tok.cls_token_id, tok.sep_token_id)]
    return ids, out["pixel_values"], out.get("pixel_attention_mask")
