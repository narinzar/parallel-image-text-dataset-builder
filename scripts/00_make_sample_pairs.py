"""Generate synthetic image-text pairs with planted near-duplicates.

Each base image is a procedurally colored pattern (gradients + shapes) with a
caption describing it. To give the dedup stage something real to find, a fraction
of the images are re-emitted as near-duplicates: the same base image resized to a
different resolution and re-saved as JPEG at a lower quality, which is exactly the
kind of transformation perceptual hashing is meant to catch.

Outputs:
    data/images/*.jpg
    data/pairs.jsonl        one {"image": ..., "caption": ..., "group": ...} per line

`group` records which base image a sample came from so tests / analysis can check
that planted near-duplicates are correctly clustered.

Usage:
    python scripts/00_make_sample_pairs.py --num-base 1500 --dup-frac 0.3
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random

import numpy as np
from PIL import Image
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")
IMAGES_DIR = os.path.join(DATA_DIR, "images")

COLORS = ["red", "green", "blue", "amber", "violet", "teal", "crimson", "olive"]
SHAPES = ["circle", "square", "triangle", "band", "ring", "grid"]
SCENES = ["sky", "field", "ocean", "desert", "forest", "city", "cavern", "dune"]


def _make_base_image(rng: np.random.Generator, size: int = 128) -> Image.Image:
    """Build a deterministic-looking but varied RGB image from numpy."""
    ys = np.linspace(0, 1, size, dtype=np.float32)[:, None]
    xs = np.linspace(0, 1, size, dtype=np.float32)[None, :]

    r = (0.5 + 0.5 * np.sin(6.0 * xs + rng.uniform(0, 6)))
    g = (0.5 + 0.5 * np.cos(6.0 * ys + rng.uniform(0, 6)))
    b = (0.5 + 0.5 * np.sin(4.0 * (xs + ys) + rng.uniform(0, 6)))
    arr = np.stack([r, g, b], axis=-1)

    # Add a bright blob so images are visually distinct and not pure gradients.
    cx, cy = rng.uniform(0.2, 0.8), rng.uniform(0.2, 0.8)
    radius = rng.uniform(0.1, 0.3)
    dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    blob = np.clip(1.0 - dist / radius, 0.0, 1.0)[..., None]
    arr = np.clip(arr + 0.5 * blob, 0.0, 1.0)

    return Image.fromarray((arr * 255).astype(np.uint8), mode="RGB")


def _make_caption(rng: np.random.Generator) -> str:
    color = COLORS[rng.integers(len(COLORS))]
    shape = SHAPES[rng.integers(len(SHAPES))]
    scene = SCENES[rng.integers(len(SCENES))]
    return f"a {color} {shape} over a {scene}"


def _near_duplicate(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    """Resize + JPEG-recompress to produce a near (not exact) duplicate."""
    scale = rng.choice([0.7, 0.85, 1.15, 1.3])
    new_size = max(16, int(image.width * scale))
    resized = image.resize((new_size, new_size), Image.Resampling.BILINEAR)
    buf = io.BytesIO()
    quality = int(rng.integers(35, 70))
    resized.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB").resize(
        (image.width, image.height), Image.Resampling.BILINEAR
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic image-text pairs.")
    parser.add_argument("--num-base", type=int, default=1500,
                        help="number of unique base images")
    parser.add_argument("--dup-frac", type=float, default=0.3,
                        help="fraction of base images that also get a near-duplicate")
    parser.add_argument("--size", type=int, default=128, help="image side length")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    os.makedirs(IMAGES_DIR, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    py_rng = random.Random(args.seed)

    pairs = []
    sample_idx = 0
    for base_id in tqdm(range(args.num_base), desc="Generating"):
        base = _make_base_image(rng, size=args.size)
        caption = _make_caption(rng)

        img_name = f"img_{sample_idx:07d}.jpg"
        base.save(os.path.join(IMAGES_DIR, img_name), format="JPEG", quality=95)
        pairs.append({"image": img_name, "caption": caption, "group": base_id})
        sample_idx += 1

        if py_rng.random() < args.dup_frac:
            dup = _near_duplicate(base, rng)
            dup_name = f"img_{sample_idx:07d}.jpg"
            dup.save(os.path.join(IMAGES_DIR, dup_name), format="JPEG", quality=95)
            # Same caption on purpose: a near-dup of the same content.
            pairs.append({"image": dup_name, "caption": caption, "group": base_id})
            sample_idx += 1

    # Shuffle so duplicates are not adjacent, mimicking a real crawl.
    py_rng.shuffle(pairs)

    out_path = os.path.join(DATA_DIR, "pairs.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for row in pairs:
            f.write(json.dumps(row) + "\n")

    num_dups = sample_idx - args.num_base
    print(f"Wrote {sample_idx} images ({args.num_base} base + {num_dups} near-dups)")
    print(f"Pairs manifest: {out_path}")
    print(f"Images dir:     {IMAGES_DIR}")


if __name__ == "__main__":
    main()
