"""Tests for perceptual hashing stability and discrimination."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from src.phash import HASH_BITS, ahash, dhash, hamming_distance


def _make_image(seed: int, size: int = 128) -> Image.Image:
    """Sum of random low-frequency plane waves: distinctive but resize-stable."""
    rng = np.random.default_rng(seed)
    ys = np.linspace(0, 1, size)[:, None]
    xs = np.linspace(0, 1, size)[None, :]
    lum = np.zeros((size, size))
    for _ in range(4):
        fx = rng.uniform(-3.5, 3.5)
        fy = rng.uniform(-3.5, 3.5)
        phase = rng.uniform(0, 2 * np.pi)
        lum += np.sin(2 * np.pi * (fx * xs + fy * ys) + phase)
    lum = (lum - lum.min()) / (np.ptp(lum) + 1e-6)
    tint = rng.uniform(0.2, 1.0, size=3)
    arr = np.clip(lum[..., None] * tint[None, None, :], 0, 1)
    return Image.fromarray((arr * 255).astype(np.uint8), "RGB")


def _resize_recompress(image: Image.Image, scale: float, quality: int) -> Image.Image:
    new_size = max(16, int(image.width * scale))
    resized = image.resize((new_size, new_size), Image.Resampling.BILINEAR)
    buf = io.BytesIO()
    resized.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def test_hash_width_is_64_bits():
    img = _make_image(0)
    for h in (dhash(img), ahash(img)):
        assert 0 <= h < (1 << HASH_BITS)


def test_identical_image_zero_distance():
    img = _make_image(1)
    assert hamming_distance(dhash(img), dhash(img)) == 0


def test_dhash_stable_under_resize_and_jpeg():
    """A resized + recompressed copy stays close to the original hash."""
    img = _make_image(2)
    original = dhash(img)
    for scale, quality in [(0.75, 50), (1.25, 40), (0.85, 60)]:
        variant = dhash(_resize_recompress(img, scale, quality))
        dist = hamming_distance(original, variant)
        assert dist <= 8, f"near-dup distance too large: {dist}"


def test_different_images_have_large_distance():
    """Unrelated images should differ in many bits."""
    distances = []
    base = dhash(_make_image(100))
    for seed in range(101, 120):
        other = dhash(_make_image(seed))
        distances.append(hamming_distance(base, other))
    assert min(distances) >= 12, f"images not distinct enough: min={min(distances)}"
    assert np.mean(distances) >= 20


def test_near_dup_closer_than_unrelated():
    """The near-dup gap: a recompressed copy is closer than any random image."""
    img = _make_image(7)
    base = dhash(img)
    near = hamming_distance(base, dhash(_resize_recompress(img, 0.8, 45)))
    unrelated = [
        hamming_distance(base, dhash(_make_image(seed))) for seed in range(200, 215)
    ]
    assert near < min(unrelated)


def test_ahash_also_stable():
    img = _make_image(9)
    original = ahash(img)
    variant = ahash(_resize_recompress(img, 0.8, 55))
    assert hamming_distance(original, variant) <= 10


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
