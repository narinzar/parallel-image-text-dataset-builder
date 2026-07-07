"""Perceptual hashing from scratch with PIL + numpy.

Two hash families are provided, both returning 64-bit integers:

    ahash (average hash)
        Downscale to 8x8 grayscale, threshold each pixel against the mean.

    dhash (difference hash)
        Downscale to 9x8 grayscale, compare each pixel to its right neighbor
        (8 comparisons per row * 8 rows = 64 bits). dhash is more stable than
        ahash under brightness shifts and mild resize/recompression because it
        encodes gradients rather than absolute levels.

Hashes are plain Python ints so they are trivial to store, compare, and slice
into prefix bands for the LSH-style bucketing in dedup.py.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

HASH_BITS = 64


def _to_gray_array(image: Image.Image, width: int, height: int) -> np.ndarray:
    """Downscale to (height, width) grayscale and return a float array."""
    gray = image.convert("L").resize((width, height), Image.Resampling.LANCZOS)
    return np.asarray(gray, dtype=np.float64)


def _bits_to_int(bits: np.ndarray) -> int:
    """Pack a flat boolean array (MSB first) into a Python int."""
    value = 0
    for bit in bits.astype(np.uint8).ravel():
        value = (value << 1) | int(bit)
    return value


def ahash(image: Image.Image) -> int:
    """Average hash: 8x8 grayscale thresholded at the mean. Returns 64-bit int."""
    pixels = _to_gray_array(image, 8, 8)
    bits = pixels > pixels.mean()
    return _bits_to_int(bits)


def dhash(image: Image.Image) -> int:
    """Difference hash: horizontal gradient over a 9x8 grid. Returns 64-bit int."""
    pixels = _to_gray_array(image, 9, 8)
    # Compare each pixel with the one to its right -> 8 bits per row.
    bits = pixels[:, 1:] > pixels[:, :-1]
    return _bits_to_int(bits)


def hamming_distance(a: int, b: int) -> int:
    """Number of differing bits between two 64-bit hashes."""
    return int(bin(a ^ b).count("1"))


def hash_from_path(path: str, kind: str = "dhash") -> int:
    """Convenience: load an image path and return its hash of the given kind."""
    with Image.open(path) as image:
        if kind == "dhash":
            return dhash(image)
        if kind == "ahash":
            return ahash(image)
    raise ValueError(f"unknown hash kind: {kind!r}")
