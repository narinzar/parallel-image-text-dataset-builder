"""LSH-style near-duplicate detection over 64-bit perceptual hashes.

Naive dedup compares every pair of images, which is O(n^2) and does not scale:
a million images would need ~5 * 10^11 comparisons.

The bucketing strategy here splits each 64-bit hash into `bands` contiguous
slices of equal width. Two images that are near-duplicates (small hamming
distance) must, by pigeonhole, share at least one band exactly whenever the
hamming distance is smaller than the number of bands. So we:

    1. group images by (band_index, band_value) into buckets,
    2. only compare pairs that land in the same bucket,
    3. union near-duplicates within a hamming threshold.

This turns the all-pairs blowup into work proportional to the number of within
-bucket pairs, which stays small for realistic data because most buckets hold a
single image. It is an approximate method: it can miss a pair whose distance is
at least the band count and whose bands never collide, so `bands` is chosen so
that `hamming_threshold < bands` to guarantee the planted near-duplicates in
this project are always caught.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

from .phash import HASH_BITS, hamming_distance


@dataclass
class DedupResult:
    """Outcome of a dedup pass.

    kept        - indices of the representative (surviving) items
    duplicates  - mapping of duplicate index -> representative index it maps to
    exact_dups  - count of items removed as exact hash matches
    near_dups   - count of items removed as near (non-exact) matches
    total       - number of input items
    """

    kept: list[int]
    duplicates: dict[int, int]
    exact_dups: int
    near_dups: int
    total: int
    _band_pairs_checked: int = field(default=0, repr=False)

    @property
    def removed(self) -> int:
        return self.exact_dups + self.near_dups

    @property
    def dedup_rate(self) -> float:
        """Fraction of input items removed as duplicates."""
        if self.total == 0:
            return 0.0
        return self.removed / self.total


def _band_slices(hash_bits: int, bands: int) -> list[tuple[int, int]]:
    """Return (shift, mask) for each band so band value = (h >> shift) & mask."""
    if hash_bits % bands != 0:
        raise ValueError(f"bands={bands} must divide hash_bits={hash_bits}")
    width = hash_bits // bands
    mask = (1 << width) - 1
    return [(i * width, mask) for i in range(bands)]


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        # path compression
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        # Keep the smaller index as the representative so the first-seen item wins.
        lo, hi = (ra, rb) if ra < rb else (rb, ra)
        self.parent[hi] = lo


def find_duplicates(
    hashes: list[int],
    hamming_threshold: int = 5,
    bands: int = 8,
    hash_bits: int = HASH_BITS,
) -> DedupResult:
    """Bucket hashes into bands and flag near-duplicates within the threshold.

    Args:
        hashes: 64-bit perceptual hashes, one per item, in input order.
        hamming_threshold: max bit distance for two items to be "near duplicates".
        bands: number of prefix/segment bands. Must divide hash_bits and, to
            guarantee recall, should exceed hamming_threshold.
        hash_bits: hash width in bits (64 for dhash/ahash here).

    Returns:
        DedupResult with kept representatives, duplicate mapping, and counts.
    """
    n = len(hashes)
    if hamming_threshold >= bands:
        # Not fatal, but recall is no longer guaranteed by pigeonhole.
        # Callers in this project keep threshold < bands.
        pass

    slices = _band_slices(hash_bits, bands)
    uf = _UnionFind(n)

    # Build candidate buckets: key = (band_index, band_value).
    buckets: dict[tuple[int, int], list[int]] = {}
    for idx, h in enumerate(hashes):
        for band_idx, (shift, mask) in enumerate(slices):
            key = (band_idx, (h >> shift) & mask)
            buckets.setdefault(key, []).append(idx)

    # Compare only within-bucket pairs, deduped across bands with a seen set.
    checked: set[tuple[int, int]] = set()
    pairs_checked = 0
    for members in buckets.values():
        if len(members) < 2:
            continue
        for a, b in combinations(members, 2):
            pair = (a, b) if a < b else (b, a)
            if pair in checked:
                continue
            checked.add(pair)
            pairs_checked += 1
            if hamming_distance(hashes[a], hashes[b]) <= hamming_threshold:
                uf.union(a, b)

    # Resolve clusters. Representative = smallest index in each cluster.
    duplicates: dict[int, int] = {}
    kept: list[int] = []
    exact_dups = 0
    near_dups = 0
    for idx in range(n):
        rep = uf.find(idx)
        if rep == idx:
            kept.append(idx)
        else:
            duplicates[idx] = rep
            if hashes[idx] == hashes[rep]:
                exact_dups += 1
            else:
                near_dups += 1

    return DedupResult(
        kept=kept,
        duplicates=duplicates,
        exact_dups=exact_dups,
        near_dups=near_dups,
        total=n,
        _band_pairs_checked=pairs_checked,
    )
