"""Tests for LSH-style bucketing dedup: grouping and dedup-rate correctness."""

from __future__ import annotations

import pytest

from src.dedup import _band_slices, find_duplicates


def test_band_slices_cover_all_bits_without_overlap():
    slices = _band_slices(64, 8)
    assert len(slices) == 8
    # Reconstruct the full 64-bit mask from every band; should be all ones.
    full = 0
    for shift, mask in slices:
        full |= mask << shift
    assert full == (1 << 64) - 1


def test_exact_duplicates_detected():
    # Three copies of hash A, two of hash B, one unique C.
    a, b, c = 0xDEADBEEF, 0x0123456789ABCDEF, 0xFFFF0000FFFF0000
    hashes = [a, a, a, b, b, c]
    result = find_duplicates(hashes, hamming_threshold=0, bands=8)
    assert result.total == 6
    assert result.exact_dups == 3  # 2 extra A + 1 extra B removed
    assert result.near_dups == 0
    assert len(result.kept) == 3
    assert result.dedup_rate == pytest.approx(3 / 6)


def test_near_duplicates_grouped_within_threshold():
    base = 0
    # Flip a few low bits to create near-dups within hamming distance 3.
    near1 = base ^ 0b0001
    near2 = base ^ 0b0110
    far = (1 << 63) | (1 << 40) | (1 << 20) | 0xF  # many bits set, far away
    hashes = [base, near1, near2, far]
    result = find_duplicates(hashes, hamming_threshold=3, bands=8)
    # base, near1, near2 collapse to one representative; far stays.
    assert len(result.kept) == 2
    assert result.near_dups == 2
    assert result.exact_dups == 0
    # All near-dups must map to the smallest index (0).
    for dup_idx, rep in result.duplicates.items():
        assert rep == 0


def test_planted_near_dups_cluster_and_rate_is_correct():
    """Known set: 5 base hashes, each with one planted near-dup (distance 2)."""
    bases = [
        0x0000000000000000,
        0x1111111111111111,
        0x2222222222222222,
        0x4444444444444444,
        0x8888888888888888,
    ]
    hashes = []
    for h in bases:
        hashes.append(h)
        hashes.append(h ^ 0b11)  # flip 2 bits -> a near-dup
    # 10 items, 5 unique groups -> expect 5 removed -> rate 0.5.
    result = find_duplicates(hashes, hamming_threshold=4, bands=8)
    assert result.total == 10
    assert len(result.kept) == 5
    assert result.near_dups == 5
    assert result.exact_dups == 0
    assert result.dedup_rate == pytest.approx(0.5)


def test_bucketing_checks_far_fewer_pairs_than_all_pairs():
    """With mostly-unique hashes, within-bucket pairs << all-pairs count."""
    # 200 well-separated hashes: spread bits so bands rarely collide.
    hashes = [(i * 0x9E3779B97F4A7C15) & ((1 << 64) - 1) for i in range(200)]
    result = find_duplicates(hashes, hamming_threshold=3, bands=8)
    all_pairs = 200 * 199 // 2
    assert result._band_pairs_checked < all_pairs
    # Almost everything should survive as unique.
    assert len(result.kept) >= 190


def test_empty_input():
    result = find_duplicates([], hamming_threshold=5, bands=8)
    assert result.total == 0
    assert result.dedup_rate == 0.0
    assert result.kept == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
