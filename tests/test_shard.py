"""Tests for the tar shard writer: sizing and exactly-once writing."""

from __future__ import annotations

import os
import tarfile

import numpy as np
import pytest
from PIL import Image

from src.shard import read_shard_keys, write_shards


def _make_samples(tmp_path, n: int):
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    samples = []
    rng = np.random.default_rng(0)
    for i in range(n):
        arr = (rng.random((32, 32, 3)) * 255).astype("uint8")
        path = img_dir / f"im_{i:05d}.jpg"
        Image.fromarray(arr, "RGB").save(path, format="JPEG")
        samples.append((str(path), f"caption number {i}"))
    return samples


def test_shard_sizes_respect_target(tmp_path):
    samples = _make_samples(tmp_path, 25)
    out_dir = tmp_path / "shards"
    stats = write_shards(samples, str(out_dir), samples_per_shard=10)

    # 25 samples / 10 per shard -> shards of 10, 10, 5.
    assert stats.num_shards == 3
    assert stats.shard_counts == [10, 10, 5]
    # Every shard except the last holds exactly the target.
    for count in stats.shard_counts[:-1]:
        assert count == 10
    assert stats.shard_counts[-1] <= 10


def test_all_samples_written_exactly_once(tmp_path):
    samples = _make_samples(tmp_path, 23)
    out_dir = tmp_path / "shards"
    stats = write_shards(samples, str(out_dir), samples_per_shard=7)

    assert stats.num_samples == 23
    assert sum(stats.shard_counts) == 23

    # Collect all keys across shards; each must be unique and appear once.
    all_keys = []
    for path in stats.shard_paths:
        all_keys.extend(read_shard_keys(path))
    assert len(all_keys) == 23
    assert len(set(all_keys)) == 23


def test_each_sample_has_jpg_and_txt(tmp_path):
    samples = _make_samples(tmp_path, 5)
    out_dir = tmp_path / "shards"
    stats = write_shards(samples, str(out_dir), samples_per_shard=100)

    with tarfile.open(stats.shard_paths[0], "r") as tar:
        names = tar.getnames()
    jpgs = [n for n in names if n.endswith(".jpg")]
    txts = [n for n in names if n.endswith(".txt")]
    assert len(jpgs) == 5
    assert len(txts) == 5
    # Keys line up between the image and caption members.
    assert sorted(n[:-4] for n in jpgs) == sorted(n[:-4] for n in txts)


def test_captions_round_trip(tmp_path):
    samples = _make_samples(tmp_path, 3)
    out_dir = tmp_path / "shards"
    stats = write_shards(samples, str(out_dir), samples_per_shard=100)

    recovered = {}
    with tarfile.open(stats.shard_paths[0], "r") as tar:
        for member in tar.getmembers():
            if member.name.endswith(".txt"):
                recovered[member.name[:-4]] = tar.extractfile(member).read().decode()
    # Three distinct captions written -> three distinct recovered strings.
    assert len(set(recovered.values())) == 3


def test_index_file_written(tmp_path):
    samples = _make_samples(tmp_path, 12)
    out_dir = tmp_path / "shards"
    write_shards(samples, str(out_dir), samples_per_shard=5)
    assert os.path.exists(out_dir / "shard_index.json")


def test_zero_samples_per_shard_rejected(tmp_path):
    samples = _make_samples(tmp_path, 2)
    with pytest.raises(ValueError):
        write_shards(samples, str(tmp_path / "s"), samples_per_shard=0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
