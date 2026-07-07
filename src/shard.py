"""Fixed-size shard writer (webdataset-style .tar).

Surviving image-text pairs are written into tar shards using the stdlib tarfile
module. Each sample contributes two members that share a key:

    {key}.jpg   - the image bytes
    {key}.txt   - the caption

A new shard is started once the current one reaches `samples_per_shard`, so every
shard except possibly the last holds exactly the target sample count. This layout
streams well for training and needs no index.
"""

from __future__ import annotations

import io
import json
import os
import tarfile
from dataclasses import dataclass

from PIL import Image
from tqdm import tqdm


@dataclass
class ShardStats:
    num_shards: int
    num_samples: int
    samples_per_shard: int
    shard_paths: list[str]
    shard_counts: list[int]


def _write_member(tar: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(payload)
    tar.addfile(info, io.BytesIO(payload))


def write_shards(
    samples: list[tuple[str, str]],
    out_dir: str,
    samples_per_shard: int = 1000,
    prefix: str = "shard",
    jpeg_quality: int = 95,
) -> ShardStats:
    """Write (image_path, caption) samples into fixed-size tar shards.

    Args:
        samples: list of (image_path, caption) for surviving pairs.
        out_dir: directory to write shards into (created if missing).
        samples_per_shard: target sample count per shard.
        prefix: shard filename prefix; files are prefix-00000.tar, etc.
        jpeg_quality: re-encode quality for image members.

    Returns:
        ShardStats describing the written shards.
    """
    if samples_per_shard <= 0:
        raise ValueError("samples_per_shard must be positive")
    os.makedirs(out_dir, exist_ok=True)

    shard_paths: list[str] = []
    shard_counts: list[int] = []
    tar: tarfile.TarFile | None = None
    in_current = 0
    written = 0

    def _open_new_shard() -> tarfile.TarFile:
        shard_idx = len(shard_paths)
        path = os.path.join(out_dir, f"{prefix}-{shard_idx:05d}.tar")
        shard_paths.append(path)
        shard_counts.append(0)
        return tarfile.open(path, "w")

    try:
        for image_path, caption in tqdm(samples, desc="Sharding"):
            if tar is None or in_current >= samples_per_shard:
                if tar is not None:
                    tar.close()
                tar = _open_new_shard()
                in_current = 0

            key = f"{written:08d}"
            with Image.open(image_path) as im:
                buf = io.BytesIO()
                im.convert("RGB").save(buf, format="JPEG", quality=jpeg_quality)
                image_bytes = buf.getvalue()

            _write_member(tar, f"{key}.jpg", image_bytes)
            _write_member(tar, f"{key}.txt", caption.encode("utf-8"))

            in_current += 1
            written += 1
            shard_counts[-1] += 1
    finally:
        if tar is not None:
            tar.close()

    stats = ShardStats(
        num_shards=len(shard_paths),
        num_samples=written,
        samples_per_shard=samples_per_shard,
        shard_paths=shard_paths,
        shard_counts=shard_counts,
    )
    with open(os.path.join(out_dir, "shard_index.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "num_shards": stats.num_shards,
                "num_samples": stats.num_samples,
                "samples_per_shard": stats.samples_per_shard,
                "shards": [
                    {"path": os.path.basename(p), "count": c}
                    for p, c in zip(stats.shard_paths, stats.shard_counts)
                ],
            },
            f,
            indent=2,
        )
    return stats


def read_shard_keys(shard_path: str) -> list[str]:
    """Return the sample keys stored in a shard (one per .jpg member)."""
    keys: list[str] = []
    with tarfile.open(shard_path, "r") as tar:
        for name in tar.getnames():
            if name.endswith(".jpg"):
                keys.append(name[: -len(".jpg")])
    return keys
