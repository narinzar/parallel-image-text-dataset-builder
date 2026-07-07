"""End-to-end pipeline: ingest -> dedup -> CLIP filter -> shard.

Perceptual hashing is CPU-bound and embarrassingly parallel, so hashing is fanned
out across a ProcessPoolExecutor. The bucketing dedup, CLIP scoring, and shard
writing then run on the surviving set. Throughput (images/sec) is measured over
the full run and reported alongside the dedup rate.

The CLIP step is optional (`run_clip=False`) so the hashing + dedup + shard path
can be exercised without a model download or GPU, which the tests rely on.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Optional

from tqdm import tqdm

from . import phash
from .dedup import find_duplicates
from .shard import write_shards


@dataclass
class PipelineConfig:
    hash_kind: str = "dhash"
    hamming_threshold: int = 5
    bands: int = 8
    num_workers: int = 4
    run_clip: bool = True
    clip_threshold: float = 0.25
    clip_batch_size: int = 64
    samples_per_shard: int = 1000
    shard_prefix: str = "shard"


@dataclass
class PipelineStats:
    num_input: int
    num_after_dedup: int
    num_after_clip: int
    num_sharded: int
    exact_dups: int
    near_dups: int
    dedup_rate: float
    num_shards: int
    num_workers: int
    hash_kind: str
    elapsed_sec: float
    images_per_sec: float
    band_pairs_checked: int
    clip_ran: bool
    extra: dict = field(default_factory=dict)


def _hash_one(args: tuple[str, str]) -> int:
    """Worker: hash a single image path. `args` is (path, kind)."""
    path, kind = args
    return phash.hash_from_path(path, kind=kind)


def hash_all(
    image_paths: list[str], kind: str, num_workers: int
) -> list[int]:
    """Hash every image path, parallelised across a process pool when workers > 1."""
    tasks = [(p, kind) for p in image_paths]
    if num_workers <= 1:
        return [_hash_one(t) for t in tqdm(tasks, desc=f"Hashing ({kind})")]

    hashes: list[int] = [0] * len(tasks)
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        # chunksize keeps per-task IPC overhead low for many small images.
        chunk = max(1, len(tasks) // (num_workers * 4))
        for idx, h in enumerate(
            tqdm(
                pool.map(_hash_one, tasks, chunksize=chunk),
                total=len(tasks),
                desc=f"Hashing ({kind}, {num_workers}w)",
            )
        ):
            hashes[idx] = h
    return hashes


def run_pipeline(
    image_paths: list[str],
    captions: list[str],
    out_dir: str,
    cfg: Optional[PipelineConfig] = None,
    clip_backend=None,
) -> PipelineStats:
    """Run the full pipeline and return timing + dedup statistics.

    Args:
        image_paths: input image file paths.
        captions: caption per image, aligned with image_paths.
        out_dir: directory for shards and shard_index.json.
        cfg: pipeline configuration.
        clip_backend: optional preloaded CLIP backend (injected in tests / reuse).
    """
    cfg = cfg or PipelineConfig()
    if len(image_paths) != len(captions):
        raise ValueError("image_paths and captions must be the same length")

    start = time.perf_counter()
    n_input = len(image_paths)

    # 1. Hash (parallel).
    hashes = hash_all(image_paths, cfg.hash_kind, cfg.num_workers)

    # 2. Dedup via LSH-style bucketing.
    dedup = find_duplicates(
        hashes,
        hamming_threshold=cfg.hamming_threshold,
        bands=cfg.bands,
    )
    kept_idx = dedup.kept
    kept_paths = [image_paths[i] for i in kept_idx]
    kept_caps = [captions[i] for i in kept_idx]

    # 3. CLIP filter (optional).
    clip_ran = False
    clip_scores: list[float] = []
    if cfg.run_clip:
        from .clip_filter import ClipConfig, filter_pairs

        clip_cfg = ClipConfig(
            batch_size=cfg.clip_batch_size, score_threshold=cfg.clip_threshold
        )
        keep_positions, clip_scores = filter_pairs(
            kept_paths, kept_caps, cfg=clip_cfg, backend=clip_backend
        )
        kept_paths = [kept_paths[i] for i in keep_positions]
        kept_caps = [kept_caps[i] for i in keep_positions]
        clip_ran = True
    n_after_clip = len(kept_paths)

    # 4. Shard.
    samples = list(zip(kept_paths, kept_caps))
    shard_stats = write_shards(
        samples,
        out_dir,
        samples_per_shard=cfg.samples_per_shard,
        prefix=cfg.shard_prefix,
    )

    elapsed = time.perf_counter() - start
    images_per_sec = n_input / elapsed if elapsed > 0 else 0.0

    stats = PipelineStats(
        num_input=n_input,
        num_after_dedup=len(kept_idx),
        num_after_clip=n_after_clip,
        num_sharded=shard_stats.num_samples,
        exact_dups=dedup.exact_dups,
        near_dups=dedup.near_dups,
        dedup_rate=dedup.dedup_rate,
        num_shards=shard_stats.num_shards,
        num_workers=cfg.num_workers,
        hash_kind=cfg.hash_kind,
        elapsed_sec=elapsed,
        images_per_sec=images_per_sec,
        band_pairs_checked=dedup._band_pairs_checked,
        clip_ran=clip_ran,
        extra={
            "clip_threshold": cfg.clip_threshold if clip_ran else None,
            "clip_score_min": min(clip_scores) if clip_scores else None,
            "clip_score_max": max(clip_scores) if clip_scores else None,
        },
    )
    return stats


def save_stats(stats: PipelineStats, path: str) -> None:
    """Write PipelineStats to a JSON file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(stats), f, indent=2)
