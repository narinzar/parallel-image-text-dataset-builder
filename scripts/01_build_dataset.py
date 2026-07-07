"""Run the full pipeline over data/pairs.jsonl and write outputs/stats.json.

Reads the manifest produced by 00_make_sample_pairs.py, runs
ingest -> dedup -> (optional) CLIP filter -> shard, prints a summary, and saves
statistics (images/sec, dedup rate, shard counts) to outputs/stats.json.

Examples:
    # hashing + dedup + shard only (no model download / GPU needed):
    python scripts/01_build_dataset.py --workers 8 --no-clip

    # full pipeline including CLIP score filtering:
    python scripts/01_build_dataset.py --workers 8 --clip-threshold 0.22
"""

from __future__ import annotations

import argparse
import json
import os

from dotenv import load_dotenv

from src.pipeline import PipelineConfig, run_pipeline, save_stats

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
OUTPUTS_DIR = os.path.join(REPO_ROOT, "outputs")


def load_manifest(path: str) -> tuple[list[str], list[str]]:
    image_paths: list[str] = []
    captions: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            image_paths.append(os.path.join(IMAGES_DIR, row["image"]))
            captions.append(row["caption"])
    return image_paths, captions


def main() -> None:
    load_dotenv()  # picks up optional HF_TOKEN from .env

    parser = argparse.ArgumentParser(description="Build the sharded dataset.")
    parser.add_argument("--workers", type=int, default=4,
                        help="process-pool workers for hashing")
    parser.add_argument("--hash-kind", choices=["dhash", "ahash"], default="dhash")
    parser.add_argument("--hamming-threshold", type=int, default=5)
    parser.add_argument("--bands", type=int, default=8)
    parser.add_argument("--no-clip", action="store_true",
                        help="skip CLIP scoring (no model / GPU needed)")
    parser.add_argument("--clip-threshold", type=float, default=0.25)
    parser.add_argument("--clip-batch-size", type=int, default=64)
    parser.add_argument("--samples-per-shard", type=int, default=1000)
    parser.add_argument("--shards-dir", default=os.path.join(OUTPUTS_DIR, "shards"))
    parser.add_argument("--stats-path", default=os.path.join(OUTPUTS_DIR, "stats.json"))
    args = parser.parse_args()

    manifest = os.path.join(DATA_DIR, "pairs.jsonl")
    if not os.path.exists(manifest):
        raise SystemExit(
            f"manifest not found: {manifest}\n"
            "Run scripts/00_make_sample_pairs.py first."
        )

    image_paths, captions = load_manifest(manifest)
    print(f"Loaded {len(image_paths)} pairs from {manifest}")

    cfg = PipelineConfig(
        hash_kind=args.hash_kind,
        hamming_threshold=args.hamming_threshold,
        bands=args.bands,
        num_workers=args.workers,
        run_clip=not args.no_clip,
        clip_threshold=args.clip_threshold,
        clip_batch_size=args.clip_batch_size,
        samples_per_shard=args.samples_per_shard,
    )

    stats = run_pipeline(image_paths, captions, args.shards_dir, cfg=cfg)
    save_stats(stats, args.stats_path)

    print("\n=== Pipeline summary ===")
    print(f"input pairs:        {stats.num_input}")
    print(f"after dedup:        {stats.num_after_dedup}")
    print(f"  exact dups:       {stats.exact_dups}")
    print(f"  near dups:        {stats.near_dups}")
    print(f"  dedup rate:       {stats.dedup_rate:.4f}")
    print(f"after CLIP filter:  {stats.num_after_clip} (clip_ran={stats.clip_ran})")
    print(f"sharded samples:    {stats.num_sharded} in {stats.num_shards} shards")
    print(f"workers:            {stats.num_workers}")
    print(f"elapsed:            {stats.elapsed_sec:.2f}s")
    print(f"images/sec:         {stats.images_per_sec:.1f}")
    print(f"band pairs checked: {stats.band_pairs_checked} "
          f"(all-pairs would be {stats.num_input * (stats.num_input - 1) // 2})")
    print(f"\nStats written to {args.stats_path}")
    print(f"Shards written to {args.shards_dir}")


if __name__ == "__main__":
    main()
