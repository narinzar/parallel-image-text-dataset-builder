# parallel-image-text-dataset-builder

A parallel pipeline that ingests image-text pairs, removes near-duplicates with
perceptual-hash bucketing, filters weak pairs by CLIP image-text similarity, and
writes fixed-size tar shards for training. It reports images/sec and the dedup
rate.

## Problem

Web-scraped image-text corpora are large, noisy, and full of duplicates: the same
picture reappears resized, recompressed, or lightly cropped, and many captions
barely describe their image. Training on that data wastes compute and biases the
model toward whatever content was duplicated most. Cleaning it is non-trivial
because exact-match dedup misses near-duplicates, and the obvious near-dup fix
(compare every pair) is O(n^2) and does not survive past a few tens of thousands
of images. The pipeline has to dedup approximately but cheaply, judge alignment
without hand labels, and stream results in a training-friendly layout.

## Approach

- Perceptual hashing written from scratch (`src/phash.py`): dhash and ahash over
  8x8 / 9x8 grayscale downscales, each packed into a 64-bit int, plus a hamming
  distance. dhash encodes horizontal gradients so it survives mild resize and JPEG
  recompression.
- LSH-style bucketing for near-dup detection (`src/dedup.py`): split each hash into
  contiguous bands and only compare images that share a band exactly. By the
  pigeonhole principle any pair within `hamming_threshold` bits collides in some
  band whenever the threshold is below the band count, so recall is guaranteed
  while the work stays proportional to within-bucket pairs instead of all pairs. A
  union-find collapses clusters to a single representative.
- CLIP score filtering (`src/clip_filter.py`): batched cosine similarity between
  image and text embeddings (open_clip preferred, transformers CLIPModel fallback),
  dropping pairs below a threshold. GPU is used automatically when available.
- Fixed-size tar sharding (`src/shard.py`): webdataset-style shards with
  `{key}.jpg` + `{key}.txt` members, each shard filled to a target sample count,
  with a `shard_index.json`.
- Parallel orchestration (`src/pipeline.py`): hashing is fanned out over a
  `ProcessPoolExecutor` (CPU-bound, embarrassingly parallel), then dedup, CLIP,
  and sharding run on survivors; the run reports images/sec and dedup rate.

## Setup

```bash
# create and activate a virtual environment
uv venv --python 3.12 .venv        # or: python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# install torch from the CUDA 12.8 index (RTX 5090 / sm_120), then the rest
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# optional env file (HF_TOKEN placeholder only)
cp .env.example .env
```

## How to run

```bash
# 1. generate synthetic pairs with planted near-duplicates
python scripts/00_make_sample_pairs.py --num-base 1500 --dup-frac 0.3

# 2a. hashing + dedup + shard only (no model download or GPU needed)
python scripts/01_build_dataset.py --workers 8 --no-clip

# 2b. full pipeline including CLIP score filtering
python scripts/01_build_dataset.py --workers 8 --clip-threshold 0.22

# tests
pytest -q
```

Useful flags for `01_build_dataset.py`: `--hash-kind {dhash,ahash}`,
`--hamming-threshold`, `--bands`, `--samples-per-shard`, `--workers`.

## Results

Numbers below are produced by running the commands above; this repo ships the
code, run it to populate them.

Reproduction commands and the behavior to expect:

- Dedup finds planted near-dups without the all-pairs blowup. After step 1,
  `data/pairs.jsonl` contains base images plus resized/recompressed copies. Step 2
  prints `band pairs checked` next to the all-pairs count; the bucketed count
  should be far smaller, and the reported `dedup rate` should be close to the
  planted duplicate fraction (`dup-frac`). Compare hash kinds with
  `--hash-kind dhash` vs `--hash-kind ahash`.
- CLIP filtering drops low-alignment pairs. Running with `--clip-threshold` set
  higher removes more pairs (`after CLIP filter` falls); running with `--no-clip`
  keeps every deduped pair. The synthetic captions are only loosely tied to the
  images, so the survival rate is threshold-sensitive by design.
- Throughput scales with workers. Re-run step 2 with `--workers 1`, `2`, `4`, `8`
  and read `images/sec`. It should rise roughly with worker count up to the CPU
  core count, then plateau (and dip past it as processes contend).

| run | command | images/sec | dedup rate | shards |
| --- | ------- | ---------- | ---------- | ------ |
| dedup only, 1 worker  | `01_build_dataset.py --workers 1 --no-clip` | TBD (run) | TBD (run) | TBD (run) |
| dedup only, 8 workers | `01_build_dataset.py --workers 8 --no-clip` | TBD (run) | TBD (run) | TBD (run) |
| full pipeline         | `01_build_dataset.py --workers 8`           | TBD (run) | TBD (run) | TBD (run) |

Machine-readable results land in `outputs/stats.json`.

## What I'd do next at larger scale

Move ingest to sharded streaming so images never all sit in memory, and hash
directly off the input tar/parquet shards with the process pool feeding a bounded
queue. Replace the single-machine union-find with a partitioned banded index (one
reducer per band) so dedup runs across nodes, and swap the CLIP step for a
persistent GPU worker pool that batches across shards to keep the accelerator saturated.
