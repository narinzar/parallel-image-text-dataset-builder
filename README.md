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
python scripts/00_make_sample_pairs.py --num-base 3000 --dup-frac 0.3

# 2a. hashing + dedup + shard only (no model download or GPU needed)
python scripts/01_build_dataset.py --workers 8 --no-clip

# 2b. full pipeline including CLIP score filtering
python scripts/01_build_dataset.py --workers 8 --clip-threshold 0.10

# tests
pytest -q
```

Useful flags for `01_build_dataset.py`: `--hash-kind {dhash,ahash}`,
`--hamming-threshold`, `--bands`, `--samples-per-shard`, `--workers`.

## Results

Measured on a single RTX 5090 (24 GB), open_clip `ViT-B-32` `laion2b_s34b_b79k`
on GPU, hashing on a Windows CPU process pool. Input: 3840 synthetic pairs (3000
unique base images + 840 planted near-duplicates) at `--num-base 3000
--dup-frac 0.3`. This is a small-scale run meant to exercise the full path end to
end; the numbers are real measurements from that run, not a large-corpus
benchmark.

| run | command | images/sec | dedup rate | shards |
| --- | ------- | ---------- | ---------- | ------ |
| dedup only, 1 worker  | `01_build_dataset.py --workers 1 --no-clip`      | 1970.7 | 0.2188 | 3 |
| dedup only, 8 workers | `01_build_dataset.py --workers 8 --no-clip`      | 1921.1 | 0.2188 | 3 |
| full pipeline         | `01_build_dataset.py --workers 8 --clip-threshold 0.10` | 259.0 | 0.2188 | 3 |

What the run shows:

- Dedup finds exactly the planted near-dups without the all-pairs blowup. Of the
  3840 inputs, dedup removed 840 (637 collapsed to a bit-identical dhash and were
  counted as exact dups, 203 as near dups within the 5-bit hamming threshold),
  leaving all 3000 base images. The `dedup rate` of 0.2188 equals the planted
  duplicate fraction (840 / 3840) exactly. Bucketing checked 556,812 band pairs
  versus 7,370,880 for all-pairs, about 13x less work, and recall stayed complete.
- CLIP filtering drops low-alignment pairs. At `--clip-threshold 0.10`, 2536 of
  the 3000 deduped pairs survived (scores ranged 0.036 to 0.318). The synthetic
  captions are only loosely tied to the abstract images, so absolute CLIP scores
  are low and the survival rate is threshold-sensitive by design; raising the
  threshold removes more pairs, `--no-clip` keeps all 3000.
- Throughput here does not rise with workers: at 128x128 images the per-image
  dhash is so cheap that Windows process-spawn and IPC overhead dominates, so 8
  workers (1921 img/s) is marginally slower than 1 (1971 img/s). The full pipeline
  drops to 259 img/s because it is bounded by GPU CLIP inference plus image
  decode, not by hashing. Parallel hashing pays off once images are larger or the
  corpus is big enough to amortize pool startup.

Machine-readable results land in `outputs/stats.json`.

## What I'd do next at larger scale

Move ingest to sharded streaming so images never all sit in memory, and hash
directly off the input tar/parquet shards with the process pool feeding a bounded
queue. Replace the single-machine union-find with a partitioned banded index (one
reducer per band) so dedup runs across nodes, and swap the CLIP step for a
persistent GPU worker pool that batches across shards to keep the accelerator saturated.
