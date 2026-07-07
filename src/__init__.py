"""Parallel image-text dataset builder.

Modules:
    phash       - perceptual hashes (dhash, ahash) and hamming distance
    dedup       - LSH-style bucketing for near-duplicate detection
    clip_filter - CLIP image-text similarity filtering
    shard       - fixed-size tar shard writer
    pipeline    - end-to-end orchestration with a process pool
"""

__all__ = ["phash", "dedup", "clip_filter", "shard", "pipeline"]
