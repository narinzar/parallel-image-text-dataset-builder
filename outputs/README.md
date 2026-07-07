# outputs/

Pipeline results. Nothing here is committed except this file.

After running `scripts/01_build_dataset.py` you will have:

```
outputs/
  shards/
    shard-00000.tar      webdataset-style tar: {key}.jpg + {key}.txt per sample
    shard-00001.tar
    ...
    shard_index.json     per-shard sample counts
  stats.json             images/sec, dedup rate, exact/near dup counts, shard totals
```

Each `.tar` shard holds up to `--samples-per-shard` samples. `stats.json` is the
machine-readable summary of the run.
