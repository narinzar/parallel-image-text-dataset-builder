# data/

Generated input for the pipeline. Nothing here is committed except this file.

After running `scripts/00_make_sample_pairs.py` you will have:

```
data/
  images/          synthetic JPEG images (base images + planted near-duplicates)
  pairs.jsonl      one record per image: {"image", "caption", "group"}
```

`group` is the id of the base image a sample came from, so planted near-duplicates
share a group. To use your own data instead, drop images under `data/images/` and
write a `pairs.jsonl` with the same fields (image is a filename relative to
`data/images/`, caption is the text, group is optional).
