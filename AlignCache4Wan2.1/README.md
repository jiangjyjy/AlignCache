# AlignCache for Wan2.1

This directory keeps Wan2.1 in the same EasyCache-style per-model layout, with one standalone script:

- `aligncache_generate.py`

## Usage

```bash
python AlignCache4Wan2.1/aligncache_generate.py \
  --task t2v-1.3B \
  --size "832*480" \
  --ckpt_dir ./Wan2.1-T2V-1.3B \
  --prompt "A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage." \
  --base_seed 0
```

This `aligncache` script is the feedback-enabled AlignCache variant, renamed to `aligncache` for the repository interface. The directory is intended to be copied into the root of a local Wan2.1 checkout and run from that environment.

The implementation matches the VBench Wan2.1 T2V AlignCache backend. It supports the vendored T2V tasks (`t2v-1.3B` and `t2v-14B`) with experiment defaults `--task t2v-1.3B`, `--size 832*480`, `--base_thresh 0.065`, `--warmup_pairs 12`, and balanced phase scaling.
