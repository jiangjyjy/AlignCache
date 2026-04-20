# AlignCache for Wan2.1

This directory keeps the two Wan2.1 scripts in the same EasyCache-style per-model layout:

- `aligncache_generate.py`: base AlignCache schedule
- `aligncache_generate_fb.py`: AlignCache-FB with online threshold feedback

`task_config_utils.py` is included as the local helper used by both scripts.

## Usage

Base AlignCache:

```bash
python AlignCache4Wan2.1/aligncache_generate.py \
  --task t2v-14B \
  --size "1280*720" \
  --ckpt_dir ./Wan2.1-T2V-14B \
  --prompt "A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage." \
  --base_seed 0
```

AlignCache-FB:

```bash
python AlignCache4Wan2.1/aligncache_generate_fb.py \
  --task t2v-14B \
  --size "1280*720" \
  --ckpt_dir ./Wan2.1-T2V-14B \
  --prompt "A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage." \
  --base_seed 0
```

The Wan2.1 scripts are intended to be run from a Wan2.1 environment after copying this directory into the root of the local Wan2.1 checkout.
