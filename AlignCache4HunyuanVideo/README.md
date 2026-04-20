# AlignCache for HunyuanVideo

This directory mirrors the per-model layout used by the EasyCache repository and provides two entry scripts for HunyuanVideo:

- `aligncache_sample_video.py`: base AlignCache schedule
- `aligncache_fb_sample_video.py`: AlignCache-FB with online threshold feedback

## Layout

Keep this folder together with the sibling `aligncache_runtime/` package from the repository root. A simple setup is:

1. Clone this repository.
2. Copy `AlignCache4HunyuanVideo/` and `aligncache_runtime/` into the root of your local `HunyuanVideo` checkout.
3. Run the scripts from that `HunyuanVideo` root.

## Usage

Base AlignCache:

```bash
python AlignCache4HunyuanVideo/aligncache_sample_video.py \
  --video-size 544 960 \
  --video-length 129 \
  --infer-steps 50 \
  --prompt "A cat walks on the grass, realistic style." \
  --flow-reverse \
  --use-cpu-offload \
  --save-path ./results \
  --seed 42
```

AlignCache-FB:

```bash
python AlignCache4HunyuanVideo/aligncache_fb_sample_video.py \
  --video-size 544 960 \
  --video-length 129 \
  --infer-steps 50 \
  --prompt "A cat walks on the grass, realistic style." \
  --flow-reverse \
  --use-cpu-offload \
  --save-path ./results \
  --seed 42
```

## Optional AlignCache arguments

The scripts accept a small set of extra AlignCache controls before the normal HunyuanVideo arguments:

- `--base-thresh`
- `--warmup-pairs`
- `--phase-scale-profile`
- `--pair-enable`
- `--pair-aggr`
- `--pair-stride`
- `--anchor-enable`

`aligncache_fb_sample_video.py` also accepts:

- `--fb-ratio-ema-decay`
- `--fb-ratio-clip-low`
- `--fb-ratio-clip-high`
