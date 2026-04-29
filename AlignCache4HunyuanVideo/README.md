# AlignCache for HunyuanVideo

This directory mirrors the per-model layout used by the EasyCache repository and provides one standalone AlignCache script for HunyuanVideo:

- `aligncache_sample_video.py`

## Layout

This script is standalone. A simple setup is:

1. Clone this repository.
2. Copy `AlignCache4HunyuanVideo/` into the root of your local `HunyuanVideo` checkout.
3. Run the scripts from that `HunyuanVideo` root.

## Usage

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

## Optional AlignCache arguments

The scripts accept a small set of extra AlignCache controls before the normal HunyuanVideo arguments:

- `--base-thresh`
- `--warmup-pairs`
- `--force-cfg-pair`
- `--phase-scale-profile`
- `--pair-enable`
- `--pair-aggr`
- `--pair-stride`
- `--anchor-enable`

This `aligncache` script is the feedback-enabled AlignCache variant, renamed to `aligncache` for the repository interface. It also accepts:

- `--ratio-ema-decay`
- `--ratio-clip-low`
- `--ratio-clip-high`

The script uses the HunyuanVideo incremental experiment defaults from VBench-exp: `--base-thresh 0.024`, `--warmup-pairs 10`, `--force-cfg-pair false`, phase scales `{early: 1.05, mid: 1.20, late: 1.35}`, and age caps of 6 for all phases.
