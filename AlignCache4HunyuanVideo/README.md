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

For FP8 inference, pass `--use-fp8`. If `--dit-weight` is left at the official default, the script will use `ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states_fp8.pt` under `--model-base` and expects the matching `mp_rank_00_model_states_fp8_map.pt` in the same directory. You can also pass `--dit-weight /path/to/mp_rank_00_model_states_fp8.pt` explicitly.

## Optional AlignCache arguments

The scripts accept a small set of extra AlignCache controls before the normal HunyuanVideo arguments:

- `--base_thresh`
- `--warmup_pairs`
- `--force_cfg_pair`
- `--phase_scale_profile`
- `--pair_enable`
- `--pair_aggr`
- `--pair_stride`
- `--anchor_enable`

This `aligncache` script is the feedback-enabled AlignCache variant, renamed to `aligncache` for the repository interface. It also accepts:

- `--fb_ratio_ema_decay`
- `--fb_ratio_clip_low`
- `--fb_ratio_clip_high`

The script uses the HunyuanVideo incremental experiment defaults: `--base_thresh 0.024`, `--warmup_pairs 10`, `--force_cfg_pair false`, phase scales `{early: 1.05, mid: 1.20, late: 1.35}`, and age caps of 6 for all phases.
