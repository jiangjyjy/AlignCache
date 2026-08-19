# AlignCache for Wan2.2

This directory keeps Wan2.2 in the same EasyCache-style per-model layout, with one standalone script:

- `aligncache_generate.py`

## Supported Tasks

- `t2v-A14B`
- `i2v-A14B`
- `ti2v-5B`

`s2v-14B` and `animate-14B` are intentionally not enabled in this runner because their audio, pose, LoRA, and animation paths need task-specific AlignCache adapters.

## Usage

```bash
python AlignCache4Wan2.2/aligncache_generate.py \
  --task t2v-A14B \
  --size "1280*720" \
  --ckpt_dir ./Wan2.2-T2V-A14B \
  --prompt "A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage." \
  --base_seed 0
```

The runner is intended to be copied into the root of a local Wan2.2 checkout and run from that environment. It preserves AlignCache's warmup/tail exact pairs, feedback-scaled phase thresholds, pair residual predictor, online scale adaptation, and anchor uncond predictor.

Wan2.2 A14B tasks use separate `low_noise_model` and `high_noise_model` DiTs. This runner configures AlignCache independently for each model and counts each model's actual sampling steps using Wan2.2's official timestep boundary rule.

Default AlignCache settings use the Wan2.2 `aligncache1` profile derived from EasyCache's Wan2.1 to Wan2.2 parameter ratios: `t2v-A14B` uses `--base_thresh 0.078`, `i2v-A14B` and `ti2v-5B` use `--base_thresh 0.065`, all tasks use `--warmup_pairs 9`, and phase scaling remains balanced.
