# AlignCache for OpenSora v1.2

This directory contains a self-contained OpenSora v1.2 AlignCache runner:

- `aligncache_inference.py`
- `opensora/`
- `configs/opensora-v1-2/inference/sample.py`

The script prioritizes the vendored `opensora/` package next to it, so it does
not require any external OpenSora package checkout.

## Checkpoints

Model weights are not included in this repository. By default, the bundled
config looks for checkpoints under:

```text
AlignCache4OpenSora-v1.2/
  checkpoints/
    OpenSora-STDiT-v3/
    OpenSora-VAE-v1.2/
    t5-v1_1-xxl/
    pixart_sigma_sdxlvae_T5_diffusers/
```

You may store the weights elsewhere and point the script to them with
environment variables:

```bash
export ALIGNCACHE_CHECKPOINTS_DIR=/path/to/checkpoints
```

The checkpoint root should contain the four directories listed above. You can
also override individual paths:

```bash
export ALIGNCACHE_OPENSORA_MODEL_PATH=/path/to/OpenSora-STDiT-v3
export ALIGNCACHE_OPENSORA_VAE_PATH=/path/to/OpenSora-VAE-v1.2
export ALIGNCACHE_OPENSORA_T5_PATH=/path/to/t5-v1_1-xxl
export ALIGNCACHE_OPENSORA_PIXART_VAE_PATH=/path/to/pixart_sigma_sdxlvae_T5_diffusers
```

The expected sources are:

- `hpcai-tech/OpenSora-STDiT-v3`
- `hpcai-tech/OpenSora-VAE-v1.2`
- `DeepFloyd/t5-v1_1-xxl`
- `PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers`

If all weights are available locally, offline mode is recommended:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

## Usage

Run the script from the repository root or by absolute path:

```bash
python AlignCache4OpenSora-v1.2/aligncache_inference.py \
  AlignCache4OpenSora-v1.2/configs/opensora-v1-2/inference/sample.py \
  --ckpt-path "$ALIGNCACHE_OPENSORA_MODEL_PATH" \
  --save-dir ./samples \
  --prompt "A cinematic drone shot of waves crashing against black volcanic rocks."
```

If you use `ALIGNCACHE_CHECKPOINTS_DIR` instead of individual path variables,
pass the model directory explicitly:

```bash
python AlignCache4OpenSora-v1.2/aligncache_inference.py \
  AlignCache4OpenSora-v1.2/configs/opensora-v1-2/inference/sample.py \
  --ckpt-path "$ALIGNCACHE_CHECKPOINTS_DIR/OpenSora-STDiT-v3" \
  --save-dir ./samples \
  --prompt "A cinematic drone shot of waves crashing against black volcanic rocks."
```

Older checkpoint environment variable names are still accepted for
compatibility, but they are not required.

## Optional AlignCache arguments

Common controls:

- `--base_thresh`
- `--warmup_pairs`
- `--phase_scale_profile`
- `--fb_ratio_ema_decay`
- `--fb_ratio_clip_low`
- `--fb_ratio_clip_high`

V2 pair controls:

- `--pair_enable`
- `--pair_aggr`
- `--pair_second_order`
- `--pair_online_scale`
- `--pair_stride`
- `--pair_ratio_ema`
- `--pair_dir_ema`
- `--pair_beta_mid`
- `--pair_beta_late`
- `--pair_ratio_tol_mid`
- `--pair_ratio_tol_late`
- `--pair_dir_min_mid`
- `--pair_dir_min_late`
- `--pair_scale_ema`
- `--pair_scale_low`
- `--pair_scale_high`

B2 anchor controls:

- `--anchor_enable`
- `--anchor_exact_early`
- `--anchor_age_mid`
- `--anchor_age_late`
- `--anchor_cond_cap_mid`
- `--anchor_cond_cap_late`
- `--anchor_alpha_early`
- `--anchor_alpha_mid`
- `--anchor_alpha_late`
- `--anchor_alpha_ema`
- `--anchor_blend_mid`
- `--anchor_blend_late`
- `--anchor_diff_cap_mid`
- `--anchor_diff_cap_late`

Legacy hyphenated names such as `--base-thresh` and `--pair-enable` are still
accepted.

Defaults: `--base_thresh 0.11`, `--warmup_pairs 8`, OpenSora phase scaling, and
pred-sigma auxiliary channel handling.
