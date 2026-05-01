# AlignCache for OpenSora v1.2

This directory contains the OpenSora v1.2 AlignCache runner without vendoring
the upstream OpenSora source tree:

- `aligncache_inference.py`
- `configs/opensora-v1-2/inference/sample.py`

For anonymous review, the upstream `opensora/` package is intentionally not
included in this repository. Use an external OpenSora v1.2 checkout or
installation instead.

## OpenSora v1.2 dependency

Clone or install OpenSora v1.2 outside this anonymized repository:

```bash
git clone https://github.com/XXX/Open-Sora.git
cd Open-Sora
git checkout v1.2.0
pip install -e .
```

If you do not install it as a package, put the external checkout on
`PYTHONPATH` before running this script:

```bash
export PYTHONPATH=/path/to/Open-Sora:$PYTHONPATH
```

Do not copy the upstream `opensora/` directory into this repository for an
anonymous submission.

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

## OpenSora source changes for local checkpoints

Stock OpenSora v1.2 works when the model sources are loaded from Hugging Face
repo IDs or from OpenSora sharded checkpoint directories. If you must run fully
offline from local Hugging Face snapshot directories, apply the following
changes in the external OpenSora v1.2 checkout, not in this repository:

1. In `opensora/models/text_encoder/t5.py`, allow local T5 directories by
   treating an existing filesystem path as a valid `from_pretrained` source
   before the `assert from_pretrained in self.available_models` check.
2. In `opensora/models/vae/vae.py`, resolve the 2D PixArt VAE source from
   `ALIGNCACHE_OPENSORA_PIXART_VAE_PATH` first, then from
   `$ALIGNCACHE_CHECKPOINTS_DIR/pixart_sigma_sdxlvae_T5_diffusers`, and fall
   back to `PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers`.
3. In `opensora/models/stdit/stdit3.py` and `opensora/models/vae/vae.py`,
   distinguish local Hugging Face snapshot directories from OpenSora sharded
   checkpoint directories. Directories with `config.json` or
   `model_index.json` plus standard weight files such as `model.safetensors`
   should be loaded with `.from_pretrained(path)`. Plain `.pt`/`.pth` files and
   sharded checkpoint directories should continue to use `load_checkpoint`.
4. If you run OpenSora's original inference entrypoint with this config, update
   `opensora/utils/config_utils.py` so `read_config()` calls
   `Config.fromfile(config_path, lazy_import=False)`. This runner already loads
   configs eagerly, so this step is only needed for the original OpenSora
   scripts.

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

## Optional AlignCache arguments

Common controls:

- `--base_thresh`
- `--warmup_pairs`
- `--phase_scale_profile`
- `--fb_ratio_ema_decay`
- `--fb_ratio_clip_low`
- `--fb_ratio_clip_high`

Pair predictor controls:

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

Anchor controls:

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
