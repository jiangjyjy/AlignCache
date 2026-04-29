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

The script accepts a small set of extra AlignCache controls before the normal
OpenSora arguments:

- `--base-thresh`
- `--warmup-pairs`
- `--phase-scale-profile`
- `--pair-enable`
- `--pair-aggr`
- `--pair-stride`
- `--anchor-enable`
- `--ratio-ema-decay`
- `--ratio-clip-low`
- `--ratio-clip-high`

The default OpenSora AlignCache settings are `--base-thresh 0.11`,
`--warmup-pairs 8`, OpenSora-specific phase scaling, and pred-sigma auxiliary
channel handling.
