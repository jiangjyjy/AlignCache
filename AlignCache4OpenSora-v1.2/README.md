# AlignCache for OpenSora v1.2

This directory adds OpenSora v1.2 in the same per-model style used by the EasyCache repository, with one standalone AlignCache script:

- `aligncache_inference.py`

## Layout

This directory is self-contained for the OpenSora Python package used by
inference:

- `aligncache_inference.py`
- `opensora/`
- `configs/opensora-v1-2/inference/sample.py`

A straightforward setup is:

1. Clone this repository.
2. Point the environment variables below to your local checkpoint files.
3. Run `aligncache_inference.py` directly from this directory or by absolute path.

The script prioritizes the vendored `opensora/` package next to it, so it does
not need the `opensora` package from VBench-exp or a separate OpenSora checkout.

## Usage

```bash
cd /root/autodl-tmp/AlignCache

export VBENCH_CHECKPOINTS_DIR=/root/autodl-tmp/VBench-exp/checkpoints
export VBENCH_OPENSORA_MODEL_PATH=$VBENCH_CHECKPOINTS_DIR/OpenSora-STDiT-v3
export VBENCH_OPENSORA_VAE_PATH=$VBENCH_CHECKPOINTS_DIR/OpenSora-VAE-v1.2
export VBENCH_OPENSORA_T5_PATH=$VBENCH_CHECKPOINTS_DIR/t5-v1_1-xxl
export VBENCH_OPENSORA_PIXART_VAE_PATH=$VBENCH_CHECKPOINTS_DIR/pixart_sigma_sdxlvae_T5_diffusers
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python AlignCache4OpenSora-v1.2/aligncache_inference.py \
  AlignCache4OpenSora-v1.2/configs/opensora-v1-2/inference/sample.py \
  --ckpt-path "$VBENCH_OPENSORA_MODEL_PATH" \
  --save-dir ./samples \
  --prompt "A cinematic drone shot of waves crashing against black volcanic rocks."
```

## Optional AlignCache arguments

The scripts accept a small set of extra AlignCache controls before the normal OpenSora arguments:

- `--base-thresh`
- `--warmup-pairs`
- `--phase-scale-profile`
- `--pair-enable`
- `--pair-aggr`
- `--pair-stride`
- `--anchor-enable`

This `aligncache` script is the feedback-enabled AlignCache variant, renamed to `aligncache` for the repository interface. It also accepts:

- `--ratio-ema-decay`
- `--ratio-clip-low`
- `--ratio-clip-high`

The script uses the OpenSora defaults from the VBench AlignCache backend: `--base-thresh 0.11`, `--warmup-pairs 8`, OpenSora-specific phase scaling, and pred-sigma auxiliary channel handling.
