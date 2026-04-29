# AlignCache for OpenSora v1.2

This directory adds OpenSora v1.2 in the same per-model style used by the EasyCache repository, with one standalone AlignCache script:

- `aligncache_inference.py`

## Layout

This script is standalone.

A straightforward setup is:

1. Clone this repository.
2. Copy `AlignCache4OpenSora-v1.2/` into the root of your local OpenSora v1.2 checkout.
3. Run the scripts from that OpenSora root.

## Usage

```bash
python AlignCache4OpenSora-v1.2/aligncache_inference.py configs/opensora-v1-2/inference/sample.py \
  --ckpt-path /path/to/weights \
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
