# AlignCache for OpenSora v1.2

This directory adds OpenSora v1.2 scripts in the same per-model style used by the EasyCache repository:

- `aligncache_inference.py`: base AlignCache schedule
- `aligncache_fb_inference.py`: AlignCache-FB with online threshold feedback

## Layout

Keep this folder together with the sibling `aligncache_runtime/` package from the repository root.

A straightforward setup is:

1. Clone this repository.
2. Copy `AlignCache4OpenSora-v1.2/` and `aligncache_runtime/` into the root of your local OpenSora v1.2 checkout.
3. Run the scripts from that OpenSora root.

## Usage

Base AlignCache:

```bash
python AlignCache4OpenSora-v1.2/aligncache_inference.py configs/opensora-v1-2/inference/sample.py \
  --ckpt-path /path/to/weights \
  --save-dir ./samples \
  --prompt "A cinematic drone shot of waves crashing against black volcanic rocks."
```

AlignCache-FB:

```bash
python AlignCache4OpenSora-v1.2/aligncache_fb_inference.py configs/opensora-v1-2/inference/sample.py \
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

`aligncache_fb_inference.py` also accepts:

- `--fb-ratio-ema-decay`
- `--fb-ratio-clip-low`
- `--fb-ratio-clip-high`
