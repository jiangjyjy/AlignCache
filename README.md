# AlignCache

EasyCache-style AlignCache runners:

- `AlignCache4Wan2.1/`
- `AlignCache4HunyuanVideo/`
- `AlignCache4OpenSora-v1.2/`

## Structure

### `AlignCache4Wan2.1`

- `aligncache_generate.py`
- `README.md`

### `AlignCache4HunyuanVideo`

- `aligncache_sample_video.py`
- `README.md`

### `AlignCache4OpenSora-v1.2`

- `aligncache_inference.py`
- `opensora/`
- `configs/opensora-v1-2/inference/sample.py`
- `README.md`

## Checkpoints

Model weights are not included. See each model directory README for the expected
checkpoint layout and environment variables.

`AlignCache4OpenSora-v1.2` includes its own `opensora/` package and config file,
so it does not require an external OpenSora checkout.

## Implementation note

The public `aligncache` method is the feedback-enabled AlignCache variant.
Defaults follow the corresponding experiment configs; incremental configs are
used when available.

See each model directory README for concrete commands.
