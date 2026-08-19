# AlignCache

EasyCache-style AlignCache runners:

- `AlignCache4Wan2.1/`
- `AlignCache4Wan2.2/`
- `AlignCache4HunyuanVideo/`
- `AlignCache4OpenSora-v1.2/`

## Structure

### `AlignCache4Wan2.1`

- `aligncache_generate.py`
- `README.md`

### `AlignCache4Wan2.2`

- `aligncache_generate.py`
- `README.md`

### `AlignCache4HunyuanVideo`

- `aligncache_sample_video.py`
- `README.md`

### `AlignCache4OpenSora-v1.2`

- `aligncache_inference.py`
- `configs/opensora-v1-2/inference/sample.py`
- `README.md`

## Checkpoints

Model weights are not included. See each model directory README for the expected
checkpoint layout and environment variables.

`AlignCache4OpenSora-v1.2` intentionally does not include the upstream
`opensora/` package. Install or add an external OpenSora v1.2 checkout to
`PYTHONPATH`; see `AlignCache4OpenSora-v1.2/README.md` for the exact setup and
local-checkpoint source patch notes.

## Implementation note

The public `aligncache` method is the feedback-enabled AlignCache variant.
Defaults follow the corresponding experiment configs; incremental configs are
used when available.

See each model directory README for concrete commands.
