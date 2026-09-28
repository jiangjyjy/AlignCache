# AlignCache

EasyCache-style AlignCache runners:

- `AlignCache4Wan2.1/`
- `AlignCache4Wan2.2/`
- `AlignCache4HunyuanVideo/`
- `AlignCache4OpenSora-v1.2/`
- [`AlignCache4QwenImage21/`](AlignCache4QwenImage21/README.md): experimental
  ComfyUI Qwen-Image-2.1 node, trace logging, and native contract tests.
  Includes a fixed experimental T2I workflow validated on RTX 3090: median 2.28x sampling speedup and 33.47 dB PSNR on 10 image/seed pairs.
  The paper-adapted flow/CFG policy adds guided-MSE feedback, phase limits and unconditional anchor correction.

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
