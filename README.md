# AlignCache

This repository reorganizes AlignCache applications in an EasyCache-style layout:

- `AlignCache4Wan2.1/`
- `AlignCache4HunyuanVideo/`
- `AlignCache4OpenSora-v1.2/`

The goal is to keep each model family in its own directory with ready-to-run entry scripts, while a small shared package, `aligncache_runtime/`, carries the extracted reusable runtime logic for HunyuanVideo and OpenSora.

## Structure

### `AlignCache4Wan2.1`

- `aligncache_generate.py`
- `aligncache_generate_fb.py`
- `task_config_utils.py`

### `AlignCache4HunyuanVideo`

- `aligncache_sample_video.py`
- `aligncache_fb_sample_video.py`
- `_aligncache_sample_video_impl.py`
- `README.md`

### `AlignCache4OpenSora-v1.2`

- `aligncache_inference.py`
- `aligncache_fb_inference.py`
- `_aligncache_inference_impl.py`
- `README.md`

### `aligncache_runtime`

Shared runtime helpers extracted from the working VBench implementation:

- `aligncache.py`
- `easycache.py`
- `teacache.py`
- `opensora_accel.py`

## Usage pattern

This repository follows the same broad organization as EasyCache: one directory per model family, with scripts grouped by target model. For Wan2.1 the scripts are self-contained in that directory. For HunyuanVideo and OpenSora v1.2, keep the model directory together with the sibling `aligncache_runtime/` package when copying files into the target official repository.

See each model directory README for concrete commands.
