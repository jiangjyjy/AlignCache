# AlignCache

This repository reorganizes AlignCache applications in an EasyCache-style layout:

- `AlignCache4Wan2.1/`
- `AlignCache4HunyuanVideo/`
- `AlignCache4OpenSora-v1.2/`

The goal is to keep each model family in its own directory with a single standalone AlignCache script, matching the per-model organization style used by EasyCache.

## Structure

### `AlignCache4Wan2.1`

- `aligncache_generate.py`
- `README.md`

### `AlignCache4HunyuanVideo`

- `aligncache_sample_video.py`
- `README.md`

### `AlignCache4OpenSora-v1.2`

- `aligncache_inference.py`
- `README.md`

## Usage pattern

This repository follows the same broad organization as EasyCache: one directory per model family, each with its own README and one standalone AlignCache entry script. In this repository, the public `aligncache` method corresponds to the feedback-enabled AlignCache variant, but it is exposed simply as `aligncache`.

See each model directory README for concrete commands.
