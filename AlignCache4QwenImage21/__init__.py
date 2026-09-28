"""ComfyUI node registration without importing Torch or Comfy at discovery."""

from .benchmark import AlignCacheQwenImage21Benchmark
from .comfyui_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

NODE_CLASS_MAPPINGS["AlignCacheQwenImage21Benchmark"] = AlignCacheQwenImage21Benchmark
NODE_DISPLAY_NAME_MAPPINGS["AlignCacheQwenImage21Benchmark"] = (
    "Qwen-Image-2.1 Benchmark (Measurement Only)"
)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
