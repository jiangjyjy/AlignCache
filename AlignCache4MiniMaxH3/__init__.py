"""ComfyUI entrypoint for MiniMax-H3 trace and closed-loop acceleration.

Copy this whole directory into ComfyUI's ``custom_nodes/AlignCacheMiniMaxH3Trace``
directory. The P0 collector remains in ``aligncache_generate.py`` and the
``on`` route shares the Diffusers controller in ``aligncache_accelerate.py``.
"""

from .comfyui_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
