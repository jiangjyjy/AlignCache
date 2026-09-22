"""Run against real ComfyUI; optional checkout supplied by COMFYUI_ROOT.

No stub modules are installed. Tiny random weights test contracts, not quality.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
COMFY = os.environ.get("COMFYUI_ROOT")


@pytest.mark.skipif(not COMFY, reason="Set COMFYUI_ROOT to the locked ComfyUI checkout")
def test_native_wrapper_patcher_options_and_qwen_lifecycle():
    result = subprocess.run(
        [sys.executable, str(ROOT / "tests/qwen21_native_runner.py"), "--cpu"],
        env={**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT), COMFY])},
        text=True,
        capture_output=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
