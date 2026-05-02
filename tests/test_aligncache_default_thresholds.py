import math
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PHASES = ("early", "mid", "late")

EXPECTED_DEFAULTS = {
    "wan2.1": {
        "base_thresh": 0.065,
        "warmup_pairs": 12,
        "phase_thresholds": {"early": 0.07475, "mid": 0.065, "late": 0.052},
    },
    "hunyuan": {
        "base_thresh": 0.024,
        "warmup_pairs": 10,
        "phase_thresholds": {"early": 0.0252, "mid": 0.0288, "late": 0.0324},
    },
    "opensora": {
        "base_thresh": 0.11,
        "warmup_pairs": 8,
        "phase_thresholds": {"early": 0.1045, "mid": 0.1265, "late": 0.11},
    },
}


def _load_resolver_module(
    name: str,
    relative_path: str,
    *,
    start_marker: str | None = None,
    stop_marker: str = "\ndef _tensor_list(",
):
    source = (ROOT / relative_path).read_text(encoding="utf-8")
    if start_marker is not None:
        source = source[source.index(start_marker) :]
    source = source[: source.index(stop_marker)]
    source = source.replace("import torch\n\n\n_WAN_SUPPORTED_TASKS", "\n\n_WAN_SUPPORTED_TASKS")
    module = types.ModuleType(name)
    sys.modules[name] = module
    exec(compile(source, str(ROOT / relative_path), "exec"), module.__dict__)
    return module


def _resolve_defaults(module, model_name: str) -> dict:
    if model_name == "wan2.1":
        return module.resolve_wan_aligncache_config(task="t2v-1.3B", method_config=None, sample_steps=50)
    if model_name == "hunyuan":
        return module.resolve_hunyuan_aligncache_config(None, sample_steps=50)
    if model_name == "opensora":
        return module.resolve_opensora_aligncache_config(None, sample_steps=50)
    raise AssertionError(f"Unhandled model: {model_name}")


def test_aligncache_default_thresholds_match_selected_experiments():
    modules = {
        "wan_script": _load_resolver_module(
            "aligncache_wan_defaults",
            "AlignCache4Wan2.1/aligncache_generate.py",
            start_marker="import types\nfrom collections.abc import Sequence\nfrom typing import Any",
        ),
        "hunyuan_script": _load_resolver_module(
            "aligncache_hunyuan_defaults",
            "AlignCache4HunyuanVideo/aligncache_sample_video.py",
        ),
        "opensora_script": _load_resolver_module(
            "aligncache_opensora_defaults",
            "AlignCache4OpenSora-v1.2/aligncache_inference.py",
        ),
    }

    for script_name, module in modules.items():
        for model_name, expected in EXPECTED_DEFAULTS.items():
            resolved = _resolve_defaults(module, model_name)
            assert resolved["base_thresh"] == expected["base_thresh"], (script_name, model_name)
            assert resolved["warmup_pairs"] == expected["warmup_pairs"], (script_name, model_name)
            for phase in PHASES:
                assert math.isclose(
                    resolved["phase_thresholds"][phase],
                    expected["phase_thresholds"][phase],
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ), (script_name, model_name, phase)
