import math
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PHASES = ("early", "mid", "late")

SCRIPT_DEFAULTS = {
    "wan2.1": (
        "AlignCache4Wan2.1/aligncache_generate.py",
        "resolve_wan_aligncache_config",
        {
            "base_thresh": 0.065,
            "warmup_pairs": 12,
            "phase_thresholds": {"early": 0.07475, "mid": 0.065, "late": 0.052},
        },
    ),
    "hunyuan": (
        "AlignCache4HunyuanVideo/aligncache_sample_video.py",
        "resolve_hunyuan_aligncache_config",
        {
            "base_thresh": 0.024,
            "warmup_pairs": 10,
            "phase_thresholds": {"early": 0.0252, "mid": 0.0288, "late": 0.0324},
        },
    ),
    "opensora": (
        "AlignCache4OpenSora-v1.2/aligncache_inference.py",
        "resolve_opensora_aligncache_config",
        {
            "base_thresh": 0.11,
            "warmup_pairs": 8,
            "phase_thresholds": {"early": 0.1045, "mid": 0.1265, "late": 0.11},
        },
    ),
}

RESOLVER_NAMES = {
    "resolve_wan_aligncache_config",
    "resolve_hunyuan_aligncache_config",
    "resolve_opensora_aligncache_config",
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
    source = source.replace("import torch\n", "")
    module = types.ModuleType(name)
    sys.modules[name] = module
    exec(compile(source, str(ROOT / relative_path), "exec"), module.__dict__)
    return module


def _resolve_defaults(module, resolver_name: str) -> dict:
    resolver = getattr(module, resolver_name)
    if resolver_name == "resolve_wan_aligncache_config":
        return resolver(task="t2v-1.3B", method_config=None, sample_steps=50)
    return resolver(None, sample_steps=50)


def test_aligncache_scripts_only_define_their_own_resolver():
    for _model_name, (relative_path, expected_resolver, _expected) in SCRIPT_DEFAULTS.items():
        source = (ROOT / relative_path).read_text(encoding="utf-8")
        for resolver_name in RESOLVER_NAMES - {expected_resolver}:
            assert f"def {resolver_name}(" not in source


def test_aligncache_default_thresholds_match_selected_experiments():
    for model_name, (relative_path, resolver_name, expected) in SCRIPT_DEFAULTS.items():
        module = _load_resolver_module(
            f"aligncache_{model_name.replace('.', '_')}_defaults",
            relative_path,
            start_marker="import types\nfrom collections.abc import Sequence\nfrom typing import Any"
            if model_name == "wan2.1"
            else None,
        )
        resolved = _resolve_defaults(module, resolver_name)
        assert resolved["base_thresh"] == expected["base_thresh"], model_name
        assert resolved["warmup_pairs"] == expected["warmup_pairs"], model_name
        for phase in PHASES:
            assert math.isclose(
                resolved["phase_thresholds"][phase],
                expected["phase_thresholds"][phase],
                rel_tol=0.0,
                abs_tol=1e-12,
            ), (model_name, phase)
