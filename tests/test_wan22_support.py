import argparse
import math
import sys
import types
from collections.abc import Iterable
from types import SimpleNamespace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "AlignCache4Wan2.2" / "aligncache_generate.py"


class _FakeTorch:
    class Tensor:
        pass


def _load_wan22_task_module():
    source = SCRIPT.read_text(encoding="utf-8")
    source = source[source.index("KNOWN_WAN_TASKS = ") : source.index("\nfrom wan.configs")]
    module = types.ModuleType("wan22_task_support")
    module.__dict__.update(
        {
            "argparse": argparse,
            "Iterable": Iterable,
            "Path": Path,
        }
    )
    sys.modules[module.__name__] = module
    exec(compile(source, str(SCRIPT), "exec"), module.__dict__)
    return module


def _load_wan22_aligncache_module():
    source = SCRIPT.read_text(encoding="utf-8")
    source = source[source.index("import types\nfrom collections.abc import Sequence\nfrom typing import Any") :]
    source = source[: source.index("\ndef _method_config_from_args(")]
    module = types.ModuleType("wan22_aligncache_support")
    module.__dict__["torch"] = _FakeTorch
    sys.modules[module.__name__] = module
    exec(compile(source, str(SCRIPT), "exec"), module.__dict__)
    return module


def _load_wan22_validation_module():
    raw_source = SCRIPT.read_text(encoding="utf-8")
    source = raw_source[raw_source.index("KNOWN_WAN_TASKS = ") : raw_source.index("\nfrom wan.configs")]
    source += "\n"
    source += raw_source[raw_source.index("EXAMPLE_PROMPT = ") : raw_source.index("\ndef _init_logging(")]
    module = types.ModuleType("wan22_validation_support")
    module.__dict__.update(
        {
            "argparse": argparse,
            "Iterable": Iterable,
            "Path": Path,
            "random": types.SimpleNamespace(randint=lambda _low, _high: 123),
            "sys": types.SimpleNamespace(maxsize=999),
        }
    )
    sys.modules[module.__name__] = module
    exec(compile(source, str(SCRIPT), "exec"), module.__dict__)
    module._normalize_wan22_guide_scale = _load_wan22_aligncache_module()._normalize_wan22_guide_scale
    return module


def _load_wan22_parse_module():
    raw_source = SCRIPT.read_text(encoding="utf-8")
    source = raw_source[raw_source.index("def _parse_args():") : raw_source.index("\ndef _build_prompt_expander(")]
    module = types.ModuleType("wan22_parse_support")
    module.__dict__.update(
        {
            "argparse": argparse,
            "_WAN_SUPPORTED_TASKS": {"t2v-A14B", "i2v-A14B", "ti2v-5B"},
            "parse_task_name_arg": lambda value, allowed_tasks=None: value,
            "SIZE_CONFIGS": {"1280*720": (1280, 720)},
            "str2bool": lambda value: str(value).lower() in {"1", "true", "yes"},
            "FB_DEFAULT_PHASE_SCALES": {"early": 1.15, "mid": 1.0, "late": 0.8},
            "FB_DEFAULTS": {"ratio_ema_decay": 0.90, "ratio_clip_low": 0.85, "ratio_clip_high": 1.25},
            "FB_DEFAULT_AGE_CAP": {"early": 5, "mid": 4, "late": 3},
            "FB_DEFAULT_SCALE_INIT": {"early": 1.0, "mid": 1.0, "late": 1.0},
            "FB_DEFAULT_SCALE_LOW": {"early": 0.95, "mid": 0.90, "late": 0.85},
            "FB_DEFAULT_SCALE_HIGH": {"early": 1.03, "mid": 1.06, "late": 1.0},
            "normalize_ckpt_dir_for_task": lambda task, ckpt_dir, size_text=None: ckpt_dir or "ckpt",
            "_validate_args": lambda args: None,
            "_validate_pair_args": lambda args: None,
            "_validate_fb_args": lambda args: None,
        }
    )
    sys.modules[module.__name__] = module
    exec(compile(source, str(SCRIPT), "exec"), module.__dict__)
    return module


def test_wan22_task_names_are_official_and_case_insensitive():
    module = _load_wan22_task_module()

    assert module.KNOWN_WAN_TASKS == ("t2v-A14B", "i2v-A14B", "ti2v-5B")
    assert module.canonicalize_task_name("T2V-A14B") == "t2v-A14B"
    assert module.canonicalize_task_name("i2v-a14b") == "i2v-A14B"
    assert module.canonicalize_task_name("TI2V-5b") == "ti2v-5B"


def test_wan22_checkpoint_defaults_match_official_model_names():
    module = _load_wan22_task_module()

    expected = {
        "t2v-A14B": "Wan2.2-T2V-A14B",
        "i2v-A14B": "Wan2.2-I2V-A14B",
        "ti2v-5B": "Wan2.2-TI2V-5B",
    }
    for task, dirname in expected.items():
        assert module.checkpoint_dir_name_for_task(task) == dirname
        assert Path(module.default_checkpoint_dir_for_task(task)).name == dirname


def test_wan22_validation_uses_official_sampling_defaults():
    module = _load_wan22_validation_module()
    module.SUPPORTED_SIZES = {
        "t2v-A14B": ("1280*720",),
        "i2v-A14B": ("1280*720",),
        "ti2v-5B": ("1280*704",),
    }
    module.WAN_CONFIGS = {
        "t2v-A14B": SimpleNamespace(sample_steps=40, sample_shift=12.0, sample_guide_scale=(3.0, 4.0), frame_num=81),
        "i2v-A14B": SimpleNamespace(sample_steps=40, sample_shift=5.0, sample_guide_scale=(3.5, 3.5), frame_num=81),
        "ti2v-5B": SimpleNamespace(sample_steps=50, sample_shift=5.0, sample_guide_scale=5.0, frame_num=121),
    }

    t2v_args = SimpleNamespace(
        task="t2v-A14B",
        ckpt_dir="ckpt",
        prompt=None,
        image=None,
        sample_steps=None,
        sample_shift=None,
        sample_guide_scale=None,
        sample_guide_scale_low=None,
        sample_guide_scale_high=None,
        frame_num=None,
        base_seed=-1,
        size="1280*720",
    )
    module._validate_args(t2v_args)
    assert t2v_args.sample_steps == 40
    assert t2v_args.sample_shift == 12.0
    assert t2v_args.sample_guide_scale == (3.0, 4.0)
    assert t2v_args.frame_num == 81

    ti2v_args = SimpleNamespace(
        task="ti2v-5B",
        ckpt_dir="ckpt",
        prompt=None,
        image=None,
        sample_steps=None,
        sample_shift=None,
        sample_guide_scale=None,
        sample_guide_scale_low=None,
        sample_guide_scale_high=None,
        frame_num=None,
        base_seed=-1,
        size="1280*704",
    )
    module._validate_args(ti2v_args)
    assert ti2v_args.sample_steps == 50
    assert ti2v_args.sample_shift == 5.0
    assert ti2v_args.sample_guide_scale == 5.0
    assert ti2v_args.frame_num == 121


def test_wan22_aligncache1_defaults_follow_easycache_wan22_task_ratios():
    module = _load_wan22_aligncache_module()

    expected_by_task = {
        "t2v-A14B": {
            "base_thresh": 0.078,
            "warmup_pairs": 9,
            "phase_thresholds": {"early": 0.0897, "mid": 0.078, "late": 0.0624},
        },
        "i2v-A14B": {
            "base_thresh": 0.065,
            "warmup_pairs": 9,
            "phase_thresholds": {"early": 0.07475, "mid": 0.065, "late": 0.052},
        },
        "ti2v-5B": {
            "base_thresh": 0.065,
            "warmup_pairs": 9,
            "phase_thresholds": {"early": 0.07475, "mid": 0.065, "late": 0.052},
        },
    }
    for task, expected in expected_by_task.items():
        resolved = module.resolve_wan22_aligncache_config(
            task=task,
            method_config=None,
            sample_steps=40,
        )

        assert resolved["base_thresh"] == expected["base_thresh"], task
        assert resolved["warmup_pairs"] == expected["warmup_pairs"], task
        assert resolved["phase_scale_profile"] == "balanced"
        for phase, threshold in expected["phase_thresholds"].items():
            assert math.isclose(resolved["phase_thresholds"][phase], threshold, abs_tol=1e-12), (task, phase)


def test_wan22_cli_defers_aligncache1_threshold_defaults_to_task_profile(monkeypatch):
    module = _load_wan22_parse_module()
    monkeypatch.setattr(sys, "argv", ["aligncache_generate.py"])

    args = module._parse_args()

    assert args.base_thresh is None
    assert args.warmup_pairs is None


def test_wan22_aligncache_rejects_unimplemented_wan22_tasks():
    module = _load_wan22_aligncache_module()

    try:
        module.resolve_wan22_aligncache_config(
            task="s2v-14B",
            method_config=None,
            sample_steps=40,
        )
    except ValueError as exc:
        assert "t2v-A14B" in str(exc)
        assert "i2v-A14B" in str(exc)
        assert "ti2v-5B" in str(exc)
    else:
        raise AssertionError("Expected unsupported Wan2.2 task to be rejected.")


class _FakeModel:
    def __init__(self):
        self.forward_calls = 0

    def forward(self, *args, **kwargs):
        self.forward_calls += 1
        return ["exact"]


class _FakeA14BPipeline:
    def __init__(self):
        self.config = SimpleNamespace(sample_guide_scale=(3.0, 4.0))
        self.low_noise_model = _FakeModel()
        self.high_noise_model = _FakeModel()


class _FakeTI2VPipeline:
    def __init__(self):
        self.model = _FakeModel()


def test_wan22_configures_a14b_low_and_high_noise_models_with_stage_guide_scales():
    module = _load_wan22_aligncache_module()
    pipeline = _FakeA14BPipeline()

    resolved = module.configure_wan22_aligncache(
        pipeline,
        method_name="aligncache",
        method_config={"sample_guide_scale": (3.0, 4.0)},
        task="t2v-A14B",
        sample_steps=40,
    )

    assert resolved is not None
    assert pipeline.low_noise_model.enable_aligncache is True
    assert pipeline.high_noise_model.enable_aligncache is True
    assert pipeline.low_noise_model.sample_guide_scale == 3.0
    assert pipeline.high_noise_model.sample_guide_scale == 4.0
    assert pipeline.low_noise_model.forward.__func__ is module._wan22_aligncache_forward
    assert pipeline.high_noise_model.forward.__func__ is module._wan22_aligncache_forward


def test_wan22_configure_uses_official_a14b_guide_scale_when_not_overridden():
    module = _load_wan22_aligncache_module()
    pipeline = _FakeA14BPipeline()

    module.configure_wan22_aligncache(
        pipeline,
        method_name="aligncache",
        method_config=None,
        task="t2v-A14B",
        sample_steps=40,
    )

    assert pipeline.low_noise_model.sample_guide_scale == 3.0
    assert pipeline.high_noise_model.sample_guide_scale == 4.0


def test_wan22_configures_a14b_models_with_their_actual_step_counts():
    module = _load_wan22_aligncache_module()
    pipeline = _FakeA14BPipeline()

    module.configure_wan22_aligncache(
        pipeline,
        method_name="aligncache",
        method_config={
            "sample_guide_scale": (3.0, 4.0),
            "model_sample_steps": {
                "low_noise_model": 35,
                "high_noise_model": 5,
            },
        },
        task="t2v-A14B",
        sample_steps=40,
    )

    assert pipeline.low_noise_model.total_calls == 70
    assert pipeline.high_noise_model.total_calls == 10
    assert pipeline.low_noise_model.tail_call_start == 68
    assert pipeline.high_noise_model.tail_call_start == 8


def test_wan22_counts_a14b_model_steps_using_official_boundary_rule():
    module = _load_wan22_aligncache_module()

    counts = module._wan22_count_model_sample_steps(
        task="t2v-A14B",
        timesteps=[999, 900, 874, 300],
        boundary=875,
    )

    assert counts == {"low_noise_model": 2, "high_noise_model": 2}
    assert module._wan22_count_model_sample_steps(
        task="ti2v-5B",
        timesteps=[999, 500, 0],
        boundary=None,
    ) == {"model": 3}


def test_wan22_configures_ti2v_single_model_with_scalar_guide_scale():
    module = _load_wan22_aligncache_module()
    pipeline = _FakeTI2VPipeline()

    resolved = module.configure_wan22_aligncache(
        pipeline,
        method_name="aligncache",
        method_config={"sample_guide_scale": 5.0},
        task="ti2v-5B",
        sample_steps=50,
    )

    assert resolved is not None
    assert pipeline.model.enable_aligncache is True
    assert pipeline.model.sample_guide_scale == 5.0
    assert pipeline.model.total_calls == 100
    assert pipeline.model.forward.__func__ is module._wan22_aligncache_forward
