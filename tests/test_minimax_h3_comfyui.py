import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = ROOT / "AlignCache4MiniMaxH3"


def _load_package():
    name = "aligncache_h3_comfy_package"
    spec = importlib.util.spec_from_file_location(
        name,
        PACKAGE_DIR / "__init__.py",
        submodule_search_locations=[str(PACKAGE_DIR)],
    )
    assert spec is not None
    assert spec.loader is not None
    package = importlib.util.module_from_spec(spec)
    sys.modules[name] = package
    spec.loader.exec_module(package)
    return sys.modules[f"{name}.comfyui_nodes"]


MODULE = _load_package()


def test_comfy_q8_defaults_are_conservative_and_velocity_space_is_recorded():
    config = MODULE.ComfyH3TraceConfig(mode="on", trace_jsonl="", summary_json="")

    assert config.video_risk_threshold == 0.33
    assert config.audio_risk_threshold == 0.165
    assert config.max_skip_age == 4
    assert config.start_skip_step == 5
    assert MODULE.AlignCacheH3ComfyTrace.INPUT_TYPES()["required"]["max_skip_age"][1]["default"] == 4
    assert MODULE.AlignCacheH3ComfyTrace.INPUT_TYPES()["required"]["start_skip_step"][1]["default"] == 5


class _FakeDiffusionExecutor:
    def __init__(self):
        self.class_obj = SimpleNamespace(
            sigma_shift_video=12.0,
            sigma_shift_audio=3.0,
            model_patcher=SimpleNamespace(model=SimpleNamespace(diffusion_model=SimpleNamespace())),
        )
        self.calls = 0
        self.outputs = []

    def __call__(self, x, timestep, context, transformer_options, minimax_payload=None, **kwargs):
        self.calls += 1
        output = [
            x[0] * 2.0 + self.calls,
            x[1] * 3.0 - self.calls,
        ]
        self.outputs.append(output)
        return output


def _run_fake_trace(mode):
    config = MODULE.ComfyH3TraceConfig(mode=mode, trace_jsonl="", summary_json="")
    sigmas = torch.tensor([1.0, 0.6, 0.2, 0.0])
    run_state = MODULE.ComfyH3RunState.begin(config, sigmas)
    transformer_options = {MODULE.RUN_STATE_KEY: run_state}
    executor = _FakeDiffusionExecutor()
    context = torch.zeros(1, 1, 1)
    outputs = []

    for step, sigma in enumerate(sigmas[:-1]):
        x = [
            torch.tensor([[1.0 + step, 2.0 + step]]),
            torch.tensor([[4.0 + step, 5.0 + step]]),
        ]
        output = MODULE.minimax_h3_diffusion_model_wrapper(
            executor,
            x,
            (sigma * 1000.0).reshape(1),
            context,
            transformer_options,
            minimax_payload={"audio_scale": 1.0},
        )
        assert executor.calls == step + 1
        assert output is executor.outputs[-1]
        outputs.append(output)

    return run_state, executor, outputs


def test_trace_wrapper_delegates_once_per_step_and_keeps_exact_dual_output():
    run_state, executor, outputs = _run_fake_trace("trace")
    state = run_state.finish()

    assert executor.calls == 3
    assert state.counters == {"exact": 3, "skip": 0}
    assert len(state.trace_records) == 3
    for output, exact in zip(outputs, executor.outputs):
        assert output is exact
    torch.testing.assert_close(outputs[-1][0], torch.tensor([[9.0, 11.0]]))
    torch.testing.assert_close(outputs[-1][1], torch.tensor([[15.0, 18.0]]))


def test_off_wrapper_still_calls_the_h3_executor_once_and_records_no_shadow_rows():
    run_state, executor, _ = _run_fake_trace("off")
    state = run_state.finish()

    assert executor.calls == 3
    assert state.counters == {"exact": 3, "skip": 0}
    assert state.reason_counts == {"off_exact": 3}
    assert state.trace_records == []


def test_comfy_audio_timestep_uses_the_h3_shift_and_finalizes_actual_call_count():
    run_state, executor, _ = _run_fake_trace("trace")
    state = run_state.finish()

    expected_audio_timestep = 1.0 - MODULE._time_shift_sigma(0.6, 12.0, 3.0)
    second_record = state.trace_records[1]
    first_video_anchor = next(anchor for anchor in state.video.exact_history if anchor.step == 0)
    first_audio_anchor = next(anchor for anchor in state.audio.exact_history if anchor.step == 0)
    second_audio_anchor = next(anchor for anchor in state.audio.exact_history if anchor.step == 1)
    assert math.isclose(first_video_anchor.timestep, 0.0)
    assert math.isclose(first_audio_anchor.timestep, 0.0)
    assert math.isclose(second_audio_anchor.timestep, expected_audio_timestep)
    assert second_record["total_steps"] == 3
    assert state.total_steps == 3


def _run_fake_on(
    *,
    video_threshold,
    audio_threshold,
    max_skip_age=1,
    start_skip_step=0,
    steps=4,
    drop_audio_history_before_step=None,
):
    config = MODULE.ComfyH3TraceConfig(
        mode="on",
        trace_jsonl="",
        summary_json="",
        predictor="scaled_residual_momentum",
        video_risk_threshold=video_threshold,
        audio_risk_threshold=audio_threshold,
        max_skip_age=max_skip_age,
        start_skip_step=start_skip_step,
    )
    sigmas = torch.linspace(1.0, 0.0, steps + 1)
    run_state = MODULE.ComfyH3RunState.begin(config, sigmas)
    transformer_options = {MODULE.RUN_STATE_KEY: run_state}
    executor = _FakeDiffusionExecutor()
    context = torch.zeros(1, 1, 1)
    outputs = []
    inputs = []

    for step, sigma in enumerate(sigmas[:-1]):
        if step == drop_audio_history_before_step:
            assert run_state.on_controller is not None
            run_state.on_controller.state.audio.previous_input = None
        x = [
            torch.tensor([[1.0 + step, 2.0 + step]]),
            torch.tensor([[4.0 + step, 5.0 + step]]),
        ]
        output = MODULE.minimax_h3_diffusion_model_wrapper(
            executor,
            x,
            (sigma * 1000.0).reshape(1),
            context,
            transformer_options,
            minimax_payload={"audio_scale": 1.0},
        )
        inputs.append(x)
        outputs.append(output)

        # The route must keep its own immutable anchors even though Comfy's
        # scheduler normally owns and may rewrite the live latent tensors.
        if step == 1:
            x[0].add_(1000.0)
            x[1].sub_(1000.0)

    return run_state, executor, outputs, inputs


def test_on_route_skips_one_joint_executor_call_only_after_two_exact_bootstraps():
    run_state, executor, outputs, _ = _run_fake_on(video_threshold=1.0, audio_threshold=1.0)
    state = run_state.finish()

    # Steps 0 and 1 are exact warm-up anchors; step 2 reuses both modalities;
    # the final step is the exact protected tail.
    assert executor.calls == 3
    assert state.counters == {"exact": 3, "skip": 1}
    assert state.reason_counts == {
        "warmup_tail": 3,
        "budget_skip": 1,
    }
    assert state.total_steps == 4
    assert len(run_state.records()) == 4
    assert run_state.records()[2]["transformer_ms"] == 0.0
    torch.testing.assert_close(outputs[2][0], torch.tensor([[6.1620, 8.2160]]), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(outputs[2][1], torch.tensor([[13.5833, 16.6543]]), rtol=1e-4, atol=1e-4)

    # The anchor from step 1 did not alias the input tensors mutated above.
    step_one_video_anchor = next(anchor for anchor in state.video.exact_history if anchor.step == 1)
    step_one_audio_anchor = next(anchor for anchor in state.audio.exact_history if anchor.step == 1)
    torch.testing.assert_close(step_one_video_anchor.model_input, torch.tensor([[2.0, 3.0]]))
    torch.testing.assert_close(step_one_audio_anchor.model_input, torch.tensor([[5.0, 6.0]]))


def test_on_route_forces_a_joint_exact_call_when_audio_budget_exceeds_its_own_threshold():
    run_state, executor, outputs, _ = _run_fake_on(
        video_threshold=1.0,
        audio_threshold=0.01,
        steps=4,
    )
    state = run_state.finish()

    assert executor.calls == 4
    assert state.counters == {"exact": 4, "skip": 0}
    assert state.reason_counts == {"warmup_tail": 3, "budget_exact": 1}
    assert run_state.records()[2]["video"]["budget"] is not None
    assert run_state.records()[2]["audio"]["budget"] > 0.01
    assert run_state.records()[2]["triggered_modalities"] == ["audio"]
    assert set(run_state.records()[2]["feedback"]) == {"video", "audio"}
    assert outputs[-1] is executor.outputs[-1]


def test_on_default_zero_thresholds_keep_the_route_exact_until_calibrated():
    run_state, executor, _, _ = _run_fake_on(video_threshold=0.0, audio_threshold=0.0, steps=4)
    state = run_state.finish()

    assert executor.calls == 4
    assert state.counters == {"exact": 4, "skip": 0}
    assert state.reason_counts == {"warmup_tail": 3, "threshold_disabled": 1}


def test_on_route_keeps_joint_steps_exact_until_the_configured_start_step():
    run_state, executor, _, _ = _run_fake_on(
        video_threshold=1.0,
        audio_threshold=1.0,
        start_skip_step=3,
        steps=5,
    )
    state = run_state.finish()

    assert executor.calls == 4
    assert state.counters == {"exact": 4, "skip": 1}
    assert state.reason_counts == {
        "warmup_tail": 4,
        "budget_skip": 1,
    }
    assert run_state.records()[2]["reason"] == "warmup_tail"
    assert run_state.records()[3]["route"] == "skip"


def test_on_route_falls_back_to_one_joint_exact_call_when_a_modality_history_is_missing():
    run_state, executor, _, _ = _run_fake_on(
        video_threshold=1.0,
        audio_threshold=1.0,
        steps=4,
        drop_audio_history_before_step=2,
    )
    state = run_state.finish()

    assert executor.calls == 4
    assert state.counters == {"exact": 4, "skip": 0}
    assert state.reason_counts == {"warmup_tail": 3, "bootstrap": 1}


def test_on_summary_reports_route_counts_config_and_outer_sampler_time(tmp_path):
    run_state, executor, _, _ = _run_fake_on(video_threshold=1.0, audio_threshold=1.0)
    run_state.state().denoise_seconds = 0.125
    summary_path = tmp_path / "on-summary.json"

    MODULE._write_comfy_summary(summary_path, run_state, executor)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert summary["actual_timesteps"] == 4
    assert summary["exact_count"] == 3
    assert summary["skip_count"] == 1
    assert summary["route_rows"] == 4
    assert summary["timings_ms"]["denoise"] == 125.0
    assert summary["on_config"]["predictor"] == "scaled_residual_momentum"
    assert summary["on_config"]["predictor_space"] == "velocity"
    assert summary["on_config"]["video_base_threshold"] == 1.0
    assert summary["on_config"]["audio_base_threshold"] == 1.0
    assert summary["on_config"]["age_caps"] == {"early": 1, "mid": 1, "late": 1}
    assert summary["on_config"]["unconditional_correction"] is False
    assert set(summary["on_feedback"]) == {"video", "audio"}
