import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "AlignCache4MiniMaxH3" / "aligncache_generate.py"


def _load_module():
    name = "aligncache_minimax_h3_support"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load_module()


class _FakeBaseDenoiser:
    def __init__(self, transformer_name="transformer"):
        self.transformer_name = transformer_name

    def __call__(self, components, block_state, i, t):
        components.transformer_calls += 1
        block_state.noise_pred = (block_state.latents * 2.0 + float(i))[None]
        block_state.audio_noise_pred = (block_state.audio_latents * 3.0 - float(i))[None]
        return components, block_state


class _FakeScheduler:
    def __init__(self):
        self.calls = 0
        self.inputs = []

    def step(self, prediction, timestep, latents):
        self.calls += 1
        self.inputs.append(prediction.clone())
        latents.add_(prediction.mean())
        return (latents,)


class _FakeBaseSchedulerStep:
    def __call__(self, components, block_state, i, t):
        block_state.latents = components.video_scheduler.step(
            block_state.noise_pred[0], t, block_state.latents
        )[0]
        block_state.audio_latents = components.audio_scheduler.step(
            block_state.audio_noise_pred[0], block_state.audio_timesteps[i], block_state.audio_latents
        )[0]
        return components, block_state


def _run_fake_loop(mode):
    denoiser_cls = MODULE.build_aligncache_h3_loop_denoiser(_FakeBaseDenoiser)
    scheduler_step_cls = MODULE.build_aligncache_h3_loop_scheduler_step(_FakeBaseSchedulerStep)
    denoiser = denoiser_cls(mode)
    update = scheduler_step_cls()
    components = SimpleNamespace(
        transformer_calls=0,
        video_scheduler=_FakeScheduler(),
        audio_scheduler=_FakeScheduler(),
    )
    block_state = SimpleNamespace(
        timesteps=[0.1, 0.2, 0.5],
        audio_timesteps=[0.2, 0.4, 0.55],
        latents=torch.tensor([1.0, 2.0], dtype=torch.float32),
        audio_latents=torch.tensor([4.0, 5.0], dtype=torch.float32),
    )
    for i, timestep in enumerate(block_state.timesteps):
        components, block_state = denoiser(components, block_state, i, torch.tensor(timestep))
        components, block_state = update(components, block_state, i, torch.tensor(timestep))
    return denoiser.collector.state, components


def _complete(collector, step, video_input, audio_input, video_timestep, audio_timestep, video_output, audio_output):
    pending = collector.prepare_exact(
        step=step,
        video_input=video_input,
        audio_input=audio_input,
        video_timestep=video_timestep,
        audio_timestep=audio_timestep,
    )
    return collector.complete_exact(
        pending=pending,
        video_output=video_output,
        audio_output=audio_output,
        transformer_ms=1.0,
    )


def test_actual_timestep_count_uses_the_scheduler_list_not_requested_grid_size():
    assert MODULE.resolve_actual_step_count([0.9, 0.4, 0.1]) == 3


def test_velocity_draft_uses_each_modalitys_nonuniform_timestep_spacing():
    video_previous = MODULE.ExactAnchor(0, 0.1, torch.zeros(1), torch.tensor([1.0]))
    video_anchor = MODULE.ExactAnchor(1, 0.2, torch.zeros(1), torch.tensor([3.0]))
    audio_previous = MODULE.ExactAnchor(0, 0.2, torch.zeros(1), torch.tensor([2.0]))
    audio_anchor = MODULE.ExactAnchor(1, 0.4, torch.zeros(1), torch.tensor([6.0]))
    target = torch.zeros(1)

    assert math.isclose(MODULE.velocity_time_ratio(video_anchor, video_previous, 0.5), 3.0)
    assert math.isclose(MODULE.velocity_time_ratio(audio_anchor, audio_previous, 0.55), 0.75)
    torch.testing.assert_close(MODULE.velocity_draft(video_anchor, video_previous, target, 0.5), torch.tensor([9.0]))
    torch.testing.assert_close(MODULE.velocity_draft(audio_anchor, audio_previous, target, 0.55), torch.tensor([9.0]))


def test_trace_keeps_both_scheduler_inputs_exact_and_steps_every_actual_timestep():
    off_state, off_components = _run_fake_loop("off")
    trace_state, trace_components = _run_fake_loop("trace")

    assert off_components.transformer_calls == 3
    assert trace_components.transformer_calls == 3
    assert off_components.video_scheduler.calls == trace_components.video_scheduler.calls == 3
    assert off_components.audio_scheduler.calls == trace_components.audio_scheduler.calls == 3
    for off_value, trace_value in zip(off_components.video_scheduler.inputs, trace_components.video_scheduler.inputs):
        assert torch.equal(off_value, trace_value)
    for off_value, trace_value in zip(off_components.audio_scheduler.inputs, trace_components.audio_scheduler.inputs):
        assert torch.equal(off_value, trace_value)

    assert off_state.total_steps == trace_state.total_steps == 3
    assert off_state.counters == {"exact": 3, "skip": 0}
    assert trace_state.counters == {"exact": 3, "skip": 0}
    assert len(trace_state.trace_records) == 3
    assert trace_state.denoise_seconds > 0.0


def test_anchor_snapshots_survive_the_schedulers_inplace_latent_update():
    trace_state, components = _run_fake_loop("trace")

    first_video_anchor = next(anchor for anchor in trace_state.video.exact_history if anchor.step == 0)
    first_audio_anchor = next(anchor for anchor in trace_state.audio.exact_history if anchor.step == 0)
    torch.testing.assert_close(first_video_anchor.model_input, torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(first_audio_anchor.model_input, torch.tensor([4.0, 5.0]))
    torch.testing.assert_close(first_video_anchor.model_output, torch.tensor([2.0, 4.0]))
    torch.testing.assert_close(first_audio_anchor.model_output, torch.tensor([12.0, 15.0]))
    assert not torch.equal(first_video_anchor.model_input, components.video_scheduler.inputs[-1])


def test_drafts_match_current_target_shape_and_dtype():
    anchor = MODULE.ExactAnchor(
        step=1,
        timestep=0.2,
        model_input=torch.tensor([1.0, 2.0], dtype=torch.float32),
        model_output=torch.tensor([4.0, 6.0], dtype=torch.float32),
    )
    previous = MODULE.ExactAnchor(
        step=0,
        timestep=0.1,
        model_input=torch.zeros(2),
        model_output=torch.tensor([2.0, 3.0]),
    )
    current = torch.tensor([3.0, 5.0], dtype=torch.bfloat16)

    drafts = (
        MODULE.reuse_draft(anchor, current),
        MODULE.residual_draft(anchor, current),
        MODULE.velocity_draft(anchor, previous, current, 0.5),
    )
    for draft in drafts:
        assert draft.shape == current.shape
        assert draft.dtype == current.dtype
    torch.testing.assert_close(drafts[0].float(), torch.tensor([4.0, 6.0]))
    torch.testing.assert_close(drafts[1].float(), torch.tensor([6.0, 9.0]))
    torch.testing.assert_close(drafts[2].float(), torch.tensor([10.0, 15.0]))


def test_anchor_shortage_is_null_and_velocity_waits_for_two_exact_anchors():
    collector = MODULE.AlignCacheH3TraceCollector("trace")
    collector.reset(total_steps=3)
    first = _complete(
        collector,
        0,
        torch.tensor([1.0]),
        torch.tensor([2.0]),
        0.1,
        0.2,
        torch.tensor([3.0]),
        torch.tensor([4.0]),
    )
    second = _complete(
        collector,
        1,
        torch.tensor([1.5]),
        torch.tensor([2.5]),
        0.2,
        0.4,
        torch.tensor([5.0]),
        torch.tensor([7.0]),
    )

    assert first["video"]["candidates"][0]["anchor_step"] is None
    candidate = second["video"]["candidates"][0]
    assert candidate["anchor_step"] == 0
    assert candidate["reuse_rel_l2"] is not None
    assert candidate["residual_rel_l2"] is not None
    assert candidate["velocity_rel_l2"] is None
    assert candidate["budget"] is None


def test_trace_records_separate_video_and_audio_velocity_ratios():
    collector = MODULE.AlignCacheH3TraceCollector("trace")
    collector.reset(total_steps=3)
    _complete(
        collector,
        0,
        torch.tensor([0.0]),
        torch.tensor([0.0]),
        0.1,
        0.2,
        torch.tensor([1.0]),
        torch.tensor([2.0]),
    )
    _complete(
        collector,
        1,
        torch.tensor([1.0]),
        torch.tensor([1.0]),
        0.2,
        0.4,
        torch.tensor([3.0]),
        torch.tensor([6.0]),
    )
    third = _complete(
        collector,
        2,
        torch.tensor([2.0]),
        torch.tensor([2.0]),
        0.5,
        0.55,
        torch.tensor([9.0]),
        torch.tensor([9.0]),
    )

    video_candidate = third["video"]["candidates"][0]
    audio_candidate = third["audio"]["candidates"][0]
    assert math.isclose(video_candidate["velocity_time_ratio"], 3.0)
    assert math.isclose(audio_candidate["velocity_time_ratio"], 0.75)
    assert math.isclose(video_candidate["velocity_rel_l2"], 0.0)
    assert math.isclose(audio_candidate["velocity_rel_l2"], 0.0)


def test_cli_parses_without_importing_the_pinned_minimax_diffusers_runtime():
    args = MODULE.parse_args(
        [
            "--prompt",
            "A test prompt.",
            "--aligncache_mode",
            "trace",
            "--output",
            "output/test.mp4",
        ]
    )
    assert args.aligncache_mode == "trace"
    assert args.num_inference_steps == 20
