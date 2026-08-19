import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "AlignCache4MiniMaxH3"
SCRIPT = SCRIPT_DIR / "aligncache_accelerate.py"


def _load_module():
    sys.path.insert(0, str(SCRIPT_DIR))
    name = "aligncache_minimax_h3_accelerate_support"
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
        block_state.noise_pred = torch.ones_like(block_state.latents)[None]
        block_state.audio_noise_pred = torch.full_like(block_state.audio_latents, 2.0)[None]
        return components, block_state


class _FakeScheduler:
    def __init__(self):
        self.calls = 0

    def step(self, prediction, timestep, latents):
        self.calls += 1
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


def test_wan_style_route_keeps_warmup_tail_and_age_refresh_joint_while_advancing_both_schedulers():
    denoiser_cls = MODULE.build_aligncache_h3_loop_denoiser(_FakeBaseDenoiser)
    scheduler_step_cls = MODULE.core.build_aligncache_h3_loop_scheduler_step(_FakeBaseSchedulerStep)
    denoiser = denoiser_cls()
    update = scheduler_step_cls()
    components = SimpleNamespace(
        transformer_calls=0,
        video_scheduler=_FakeScheduler(),
        audio_scheduler=_FakeScheduler(),
    )
    block_state = SimpleNamespace(
        timesteps=list(range(8)),
        audio_timesteps=list(range(8)),
        latents=torch.tensor([1.0]),
        audio_latents=torch.tensor([2.0]),
    )

    for step, timestep in enumerate(block_state.timesteps):
        components, block_state = denoiser(components, block_state, step, torch.tensor(timestep))
        components, block_state = update(components, block_state, step, torch.tensor(timestep))

    state = denoiser.controller.state
    assert state is not None
    assert components.transformer_calls == 4
    assert components.video_scheduler.calls == components.audio_scheduler.calls == 8
    assert state.counters == {"exact": 4, "skip": 4}
    assert state.reason_counts == {"warmup_tail": 3, "budget_skip": 4, "max_age": 1}
    assert len(state.trace_records) == 8


def test_audio_budget_forces_one_joint_exact_call():
    controller = MODULE.AlignCacheH3Controller(
        MODULE.AlignCacheH3Config(
            video_base_threshold=10.0,
            audio_base_threshold=0.1,
            warmup_steps=0,
            tail_steps=0,
            phase_threshold_scales=(1.0, 1.0, 1.0),
            age_caps=(10, 10, 10),
        )
    )
    controller.reset(3)

    first = controller.decide(0, torch.tensor([0.0]), torch.tensor([0.0]), 0.0, 0.0)
    controller.record_exact(first, torch.tensor([[1.0]]), torch.tensor([[1.0]]), 1.0)
    second = controller.decide(1, torch.tensor([1.0]), torch.tensor([1.0]), 1.0, 1.0)
    controller.record_exact(second, torch.tensor([[2.0]]), torch.tensor([[2.0]]), 1.0)

    decision = controller.decide(2, torch.tensor([1.01]), torch.tensor([2.0]), 2.0, 2.0)

    assert decision.action == "exact"
    assert decision.reason == "budget_exact"
    assert decision.triggered_modalities == ("audio",)
    assert decision.video.budget is not None and decision.video.budget < decision.video.threshold
    assert decision.audio.budget is not None and decision.audio.budget >= decision.audio.threshold


def test_budget_exact_updates_each_modality_feedback_scale_from_counterfactual_error():
    controller = MODULE.AlignCacheH3Controller(
        MODULE.AlignCacheH3Config(
            video_base_threshold=0.1,
            audio_base_threshold=0.1,
            warmup_steps=0,
            tail_steps=0,
            phase_threshold_scales=(1.0, 1.0, 1.0),
            age_caps=(10, 10, 10),
            feedback_ratio_ema_decay=0.0,
        )
    )
    controller.reset(3)

    first = controller.decide(0, torch.tensor([0.0]), torch.tensor([0.0]), 0.0, 0.0)
    controller.record_exact(first, torch.tensor([[1.0]]), torch.tensor([[1.0]]), 1.0)
    second = controller.decide(1, torch.tensor([1.0]), torch.tensor([1.0]), 1.0, 1.0)
    controller.record_exact(second, torch.tensor([[2.0]]), torch.tensor([[2.0]]), 1.0)

    decision = controller.decide(2, torch.tensor([2.0]), torch.tensor([2.0]), 2.0, 2.0)
    assert decision.feedback_eligible
    drafts = controller.build_feedback_drafts(decision)
    assert drafts is not None
    record = controller.record_exact(
        decision,
        torch.tensor([[0.0]]),
        torch.tensor([[0.0]]),
        1.0,
        feedback_drafts=drafts,
    )

    assert set(record["feedback"]) == {"video", "audio"}
    for modality_name in ("video", "audio"):
        runtime = controller.runtime[modality_name]
        assert runtime.feedback_updates == 1
        assert runtime.threshold_scale["late"] == 0.85


def test_skip_predictions_do_not_replace_the_two_exact_predictor_anchors():
    controller = MODULE.AlignCacheH3Controller(
        MODULE.AlignCacheH3Config(
            video_base_threshold=100.0,
            audio_base_threshold=100.0,
            warmup_steps=0,
            tail_steps=0,
            phase_threshold_scales=(1.0, 1.0, 1.0),
            age_caps=(10, 10, 10),
        )
    )
    controller.reset(5)
    first = controller.decide(0, torch.tensor([0.0]), torch.tensor([0.0]), 0.0, 0.0)
    controller.record_exact(first, torch.tensor([[1.0]]), torch.tensor([[1.0]]), 1.0)
    second = controller.decide(1, torch.tensor([1.0]), torch.tensor([1.0]), 1.0, 1.0)
    controller.record_exact(second, torch.tensor([[2.0]]), torch.tensor([[2.0]]), 1.0)

    third = controller.decide(2, torch.tensor([2.0]), torch.tensor([2.0]), 2.0, 2.0)
    video_draft, audio_draft = controller.build_skip_output(third)
    controller.record_skip(third)

    assert [anchor.step for anchor in controller.state.video.exact_history] == [0, 1]
    assert [anchor.step for anchor in controller.state.audio.exact_history] == [0, 1]
    torch.testing.assert_close(video_draft, torch.tensor([3.0]))
    torch.testing.assert_close(audio_draft, torch.tensor([3.0]))


def test_fp8_exact_outputs_are_promoted_for_risk_and_keep_scheduler_dtype():
    if not hasattr(torch, "float8_e4m3fn"):
        return
    config = MODULE.AlignCacheH3Config(
        video_base_threshold=100.0,
        audio_base_threshold=100.0,
        warmup_steps=0,
        tail_steps=0,
        phase_threshold_scales=(1.0, 1.0, 1.0),
        age_caps=(10, 10, 10),
    )
    controller = MODULE.AlignCacheH3Controller(config)
    controller.reset(5)
    inputs = [
        torch.tensor([0.0, 1.0], dtype=torch.bfloat16),
        torch.tensor([1.0, 2.0], dtype=torch.bfloat16),
    ]
    outputs = [
        torch.tensor([[1.0, 2.0]], dtype=torch.float8_e4m3fn),
        torch.tensor([[2.0, 3.0]], dtype=torch.float8_e4m3fn),
    ]
    for step, (current_input, exact_output) in enumerate(zip(inputs, outputs)):
        decision = controller.decide(step, current_input, current_input, 0.1 + step, 0.2 + step)
        controller.record_exact(decision, exact_output, exact_output, 1.0)

    decision = controller.decide(
        2,
        torch.tensor([2.0, 3.0], dtype=torch.bfloat16),
        torch.tensor([2.0, 3.0], dtype=torch.bfloat16),
        2.1,
        2.2,
    )
    video_draft, audio_draft = controller.build_skip_output(decision)
    assert decision.action == "skip"
    assert video_draft.dtype == audio_draft.dtype == torch.bfloat16


def test_cli_uses_the_resolved_closed_loop_defaults_not_a_mode_switch():
    args = MODULE.parse_args(["--prompt", "A test prompt.", "--output", "output/test.mp4"])

    assert not hasattr(args, "aligncache_mode")
    assert MODULE.VIDEO_RISK_THRESHOLD == 0.40
    assert MODULE.AUDIO_RISK_THRESHOLD == 0.20
    assert MODULE.PHASE_THRESHOLD_SCALES == (1.15, 1.0, 0.8)
    assert MODULE.PHASE_AGE_CAPS == (5, 4, 3)
    assert MODULE.AlignCacheH3Config().as_dict()["predictor"] == "scaled_residual_momentum"
