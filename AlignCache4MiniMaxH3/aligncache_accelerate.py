#!/usr/bin/env python3
"""Generate MiniMax-H3 T2VA with Wan2.1-style closed-loop AlignCache."""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any, Sequence

import torch

try:
    from . import aligncache_generate as core
except ImportError:
    import aligncache_generate as core


PHASES = ("early", "mid", "late")
VIDEO_RISK_THRESHOLD = 0.33
AUDIO_RISK_THRESHOLD = 0.165
WARMUP_STEPS = 2
TAIL_STEPS = 1
START_SKIP_STEP = 5
EARLY_END_RATIO = 0.15
LATE_START_RATIO = 0.86
PHASE_THRESHOLD_SCALES = (1.15, 1.00, 0.80)
PHASE_AGE_CAPS = (4, 4, 4)
MAX_SKIP_AGE = max(PHASE_AGE_CAPS)


def _phase_mapping(values: tuple[float, float, float] | tuple[int, int, int]) -> dict[str, float | int]:
    return {phase: values[index] for index, phase in enumerate(PHASES)}


@dataclass(frozen=True)
class AlignCacheH3Config:
    """Resolved H3 controller settings following the Wan2.1 AlignCache structure."""

    video_base_threshold: float = VIDEO_RISK_THRESHOLD
    audio_base_threshold: float = AUDIO_RISK_THRESHOLD
    warmup_steps: int = WARMUP_STEPS
    tail_steps: int = TAIL_STEPS
    start_skip_step: int = START_SKIP_STEP
    early_end_ratio: float = EARLY_END_RATIO
    late_start_ratio: float = LATE_START_RATIO
    phase_threshold_scales: tuple[float, float, float] = PHASE_THRESHOLD_SCALES
    age_caps: tuple[int, int, int] = PHASE_AGE_CAPS
    feedback_ratio_ema_decay: float = 0.90
    feedback_ratio_clip: tuple[float, float] = (0.85, 1.25)
    feedback_scale_low: tuple[float, float, float] = (0.95, 0.90, 0.85)
    feedback_scale_high: tuple[float, float, float] = (1.03, 1.06, 1.00)
    predictor_subsample_stride: int = 64
    predictor_ratio_ema_decay: float = 0.80
    predictor_direction_ema_decay: float = 0.80
    predictor_beta: tuple[float, float, float] = (0.0, 0.35, 0.15)
    predictor_ratio_tolerance: tuple[float, float, float] = (0.0, 0.30, 0.18)
    predictor_direction_minimum: tuple[float, float, float] = (1.0, 0.80, 0.88)
    predictor_scale_ema_decay: float = 0.85
    predictor_scale_bounds: tuple[float, float] = (0.85, 1.18)
    score_eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.video_base_threshold < 0.0 or self.audio_base_threshold < 0.0:
            raise ValueError("MiniMax-H3 risk thresholds must be non-negative.")
        if self.warmup_steps < 0 or self.tail_steps < 0 or self.start_skip_step < 0:
            raise ValueError("MiniMax-H3 warm-up, tail, and start-skip steps must be non-negative.")
        if not 0.0 <= self.early_end_ratio < self.late_start_ratio <= 1.0:
            raise ValueError("MiniMax-H3 phase boundaries must satisfy 0 <= early < late <= 1.")
        if any(value < 0 for value in self.age_caps):
            raise ValueError("MiniMax-H3 phase age caps must be non-negative.")

    @staticmethod
    def _phase_value(values: tuple[Any, Any, Any], phase: str) -> Any:
        return values[PHASES.index(phase)]

    def phase_for_step(self, step: int, total_steps: int) -> str:
        progress = float(step) / max(int(total_steps) - 1, 1)
        if progress < self.early_end_ratio:
            return "early"
        if progress < self.late_start_ratio:
            return "mid"
        return "late"

    def threshold_scale_for_phase(self, phase: str) -> float:
        return float(self._phase_value(self.phase_threshold_scales, phase))

    def age_cap_for_phase(self, phase: str) -> int:
        return int(self._phase_value(self.age_caps, phase))

    def feedback_scale_bounds_for_phase(self, phase: str) -> tuple[float, float]:
        return (
            float(self._phase_value(self.feedback_scale_low, phase)),
            float(self._phase_value(self.feedback_scale_high, phase)),
        )

    def predictor_params_for_phase(self, phase: str) -> tuple[float, float, float]:
        return (
            float(self._phase_value(self.predictor_beta, phase)),
            float(self._phase_value(self.predictor_ratio_tolerance, phase)),
            float(self._phase_value(self.predictor_direction_minimum, phase)),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "predictor": "scaled_residual_momentum",
            "predictor_space": "velocity",
            "video_base_threshold": float(self.video_base_threshold),
            "audio_base_threshold": float(self.audio_base_threshold),
            "warmup_steps": int(self.warmup_steps),
            "tail_steps": int(self.tail_steps),
            "start_skip_step": int(self.start_skip_step),
            "phase_boundaries": {
                "early_end_ratio": float(self.early_end_ratio),
                "late_start_ratio": float(self.late_start_ratio),
            },
            "phase_threshold_scales": _phase_mapping(self.phase_threshold_scales),
            "age_caps": _phase_mapping(self.age_caps),
            "feedback_ratio_ema_decay": float(self.feedback_ratio_ema_decay),
            "feedback_ratio_clip": list(self.feedback_ratio_clip),
            "feedback_error_metric": "relative_l2",
            "feedback_scale_low": _phase_mapping(self.feedback_scale_low),
            "feedback_scale_high": _phase_mapping(self.feedback_scale_high),
            "predictor_beta": _phase_mapping(self.predictor_beta),
            "predictor_scale_bounds": list(self.predictor_scale_bounds),
            "unconditional_correction": False,
        }


@dataclass
class _ModalityRuntime:
    threshold_scale: dict[str, float] = field(default_factory=lambda: {phase: 1.0 for phase in PHASES})
    calibration_ratio_ema: dict[str, float] = field(default_factory=lambda: {phase: 1.0 for phase in PHASES})
    velocity_scale: dict[str, float] = field(default_factory=lambda: {phase: 1.0 for phase in PHASES})
    velocity_ratio_ema: float | None = None
    velocity_direction_ema: float | None = None
    feedback_updates: int = 0
    feedback_skipped: int = 0
    last_feedback: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModalityRisk:
    input_delta: float | None
    gain: float | None
    risk_increment: float | None
    budget: float | None
    base_phase_threshold: float
    threshold_scale: float
    threshold: float


@dataclass(frozen=True)
class RouteDecision:
    step: int
    phase: str
    video_input: torch.Tensor
    audio_input: torch.Tensor
    video_timestep: float
    audio_timestep: float
    action: str
    reason: str
    age_since_exact: int | None
    feedback_eligible: bool
    triggered_modalities: tuple[str, ...]
    video: ModalityRisk
    audio: ModalityRisk


class AlignCacheH3Controller:
    """Wan2.1-style closed-loop controller adapted to H3's joint AV forward."""

    def __init__(self, config: AlignCacheH3Config | None = None):
        self.config = config or AlignCacheH3Config()
        self.state: core.AlignCacheH3State | None = None
        self.runtime: dict[str, _ModalityRuntime] = {}

    def reset(self, total_steps: int) -> core.AlignCacheH3State:
        self.state = core.AlignCacheH3State(total_steps=total_steps)
        self.state.video.exact_history = deque(maxlen=2)
        self.state.audio.exact_history = deque(maxlen=2)
        self.runtime = {"video": _ModalityRuntime(), "audio": _ModalityRuntime()}
        return self.state

    def _state(self) -> core.AlignCacheH3State:
        if self.state is None:
            raise RuntimeError("The MiniMax-H3 AlignCache controller must be reset before denoising.")
        return self.state

    @staticmethod
    def _update_ema(previous: float | None, current: float, decay: float) -> float:
        if previous is None:
            return float(current)
        return float(decay) * float(previous) + (1.0 - float(decay)) * float(current)

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(float(low), min(float(high), float(value)))

    def _threshold_terms(self, modality_name: str, phase: str) -> tuple[float, float, float]:
        base_threshold = (
            self.config.video_base_threshold if modality_name == "video" else self.config.audio_base_threshold
        ) * self.config.threshold_scale_for_phase(phase)
        scale = float(self.runtime[modality_name].threshold_scale[phase])
        return float(base_threshold), scale, float(base_threshold) * scale

    def _empty_risk(self, modality_name: str, phase: str) -> ModalityRisk:
        base_threshold, threshold_scale, threshold = self._threshold_terms(modality_name, phase)
        return ModalityRisk(None, None, None, None, base_threshold, threshold_scale, threshold)

    def _risk(
        self,
        modality_name: str,
        modality: core.ModalityState,
        current_input: torch.Tensor,
        phase: str,
    ) -> ModalityRisk:
        latest_anchor = modality.exact_history[-1]
        previous_anchor = modality.exact_history[-2]
        input_delta = core.mean_abs(
            current_input.detach().to(torch.float32)
            - modality.previous_input.detach().to(torch.float32)
        )
        output_change = core.mean_abs(
            latest_anchor.model_output.detach().to(torch.float32)
            - previous_anchor.model_output.detach().to(torch.float32)
        )
        input_change = core.mean_abs(
            latest_anchor.model_input.detach().to(torch.float32)
            - previous_anchor.model_input.detach().to(torch.float32)
        )
        gain = output_change / max(input_change, core.EPS)
        risk_increment = gain * input_delta / max(core.mean_abs(latest_anchor.model_output), core.EPS)
        base_threshold, threshold_scale, threshold = self._threshold_terms(modality_name, phase)
        return ModalityRisk(
            input_delta=input_delta,
            gain=gain,
            risk_increment=risk_increment,
            budget=modality.budget + risk_increment,
            base_phase_threshold=base_threshold,
            threshold_scale=threshold_scale,
            threshold=threshold,
        )

    def decide(
        self,
        step: int,
        video_input: torch.Tensor,
        audio_input: torch.Tensor,
        video_timestep: float,
        audio_timestep: float,
    ) -> RouteDecision:
        state = self._state()
        video_snapshot = core._snapshot(video_input)
        audio_snapshot = core._snapshot(audio_input)
        phase = self.config.phase_for_step(step, state.total_steps)
        age_since_exact = step - state.last_exact_step if state.last_exact_step >= 0 else None
        video_empty = self._empty_risk("video", phase)
        audio_empty = self._empty_risk("audio", phase)

        warmup_limit = max(int(self.config.warmup_steps), int(self.config.start_skip_step))
        tail_start = max(int(state.total_steps) - int(self.config.tail_steps), 0)
        in_tail = self.config.tail_steps > 0 and step >= tail_start
        if step < warmup_limit or in_tail:
            return RouteDecision(
                step,
                phase,
                video_snapshot,
                audio_snapshot,
                video_timestep,
                audio_timestep,
                "exact",
                "warmup_tail",
                age_since_exact,
                False,
                (),
                video_empty,
                audio_empty,
            )

        if (
            len(state.video.exact_history) < 2
            or len(state.audio.exact_history) < 2
            or state.video.previous_input is None
            or state.audio.previous_input is None
        ):
            return RouteDecision(
                step,
                phase,
                video_snapshot,
                audio_snapshot,
                video_timestep,
                audio_timestep,
                "exact",
                "bootstrap",
                age_since_exact,
                False,
                (),
                video_empty,
                audio_empty,
            )

        video_risk = self._risk("video", state.video, video_snapshot, phase)
        audio_risk = self._risk("audio", state.audio, audio_snapshot, phase)
        if video_risk.threshold <= 0.0 or audio_risk.threshold <= 0.0:
            action, reason, feedback_eligible, triggered = "exact", "threshold_disabled", False, ()
        elif age_since_exact is None or age_since_exact > self.config.age_cap_for_phase(phase):
            action, reason, feedback_eligible, triggered = "exact", "max_age", False, ()
        else:
            triggered = tuple(
                name
                for name, risk in (("video", video_risk), ("audio", audio_risk))
                if risk.budget is not None and risk.budget >= risk.threshold
            )
            if triggered:
                action, reason, feedback_eligible = "exact", "budget_exact", True
            else:
                action, reason, feedback_eligible = "skip", "budget_skip", False

        return RouteDecision(
            step,
            phase,
            video_snapshot,
            audio_snapshot,
            video_timestep,
            audio_timestep,
            action,
            reason,
            age_since_exact,
            feedback_eligible,
            triggered,
            video_risk,
            audio_risk,
        )

    @staticmethod
    def _risk_record(risk: ModalityRisk) -> dict[str, float | None]:
        return {
            "input_delta": risk.input_delta,
            "gain": risk.gain,
            "risk_increment": risk.risk_increment,
            "budget": risk.budget,
            "base_phase_threshold": risk.base_phase_threshold,
            "threshold_scale": risk.threshold_scale,
            "threshold": risk.threshold,
        }

    def _record(
        self,
        decision: RouteDecision,
        transformer_ms: float,
        feedback: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        state = self._state()
        record = {
            "step": decision.step,
            "total_steps": state.total_steps,
            "phase": decision.phase,
            "route": decision.action,
            "reason": decision.reason,
            "age_since_exact": decision.age_since_exact,
            "feedback_eligible": decision.feedback_eligible,
            "triggered_modalities": list(decision.triggered_modalities),
            "video": self._risk_record(decision.video),
            "audio": self._risk_record(decision.audio),
            "feedback": feedback or {},
            "transformer_ms": transformer_ms,
            "controller_ms": None,
        }
        state.trace_records.append(record)
        return record

    @staticmethod
    def _prediction_for_input(prediction: torch.Tensor, current_input: torch.Tensor, label: str) -> torch.Tensor:
        detached = prediction.detach()
        if detached.shape == current_input.shape:
            return detached.clone()
        if detached.ndim == current_input.ndim + 1 and detached.shape[0] == 1 and detached.shape[1:] == current_input.shape:
            return detached[0].clone()
        raise ValueError(
            f"MiniMax-H3 {label} prediction shape {tuple(detached.shape)} does not match input "
            f"shape {tuple(current_input.shape)}."
        )

    def _anchor_velocity(self, anchor: core.ExactAnchor) -> torch.Tensor:
        """Return an H3 model velocity in the model's own output space.

        MiniMax-H3 runs through ComfyUI's FLOW_AV sampler.  Its diffusion
        model output is a velocity and the sampler applies the sigma-dependent
        denoising transform afterwards; it is not an x0/latent tensor.  Keep
        predictor arithmetic entirely in that velocity space.
        """

        return anchor.model_output.detach().to(torch.float32)

    def _subsample(self, value: torch.Tensor) -> torch.Tensor:
        stride = max(int(self.config.predictor_subsample_stride), 1)
        return value.detach().reshape(-1)[::stride].to(torch.float32)

    @staticmethod
    def _cosine(first: torch.Tensor, second: torch.Tensor) -> float | None:
        denominator = torch.linalg.vector_norm(first) * torch.linalg.vector_norm(second)
        if float(denominator.item()) <= 1e-12:
            return None
        return float((torch.dot(first, second) / denominator).clamp(-1.0, 1.0).item())

    @staticmethod
    def _ratio_gate(ratio_ema: float | None, tolerance: float) -> float:
        if ratio_ema is None or tolerance <= 0.0:
            return 0.0
        return max(0.0, 1.0 - abs(float(ratio_ema) - 1.0) / max(float(tolerance), 1e-6))

    @staticmethod
    def _direction_gate(direction_ema: float | None, minimum: float) -> float:
        if direction_ema is None:
            return 0.0
        return max(0.0, min(1.0, (float(direction_ema) - float(minimum)) / max(1.0 - float(minimum), 1e-6)))

    def _build_modality_draft(
        self,
        modality_name: str,
        modality: core.ModalityState,
        current_input: torch.Tensor,
        phase: str,
    ) -> torch.Tensor:
        if not modality.exact_history:
            raise RuntimeError(f"MiniMax-H3 {modality_name} predictor has no exact anchor.")
        runtime = self.runtime[modality_name]
        latest_velocity = self._anchor_velocity(modality.exact_history[-1])
        predicted_velocity = latest_velocity * float(runtime.velocity_scale[phase])

        beta, ratio_tolerance, direction_minimum = self.config.predictor_params_for_phase(phase)
        if beta > 0.0 and len(modality.exact_history) >= 2:
            previous_velocity = self._anchor_velocity(modality.exact_history[-2])
            gate = min(
                self._ratio_gate(runtime.velocity_ratio_ema, ratio_tolerance),
                self._direction_gate(runtime.velocity_direction_ema, direction_minimum),
            )
            if gate > 1e-4:
                predicted_velocity = predicted_velocity + beta * gate * (latest_velocity - previous_velocity)

        return predicted_velocity.to(dtype=current_input.dtype)

    def build_feedback_drafts(self, decision: RouteDecision) -> tuple[torch.Tensor, torch.Tensor] | None:
        if not decision.feedback_eligible:
            return None
        state = self._state()
        return (
            self._build_modality_draft("video", state.video, decision.video_input, decision.phase),
            self._build_modality_draft("audio", state.audio, decision.audio_input, decision.phase),
        )

    def _fit_velocity_scale(self, target: torch.Tensor, basis: torch.Tensor, default: float) -> float:
        target_f = target.detach().to(torch.float32).reshape(-1)
        basis_f = basis.detach().to(torch.float32).reshape(-1)
        denominator = torch.dot(basis_f, basis_f)
        fitted = float(default)
        if float(denominator.item()) > 1e-12:
            fitted = float((torch.dot(target_f, basis_f) / denominator).item())
        low, high = self.config.predictor_scale_bounds
        return self._clamp(fitted, low, high)

    def _update_predictor_from_exact(
        self,
        modality_name: str,
        modality: core.ModalityState,
        current_input: torch.Tensor,
        exact_output: torch.Tensor,
        phase: str,
    ) -> None:
        if not modality.exact_history:
            return
        runtime = self.runtime[modality_name]
        current_velocity = exact_output.detach().to(torch.float32)
        latest_velocity = self._anchor_velocity(modality.exact_history[-1])
        fitted_scale = self._fit_velocity_scale(current_velocity, latest_velocity, runtime.velocity_scale[phase])
        runtime.velocity_scale[phase] = self._update_ema(
            runtime.velocity_scale[phase],
            fitted_scale,
            self.config.predictor_scale_ema_decay,
        )

        current_sub = self._subsample(current_velocity)
        latest_sub = self._subsample(latest_velocity)
        latest_norm = torch.linalg.vector_norm(latest_sub)
        if float(latest_norm.item()) > 1e-12:
            ratio = float((torch.linalg.vector_norm(current_sub) / latest_norm).item())
            runtime.velocity_ratio_ema = self._update_ema(
                runtime.velocity_ratio_ema,
                ratio,
                self.config.predictor_ratio_ema_decay,
            )
        direction = self._cosine(current_sub, latest_sub)
        if direction is not None:
            runtime.velocity_direction_ema = self._update_ema(
                runtime.velocity_direction_ema,
                direction,
                self.config.predictor_direction_ema_decay,
            )

    @staticmethod
    def _relative_mse(prediction: torch.Tensor, reference: torch.Tensor) -> float:
        prediction_f = prediction.detach().to(torch.float32)
        reference_f = reference.detach().to(torch.float32)
        numerator = torch.sum((prediction_f - reference_f) ** 2, dtype=torch.float32)
        denominator = torch.sum(reference_f * reference_f, dtype=torch.float32).clamp_min(1e-12)
        return float((numerator / denominator).item())

    @staticmethod
    def _relative_l2(prediction: torch.Tensor, reference: torch.Tensor) -> float:
        prediction_f = prediction.detach().to(torch.float32)
        reference_f = reference.detach().to(torch.float32)
        numerator = torch.linalg.vector_norm(prediction_f - reference_f)
        denominator = torch.linalg.vector_norm(reference_f).clamp_min(1e-12)
        return float((numerator / denominator).item())

    def _update_feedback(
        self,
        modality_name: str,
        phase: str,
        risk: ModalityRisk,
        draft: torch.Tensor | None,
        exact_output: torch.Tensor,
    ) -> dict[str, Any] | None:
        runtime = self.runtime[modality_name]
        if draft is None or risk.budget is None:
            runtime.feedback_skipped += 1
            return None
        observed_error = self._relative_l2(draft, exact_output)
        calibration_ratio = observed_error / max(float(risk.budget), float(self.config.score_eps))
        clip_low, clip_high = self.config.feedback_ratio_clip
        clipped_ratio = self._clamp(calibration_ratio, clip_low, clip_high)
        ratio_ema = self._update_ema(
            runtime.calibration_ratio_ema[phase],
            clipped_ratio,
            self.config.feedback_ratio_ema_decay,
        )
        runtime.calibration_ratio_ema[phase] = ratio_ema
        scale_low, scale_high = self.config.feedback_scale_bounds_for_phase(phase)
        runtime.threshold_scale[phase] = self._clamp(
            1.0 / max(ratio_ema, self.config.score_eps),
            scale_low,
            scale_high,
        )
        runtime.feedback_updates += 1
        runtime.last_feedback = {
            "phase": phase,
            "observed_relative_l2": observed_error,
            "error_metric": "relative_l2",
            "calibration_ratio": calibration_ratio,
            "clipped_ratio": clipped_ratio,
            "ratio_ema": ratio_ema,
            "threshold_scale": runtime.threshold_scale[phase],
        }
        return dict(runtime.last_feedback)

    def record_exact(
        self,
        decision: RouteDecision,
        video_output: torch.Tensor,
        audio_output: torch.Tensor,
        transformer_ms: float,
        feedback_drafts: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> dict[str, Any]:
        state = self._state()
        video_exact = self._prediction_for_input(video_output, decision.video_input, "video")
        audio_exact = self._prediction_for_input(audio_output, decision.audio_input, "audio")
        feedback: dict[str, Any] = {}
        if decision.feedback_eligible:
            drafts = feedback_drafts or self.build_feedback_drafts(decision)
            video_draft = None if drafts is None else drafts[0]
            audio_draft = None if drafts is None else drafts[1]
            video_feedback = self._update_feedback(
                "video", decision.phase, decision.video, video_draft, video_exact
            )
            audio_feedback = self._update_feedback(
                "audio", decision.phase, decision.audio, audio_draft, audio_exact
            )
            if video_feedback is not None:
                feedback["video"] = video_feedback
            if audio_feedback is not None:
                feedback["audio"] = audio_feedback

        self._update_predictor_from_exact(
            "video", state.video, decision.video_input, video_exact, decision.phase
        )
        self._update_predictor_from_exact(
            "audio", state.audio, decision.audio_input, audio_exact, decision.phase
        )
        state.video.exact_history.append(
            core.ExactAnchor(
                step=decision.step,
                timestep=decision.video_timestep,
                model_input=decision.video_input,
                model_output=video_exact,
            )
        )
        state.audio.exact_history.append(
            core.ExactAnchor(
                step=decision.step,
                timestep=decision.audio_timestep,
                model_input=decision.audio_input,
                model_output=audio_exact,
            )
        )
        state.video.previous_input = decision.video_input
        state.audio.previous_input = decision.audio_input
        state.video.budget = 0.0
        state.audio.budget = 0.0
        state.last_exact_step = decision.step
        state.counters["exact"] += 1
        state.reason_counts[decision.reason] = state.reason_counts.get(decision.reason, 0) + 1
        core._record_controller_memory(state)
        return self._record(decision, transformer_ms, feedback)

    def build_skip_output(self, decision: RouteDecision) -> tuple[torch.Tensor, torch.Tensor]:
        state = self._state()
        if decision.action != "skip":
            raise ValueError("MiniMax-H3 can build cached outputs only for a skip decision.")
        video = self._build_modality_draft("video", state.video, decision.video_input, decision.phase)
        audio = self._build_modality_draft("audio", state.audio, decision.audio_input, decision.phase)
        state.video.previous_input = decision.video_input
        state.audio.previous_input = decision.audio_input
        state.video.budget = decision.video.budget
        state.audio.budget = decision.audio.budget
        state.counters["skip"] += 1
        state.reason_counts[decision.reason] = state.reason_counts.get(decision.reason, 0) + 1
        core._record_controller_memory(state)
        return video, audio

    def record_skip(self, decision: RouteDecision) -> dict[str, Any]:
        return self._record(decision, transformer_ms=0.0)

    def add_transformer_seconds(self, seconds: float) -> None:
        self._state().transformer_seconds += seconds

    def add_controller_seconds(self, seconds: float) -> None:
        self._state().controller_seconds += seconds

    def add_denoise_seconds(self, seconds: float) -> None:
        self._state().denoise_seconds += seconds

    def feedback_summary(self) -> dict[str, Any]:
        return {
            name: {
                "threshold_scale": dict(runtime.threshold_scale),
                "calibration_ratio_ema": dict(runtime.calibration_ratio_ema),
                "velocity_scale": dict(runtime.velocity_scale),
                "velocity_ratio_ema": runtime.velocity_ratio_ema,
                "velocity_direction_ema": runtime.velocity_direction_ema,
                "feedback_updates": int(runtime.feedback_updates),
                "feedback_skipped": int(runtime.feedback_skipped),
                "last_feedback": dict(runtime.last_feedback),
            }
            for name, runtime in self.runtime.items()
        }


def build_aligncache_h3_loop_denoiser(
    base_denoiser_cls: type[Any],
    config: AlignCacheH3Config | None = None,
) -> type[Any]:
    class AlignCacheH3LoopDenoiser(base_denoiser_cls):
        def __init__(self):
            super().__init__(transformer_name="transformer")
            self.controller = AlignCacheH3Controller(config)

        @torch.no_grad()
        def __call__(self, components, block_state, i: int, t: torch.Tensor):
            if i == 0:
                self.controller.reset(core.resolve_actual_step_count(block_state.timesteps))

            block_state._aligncache_h3_collector = self.controller
            block_state._aligncache_h3_step_started_at = perf_counter()
            core._synchronize_for_timing(block_state.latents)
            controller_start = perf_counter()
            decision = self.controller.decide(
                step=i,
                video_input=block_state.latents,
                audio_input=block_state.audio_latents,
                video_timestep=core._as_float(block_state.timesteps[i]),
                audio_timestep=core._as_float(block_state.audio_timesteps[i]),
            )
            feedback_drafts = self.controller.build_feedback_drafts(decision)

            if decision.action == "skip":
                video_draft, audio_draft = self.controller.build_skip_output(decision)
                block_state.noise_pred = video_draft[None]
                block_state.audio_noise_pred = audio_draft[None]
                record = self.controller.record_skip(decision)
                core._synchronize_for_timing(block_state.latents)
                controller_seconds = perf_counter() - controller_start
            else:
                core._synchronize_for_timing(block_state.latents)
                controller_before = perf_counter() - controller_start
                transformer_start = perf_counter()
                components, block_state = super().__call__(components, block_state, i, t)
                core._synchronize_for_timing(block_state.latents)
                transformer_seconds = perf_counter() - transformer_start
                self.controller.add_transformer_seconds(transformer_seconds)

                controller_start = perf_counter()
                record = self.controller.record_exact(
                    decision,
                    block_state.noise_pred,
                    block_state.audio_noise_pred,
                    transformer_seconds * 1000.0,
                    feedback_drafts=feedback_drafts,
                )
                core._synchronize_for_timing(block_state.latents)
                controller_seconds = controller_before + perf_counter() - controller_start

            self.controller.add_controller_seconds(controller_seconds)
            record["controller_ms"] = controller_seconds * 1000.0
            return components, block_state

    AlignCacheH3LoopDenoiser.__name__ = "AlignCacheH3LoopDenoiser"
    return AlignCacheH3LoopDenoiser


def _build_t2va_pipeline(api: SimpleNamespace, args: argparse.Namespace, model_snapshot: dict[str, Any]):
    blocks = api.MiniMaxH3Blocks().get_workflow("t2va")
    loop = blocks.sub_blocks["denoise.denoise"]
    denoiser_cls = build_aligncache_h3_loop_denoiser(api.MiniMaxH3LoopDenoiser)
    scheduler_step_cls = core.build_aligncache_h3_loop_scheduler_step(api.MiniMaxH3LoopSchedulerStep)
    loop.sub_blocks["denoiser"] = denoiser_cls()
    loop.sub_blocks["update"] = scheduler_step_cls()

    revision = model_snapshot["resolved_revision"]
    pipeline_kwargs: dict[str, Any] = {
        "blocks": blocks,
        "pretrained_model_name_or_path": model_snapshot["model_id"],
    }
    if revision is not None:
        pipeline_kwargs["revision"] = revision
    pipe = api.MiniMaxH3ModularPipeline(**pipeline_kwargs)

    load_kwargs: dict[str, Any] = {"dtype": torch.bfloat16}
    if revision is not None:
        load_kwargs["revision"] = revision
    pipe.load_components(**load_kwargs)
    pipe.to("cuda")
    return pipe


def _installed_controller(pipe) -> AlignCacheH3Controller:
    denoiser = pipe._blocks.sub_blocks["denoise.denoise"].sub_blocks["denoiser"]
    return denoiser.controller


def build_summary(
    controller: AlignCacheH3Controller,
    args: argparse.Namespace,
    model_snapshot: dict[str, Any],
    end_to_end_seconds: float,
) -> dict[str, Any]:
    state = controller._state()
    return {
        "mode": "aligncache",
        "aligncache_config": controller.config.as_dict(),
        "aligncache_feedback": controller.feedback_summary(),
        "seed": args.seed,
        "model_snapshot": model_snapshot,
        "diffusers_commit": core.DIFFUSERS_COMMIT,
        "parameters": {
            "model_id": args.model_id,
            "model_revision": args.model_revision,
            "prompt": args.prompt,
            "num_frames": args.num_frames,
            "num_inference_steps": args.num_inference_steps,
        },
        "actual_timesteps": state.total_steps,
        "exact_count": state.counters["exact"],
        "skip_count": state.counters["skip"],
        "reason_counts": state.reason_counts,
        "trace_rows": len(state.trace_records),
        "timings_ms": {
            "transformer": state.transformer_seconds * 1000.0,
            "controller": state.controller_seconds * 1000.0,
            "denoise": state.denoise_seconds * 1000.0,
            "end_to_end": end_to_end_seconds * 1000.0,
        },
        "controller_cache": {
            "current_tensor_bytes": state.current_tensor_bytes,
            "peak_tensor_bytes": state.peak_tensor_bytes,
        },
        "hardware": core._hardware_summary(),
        "dependencies": core._dependency_versions(),
    }


def run_generation(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("MiniMax-H3 AlignCache runs only on the resident BF16 CUDA path.")

    api = core._load_minimax_h3_api()
    model_snapshot = core.resolve_model_snapshot(args.model_id, args.model_revision)
    pipe = _build_t2va_pipeline(api, args, model_snapshot)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    start = perf_counter()
    results = pipe(
        prompt=args.prompt,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        generator=generator,
        output=["videos", "audio", "sampling_rate"],
    )
    api.encode_video(
        results["videos"][0],
        fps=24,
        output_path=str(output_path),
        audio=results["audio"][0],
        audio_sample_rate=results["sampling_rate"],
    )
    torch.cuda.synchronize()
    end_to_end_seconds = perf_counter() - start

    controller = _installed_controller(pipe)
    if controller.state is None:
        raise RuntimeError("MiniMax-H3 denoiser did not initialize the AlignCache state.")
    summary = build_summary(controller, args, model_snapshot, end_to_end_seconds)
    summary["output"] = str(output_path)

    if args.trace_jsonl is not None:
        core._write_jsonl(Path(args.trace_jsonl), controller.state.trace_records)
    if args.summary_json is not None:
        core._write_json(Path(args.summary_json), summary)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate MiniMax-H3 T2VA with AlignCache.")
    parser.add_argument("--model_id", default=core.DEFAULT_MODEL_ID, help="Hugging Face model ID or fixed local snapshot.")
    parser.add_argument("--model_revision", default=None, help="Optional immutable Hub revision.")
    parser.add_argument("--prompt", required=True, help="T2VA prompt.")
    parser.add_argument("--num_frames", type=int, default=124, help="Requested MiniMax-H3 frame count.")
    parser.add_argument("--num_inference_steps", type=int, default=20, help="Sigma-grid point count (default: 20).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--trace_jsonl", default=None, help="Optional per-step route trace JSONL path.")
    parser.add_argument("--summary_json", default=None, help="Optional run summary JSON path.")
    parser.add_argument("--output", required=True, help="Muxed audio-video output path, normally .mp4.")
    args = parser.parse_args(argv)
    if args.num_frames < 1:
        parser.error("--num_frames must be positive.")
    if args.num_inference_steps < 2:
        parser.error("--num_inference_steps must be at least 2.")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    summary = run_generation(parse_args(argv))
    print(
        json.dumps(
            {
                "output": summary["output"],
                "actual_timesteps": summary["actual_timesteps"],
                "exact_count": summary["exact_count"],
                "skip_count": summary["skip_count"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
