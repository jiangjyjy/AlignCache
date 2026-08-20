#!/usr/bin/env python3
"""P0 AlignCache trace runner for the fixed MiniMax-H3 T2VA Diffusers workflow.

`trace` deliberately keeps every MiniMax-H3 Transformer evaluation exact.  It
only evaluates counterfactual drafts and their risk statistics, so its video
and audio scheduler inputs are identical to `off` for a paired run.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any, Iterable, Sequence

import torch


DIFFUSERS_COMMIT = "9558d918ba740dc8aa18bb06da423ee1aac9fef3"
DEFAULT_MODEL_ID = "MiniMaxAI/MiniMax-H3"
TRACE_ANCHOR_AGES = (1, 2, 3)
EPS = 1e-12


@dataclass
class ExactAnchor:
    """One immutable exact point on a single modality's denoising trajectory."""

    step: int
    timestep: float
    model_input: torch.Tensor
    model_output: torch.Tensor


@dataclass
class ModalityState:
    """Trace state for either video or audio target rows."""

    previous_input: torch.Tensor | None = None
    exact_history: deque[ExactAnchor] = field(default_factory=lambda: deque(maxlen=4))
    budget: float = 0.0
    budget_increments: dict[int, float | None] = field(default_factory=dict)


@dataclass
class AlignCacheH3State:
    """State owned by one CLI invocation; it is never shared across requests."""

    total_steps: int
    last_exact_step: int = -1
    video: ModalityState = field(default_factory=ModalityState)
    audio: ModalityState = field(default_factory=ModalityState)
    counters: dict[str, int] = field(default_factory=lambda: {"exact": 0, "skip": 0})
    reason_counts: dict[str, int] = field(default_factory=dict)
    trace_records: list[dict[str, Any]] = field(default_factory=list)
    transformer_seconds: float = 0.0
    controller_seconds: float = 0.0
    denoise_seconds: float = 0.0
    current_tensor_bytes: int = 0
    peak_tensor_bytes: int = 0


@dataclass
class _PreparedModality:
    input_delta: float | None
    gain: float | None
    risk_increment: float | None


@dataclass
class PendingTraceStep:
    """Snapshots made before an exact forward and finalized after it returns."""

    step: int
    video_input: torch.Tensor
    audio_input: torch.Tensor
    video_timestep: float
    audio_timestep: float
    video: _PreparedModality
    audio: _PreparedModality


def resolve_actual_step_count(timesteps: Sequence[Any]) -> int:
    """Return the number of model evaluations, never the requested sigma-grid size."""

    return len(timesteps)


def _as_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().item())
    return float(value)


def _snapshot(value: torch.Tensor) -> torch.Tensor:
    """Detach and clone because the upstream scheduler writes latent rows in place."""

    return value.detach().clone()


def mean_abs(value: torch.Tensor) -> float:
    return float(value.detach().float().abs().mean().item())


def relative_l2_and_mse(prediction: torch.Tensor, reference: torch.Tensor, eps: float = EPS) -> tuple[float, float]:
    """Return the P0 relative-L2 error and its squared relative-MSE counterpart."""

    numerator = torch.linalg.vector_norm((prediction - reference).detach().float())
    denominator = torch.linalg.vector_norm(reference.detach().float())
    relative_l2 = float((numerator / denominator.clamp_min(eps)).item())
    return relative_l2, relative_l2 * relative_l2


def reuse_draft(anchor: ExactAnchor, current_input: torch.Tensor) -> torch.Tensor:
    """Use the exact anchor velocity without changing target shape or dtype."""

    return anchor.model_output.to(dtype=current_input.dtype)


def residual_draft(anchor: ExactAnchor, current_input: torch.Tensor) -> torch.Tensor:
    """Apply the anchored `Y - X` residual with the P0 fixed alpha of one."""

    draft = current_input + (anchor.model_output - anchor.model_input)
    return draft.to(dtype=current_input.dtype)


def velocity_time_ratio(anchor: ExactAnchor, previous_anchor: ExactAnchor, current_timestep: float, eps: float = EPS) -> float:
    """First-order extrapolation ratio on the modality's own non-uniform timetable."""

    numerator = current_timestep - anchor.timestep
    denominator = max(anchor.timestep - previous_anchor.timestep, eps)
    return numerator / denominator


def velocity_draft(
    anchor: ExactAnchor,
    previous_anchor: ExactAnchor,
    current_input: torch.Tensor,
    current_timestep: float,
) -> torch.Tensor:
    """First-order velocity extrapolation, preserving the current target dtype."""

    ratio = velocity_time_ratio(anchor, previous_anchor, current_timestep)
    draft = anchor.model_output + ratio * (anchor.model_output - previous_anchor.model_output)
    return draft.to(dtype=current_input.dtype)


def _find_anchor(history: Iterable[ExactAnchor], step: int) -> ExactAnchor | None:
    for anchor in history:
        if anchor.step == step:
            return anchor
    return None


def _prepare_modality_step(modality: ModalityState, current_input: torch.Tensor, step: int) -> _PreparedModality:
    input_delta = None
    if modality.previous_input is not None:
        input_delta = mean_abs(current_input - modality.previous_input)

    gain = None
    if len(modality.exact_history) >= 2:
        latest_anchor = modality.exact_history[-1]
        previous_anchor = modality.exact_history[-2]
        output_change = mean_abs(latest_anchor.model_output - previous_anchor.model_output)
        input_change = mean_abs(latest_anchor.model_input - previous_anchor.model_input)
        gain = output_change / max(input_change, EPS)

    risk_increment = None
    if input_delta is not None and gain is not None and modality.exact_history:
        anchor_output_scale = mean_abs(modality.exact_history[-1].model_output)
        risk_increment = gain * input_delta / max(anchor_output_scale, EPS)

    modality.budget_increments[step] = risk_increment
    return _PreparedModality(input_delta=input_delta, gain=gain, risk_increment=risk_increment)


def _budget_since_anchor(modality: ModalityState, anchor_step: int, current_step: int) -> float | None:
    increments = [modality.budget_increments.get(step) for step in range(anchor_step + 1, current_step + 1)]
    if not increments or any(increment is None for increment in increments):
        return None
    return float(sum(increment for increment in increments if increment is not None))


def _empty_candidate(anchor_age: int) -> dict[str, Any]:
    candidate: dict[str, Any] = {
        "anchor_age": anchor_age,
        "anchor_step": None,
        "budget": None,
        "velocity_time_ratio": None,
    }
    for predictor in ("reuse", "residual", "velocity"):
        for metric in ("rel_l2", "rel_mse", "r_l2", "r_mse", "r_quad"):
            candidate[f"{predictor}_{metric}"] = None
    return candidate


def _record_prediction_metrics(
    candidate: dict[str, Any],
    predictor: str,
    draft: torch.Tensor,
    exact_output: torch.Tensor,
    budget: float | None,
) -> None:
    relative_l2, relative_mse = relative_l2_and_mse(draft, exact_output)
    candidate[f"{predictor}_rel_l2"] = relative_l2
    candidate[f"{predictor}_rel_mse"] = relative_mse
    if budget is None:
        return

    denominator = max(budget, EPS)
    candidate[f"{predictor}_r_l2"] = relative_l2 / denominator
    candidate[f"{predictor}_r_mse"] = relative_mse / denominator
    candidate[f"{predictor}_r_quad"] = relative_mse / max(budget * budget, EPS)


def build_shadow_candidates(
    modality: ModalityState,
    current_input: torch.Tensor,
    exact_output: torch.Tensor,
    current_timestep: float,
    step: int,
) -> list[dict[str, Any]]:
    """Measure age-1..3 counterfactual drafts against this exact output.

    The caller invokes this before appending the current exact anchor, which
    makes every candidate teacher-forced from only earlier exact anchors.
    """

    candidates: list[dict[str, Any]] = []
    for age in TRACE_ANCHOR_AGES:
        candidate = _empty_candidate(age)
        anchor = _find_anchor(modality.exact_history, step - age)
        if anchor is None:
            candidates.append(candidate)
            continue

        budget = _budget_since_anchor(modality, anchor.step, step)
        candidate["anchor_step"] = anchor.step
        candidate["budget"] = budget

        draft = reuse_draft(anchor, current_input)
        _record_prediction_metrics(candidate, "reuse", draft, exact_output, budget)
        del draft

        draft = residual_draft(anchor, current_input)
        _record_prediction_metrics(candidate, "residual", draft, exact_output, budget)
        del draft

        previous_anchor = _find_anchor(modality.exact_history, anchor.step - 1)
        if previous_anchor is not None:
            candidate["velocity_time_ratio"] = velocity_time_ratio(anchor, previous_anchor, current_timestep)
            draft = velocity_draft(anchor, previous_anchor, current_input, current_timestep)
            _record_prediction_metrics(candidate, "velocity", draft, exact_output, budget)
            del draft

        candidates.append(candidate)
    return candidates


def _strip_prediction_batch(prediction: torch.Tensor, label: str) -> torch.Tensor:
    if prediction.ndim == 0 or prediction.shape[0] != 1:
        raise ValueError(f"MiniMax-H3 {label} prediction must have batch size 1, got {tuple(prediction.shape)}.")
    return prediction[0]


def _tensor_storage_bytes(value: torch.Tensor, seen: set[tuple[Any, ...]]) -> int:
    try:
        storage = value.untyped_storage()
        key = (value.device.type, value.device.index, storage.data_ptr())
        if key in seen:
            return 0
        seen.add(key)
        return int(storage.nbytes())
    except AttributeError:
        key = (id(value),)
        if key in seen:
            return 0
        seen.add(key)
        return int(value.numel() * value.element_size())


def controller_tensor_bytes(state: AlignCacheH3State) -> int:
    seen: set[tuple[Any, ...]] = set()
    total = 0
    for modality in (state.video, state.audio):
        if modality.previous_input is not None:
            total += _tensor_storage_bytes(modality.previous_input, seen)
        for anchor in modality.exact_history:
            total += _tensor_storage_bytes(anchor.model_input, seen)
            total += _tensor_storage_bytes(anchor.model_output, seen)
    return total


def _record_controller_memory(state: AlignCacheH3State) -> None:
    state.current_tensor_bytes = controller_tensor_bytes(state)
    state.peak_tensor_bytes = max(state.peak_tensor_bytes, state.current_tensor_bytes)


class AlignCacheH3TraceCollector:
    """P0-only trace collector; it contains no route, threshold, or feedback state."""

    def __init__(self, mode: str):
        if mode not in {"off", "trace"}:
            raise ValueError(f"Unsupported AlignCache H3 mode {mode!r}; expected 'off' or 'trace'.")
        self.mode = mode
        self.state: AlignCacheH3State | None = None

    def reset(self, total_steps: int) -> AlignCacheH3State:
        self.state = AlignCacheH3State(total_steps=total_steps)
        return self.state

    def _state(self) -> AlignCacheH3State:
        if self.state is None:
            raise RuntimeError("The H3 trace collector must be reset before a denoising step.")
        return self.state

    @staticmethod
    def _count_exact(state: AlignCacheH3State, step: int, reason: str) -> None:
        state.last_exact_step = step
        state.counters["exact"] = state.counters.get("exact", 0) + 1
        state.reason_counts[reason] = state.reason_counts.get(reason, 0) + 1

    def record_off_exact(self, step: int) -> None:
        self._count_exact(self._state(), step, "off_exact")

    def prepare_exact(
        self,
        step: int,
        video_input: torch.Tensor,
        audio_input: torch.Tensor,
        video_timestep: float,
        audio_timestep: float,
    ) -> PendingTraceStep:
        state = self._state()
        video_snapshot = _snapshot(video_input)
        audio_snapshot = _snapshot(audio_input)
        video = _prepare_modality_step(state.video, video_snapshot, step)
        audio = _prepare_modality_step(state.audio, audio_snapshot, step)
        return PendingTraceStep(
            step=step,
            video_input=video_snapshot,
            audio_input=audio_snapshot,
            video_timestep=video_timestep,
            audio_timestep=audio_timestep,
            video=video,
            audio=audio,
        )

    def complete_exact(
        self,
        pending: PendingTraceStep,
        video_output: torch.Tensor,
        audio_output: torch.Tensor,
        transformer_ms: float,
    ) -> dict[str, Any]:
        state = self._state()
        video_candidates = build_shadow_candidates(
            state.video,
            pending.video_input,
            video_output,
            pending.video_timestep,
            pending.step,
        )
        audio_candidates = build_shadow_candidates(
            state.audio,
            pending.audio_input,
            audio_output,
            pending.audio_timestep,
            pending.step,
        )

        state.video.exact_history.append(
            ExactAnchor(
                step=pending.step,
                timestep=pending.video_timestep,
                model_input=pending.video_input,
                model_output=_snapshot(video_output),
            )
        )
        state.audio.exact_history.append(
            ExactAnchor(
                step=pending.step,
                timestep=pending.audio_timestep,
                model_input=pending.audio_input,
                model_output=_snapshot(audio_output),
            )
        )
        state.video.previous_input = pending.video_input
        state.audio.previous_input = pending.audio_input
        state.video.budget = 0.0
        state.audio.budget = 0.0
        self._count_exact(state, pending.step, "trace_exact")
        _record_controller_memory(state)

        record = {
            "step": pending.step,
            "total_steps": state.total_steps,
            "mode": "trace",
            "video": {
                "input_delta": pending.video.input_delta,
                "gain": pending.video.gain,
                "risk_increment": pending.video.risk_increment,
                "candidates": video_candidates,
            },
            "audio": {
                "input_delta": pending.audio.input_delta,
                "gain": pending.audio.gain,
                "risk_increment": pending.audio.risk_increment,
                "candidates": audio_candidates,
            },
            "transformer_ms": transformer_ms,
            "controller_ms": None,
        }
        state.trace_records.append(record)
        return record

    def add_transformer_seconds(self, seconds: float) -> None:
        self._state().transformer_seconds += seconds

    def add_controller_seconds(self, seconds: float) -> None:
        self._state().controller_seconds += seconds

    def add_denoise_seconds(self, seconds: float) -> None:
        self._state().denoise_seconds += seconds


def _synchronize_for_timing(value: torch.Tensor) -> None:
    if value.is_cuda:
        torch.cuda.synchronize(value.device)


def build_aligncache_h3_loop_denoiser(base_denoiser_cls: type[Any]) -> type[Any]:
    """Build the fixed-commit block subclass without importing Diffusers at module import time."""

    class AlignCacheH3LoopDenoiser(base_denoiser_cls):
        def __init__(self, mode: str):
            super().__init__(transformer_name="transformer")
            self.mode = mode
            self.collector = AlignCacheH3TraceCollector(mode)

        @torch.no_grad()
        def __call__(self, components, block_state, i: int, t: torch.Tensor):
            if i == 0:
                self.collector.reset(resolve_actual_step_count(block_state.timesteps))

            block_state._aligncache_h3_collector = self.collector
            block_state._aligncache_h3_step_started_at = perf_counter()

            pending = None
            controller_before = 0.0
            if self.mode == "trace":
                _synchronize_for_timing(block_state.latents)
                controller_start = perf_counter()
                pending = self.collector.prepare_exact(
                    step=i,
                    video_input=block_state.latents,
                    audio_input=block_state.audio_latents,
                    video_timestep=_as_float(block_state.timesteps[i]),
                    audio_timestep=_as_float(block_state.audio_timesteps[i]),
                )
                _synchronize_for_timing(block_state.latents)
                controller_before = perf_counter() - controller_start

            _synchronize_for_timing(block_state.latents)
            transformer_start = perf_counter()
            components, block_state = super().__call__(components, block_state, i, t)
            _synchronize_for_timing(block_state.latents)
            transformer_seconds = perf_counter() - transformer_start
            self.collector.add_transformer_seconds(transformer_seconds)

            if self.mode == "off":
                self.collector.record_off_exact(i)
                return components, block_state

            assert pending is not None
            _synchronize_for_timing(block_state.latents)
            controller_start = perf_counter()
            record = self.collector.complete_exact(
                pending=pending,
                video_output=_strip_prediction_batch(block_state.noise_pred, "video"),
                audio_output=_strip_prediction_batch(block_state.audio_noise_pred, "audio"),
                transformer_ms=transformer_seconds * 1000.0,
            )
            _synchronize_for_timing(block_state.latents)
            controller_seconds = controller_before + perf_counter() - controller_start
            self.collector.add_controller_seconds(controller_seconds)
            record["controller_ms"] = controller_seconds * 1000.0
            return components, block_state

    AlignCacheH3LoopDenoiser.__name__ = "AlignCacheH3LoopDenoiser"
    return AlignCacheH3LoopDenoiser


def build_aligncache_h3_loop_scheduler_step(base_scheduler_step_cls: type[Any]) -> type[Any]:
    """Time the unchanged two-scheduler update after it has completed."""

    class AlignCacheH3LoopSchedulerStep(base_scheduler_step_cls):
        @torch.no_grad()
        def __call__(self, components, block_state, i: int, t: torch.Tensor):
            components, block_state = super().__call__(components, block_state, i, t)
            _synchronize_for_timing(block_state.latents)
            collector = getattr(block_state, "_aligncache_h3_collector", None)
            started_at = getattr(block_state, "_aligncache_h3_step_started_at", None)
            if collector is not None and started_at is not None:
                collector.add_denoise_seconds(perf_counter() - started_at)
            return components, block_state

    AlignCacheH3LoopSchedulerStep.__name__ = "AlignCacheH3LoopSchedulerStep"
    return AlignCacheH3LoopSchedulerStep


def _quantile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def summarize_predictor_errors(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Summarize the median and worst-decile relative-L2 by modality/age/predictor."""

    values: dict[str, dict[int, dict[str, list[float]]]] = {
        modality: {age: {predictor: [] for predictor in ("reuse", "residual", "velocity")} for age in TRACE_ANCHOR_AGES}
        for modality in ("video", "audio")
    }
    for record in records:
        for modality in ("video", "audio"):
            for candidate in record[modality]["candidates"]:
                age = int(candidate["anchor_age"])
                for predictor in ("reuse", "residual", "velocity"):
                    value = candidate[f"{predictor}_rel_l2"]
                    if value is not None:
                        values[modality][age][predictor].append(float(value))

    summary: dict[str, Any] = {}
    for modality, by_age in values.items():
        summary[modality] = {}
        for age, by_predictor in by_age.items():
            summary[modality][f"age_{age}"] = {}
            for predictor, predictor_values in by_predictor.items():
                summary[modality][f"age_{age}"][predictor] = {
                    "count": len(predictor_values),
                    "median": _quantile(predictor_values, 0.5),
                    "worst_decile": _quantile(predictor_values, 0.9),
                    "max": max(predictor_values) if predictor_values else None,
                }
    return summary


def _dependency_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"python": platform.python_version(), "torch": torch.__version__}
    for package in ("diffusers", "transformers", "huggingface_hub"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _hardware_summary() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    return {
        "cuda_available": True,
        "gpu": torch.cuda.get_device_name(torch.cuda.current_device()),
        "gpu_count": torch.cuda.device_count(),
    }


def build_summary(
    state: AlignCacheH3State,
    args: argparse.Namespace,
    model_snapshot: dict[str, Any],
    end_to_end_seconds: float,
) -> dict[str, Any]:
    return {
        "mode": args.aligncache_mode,
        "seed": args.seed,
        "model_snapshot": model_snapshot,
        "diffusers_commit": DIFFUSERS_COMMIT,
        "parameters": {
            "model_id": args.model_id,
            "model_revision": args.model_revision,
            "prompt": args.prompt,
            "num_frames": args.num_frames,
            "num_inference_steps": args.num_inference_steps,
        },
        "actual_timesteps": state.total_steps,
        "exact_count": state.counters.get("exact", 0),
        "skip_count": state.counters.get("skip", 0),
        "reason_counts": state.reason_counts,
        "trace_rows": len(state.trace_records),
        "predictor_relative_l2": summarize_predictor_errors(state.trace_records),
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
        "hardware": _hardware_summary(),
        "dependencies": _dependency_versions(),
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def _load_minimax_h3_api() -> SimpleNamespace:
    """Import only on execution so unit tests do not need the 33B runtime dependency."""

    try:
        from diffusers.modular_pipelines.minimax_h3.denoise import (
            MiniMaxH3LoopDenoiser,
            MiniMaxH3LoopSchedulerStep,
        )
        from diffusers.modular_pipelines.minimax_h3.modular_blocks_minimax_h3 import MiniMaxH3Blocks
        from diffusers.modular_pipelines.minimax_h3.modular_pipeline import MiniMaxH3ModularPipeline
        from diffusers.utils.export_utils import encode_video
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "MiniMax-H3 requires Diffusers commit "
            f"{DIFFUSERS_COMMIT}. Install the pinned commit documented in AlignCache4MiniMaxH3/README.md."
        ) from exc
    return SimpleNamespace(
        MiniMaxH3Blocks=MiniMaxH3Blocks,
        MiniMaxH3LoopDenoiser=MiniMaxH3LoopDenoiser,
        MiniMaxH3LoopSchedulerStep=MiniMaxH3LoopSchedulerStep,
        MiniMaxH3ModularPipeline=MiniMaxH3ModularPipeline,
        encode_video=encode_video,
    )


def resolve_model_snapshot(model_id: str, requested_revision: str | None) -> dict[str, Any]:
    """Resolve one immutable Hub SHA, or record the explicit local snapshot path."""

    local_path = Path(model_id).expanduser()
    if local_path.exists():
        if requested_revision is not None:
            raise ValueError("--model_revision applies to Hub model IDs; a local snapshot already selects its files.")
        return {
            "source": "local",
            "model_id": str(local_path.resolve()),
            "requested_revision": None,
            "resolved_revision": None,
        }

    from huggingface_hub import HfApi

    requested = requested_revision or "main"
    info = HfApi().model_info(model_id, revision=requested)
    return {
        "source": "hub",
        "model_id": model_id,
        "requested_revision": requested,
        "resolved_revision": info.sha,
    }


def _build_t2va_pipeline(api: SimpleNamespace, args: argparse.Namespace, model_snapshot: dict[str, Any]):
    blocks = api.MiniMaxH3Blocks().get_workflow("t2va")
    loop = blocks.sub_blocks["denoise.denoise"]
    denoiser_cls = build_aligncache_h3_loop_denoiser(api.MiniMaxH3LoopDenoiser)
    scheduler_step_cls = build_aligncache_h3_loop_scheduler_step(api.MiniMaxH3LoopSchedulerStep)
    loop.sub_blocks["denoiser"] = denoiser_cls(args.aligncache_mode)
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


def _installed_trace_collector(pipe) -> AlignCacheH3TraceCollector:
    denoiser = pipe._blocks.sub_blocks["denoise.denoise"].sub_blocks["denoiser"]
    return denoiser.collector


def run_generation(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("MiniMax-H3 P0 runs only on the fixed resident BF16 CUDA path.")

    api = _load_minimax_h3_api()
    model_snapshot = resolve_model_snapshot(args.model_id, args.model_revision)
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

    collector = _installed_trace_collector(pipe)
    if collector.state is None:
        raise RuntimeError("MiniMax-H3 denoiser did not initialize the AlignCache trace state.")
    summary = build_summary(collector.state, args, model_snapshot, end_to_end_seconds)
    summary["output"] = str(output_path)

    if args.aligncache_mode == "trace" and args.trace_jsonl is not None:
        _write_jsonl(Path(args.trace_jsonl), collector.state.trace_records)
    if args.summary_json is not None:
        _write_json(Path(args.summary_json), summary)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MiniMax-H3 T2VA with AlignCache P0 off/trace modes.")
    parser.add_argument("--model_id", default=DEFAULT_MODEL_ID, help="Hugging Face model ID or fixed local snapshot.")
    parser.add_argument(
        "--model_revision",
        default=None,
        help="Optional Hub revision; it is resolved to and loaded by its immutable commit SHA.",
    )
    parser.add_argument("--prompt", required=True, help="T2VA prompt.")
    parser.add_argument("--num_frames", type=int, default=124, help="Requested MiniMax-H3 frame count.")
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=20,
        help="Sigma-grid point count (default: 20); the denoiser runs once per resulting timestep.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the shared MiniMax-H3 generator.")
    parser.add_argument(
        "--aligncache_mode",
        choices=("off", "trace"),
        default="off",
        help="off uses exact upstream denoising; trace adds counterfactual P0 statistics only.",
    )
    parser.add_argument("--trace_jsonl", default=None, help="Optional per-step P0 trace JSONL path.")
    parser.add_argument("--summary_json", default=None, help="Optional run summary JSON path.")
    parser.add_argument("--output", required=True, help="Muxed audio-video output path, normally .mp4.")
    args = parser.parse_args(argv)
    if args.num_frames < 1:
        parser.error("--num_frames must be positive.")
    if args.num_inference_steps < 2:
        parser.error("--num_inference_steps must be at least 2 because H3 keeps the terminal sigma zero.")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_generation(args)
    print(
        json.dumps(
            {
                "mode": summary["mode"],
                "output": summary["output"],
                "actual_timesteps": summary["actual_timesteps"],
                "exact_count": summary["exact_count"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
