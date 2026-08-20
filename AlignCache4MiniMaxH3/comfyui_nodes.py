"""MiniMax-H3 trace and closed-loop joint-cache node for ComfyUI.

The node adds an ``OUTER_SAMPLE`` wrapper for per-run state and a
``DIFFUSION_MODEL`` wrapper around ComfyUI's MiniMax-H3 model.  Its trace mode
always delegates to the original transformer once and returns its exact
``[video, audio]`` output unchanged. The separate experimental ``on`` mode
uses the same phase-aware residual-momentum predictor and counterfactual
feedback controller as the Diffusers accelerator. Both modality budgets must
permit a skip; either modality can force the one joint exact forward.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

from . import aligncache_generate as core
from . import aligncache_accelerate as accelerate


CONFIG_KEY = "aligncache_h3_comfy_config"
RUN_STATE_KEY = "aligncache_h3_comfy_run_state"
WRAPPER_KEY = "aligncache_h3_p0_trace"


@dataclass(frozen=True)
class ComfyH3TraceConfig:
    """Configuration which can safely be shared by cloned Comfy model patchers.

    A zero risk threshold deliberately disables skips for that modality. This
    keeps the default experimental route exact until H3-specific thresholds
    have been selected.
    """

    mode: str
    trace_jsonl: str
    summary_json: str
    predictor: str = "scaled_residual_momentum"
    video_risk_threshold: float = 0.33
    audio_risk_threshold: float = 0.165
    max_skip_age: int = 4
    start_skip_step: int = 5


class ComfyH3OnController(accelerate.AlignCacheH3Controller):
    """Comfy adapter for the shared Wan2.1-style H3 controller."""

    def __init__(self, config: ComfyH3TraceConfig, planned_steps: int):
        if config.predictor != "scaled_residual_momentum":
            raise ValueError("MiniMax-H3 on mode uses the scaled_residual_momentum predictor.")
        uniform_age_override = int(config.max_skip_age)
        phase_age_caps = (
            accelerate.PHASE_AGE_CAPS
            if uniform_age_override <= 0 or uniform_age_override == accelerate.MAX_SKIP_AGE
            else (uniform_age_override,) * 3
        )
        controller_config = accelerate.AlignCacheH3Config(
            video_base_threshold=float(config.video_risk_threshold),
            audio_base_threshold=float(config.audio_risk_threshold),
            warmup_steps=accelerate.WARMUP_STEPS,
            tail_steps=accelerate.TAIL_STEPS,
            start_skip_step=int(config.start_skip_step),
            age_caps=phase_age_caps,
        )
        super().__init__(controller_config)
        self.comfy_config = config
        self.reset(planned_steps)

    @property
    def route_records(self) -> list[dict[str, Any]]:
        return self._state().trace_records


@dataclass
class ComfyH3RunState:
    """Mutable state scoped to one Comfy sampler invocation."""

    config: ComfyH3TraceConfig
    collector: core.AlignCacheH3TraceCollector | None
    on_controller: ComfyH3OnController | None
    next_step: int = 0
    planned_steps: int = 0

    @classmethod
    def begin(cls, config: ComfyH3TraceConfig, sigmas: Any) -> "ComfyH3RunState":
        planned_steps = max(len(sigmas) - 1, 0)
        if config.mode == "on":
            return cls(
                config=config,
                collector=None,
                on_controller=ComfyH3OnController(config, planned_steps),
                planned_steps=planned_steps,
            )
        collector = core.AlignCacheH3TraceCollector(config.mode)
        collector.reset(planned_steps)
        return cls(config=config, collector=collector, on_controller=None, planned_steps=planned_steps)

    def state(self) -> core.AlignCacheH3State:
        if self.on_controller is not None:
            return self.on_controller.state
        if self.collector is None:
            raise RuntimeError("MiniMax-H3 run state has neither a trace collector nor an on controller.")
        return self.collector._state()

    def records(self) -> list[dict[str, Any]]:
        if self.on_controller is not None:
            return self.on_controller.route_records
        return self.state().trace_records

    def finish(self) -> core.AlignCacheH3State:
        state = self.state()
        actual_steps = self.next_step
        state.total_steps = actual_steps
        for record in state.trace_records:
            record["total_steps"] = actual_steps
        for record in self.records():
            record["total_steps"] = actual_steps
        return state


def _time_shift_sigma(sigma: float, from_shift: float, to_shift: float) -> float:
    """Match ComfyUI MiniMax-H3's closed-form video-to-audio sigma mapping."""

    base = sigma / (from_shift + sigma * (1.0 - from_shift))
    return to_shift * base / (1.0 + (to_shift - 1.0) * base)


def _modality_timesteps(executor: Any, timestep: torch.Tensor, transformer_options: dict[str, Any]) -> tuple[float, float]:
    """Read Comfy's sigma grid and derive H3's per-stream model timesteps."""

    video_sigma = core._as_float(timestep) / 1000.0
    model = executor.class_obj
    video_shift = float(
        transformer_options.get("minimax_h3_sigma_shift_video", getattr(model, "sigma_shift_video", 12.0))
    )
    audio_shift = float(
        transformer_options.get("minimax_h3_sigma_shift_audio", getattr(model, "sigma_shift_audio", 3.0))
    )
    audio_sigma = _time_shift_sigma(video_sigma, video_shift, audio_shift)
    return 1.0 - video_sigma, 1.0 - audio_sigma


def _run_state(transformer_options: dict[str, Any]) -> ComfyH3RunState | None:
    state = transformer_options.get(RUN_STATE_KEY)
    if state is None:
        return None
    if not isinstance(state, ComfyH3RunState):
        raise TypeError(f"{RUN_STATE_KEY} must be a ComfyH3RunState.")
    return state


def _run_on_diffusion_model_step(
    run_state: ComfyH3RunState,
    executor: Any,
    x: list[torch.Tensor],
    timestep: torch.Tensor,
    context: torch.Tensor,
    transformer_options: dict[str, Any],
    minimax_payload: dict[str, Any] | None,
    **kwargs: Any,
) -> list[torch.Tensor]:
    """Run one atomic H3 on-route: one joint exact call or two cached drafts."""

    controller = run_state.on_controller
    if controller is None:
        raise RuntimeError("MiniMax-H3 on mode requires its per-run on controller.")
    step = run_state.next_step
    video_timestep, audio_timestep = _modality_timesteps(executor, timestep, transformer_options)

    core._synchronize_for_timing(x[0])
    controller_start = perf_counter()
    decision = controller.decide(step, x[0], x[1], video_timestep, audio_timestep)
    feedback_drafts = controller.build_feedback_drafts(decision)
    core._synchronize_for_timing(x[0])
    controller_before = perf_counter() - controller_start

    if decision.action == "skip":
        core._synchronize_for_timing(x[0])
        controller_start = perf_counter()
        output = list(controller.build_skip_output(decision))
        record = controller.record_skip(decision)
        core._synchronize_for_timing(x[0])
        controller_seconds = controller_before + perf_counter() - controller_start
        controller.state.controller_seconds += controller_seconds
        record["controller_ms"] = controller_seconds * 1000.0
        run_state.next_step += 1
        return output

    core._synchronize_for_timing(x[0])
    transformer_start = perf_counter()
    output = executor(x, timestep, context, transformer_options, minimax_payload=minimax_payload, **kwargs)
    core._synchronize_for_timing(x[0])
    transformer_seconds = perf_counter() - transformer_start
    controller.state.transformer_seconds += transformer_seconds

    core._synchronize_for_timing(x[0])
    controller_start = perf_counter()
    record = controller.record_exact(
        decision,
        video_output=output[0],
        audio_output=output[1],
        transformer_ms=transformer_seconds * 1000.0,
        feedback_drafts=feedback_drafts,
    )
    core._synchronize_for_timing(x[0])
    controller_seconds = controller_before + perf_counter() - controller_start
    controller.state.controller_seconds += controller_seconds
    record["controller_ms"] = controller_seconds * 1000.0
    run_state.next_step += 1
    return output


def minimax_h3_diffusion_model_wrapper(
    executor: Any,
    x: list[torch.Tensor],
    timestep: torch.Tensor,
    context: torch.Tensor,
    transformer_options: dict[str, Any],
    minimax_payload: dict[str, Any] | None = None,
    **kwargs: Any,
) -> list[torch.Tensor]:
    """Run one P0 exact step or one experimental atomic joint-cache route."""

    run_state = _run_state(transformer_options)
    if run_state is None:
        return executor(x, timestep, context, transformer_options, minimax_payload=minimax_payload, **kwargs)

    if not isinstance(x, (list, tuple)) or len(x) != 2:
        raise ValueError("AlignCache MiniMax-H3 expects [video_latents, audio_latents].")

    if run_state.config.mode == "on":
        return _run_on_diffusion_model_step(
            run_state,
            executor,
            x,
            timestep,
            context,
            transformer_options,
            minimax_payload,
            **kwargs,
        )

    collector = run_state.collector
    if collector is None:
        raise RuntimeError("MiniMax-H3 trace mode requires its per-run trace collector.")
    step = run_state.next_step
    video_timestep, audio_timestep = _modality_timesteps(executor, timestep, transformer_options)

    pending = None
    controller_before = 0.0
    if run_state.config.mode == "trace":
        core._synchronize_for_timing(x[0])
        controller_start = perf_counter()
        pending = collector.prepare_exact(
            step=step,
            video_input=x[0],
            audio_input=x[1],
            video_timestep=video_timestep,
            audio_timestep=audio_timestep,
        )
        core._synchronize_for_timing(x[0])
        controller_before = perf_counter() - controller_start

    core._synchronize_for_timing(x[0])
    transformer_start = perf_counter()
    output = executor(x, timestep, context, transformer_options, minimax_payload=minimax_payload, **kwargs)
    core._synchronize_for_timing(x[0])
    transformer_seconds = perf_counter() - transformer_start
    collector.add_transformer_seconds(transformer_seconds)

    if run_state.config.mode == "off":
        collector.record_off_exact(step)
    else:
        assert pending is not None
        core._synchronize_for_timing(x[0])
        controller_start = perf_counter()
        record = collector.complete_exact(
            pending=pending,
            video_output=core._strip_prediction_batch(output[0], "video"),
            audio_output=core._strip_prediction_batch(output[1], "audio"),
            transformer_ms=transformer_seconds * 1000.0,
        )
        core._synchronize_for_timing(x[0])
        controller_seconds = controller_before + perf_counter() - controller_start
        collector.add_controller_seconds(controller_seconds)
        record["controller_ms"] = controller_seconds * 1000.0

    run_state.next_step += 1
    return output


def _relative_output_path(folder_paths: Any, filename: str) -> Path | None:
    if not filename:
        return None
    path = Path(filename)
    if path.is_absolute():
        return path
    return Path(folder_paths.get_output_directory()) / path


def _synchronize_cuda_if_available() -> None:
    """Make outer-sampler timing comparable without requiring a latent argument."""

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _write_comfy_summary(path: Path, run_state: ComfyH3RunState, executor: Any) -> None:
    state = run_state.finish()
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "backend": "comfyui",
        "mode": run_state.config.mode,
        "model_class": executor.class_obj.model_patcher.model.diffusion_model.__class__.__name__,
        "planned_timesteps": run_state.planned_steps,
        "actual_timesteps": state.total_steps,
        "exact_count": state.counters.get("exact", 0),
        "skip_count": state.counters.get("skip", 0),
        "reason_counts": state.reason_counts,
        "trace_rows": len(state.trace_records),
        "route_rows": len(run_state.records()),
        "predictor_relative_l2": (
            core.summarize_predictor_errors(state.trace_records) if run_state.config.mode == "trace" else {}
        ),
        "timings_ms": {
            "transformer": state.transformer_seconds * 1000.0,
            "controller": state.controller_seconds * 1000.0,
            "denoise": state.denoise_seconds * 1000.0,
        },
        "controller_cache": {
            "current_tensor_bytes": state.current_tensor_bytes,
            "peak_tensor_bytes": state.peak_tensor_bytes,
        },
        "hardware": core._hardware_summary(),
        "dependencies": core._dependency_versions(),
    }
    if run_state.config.mode == "on":
        assert run_state.on_controller is not None
        summary["on_config"] = run_state.on_controller.config.as_dict()
        summary["on_feedback"] = run_state.on_controller.feedback_summary()
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def make_outer_sample_wrapper(folder_paths: Any):
    """Create the Comfy ``OUTER_SAMPLE`` wrapper with per-sample trace state."""

    def outer_sample_wrapper(executor: Any, *args: Any, **kwargs: Any):
        guider = executor.class_obj
        original_model_options = guider.model_options
        transformer_options = original_model_options["transformer_options"]
        config = transformer_options[CONFIG_KEY]
        if not isinstance(config, ComfyH3TraceConfig):
            raise TypeError(f"{CONFIG_KEY} must be a ComfyH3TraceConfig.")
        if float(guider.cfg) != 1.0:
            raise ValueError("MiniMax-H3 trace/on requires CFG 1.0 so each step has one joint transformer call.")

        sigmas = args[3]
        run_state = ComfyH3RunState.begin(config, sigmas)
        from comfy.model_patcher import create_model_options_clone

        guider.model_options = create_model_options_clone(original_model_options)
        guider.model_options["transformer_options"][RUN_STATE_KEY] = run_state
        try:
            _synchronize_cuda_if_available()
            denoise_start = perf_counter()
            output = executor(*args, **kwargs)
            _synchronize_cuda_if_available()
            run_state.state().denoise_seconds += perf_counter() - denoise_start
        except Exception:
            raise
        else:
            state = run_state.finish()
            trace_path = _relative_output_path(folder_paths, config.trace_jsonl)
            summary_path = _relative_output_path(folder_paths, config.summary_json)
            if config.mode in {"trace", "on"} and trace_path is not None:
                core._write_jsonl(trace_path, run_state.records())
            if summary_path is not None:
                _write_comfy_summary(summary_path, run_state, executor)
            return output
        finally:
            guider.model_options = original_model_options

    return outer_sample_wrapper


class AlignCacheH3ComfyTrace:
    """Attach the MiniMax-H3 trace or experimental joint-cache wrapper."""

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "model": ("MODEL",),
                "aligncache_mode": (["off", "trace", "on"], {"default": "trace"}),
                "trace_jsonl": ("STRING", {"default": "aligncache_h3_trace.jsonl"}),
                "summary_json": ("STRING", {"default": "aligncache_h3_summary.json"}),
                "aligncache_predictor": (["scaled_residual_momentum"], {"default": "scaled_residual_momentum"}),
                "video_risk_threshold": ("FLOAT", {"default": 0.33, "min": 0.0, "step": 0.0001}),
                "audio_risk_threshold": ("FLOAT", {"default": 0.165, "min": 0.0, "step": 0.0001}),
                "max_skip_age": ("INT", {"default": 4, "min": 0, "max": 32}),
                "start_skip_step": ("INT", {"default": 5, "min": 0, "max": 10000}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "attach"
    CATEGORY = "AlignCache/MiniMax-H3"

    def attach(
        self,
        model: Any,
        aligncache_mode: str,
        trace_jsonl: str,
        summary_json: str,
        aligncache_predictor: str = "scaled_residual_momentum",
        video_risk_threshold: float = 0.33,
        audio_risk_threshold: float = 0.165,
        max_skip_age: int = 4,
        start_skip_step: int = 5,
    ):
        if aligncache_mode not in {"off", "trace", "on"}:
            raise ValueError("MiniMax-H3 supports only 'off', 'trace', and experimental 'on'.")

        import comfy.patcher_extension
        import folder_paths

        patched = model.clone()
        transformer_options = patched.model_options.setdefault("transformer_options", {})
        transformer_options[CONFIG_KEY] = ComfyH3TraceConfig(
            mode=aligncache_mode,
            trace_jsonl=trace_jsonl,
            summary_json=summary_json,
            predictor=aligncache_predictor,
            video_risk_threshold=float(video_risk_threshold),
            audio_risk_threshold=float(audio_risk_threshold),
            max_skip_age=int(max_skip_age),
            start_skip_step=int(start_skip_step),
        )
        patched.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
            WRAPPER_KEY,
            make_outer_sample_wrapper(folder_paths),
        )
        patched.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
            WRAPPER_KEY,
            minimax_h3_diffusion_model_wrapper,
        )
        return (patched,)


NODE_CLASS_MAPPINGS = {"AlignCacheH3ComfyTrace": AlignCacheH3ComfyTrace}
NODE_DISPLAY_NAME_MAPPINGS = {"AlignCacheH3ComfyTrace": "AlignCache MiniMax-H3 Trace / Experimental On"}
