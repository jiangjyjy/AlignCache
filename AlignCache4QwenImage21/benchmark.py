"""Optional measurement-only nodes for paired native/AlignCache benchmarks.

They neither route nor predict. Normal measurements synchronize only at outer
boundaries; explicit profile mode records CUDA events and resolves them once.
"""

from __future__ import annotations

import inspect
import json
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

BENCHMARK_KEY = "aligncache_qwen21_measurement"
BENCHMARK_CONFIG_KEY = BENCHMARK_KEY + "_config"
BENCHMARK_STATE_KEY = BENCHMARK_KEY + "_state"
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MeasurementConfig:
    output_root: str
    run_nonce: str
    profile: bool = False


@dataclass
class MeasurementState:
    config: MeasurementConfig
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    forward_attempts: int = 0
    forward_completed: int = 0
    events: list = field(default_factory=list)
    rows: list = field(default_factory=list)


def benchmark_model_wrapper(
    executor,
    x,
    timestep,
    context,
    ref_latents=None,
    image_slots=None,
    transformer_options=None,
    **kwargs,
):
    state = (transformer_options or {}).get(BENCHMARK_STATE_KEY)

    def exact():
        return executor(
            x,
            timestep,
            context,
            ref_latents,
            image_slots,
            transformer_options,
            **kwargs,
        )

    if state is None:
        return exact()
    index = state.forward_attempts
    state.forward_attempts += 1
    events = None
    if state.config.profile:
        if x.device.type == "cuda":
            import torch

            stream = torch.cuda.current_stream(x.device)
            events = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            events[0].record(stream)
        started = perf_counter()
    try:
        result = exact()
        state.forward_completed += 1
        return result
    finally:
        if state.config.profile:
            host_seconds = perf_counter() - started
            if events is not None:
                try:
                    events[1].record(stream)
                except Exception:
                    logger.exception("Could not record the final profiling event")
                    events = None
            state.events.append(events)
            state.rows.append(
                {
                    "forward_index": index,
                    "host_seconds": host_seconds,
                    "cuda_seconds": None,
                }
            )


def write_measurement(state, summary):
    destination = (
        Path(state.config.output_root) / "aligncache_qwen21_benchmark" / state.run_id
    )
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "measurement.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def benchmark_outer_wrapper(executor, *args, **kwargs):
    import torch
    from comfy.model_patcher import create_model_options_clone

    guider = executor.class_obj
    original = guider.model_options
    config = original["transformer_options"][BENCHMARK_CONFIG_KEY]
    state = MeasurementState(config)
    bound = inspect.signature(executor.original).bind(*args, **kwargs)
    bound.apply_defaults()
    sigmas = bound.arguments["sigmas"]
    device = torch.device(guider.model_patcher.load_device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError(
            "Qwen21 benchmark timing currently supports CPU and CUDA devices only"
        )
    options = create_model_options_clone(original)
    options["transformer_options"][BENCHMARK_STATE_KEY] = state
    guider.model_options = options
    status, failure = "failed", None
    sampling_seconds = None
    start = None
    baseline_allocated = peak_allocated = None
    try:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            baseline_allocated = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
        start = perf_counter()
        result = executor(*args, **kwargs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        sampling_seconds = perf_counter() - start
        status = "completed"
        return result
    except BaseException as error:
        status = (
            "interrupted"
            if isinstance(error, KeyboardInterrupt)
            or "Interrupt" in type(error).__name__
            else "failed"
        )
        failure = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        guider.model_options = original
        # Diagnostic failure, including failed CUDA event resolution, must not
        # replace a model exception or keep references to device events alive.
        try:
            if sampling_seconds is None and start is not None:
                sampling_seconds = perf_counter() - start
            if device.type == "cuda":
                peak_allocated = torch.cuda.max_memory_allocated(device)
                if status == "completed":
                    for row, events in zip(state.rows, state.events):
                        if events is not None:
                            row["cuda_seconds"] = (
                                events[0].elapsed_time(events[1]) / 1000.0
                            )
            summary = {
                "schema_version": 1,
                "run_id": state.run_id,
                "run_nonce": config.run_nonce,
                "status": status,
                "failure": failure,
                "profile": config.profile,
                "performance_sample": not config.profile,
                "planned_steps": max(0, len(sigmas) - 1),
                "forward_attempts": state.forward_attempts,
                "forward_completed": state.forward_completed,
                "sampling_seconds": sampling_seconds,
                "device": str(device),
                "memory_allocated_at_entry_bytes": baseline_allocated,
                "peak_allocated_bytes": peak_allocated,
                "peak_above_entry_bytes": max(0, peak_allocated - baseline_allocated)
                if peak_allocated is not None
                else None,
                "memory_scope": "outer sample, including loading/cleanup; resets process CUDA peak counters; serial benchmarks only",
                "timing_scope": "outer sample, including AlignCache setup/cleanup/logging; excludes measurement export",
                "forwards": state.rows,
            }
            write_measurement(state, summary)
        except Exception:
            logger.exception("Could not export Qwen21 benchmark measurement")
        finally:
            state.events.clear()
            state.rows.clear()


def trusted_measurement_wrapper(kind, key, wrappers):
    if key != BENCHMARK_KEY:
        return False
    expected = {
        "outer_sample": benchmark_outer_wrapper,
        "diffusion_model": benchmark_model_wrapper,
    }.get(kind)
    return expected is not None and len(wrappers) == 1 and wrappers[0] is expected


class AlignCacheQwenImage21Benchmark:
    """Attach last, for serial experiments only. Does not enable acceleration."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "run_nonce": ("STRING", {"default": ""}),
                "profile": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "attach"
    CATEGORY = "AlignCache/Qwen-Image-2.1/Experiments"

    def attach(self, model, run_nonce="", profile=False):
        import folder_paths
        from comfy.patcher_extension import WrappersMP

        from .runtime import require_model

        require_model(model)
        patched = model.clone()
        options = patched.model_options.setdefault("transformer_options", {})
        options.pop(BENCHMARK_STATE_KEY, None)
        options[BENCHMARK_CONFIG_KEY] = MeasurementConfig(
            folder_paths.get_output_directory(), run_nonce, bool(profile)
        )
        for kind, wrapper in [
            (WrappersMP.OUTER_SAMPLE, benchmark_outer_wrapper),
            (WrappersMP.DIFFUSION_MODEL, benchmark_model_wrapper),
        ]:
            patched.remove_wrappers_with_key(kind, BENCHMARK_KEY)
            options.get("wrappers", {}).get(kind, {}).pop(BENCHMARK_KEY, None)
            patched.add_wrapper_with_key(kind, BENCHMARK_KEY, wrapper)
        # Outer timer surrounds AlignCache setup/cleanup. The inner observer
        # follows routing so skipped evaluations produce no native-forward event.
        outer = patched.wrappers[WrappersMP.OUTER_SAMPLE]
        patched.wrappers[WrappersMP.OUTER_SAMPLE] = {
            BENCHMARK_KEY: outer[BENCHMARK_KEY],
            **{k: v for k, v in outer.items() if k != BENCHMARK_KEY},
        }
        return (patched,)
