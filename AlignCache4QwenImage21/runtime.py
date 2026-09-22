"""Compatibility, condition signatures and buffered runtime reporting.

Keep discovery lightweight: Torch and Comfy imports belong inside functions.
"""

from __future__ import annotations

import json
import logging
import math
import platform
from dataclasses import asdict
from pathlib import Path

WRAPPER_KEY = "aligncache_qwen21"
CONFIG_KEY = WRAPPER_KEY + "_config"
RUN_STATE_KEY = WRAPPER_KEY + "_run"
COMFY_COMMIT = "b0f4b7b294ce482a2e071d9d762c133d38c7aa07"
CONDITION_FIELDS = frozenset(
    {
        "model_conds",
        "cross_attn",
        "pooled_output",
        "attention_mask",
        "image_slots",
        "uuid",
        "reference_latents",
    }
)


def wrapper_reason(wrappers):
    from .benchmark import BENCHMARK_KEY, trusted_measurement_wrapper

    for kind, groups in wrappers.items():
        for key, value in groups.items():
            if (
                value
                and key != WRAPPER_KEY
                and not trusted_measurement_wrapper(kind, key, value)
            ):
                return "external_wrapper"
        if (
            kind == "diffusion_model"
            and groups.get(WRAPPER_KEY)
            and groups.get(BENCHMARK_KEY)
        ):
            if list(groups).index(BENCHMARK_KEY) < list(groups).index(WRAPPER_KEY):
                return "measurement_must_follow_routing"
    return None


def require_model(patcher):
    cls = type(patcher.model.diffusion_model)
    if (
        cls.__name__ != "QwenImage21Transformer2DModel"
        or cls.__module__ != "comfy.ldm.qwen_image21.model"
    ):
        raise ValueError(
            "AlignCacheQwenImage21 requires the native QwenImage21Transformer2DModel"
        )


def option_reason(options):
    from .benchmark import BENCHMARK_CONFIG_KEY, BENCHMARK_STATE_KEY

    known = {
        CONFIG_KEY,
        RUN_STATE_KEY,
        BENCHMARK_CONFIG_KEY,
        BENCHMARK_STATE_KEY,
        "patches",
        "patches_replace",
        "wrappers",
        "callbacks",
        "cond_or_uncond",
        "uuids",
        "sigmas",
        "sample_sigmas",
        "qwen_image21_cache",
        "prefetch_dynamic_vbars",
        "total_blocks",
        "block_type",
        "block_index",
    }
    if set(options) - known:
        return "unknown_transformer_options"
    if options.get("patches") or options.get("patches_replace"):
        return "transformer_patches"
    reason = wrapper_reason(options.get("wrappers", {}))
    if reason:
        return reason
    if options.get("callbacks"):
        return "external_callback"
    if options.get("prefetch_dynamic_vbars"):
        return "dynamic_vram_unvalidated"
    return None


def run_reason(guider, sampler, noise, sigmas, denoise_mask=None):
    import torch

    config = guider.model_options.get("transformer_options", {}).get(CONFIG_KEY)
    paper = config is not None and config.policy == "paper"
    cfg = float(getattr(guider, "cfg", float("nan")))
    if not math.isfinite(cfg) or cfg < 1 or (not paper and cfg != 1):
        return "unsupported_cfg"
    from comfy.k_diffusion.sampling import sample_euler

    if getattr(sampler, "sampler_function", None) is not sample_euler:
        return "unsupported_sampler"
    if sampler.extra_options.get("s_churn", 0) != 0 or any(
        k not in {"s_churn", "s_tmin", "s_tmax", "s_noise"}
        for k in sampler.extra_options
    ):
        return "unsupported_sampler_options"
    if noise.ndim != 4 or noise.shape[0] != 1 or denoise_mask is not None:
        return "unsupported_batch_or_mask"
    if (
        sigmas.ndim != 1
        or len(sigmas) < 2
        or not torch.isfinite(sigmas).all()
        or not torch.all(sigmas[:-1] > sigmas[1:])
        or sigmas[-1] < 0
    ):
        return "unsupported_schedule"
    conds = getattr(guider, "conds", {})
    if len(conds.get("positive", [])) != 1:
        return "partitioned_conditioning"
    for branch in ["positive", "negative"] if paper and cfg != 1 else ["positive"]:
        if len(conds.get(branch, [])) != 1:
            return "partitioned_conditioning"
        unsupported = sorted(set(conds[branch][0]) - CONDITION_FIELDS)
        if unsupported:
            return "unsupported_condition_metadata:" + ",".join(unsupported)
    patcher = guider.model_patcher
    diffusion = patcher.model.diffusion_model
    if getattr(diffusion, "_compiled_call_impl", None) is not None or "forward" in vars(
        diffusion
    ):
        return "compiled_or_replaced_forward"
    if getattr(patcher, "hook_patches", {}) or getattr(patcher, "object_patches", {}):
        return "model_hooks_or_replacements"
    if getattr(patcher, "weight_wrapper_patches", {}) or getattr(
        patcher, "injections", {}
    ):
        return "model_weight_wrappers_or_injections"
    if getattr(patcher, "additional_models", {}):
        return "additional_models"
    reason = wrapper_reason(getattr(patcher, "wrappers", {}))
    if reason:
        return reason
    opts = guider.model_options
    if set(opts) - {
        "transformer_options",
        "disable_cfg1_optimization",
        "to_load_options",
    }:
        return "unsupported_model_options"
    reason = option_reason(opts.get("to_load_options", {}))
    if reason:
        return reason
    if opts.get("disable_cfg1_optimization"):
        return "cfg1_optimization_disabled"
    return option_reason(opts.get("transformer_options", {}))


def snapshot(value):
    """Full device-resident copies: same-shape replacements cannot collide."""
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {k: snapshot(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(snapshot(v) for v in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    # UUIDs appear in Comfy conditioning identities.
    import uuid

    if isinstance(value, uuid.UUID):
        return str(value)
    raise TypeError(f"Unsupported condition signature type: {type(value).__name__}")


def same(a, b):
    import torch

    if isinstance(a, torch.Tensor):
        return (
            isinstance(b, torch.Tensor)
            and a.shape == b.shape
            and a.dtype == b.dtype
            and a.device == b.device
            and torch.equal(a, b)
        )
    if isinstance(a, dict):
        return (
            isinstance(b, dict)
            and a.keys() == b.keys()
            and all(same(a[k], b[k]) for k in a)
        )
    if isinstance(a, tuple):
        return (
            isinstance(b, (tuple, list))
            and len(a) == len(b)
            and all(same(x, y) for x, y in zip(a, b))
        )
    return a == b or (isinstance(a, str) and a == str(b))


def condition_payload(value):
    """Compare actual native conditioning contents before bypassing any forward."""
    if isinstance(value, dict):
        return {k: condition_payload(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(condition_payload(v) for v in value)
    cls = type(value)
    if cls.__module__ == "comfy.conds" and cls.__name__ in {
        "CONDRegular",
        "CONDNoiseShape",
        "CONDCrossAttn",
        "CONDConstant",
        "CONDList",
    }:
        return {"class": cls.__name__, "attributes": condition_payload(vars(value))}
    return value


def validate_pair(state, model, conds, x, sigma, options):
    import torch
    from comfy.model_sampling import CONST

    if state.reasons:
        return
    reason = option_reason(options.get("transformer_options", {}))
    if reason:
        state.disable(reason)
        return
    patcher = state.guider.model_patcher
    if (
        id(model.diffusion_model) != state.model_id
        or str(patcher.patches_uuid) != state.patch_id
    ):
        state.disable("model_or_patches_changed")
    if state.guider.cfg != state.controller.cfg:
        state.disable("guidance_scale_changed")
    if any(
        getattr(patcher, k, {})
        for k in (
            "hook_patches",
            "object_patches",
            "weight_wrapper_patches",
            "injections",
        )
    ):
        state.disable("model_hooks_or_replacements")
    sampling = model.model_sampling
    if (
        getattr(sampling.calculate_input, "__func__", None) is not CONST.calculate_input
        or getattr(sampling.calculate_denoised, "__func__", None)
        is not CONST.calculate_denoised
    ):
        state.disable("unsupported_flow_parameterization")
    if x.ndim != 4 or x.shape[0] != 1 or not torch.isfinite(x).all():
        state.disable("unsupported_batch_or_nonfinite_input")
    if (
        sigma.numel() != 1
        or not torch.isfinite(sigma).all()
        or float(sigma.flatten()[0]) <= 1e-6
    ):
        state.disable("unsupported_sigma")
    elif state.index >= len(state.sigmas) - 1 or not torch.allclose(
        sigma,
        state.sigmas[state.index].to(sigma).expand_as(sigma),
        rtol=1e-5,
        atol=1e-7,
    ):
        state.disable("schedule_changed")
    count = 1 if state.controller.cfg == 1 else 2
    if (
        len(conds) != 2
        or any(conds[i] is None or len(conds[i]) != 1 for i in range(count))
        or (count == 1 and conds[1] is not None)
    ):
        state.disable("unsupported_condition_branch")
        return
    for branch in conds[:count]:
        if set(branch[0]) - CONDITION_FIELDS:
            state.disable("unsupported_condition_metadata")
        if branch[0].get("reference_latents") or "ref_latents" in branch[0].get(
            "model_conds", {}
        ):
            state.disable("editing_not_calibrated")
    if state.reasons:
        return
    signature = {
        "conditions": condition_payload(conds),
        "layout": (tuple(x.shape), str(x.dtype), str(x.device)),
        "kv": options.get("transformer_options", {}).get("qwen_image21_cache"),
    }
    try:
        if state.signature is None:
            state.signature = snapshot(signature)
            state.manifest["kv_config"] = snapshot(signature["kv"]) or {
                "device": "auto",
                "dtype": "default",
            }
        elif not same(state.signature, signature):
            state.disable("condition_signature_changed")
    except (TypeError, ValueError, RuntimeError):
        state.disable("unverifiable_signature")


def finite_json(value):
    """Retain failure artifacts when a rejected schedule/statistic is NaN/Inf."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(item) for item in value]
    return value


def write_run(root, state, status, seconds):
    import torch

    load_device = torch.device(state.guider.model_patcher.load_device)
    summary = {
        "schema_version": 1,
        "run_id": state.run_id,
        "status": status,
        "config": asdict(state.config),
        "planned_steps": state.controller.steps,
        "actual_model_evaluations": state.eval_attempts,
        "completed_model_evaluations": state.index,
        "failure": state.failure,
        "exact_count": state.exact_count,
        "skip_count": state.skip_count,
        "counter_unit": "condition_pair"
        if state.config.policy == "paper"
        else "model_evaluation",
        "native_forward_count": state.native_forward_count
        if state.config.policy == "paper"
        else state.exact_count,
        "degradation_reasons": state.reasons,
        "sampling_seconds": seconds,
        "end_to_end_seconds": None,
        "controller_seconds": state.controller_seconds,
        "signature_seconds": state.signature_seconds,
        "peak_anchor_bytes": state.controller.peak_bytes,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(load_device)
        if load_device.type == "cuda"
        else None,
        "peak_memory_scope": "process_since_last_external_reset",
        "manifest": {
            "plugin_version": "0.2.0-experimental",
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "tested_comfy_source_commit": COMFY_COMMIT,
            **state.manifest,
        },
        "kv_actual_hit": "unknown",
        "kv_actual_device": "unknown",
        "timing_note": "Boundary CUDA synchronization only; controller time includes scalar synchronization.",
    }
    logging.info(
        "AlignCache Qwen21 %s: %s exact, %s skip; %s",
        state.run_id,
        state.exact_count,
        state.skip_count,
        status,
    )
    if not state.config.export_trace:
        return summary
    try:
        destination = Path(root) / "aligncache_qwen21" / state.run_id
        destination.mkdir(parents=True, exist_ok=False)
        (destination / "summary.json").write_text(
            json.dumps(
                finite_json(summary), indent=2, ensure_ascii=False, allow_nan=False
            )
            + "\n"
        )
        (destination / "trace.jsonl").write_text(
            "".join(
                json.dumps(finite_json(row), allow_nan=False) + "\n"
                for row in state.rows
            )
        )
    except Exception:
        logging.exception("Could not export AlignCache Qwen21 telemetry")
    return summary
