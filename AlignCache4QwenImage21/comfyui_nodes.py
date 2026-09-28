"""Native six-argument Qwen wrapper and isolated outer-sample lifecycle."""

from __future__ import annotations

import inspect
import math
import uuid
from dataclasses import asdict, replace
from time import perf_counter

from .runtime import CONFIG_KEY, RUN_STATE_KEY, WRAPPER_KEY


class RunState:
    def __init__(self, config, sigmas, guider, reason=None):
        from .aligncache_accelerate import SAMPLING_RULE, Controller, PairController

        self.config = config
        self.run_id = uuid.uuid4().hex
        self.sigmas = sigmas.detach().clone()
        self.guider = guider
        steps = max(0, len(sigmas) - 1)
        self.controller = (
            PairController(config, steps, float(guider.cfg))
            if config.policy == "paper"
            else Controller(config, steps)
        )
        self.native_forward_count = 0
        self.index = self.exact_count = self.skip_count = 0
        self.eval_attempts = 0
        self.failure = None
        self.rows, self.reasons = [], []
        self.signature = self.layout = self.output_layout = None
        self.controller_seconds = self.signature_seconds = 0.0
        patcher = guider.model_patcher
        self.patch_id = str(patcher.patches_uuid)
        self.model_id = id(patcher.model.diffusion_model)
        self.manifest = {
            "statistic_sampling": SAMPLING_RULE,
            "model_class": type(patcher.model.diffusion_model).__name__,
            "model_loading_identity": self.model_id,
            "patches_uuid": self.patch_id,
            "weight_hash": "unknown",
            "quantization": "unknown",
            "attention_backend": "unknown",
            "workflow_hash": "provided_by_benchmark",
            "runtime_comfy_commit": "unknown",
            "sigmas": sigmas.detach().cpu().tolist(),
        }
        if reason:
            self.disable(reason)

    def disable(self, reason):
        if reason not in self.reasons:
            self.reasons.append(reason)
        self.controller.clear()

    def validate(self, x, timestep, context, refs, slots, options, kwargs):
        import torch

        from .runtime import option_reason, same, snapshot

        started = perf_counter()
        try:
            if self.reasons:
                return
            reason = option_reason(options)
            if reason:
                self.disable(reason)
            if kwargs.get("control") is not None or set(kwargs) - {"control"}:
                self.disable("unsupported_model_kwargs")
            if options.get("cond_or_uncond") != [0]:
                self.disable("unsupported_condition_branch")
            if refs and self.config.mode == "on":
                self.disable("editing_not_calibrated")
            patcher = self.guider.model_patcher
            if self.model_id != id(
                patcher.model.diffusion_model
            ) or self.patch_id != str(patcher.patches_uuid):
                self.disable("model_or_patches_changed")
            if getattr(patcher, "hook_patches", {}) or getattr(
                patcher, "object_patches", {}
            ):
                self.disable("model_hooks_or_replacements")
            layout = (tuple(x.shape), x.dtype, x.device)
            if self.layout is None:
                self.layout = layout
            elif self.layout != layout:
                self.disable("input_layout_changed")
            if x.ndim != 4 or x.shape[0] != 1:
                self.disable("unsupported_batch")
            if self.index >= len(self.sigmas) - 1:
                self.disable("extra_evaluation")
            else:
                sampling = patcher.model.model_sampling
                expected = sampling.timestep(self.sigmas[self.index].reshape(1)).to(
                    timestep
                )
                tolerance = max(1e-7, torch.finfo(timestep.dtype).eps * 2)
                if not torch.isfinite(timestep).all() or not torch.allclose(
                    timestep,
                    expected.expand_as(timestep),
                    rtol=tolerance,
                    atol=tolerance,
                ):
                    self.disable("timestep_mapping_mismatch")
            if "sample_sigmas" in options and not same(
                self.sigmas.to(options["sample_sigmas"]), options["sample_sigmas"]
            ):
                self.disable("schedule_changed")
            if self.reasons:
                return
            signature = {
                "context": context,
                "refs": refs,
                "slots": slots,
                "condition_kwargs": {
                    key: kwargs[key]
                    for key in ("attention_mask", "pooled_output")
                    if key in kwargs
                },
                "branch": options.get("cond_or_uncond"),
                "uuids": options.get("uuids"),
                "kv": options.get("qwen_image21_cache")
                or {"device": "auto", "dtype": "default", "source": "native_default"},
            }
            if self.signature is None:
                self.signature = snapshot(signature)
                self.manifest["kv_config"] = snapshot(signature["kv"])
                self.manifest["model_input_dtype"] = str(x.dtype)
            elif not same(self.signature, signature):
                self.disable("condition_signature_changed")
        except (TypeError, ValueError, RuntimeError) as error:
            self.disable("unverifiable_signature:" + type(error).__name__)
        finally:
            self.signature_seconds += perf_counter() - started


def qwen21_diffusion_wrapper(
    executor,
    x,
    timestep,
    context,
    ref_latents=None,
    image_slots=None,
    transformer_options=None,
    **kwargs,
):
    import torch

    from .aligncache_accelerate import errors

    options = transformer_options if transformer_options is not None else {}
    state = options.get(RUN_STATE_KEY)

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
    if state.config.policy == "paper":
        # Routing happens once per logical CFG pair above Comfy's batching.
        from .runtime import option_reason

        reason = option_reason(options)
        if reason:
            state.disable(reason)
        state.native_forward_count += 1
        return exact()
    state.eval_attempts += 1
    started = perf_counter()
    state.validate(x, timestep, context, ref_latents, image_slots, options, kwargs)
    time_value = float(timestep.flatten()[0])
    if not math.isfinite(time_value):
        state.disable("nonfinite_timestep")
    if not torch.isfinite(x).all():
        state.disable("nonfinite_input")
    decision = state.controller.decide(
        x, state.index, time_value, state.reasons[0] if state.reasons else None
    )
    draft = None
    if not state.reasons:
        try:
            draft = state.controller.predict(decision, x)
            if draft is not None and not torch.isfinite(draft).all():
                raise ValueError("nonfinite_prediction")
        except ValueError:
            draft = None
            decision = replace(
                decision,
                action="exact",
                reason="unsafe_prediction",
                feedback_eligible=False,
            )
    state.controller_seconds += perf_counter() - started
    skipping = decision.action == "skip" and state.config.mode == "on"
    if not skipping:
        state.exact_count += 1
    output = draft if skipping else exact()
    started = perf_counter()
    metrics = {}
    if skipping:
        state.skip_count += 1
        state.controller.commit_skip(decision, x)
    else:
        if not isinstance(output, torch.Tensor) or not torch.isfinite(output).all():
            raise FloatingPointError("QwenImage21 exact output must be a finite tensor")
        layout = (tuple(output.shape), output.dtype, output.device)
        if state.output_layout is not None and state.output_layout != layout:
            state.disable("output_layout_changed")
        state.output_layout = layout
        if state.reasons:
            pass
        elif decision.action == "skip":
            # Trace observes the truth but advances only the shadow policy.
            metrics = errors(draft, output)
            state.controller.commit_skip(decision, x)
        else:
            metrics = state.controller.commit_exact(decision, x, output, draft)
    record = {
        **asdict(decision),
        "run_id": state.run_id,
        "logical_step": state.index if not state.reasons else None,
        "eval_index": state.index,
        "sigma": float(state.sigmas[state.index])
        if state.index < len(state.sigmas)
        else None,
        "timestep": time_value if math.isfinite(time_value) else None,
        "shadow_action": decision.action,
        "consecutive_skips": state.controller.age,
        "action": "skip" if skipping else "exact",
        "predictor": state.config.predictor,
        "feedback_error": metrics,
        "feedback_scale": state.controller.feedback_scale(decision.phase),
        "condition_signature_version": 1,
        "model_output_dtype": str(output.dtype),
        "kv_config": state.manifest.get("kv_config", {}),
        "kv_hit": "unknown",
    }
    state.rows.append(record)
    state.index += 1
    state.controller_seconds += perf_counter() - started
    return output


def make_outer_sample_wrapper(output_root):
    def outer(executor, *args, **kwargs):
        import torch
        from comfy.model_patcher import create_model_options_clone

        from .runtime import run_reason, write_run

        guider = executor.class_obj
        bound = inspect.signature(executor.original).bind(*args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
        original = guider.model_options
        config = original["transformer_options"][CONFIG_KEY]
        reason = run_reason(
            guider,
            values["sampler"],
            values["noise"],
            values["sigmas"],
            values.get("denoise_mask"),
        )
        state = RunState(config, values["sigmas"], guider, reason)
        load_device = torch.device(guider.model_patcher.load_device)
        guider.model_options = create_model_options_clone(original)
        guider.model_options["transformer_options"][RUN_STATE_KEY] = state
        status, start = "failed", perf_counter()
        try:
            if load_device.type == "cuda":
                torch.cuda.synchronize(load_device)
            start = perf_counter()
            result = executor(*args, **kwargs)
            if load_device.type == "cuda":
                torch.cuda.synchronize(load_device)
            status = "completed"
            if state.index != state.controller.steps:
                state.disable("evaluation_count_mismatch")
            return result
        except BaseException as error:
            state.failure = {"type": type(error).__name__, "message": str(error)}
            status = (
                "interrupted"
                if isinstance(error, KeyboardInterrupt)
                or "Interrupt" in type(error).__name__
                else "failed"
            )
            raise
        finally:
            guider.model_options = original
            try:
                try:
                    write_run(output_root, state, status, perf_counter() - start)
                except Exception:
                    import logging

                    logging.exception("Could not finalize AlignCache Qwen21 telemetry")
            finally:
                state.controller.clear()
                state.signature = None
                state.guider = None

    return outer


class AlignCacheQwenImage21:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "mode": (["off", "trace", "on"], {"default": "trace"}),
                "base_threshold": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
                "max_consecutive_skips": ("INT", {"default": 1, "min": 0, "max": 8}),
                "warmup_steps": ("INT", {"default": -1, "min": -1, "max": 10000}),
                "tail_steps": ("INT", {"default": -1, "min": -1, "max": 10000}),
                "predictor": (
                    [
                        "scaled_output_momentum",
                        "reuse",
                        "scaled_input_residual",
                        "time_normalized_output",
                    ],
                ),
                "export_trace": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "run_nonce": ("STRING", {"default": ""}),
                "advanced_config": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "attach"
    CATEGORY = "AlignCache/Qwen-Image-2.1"

    def attach(
        self,
        model,
        mode="trace",
        base_threshold=0.0,
        max_consecutive_skips=1,
        warmup_steps=-1,
        tail_steps=-1,
        predictor="scaled_output_momentum",
        export_trace=True,
        run_nonce="",
        advanced_config="",
    ):
        try:
            import folder_paths
            from comfy.patcher_extension import WrappersMP

            from .aligncache_accelerate import Config
        except ImportError as error:
            raise RuntimeError(
                "AlignCacheQwenImage21 requires Torch and a ComfyUI installation with model wrappers"
            ) from error
        from .runtime import require_model

        require_model(model)
        advanced = {}
        if advanced_config:
            import json
            from pathlib import Path

            presets = {
                "preset:fixed_t2i": "fixed_t2i_advanced.json",
                "preset:paper_qwen": "paper_qwen_advanced.json",
            }
            config_path = (
                Path(__file__).resolve().parent / "presets" / presets[advanced_config]
                if advanced_config in presets
                else Path(advanced_config)
            )
            advanced = json.loads(config_path.read_text(encoding="utf-8"))
            schema = advanced.pop("schema_version", None)
            allowed = {"beta", "phase_scales", "max_time_gap"}
            if schema == 2:
                allowed |= {
                    "policy",
                    "phase_age_caps",
                    "feedback_scale_min",
                    "feedback_scale_max",
                    "feedback_decay",
                    "feedback_ratio_bounds",
                    "phase_beta_cond",
                    "phase_beta_uncond",
                    "anchor_strength",
                    "anchor_cond_cap",
                    "anchor_disagreement_cap",
                }
            if schema not in (1, 2) or set(advanced) - allowed:
                raise ValueError("Unsupported advanced config schema or fields")
            advanced["schema_version"] = schema
        config = Config(
            mode,
            base_threshold,
            max_consecutive_skips,
            warmup_steps,
            tail_steps,
            predictor,
            export_trace,
            run_nonce=run_nonce,
            **advanced,
        )
        patched = model.clone()
        options = patched.model_options.setdefault("transformer_options", {})
        options.pop(CONFIG_KEY, None)
        options.pop(RUN_STATE_KEY, None)
        for kind in (
            WrappersMP.OUTER_SAMPLE,
            WrappersMP.DIFFUSION_MODEL,
            WrappersMP.CALC_COND_BATCH,
        ):
            patched.remove_wrappers_with_key(kind, WRAPPER_KEY)
            # Also remove propagated copies if attaching to prepared options.
            options.get("wrappers", {}).get(kind, {}).pop(WRAPPER_KEY, None)
        if mode != "off":
            options[CONFIG_KEY] = config
            patched.add_wrapper_with_key(
                WrappersMP.OUTER_SAMPLE,
                WRAPPER_KEY,
                make_outer_sample_wrapper(folder_paths.get_output_directory()),
            )
            patched.add_wrapper_with_key(
                WrappersMP.DIFFUSION_MODEL, WRAPPER_KEY, qwen21_diffusion_wrapper
            )
            if config.policy == "paper":
                patched.add_wrapper_with_key(
                    WrappersMP.CALC_COND_BATCH, WRAPPER_KEY, qwen21_pair_wrapper
                )
        return (patched,)


NODE_CLASS_MAPPINGS = {"AlignCacheQwenImage21": AlignCacheQwenImage21}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AlignCacheQwenImage21": "AlignCache Qwen-Image-2.1 (Experimental)"
}


def qwen21_pair_wrapper(executor, model, conds, x_in, timestep, model_options):
    import torch

    from .aligncache_accelerate import errors, guided
    from .runtime import validate_pair

    state = model_options.get("transformer_options", {}).get(RUN_STATE_KEY)
    if state is None or state.config.policy != "paper":
        return executor(model, conds, x_in, timestep, model_options)
    state.eval_attempts += 1
    started = perf_counter()
    validate_pair(state, model, conds, x_in, timestep, model_options)
    state.signature_seconds += perf_counter() - started
    controller = state.controller
    sigma = float(timestep.flatten()[0])
    time_value = float(model.model_sampling.timestep(timestep).flatten()[0])
    decision = controller.decide(
        x_in, state.index, time_value, state.reasons[0] if state.reasons else None
    )
    drafts = None
    if not state.reasons:
        try:
            drafts = controller.predict_pair(decision, x_in)
            if drafts is not None and any(not torch.isfinite(v).all() for v in drafts):
                raise ValueError("nonfinite_prediction")
        except ValueError:
            drafts = None
            decision = replace(
                decision,
                action="exact",
                reason="unsafe_prediction",
                feedback_eligible=False,
            )
    skipping = (
        not state.reasons
        and drafts is not None
        and decision.action == "skip"
        and state.config.mode == "on"
    )
    state.controller_seconds += perf_counter() - started
    if skipping:
        result = [
            model.model_sampling.calculate_denoised(timestep, v.float(), x_in)
            for v in drafts
        ]
        if len(result) == 1:
            result.append(torch.zeros_like(x_in))
        state.skip_count += 1
        controller.commit_skip(decision, x_in)
        metrics = {}
    else:
        state.exact_count += 1
        result = executor(model, conds, x_in, timestep, model_options)
        started = perf_counter()
        metrics = {}
        if not state.reasons:
            # Invert only the audited native flow conversion; exact/trace always
            # return the untouched native results, never a reconstructed tensor.
            count = 1 if controller.cfg == 1 else 2
            outputs = [(x_in.float() - y.float()) / sigma for y in result[:count]]
            if any(not torch.isfinite(v).all() for v in outputs):
                raise FloatingPointError("Nonfinite exact Qwen flow output")
            if decision.action == "skip":
                metrics = errors(
                    guided(drafts, controller.cfg), guided(outputs, controller.cfg)
                )
                controller.commit_skip(decision, x_in)
            else:
                metrics = controller.commit_pair(decision, x_in, outputs, drafts)
        state.controller_seconds += perf_counter() - started
    state.rows.append(
        {
            **asdict(decision),
            "run_id": state.run_id,
            "logical_step": state.index,
            "eval_index": state.index,
            "sigma": sigma,
            "action": "skip" if skipping else "exact",
            "shadow_action": decision.action,
            "feedback_error": metrics,
            "feedback_metric": "guided_velocity_relative_mse",
            "feedback_scale": controller.feedback_scale(decision.phase),
            "anchor_weight": controller.anchor_weight,
            "transfer_alpha": controller.transfer_alpha[decision.phase],
            "consecutive_skips": controller.age,
            "counter_unit": "condition_pair",
        }
    )
    state.index += 1
    return result
