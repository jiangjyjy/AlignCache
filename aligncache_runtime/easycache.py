from __future__ import annotations

import types
from typing import Any

import torch
from einops import rearrange


_WAN_SUPPORTED_TASKS = frozenset({"t2v-1.3B", "t2v-14B"})


def _coerce_optional_float(value: Any, *, label: str) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric when provided. Received {value!r}.") from exc


def _coerce_optional_int(value: Any, *, label: str) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer when provided. Received {value!r}.") from exc


def _mean_abs(value: torch.Tensor) -> float:
    return float(value.detach().abs().mean().cpu().item())


def _require_non_negative_float(value: float | None, *, label: str) -> None:
    if value is not None and value < 0:
        raise ValueError(f"{label} must be non-negative when provided. Received {value!r}.")


def _require_non_negative_int(value: int | None, *, label: str) -> None:
    if value is not None and value < 0:
        raise ValueError(f"{label} must be non-negative when provided. Received {value!r}.")


def resolve_wan_easycache_config(*, task: str, method_config: dict[str, Any] | None) -> dict[str, Any]:
    resolved_task = str(task)
    if resolved_task not in _WAN_SUPPORTED_TASKS:
        supported = ", ".join(sorted(_WAN_SUPPORTED_TASKS))
        raise ValueError(
            "EasyCache for Wan currently supports the vendored T2V tasks only. "
            f"Received task={resolved_task!r}. Supported tasks: {supported}"
        )

    raw = dict(method_config or {})
    thresh = _coerce_optional_float(
        raw.get("easycache_thresh", raw.get("cache_thresh", raw.get("thresh", raw.get("threshold")))),
        label="wan.easycache.easycache_thresh",
    )
    ret_steps = _coerce_optional_int(
        raw.get("ret_steps", raw.get("cache_ret_steps")),
        label="wan.easycache.ret_steps",
    )
    _require_non_negative_float(thresh, label="wan.easycache.easycache_thresh")
    _require_non_negative_int(ret_steps, label="wan.easycache.ret_steps")
    return {
        "easycache_thresh": float(0.05 if thresh is None else thresh),
        "ret_steps": int(10 if ret_steps is None else ret_steps),
        "cutoff_steps_offset": 2,
    }


def resolve_hunyuan_easycache_config(method_config: dict[str, Any] | None) -> dict[str, Any]:
    raw = dict(method_config or {})
    thresh = _coerce_optional_float(
        raw.get("easycache_thresh", raw.get("cache_thresh", raw.get("thresh", raw.get("threshold")))),
        label="hunyuan.easycache.easycache_thresh",
    )
    ret_steps = _coerce_optional_int(
        raw.get("ret_steps", raw.get("cache_ret_steps")),
        label="hunyuan.easycache.ret_steps",
    )
    _require_non_negative_float(thresh, label="hunyuan.easycache.easycache_thresh")
    _require_non_negative_int(ret_steps, label="hunyuan.easycache.ret_steps")
    return {
        "easycache_thresh": float(0.025 if thresh is None else thresh),
        "ret_steps": int(5 if ret_steps is None else ret_steps),
    }


def resolve_opensora_easycache_config(method_config: dict[str, Any] | None) -> dict[str, Any]:
    raw = dict(method_config or {})
    thresh = _coerce_optional_float(
        raw.get("easycache_thresh", raw.get("cache_thresh", raw.get("thresh", raw.get("threshold")))),
        label="opensora.easycache.easycache_thresh",
    )
    ret_steps = _coerce_optional_int(
        raw.get("ret_steps", raw.get("cache_ret_steps")),
        label="opensora.easycache.ret_steps",
    )
    _require_non_negative_float(thresh, label="opensora.easycache.easycache_thresh")
    _require_non_negative_int(ret_steps, label="opensora.easycache.ret_steps")
    return {
        "easycache_thresh": float(0.03 if thresh is None else thresh),
        "ret_steps": int(5 if ret_steps is None else ret_steps),
    }


def configure_wan_easycache(
    pipeline: Any,
    *,
    method_name: str,
    method_config: dict[str, Any] | None,
    task: str,
    sample_steps: int,
) -> dict[str, Any] | None:
    model = pipeline.model
    if not hasattr(model, "_vbench_easycache_original_forward"):
        model._vbench_easycache_original_forward = model.forward

    enabled = str(method_name) == "easycache"
    if enabled:
        model.forward = types.MethodType(_wan_easycache_forward, model)
        model._vbench_easycache_patch_applied = True
    else:
        original_forward = getattr(model, "_vbench_easycache_original_forward", None)
        if original_forward is not None:
            model.forward = original_forward

    model.enable_easycache = bool(enabled)
    model.cnt = 0
    model.num_steps = max(int(sample_steps) * 2, 1)
    model.easycache_thresh = 0.0
    model.ret_steps = 0
    model.cutoff_steps = max(int(sample_steps) * 2, 1)
    model.accumulated_error_even = 0.0
    model.should_calc_current_pair = True
    model.k = None
    model.previous_raw_input_even = None
    model.previous_raw_output_even = None
    model.previous_raw_output_odd = None
    model.prev_prev_raw_input_even = None
    model.cache_even = None
    model.cache_odd = None

    if not enabled:
        return None

    resolved = resolve_wan_easycache_config(task=str(task), method_config=method_config)
    model.easycache_thresh = float(resolved["easycache_thresh"])
    model.ret_steps = int(resolved["ret_steps"]) * 2
    model.cutoff_steps = max(int(sample_steps) * 2 - int(resolved["cutoff_steps_offset"]), 1)
    return {
        "easycache_thresh": float(resolved["easycache_thresh"]),
        "ret_steps": int(resolved["ret_steps"]),
    }


def restore_wan_easycache_state(pipeline: Any) -> None:
    model = pipeline.model
    original_forward = getattr(model, "_vbench_easycache_original_forward", None)
    if original_forward is not None:
        model.forward = original_forward
    model.enable_easycache = False
    model.cnt = 0
    model.num_steps = max(int(getattr(model, "num_steps", 1) or 1), 1)
    model.easycache_thresh = 0.0
    model.ret_steps = 0
    model.cutoff_steps = max(int(getattr(model, "num_steps", 1) or 1), 1)
    model.accumulated_error_even = 0.0
    model.should_calc_current_pair = True
    model.k = None
    model.previous_raw_input_even = None
    model.previous_raw_output_even = None
    model.previous_raw_output_odd = None
    model.prev_prev_raw_input_even = None
    model.cache_even = None
    model.cache_odd = None


def configure_hunyuan_easycache(
    sampler: Any,
    *,
    method_name: str,
    method_config: dict[str, Any] | None,
    sample_steps: int,
) -> dict[str, Any] | None:
    from . import teacache

    transformer = sampler.pipeline.transformer
    if not getattr(transformer, "_vbench_teacache_patch_applied", False):
        transformer.forward = types.MethodType(teacache._hunyuan_teacache_forward, transformer)
        transformer._vbench_teacache_patch_applied = True

    enabled = str(method_name) == "easycache"
    transformer.enable_easycache = bool(enabled)
    transformer.enable_teacache = False
    transformer.cnt = 0
    transformer.num_steps = max(int(sample_steps), 1)
    transformer.easycache_thresh = 0.0
    transformer.ret_steps = 0
    transformer.accumulated_error = 0.0
    transformer.previous_raw_input = None
    transformer.previous_output = None
    transformer.prev_prev_raw_input = None
    transformer.cache = None
    transformer.k = None

    if not enabled:
        return None

    resolved = resolve_hunyuan_easycache_config(method_config)
    transformer.easycache_thresh = float(resolved["easycache_thresh"])
    transformer.ret_steps = int(resolved["ret_steps"])
    return dict(resolved)


def _wan_easycache_forward(
    self,
    x,
    t,
    context,
    seq_len,
    clip_fea=None,
    y=None,
):
    from torch.cuda import amp

    from wan.modules.model import sinusoidal_embedding_1d

    if self.model_type in {"i2v", "flf2v"}:
        assert clip_fea is not None and y is not None

    raw_input = [u.clone() for u in x]
    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)

    if y is not None:
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

    self.is_even = bool(int(getattr(self, "cnt", 0)) % 2 == 0)
    if self.is_even:
        if self.cnt < getattr(self, "ret_steps", 0) or self.cnt >= getattr(self, "cutoff_steps", 0):
            self.should_calc_current_pair = True
            self.accumulated_error_even = 0.0
        else:
            prev_input = getattr(self, "previous_raw_input_even", None)
            prev_output = getattr(self, "previous_raw_output_even", None)
            if prev_input is not None and prev_output is not None and getattr(self, "k", None) is not None:
                raw_input_change = torch.cat(
                    [(current - previous).flatten() for current, previous in zip(raw_input, prev_input)]
                ).abs().mean()
                output_norm = torch.cat([value.flatten() for value in prev_output]).abs().mean()
                if float(output_norm.detach().cpu().item()) == 0.0:
                    self.should_calc_current_pair = True
                else:
                    predicted_change = float(
                        getattr(self, "k", 0.0) * (raw_input_change / output_norm).detach().cpu().item()
                    )
                    self.accumulated_error_even += predicted_change
                    self.should_calc_current_pair = self.accumulated_error_even >= float(
                        getattr(self, "easycache_thresh", 0.0)
                    )
                    if self.should_calc_current_pair:
                        self.accumulated_error_even = 0.0
            else:
                self.should_calc_current_pair = True
        self.previous_raw_input_even = [u.clone() for u in raw_input]

    if self.is_even and not self.should_calc_current_pair and getattr(self, "cache_even", None) is not None:
        self.cnt += 1
        if self.cnt >= max(int(getattr(self, "num_steps", 1)), 1):
            self.cnt = 0
        return [(raw + cached).float() for raw, cached in zip(raw_input, self.cache_even)]

    if not self.is_even and not self.should_calc_current_pair and getattr(self, "cache_odd", None) is not None:
        self.cnt += 1
        if self.cnt >= max(int(getattr(self, "num_steps", 1)), 1):
            self.cnt = 0
        return [(raw + cached).float() for raw, cached in zip(raw_input, self.cache_odd)]

    x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
    grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
    x = [u.flatten(2).transpose(1, 2) for u in x]
    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
    assert seq_lens.max() <= seq_len
    x = torch.cat(
        [torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1) for u in x]
    )

    with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).float())
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))
        assert e.dtype == torch.float32 and e0.dtype == torch.float32

    context_lens = None
    context = self.text_embedding(
        torch.stack(
            [torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context]
        )
    )
    if clip_fea is not None:
        context_clip = self.img_emb(clip_fea)
        context = torch.concat([context_clip, context], dim=1)

    kwargs = {
        "e": e0,
        "seq_lens": seq_lens,
        "grid_sizes": grid_sizes,
        "freqs": self.freqs,
        "context": context,
        "context_lens": context_lens,
    }

    for block in self.blocks:
        x = block(x, **kwargs)

    x = self.head(x, e)
    output = self.unpatchify(x, grid_sizes)

    if self.is_even:
        prev_output = getattr(self, "previous_raw_output_even", None)
        prev_prev_input = getattr(self, "prev_prev_raw_input_even", None)
        if prev_output is not None and prev_prev_input is not None:
            output_change = torch.cat(
                [(current - previous).flatten() for current, previous in zip(output, prev_output)]
            ).abs().mean()
            input_change = torch.cat(
                [(current - previous).flatten() for current, previous in zip(self.previous_raw_input_even, prev_prev_input)]
            ).abs().mean()
            if float(input_change.detach().cpu().item()) != 0.0:
                self.k = float((output_change / input_change).detach().cpu().item())
        self.prev_prev_raw_input_even = getattr(self, "previous_raw_input_even", None)
        self.previous_raw_output_even = [u.clone() for u in output]
        self.cache_even = [generated - original for generated, original in zip(output, raw_input)]
    else:
        self.previous_raw_output_odd = [u.clone() for u in output]
        self.cache_odd = [generated - original for generated, original in zip(output, raw_input)]

    self.cnt += 1
    if self.cnt >= max(int(getattr(self, "num_steps", 1)), 1):
        self.cnt = 0
    return [u.float() for u in output]
