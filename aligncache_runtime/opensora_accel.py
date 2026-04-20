from __future__ import annotations

import gc
import types
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from einops import rearrange

from . import aligncache, easycache, teacache

_DEFAULT_PAB_CONFIG = {
    "spatial_broadcast": True,
    "spatial_threshold": [450, 930],
    "spatial_range": 2,
    "temporal_broadcast": True,
    "temporal_threshold": [450, 930],
    "temporal_range": 4,
    "cross_broadcast": True,
    "cross_threshold": [450, 930],
    "cross_range": 6,
    "mlp_broadcast": True,
    "mlp_spatial_broadcast_config": {
        676: {"block": [0, 1, 2, 3, 4], "skip_count": 2},
        788: {"block": [0, 1, 2, 3, 4], "skip_count": 2},
        864: {"block": [0, 1, 2, 3, 4], "skip_count": 2},
    },
    "mlp_temporal_broadcast_config": {
        676: {"block": [0, 1, 2, 3, 4], "skip_count": 2},
        788: {"block": [0, 1, 2, 3, 4], "skip_count": 2},
        864: {"block": [0, 1, 2, 3, 4], "skip_count": 2},
    },
}


def _coerce_optional_bool(value: Any, *, label: str) -> bool | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y", "on"}:
            return True
        if lowered in {"false", "0", "no", "n", "off"}:
            return False
    raise ValueError(f"{label} must be a boolean when provided. Received {value!r}.")


def _coerce_optional_int(value: Any, *, label: str) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer when provided. Received {value!r}.") from exc


def _coerce_threshold_pair(value: Any, *, label: str, default: list[int]) -> list[int]:
    if value in (None, ""):
        return [int(item) for item in default]
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{label} must be a list of two integers. Received {value!r}.")
    return [int(value[0]), int(value[1])]


def _normalize_mlp_broadcast_config(value: Any, *, label: str, default: dict[int, dict[str, Any]]) -> dict[int, dict[str, Any]]:
    source = default if value in (None, "") else value
    if not isinstance(source, dict):
        raise ValueError(f"{label} must be an object mapping timestep -> config. Received {type(source).__name__}.")
    normalized: dict[int, dict[str, Any]] = {}
    for timestep, row in source.items():
        if not isinstance(row, dict):
            raise ValueError(f"{label}[{timestep!r}] must be an object. Received {type(row).__name__}.")
        block_ids = row.get("block")
        if not isinstance(block_ids, list) or not block_ids:
            raise ValueError(f"{label}[{timestep!r}].block must be a non-empty list of integers.")
        skip_count = _coerce_optional_int(row.get("skip_count"), label=f"{label}[{timestep!r}].skip_count")
        if skip_count is None or skip_count < 0:
            raise ValueError(f"{label}[{timestep!r}].skip_count must be a non-negative integer.")
        normalized[int(timestep)] = {
            "block": [int(block_id) for block_id in block_ids],
            "skip_count": int(skip_count),
        }
    return normalized


def resolve_opensora_pab_config(method_config: dict[str, Any] | None) -> dict[str, Any]:
    raw = dict(method_config or {})
    pab_raw = dict(raw.get("pab_config") or raw)
    resolved = {
        "spatial_broadcast": bool(
            _coerce_optional_bool(
                pab_raw.get("spatial_broadcast"),
                label="opensora.pab.spatial_broadcast",
            )
            if pab_raw.get("spatial_broadcast") is not None
            else _DEFAULT_PAB_CONFIG["spatial_broadcast"]
        ),
        "spatial_threshold": _coerce_threshold_pair(
            pab_raw.get("spatial_threshold"),
            label="opensora.pab.spatial_threshold",
            default=_DEFAULT_PAB_CONFIG["spatial_threshold"],
        ),
        "spatial_range": int(
            _coerce_optional_int(pab_raw.get("spatial_range"), label="opensora.pab.spatial_range")
            or _DEFAULT_PAB_CONFIG["spatial_range"]
        ),
        "temporal_broadcast": bool(
            _coerce_optional_bool(
                pab_raw.get("temporal_broadcast"),
                label="opensora.pab.temporal_broadcast",
            )
            if pab_raw.get("temporal_broadcast") is not None
            else _DEFAULT_PAB_CONFIG["temporal_broadcast"]
        ),
        "temporal_threshold": _coerce_threshold_pair(
            pab_raw.get("temporal_threshold"),
            label="opensora.pab.temporal_threshold",
            default=_DEFAULT_PAB_CONFIG["temporal_threshold"],
        ),
        "temporal_range": int(
            _coerce_optional_int(pab_raw.get("temporal_range"), label="opensora.pab.temporal_range")
            or _DEFAULT_PAB_CONFIG["temporal_range"]
        ),
        "cross_broadcast": bool(
            _coerce_optional_bool(
                pab_raw.get("cross_broadcast"),
                label="opensora.pab.cross_broadcast",
            )
            if pab_raw.get("cross_broadcast") is not None
            else _DEFAULT_PAB_CONFIG["cross_broadcast"]
        ),
        "cross_threshold": _coerce_threshold_pair(
            pab_raw.get("cross_threshold"),
            label="opensora.pab.cross_threshold",
            default=_DEFAULT_PAB_CONFIG["cross_threshold"],
        ),
        "cross_range": int(
            _coerce_optional_int(pab_raw.get("cross_range"), label="opensora.pab.cross_range")
            or _DEFAULT_PAB_CONFIG["cross_range"]
        ),
        "mlp_broadcast": bool(
            _coerce_optional_bool(
                pab_raw.get("mlp_broadcast"),
                label="opensora.pab.mlp_broadcast",
            )
            if pab_raw.get("mlp_broadcast") is not None
            else _DEFAULT_PAB_CONFIG["mlp_broadcast"]
        ),
        "mlp_spatial_broadcast_config": _normalize_mlp_broadcast_config(
            pab_raw.get("mlp_spatial_broadcast_config"),
            label="opensora.pab.mlp_spatial_broadcast_config",
            default=_DEFAULT_PAB_CONFIG["mlp_spatial_broadcast_config"],
        ),
        "mlp_temporal_broadcast_config": _normalize_mlp_broadcast_config(
            pab_raw.get("mlp_temporal_broadcast_config"),
            label="opensora.pab.mlp_temporal_broadcast_config",
            default=_DEFAULT_PAB_CONFIG["mlp_temporal_broadcast_config"],
        ),
    }
    return resolved


def _opensora_pab_runtime_enabled(config: dict[str, Any]) -> bool:
    return bool(
        config.get("spatial_broadcast")
        or config.get("temporal_broadcast")
        or config.get("cross_broadcast")
    )


def _ratio_delta(current: torch.Tensor, previous: torch.Tensor) -> float:
    denom = previous.abs().mean()
    if float(denom) == 0.0:
        return 0.0
    return float(((current - previous).abs().mean() / denom).detach().cpu().item())


def _iter_blocks(model: Any) -> list[Any]:
    return [*list(getattr(model, "spatial_blocks", [])), *list(getattr(model, "temporal_blocks", []))]


def _assign_block_indices(model: Any) -> None:
    for index, block in enumerate(list(getattr(model, "spatial_blocks", []))):
        block._vbench_block_idx = int(index)
    for index, block in enumerate(list(getattr(model, "temporal_blocks", []))):
        block._vbench_block_idx = int(index)


def _reset_pab_state(model: Any) -> None:
    manager = getattr(model, "_vbench_pab_manager", None)
    if manager is not None:
        try:
            manager._mlp_temporal_outputs.clear()
        except Exception:
            pass
        try:
            manager._mlp_spatial_outputs.clear()
        except Exception:
            pass
    model.enable_pab = False
    model._vbench_pab_manager = None
    for block in _iter_blocks(model):
        block._vbench_pab_manager = None
        block._vbench_pab_attn_count = 0
        block._vbench_pab_last_attn = None
        block._vbench_pab_cross_count = 0
        block._vbench_pab_last_cross = None
        block._vbench_pab_mlp_count = 0


def _reset_teacache_state(model: Any, *, sample_steps: int) -> None:
    model.accumulated_rel_l1_distance = 0.0
    model.previous_modulated_input = None
    model.previous_residual = None
    model.rel_l1_thresh = 0.0
    model.enable_teacache = False
    model.num_steps = max(int(sample_steps), 1)
    model.cnt = 0


def _reset_easycache_state(model: Any, *, sample_steps: int) -> None:
    model.enable_easycache = False
    model.num_steps = max(int(sample_steps), 1)
    model.cnt = 0
    model.easycache_thresh = 0.0
    model.ret_steps = 0
    model.accumulated_error = 0.0
    model.previous_raw_input = None
    model.previous_output = None
    model.prev_prev_raw_input = None
    model.cache = None
    model.k = None


def _reset_aligncache_state(model: Any, *, sample_steps: int) -> None:
    model.enable_aligncache = False
    aligncache.reset_single_aligncache_state(model, sample_steps=max(int(sample_steps), 1))


def reset_opensora_method_state(model: Any, *, sample_steps: int | None = None) -> None:
    resolved_steps = sample_steps
    if resolved_steps in (None, ""):
        resolved_steps = getattr(model, "num_steps", 1)
    _reset_teacache_state(model, sample_steps=max(int(resolved_steps), 1))
    _reset_easycache_state(model, sample_steps=max(int(resolved_steps), 1))
    _reset_aligncache_state(model, sample_steps=max(int(resolved_steps), 1))
    _reset_pab_state(model)


def release_opensora_cuda_memory(*, device: str | None) -> None:
    gc.collect()
    if not torch.cuda.is_available():
        return
    if device not in (None, ""):
        try:
            torch.cuda.synchronize(device=str(device))
        except Exception:
            pass
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    ipc_collect = getattr(torch.cuda, "ipc_collect", None)
    if callable(ipc_collect):
        try:
            ipc_collect()
        except Exception:
            pass


class OpenSoraPABManager:
    def __init__(self, config: dict[str, Any], *, steps: int) -> None:
        self.config = dict(config)
        self.steps = max(int(steps), 1)
        self._mlp_temporal_outputs: dict[tuple[int, int], Any] = {}
        self._mlp_spatial_outputs: dict[tuple[int, int], Any] = {}

    def _next_count(self, count: int) -> int:
        return (int(count) + 1) % self.steps

    def if_broadcast_cross(self, timestep: int | None, count: int) -> tuple[bool, int]:
        enabled = bool(self.config.get("cross_broadcast"))
        threshold = list(self.config.get("cross_threshold") or [])
        broadcast_range = max(int(self.config.get("cross_range") or 1), 1)
        flag = bool(
            enabled
            and timestep is not None
            and count % broadcast_range != 0
            and threshold[0] < timestep < threshold[1]
        )
        return flag, self._next_count(count)

    def if_broadcast_spatial(self, timestep: int | None, count: int) -> tuple[bool, int]:
        enabled = bool(self.config.get("spatial_broadcast"))
        threshold = list(self.config.get("spatial_threshold") or [])
        broadcast_range = max(int(self.config.get("spatial_range") or 1), 1)
        flag = bool(
            enabled
            and timestep is not None
            and count % broadcast_range != 0
            and threshold[0] < timestep < threshold[1]
        )
        return flag, self._next_count(count)

    def if_broadcast_temporal(self, timestep: int | None, count: int) -> tuple[bool, int]:
        enabled = bool(self.config.get("temporal_broadcast"))
        threshold = list(self.config.get("temporal_threshold") or [])
        broadcast_range = max(int(self.config.get("temporal_range") or 1), 1)
        flag = bool(
            enabled
            and timestep is not None
            and count % broadcast_range != 0
            and threshold[0] < timestep < threshold[1]
        )
        return flag, self._next_count(count)

    @staticmethod
    def _resolve_skip_window(all_timesteps: list[int], timestep: int | None, config: dict[int, dict[str, Any]]) -> tuple[bool, list[int] | None]:
        if timestep is None:
            return False, None
        for anchor, row in config.items():
            if int(anchor) not in all_timesteps:
                continue
            anchor_index = all_timesteps.index(int(anchor))
            skip_count = int(row.get("skip_count") or 0)
            end_index = anchor_index + skip_count
            if end_index >= len(all_timesteps):
                continue
            skip_window = all_timesteps[anchor_index : end_index + 1]
            if timestep in skip_window:
                return True, [skip_window[0], skip_window[-1]]
        return False, None

    def if_broadcast_mlp(
        self,
        timestep: int | None,
        count: int,
        *,
        block_idx: int,
        all_timesteps: list[int],
        is_temporal: bool,
    ) -> tuple[bool, int, bool, list[int] | None]:
        if not bool(self.config.get("mlp_broadcast")):
            return False, int(count), False, None
        cur_config = (
            dict(self.config.get("mlp_temporal_broadcast_config") or {})
            if is_temporal
            else dict(self.config.get("mlp_spatial_broadcast_config") or {})
        )
        in_skip_window, skip_window = self._resolve_skip_window(all_timesteps, timestep, cur_config)
        if timestep is not None and int(timestep) in cur_config and int(block_idx) in set(cur_config[int(timestep)]["block"]):
            return False, int(count) + 1, True, None
        if in_skip_window and skip_window is not None:
            anchor = int(skip_window[0])
            if int(block_idx) in set(cur_config[anchor]["block"]):
                return True, 0, False, skip_window
        return False, int(count), False, None

    def save_mlp_output(self, *, timestep: int, block_idx: int, ff_output: Any, is_temporal: bool) -> None:
        if is_temporal:
            self._mlp_temporal_outputs[(int(timestep), int(block_idx))] = ff_output
        else:
            self._mlp_spatial_outputs[(int(timestep), int(block_idx))] = ff_output

    def get_mlp_output(self, *, skip_window: list[int], timestep: int, block_idx: int, is_temporal: bool) -> Any:
        anchor = int(skip_window[0])
        key = (anchor, int(block_idx))
        outputs = self._mlp_temporal_outputs if is_temporal else self._mlp_spatial_outputs
        if key not in outputs:
            raise ValueError(
                f"No stored PAB MLP output found for timestep {timestep}, skip window {skip_window}, block {block_idx}."
            )
        value = outputs[key]
        if int(timestep) == int(skip_window[-1]):
            del outputs[key]
        return value


def configure_opensora_method(
    model: Any,
    *,
    method_name: str,
    method_config: dict[str, Any] | None,
    sample_steps: int,
) -> dict[str, Any] | None:
    resolved_method_name = str(method_name)
    if resolved_method_name not in {"baseline", "teacache", "easycache", "aligncache", "pab"}:
        raise ValueError(f"Unsupported OpenSora acceleration method: {method_name!r}.")

    if not getattr(model, "_vbench_accel_patch_applied", False):
        model.forward = types.MethodType(_opensora_accel_forward, model)
        model._vbench_accel_patch_applied = True
        for block in _iter_blocks(model):
            if getattr(block, "_vbench_pab_patch_applied", False):
                continue
            block.forward = types.MethodType(_opensora_pab_block_forward, block)
            block._vbench_pab_patch_applied = True
    _assign_block_indices(model)

    _reset_teacache_state(model, sample_steps=int(sample_steps))
    _reset_easycache_state(model, sample_steps=int(sample_steps))
    _reset_aligncache_state(model, sample_steps=int(sample_steps))
    _reset_pab_state(model)

    if resolved_method_name == "baseline":
        model.enable_pab = False
        return None

    if resolved_method_name == "teacache":
        resolved = teacache.resolve_opensora_teacache_config(method_config)
        model.enable_teacache = True
        model.rel_l1_thresh = float(resolved["rel_l1_thresh"])
        model.enable_pab = False
        return dict(resolved)

    if resolved_method_name == "easycache":
        resolved = easycache.resolve_opensora_easycache_config(method_config)
        model.enable_easycache = True
        model.easycache_thresh = float(resolved["easycache_thresh"])
        model.ret_steps = int(resolved["ret_steps"])
        model.enable_pab = False
        return dict(resolved)

    if resolved_method_name == "aligncache":
        resolved = aligncache.configure_opensora_aligncache(
            model,
            enabled=True,
            method_config=method_config,
            sample_steps=int(sample_steps),
        )
        model.enable_pab = False
        return dict(resolved or {})

    resolved = resolve_opensora_pab_config(method_config)
    if not _opensora_pab_runtime_enabled(resolved):
        model.enable_pab = False
        return dict(resolved)
    manager = OpenSoraPABManager(resolved, steps=int(sample_steps))
    model.enable_pab = True
    model._vbench_pab_manager = manager
    for block in _iter_blocks(model):
        block._vbench_pab_manager = manager
    return dict(resolved)


def _extract_timestep_value(timestep: Any) -> int | None:
    if timestep is None:
        return None
    if isinstance(timestep, torch.Tensor):
        return int(round(float(timestep[0].detach().cpu().item())))
    return int(round(float(timestep)))


def _normalize_timestep_sequence(all_timesteps: Any) -> list[int]:
    sequence: list[int] = []
    for item in list(all_timesteps or []):
        sequence.append(_extract_timestep_value(item))
    return [int(value) for value in sequence if value is not None]


def _prepare_easycache_state_for_new_sampling_sequence(
    model: Any,
    *,
    timestep: Any,
    all_timesteps: Any,
) -> None:
    if not getattr(model, "enable_easycache", False):
        return
    if int(getattr(model, "cnt", 0)) != 0:
        return

    current_timestep = _extract_timestep_value(timestep)
    timestep_sequence = _normalize_timestep_sequence(all_timesteps)
    if timestep_sequence and current_timestep is not None and current_timestep != timestep_sequence[0]:
        return

    model.accumulated_error = 0.0
    model.previous_raw_input = None
    model.previous_output = None
    model.prev_prev_raw_input = None
    model.cache = None
    model.k = None


def _opensora_accel_forward(
    self,
    x,
    timestep,
    y,
    mask=None,
    x_mask=None,
    fps=None,
    height=None,
    width=None,
    **kwargs,
):
    from opensora.acceleration.checkpoint import auto_grad_checkpoint
    from opensora.acceleration.communications import gather_forward_split_backward, split_forward_gather_backward
    from opensora.acceleration.parallel_states import get_sequence_parallel_group
    from opensora.models.layers.blocks import t2i_modulate

    all_timesteps = kwargs.get("all_timesteps")
    _prepare_easycache_state_for_new_sampling_sequence(
        self,
        timestep=timestep,
        all_timesteps=all_timesteps,
    )
    dtype = self.x_embedder.proj.weight.dtype
    raw_input = x.clone() if getattr(self, "enable_easycache", False) else None
    if getattr(self, "enable_aligncache", False):
        raw_input = x.clone()
    x = x.to(dtype)
    timestep = timestep.to(dtype)
    y = y.to(dtype)

    _, _, tx, hx, wx = x.size()
    t_size, h_size, w_size = self.get_dynamic_size(x)
    s_size = h_size * w_size
    base_size = round(s_size**0.5)
    resolution_sq = (height[0].item() * width[0].item()) ** 0.5
    scale = resolution_sq / self.input_sq_size
    pos_emb = self.pos_embed(x, h_size, w_size, scale=scale, base_size=base_size)

    t = self.t_embedder(timestep, dtype=x.dtype)
    fps = self.fps_embedder(fps.unsqueeze(1), x.size(0))
    t = t + fps
    t_mlp = self.t_block(t)
    t0 = t0_mlp = None
    if x_mask is not None:
        t0_timestep = torch.zeros_like(timestep)
        t0 = self.t_embedder(t0_timestep, dtype=x.dtype)
        t0 = t0 + fps
        t0_mlp = self.t_block(t0)

    if self.config.skip_y_embedder:
        y_lens = mask
        if isinstance(y_lens, torch.Tensor):
            y_lens = y_lens.long().tolist()
    else:
        y, y_lens = self.encode_text(y, mask)

    x = self.x_embedder(x)
    x = rearrange(x, "B (T S) C -> B T S C", T=t_size, S=s_size)
    x = x + pos_emb

    aligncache_decision = None
    if getattr(self, "enable_aligncache", False):
        aligncache_decision = aligncache.prepare_single_aligncache_step(
            self,
            raw_input=raw_input,
        )
        if aligncache_decision is not None and aligncache_decision["exec_mode"] == "skip":
            predicted_output = aligncache.complete_single_aligncache_skip(
                self,
                raw_input=raw_input,
                predicted_output=aligncache_decision["predicted_output"],
                decision=aligncache_decision,
            )[0]
            return predicted_output.to(torch.float32)

    should_calc = True
    if getattr(self, "enable_easycache", False):
        if self.cnt < getattr(self, "ret_steps", 0) or self.cnt >= max(int(getattr(self, "num_steps", 1)) - 1, 0):
            should_calc = True
            self.accumulated_error = 0.0
        else:
            prev_input = getattr(self, "previous_raw_input", None)
            prev_output = getattr(self, "previous_output", None)
            if prev_input is not None and prev_output is not None and getattr(self, "k", None) is not None:
                raw_input_change = (raw_input - prev_input).abs().mean()
                output_norm = prev_output.abs().mean()
                if float(output_norm.detach().cpu().item()) == 0.0:
                    should_calc = True
                else:
                    predicted_change = float(
                        getattr(self, "k", 0.0) * (raw_input_change / output_norm).detach().cpu().item()
                    )
                    self.accumulated_error += predicted_change
                    should_calc = self.accumulated_error >= float(getattr(self, "easycache_thresh", 0.0))
                    if should_calc:
                        self.accumulated_error = 0.0
            else:
                should_calc = True
        self.previous_raw_input = raw_input.clone()
        if not should_calc and getattr(self, "cache", None) is not None:
            result = raw_input + self.cache
            self.cnt += 1
            if self.cnt >= max(int(getattr(self, "num_steps", 1)), 1):
                self.cnt = 0
            return result.to(torch.float32)

    if getattr(self, "enable_teacache", False):
        modulated_inp = rearrange(x.clone(), "B T S C -> B (T S) C", T=t_size, S=s_size)
        batch_size = modulated_inp.shape[0]
        shift_msa, scale_msa, _gate_msa, _shift_mlp, _scale_mlp, _gate_mlp = (
            self.spatial_blocks[0].scale_shift_table[None] + t_mlp.reshape(batch_size, 6, -1)
        ).chunk(6, dim=1)
        modulated_inp = t2i_modulate(self.spatial_blocks[0].norm1(modulated_inp), shift_msa, scale_msa)
        current_step = float(timestep[0].detach().cpu().item())
        first_step = None
        last_step = None
        if all_timesteps:
            first_step = float(all_timesteps[0][0].detach().cpu().item())
            last_step = float(all_timesteps[-1][0].detach().cpu().item())
        if (
            getattr(self, "previous_modulated_input", None) is None
            or first_step is None
            or current_step == first_step
            or current_step == last_step
        ):
            should_calc = True
            self.accumulated_rel_l1_distance = 0.0
        else:
            rescale_func = np.poly1d(teacache._OPENSORA_COEFFICIENTS)
            self.accumulated_rel_l1_distance += float(
                rescale_func(_ratio_delta(modulated_inp, self.previous_modulated_input))
            )
            should_calc = self.accumulated_rel_l1_distance >= float(getattr(self, "rel_l1_thresh", 0.0))
            if should_calc:
                self.accumulated_rel_l1_distance = 0.0
        self.previous_modulated_input = modulated_inp

    if self.enable_sequence_parallelism:
        x = split_forward_gather_backward(x, get_sequence_parallel_group(), dim=2, grad_scale="down")
        s_size = s_size // dist.get_world_size(get_sequence_parallel_group())

    x = rearrange(x, "B T S C -> B (T S) C", T=t_size, S=s_size)

    if getattr(self, "enable_teacache", False) and not should_calc and getattr(self, "previous_residual", None) is not None:
        x = x + self.previous_residual
    else:
        origin_x = x.clone()
        for spatial_block, temporal_block in zip(self.spatial_blocks, self.temporal_blocks):
            x = auto_grad_checkpoint(
                spatial_block,
                x,
                y,
                t_mlp,
                y_lens,
                x_mask,
                t0_mlp,
                t_size,
                s_size,
                timestep,
                all_timesteps=all_timesteps,
            )
            x = auto_grad_checkpoint(
                temporal_block,
                x,
                y,
                t_mlp,
                y_lens,
                x_mask,
                t0_mlp,
                t_size,
                s_size,
                timestep,
                all_timesteps=all_timesteps,
            )
        if getattr(self, "enable_teacache", False):
            self.previous_residual = x - origin_x

    if self.enable_sequence_parallelism:
        x = rearrange(x, "B (T S) C -> B T S C", T=t_size, S=s_size)
        x = gather_forward_split_backward(x, get_sequence_parallel_group(), dim=2, grad_scale="up")
        s_size = s_size * dist.get_world_size(get_sequence_parallel_group())
        x = rearrange(x, "B T S C -> B (T S) C", T=t_size, S=s_size)

    x = self.final_layer(x, t, x_mask, t0, t_size, s_size)
    x = self.unpatchify(x, t_size, h_size, w_size, tx, hx, wx)
    if getattr(self, "enable_aligncache", False):
        resolved_output = aligncache.complete_single_aligncache_exact(
            self,
            raw_input=raw_input,
            exact_output=x,
            decision=aligncache_decision or {"phase": "mid", "step_idx": int(getattr(self, "cnt", 0))},
        )[0]
        return resolved_output.to(torch.float32)
    if getattr(self, "enable_easycache", False):
        self.cache = x - raw_input
        previous_output = getattr(self, "previous_output", None)
        prev_prev_raw_input = getattr(self, "prev_prev_raw_input", None)
        if previous_output is not None and prev_prev_raw_input is not None:
            output_change = (x - previous_output).abs().mean()
            input_change = (self.previous_raw_input - prev_prev_raw_input).abs().mean()
            if float(input_change.detach().cpu().item()) != 0.0:
                self.k = float((output_change / input_change).detach().cpu().item())
        self.prev_prev_raw_input = getattr(self, "previous_raw_input", None)
        self.previous_output = x.clone()
        self.cnt += 1
        if self.cnt >= max(int(getattr(self, "num_steps", 1)), 1):
            self.cnt = 0
    return x.to(torch.float32)


def _opensora_pab_block_forward(
    self,
    x,
    y,
    t,
    mask=None,
    x_mask=None,
    t0=None,
    T=None,
    S=None,
    timestep=None,
    all_timesteps=None,
):
    from opensora.models.layers.blocks import t2i_modulate

    manager: OpenSoraPABManager | None = getattr(self, "_vbench_pab_manager", None)
    timestep_value = _extract_timestep_value(timestep)
    timestep_sequence = _normalize_timestep_sequence(all_timesteps)

    batch_size, _, _ = x.shape
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
        self.scale_shift_table[None] + t.reshape(batch_size, 6, -1)
    ).chunk(6, dim=1)
    if x_mask is not None:
        shift_msa_zero, scale_msa_zero, gate_msa_zero, shift_mlp_zero, scale_mlp_zero, gate_mlp_zero = (
            self.scale_shift_table[None] + t0.reshape(batch_size, 6, -1)
        ).chunk(6, dim=1)

    broadcast_attn = False
    if manager is not None:
        if getattr(self, "temporal", False):
            broadcast_attn, self._vbench_pab_attn_count = manager.if_broadcast_temporal(
                timestep_value,
                getattr(self, "_vbench_pab_attn_count", 0),
            )
        else:
            broadcast_attn, self._vbench_pab_attn_count = manager.if_broadcast_spatial(
                timestep_value,
                getattr(self, "_vbench_pab_attn_count", 0),
            )

    if manager is not None and broadcast_attn and getattr(self, "_vbench_pab_last_attn", None) is not None:
        x_m_s = self._vbench_pab_last_attn
    else:
        x_m = t2i_modulate(self.norm1(x), shift_msa, scale_msa)
        if x_mask is not None:
            x_m_zero = t2i_modulate(self.norm1(x), shift_msa_zero, scale_msa_zero)
            x_m = self.t_mask_select(x_mask, x_m, x_m_zero, T, S)

        if getattr(self, "temporal", False):
            x_m = rearrange(x_m, "B (T S) C -> (B S) T C", T=T, S=S)
            x_m = self.attn(x_m)
            x_m = rearrange(x_m, "(B S) T C -> B (T S) C", T=T, S=S)
        else:
            x_m = rearrange(x_m, "B (T S) C -> (B T) S C", T=T, S=S)
            x_m = self.attn(x_m)
            x_m = rearrange(x_m, "(B T) S C -> B (T S) C", T=T, S=S)

        x_m_s = gate_msa * x_m
        if x_mask is not None:
            x_m_s_zero = gate_msa_zero * x_m
            x_m_s = self.t_mask_select(x_mask, x_m_s, x_m_s_zero, T, S)
        if manager is not None:
            self._vbench_pab_last_attn = x_m_s

    x = x + self.drop_path(x_m_s)

    broadcast_cross = False
    if manager is not None:
        broadcast_cross, self._vbench_pab_cross_count = manager.if_broadcast_cross(
            timestep_value,
            getattr(self, "_vbench_pab_cross_count", 0),
        )

    if manager is not None and broadcast_cross and getattr(self, "_vbench_pab_last_cross", None) is not None:
        x = x + self._vbench_pab_last_cross
    else:
        x_cross = self.cross_attn(x, y, mask)
        if manager is not None:
            self._vbench_pab_last_cross = x_cross
        x = x + x_cross

    broadcast_mlp = False
    broadcast_next = False
    skip_window = None
    if manager is not None:
        broadcast_mlp, self._vbench_pab_mlp_count, broadcast_next, skip_window = manager.if_broadcast_mlp(
            timestep_value,
            getattr(self, "_vbench_pab_mlp_count", 0),
            block_idx=int(getattr(self, "_vbench_block_idx", 0)),
            all_timesteps=timestep_sequence,
            is_temporal=bool(getattr(self, "temporal", False)),
        )

    if manager is not None and broadcast_mlp and skip_window is not None:
        x_m_s = manager.get_mlp_output(
            skip_window=skip_window,
            timestep=int(timestep_value),
            block_idx=int(getattr(self, "_vbench_block_idx", 0)),
            is_temporal=bool(getattr(self, "temporal", False)),
        )
    else:
        x_m = t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)
        if x_mask is not None:
            x_m_zero = t2i_modulate(self.norm2(x), shift_mlp_zero, scale_mlp_zero)
            x_m = self.t_mask_select(x_mask, x_m, x_m_zero, T, S)

        x_m = self.mlp(x_m)
        x_m_s = gate_mlp * x_m
        if x_mask is not None:
            x_m_s_zero = gate_mlp_zero * x_m
            x_m_s = self.t_mask_select(x_mask, x_m_s, x_m_s_zero, T, S)

        if manager is not None and broadcast_next and timestep_value is not None:
            manager.save_mlp_output(
                timestep=int(timestep_value),
                block_idx=int(getattr(self, "_vbench_block_idx", 0)),
                ff_output=x_m_s,
                is_temporal=bool(getattr(self, "temporal", False)),
            )

    x = x + self.drop_path(x_m_s)
    return x
