from __future__ import annotations

import types
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from einops import rearrange


_WAN_T2V_THRESHOLDS = {
    ("t2v-1.3B", False): 0.06,
    ("t2v-1.3B", True): 0.06,
    ("t2v-14B", False): 0.06,
    ("t2v-14B", True): 0.06,
}

_WAN_T2V_COEFFICIENTS = {
    ("t2v-1.3B", False): [2.39676752e03, -1.31110545e03, 2.01331979e02, -8.29855975e00, 1.37887774e-01],
    ("t2v-1.3B", True): [-5.21862437e04, 9.23041404e03, -5.28275948e02, 1.36987616e01, -4.99875664e-02],
    ("t2v-14B", False): [-5.78454975e03, 5.44950912e03, -1.81116592e03, 2.56271784e02, -1.30225240e01],
    ("t2v-14B", True): [-3.03318725e05, 4.90537029e04, -2.65530556e03, 5.87365115e01, -3.15583525e-01],
}

_HUNYUAN_COEFFICIENTS = [7.33226126e02, -4.01131952e02, 6.75869174e01, -3.14987800e00, 9.61237896e-02]
_OPENSORA_COEFFICIENTS = [2.17546007e02, -1.18329252e02, 2.68662585e01, -4.59364272e-02, 4.84426240e-02]


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


def _coerce_optional_float(value: Any, *, label: str) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric when provided. Received {value!r}.") from exc


def _ratio_delta(current: torch.Tensor, previous: torch.Tensor) -> float:
    denom = previous.abs().mean()
    if float(denom) == 0.0:
        return 0.0
    return float(((current - previous).abs().mean() / denom).detach().cpu().item())


def resolve_wan_teacache_config(*, task: str, method_config: dict[str, Any] | None) -> dict[str, Any]:
    resolved_task = str(task)
    raw = dict(method_config or {})
    use_ret_steps = _coerce_optional_bool(
        raw.get("use_ret_steps", raw.get("use_ref_steps")),
        label="wan.teacache.use_ret_steps",
    )
    use_ret_steps = bool(use_ret_steps) if use_ret_steps is not None else False
    teacache_thresh = _coerce_optional_float(
        raw.get("teacache_thresh", raw.get("threshold")),
        label="wan.teacache.teacache_thresh",
    )
    if teacache_thresh is None:
        teacache_thresh = _WAN_T2V_THRESHOLDS.get((resolved_task, use_ret_steps))
    coefficients = _WAN_T2V_COEFFICIENTS.get((resolved_task, use_ret_steps))
    if teacache_thresh is None or coefficients is None:
        raise ValueError(
            "TeaCache for Wan currently supports the vendored T2V tasks only. "
            f"Received task={resolved_task!r}, use_ret_steps={use_ret_steps!r}."
        )
    ret_steps = 10 if use_ret_steps else 2
    return {
        "teacache_thresh": float(teacache_thresh),
        "use_ret_steps": use_ret_steps,
        "coefficients": list(coefficients),
        "ret_steps": int(ret_steps),
        "cutoff_steps_offset": 0 if use_ret_steps else 2,
    }


def resolve_hunyuan_teacache_config(method_config: dict[str, Any] | None) -> dict[str, Any]:
    raw = dict(method_config or {})
    rel_l1_thresh = _coerce_optional_float(
        raw.get("rel_l1_thresh", raw.get("teacache_rel_l1_thresh", raw.get("teacache_thresh"))),
        label="hunyuan.teacache.rel_l1_thresh",
    )
    return {
        "rel_l1_thresh": float(0.15 if rel_l1_thresh is None else rel_l1_thresh),
    }


def resolve_opensora_teacache_config(method_config: dict[str, Any] | None) -> dict[str, Any]:
    raw = dict(method_config or {})
    rel_l1_thresh = _coerce_optional_float(
        raw.get("rel_l1_thresh", raw.get("teacache_rel_l1_thresh", raw.get("teacache_thresh"))),
        label="opensora.teacache.rel_l1_thresh",
    )
    return {
        "rel_l1_thresh": float(0.10 if rel_l1_thresh is None else rel_l1_thresh),
    }


def configure_wan_teacache(
    pipeline: Any,
    *,
    method_name: str,
    method_config: dict[str, Any] | None,
    task: str,
    sample_steps: int,
) -> dict[str, Any] | None:
    model = pipeline.model
    if not hasattr(model, "_vbench_teacache_original_forward"):
        model._vbench_teacache_original_forward = model.forward

    enabled = str(method_name) == "teacache"
    if enabled:
        model.forward = types.MethodType(_wan_teacache_forward, model)
        model._vbench_teacache_patch_applied = True
    else:
        original_forward = getattr(model, "_vbench_teacache_original_forward", None)
        if original_forward is not None:
            model.forward = original_forward
    model.enable_teacache = enabled
    model.cnt = 0
    model.num_steps = max(int(sample_steps) * 2, 1)
    model.accumulated_rel_l1_distance_even = 0.0
    model.accumulated_rel_l1_distance_odd = 0.0
    model.previous_e0_even = None
    model.previous_e0_odd = None
    model.previous_residual_even = None
    model.previous_residual_odd = None

    if not enabled:
        model.use_ref_steps = False
        model.teacache_thresh = 0.0
        model.coefficients = None
        model.ret_steps = 0
        model.cutoff_steps = max(int(sample_steps) * 2, 1)
        return None

    resolved = resolve_wan_teacache_config(task=str(task), method_config=method_config)
    model.use_ref_steps = bool(resolved["use_ret_steps"])
    model.teacache_thresh = float(resolved["teacache_thresh"])
    model.coefficients = list(resolved["coefficients"])
    model.ret_steps = int(resolved["ret_steps"])
    model.cutoff_steps = max(int(sample_steps) * 2 - int(resolved["cutoff_steps_offset"]), 1)
    return {
        "teacache_thresh": float(resolved["teacache_thresh"]),
        "use_ret_steps": bool(resolved["use_ret_steps"]),
    }


def restore_wan_teacache_state(pipeline: Any) -> None:
    model = pipeline.model
    original_forward = getattr(model, "_vbench_teacache_original_forward", None)
    if original_forward is not None:
        model.forward = original_forward
    model.enable_teacache = False
    model.cnt = 0
    model.accumulated_rel_l1_distance_even = 0.0
    model.accumulated_rel_l1_distance_odd = 0.0
    model.previous_e0_even = None
    model.previous_e0_odd = None
    model.previous_residual_even = None
    model.previous_residual_odd = None
    model.use_ref_steps = False
    model.teacache_thresh = 0.0
    model.coefficients = None
    model.ret_steps = 0


def configure_hunyuan_teacache(
    sampler: Any,
    *,
    method_name: str,
    method_config: dict[str, Any] | None,
    sample_steps: int,
) -> dict[str, Any] | None:
    transformer = sampler.pipeline.transformer
    if not getattr(transformer, "_vbench_teacache_patch_applied", False):
        transformer.forward = types.MethodType(_hunyuan_teacache_forward, transformer)
        transformer._vbench_teacache_patch_applied = True

    enabled = str(method_name) == "teacache"
    transformer.enable_teacache = enabled
    transformer.cnt = 0
    transformer.num_steps = max(int(sample_steps), 1)
    transformer.accumulated_rel_l1_distance = 0.0
    transformer.previous_modulated_input = None
    transformer.previous_residual = None

    if not enabled:
        transformer.rel_l1_thresh = 0.0
        return None

    resolved = resolve_hunyuan_teacache_config(method_config)
    transformer.rel_l1_thresh = float(resolved["rel_l1_thresh"])
    return dict(resolved)


def configure_opensora_teacache(
    model: Any,
    *,
    enabled: bool,
    method_config: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not getattr(model, "_vbench_teacache_patch_applied", False):
        model.forward = types.MethodType(_opensora_teacache_forward, model)
        model._vbench_teacache_patch_applied = True

    model.enable_teacache = bool(enabled)
    model.accumulated_rel_l1_distance = 0.0
    model.previous_modulated_input = None
    model.previous_residual = None

    if not enabled:
        model.rel_l1_thresh = 0.0
        return None

    resolved = resolve_opensora_teacache_config(method_config)
    model.rel_l1_thresh = float(resolved["rel_l1_thresh"])
    return dict(resolved)


def _wan_teacache_forward(
    self,
    x,
    t,
    context,
    seq_len,
    clip_fea=None,
    y=None,
):
    from wan.modules.model import sinusoidal_embedding_1d

    if self.model_type in {"i2v", "flf2v"}:
        assert clip_fea is not None and y is not None
    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)

    if y is not None:
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

    x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
    grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
    x = [u.flatten(2).transpose(1, 2) for u in x]
    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
    assert seq_lens.max() <= seq_len
    x = torch.cat(
        [torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1) for u in x]
    )

    with torch.amp.autocast("cuda", dtype=torch.float32):
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

    should_calc_even = True
    should_calc_odd = True
    use_even_branch = True
    if getattr(self, "enable_teacache", False):
        modulated_inp = e0 if bool(getattr(self, "use_ref_steps", False)) else e
        coefficients = np.poly1d(getattr(self, "coefficients", []))
        if self.cnt % 2 == 0:
            use_even_branch = True
            if self.cnt < getattr(self, "ret_steps", 0) or self.cnt >= getattr(self, "cutoff_steps", 0):
                self.accumulated_rel_l1_distance_even = 0.0
                should_calc_even = True
            else:
                prev = getattr(self, "previous_e0_even", None)
                if prev is None:
                    should_calc_even = True
                    self.accumulated_rel_l1_distance_even = 0.0
                else:
                    self.accumulated_rel_l1_distance_even += float(coefficients(_ratio_delta(modulated_inp, prev)))
                    should_calc_even = self.accumulated_rel_l1_distance_even >= float(getattr(self, "teacache_thresh", 0.0))
                    if should_calc_even:
                        self.accumulated_rel_l1_distance_even = 0.0
            self.previous_e0_even = modulated_inp.clone()
        else:
            use_even_branch = False
            if self.cnt < getattr(self, "ret_steps", 0) or self.cnt >= getattr(self, "cutoff_steps", 0):
                self.accumulated_rel_l1_distance_odd = 0.0
                should_calc_odd = True
            else:
                prev = getattr(self, "previous_e0_odd", None)
                if prev is None:
                    should_calc_odd = True
                    self.accumulated_rel_l1_distance_odd = 0.0
                else:
                    self.accumulated_rel_l1_distance_odd += float(coefficients(_ratio_delta(modulated_inp, prev)))
                    should_calc_odd = self.accumulated_rel_l1_distance_odd >= float(getattr(self, "teacache_thresh", 0.0))
                    if should_calc_odd:
                        self.accumulated_rel_l1_distance_odd = 0.0
            self.previous_e0_odd = modulated_inp.clone()

    if getattr(self, "enable_teacache", False):
        if use_even_branch:
            if not should_calc_even and getattr(self, "previous_residual_even", None) is not None:
                x = x + self.previous_residual_even
            else:
                origin_x = x.clone()
                for block in self.blocks:
                    x = block(x, **kwargs)
                self.previous_residual_even = x - origin_x
        else:
            if not should_calc_odd and getattr(self, "previous_residual_odd", None) is not None:
                x = x + self.previous_residual_odd
            else:
                origin_x = x.clone()
                for block in self.blocks:
                    x = block(x, **kwargs)
                self.previous_residual_odd = x - origin_x
    else:
        for block in self.blocks:
            x = block(x, **kwargs)

    x = self.head(x, e)
    x = self.unpatchify(x, grid_sizes)
    self.cnt += 1
    if self.cnt >= max(int(getattr(self, "num_steps", 1)), 1):
        self.cnt = 0
    return [u.float() for u in x]


def _hunyuan_teacache_forward(
    self,
    x: torch.Tensor,
    t: torch.Tensor,
    text_states: torch.Tensor = None,
    text_mask: torch.Tensor = None,
    text_states_2: torch.Tensor | None = None,
    freqs_cos: torch.Tensor | None = None,
    freqs_sin: torch.Tensor | None = None,
    guidance: torch.Tensor = None,
    return_dict: bool = True,
):
    from hyvideo.modules.modulate_layers import modulate
    from hyvideo.modules.attenion import get_cu_seqlens

    out = {}
    raw_input = x.clone() if (getattr(self, "enable_easycache", False) or getattr(self, "enable_aligncache", False)) else None
    img = x
    txt = text_states
    _, _, ot, oh, ow = x.shape
    tt, th, tw = (
        ot // self.patch_size[0],
        oh // self.patch_size[1],
        ow // self.patch_size[2],
    )

    vec = self.time_in(t)
    vec = vec + self.vector_in(text_states_2)
    if self.guidance_embed:
        if guidance is None:
            raise ValueError("Didn't get guidance strength for guidance distilled model.")
        vec = vec + self.guidance_in(guidance)

    img = self.img_in(img)
    if self.text_projection == "linear":
        txt = self.txt_in(txt)
    elif self.text_projection == "single_refiner":
        txt = self.txt_in(txt, t, text_mask if self.use_attention_mask else None)
    else:
        raise NotImplementedError(f"Unsupported text_projection: {self.text_projection}")

    txt_seq_len = txt.shape[1]
    img_seq_len = img.shape[1]
    cu_seqlens_q = get_cu_seqlens(text_mask, img_seq_len)
    cu_seqlens_kv = cu_seqlens_q
    max_seqlen_q = img_seq_len + txt_seq_len
    max_seqlen_kv = max_seqlen_q
    freqs_cis = (freqs_cos, freqs_sin) if freqs_cos is not None else None

    if getattr(self, "enable_pab", False):
        from . import pab as pab_runtime

        pab_runtime.prepare_hunyuan_pab_state(self, t)

    aligncache_decision = None
    if getattr(self, "enable_aligncache", False):
        from . import aligncache as aligncache_runtime

        aligncache_decision = aligncache_runtime.prepare_single_aligncache_step(
            self,
            raw_input=raw_input,
        )
        if aligncache_decision is not None and aligncache_decision["exec_mode"] == "skip":
            predicted_output = aligncache_runtime.complete_single_aligncache_skip(
                self,
                raw_input=raw_input,
                predicted_output=aligncache_decision["predicted_output"],
                decision=aligncache_decision,
            )[0]
            if return_dict:
                out["x"] = predicted_output
                return out
            return predicted_output

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
            if return_dict:
                out["x"] = result
                return out
            return result

    if not getattr(self, "enable_easycache", False):
        should_calc = True
    if getattr(self, "enable_teacache", False):
        (
            img_mod1_shift,
            img_mod1_scale,
            _img_mod1_gate,
            _img_mod2_shift,
            _img_mod2_scale,
            _img_mod2_gate,
        ) = self.double_blocks[0].img_mod(vec.clone()).chunk(6, dim=-1)
        modulated_inp = modulate(
            self.double_blocks[0].img_norm1(img.clone()),
            shift=img_mod1_shift,
            scale=img_mod1_scale,
        )
        if self.cnt in {0, max(int(getattr(self, "num_steps", 1)) - 1, 0)} or getattr(
            self, "previous_modulated_input", None
        ) is None:
            should_calc = True
            self.accumulated_rel_l1_distance = 0.0
        else:
            rescale_func = np.poly1d(_HUNYUAN_COEFFICIENTS)
            self.accumulated_rel_l1_distance += float(
                rescale_func(_ratio_delta(modulated_inp, self.previous_modulated_input))
            )
            should_calc = self.accumulated_rel_l1_distance >= float(getattr(self, "rel_l1_thresh", 0.0))
            if should_calc:
                self.accumulated_rel_l1_distance = 0.0
        self.previous_modulated_input = modulated_inp
        self.cnt += 1
        if self.cnt >= max(int(getattr(self, "num_steps", 1)), 1):
            self.cnt = 0

    if getattr(self, "enable_teacache", False) and not should_calc and getattr(self, "previous_residual", None) is not None:
        img = img + self.previous_residual
    else:
        origin_img = img.clone()
        for block in self.double_blocks:
            img, txt = block(
                img,
                txt,
                vec,
                cu_seqlens_q,
                cu_seqlens_kv,
                max_seqlen_q,
                max_seqlen_kv,
                freqs_cis,
            )

        x = torch.cat((img, txt), 1)
        if len(self.single_blocks) > 0:
            for block in self.single_blocks:
                x = block(
                    x,
                    vec,
                    txt_seq_len,
                    cu_seqlens_q,
                    cu_seqlens_kv,
                    max_seqlen_q,
                    max_seqlen_kv,
                    (freqs_cos, freqs_sin),
                )

        img = x[:, :img_seq_len, ...]
        if getattr(self, "enable_teacache", False):
            self.previous_residual = img - origin_img

    img = self.final_layer(img, vec)
    img = self.unpatchify(img, tt, th, tw)
    if getattr(self, "enable_aligncache", False):
        from . import aligncache as aligncache_runtime

        resolved_output = aligncache_runtime.complete_single_aligncache_exact(
            self,
            raw_input=raw_input,
            exact_output=img,
            decision=aligncache_decision or {"phase": "mid", "step_idx": int(getattr(self, "cnt", 0))},
        )[0]
        if return_dict:
            out["x"] = resolved_output
            return out
        return resolved_output
    if getattr(self, "enable_easycache", False):
        self.cache = img - raw_input
        previous_output = getattr(self, "previous_output", None)
        prev_prev_raw_input = getattr(self, "prev_prev_raw_input", None)
        if previous_output is not None and prev_prev_raw_input is not None:
            output_change = (img - previous_output).abs().mean()
            input_change = (self.previous_raw_input - prev_prev_raw_input).abs().mean()
            if float(input_change.detach().cpu().item()) != 0.0:
                self.k = float((output_change / input_change).detach().cpu().item())
        self.prev_prev_raw_input = getattr(self, "previous_raw_input", None)
        self.previous_output = img.clone()
        self.cnt += 1
        if self.cnt >= max(int(getattr(self, "num_steps", 1)), 1):
            self.cnt = 0
    if return_dict:
        out["x"] = img
        return out
    return img


def _opensora_teacache_forward(
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
    dtype = self.x_embedder.proj.weight.dtype
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

    should_calc = True
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
            rescale_func = np.poly1d(_OPENSORA_COEFFICIENTS)
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
            x = auto_grad_checkpoint(spatial_block, x, y, t_mlp, y_lens, x_mask, t0_mlp, t_size, s_size)
            x = auto_grad_checkpoint(temporal_block, x, y, t_mlp, y_lens, x_mask, t0_mlp, t_size, s_size)
        if getattr(self, "enable_teacache", False):
            self.previous_residual = x - origin_x

    if self.enable_sequence_parallelism:
        x = rearrange(x, "B (T S) C -> B T S C", T=t_size, S=s_size)
        x = gather_forward_split_backward(x, get_sequence_parallel_group(), dim=2, grad_scale="up")
        s_size = s_size * dist.get_world_size(get_sequence_parallel_group())
        x = rearrange(x, "B T S C -> B (T S) C", T=t_size, S=s_size)

    x = self.final_layer(x, t, x_mask, t0, t_size, s_size)
    x = self.unpatchify(x, t_size, h_size, w_size, tx, hx, wx)
    return x.to(torch.float32)
