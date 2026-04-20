import argparse
import gc
import logging
import math
import os
import random
import sys
import warnings
from contextlib import contextmanager
from datetime import datetime
from time import time

from PIL import Image
import numpy as np
import torch
import torch.cuda.amp as amp
amp.autocast = lambda *args, **kwargs: torch.autocast("cuda", *args, **kwargs)
import torch.distributed as dist
import torchvision.transforms.functional as TF
from tqdm import tqdm

import wan
from task_config_utils import normalize_ckpt_dir_for_task, parse_task_name_arg
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.modules.model import sinusoidal_embedding_1d
from wan.utils.fm_solvers import (FlowDPMSolverMultistepScheduler, get_sampling_sigmas, retrieve_timesteps)
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from wan.utils.utils import cache_image, cache_video, str2bool

warnings.filterwarnings('ignore')

EXAMPLE_PROMPT = {
    "t2v-1.3B": {
        "prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "t2v-14B": {
        "prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "t2i-14B": {
        "prompt": "一个朴素端庄的美人",
    },
    "i2v-14B": {
        "prompt": "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside.",
        "image": "examples/i2v_input.JPG",
    },
}


def t2v_generate(self,
                 input_prompt,
                 size=(1280, 720),
                 frame_num=81,
                 shift=5.0,
                 sample_solver="unipc",
                 sampling_steps=50,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True):
    f_count = frame_num
    target_shape = (self.vae.model.z_dim, (f_count - 1) // self.vae_stride[0] + 1,
                    size[1] // self.vae_stride[1],
                    size[0] // self.vae_stride[2])

    seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                        (self.patch_size[1] * self.patch_size[2]) *
                        target_shape[1] / self.sp_size) * self.sp_size

    if n_prompt == "":
        n_prompt = self.sample_neg_prompt
    seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
    seed_g = torch.Generator(device=self.device)
    seed_g.manual_seed(seed)

    if not self.t5_cpu:
        self.text_encoder.model.to(self.device)
        context = self.text_encoder([input_prompt], self.device)
        context_null = self.text_encoder([n_prompt], self.device)
        if offload_model:
            self.text_encoder.model.cpu()
    else:
        context = self.text_encoder([input_prompt], torch.device("cpu"))
        context_null = self.text_encoder([n_prompt], torch.device("cpu"))
        context = [tensor.to(self.device) for tensor in context]
        context_null = [tensor.to(self.device) for tensor in context_null]

    noise = [
        torch.randn(
            target_shape[0],
            target_shape[1],
            target_shape[2],
            target_shape[3],
            dtype=torch.float32,
            device=self.device,
            generator=seed_g,
        )
    ]

    @contextmanager
    def noop_no_sync():
        yield

    no_sync = getattr(self.model, "no_sync", noop_no_sync)

    with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():
        if sample_solver == "unipc":
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False,
            )
            sample_scheduler.set_timesteps(
                sampling_steps, device=self.device, shift=shift)
            timesteps = sample_scheduler.timesteps
        elif sample_solver == "dpm++":
            sample_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False,
            )
            sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
            timesteps, _ = retrieve_timesteps(
                sample_scheduler,
                device=self.device,
                sigmas=sampling_sigmas,
            )
        else:
            raise NotImplementedError("Unsupported solver.")

        latents = noise
        arg_c = {"context": context, "seq_len": seq_len}
        arg_null = {"context": context_null, "seq_len": seq_len}

        for _, timestep_item in enumerate(tqdm(timesteps)):
            torch.cuda.synchronize()
            start_time = time()

            latent_model_input = latents
            timestep_tensor = torch.stack([timestep_item])

            self.model.to(self.device)
            noise_pred_cond = self.model(
                latent_model_input, t=timestep_tensor, **arg_c)[0]
            noise_pred_uncond = self.model(
                latent_model_input, t=timestep_tensor, **arg_null)[0]
            noise_pred = noise_pred_uncond + guide_scale * (
                noise_pred_cond - noise_pred_uncond)

            torch.cuda.synchronize()
            self.cost_time += (time() - start_time)

            temp_x0 = sample_scheduler.step(
                noise_pred.unsqueeze(0),
                timestep_item,
                latents[0].unsqueeze(0),
                return_dict=False,
                generator=seed_g,
            )[0]
            latents = [temp_x0.squeeze(0)]

        x0 = latents
        if offload_model:
            self.model.cpu()
            torch.cuda.empty_cache()
        if self.rank == 0:
            videos = self.vae.decode(x0)

    del noise, latents
    del sample_scheduler
    if offload_model:
        gc.collect()
        torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()

    return videos[0] if self.rank == 0 else None


def i2v_generate(self,
                 input_prompt,
                 img,
                 max_area=720 * 1280,
                 frame_num=81,
                 shift=5.0,
                 sample_solver="unipc",
                 sampling_steps=40,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True):
    img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)

    f_count = frame_num
    h, w = img.shape[1:]
    aspect_ratio = h / w
    lat_h = round(
        np.sqrt(max_area * aspect_ratio) // self.vae_stride[1] //
        self.patch_size[1] * self.patch_size[1])
    lat_w = round(
        np.sqrt(max_area / aspect_ratio) // self.vae_stride[2] //
        self.patch_size[2] * self.patch_size[2])
    h = lat_h * self.vae_stride[1]
    w = lat_w * self.vae_stride[2]

    max_seq_len = ((f_count - 1) // self.vae_stride[0] + 1) * lat_h * lat_w // (
        self.patch_size[1] * self.patch_size[2])
    max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size

    seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
    seed_g = torch.Generator(device=self.device)
    seed_g.manual_seed(seed)
    noise = torch.randn(
        self.vae.model.z_dim,
        (f_count - 1) // self.vae_stride[0] + 1,
        lat_h,
        lat_w,
        dtype=torch.float32,
        generator=seed_g,
        device=self.device,
    )

    msk = torch.ones(1, f_count, lat_h, lat_w, device=self.device)
    msk[:, 1:] = 0
    msk = torch.concat([
        torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1),
        msk[:, 1:]
    ], dim=1)
    msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
    msk = msk.transpose(1, 2)[0]

    if n_prompt == "":
        n_prompt = self.sample_neg_prompt

    if not self.t5_cpu:
        self.text_encoder.model.to(self.device)
        context = self.text_encoder([input_prompt], self.device)
        context_null = self.text_encoder([n_prompt], self.device)
        if offload_model:
            self.text_encoder.model.cpu()
    else:
        context = self.text_encoder([input_prompt], torch.device("cpu"))
        context_null = self.text_encoder([n_prompt], torch.device("cpu"))
        context = [tensor.to(self.device) for tensor in context]
        context_null = [tensor.to(self.device) for tensor in context_null]

    self.clip.model.to(self.device)
    clip_context = self.clip.visual([img[:, None, :, :]])
    if offload_model:
        self.clip.model.cpu()

    y = self.vae.encode([
        torch.concat([
            torch.nn.functional.interpolate(
                img[None].cpu(), size=(h, w), mode="bicubic").transpose(0, 1),
            torch.zeros(3, f_count - 1, h, w),
        ], dim=1).to(self.device)
    ])[0]
    y = torch.concat([msk, y])

    @contextmanager
    def noop_no_sync():
        yield

    no_sync = getattr(self.model, "no_sync", noop_no_sync)

    with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():
        if sample_solver == "unipc":
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False,
            )
            sample_scheduler.set_timesteps(
                sampling_steps, device=self.device, shift=shift)
            timesteps = sample_scheduler.timesteps
        elif sample_solver == "dpm++":
            sample_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False,
            )
            sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
            timesteps, _ = retrieve_timesteps(
                sample_scheduler,
                device=self.device,
                sigmas=sampling_sigmas,
            )
        else:
            raise NotImplementedError("Unsupported solver.")

        latent = noise
        arg_c = {
            "context": [context[0]],
            "clip_fea": clip_context,
            "seq_len": max_seq_len,
            "y": [y],
        }
        arg_null = {
            "context": context_null,
            "clip_fea": clip_context,
            "seq_len": max_seq_len,
            "y": [y],
        }

        if offload_model:
            torch.cuda.empty_cache()

        self.model.to(self.device)
        for _, timestep_item in enumerate(tqdm(timesteps)):
            torch.cuda.synchronize()
            start_time = time()

            latent_model_input = [latent.to(self.device)]
            timestep_tensor = torch.stack([timestep_item]).to(self.device)

            noise_pred_cond = self.model(
                latent_model_input, t=timestep_tensor, **arg_c)[0].to(
                torch.device("cpu") if offload_model else self.device)
            if offload_model:
                torch.cuda.empty_cache()

            noise_pred_uncond = self.model(
                latent_model_input, t=timestep_tensor, **arg_null)[0].to(
                torch.device("cpu") if offload_model else self.device)
            if offload_model:
                torch.cuda.empty_cache()

            noise_pred = noise_pred_uncond + guide_scale * (
                noise_pred_cond - noise_pred_uncond)

            latent = latent.to(torch.device("cpu") if offload_model else self.device)

            torch.cuda.synchronize()
            self.cost_time += (time() - start_time)

            temp_x0 = sample_scheduler.step(
                noise_pred.unsqueeze(0),
                timestep_item,
                latent.unsqueeze(0),
                return_dict=False,
                generator=seed_g,
            )[0]
            latent = temp_x0.squeeze(0)
            x0 = [latent.to(self.device)]
            del latent_model_input, timestep_tensor

        if offload_model:
            self.model.cpu()
            torch.cuda.empty_cache()

        if self.rank == 0:
            videos = self.vae.decode(x0)

    del noise, latent
    del sample_scheduler
    if offload_model:
        gc.collect()
        torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()

    return videos[0] if self.rank == 0 else None


def _validate_args(args):
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.task in WAN_CONFIGS, f"Unsupport task: {args.task}"
    assert args.task in EXAMPLE_PROMPT, f"Unsupport task: {args.task}"

    if args.sample_steps is None:
        args.sample_steps = 40 if "i2v" in args.task else 50

    if args.sample_shift is None:
        args.sample_shift = 5.0
        if "i2v" in args.task and args.size in ["832*480", "480*832"]:
            args.sample_shift = 3.0

    if args.frame_num is None:
        args.frame_num = 1 if "t2i" in args.task else 81

    if "t2i" in args.task:
        assert args.frame_num == 1, (
            f"Unsupport frame_num {args.frame_num} for task {args.task}")

    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(
        0, sys.maxsize)

    assert args.size in SUPPORTED_SIZES[args.task], (
        f"Unsupport size {args.size} for task {args.task}, supported sizes are: "
        f"{', '.join(SUPPORTED_SIZES[args.task])}")


def _init_logging(rank):
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)],
        )
    else:
        logging.basicConfig(level=logging.ERROR)

V2_DEFAULTS = dict(
    enable=True,
    early_end_ratio=0.15,
    late_start_ratio=0.86,
    subsample_stride=64,
    qc_ratio_ema_decay=0.8,
    qc_dir_ema_decay=0.8,
    beta_mid=0.35,
    beta_late=0.15,
    ratio_tol_mid=0.30,
    ratio_tol_late=0.18,
    dir_min_mid=0.80,
    dir_min_late=0.88,
    use_second_order=True,
    use_online_scale=True,
    scale_ema_decay=0.85,
    scale_clamp_low=0.85,
    scale_clamp_high=1.18,
    b2_enable=True,
    b2_exact_early=True,
    b2_max_pred_age_mid=1,
    b2_max_pred_age_late=2,
    b2_cond_rel_cap_mid=0.0050,
    b2_cond_rel_cap_late=0.0200,
    b2_alpha_prior_early=1.279282,
    b2_alpha_prior_mid=1.018750,
    b2_alpha_prior_late=0.972800,
    b2_alpha_ema_decay=0.85,
    b2_blend_mid=0.35,
    b2_blend_late=0.20,
    b2_disagree_rel_cap_mid=0.0200,
    b2_disagree_rel_cap_late=0.0600,
)


def _clone_tensor_list(values):
    return [value.detach().clone() for value in values]


def _tensor_list_subsample(values, stride):
    stride = max(int(stride), 1)
    chunks = []
    for value in values:
        chunks.append(value.detach().reshape(-1)[::stride].to(torch.float32).clone())
    return torch.cat(chunks, dim=0) if chunks else None


def _tensor_list_cosine(first, second):
    if first is None or second is None:
        return None
    first_vec = first.reshape(-1).float()
    second_vec = second.reshape(-1).float()
    denom = torch.norm(first_vec, p=2) * torch.norm(second_vec, p=2)
    if float(denom.item()) <= 1e-12:
        return None
    value = (first_vec.dot(second_vec) / denom).clamp(min=-1.0, max=1.0)
    return float(value.item())


def _tensor_list_rel_mse(prediction, reference):
    if prediction is None or reference is None:
        return None
    numer = None
    denom = None
    for pred_value, ref_value in zip(prediction, reference):
        pred_f = pred_value.detach().float()
        ref_f = ref_value.detach().float()
        current_numer = torch.sum((pred_f - ref_f) ** 2, dtype=torch.float32)
        current_denom = torch.sum(ref_f * ref_f, dtype=torch.float32)
        numer = current_numer if numer is None else numer + current_numer
        denom = current_denom if denom is None else denom + current_denom
    if numer is None or denom is None:
        return None
    return float((numer / denom.clamp_min(1e-12)).item())


def _clamp_float(value, low, high):
    return max(float(low), min(float(high), float(value)))


def _fit_affine_scale_list(target, basis, default_alpha=1.0):
    alpha = float(default_alpha)
    numer = None
    denom = None
    for target_value, basis_value in zip(target, basis):
        target_f = target_value.detach().float()
        basis_f = basis_value.detach().float()
        current_numer = torch.sum(target_f * basis_f, dtype=torch.float32)
        current_denom = torch.sum(basis_f * basis_f, dtype=torch.float32)
        numer = current_numer if numer is None else numer + current_numer
        denom = current_denom if denom is None else denom + current_denom
    if numer is not None and denom is not None and float(denom.item()) > 1e-12:
        alpha = float((numer / denom.clamp_min(1e-12)).item())
    return _clamp_float(alpha, -1.5, 1.5)


def _blend_tensor_lists(base, target, weight):
    weight = float(weight)
    if weight <= 1e-6:
        return [value.detach().clone() for value in base]
    if weight >= 1.0:
        return [
            value.detach().to(dtype=base_value.dtype)
            for base_value, value in zip(base, target)
        ]
    return [
        base_value + (value.detach().to(dtype=base_value.dtype) - base_value) * weight
        for base_value, value in zip(base, target)
    ]


def _update_running_ema(previous, current, decay):
    if current is None:
        return previous
    if previous is None:
        return current
    return decay * previous + (1.0 - decay) * current


def _get_phase(step_idx, total_steps, cfg):
    ratio = step_idx / max(total_steps - 1, 1)
    if ratio < cfg['early_end_ratio']:
        return 'early'
    if ratio < cfg['late_start_ratio']:
        return 'mid'
    return 'late'


def _correction_params(cfg, phase):
    if phase == 'mid':
        return cfg['beta_mid'], cfg['ratio_tol_mid'], cfg['dir_min_mid']
    if phase == 'late':
        return cfg['beta_late'], cfg['ratio_tol_late'], cfg['dir_min_late']
    return 0.0, 0.0, 1.0


def _init_phase_ema(value):
    stored = None if value is None else float(value)
    return {
        'early': stored,
        'mid': stored,
        'late': stored,
    }


def _ratio_gate(ratio_ema, ratio_tol):
    if ratio_ema is None:
        return 0.0
    drift = abs(float(ratio_ema) - 1.0)
    return max(0.0, 1.0 - drift / max(float(ratio_tol), 1e-6))


def _dir_gate(direction, dir_min):
    if direction is None:
        return 0.0
    return max(0.0, min(1.0, (float(direction) - float(dir_min)) / max(1.0 - float(dir_min), 1e-6)))


def _phase_skip_threshold(self, phase):
    phase_thresholds = getattr(self, 'phase_skip_thresholds', None)
    if isinstance(phase_thresholds, dict):
        if phase in phase_thresholds and phase_thresholds[phase] is not None:
            return float(phase_thresholds[phase])
    attr_name = f'phase_thresh_{phase}'
    if hasattr(self, attr_name):
        value = getattr(self, attr_name)
        if value is not None:
            return float(value)
    return float(getattr(self, 'thresh', 0.0))


def _apply_v2_aggressiveness(cfg, aggressiveness):
    cfg = dict(cfg)
    aggressiveness = float(aggressiveness)
    beta_scale = _clamp_float(1.0 + 0.60 * aggressiveness, 0.50, 1.35)
    blend_scale = _clamp_float(1.0 + 0.50 * aggressiveness, 0.50, 1.40)
    clamp_delta = 0.08 * aggressiveness

    cfg['beta_mid'] *= beta_scale
    cfg['beta_late'] *= beta_scale
    cfg['b2_blend_mid'] *= blend_scale
    cfg['b2_blend_late'] *= blend_scale
    cfg['scale_clamp_low'] = max(0.60, cfg['scale_clamp_low'] - clamp_delta)
    cfg['scale_clamp_high'] = max(
        cfg['scale_clamp_low'],
        cfg['scale_clamp_high'] + clamp_delta,
    )
    return cfg


def _build_v2_cfg(args):
    cfg = dict(V2_DEFAULTS)
    cfg['enable'] = args.v2_enable
    cfg['use_second_order'] = args.v2_use_second_order
    cfg['use_online_scale'] = args.v2_use_online_scale
    cfg['subsample_stride'] = args.v2_subsample_stride
    cfg['qc_ratio_ema_decay'] = args.v2_ratio_ema_decay
    cfg['qc_dir_ema_decay'] = args.v2_dir_ema_decay
    cfg['beta_mid'] = args.v2_beta_mid
    cfg['beta_late'] = args.v2_beta_late
    cfg['ratio_tol_mid'] = args.v2_ratio_tol_mid
    cfg['ratio_tol_late'] = args.v2_ratio_tol_late
    cfg['dir_min_mid'] = args.v2_dir_min_mid
    cfg['dir_min_late'] = args.v2_dir_min_late
    cfg['scale_ema_decay'] = args.v2_scale_ema_decay
    cfg['scale_clamp_low'] = args.v2_scale_clamp_low
    cfg['scale_clamp_high'] = args.v2_scale_clamp_high
    cfg['b2_enable'] = args.v2_b2_enable
    cfg['b2_exact_early'] = args.v2_b2_exact_early
    cfg['b2_max_pred_age_mid'] = args.v2_b2_max_pred_age_mid
    cfg['b2_max_pred_age_late'] = args.v2_b2_max_pred_age_late
    cfg['b2_cond_rel_cap_mid'] = args.v2_b2_cond_rel_cap_mid
    cfg['b2_cond_rel_cap_late'] = args.v2_b2_cond_rel_cap_late
    cfg['b2_alpha_prior_early'] = args.v2_b2_alpha_prior_early
    cfg['b2_alpha_prior_mid'] = args.v2_b2_alpha_prior_mid
    cfg['b2_alpha_prior_late'] = args.v2_b2_alpha_prior_late
    cfg['b2_alpha_ema_decay'] = args.v2_b2_alpha_ema_decay
    cfg['b2_blend_mid'] = args.v2_b2_blend_mid
    cfg['b2_blend_late'] = args.v2_b2_blend_late
    cfg['b2_disagree_rel_cap_mid'] = args.v2_b2_disagree_rel_cap_mid
    cfg['b2_disagree_rel_cap_late'] = args.v2_b2_disagree_rel_cap_late
    return _apply_v2_aggressiveness(cfg, args.v2_aggressiveness)


def _reset_shadow_pair_state(self):
    self.shadow_cache_even = None
    self.shadow_cache_odd = None
    self.shadow_prev_cache_even = None
    self.shadow_prev_cache_odd = None
    self.shadow_cache_even_sub = None
    self.shadow_cache_odd_sub = None
    self.shadow_pair_ratio_ema_even = None
    self.shadow_pair_ratio_ema_odd = None
    self.shadow_pair_dir_ema_even = None
    self.shadow_pair_dir_ema_odd = None
    self.shadow_guided_pair_sub = None
    self.shadow_guided_pair_ratio_ema = None
    self.shadow_guided_pair_dir_ema = None
    self.shadow_pair_valid = False
    self.shadow_pair_scale_ema_even = _init_phase_ema(1.0)
    self.shadow_pair_scale_ema_odd = _init_phase_ema(1.0)


def _store_pair_branch_history(self, residual, is_cond):
    residual_detached = _clone_tensor_list(residual)
    residual_sub = _tensor_list_subsample(
        residual_detached, self.v2_cfg['subsample_stride'])

    if is_cond:
        last = self.shadow_cache_even
        last_sub = self.shadow_cache_even_sub
        ratio_ema = self.shadow_pair_ratio_ema_even
        dir_ema = self.shadow_pair_dir_ema_even
    else:
        last = self.shadow_cache_odd
        last_sub = self.shadow_cache_odd_sub
        ratio_ema = self.shadow_pair_ratio_ema_odd
        dir_ema = self.shadow_pair_dir_ema_odd

    if last_sub is not None and residual_sub is not None:
        prev_norm = torch.norm(last_sub, p=2).clamp_(min=1e-12)
        curr_norm = torch.norm(residual_sub, p=2)
        ratio = float((curr_norm / prev_norm).item())
        direction = _tensor_list_cosine(residual_sub, last_sub)
        ratio_ema = _update_running_ema(
            ratio_ema, ratio, self.v2_cfg['qc_ratio_ema_decay'])
        dir_ema = _update_running_ema(
            dir_ema, direction, self.v2_cfg['qc_dir_ema_decay'])

    if is_cond:
        self.shadow_prev_cache_even = last
        self.shadow_cache_even = residual_detached
        self.shadow_cache_even_sub = residual_sub
        self.shadow_pair_ratio_ema_even = ratio_ema
        self.shadow_pair_dir_ema_even = dir_ema
    else:
        self.shadow_prev_cache_odd = last
        self.shadow_cache_odd = residual_detached
        self.shadow_cache_odd_sub = residual_sub
        self.shadow_pair_ratio_ema_odd = ratio_ema
        self.shadow_pair_dir_ema_odd = dir_ema


def _compose_guided_pair_sub(self, cond_sub, uncond_sub):
    if cond_sub is None or uncond_sub is None:
        return None
    guide_scale = float(getattr(self, 'sample_guide_scale', 5.0))
    return uncond_sub + (cond_sub - uncond_sub) * guide_scale


def _update_guided_pair_history(self):
    guided_sub = _compose_guided_pair_sub(
        self,
        self.shadow_cache_even_sub,
        self.shadow_cache_odd_sub,
    )
    if guided_sub is None:
        self.shadow_pair_valid = False
        return

    prev_guided = self.shadow_guided_pair_sub
    if prev_guided is not None:
        prev_norm = torch.norm(prev_guided, p=2).clamp_(min=1e-12)
        curr_norm = torch.norm(guided_sub, p=2)
        ratio = float((curr_norm / prev_norm).item())
        direction = _tensor_list_cosine(guided_sub, prev_guided)
        self.shadow_guided_pair_ratio_ema = _update_running_ema(
            self.shadow_guided_pair_ratio_ema, ratio,
            self.v2_cfg['qc_ratio_ema_decay'])
        self.shadow_guided_pair_dir_ema = _update_running_ema(
            self.shadow_guided_pair_dir_ema, direction,
            self.v2_cfg['qc_dir_ema_decay'])

    self.shadow_guided_pair_sub = guided_sub.clone()
    self.shadow_pair_valid = True


def _pair_scale_ema(self, is_cond):
    return self.shadow_pair_scale_ema_even if is_cond else self.shadow_pair_scale_ema_odd


def _pair_scale_value(self, phase, is_cond):
    if not self.v2_cfg.get('use_online_scale', True):
        return 1.0
    scale_state = _pair_scale_ema(self, is_cond)
    scale = 1.0 if scale_state is None else float(scale_state[phase])
    return _clamp_float(
        scale,
        self.v2_cfg['scale_clamp_low'],
        self.v2_cfg['scale_clamp_high'],
    )


def _pair_blend_coeff(self, phase, is_cond):
    cfg = self.v2_cfg
    beta, ratio_tol, dir_min = _correction_params(cfg, phase)
    if beta <= 0.0:
        return 0.0, 0.0

    if is_cond:
        branch_ratio_ema = self.shadow_pair_ratio_ema_even
        branch_dir_ema = self.shadow_pair_dir_ema_even
    else:
        branch_ratio_ema = self.shadow_pair_ratio_ema_odd
        branch_dir_ema = self.shadow_pair_dir_ema_odd

    gate = min(
        _ratio_gate(self.shadow_guided_pair_ratio_ema, ratio_tol),
        _dir_gate(self.shadow_guided_pair_dir_ema, dir_min),
        _ratio_gate(branch_ratio_ema, ratio_tol),
        _dir_gate(branch_dir_ema, dir_min),
    )
    beta_eff = beta * gate
    return beta_eff, gate


def _predict_pair_residual(self, phase, is_cond):
    cfg = self.v2_cfg
    last = self.shadow_cache_even if is_cond else self.shadow_cache_odd
    prev = self.shadow_prev_cache_even if is_cond else self.shadow_prev_cache_odd
    if last is None:
        return None

    base_scale = _pair_scale_value(self, phase, is_cond)
    base_pred = [
        tensor.detach().to(torch.float32) * base_scale for tensor in last
    ]
    if not cfg.get('use_second_order', True) or prev is None:
        return base_pred

    beta_eff, blend_gate = _pair_blend_coeff(self, phase, is_cond)
    if beta_eff <= 1e-4 or blend_gate <= 1e-4:
        return base_pred

    return [
        base + (current.detach().to(torch.float32) - previous.detach().to(torch.float32)) * beta_eff
        for base, current, previous in zip(base_pred, last, prev)
    ]


def _update_pair_scale_from_exact(self, residual, phase, is_cond):
    cfg = self.v2_cfg
    if not cfg.get('use_online_scale', True):
        return

    last = self.shadow_cache_even if is_cond else self.shadow_cache_odd
    if last is None:
        return

    scale_state = _pair_scale_ema(self, is_cond)
    fitted = _fit_affine_scale_list(
        residual,
        last,
        default_alpha=scale_state[phase],
    )
    fitted = _clamp_float(
        fitted,
        cfg['scale_clamp_low'],
        cfg['scale_clamp_high'],
    )
    scale_state[phase] = _update_running_ema(
        scale_state[phase], fitted, cfg['scale_ema_decay'])


def _learn_pair_from_exact(self, residual, phase, is_cond):
    _update_pair_scale_from_exact(self, residual, phase, is_cond)


def _predict_pair_output(self, raw_input, phase, is_cond):
    cached = self.cache_even if is_cond else self.cache_odd
    if cached is None:
        return None
    base_output = [
        value + delta.to(dtype=value.dtype)
        for value, delta in zip(raw_input, cached)
    ]

    cfg = self.v2_cfg
    if not cfg.get('enable', True) or not self.shadow_pair_valid:
        return base_output

    pred_residual = _predict_pair_residual(self, phase, is_cond)
    if pred_residual is None:
        return base_output

    return [
        value + delta.to(dtype=value.dtype)
        for value, delta in zip(raw_input, pred_residual)
    ]


def _write_pair_branch_result(self, raw_input, output, is_cond, phase, learn_from_exact):
    residual = [(u - v).detach() for u, v in zip(output, raw_input)]
    if learn_from_exact:
        _learn_pair_from_exact(self, residual, phase, is_cond)
    _store_pair_branch_history(self, residual, is_cond)
    if not is_cond:
        _update_guided_pair_history(self)


def _init_v2_b2_state(cfg):
    return {
        "anchor_cond": None,
        "anchor_uncond": None,
        "last_exact_step": -1,
        "phase_alpha_ema": {
            "early": float(cfg['b2_alpha_prior_early']),
            "mid": float(cfg['b2_alpha_prior_mid']),
            "late": float(cfg['b2_alpha_prior_late']),
        },
        "counts": {
            "anchor_updates": 0,
            "applied_steps": 0,
            "reject_no_anchor": 0,
            "reject_early_exact": 0,
            "reject_max_age": 0,
            "reject_cond_gate": 0,
            "reject_disagree": 0,
            "reject_weak_gate": 0,
        },
    }


def _record_v2_b2_event(self, key):
    state = getattr(self, 'v2_b2_state', None)
    if state is None:
        return
    counts = state["counts"]
    if key in counts:
        counts[key] += 1


def _clear_v2_pending_cond_state(self):
    self.v2_pending_cond_output = None
    self.v2_pending_cond_exact = False


def _set_v2_pending_cond_state(self, output, exact_pair):
    self.v2_pending_cond_output = _clone_tensor_list(output)
    self.v2_pending_cond_exact = bool(exact_pair)


def _predict_v2_b2_uncond(self, phase, step_idx, cond_output, base_output):
    cfg = self.v2_cfg
    if not cfg.get('enable', True) or not cfg.get('b2_enable', True):
        return None

    state = getattr(self, 'v2_b2_state', None)
    if state is None or cond_output is None or base_output is None:
        _record_v2_b2_event(self, 'reject_no_anchor')
        return None

    if state["anchor_cond"] is None or state["anchor_uncond"] is None:
        _record_v2_b2_event(self, 'reject_no_anchor')
        return None

    if cfg.get('b2_exact_early', True) and phase == 'early':
        _record_v2_b2_event(self, 'reject_early_exact')
        return None

    max_pred_age = {
        'early': 0,
        'mid': int(cfg['b2_max_pred_age_mid']),
        'late': int(cfg['b2_max_pred_age_late']),
    }
    pred_age = int(step_idx) - int(state["last_exact_step"])
    if pred_age > max_pred_age[phase]:
        _record_v2_b2_event(self, 'reject_max_age')
        return None

    cond_rel_cap = {
        'early': 0.0,
        'mid': float(cfg['b2_cond_rel_cap_mid']),
        'late': float(cfg['b2_cond_rel_cap_late']),
    }
    cond_rel = _tensor_list_rel_mse(cond_output, state["anchor_cond"])
    cond_rel = 0.0 if cond_rel is None else cond_rel
    if cond_rel > cond_rel_cap[phase]:
        _record_v2_b2_event(self, 'reject_cond_gate')
        return None

    alpha = float(state["phase_alpha_ema"][phase])
    b2_output = [
        anchor_uncond.detach().to(torch.float32)
        + (cond_value.detach().to(torch.float32)
           - anchor_cond.detach().to(torch.float32)) * alpha
        for cond_value, anchor_cond, anchor_uncond in zip(
            cond_output, state["anchor_cond"], state["anchor_uncond"])
    ]

    disagree_cap = {
        'early': 0.0,
        'mid': float(cfg['b2_disagree_rel_cap_mid']),
        'late': float(cfg['b2_disagree_rel_cap_late']),
    }
    disagree_rel = _tensor_list_rel_mse(b2_output, base_output)
    disagree_rel = 0.0 if disagree_rel is None else disagree_rel
    if disagree_rel > disagree_cap[phase]:
        _record_v2_b2_event(self, 'reject_disagree')
        return None

    blend_base = {
        'early': 0.0,
        'mid': float(cfg['b2_blend_mid']),
        'late': float(cfg['b2_blend_late']),
    }
    blend_weight = blend_base[phase]
    if blend_weight <= 1e-6:
        _record_v2_b2_event(self, 'reject_weak_gate')
        return None

    cond_gate = 1.0
    if cond_rel_cap[phase] > 0.0:
        cond_gate = max(0.0, 1.0 - cond_rel / max(cond_rel_cap[phase], 1e-6))
    disagree_gate = 1.0
    if disagree_cap[phase] > 0.0:
        disagree_gate = max(0.0, 1.0 - disagree_rel / max(disagree_cap[phase], 1e-6))
    if max_pred_age[phase] <= 1:
        age_gate = 1.0
    else:
        age_gate = max(
            0.0,
            1.0 - max(float(pred_age) - 1.0, 0.0)
            / max(float(max_pred_age[phase] - 1), 1.0),
        )

    blend_weight *= min(cond_gate, disagree_gate, age_gate)
    if blend_weight <= 1e-4:
        _record_v2_b2_event(self, 'reject_weak_gate')
        return None

    _record_v2_b2_event(self, 'applied_steps')
    return _blend_tensor_lists(base_output, b2_output, blend_weight)


def _update_v2_b2_exact_state(self, phase, step_idx, cond_output, uncond_output, learn):
    cfg = self.v2_cfg
    if not cfg.get('enable', True) or not cfg.get('b2_enable', True):
        return

    state = getattr(self, 'v2_b2_state', None)
    if state is None or cond_output is None or uncond_output is None:
        return

    if learn and state["anchor_cond"] is not None and state["anchor_uncond"] is not None:
        cond_delta = [
            cond_value.detach().to(torch.float32) - anchor_value.detach().to(torch.float32)
            for cond_value, anchor_value in zip(cond_output, state["anchor_cond"])
        ]
        uncond_delta = [
            uncond_value.detach().to(torch.float32) - anchor_value.detach().to(torch.float32)
            for uncond_value, anchor_value in zip(uncond_output, state["anchor_uncond"])
        ]
        alpha = _fit_affine_scale_list(
            uncond_delta,
            cond_delta,
            default_alpha=state["phase_alpha_ema"][phase],
        )
        decay = float(cfg['b2_alpha_ema_decay'])
        state["phase_alpha_ema"][phase] = (
            decay * state["phase_alpha_ema"][phase] + (1.0 - decay) * alpha)

    state["anchor_cond"] = _clone_tensor_list(cond_output)
    state["anchor_uncond"] = _clone_tensor_list(uncond_output)
    state["last_exact_step"] = int(step_idx)
    state["counts"]["anchor_updates"] += 1


def _reset_v2_runtime_state(self):
    self.cnt = 0
    self.accumulated_error_even = 0
    self.should_calc_current_pair = True
    self.k = None
    self.is_even = True
    self.previous_raw_input_even = None
    self.previous_raw_output_even = None
    self.prev_prev_raw_input_even = None
    self.cache_even = None
    self.cache_odd = None
    self.v2_current_phase_threshold = None
    _reset_shadow_pair_state(self)
    self.v2_b2_state = _init_v2_b2_state(self.v2_cfg)
    _clear_v2_pending_cond_state(self)


def _advance_counter(self):
    self.cnt += 1
    if self.cnt >= self.num_steps:
        self.cnt = 0
        _clear_v2_pending_cond_state(self)


def aligncache_forward(self, x, t, context, seq_len, clip_fea=None, y=None):
    if self.model_type == 'i2v':
        assert clip_fea is not None and y is not None

    raw_input = _clone_tensor_list(x)
    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)

    if y is not None:
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

    self.is_even = (self.cnt % 2 == 0)
    pair_idx = self.cnt // 2
    total_pairs = max(self.num_steps // 2, 1)
    phase = _get_phase(pair_idx, total_pairs, self.v2_cfg)
    current_phase_threshold = _phase_skip_threshold(self, phase)
    self.v2_current_phase_threshold = float(current_phase_threshold)
    step_ratio = pair_idx / max(total_pairs - 1, 1)

    if self.is_even:
        if self.cnt < self.ret_steps or self.cnt >= self.cutoff_steps:
            self.should_calc_current_pair = True
            self.accumulated_error_even = 0
        else:
            if (hasattr(self, 'previous_raw_input_even')
                    and hasattr(self, 'previous_raw_output_even')
                    and self.previous_raw_input_even is not None
                    and self.previous_raw_output_even is not None):
                raw_input_change = torch.cat([
                    (u - v).flatten() for u, v in zip(raw_input, self.previous_raw_input_even)
                ]).abs().mean()

                if hasattr(self, 'k') and self.k is not None:
                    output_norm = torch.cat([
                        u.flatten() for u in self.previous_raw_output_even
                    ]).abs().mean()
                    pred_change = self.k * (raw_input_change / output_norm)
                    if not hasattr(self, 'accumulated_error_even'):
                        self.accumulated_error_even = 0
                    self.accumulated_error_even += pred_change
                    if self.accumulated_error_even < current_phase_threshold:
                        self.should_calc_current_pair = False
                    else:
                        self.should_calc_current_pair = True
                        self.accumulated_error_even = 0
                else:
                    self.should_calc_current_pair = True
            else:
                self.should_calc_current_pair = True

        self.previous_raw_input_even = _clone_tensor_list(raw_input)

    if self.is_even and not self.should_calc_current_pair and self.cache_even is not None:
        pred_output = _predict_pair_output(self, raw_input, phase, True)
        if pred_output is None:
            pred_output = [(u + v).detach() for u, v in zip(raw_input, self.cache_even)]
        _set_v2_pending_cond_state(self, pred_output, False)
        if self.v2_cfg.get('enable', True):
            _write_pair_branch_result(
                self, raw_input, pred_output, True, phase, False)
        _advance_counter(self)
        return [u.float() for u in pred_output]

    if (not self.is_even) and (not self.should_calc_current_pair) and self.cache_odd is not None:
        pred_output = _predict_pair_output(self, raw_input, phase, False)
        if pred_output is None:
            pred_output = [(u + v).detach() for u, v in zip(raw_input, self.cache_odd)]
        b2_output = _predict_v2_b2_uncond(
            self,
            phase,
            pair_idx,
            getattr(self, 'v2_pending_cond_output', None),
            pred_output,
        )
        if b2_output is not None:
            pred_output = b2_output
        if self.v2_cfg.get('enable', True):
            _write_pair_branch_result(
                self, raw_input, pred_output, False, phase, False)
        _clear_v2_pending_cond_state(self)
        _advance_counter(self)
        return [u.float() for u in pred_output]

    x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
    grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
    x = [u.flatten(2).transpose(1, 2) for u in x]
    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
    assert seq_lens.max() <= seq_len
    x = torch.cat([
        torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
        for u in x
    ])

    with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).float())
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))
        assert e.dtype == torch.float32 and e0.dtype == torch.float32

    context_lens = None
    context = self.text_embedding(torch.stack([
        torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
        for u in context
    ]))

    if clip_fea is not None:
        context_clip = self.img_emb(clip_fea)
        context = torch.concat([context_clip, context], dim=1)

    kwargs = dict(
        e=e0,
        seq_lens=seq_lens,
        grid_sizes=grid_sizes,
        freqs=self.freqs,
        context=context,
        context_lens=context_lens,
    )

    for block in self.blocks:
        x = block(x, **kwargs)

    x = self.head(x, e)
    output = self.unpatchify(x, grid_sizes)

    if self.is_even:
        if self.previous_raw_output_even is not None:
            output_change = torch.cat([
                (u - v).flatten() for u, v in zip(output, self.previous_raw_output_even)
            ]).abs().mean()

            if self.prev_prev_raw_input_even is not None:
                input_change = torch.cat([
                    (u - v).flatten() for u, v in zip(
                        self.previous_raw_input_even, self.prev_prev_raw_input_even
                    )
                ]).abs().mean()
                self.k = output_change / input_change

        self.prev_prev_raw_input_even = getattr(self, 'previous_raw_input_even', None)
        self.previous_raw_output_even = _clone_tensor_list(output)
        self.cache_even = [u.detach() - v.detach() for u, v in zip(output, raw_input)]
        _set_v2_pending_cond_state(self, output, True)
        if self.v2_cfg.get('enable', True):
            _write_pair_branch_result(
                self, raw_input, output, True, phase, True)
    else:
        self.cache_odd = [u.detach() - v.detach() for u, v in zip(output, raw_input)]
        if self.v2_cfg.get('enable', True):
            _write_pair_branch_result(
                self, raw_input, output, False, phase, True)
            _update_v2_b2_exact_state(
                self,
                phase,
                pair_idx,
                getattr(self, 'v2_pending_cond_output', None),
                output,
                getattr(self, 'v2_pending_cond_exact', False),
            )
        _clear_v2_pending_cond_state(self)

    _advance_counter(self)
    return [u.float() for u in output]


def _log_v2_b2_summary(model):
    state = getattr(model, 'v2_b2_state', None)
    if state is None:
        return
    counts = state["counts"]
    logging.info(
        "AlignCache B2: anchor_updates=%s applied=%s no_anchor=%s early=%s max_age=%s cond=%s disagree=%s weak=%s",
        counts["anchor_updates"],
        counts["applied_steps"],
        counts["reject_no_anchor"],
        counts["reject_early_exact"],
        counts["reject_max_age"],
        counts["reject_cond_gate"],
        counts["reject_disagree"],
        counts["reject_weak_gate"],
    )
    logging.info(
        "AlignCache alpha: early=%.4f mid=%.4f late=%.4f",
        state["phase_alpha_ema"]["early"],
        state["phase_alpha_ema"]["mid"],
        state["phase_alpha_ema"]["late"],
    )


def _setup_aligncache(model, args):
    model.forward = aligncache_forward
    model.num_steps = args.sample_steps * 2
    model.thresh = args.thresh
    model.ret_steps = int(args.ret_steps) * 2
    model.cutoff_steps = args.sample_steps * 2 - 2
    model.sample_guide_scale = float(args.sample_guide_scale)
    model.v2_cfg = _build_v2_cfg(args)
    _reset_v2_runtime_state(model)
    logging.info(
        "AlignCache: preserve original skip logic, add v16-style online pair correction + skipped-uncond B2  "
        "thresh=%.4f ret_steps=%d beta_mid=%.2f beta_late=%.2f online_scale=%s b2=%s",
        args.thresh,
        args.ret_steps,
        model.v2_cfg['beta_mid'],
        model.v2_cfg['beta_late'],
        model.v2_cfg['use_online_scale'],
        model.v2_cfg['b2_enable'],
    )


def _validate_v2_args(args):
    assert args.v2_subsample_stride >= 1, "v2_subsample_stride must be >= 1"
    assert 0.0 <= args.v2_ratio_ema_decay <= 1.0, "v2_ratio_ema_decay must be in [0, 1]"
    assert 0.0 <= args.v2_dir_ema_decay <= 1.0, "v2_dir_ema_decay must be in [0, 1]"
    assert args.v2_beta_mid >= 0.0, "v2_beta_mid must be non-negative"
    assert args.v2_beta_late >= 0.0, "v2_beta_late must be non-negative"
    assert args.v2_ratio_tol_mid >= 0.0, "v2_ratio_tol_mid must be non-negative"
    assert args.v2_ratio_tol_late >= 0.0, "v2_ratio_tol_late must be non-negative"
    assert -1.0 <= args.v2_dir_min_mid <= 1.0, "v2_dir_min_mid must be in [-1, 1]"
    assert -1.0 <= args.v2_dir_min_late <= 1.0, "v2_dir_min_late must be in [-1, 1]"
    assert 0.0 <= args.v2_scale_ema_decay <= 1.0, "v2_scale_ema_decay must be in [0, 1]"
    assert args.v2_scale_clamp_low > 0.0, "v2_scale_clamp_low must be positive"
    assert args.v2_scale_clamp_high >= args.v2_scale_clamp_low, (
        "v2_scale_clamp_high must be >= v2_scale_clamp_low")
    assert args.v2_b2_max_pred_age_mid >= 0, "v2_b2_max_pred_age_mid must be >= 0"
    assert args.v2_b2_max_pred_age_late >= 0, "v2_b2_max_pred_age_late must be >= 0"


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate image/video from text or image using Wan with AlignCache"
    )
    parser.add_argument(
        "--task",
        type=lambda value: parse_task_name_arg(value, allowed_tasks=WAN_CONFIGS.keys()),
        default="t2v-14B",
        choices=list(WAN_CONFIGS.keys()),
        help="The task to run.",
    )
    parser.add_argument("--size", type=str, default="1280*720", choices=list(SIZE_CONFIGS.keys()), help="Output size.")
    parser.add_argument("--frame_num", type=int, default=None, help="How many frames to sample. 4n+1 for video.")
    parser.add_argument("--ckpt_dir", type=str, default=None, help="Path to checkpoint directory.")
    parser.add_argument("--offload_model", type=str2bool, default=None, help="Whether to offload model to CPU during generation.")
    parser.add_argument("--ulysses_size", type=int, default=1, help="Ulysses degree.")
    parser.add_argument("--ring_size", type=int, default=1, help="Ring degree.")
    parser.add_argument("--t5_fsdp", action="store_true", default=False, help="Enable T5 FSDP.")
    parser.add_argument("--t5_cpu", action="store_true", default=False, help="Run T5 on CPU.")
    parser.add_argument("--dit_fsdp", action="store_true", default=False, help="Enable DiT FSDP.")
    parser.add_argument("--save_file", type=str, default=None, help="Output path.")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt.")
    parser.add_argument("--use_prompt_extend", action="store_true", default=False, help="Use prompt extension.")
    parser.add_argument("--prompt_extend_method", type=str, default="local_qwen", choices=["dashscope", "local_qwen"], help="Prompt extension method.")
    parser.add_argument("--prompt_extend_model", type=str, default=None, help="Prompt extension model.")
    parser.add_argument("--prompt_extend_target_lang", type=str, default="ch", choices=["ch", "en"], help="Prompt extension target language.")
    parser.add_argument("--base_seed", type=int, default=-1, help="Seed.")
    parser.add_argument("--image", type=str, default=None, help="Input image for i2v.")
    parser.add_argument("--sample_solver", type=str, default='unipc', choices=['unipc', 'dpm++'], help="Sampling solver.")
    parser.add_argument("--sample_steps", type=int, default=None, help="Sampling steps.")
    parser.add_argument("--sample_shift", type=float, default=None, help="Sampling shift.")
    parser.add_argument("--sample_guide_scale", type=float, default=5.0, help="CFG scale.")
    parser.add_argument("--thresh", type=float, default=0.05, help="Threshold for the preserved skip schedule.")
    parser.add_argument("--ret_steps", default=10, type=int, help="Warm-up steps for the preserved skip schedule.")
    parser.add_argument("--alpha", default=0.0, type=float)
    parser.add_argument("--beta", default=1.0, type=float)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=946)
    parser.add_argument("--out_dir", type=str, default="./output")

    parser.add_argument("--v2_enable", dest="v2_enable", type=str2bool, default=True)
    parser.add_argument("--v2_aggressiveness", dest="v2_aggressiveness", type=float, default=0.0)
    parser.add_argument("--v2_use_second_order", dest="v2_use_second_order", type=str2bool, default=True)
    parser.add_argument("--v2_use_online_scale", dest="v2_use_online_scale", type=str2bool, default=True)
    parser.add_argument("--v2_subsample_stride", dest="v2_subsample_stride", type=int, default=64)
    parser.add_argument("--v2_ratio_ema_decay", dest="v2_ratio_ema_decay", type=float, default=0.8)
    parser.add_argument("--v2_dir_ema_decay", dest="v2_dir_ema_decay", type=float, default=0.8)
    parser.add_argument("--v2_beta_mid", dest="v2_beta_mid", type=float, default=0.35)
    parser.add_argument("--v2_beta_late", dest="v2_beta_late", type=float, default=0.15)
    parser.add_argument("--v2_ratio_tol_mid", dest="v2_ratio_tol_mid", type=float, default=0.30)
    parser.add_argument("--v2_ratio_tol_late", dest="v2_ratio_tol_late", type=float, default=0.18)
    parser.add_argument("--v2_dir_min_mid", dest="v2_dir_min_mid", type=float, default=0.80)
    parser.add_argument("--v2_dir_min_late", dest="v2_dir_min_late", type=float, default=0.88)
    parser.add_argument("--v2_scale_ema_decay", dest="v2_scale_ema_decay", type=float, default=0.85)
    parser.add_argument("--v2_scale_clamp_low", dest="v2_scale_clamp_low", type=float, default=0.85)
    parser.add_argument("--v2_scale_clamp_high", dest="v2_scale_clamp_high", type=float, default=1.18)
    parser.add_argument("--v2_b2_enable", dest="v2_b2_enable", type=str2bool, default=True)
    parser.add_argument("--v2_b2_exact_early", dest="v2_b2_exact_early", type=str2bool, default=True)
    parser.add_argument("--v2_b2_max_pred_age_mid", dest="v2_b2_max_pred_age_mid", type=int, default=1)
    parser.add_argument("--v2_b2_max_pred_age_late", dest="v2_b2_max_pred_age_late", type=int, default=2)
    parser.add_argument("--v2_b2_cond_rel_cap_mid", dest="v2_b2_cond_rel_cap_mid", type=float, default=0.0050)
    parser.add_argument("--v2_b2_cond_rel_cap_late", dest="v2_b2_cond_rel_cap_late", type=float, default=0.0200)
    parser.add_argument("--v2_b2_alpha_prior_early", dest="v2_b2_alpha_prior_early", type=float, default=1.279282)
    parser.add_argument("--v2_b2_alpha_prior_mid", dest="v2_b2_alpha_prior_mid", type=float, default=1.018750)
    parser.add_argument("--v2_b2_alpha_prior_late", dest="v2_b2_alpha_prior_late", type=float, default=0.972800)
    parser.add_argument("--v2_b2_alpha_ema_decay", dest="v2_b2_alpha_ema_decay", type=float, default=0.85)
    parser.add_argument("--v2_b2_blend_mid", dest="v2_b2_blend_mid", type=float, default=0.35)
    parser.add_argument("--v2_b2_blend_late", dest="v2_b2_blend_late", type=float, default=0.20)
    parser.add_argument("--v2_b2_disagree_rel_cap_mid", dest="v2_b2_disagree_rel_cap_mid", type=float, default=0.0200)
    parser.add_argument("--v2_b2_disagree_rel_cap_late", dest="v2_b2_disagree_rel_cap_late", type=float, default=0.0600)

    args = parser.parse_args()
    args.ckpt_dir = normalize_ckpt_dir_for_task(args.task, args.ckpt_dir, size_text=args.size)
    _validate_args(args)
    _validate_v2_args(args)
    return args


def _build_prompt_expander(args, rank):
    if not args.use_prompt_extend:
        return None
    if args.prompt_extend_method == "dashscope":
        return DashScopePromptExpander(
            model_name=args.prompt_extend_model,
            is_vl="i2v" in args.task,
        )
    if args.prompt_extend_method == "local_qwen":
        return QwenPromptExpander(
            model_name=args.prompt_extend_model,
            is_vl="i2v" in args.task,
            device=rank,
        )
    raise NotImplementedError(f"Unsupport prompt_extend_method: {args.prompt_extend_method}")


def generate(args):
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        logging.info("offload_model is not specified, set to %s.", args.offload_model)

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size,
        )
    else:
        assert not (args.t5_fsdp or args.dit_fsdp), "t5_fsdp and dit_fsdp are not supported in non-distributed environments."
        assert not (args.ulysses_size > 1 or args.ring_size > 1), "context parallel are not supported in non-distributed environments."

    if args.ulysses_size > 1 or args.ring_size > 1:
        assert args.ulysses_size * args.ring_size == world_size, "ulysses_size * ring_size must equal world size."
        from xfuser.core.distributed import initialize_model_parallel, init_distributed_environment
        init_distributed_environment(rank=dist.get_rank(), world_size=dist.get_world_size())
        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=args.ring_size,
            ulysses_degree=args.ulysses_size,
        )

    prompt_expander = _build_prompt_expander(args, rank)
    cfg = WAN_CONFIGS[args.task]
    if args.ulysses_size > 1:
        assert cfg.num_heads % args.ulysses_size == 0, "num_heads must be divisible by ulysses_size."

    logging.info("Generation job args: %s", args)
    logging.info("Generation model config: %s", cfg)

    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]

    if "t2v" in args.task or "t2i" in args.task:
        if args.prompt is None:
            args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]
        logging.info("Input prompt: %s", args.prompt)
        if args.use_prompt_extend:
            assert prompt_expander is not None
            logging.info("Extending prompt ...")
            if rank == 0:
                prompt_output = prompt_expander(
                    args.prompt,
                    tar_lang=args.prompt_extend_target_lang,
                    seed=args.base_seed,
                )
                if prompt_output.status is False:
                    logging.info("Extending prompt failed: %s", prompt_output.message)
                    logging.info("Falling back to original prompt.")
                    input_prompt = args.prompt
                else:
                    input_prompt = prompt_output.prompt
                input_prompt = [input_prompt]
            else:
                input_prompt = [None]
            if dist.is_initialized():
                dist.broadcast_object_list(input_prompt, src=0)
            args.prompt = input_prompt[0]
            logging.info("Extended prompt: %s", args.prompt)

        logging.info("Creating WanT2V pipeline.")
        pipeline = wan.WanT2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
            t5_cpu=args.t5_cpu,
        )
        pipeline.__class__.generate = t2v_generate
        pipeline.__class__.cost_time = 0
        _setup_aligncache(pipeline.model.__class__, args)
        pipeline.cost_time = 0

        logging.info("Generating %s ...", 'image' if 't2i' in args.task else 'video')
        video = pipeline.generate(
            args.prompt,
            size=SIZE_CONFIGS[args.size],
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model,
        )
        _log_v2_b2_summary(pipeline.model.__class__)

        if rank == 0:
            if args.save_file is None:
                assert args.prompt is not None
                formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                formatted_prompt = args.prompt.replace(" ", "_").replace("/", "_")[:50]
                suffix = '.png' if 't2i' in args.task else '.mp4'
                args.save_file = f"{args.task}_aligncache_step{args.sample_steps}_{formatted_prompt}_{formatted_time}{suffix}"
            if 't2i' in args.task:
                logging.info("Saving generated image to %s", args.save_file)
                cache_image(
                    tensor=video.squeeze(1)[None],
                    save_file=args.save_file,
                    nrow=1,
                    normalize=True,
                    value_range=(-1, 1),
                )
            else:
                logging.info("Saving generated video to %s", args.save_file)
                cache_video(
                    tensor=video[None],
                    save_file=args.save_file,
                    fps=cfg.sample_fps,
                    nrow=1,
                    normalize=True,
                    value_range=(-1, 1),
                )
        logging.info("Finished.")
        avg_cost = float(getattr(pipeline, 'cost_time', 0.0))
    else:
        if args.prompt is None:
            args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]
        if args.image is None:
            args.image = EXAMPLE_PROMPT[args.task]["image"]
        logging.info("Input prompt: %s", args.prompt)
        logging.info("Input image: %s", args.image)
        img = Image.open(args.image).convert("RGB")

        if args.use_prompt_extend:
            assert prompt_expander is not None
            logging.info("Extending prompt ...")
            if rank == 0:
                prompt_output = prompt_expander(
                    args.prompt,
                    tar_lang=args.prompt_extend_target_lang,
                    image=img,
                    seed=args.base_seed,
                )
                if prompt_output.status is False:
                    logging.info("Extending prompt failed: %s", prompt_output.message)
                    logging.info("Falling back to original prompt.")
                    input_prompt = args.prompt
                else:
                    input_prompt = prompt_output.prompt
                input_prompt = [input_prompt]
            else:
                input_prompt = [None]
            if dist.is_initialized():
                dist.broadcast_object_list(input_prompt, src=0)
            args.prompt = input_prompt[0]
            logging.info("Extended prompt: %s", args.prompt)

        logging.info("Creating WanI2V pipeline.")
        pipeline = wan.WanI2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
            t5_cpu=args.t5_cpu,
        )
        pipeline.__class__.generate = i2v_generate
        pipeline.__class__.cost_time = 0
        _setup_aligncache(pipeline.model.__class__, args)
        pipeline.cost_time = 0

        logging.info("Generating video ...")
        video = pipeline.generate(
            args.prompt,
            img,
            max_area=MAX_AREA_CONFIGS[args.size],
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model,
        )
        _log_v2_b2_summary(pipeline.model.__class__)

        if rank == 0:
            if args.save_file is None:
                assert args.prompt is not None
                formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                formatted_prompt = args.prompt.replace(" ", "_").replace("/", "_")[:50]
                args.save_file = f"{args.task}_aligncache_step{args.sample_steps}_{formatted_prompt}_{formatted_time}.mp4"
            logging.info("Saving generated video to %s", args.save_file)
            cache_video(
                tensor=video[None],
                save_file=args.save_file,
                fps=cfg.sample_fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1),
            )
        logging.info("Finished.")
        avg_cost = float(getattr(pipeline, 'cost_time', 0.0))

    print(
        f"GPU_Device:{torch.cuda.get_device_name(0)}, number_prompt: 1, avg_cost_time: {avg_cost}"
    )


if __name__ == "__main__":
    generate(_parse_args())
