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
import argparse
from pathlib import Path
from typing import Iterable


KNOWN_WAN_TASKS = (
    "t2v-14B",
    "t2v-1.3B",
    "i2v-14B",
    "t2i-14B",
    "flf2v-14B",
    "vace-1.3B",
    "vace-14B",
)

DEFAULT_MODEL_WEIGHTS_DIR = "./model_weights"

LEGACY_T2V_13B_DEFAULT_CKPT_DIRS = frozenset(
    {
        "./model_weights/Wan2.1-T2V-1.3B",
        "model_weights/Wan2.1-T2V-1.3B",
        "./Wan2.1-T2V-1.3B",
        "Wan2.1-T2V-1.3B",
    }
)


def _canonical_task_lookup(allowed_tasks: Iterable[str] | None = None) -> dict[str, str]:
    tasks = tuple(str(task) for task in (allowed_tasks or KNOWN_WAN_TASKS))
    lookup: dict[str, str] = {}
    for task in tasks:
        lookup.setdefault(task.lower(), task)
    return lookup


def canonicalize_task_name(
    task_name: str | None,
    *,
    allowed_tasks: Iterable[str] | None = None,
) -> str | None:
    if task_name is None:
        return None

    task_text = str(task_name).strip()
    if not task_text:
        raise ValueError("Task name cannot be empty.")

    lookup = _canonical_task_lookup(allowed_tasks=allowed_tasks)
    canonical = lookup.get(task_text.lower())
    if canonical is not None:
        return canonical

    supported = ", ".join(lookup.values())
    raise ValueError(f"Unsupported task {task_name!r}. Supported tasks: {supported}")


def parse_task_name_arg(
    task_name: str,
    *,
    allowed_tasks: Iterable[str] | None = None,
) -> str:
    try:
        canonical = canonicalize_task_name(task_name, allowed_tasks=allowed_tasks)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    assert canonical is not None
    return canonical


def checkpoint_dir_name_for_task(task_name: str, *, size_text: str | None = None) -> str:
    canonical = canonicalize_task_name(task_name)
    assert canonical is not None

    if canonical == "t2v-1.3B":
        return "Wan2.1-T2V-1.3B"
    if canonical in {"t2v-14B", "t2i-14B"}:
        return "Wan2.1-T2V-14B"
    if canonical == "i2v-14B":
        if size_text in {"1280*720", "720*1280"}:
            return "Wan2.1-I2V-14B-720P"
        return "Wan2.1-I2V-14B-480P"
    if canonical == "flf2v-14B":
        return "Wan2.1-FLF2V-14B-720P"
    if canonical == "vace-1.3B":
        return "Wan2.1-VACE-1.3B"
    if canonical == "vace-14B":
        return "Wan2.1-VACE-14B"

    raise ValueError(f"Unsupported task {task_name!r}")


def default_checkpoint_dir_for_task(
    task_name: str,
    *,
    size_text: str | None = None,
    model_weights_root: str = DEFAULT_MODEL_WEIGHTS_DIR,
) -> str:
    checkpoint_dir_name = checkpoint_dir_name_for_task(task_name, size_text=size_text)
    return str(Path(model_weights_root) / checkpoint_dir_name)


def normalize_ckpt_dir_for_task(
    task_name: str,
    ckpt_dir: str | None,
    *,
    size_text: str | None = None,
    model_weights_root: str = DEFAULT_MODEL_WEIGHTS_DIR,
) -> str:
    canonical = canonicalize_task_name(task_name)
    assert canonical is not None

    if ckpt_dir is None or not str(ckpt_dir).strip():
        return default_checkpoint_dir_for_task(
            canonical,
            size_text=size_text,
            model_weights_root=model_weights_root,
        )

    ckpt_text = str(ckpt_dir).strip()
    if canonical != "t2v-1.3B" and ckpt_text in LEGACY_T2V_13B_DEFAULT_CKPT_DIRS:
        return default_checkpoint_dir_for_task(
            canonical,
            size_text=size_text,
            model_weights_root=model_weights_root,
        )

    return ckpt_text

from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.modules.model import sinusoidal_embedding_1d
from wan.utils.fm_solvers import (FlowDPMSolverMultistepScheduler, get_sampling_sigmas, retrieve_timesteps)
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from wan.utils.utils import cache_image, cache_video, str2bool

warnings.filterwarnings('ignore')

FB_DEFAULT_PHASE_SCALES = {
    "early": 1.00,
    "mid": 1.00,
    "late": 1.00,
}

FB_DEFAULT_AGE_CAP = {
    "early": 5,
    "mid": 4,
    "late": 3,
}

FB_DEFAULT_SCALE_INIT = {
    "early": 1.00,
    "mid": 1.00,
    "late": 1.00,
}

FB_DEFAULT_SCALE_LOW = {
    "early": 0.95,
    "mid": 0.90,
    "late": 0.85,
}

FB_DEFAULT_SCALE_HIGH = {
    "early": 1.03,
    "mid": 1.06,
    "late": 1.00,
}

FB_DEFAULTS = {
    "ratio_ema_decay": 0.90,
    "ratio_clip_low": 0.85,
    "ratio_clip_high": 1.25,
    "score_eps": 1e-6,
}

FB_REASON_KEYS = {
    "warmup_tail": "warmup_tail_pairs",
    "bootstrap": "bootstrap_pairs",
    "max_age": "max_age_pairs",
    "budget_exact": "budget_exact_pairs",
    "budget_skip": "budget_skip_pairs",
    "cache_miss_exact": "cache_miss_exact_pairs",
}

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

PAIR_DEFAULTS = dict(
    enable=True,
    early_end_ratio=0.15,
    late_start_ratio=0.86,
    stride=64,
    ratio_ema=0.8,
    dir_ema=0.8,
    beta_mid=0.35,
    beta_late=0.15,
    ratio_tol_mid=0.30,
    ratio_tol_late=0.18,
    dir_min_mid=0.80,
    dir_min_late=0.88,
    second_order=True,
    online_scale=True,
    scale_ema=0.85,
    scale_low=0.85,
    scale_high=1.18,
    anchor_enable=True,
    anchor_exact_early=True,
    anchor_age_mid=1,
    anchor_age_late=2,
    anchor_cond_cap_mid=0.0050,
    anchor_cond_cap_late=0.0200,
    anchor_alpha_early=1.279282,
    anchor_alpha_mid=1.018750,
    anchor_alpha_late=0.972800,
    anchor_alpha_ema=0.85,
    anchor_blend_mid=0.35,
    anchor_blend_late=0.20,
    anchor_diff_cap_mid=0.0200,
    anchor_diff_cap_late=0.0600,
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
    return float(getattr(self, 'base_thresh', 0.0))


def _apply_pair_aggr(cfg, aggressiveness):
    cfg = dict(cfg)
    aggressiveness = float(aggressiveness)
    beta_scale = _clamp_float(1.0 + 0.60 * aggressiveness, 0.50, 1.35)
    blend_scale = _clamp_float(1.0 + 0.50 * aggressiveness, 0.50, 1.40)
    clamp_delta = 0.08 * aggressiveness

    cfg['beta_mid'] *= beta_scale
    cfg['beta_late'] *= beta_scale
    cfg['anchor_blend_mid'] *= blend_scale
    cfg['anchor_blend_late'] *= blend_scale
    cfg['scale_low'] = max(0.60, cfg['scale_low'] - clamp_delta)
    cfg['scale_high'] = max(
        cfg['scale_low'],
        cfg['scale_high'] + clamp_delta,
    )
    return cfg


def _build_pair_cfg(args):
    cfg = dict(PAIR_DEFAULTS)
    cfg['enable'] = args.pair_enable
    cfg['second_order'] = args.pair_second_order
    cfg['online_scale'] = args.pair_online_scale
    cfg['stride'] = args.pair_stride
    cfg['ratio_ema'] = args.pair_ratio_ema
    cfg['dir_ema'] = args.pair_dir_ema
    cfg['beta_mid'] = args.pair_beta_mid
    cfg['beta_late'] = args.pair_beta_late
    cfg['ratio_tol_mid'] = args.pair_ratio_tol_mid
    cfg['ratio_tol_late'] = args.pair_ratio_tol_late
    cfg['dir_min_mid'] = args.pair_dir_min_mid
    cfg['dir_min_late'] = args.pair_dir_min_late
    cfg['scale_ema'] = args.pair_scale_ema
    cfg['scale_low'] = args.pair_scale_low
    cfg['scale_high'] = args.pair_scale_high
    cfg['anchor_enable'] = args.anchor_enable
    cfg['anchor_exact_early'] = args.anchor_exact_early
    cfg['anchor_age_mid'] = args.anchor_age_mid
    cfg['anchor_age_late'] = args.anchor_age_late
    cfg['anchor_cond_cap_mid'] = args.anchor_cond_cap_mid
    cfg['anchor_cond_cap_late'] = args.anchor_cond_cap_late
    cfg['anchor_alpha_early'] = args.anchor_alpha_early
    cfg['anchor_alpha_mid'] = args.anchor_alpha_mid
    cfg['anchor_alpha_late'] = args.anchor_alpha_late
    cfg['anchor_alpha_ema'] = args.anchor_alpha_ema
    cfg['anchor_blend_mid'] = args.anchor_blend_mid
    cfg['anchor_blend_late'] = args.anchor_blend_late
    cfg['anchor_diff_cap_mid'] = args.anchor_diff_cap_mid
    cfg['anchor_diff_cap_late'] = args.anchor_diff_cap_late
    return _apply_pair_aggr(cfg, args.pair_aggr)


def _reset_shadow_pair_state(self):
    self.cond_last = None
    self.uncond_last = None
    self.cond_prev = None
    self.uncond_prev = None
    self.cond_last_sub = None
    self.uncond_last_sub = None
    self.cond_ratio_ema = None
    self.uncond_ratio_ema = None
    self.cond_dir_ema = None
    self.uncond_dir_ema = None
    self.guided_sub = None
    self.guided_ratio_ema = None
    self.guided_dir_ema = None
    self.guided_ready = False
    self.cond_scale_ema = _init_phase_ema(1.0)
    self.uncond_scale_ema = _init_phase_ema(1.0)


def _store_pair_branch_history(self, residual, is_cond):
    residual_detached = _clone_tensor_list(residual)
    residual_sub = _tensor_list_subsample(
        residual_detached, self.pair_cfg['stride'])

    if is_cond:
        last = self.cond_last
        last_sub = self.cond_last_sub
        ratio_ema = self.cond_ratio_ema
        dir_ema = self.cond_dir_ema
    else:
        last = self.uncond_last
        last_sub = self.uncond_last_sub
        ratio_ema = self.uncond_ratio_ema
        dir_ema = self.uncond_dir_ema

    if last_sub is not None and residual_sub is not None:
        prev_norm = torch.norm(last_sub, p=2).clamp_(min=1e-12)
        curr_norm = torch.norm(residual_sub, p=2)
        ratio = float((curr_norm / prev_norm).item())
        direction = _tensor_list_cosine(residual_sub, last_sub)
        ratio_ema = _update_running_ema(
            ratio_ema, ratio, self.pair_cfg['ratio_ema'])
        dir_ema = _update_running_ema(
            dir_ema, direction, self.pair_cfg['dir_ema'])

    if is_cond:
        self.cond_prev = last
        self.cond_last = residual_detached
        self.cond_last_sub = residual_sub
        self.cond_ratio_ema = ratio_ema
        self.cond_dir_ema = dir_ema
    else:
        self.uncond_prev = last
        self.uncond_last = residual_detached
        self.uncond_last_sub = residual_sub
        self.uncond_ratio_ema = ratio_ema
        self.uncond_dir_ema = dir_ema


def _compose_guided_pair_sub(self, cond_sub, uncond_sub):
    if cond_sub is None or uncond_sub is None:
        return None
    guide_scale = float(getattr(self, 'sample_guide_scale', 5.0))
    return uncond_sub + (cond_sub - uncond_sub) * guide_scale


def _update_guided_pair_history(self):
    guided_sub = _compose_guided_pair_sub(
        self,
        self.cond_last_sub,
        self.uncond_last_sub,
    )
    if guided_sub is None:
        self.guided_ready = False
        return

    prev_guided = self.guided_sub
    if prev_guided is not None:
        prev_norm = torch.norm(prev_guided, p=2).clamp_(min=1e-12)
        curr_norm = torch.norm(guided_sub, p=2)
        ratio = float((curr_norm / prev_norm).item())
        direction = _tensor_list_cosine(guided_sub, prev_guided)
        self.guided_ratio_ema = _update_running_ema(
            self.guided_ratio_ema, ratio,
            self.pair_cfg['ratio_ema'])
        self.guided_dir_ema = _update_running_ema(
            self.guided_dir_ema, direction,
            self.pair_cfg['dir_ema'])

    self.guided_sub = guided_sub.clone()
    self.guided_ready = True


def _pair_scale_ema(self, is_cond):
    return self.cond_scale_ema if is_cond else self.uncond_scale_ema


def _pair_scale_value(self, phase, is_cond):
    if not self.pair_cfg.get('online_scale', True):
        return 1.0
    scale_state = _pair_scale_ema(self, is_cond)
    scale = 1.0 if scale_state is None else float(scale_state[phase])
    return _clamp_float(
        scale,
        self.pair_cfg['scale_low'],
        self.pair_cfg['scale_high'],
    )


def _pair_blend_coeff(self, phase, is_cond):
    cfg = self.pair_cfg
    beta, ratio_tol, dir_min = _correction_params(cfg, phase)
    if beta <= 0.0:
        return 0.0, 0.0

    if is_cond:
        branch_ratio_ema = self.cond_ratio_ema
        branch_dir_ema = self.cond_dir_ema
    else:
        branch_ratio_ema = self.uncond_ratio_ema
        branch_dir_ema = self.uncond_dir_ema

    gate = min(
        _ratio_gate(self.guided_ratio_ema, ratio_tol),
        _dir_gate(self.guided_dir_ema, dir_min),
        _ratio_gate(branch_ratio_ema, ratio_tol),
        _dir_gate(branch_dir_ema, dir_min),
    )
    beta_eff = beta * gate
    return beta_eff, gate


def _predict_pair_residual(self, phase, is_cond):
    cfg = self.pair_cfg
    last = self.cond_last if is_cond else self.uncond_last
    prev = self.cond_prev if is_cond else self.uncond_prev
    if last is None:
        return None

    base_scale = _pair_scale_value(self, phase, is_cond)
    base_pred = [
        tensor.detach().to(torch.float32) * base_scale for tensor in last
    ]
    if not cfg.get('second_order', True) or prev is None:
        return base_pred

    beta_eff, blend_gate = _pair_blend_coeff(self, phase, is_cond)
    if beta_eff <= 1e-4 or blend_gate <= 1e-4:
        return base_pred

    return [
        base + (current.detach().to(torch.float32) - previous.detach().to(torch.float32)) * beta_eff
        for base, current, previous in zip(base_pred, last, prev)
    ]


def _update_pair_scale_from_exact(self, residual, phase, is_cond):
    cfg = self.pair_cfg
    if not cfg.get('online_scale', True):
        return

    last = self.cond_last if is_cond else self.uncond_last
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
        cfg['scale_low'],
        cfg['scale_high'],
    )
    scale_state[phase] = _update_running_ema(
        scale_state[phase], fitted, cfg['scale_ema'])


def _learn_pair_from_exact(self, residual, phase, is_cond):
    _update_pair_scale_from_exact(self, residual, phase, is_cond)


def _predict_pair_output(self, raw_input, phase, is_cond):
    cached = self.cond_cache if is_cond else self.uncond_cache
    if cached is None:
        return None
    base_output = [
        value + delta.to(dtype=value.dtype)
        for value, delta in zip(raw_input, cached)
    ]

    cfg = self.pair_cfg
    if not cfg.get('enable', True) or not self.guided_ready:
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


def _init_anchor_state(cfg):
    return {
        "anchor_cond": None,
        "anchor_uncond": None,
        "last_exact_pair": -1,
        "alpha_ema": {
            "early": float(cfg['anchor_alpha_early']),
            "mid": float(cfg['anchor_alpha_mid']),
            "late": float(cfg['anchor_alpha_late']),
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


def _record_anchor_event(self, key, *, enabled=True):
    if not enabled:
        return
    state = getattr(self, 'anchor_state', None)
    if state is None:
        return
    counts = state["counts"]
    if key in counts:
        counts[key] += 1


def _clear_pending_cond(self):
    self.pending_cond = None
    self.pending_cond_exact = False


def _set_pending_cond(self, output, exact_pair):
    self.pending_cond = _clone_tensor_list(output)
    self.pending_cond_exact = bool(exact_pair)


def _predict_anchor_uncond(self, phase, step_idx, cond_output, base_output, *, record_event=True):
    cfg = self.pair_cfg
    if not cfg.get('enable', True) or not cfg.get('anchor_enable', True):
        return None

    state = getattr(self, 'anchor_state', None)
    if state is None or cond_output is None or base_output is None:
        _record_anchor_event(self, 'reject_no_anchor', enabled=record_event)
        return None

    if state["anchor_cond"] is None or state["anchor_uncond"] is None:
        _record_anchor_event(self, 'reject_no_anchor', enabled=record_event)
        return None

    if cfg.get('anchor_exact_early', True) and phase == 'early':
        _record_anchor_event(self, 'reject_early_exact', enabled=record_event)
        return None

    max_pred_age = {
        'early': 0,
        'mid': int(cfg['anchor_age_mid']),
        'late': int(cfg['anchor_age_late']),
    }
    anchor_age = int(step_idx) - int(state["last_exact_pair"])
    if anchor_age > max_pred_age[phase]:
        _record_anchor_event(self, 'reject_max_age', enabled=record_event)
        return None

    cond_rel_cap = {
        'early': 0.0,
        'mid': float(cfg['anchor_cond_cap_mid']),
        'late': float(cfg['anchor_cond_cap_late']),
    }
    cond_rel = _tensor_list_rel_mse(cond_output, state["anchor_cond"])
    cond_rel = 0.0 if cond_rel is None else cond_rel
    if cond_rel > cond_rel_cap[phase]:
        _record_anchor_event(self, 'reject_cond_gate', enabled=record_event)
        return None

    alpha = float(state["alpha_ema"][phase])
    anchor_output = [
        anchor_uncond.detach().to(torch.float32)
        + (cond_value.detach().to(torch.float32)
           - anchor_cond.detach().to(torch.float32)) * alpha
        for cond_value, anchor_cond, anchor_uncond in zip(
            cond_output, state["anchor_cond"], state["anchor_uncond"])
    ]

    disagree_cap = {
        'early': 0.0,
        'mid': float(cfg['anchor_diff_cap_mid']),
        'late': float(cfg['anchor_diff_cap_late']),
    }
    disagree_rel = _tensor_list_rel_mse(anchor_output, base_output)
    disagree_rel = 0.0 if disagree_rel is None else disagree_rel
    if disagree_rel > disagree_cap[phase]:
        _record_anchor_event(self, 'reject_disagree', enabled=record_event)
        return None

    blend_base = {
        'early': 0.0,
        'mid': float(cfg['anchor_blend_mid']),
        'late': float(cfg['anchor_blend_late']),
    }
    blend_weight = blend_base[phase]
    if blend_weight <= 1e-6:
        _record_anchor_event(self, 'reject_weak_gate', enabled=record_event)
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
            1.0 - max(float(anchor_age) - 1.0, 0.0)
            / max(float(max_pred_age[phase] - 1), 1.0),
        )

    blend_weight *= min(cond_gate, disagree_gate, age_gate)
    if blend_weight <= 1e-4:
        _record_anchor_event(self, 'reject_weak_gate', enabled=record_event)
        return None

    _record_anchor_event(self, 'applied_steps', enabled=record_event)
    return _blend_tensor_lists(base_output, anchor_output, blend_weight)


def _update_anchor_state(self, phase, step_idx, cond_output, uncond_output, learn):
    cfg = self.pair_cfg
    if not cfg.get('enable', True) or not cfg.get('anchor_enable', True):
        return

    state = getattr(self, 'anchor_state', None)
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
            default_alpha=state["alpha_ema"][phase],
        )
        decay = float(cfg['anchor_alpha_ema'])
        state["alpha_ema"][phase] = (
            decay * state["alpha_ema"][phase] + (1.0 - decay) * alpha)

    state["anchor_cond"] = _clone_tensor_list(cond_output)
    state["anchor_uncond"] = _clone_tensor_list(uncond_output)
    state["last_exact_pair"] = int(step_idx)
    state["counts"]["anchor_updates"] += 1


def _tensor_list_mean_abs(values):
    if values is None:
        return None
    flat = torch.cat([
        value.detach().to(torch.float32).flatten() for value in values
    ])
    if flat.numel() == 0:
        return 0.0
    return float(flat.abs().mean().item())


def _resolve_phase_mapping(defaults, overrides=None, caster=float):
    resolved = dict(defaults)
    if isinstance(overrides, dict):
        for phase in ("early", "mid", "late"):
            value = overrides.get(phase)
            if value is not None:
                resolved[phase] = caster(value)
    return resolved


def _pair_age(current_pair_idx, last_pair_idx):
    if last_pair_idx is None or int(last_pair_idx) < 0:
        return int(current_pair_idx) + 1
    return int(current_pair_idx) - int(last_pair_idx)


def resolve_fb_phase_thresholds(base_thresh, *, phase_thresholds=None, phase_threshold_scales=None):
    resolved_scales = _resolve_phase_mapping(
        FB_DEFAULT_PHASE_SCALES,
        phase_threshold_scales,
        float,
    )
    resolved_thresholds = {}
    for phase in ("early", "mid", "late"):
        explicit_value = None if not isinstance(phase_thresholds, dict) else phase_thresholds.get(phase)
        if explicit_value is None:
            explicit_value = float(base_thresh) * float(resolved_scales[phase])
        resolved_thresholds[phase] = float(explicit_value)
    return resolved_thresholds, resolved_scales


def build_fb_cfg(
    *,
    base_thresh,
    warmup_pairs,
    phase_thresholds=None,
    phase_threshold_scales=None,
    sample_steps=None,
    guide_scale=None,
    age_cap=None,
    scale_init=None,
    scale_low=None,
    scale_high=None,
    **overrides,
):
    resolved_thresholds, resolved_scales = resolve_fb_phase_thresholds(
        base_thresh,
        phase_thresholds=phase_thresholds,
        phase_threshold_scales=phase_threshold_scales,
    )
    cfg = dict(FB_DEFAULTS)
    for key, value in overrides.items():
        if value is not None:
            cfg[key] = value

    resolved_scale_init = _resolve_phase_mapping(
        FB_DEFAULT_SCALE_INIT,
        scale_init,
        float,
    )
    resolved_scale_low = _resolve_phase_mapping(
        FB_DEFAULT_SCALE_LOW,
        scale_low,
        float,
    )
    resolved_scale_high = _resolve_phase_mapping(
        FB_DEFAULT_SCALE_HIGH,
        scale_high,
        float,
    )
    resolved_age_cap = _resolve_phase_mapping(
        FB_DEFAULT_AGE_CAP,
        age_cap,
        int,
    )

    for phase in ("early", "mid", "late"):
        low_value = float(resolved_scale_low[phase])
        init_value = float(resolved_scale_init[phase])
        high_value = float(resolved_scale_high[phase])
        if low_value <= 0.0:
            raise ValueError(f"fb scale_low[{phase}] must be > 0.")
        if high_value < low_value:
            raise ValueError(f"fb scale_high[{phase}] must be >= scale_low[{phase}].")
        if init_value < low_value or init_value > high_value:
            raise ValueError(
                f"fb scale_init[{phase}] must be within [scale_low[{phase}], scale_high[{phase}]]."
            )

    cfg.update(
        {
            "method": "aligncache",
            "scheduler": "aligncache-base-plus-fb",
            "base_thresh": float(base_thresh),
            "warmup_pairs": int(warmup_pairs),
            "sample_steps": None if sample_steps is None else int(sample_steps),
            "guide_scale": None if guide_scale is None else float(guide_scale),
            "base_scheduler": {
                "method": "aligncache_preserved_accumulated_error",
                "base_thresh": float(base_thresh),
                "warmup_pairs": int(warmup_pairs),
                "phase_thresholds": dict(resolved_thresholds),
                "phase_threshold_scales": dict(resolved_scales),
            },
            "phase_thresholds": dict(resolved_thresholds),
            "phase_threshold_scales": dict(resolved_scales),
            "age_cap": dict(resolved_age_cap),
            "scale_init": dict(resolved_scale_init),
            "scale_low": dict(resolved_scale_low),
            "scale_high": dict(resolved_scale_high),
            "threshold_controller": {
                "scale_init": dict(resolved_scale_init),
                "scale_low": dict(resolved_scale_low),
                "scale_high": dict(resolved_scale_high),
            },
            "feedback": {
                "ratio_ema_decay": float(cfg["ratio_ema_decay"]),
                "ratio_clip_low": float(cfg["ratio_clip_low"]),
                "ratio_clip_high": float(cfg["ratio_clip_high"]),
                "update_reasons": ["budget_exact"],
            },
        }
    )
    return cfg


def _init_fb_state(cfg=None):
    cfg = dict(cfg or {})
    scale_init = _resolve_phase_mapping(
        FB_DEFAULT_SCALE_INIT,
        cfg.get("scale_init"),
        float,
    )
    ratio_ema = {
        phase: 1.0 / max(float(scale_init[phase]), 1e-6)
        for phase in ("early", "mid", "late")
    }
    return {
        "scale": dict(scale_init),
        "ratio_ema": dict(ratio_ema),
        "last_exact_pair_idx": -1,
        "pair_mode": None,
        "last_decision": {},
        "fb_phase": None,
        "fb_budget": None,
        "fb_ready": False,
        "cond_draft": None,
        "uncond_draft": None,
        "stats": {
            "exact_pairs": 0,
            "skipped_pairs": 0,
            "warmup_tail_pairs": 0,
            "bootstrap_pairs": 0,
            "max_age_pairs": 0,
            "budget_exact_pairs": 0,
            "budget_skip_pairs": 0,
            "cache_miss_exact_pairs": 0,
            "feedback_updates": 0,
            "feedback_skipped_pairs": 0,
            "max_anchor_age": 0,
        },
    }


def _reset_fb_state(state):
    state["fb_phase"] = None
    state["fb_budget"] = None
    state["fb_ready"] = False
    state["cond_draft"] = None
    state["uncond_draft"] = None


def _reset_fb_state_runtime(self):
    self.fb_state = _init_fb_state(getattr(self, "fb_cfg", None))


def _inc_stats(self, key):
    state = getattr(self, "fb_state", None)
    if state is None:
        return
    stats = state["stats"]
    if key in stats:
        stats[key] += 1


def _should_update_fb(reason, budget_next):
    if budget_next is None:
        return False
    return str(reason) == "budget_exact"


def _register_pair_decision(
    self,
    phase,
    *,
    exec_mode,
    reason,
    budget_next,
    pred_change_base,
    base_phase_threshold,
    phase_threshold,
    threshold_scale,
    ratio_ema,
    anchor_age,
    feedback_eligible,
):
    state = self.fb_state
    state["pair_mode"] = str(exec_mode)
    state["fb_phase"] = str(phase)
    state["fb_budget"] = None if budget_next is None else float(budget_next)
    state["fb_ready"] = bool(feedback_eligible)
    state["cond_draft"] = None
    state["uncond_draft"] = None
    state["last_decision"] = {
        "decision": str(reason),
        "exec_mode": str(exec_mode),
        "base_phase_threshold": float(base_phase_threshold),
        "phase_threshold": float(phase_threshold),
        "budget_next": None if budget_next is None else float(budget_next),
        "pred_change_base": None if pred_change_base is None else float(pred_change_base),
        "threshold_scale": float(threshold_scale),
        "ratio_ema": float(ratio_ema),
        "anchor_age": int(anchor_age),
        "feedback_eligible": bool(feedback_eligible),
    }

    summary_key = FB_REASON_KEYS.get(str(reason))
    if summary_key is not None:
        _inc_stats(self, summary_key)


def _finish_pair(self, pair_idx, exec_mode):
    state = self.fb_state
    anchor_age = _pair_age(pair_idx, state["last_exact_pair_idx"])
    state["stats"]["max_anchor_age"] = max(
        int(state["stats"]["max_anchor_age"]),
        int(anchor_age),
    )
    if exec_mode == "exact_pair":
        state["last_exact_pair_idx"] = int(pair_idx)
        state["stats"]["exact_pairs"] += 1
    else:
        state["stats"]["skipped_pairs"] += 1
    state["pair_mode"] = str(exec_mode)
    probe = state.get("last_decision")
    if isinstance(probe, dict):
        probe["exec_mode"] = str(exec_mode)


def _predict_cond_draft(self, raw_input, phase):
    draft_output = _predict_pair_output(self, raw_input, phase, True)
    if draft_output is None and self.cond_cache is not None:
        draft_output = [
            (current + delta).detach()
            for current, delta in zip(raw_input, self.cond_cache)
        ]
    return None if draft_output is None else _clone_tensor_list(draft_output)


def _predict_uncond_draft(self, raw_input, phase, pair_idx, cond_draft):
    draft_output = _predict_pair_output(self, raw_input, phase, False)
    if draft_output is None and self.uncond_cache is not None:
        draft_output = [
            (current + delta).detach()
            for current, delta in zip(raw_input, self.uncond_cache)
        ]
    if draft_output is None:
        return None
    anchor_output = _predict_anchor_uncond(
        self,
        phase,
        pair_idx,
        cond_draft,
        draft_output,
        record_event=False,
    )
    if anchor_output is not None:
        draft_output = anchor_output
    return _clone_tensor_list(draft_output)


def _guided_output(cond_output, uncond_output, guide_scale):
    return [
        uncond.detach().to(torch.float32)
        + (cond.detach().to(torch.float32) - uncond.detach().to(torch.float32)) * float(guide_scale)
        for cond, uncond in zip(cond_output, uncond_output)
    ]


def _update_fb_from_exact(self, cond_output, uncond_output):
    state = self.fb_state
    if not bool(state.get("fb_ready")):
        _reset_fb_state(state)
        return

    phase = state.get("fb_phase")
    budget = state.get("fb_budget")
    cond_draft = state.get("cond_draft")
    uncond_draft = state.get("uncond_draft")
    if (
        phase is None
        or budget is None
        or cond_draft is None
        or uncond_draft is None
        or cond_output is None
        or uncond_output is None
    ):
        state["stats"]["feedback_skipped_pairs"] += 1
        _reset_fb_state(state)
        return

    guided_draft = _guided_output(
        cond_draft,
        uncond_draft,
        getattr(self, "sample_guide_scale", 5.0),
    )
    guided_exact = _guided_output(
        cond_output,
        uncond_output,
        getattr(self, "sample_guide_scale", 5.0),
    )
    obs_err = _tensor_list_rel_mse(guided_draft, guided_exact)
    if obs_err is None:
        state["stats"]["feedback_skipped_pairs"] += 1
        _reset_fb_state(state)
        return

    cfg = self.fb_cfg
    err_ratio = float(obs_err) / max(float(budget), float(cfg["score_eps"]))
    clipped_ratio = _clamp_float(
        err_ratio,
        float(cfg["ratio_clip_low"]),
        float(cfg["ratio_clip_high"]),
    )
    prev_ratio_ema = float(state["ratio_ema"][phase])
    updated_ratio_ema = (
        float(cfg["ratio_ema_decay"]) * prev_ratio_ema
        + (1.0 - float(cfg["ratio_ema_decay"])) * float(clipped_ratio)
    )
    state["ratio_ema"][phase] = float(updated_ratio_ema)
    state["scale"][phase] = _clamp_float(
        1.0 / max(float(updated_ratio_ema), float(cfg["score_eps"])),
        float(cfg["scale_low"][phase]),
        float(cfg["scale_high"][phase]),
    )
    state["stats"]["feedback_updates"] += 1
    _reset_fb_state(state)


def _route_pair(state, phase, pair_idx, budget_next, phase_threshold, cfg):
    age = _pair_age(pair_idx, state["last_exact_pair_idx"])
    if budget_next is None:
        return {
            "exec_mode": "exact_pair",
            "reason": "bootstrap",
            "age": int(age),
        }
    if age > int(cfg["age_cap"][phase]):
        return {
            "exec_mode": "exact_pair",
            "reason": "max_age",
            "age": int(age),
        }
    if float(budget_next) < float(phase_threshold):
        return {
            "exec_mode": "skip_pair",
            "reason": "budget_skip",
            "age": int(age),
        }
    return {
        "exec_mode": "exact_pair",
        "reason": "budget_exact",
        "age": int(age),
    }


def _reset_runtime_state(self):
    self.call_idx = 0
    self.skip_budget = 0
    self.run_exact_pair = True
    self.motion_gain = None
    self.is_cond = True
    self.last_cond_input = None
    self.last_cond_output = None
    self.prev_cond_input = None
    self.cond_cache = None
    self.uncond_cache = None
    self.phase_base_thresh = None
    _reset_shadow_pair_state(self)
    self.anchor_state = _init_anchor_state(self.pair_cfg)
    _clear_pending_cond(self)
    _reset_fb_state_runtime(self)


def _advance_counter(self):
    self.call_idx += 1
    if self.call_idx >= self.total_calls:
        self.call_idx = 0
        _clear_pending_cond(self)
        if hasattr(self, "fb_state"):
            _reset_fb_state(self.fb_state)


def aligncache_forward(self, x, t, context, seq_len, clip_fea=None, y=None):
    if self.model_type == 'i2v':
        assert clip_fea is not None and y is not None

    raw_input = _clone_tensor_list(x)
    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)

    if y is not None:
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

    self.is_cond = (self.call_idx % 2 == 0)
    pair_idx = self.call_idx // 2
    total_pairs = max(self.total_calls // 2, 1)
    phase = _get_phase(pair_idx, total_pairs, self.pair_cfg)
    state = self.fb_state
    base_phase_threshold = _phase_skip_threshold(self, phase)
    threshold_scale = float(state["scale"][phase])
    current_phase_threshold = float(base_phase_threshold) * float(threshold_scale)
    current_ratio_ema = float(state["ratio_ema"][phase])
    self.phase_base_thresh = float(base_phase_threshold)

    if self.is_cond:
        pred_change_base = None
        budget_next = None
        anchor_age = _pair_age(pair_idx, state["last_exact_pair_idx"])
        exec_mode = "exact_pair"
        feedback_eligible = False
        if self.call_idx < self.warmup_calls or self.call_idx >= self.tail_call_start:
            self.run_exact_pair = True
            self.skip_budget = 0
            decision_reason = "warmup_tail"
        else:
            motion = None
            if (
                self.last_cond_input is not None
                and self.last_cond_output is not None
            ):
                raw_input_change = torch.cat([
                    (u - v).flatten() for u, v in zip(raw_input, self.last_cond_input)
                ]).abs().mean()
                output_norm = _tensor_list_mean_abs(self.last_cond_output)
                if output_norm is not None and self.motion_gain is not None:
                    motion = float(raw_input_change.item()) / max(
                        float(output_norm),
                        float(self.fb_cfg["score_eps"]),
                    )
                    pred_change_base = float(self.motion_gain) * float(motion)
                    budget_next = float(self.skip_budget) + float(pred_change_base)
            route = _route_pair(
                state,
                phase,
                pair_idx,
                budget_next,
                current_phase_threshold,
                self.fb_cfg,
            )
            anchor_age = int(route["age"])
            exec_mode = str(route["exec_mode"])
            decision_reason = str(route["reason"])
            self.run_exact_pair = exec_mode == "exact_pair"
            feedback_eligible = (
                self.run_exact_pair
                and _should_update_fb(decision_reason, budget_next)
            )
            if self.run_exact_pair:
                self.skip_budget = 0
            else:
                self.skip_budget = float(budget_next)

        if exec_mode == "skip_pair" and (self.cond_cache is None or self.uncond_cache is None):
            self.run_exact_pair = True
            exec_mode = "exact_pair"
            feedback_eligible = False
            decision_reason = "cache_miss_exact"
            self.skip_budget = 0

        _register_pair_decision(
            self,
            phase,
            exec_mode=exec_mode,
            reason=decision_reason,
            budget_next=budget_next,
            pred_change_base=pred_change_base,
            base_phase_threshold=base_phase_threshold,
            phase_threshold=current_phase_threshold,
            threshold_scale=threshold_scale,
            ratio_ema=current_ratio_ema,
            anchor_age=anchor_age,
            feedback_eligible=feedback_eligible,
        )
        if self.run_exact_pair and feedback_eligible:
            state["cond_draft"] = _predict_cond_draft(
                self,
                raw_input,
                phase,
            )

        self.last_cond_input = _clone_tensor_list(raw_input)

    if self.is_cond and not self.run_exact_pair and self.cond_cache is not None:
        pred_output = _predict_pair_output(self, raw_input, phase, True)
        if pred_output is None:
            pred_output = [(u + v).detach() for u, v in zip(raw_input, self.cond_cache)]
        _set_pending_cond(self, pred_output, False)
        if self.pair_cfg.get('enable', True):
            _write_pair_branch_result(
                self, raw_input, pred_output, True, phase, False)
        _advance_counter(self)
        return [u.float() for u in pred_output]

    pair_mode = state.get("pair_mode", "exact_pair")

    if (not self.is_cond) and pair_mode == "skip_pair" and self.uncond_cache is not None:
        pred_output = _predict_pair_output(self, raw_input, phase, False)
        if pred_output is None:
            pred_output = [(u + v).detach() for u, v in zip(raw_input, self.uncond_cache)]
        anchor_output = _predict_anchor_uncond(
            self,
            phase,
            pair_idx,
            getattr(self, 'pending_cond', None),
            pred_output,
        )
        if anchor_output is not None:
            pred_output = anchor_output
        if self.pair_cfg.get('enable', True):
            _write_pair_branch_result(
                self, raw_input, pred_output, False, phase, False)
        _clear_pending_cond(self)
        _finish_pair(self, pair_idx, "skip_pair")
        _reset_fb_state(state)
        _advance_counter(self)
        return [u.float() for u in pred_output]

    if (
        (not self.is_cond)
        and self.run_exact_pair
        and bool(state.get("fb_ready"))
        and state["uncond_draft"] is None
    ):
        state["uncond_draft"] = _predict_uncond_draft(
            self,
            raw_input,
            phase,
            pair_idx,
            state.get("cond_draft"),
        )

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

    if self.is_cond:
        if self.last_cond_output is not None:
            output_change = torch.cat([
                (u - v).flatten() for u, v in zip(output, self.last_cond_output)
            ]).abs().mean()

            if self.prev_cond_input is not None:
                input_change = torch.cat([
                    (u - v).flatten() for u, v in zip(
                        self.last_cond_input, self.prev_cond_input
                    )
                ]).abs().mean()
                self.motion_gain = float(output_change.item()) / max(
                    float(input_change.item()),
                    float(self.fb_cfg["score_eps"]),
                )

        self.prev_cond_input = getattr(self, 'last_cond_input', None)
        self.last_cond_output = _clone_tensor_list(output)
        self.cond_cache = [u.detach() - v.detach() for u, v in zip(output, raw_input)]
        _set_pending_cond(self, output, True)
        if self.pair_cfg.get('enable', True):
            _write_pair_branch_result(
                self, raw_input, output, True, phase, True)
    else:
        self.uncond_cache = [u.detach() - v.detach() for u, v in zip(output, raw_input)]
        if self.pair_cfg.get('enable', True):
            _write_pair_branch_result(
                self, raw_input, output, False, phase, True)
            _update_anchor_state(
                self,
                phase,
                pair_idx,
                getattr(self, 'pending_cond', None),
                output,
                getattr(self, 'pending_cond_exact', False),
            )
        _update_fb_from_exact(
            self,
            getattr(self, 'pending_cond', None),
            output,
        )
        _clear_pending_cond(self)
        _finish_pair(self, pair_idx, "exact_pair")

    _advance_counter(self)
    return [u.float() for u in output]


def _log_anchor_summary(model):
    state = getattr(model, 'anchor_state', None)
    if state is None:
        return
    counts = state["counts"]
    logging.info(
        "AlignCache anchor: anchor_updates=%s applied=%s no_anchor=%s early=%s max_age=%s cond=%s disagree=%s weak=%s",
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
        "AlignCache anchor alpha: early=%.4f mid=%.4f late=%.4f",
        state["alpha_ema"]["early"],
        state["alpha_ema"]["mid"],
        state["alpha_ema"]["late"],
    )


def _log_fb_summary(model):
    state = getattr(model, "fb_state", None)
    if state is None:
        return
    logging.info("AlignCache stats: %s", dict(state.get("stats", {})))
    logging.info("AlignCache threshold scale: %s", dict(state.get("scale", {})))
    logging.info("AlignCache ratio EMA: %s", dict(state.get("ratio_ema", {})))


def _setup_fb(model, args):
    model_class = model if isinstance(model, type) else model.__class__
    model_instance = model
    model_class.forward = aligncache_forward
    model_instance.total_calls = args.sample_steps * 2
    model_instance.base_thresh = args.base_thresh
    model_instance.warmup_calls = int(args.warmup_pairs) * 2
    model_instance.tail_call_start = args.sample_steps * 2 - 2
    model_instance.sample_guide_scale = float(args.sample_guide_scale)
    model_instance.pair_cfg = _build_pair_cfg(args)
    model_instance.fb_cfg = build_fb_cfg(
        base_thresh=args.base_thresh,
        warmup_pairs=args.warmup_pairs,
        sample_steps=args.sample_steps,
        guide_scale=args.sample_guide_scale,
        phase_thresholds={
            "early": args.phase_thresh_early,
            "mid": args.phase_thresh_mid,
            "late": args.phase_thresh_late,
        },
        phase_threshold_scales={
            "early": args.phase_scale_early,
            "mid": args.phase_scale_mid,
            "late": args.phase_scale_late,
        },
        age_cap={
            "early": args.fb_age_cap_early,
            "mid": args.fb_age_cap_mid,
            "late": args.fb_age_cap_late,
        },
        scale_init={
            "early": args.fb_scale_init_early,
            "mid": args.fb_scale_init_mid,
            "late": args.fb_scale_init_late,
        },
        scale_low={
            "early": args.fb_scale_low_early,
            "mid": args.fb_scale_low_mid,
            "late": args.fb_scale_low_late,
        },
        scale_high={
            "early": args.fb_scale_high_early,
            "mid": args.fb_scale_high_mid,
            "late": args.fb_scale_high_late,
        },
        ratio_ema_decay=args.fb_ratio_ema_decay,
        ratio_clip_low=args.fb_ratio_clip_low,
        ratio_clip_high=args.fb_ratio_clip_high,
    )
    model_instance.phase_skip_thresholds = dict(model_instance.fb_cfg["phase_thresholds"])
    model_instance.phase_threshold_scales = dict(model_instance.fb_cfg["phase_threshold_scales"])
    _reset_runtime_state(model_instance)
    logging.info(
        "AlignCache: phase_thresholds=%s age_cap=%s ratio_ema_decay=%.3f ratio_clip=[%.2f, %.2f] "
        "scale_init=%s scale_low=%s scale_high=%s",
        model_instance.fb_cfg["phase_thresholds"],
        model_instance.fb_cfg["age_cap"],
        float(model_instance.fb_cfg["ratio_ema_decay"]),
        float(model_instance.fb_cfg["ratio_clip_low"]),
        float(model_instance.fb_cfg["ratio_clip_high"]),
        model_instance.fb_cfg["scale_init"],
        model_instance.fb_cfg["scale_low"],
        model_instance.fb_cfg["scale_high"],
    )
    logging.info(
        "AlignCache predictor: beta_mid=%.2f beta_late=%.2f online_scale=%s anchor=%s",
        model_instance.pair_cfg['beta_mid'],
        model_instance.pair_cfg['beta_late'],
        model_instance.pair_cfg['online_scale'],
        model_instance.pair_cfg['anchor_enable'],
    )


def resolve_aligncache_phase_thresholds(*, base_thresh=None, thresh=None, **kwargs):
    resolved_base_thresh = base_thresh if base_thresh is not None else thresh
    if resolved_base_thresh is None:
        raise TypeError("base_thresh or thresh is required")
    return resolve_fb_phase_thresholds(resolved_base_thresh, **kwargs)


def build_aligncache_cfg(
    *,
    base_thresh=None,
    warmup_pairs=None,
    thresh=None,
    ret_steps=None,
    **kwargs,
):
    resolved_base_thresh = base_thresh if base_thresh is not None else thresh
    resolved_warmup_pairs = warmup_pairs if warmup_pairs is not None else ret_steps
    if resolved_base_thresh is None or resolved_warmup_pairs is None:
        raise TypeError("base_thresh/thresh and warmup_pairs/ret_steps are required")
    return build_fb_cfg(
        base_thresh=resolved_base_thresh,
        warmup_pairs=resolved_warmup_pairs,
        **kwargs,
    )


def _init_aligncache_state(cfg=None):
    return _init_fb_state(cfg)


def _setup_aligncache(model, args):
    _setup_fb(model, args)


def _log_aligncache_summary(model):
    _log_fb_summary(model)


def _reset_aligncache_state(model):
    _reset_runtime_state(model)


def _build_v2_cfg(args):
    return _build_pair_cfg(args)


def _fb_route_pair(state, phase, pair_idx, budget_next, phase_threshold, cfg):
    return _route_pair(state, phase, pair_idx, budget_next, phase_threshold, cfg)


def _fb_should_collect_feedback(reason, budget_next):
    return _should_update_fb(reason, budget_next)


def _fb_update_from_exact_feedback(self, cond_output, uncond_output):
    _update_fb_from_exact(self, cond_output, uncond_output)


def _validate_pair_args(args):
    assert args.pair_stride >= 1, "pair_stride must be >= 1"
    assert 0.0 <= args.pair_ratio_ema <= 1.0, "pair_ratio_ema must be in [0, 1]"
    assert 0.0 <= args.pair_dir_ema <= 1.0, "pair_dir_ema must be in [0, 1]"
    assert args.pair_beta_mid >= 0.0, "pair_beta_mid must be non-negative"
    assert args.pair_beta_late >= 0.0, "pair_beta_late must be non-negative"
    assert args.pair_ratio_tol_mid >= 0.0, "pair_ratio_tol_mid must be non-negative"
    assert args.pair_ratio_tol_late >= 0.0, "pair_ratio_tol_late must be non-negative"
    assert -1.0 <= args.pair_dir_min_mid <= 1.0, "pair_dir_min_mid must be in [-1, 1]"
    assert -1.0 <= args.pair_dir_min_late <= 1.0, "pair_dir_min_late must be in [-1, 1]"
    assert 0.0 <= args.pair_scale_ema <= 1.0, "pair_scale_ema must be in [0, 1]"
    assert args.pair_scale_low > 0.0, "pair_scale_low must be positive"
    assert args.pair_scale_high >= args.pair_scale_low, (
        "pair_scale_high must be >= pair_scale_low")
    assert args.anchor_age_mid >= 0, "anchor_age_mid must be >= 0"
    assert args.anchor_age_late >= 0, "anchor_age_late must be >= 0"


def _validate_fb_args(args):
    assert args.phase_scale_early > 0.0
    assert args.phase_scale_mid > 0.0
    assert args.phase_scale_late > 0.0
    assert 0.0 <= args.fb_ratio_ema_decay <= 1.0
    assert args.fb_ratio_clip_low > 0.0
    assert args.fb_ratio_clip_high >= args.fb_ratio_clip_low
    assert args.fb_age_cap_early >= 1
    assert args.fb_age_cap_mid >= 1
    assert args.fb_age_cap_late >= 1
    for phase in ("early", "mid", "late"):
        low_value = float(getattr(args, f"fb_scale_low_{phase}"))
        init_value = float(getattr(args, f"fb_scale_init_{phase}"))
        high_value = float(getattr(args, f"fb_scale_high_{phase}"))
        assert low_value > 0.0
        assert high_value >= low_value
        assert low_value <= init_value <= high_value


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate image/video from text or image using Wan with AlignCache scheduling."
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
    parser.add_argument("--base_thresh", "--thresh", dest="base_thresh", type=float, default=0.05, help="Base threshold used by AlignCache.")
    parser.add_argument("--warmup_pairs", "--ret_steps", dest="warmup_pairs", default=10, type=int, help="Warm-up exact pairs before AlignCache can skip.")
    parser.add_argument("--phase_thresh_early", type=float, default=None, help="Optional explicit early-phase threshold override.")
    parser.add_argument("--phase_thresh_mid", type=float, default=None, help="Optional explicit mid-phase threshold override.")
    parser.add_argument("--phase_thresh_late", type=float, default=None, help="Optional explicit late-phase threshold override.")
    parser.add_argument("--phase_scale_early", type=float, default=FB_DEFAULT_PHASE_SCALES["early"], help="Early-phase threshold scale.")
    parser.add_argument("--phase_scale_mid", type=float, default=FB_DEFAULT_PHASE_SCALES["mid"], help="Mid-phase threshold scale.")
    parser.add_argument("--phase_scale_late", type=float, default=FB_DEFAULT_PHASE_SCALES["late"], help="Late-phase threshold scale.")
    parser.add_argument("--fb_ratio_ema_decay", "--fb2_ratio_ema_decay", dest="fb_ratio_ema_decay", type=float, default=FB_DEFAULTS["ratio_ema_decay"])
    parser.add_argument("--fb_ratio_clip_low", "--fb2_ratio_clip_low", dest="fb_ratio_clip_low", type=float, default=FB_DEFAULTS["ratio_clip_low"])
    parser.add_argument("--fb_ratio_clip_high", "--fb2_ratio_clip_high", dest="fb_ratio_clip_high", type=float, default=FB_DEFAULTS["ratio_clip_high"])
    parser.add_argument("--fb_age_cap_early", "--fb2_age_cap_early", dest="fb_age_cap_early", type=int, default=FB_DEFAULT_AGE_CAP["early"])
    parser.add_argument("--fb_age_cap_mid", "--fb2_age_cap_mid", dest="fb_age_cap_mid", type=int, default=FB_DEFAULT_AGE_CAP["mid"])
    parser.add_argument("--fb_age_cap_late", "--fb2_age_cap_late", dest="fb_age_cap_late", type=int, default=FB_DEFAULT_AGE_CAP["late"])
    parser.add_argument("--fb_scale_init_early", "--fb2_scale_init_early", dest="fb_scale_init_early", type=float, default=FB_DEFAULT_SCALE_INIT["early"])
    parser.add_argument("--fb_scale_init_mid", "--fb2_scale_init_mid", dest="fb_scale_init_mid", type=float, default=FB_DEFAULT_SCALE_INIT["mid"])
    parser.add_argument("--fb_scale_init_late", "--fb2_scale_init_late", dest="fb_scale_init_late", type=float, default=FB_DEFAULT_SCALE_INIT["late"])
    parser.add_argument("--fb_scale_low_early", "--fb2_scale_low_early", dest="fb_scale_low_early", type=float, default=FB_DEFAULT_SCALE_LOW["early"])
    parser.add_argument("--fb_scale_low_mid", "--fb2_scale_low_mid", dest="fb_scale_low_mid", type=float, default=FB_DEFAULT_SCALE_LOW["mid"])
    parser.add_argument("--fb_scale_low_late", "--fb2_scale_low_late", dest="fb_scale_low_late", type=float, default=FB_DEFAULT_SCALE_LOW["late"])
    parser.add_argument("--fb_scale_high_early", "--fb2_scale_high_early", dest="fb_scale_high_early", type=float, default=FB_DEFAULT_SCALE_HIGH["early"])
    parser.add_argument("--fb_scale_high_mid", "--fb2_scale_high_mid", dest="fb_scale_high_mid", type=float, default=FB_DEFAULT_SCALE_HIGH["mid"])
    parser.add_argument("--fb_scale_high_late", "--fb2_scale_high_late", dest="fb_scale_high_late", type=float, default=FB_DEFAULT_SCALE_HIGH["late"])
    parser.add_argument("--alpha", default=0.0, type=float)
    parser.add_argument("--beta", default=1.0, type=float)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=946)
    parser.add_argument("--out_dir", type=str, default="./output")

    parser.add_argument("--pair_enable", dest="pair_enable", type=str2bool, default=True)
    parser.add_argument("--pair_aggr", dest="pair_aggr", type=float, default=0.0)
    parser.add_argument("--pair_second_order", dest="pair_second_order", type=str2bool, default=True)
    parser.add_argument("--pair_online_scale", dest="pair_online_scale", type=str2bool, default=True)
    parser.add_argument("--pair_stride", dest="pair_stride", type=int, default=64)
    parser.add_argument("--pair_ratio_ema", dest="pair_ratio_ema", type=float, default=0.8)
    parser.add_argument("--pair_dir_ema", dest="pair_dir_ema", type=float, default=0.8)
    parser.add_argument("--pair_beta_mid", dest="pair_beta_mid", type=float, default=0.35)
    parser.add_argument("--pair_beta_late", dest="pair_beta_late", type=float, default=0.15)
    parser.add_argument("--pair_ratio_tol_mid", dest="pair_ratio_tol_mid", type=float, default=0.30)
    parser.add_argument("--pair_ratio_tol_late", dest="pair_ratio_tol_late", type=float, default=0.18)
    parser.add_argument("--pair_dir_min_mid", dest="pair_dir_min_mid", type=float, default=0.80)
    parser.add_argument("--pair_dir_min_late", dest="pair_dir_min_late", type=float, default=0.88)
    parser.add_argument("--pair_scale_ema", dest="pair_scale_ema", type=float, default=0.85)
    parser.add_argument("--pair_scale_low", dest="pair_scale_low", type=float, default=0.85)
    parser.add_argument("--pair_scale_high", dest="pair_scale_high", type=float, default=1.18)
    parser.add_argument("--anchor_enable", dest="anchor_enable", type=str2bool, default=True)
    parser.add_argument("--anchor_exact_early", dest="anchor_exact_early", type=str2bool, default=True)
    parser.add_argument("--anchor_age_mid", dest="anchor_age_mid", type=int, default=1)
    parser.add_argument("--anchor_age_late", dest="anchor_age_late", type=int, default=2)
    parser.add_argument("--anchor_cond_cap_mid", dest="anchor_cond_cap_mid", type=float, default=0.0050)
    parser.add_argument("--anchor_cond_cap_late", dest="anchor_cond_cap_late", type=float, default=0.0200)
    parser.add_argument("--anchor_alpha_early", dest="anchor_alpha_early", type=float, default=1.279282)
    parser.add_argument("--anchor_alpha_mid", dest="anchor_alpha_mid", type=float, default=1.018750)
    parser.add_argument("--anchor_alpha_late", dest="anchor_alpha_late", type=float, default=0.972800)
    parser.add_argument("--anchor_alpha_ema", dest="anchor_alpha_ema", type=float, default=0.85)
    parser.add_argument("--anchor_blend_mid", dest="anchor_blend_mid", type=float, default=0.35)
    parser.add_argument("--anchor_blend_late", dest="anchor_blend_late", type=float, default=0.20)
    parser.add_argument("--anchor_diff_cap_mid", dest="anchor_diff_cap_mid", type=float, default=0.0200)
    parser.add_argument("--anchor_diff_cap_late", dest="anchor_diff_cap_late", type=float, default=0.0600)

    args = parser.parse_args()
    args.ckpt_dir = normalize_ckpt_dir_for_task(args.task, args.ckpt_dir, size_text=args.size)
    _validate_args(args)
    _validate_pair_args(args)
    _validate_fb_args(args)
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
        _log_aligncache_summary(pipeline.model.__class__)
        _log_anchor_summary(pipeline.model.__class__)

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
        _log_aligncache_summary(pipeline.model.__class__)
        _log_anchor_summary(pipeline.model.__class__)

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
