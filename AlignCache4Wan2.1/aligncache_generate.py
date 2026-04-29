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
    "early": 1.15,
    "mid": 1.00,
    "late": 0.80,
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


import types
from collections.abc import Sequence
from typing import Any

_WAN_SUPPORTED_TASKS = frozenset({"t2v-1.3B", "t2v-14B"})
_PHASES = ("early", "mid", "late")

_ALIGNCACHE_DEFAULT_AGE_CAP = {"early": 5, "mid": 4, "late": 3}
_ALIGNCACHE_DEFAULT_SCALE_INIT = {"early": 1.0, "mid": 1.0, "late": 1.0}
_ALIGNCACHE_DEFAULT_SCALE_LOW = {"early": 0.95, "mid": 0.90, "late": 0.85}
_ALIGNCACHE_DEFAULT_SCALE_HIGH = {"early": 1.03, "mid": 1.06, "late": 1.00}
_ALIGNCACHE_BALANCED_PHASE_THRESHOLD_SCALES = {"early": 1.15, "mid": 1.00, "late": 0.80}
_HUNYUAN_ALIGNCACHE_AGE_CAP = {"early": 5, "mid": 5, "late": 6}
_HUNYUAN_ALIGNCACHE_PHASE_THRESHOLD_SCALES = {"early": 1.00, "mid": 1.10, "late": 1.28}
_HUNYUAN_ALIGNCACHE_SCALE_INIT = {"early": 1.00, "mid": 1.00, "late": 1.00}
_HUNYUAN_ALIGNCACHE_SCALE_LOW = {"early": 1.00, "mid": 1.00, "late": 1.00}
_HUNYUAN_ALIGNCACHE_SCALE_HIGH = {"early": 1.00, "mid": 1.00, "late": 1.00}
_HUNYUAN_ALIGNCACHE_V2_DEFAULTS = {"early_end_ratio": 0.18, "late_start_ratio": 0.72}
_OPENSORA_ALIGNCACHE_PHASE_THRESHOLD_SCALES = {"early": 0.95, "mid": 1.15, "late": 1.00}
_OPENSORA_ALIGNCACHE_AGE_CAP = {"early": 5, "mid": 5, "late": 4}
_OPENSORA_ALIGNCACHE_SCALE_HIGH = {"early": 1.03, "mid": 1.10, "late": 1.05}

_ALIGNCACHE_V2_DEFAULTS = {
    "enable": True,
    "early_end_ratio": 0.15,
    "late_start_ratio": 0.86,
    "subsample_stride": 64,
    "qc_ratio_ema_decay": 0.8,
    "qc_dir_ema_decay": 0.8,
    "beta_mid": 0.35,
    "beta_late": 0.15,
    "ratio_tol_mid": 0.30,
    "ratio_tol_late": 0.18,
    "dir_min_mid": 0.80,
    "dir_min_late": 0.88,
    "use_second_order": True,
    "use_online_scale": True,
    "scale_ema_decay": 0.85,
    "scale_clamp_low": 0.85,
    "scale_clamp_high": 1.18,
    "b2_enable": True,
    "b2_exact_early": True,
    "b2_max_pred_age_mid": 1,
    "b2_max_pred_age_late": 2,
    "b2_cond_rel_cap_mid": 0.0050,
    "b2_cond_rel_cap_late": 0.0200,
    "b2_alpha_prior_early": 1.279282,
    "b2_alpha_prior_mid": 1.018750,
    "b2_alpha_prior_late": 0.972800,
    "b2_alpha_ema_decay": 0.85,
    "b2_blend_mid": 0.35,
    "b2_blend_late": 0.20,
    "b2_disagree_rel_cap_mid": 0.0200,
    "b2_disagree_rel_cap_late": 0.0600,
    "aggressiveness": 0.0,
}

_ALIGNCACHE_REASON_SUMMARY_KEYS = {
    "warmup_tail": "warmup_tail_pairs",
    "bootstrap": "bootstrap_pairs",
    "max_age": "max_age_pairs",
    "budget_exact": "budget_exact_pairs",
    "budget_skip": "budget_skip_pairs",
    "cache_miss_exact": "cache_miss_exact_pairs",
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


def _coerce_optional_str(value: Any, *, label: str) -> str | None:
    if value in (None, ""):
        return None
    if isinstance(value, str):
        return value.strip()
    raise ValueError(f"{label} must be a string when provided. Received {value!r}.")


def _clamp_float(value: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(value)))


def _resolve_phase_mapping(
    defaults: dict[str, Any],
    overrides: dict[str, Any] | None = None,
    caster=float,
) -> dict[str, Any]:
    resolved = {phase: caster(defaults[phase]) for phase in _PHASES}
    for phase in _PHASES:
        if overrides is None:
            continue
        value = overrides.get(phase)
        if value is not None:
            resolved[phase] = caster(value)
    return resolved


def _phase_mapping_from_flat_keys(
    raw: dict[str, Any],
    *,
    mapping_keys: dict[str, str],
    label_prefix: str,
    caster=float,
) -> dict[str, Any]:
    resolved: dict[str, Any] = {}
    for phase, key in mapping_keys.items():
        value = raw.get(key)
        if value in (None, ""):
            continue
        resolved[phase] = caster(value)
    return resolved


def _resolve_phase_override_mapping(
    raw: dict[str, Any],
    *,
    nested_key: str,
    flat_mapping_keys: dict[str, str],
    label_prefix: str,
    caster=float,
) -> dict[str, Any]:
    resolved: dict[str, Any] = {}
    nested = raw.get(nested_key)
    if isinstance(nested, dict):
        for phase, value in dict(nested).items():
            if value in (None, ""):
                continue
            resolved[str(phase)] = caster(value)
    resolved.update(
        _phase_mapping_from_flat_keys(
            raw,
            mapping_keys=flat_mapping_keys,
            label_prefix=label_prefix,
            caster=caster,
        )
    )
    return resolved


def _normalize_aligncache_method_config(method_config: dict[str, Any] | None) -> dict[str, Any]:
    raw = dict(method_config or {})
    nested = raw.get("aligncache")
    if isinstance(nested, dict):
        merged = dict(raw)
        merged.pop("aligncache", None)
        merged.update(nested)
        raw = merged
    return raw


def _apply_v2_aggressiveness(cfg: dict[str, Any]) -> dict[str, Any]:
    resolved = dict(cfg)
    aggressiveness = float(resolved.get("aggressiveness") or 0.0)
    if abs(aggressiveness) <= 1e-8:
        return resolved
    beta_scale = _clamp_float(1.0 + 0.60 * aggressiveness, 0.50, 1.35)
    blend_scale = _clamp_float(1.0 + 0.50 * aggressiveness, 0.50, 1.40)
    clamp_delta = 0.08 * aggressiveness
    resolved["beta_mid"] = float(resolved["beta_mid"]) * beta_scale
    resolved["beta_late"] = float(resolved["beta_late"]) * beta_scale
    resolved["b2_blend_mid"] = float(resolved["b2_blend_mid"]) * blend_scale
    resolved["b2_blend_late"] = float(resolved["b2_blend_late"]) * blend_scale
    resolved["scale_clamp_low"] = max(0.60, float(resolved["scale_clamp_low"]) - clamp_delta)
    resolved["scale_clamp_high"] = max(
        float(resolved["scale_clamp_low"]),
        float(resolved["scale_clamp_high"]) + clamp_delta,
    )
    return resolved


def _coerce_aligncache_v2_cfg_input(value: Any) -> dict[str, Any]:
    source = dict(value or {})
    resolved = dict(_ALIGNCACHE_V2_DEFAULTS)
    alias_map = {
        "enable": ("enable", "pair_enable"),
        "early_end_ratio": ("early_end_ratio", "pair_early_end_ratio"),
        "late_start_ratio": ("late_start_ratio", "pair_late_start_ratio"),
        "subsample_stride": ("subsample_stride", "pair_stride"),
        "qc_ratio_ema_decay": ("qc_ratio_ema_decay", "pair_ratio_ema"),
        "qc_dir_ema_decay": ("qc_dir_ema_decay", "pair_dir_ema"),
        "beta_mid": ("beta_mid", "pair_beta_mid"),
        "beta_late": ("beta_late", "pair_beta_late"),
        "ratio_tol_mid": ("ratio_tol_mid", "pair_ratio_tol_mid"),
        "ratio_tol_late": ("ratio_tol_late", "pair_ratio_tol_late"),
        "dir_min_mid": ("dir_min_mid", "pair_dir_min_mid"),
        "dir_min_late": ("dir_min_late", "pair_dir_min_late"),
        "use_second_order": ("use_second_order", "pair_second_order"),
        "use_online_scale": ("use_online_scale", "pair_online_scale"),
        "scale_ema_decay": ("scale_ema_decay", "pair_scale_ema"),
        "scale_clamp_low": ("scale_clamp_low", "pair_scale_low"),
        "scale_clamp_high": ("scale_clamp_high", "pair_scale_high"),
        "b2_enable": ("b2_enable", "anchor_enable"),
        "b2_exact_early": ("b2_exact_early", "anchor_exact_early"),
        "b2_max_pred_age_mid": ("b2_max_pred_age_mid", "anchor_age_mid"),
        "b2_max_pred_age_late": ("b2_max_pred_age_late", "anchor_age_late"),
        "b2_cond_rel_cap_mid": ("b2_cond_rel_cap_mid", "anchor_cond_cap_mid"),
        "b2_cond_rel_cap_late": ("b2_cond_rel_cap_late", "anchor_cond_cap_late"),
        "b2_alpha_prior_early": ("b2_alpha_prior_early", "anchor_alpha_early"),
        "b2_alpha_prior_mid": ("b2_alpha_prior_mid", "anchor_alpha_mid"),
        "b2_alpha_prior_late": ("b2_alpha_prior_late", "anchor_alpha_late"),
        "b2_alpha_ema_decay": ("b2_alpha_ema_decay", "anchor_alpha_ema"),
        "b2_blend_mid": ("b2_blend_mid", "anchor_blend_mid"),
        "b2_blend_late": ("b2_blend_late", "anchor_blend_late"),
        "b2_disagree_rel_cap_mid": ("b2_disagree_rel_cap_mid", "anchor_diff_cap_mid"),
        "b2_disagree_rel_cap_late": ("b2_disagree_rel_cap_late", "anchor_diff_cap_late"),
        "aggressiveness": ("aggressiveness", "pair_aggr"),
    }
    for key, aliases in alias_map.items():
        value_found = None
        for alias in aliases:
            if alias in source and source[alias] not in (None, ""):
                value_found = source[alias]
                break
        if value_found in (None, ""):
            continue
        if isinstance(_ALIGNCACHE_V2_DEFAULTS[key], bool):
            resolved[key] = bool(_coerce_optional_bool(value_found, label=f"aligncache.v2.{key}"))
        elif isinstance(_ALIGNCACHE_V2_DEFAULTS[key], int):
            resolved[key] = int(_coerce_optional_int(value_found, label=f"aligncache.v2.{key}"))
        else:
            resolved[key] = float(_coerce_optional_float(value_found, label=f"aligncache.v2.{key}"))
    return _apply_v2_aggressiveness(resolved)


def _pair_cfg_from_v2_cfg(v2_cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "enable": bool(v2_cfg["enable"]),
        "early_end_ratio": float(v2_cfg["early_end_ratio"]),
        "late_start_ratio": float(v2_cfg["late_start_ratio"]),
        "stride": int(v2_cfg["subsample_stride"]),
        "ratio_ema": float(v2_cfg["qc_ratio_ema_decay"]),
        "dir_ema": float(v2_cfg["qc_dir_ema_decay"]),
        "beta_mid": float(v2_cfg["beta_mid"]),
        "beta_late": float(v2_cfg["beta_late"]),
        "ratio_tol_mid": float(v2_cfg["ratio_tol_mid"]),
        "ratio_tol_late": float(v2_cfg["ratio_tol_late"]),
        "dir_min_mid": float(v2_cfg["dir_min_mid"]),
        "dir_min_late": float(v2_cfg["dir_min_late"]),
        "second_order": bool(v2_cfg["use_second_order"]),
        "online_scale": bool(v2_cfg["use_online_scale"]),
        "scale_ema": float(v2_cfg["scale_ema_decay"]),
        "scale_low": float(v2_cfg["scale_clamp_low"]),
        "scale_high": float(v2_cfg["scale_clamp_high"]),
        "anchor_enable": bool(v2_cfg["b2_enable"]),
        "anchor_exact_early": bool(v2_cfg["b2_exact_early"]),
        "anchor_age_mid": int(v2_cfg["b2_max_pred_age_mid"]),
        "anchor_age_late": int(v2_cfg["b2_max_pred_age_late"]),
        "anchor_cond_cap_mid": float(v2_cfg["b2_cond_rel_cap_mid"]),
        "anchor_cond_cap_late": float(v2_cfg["b2_cond_rel_cap_late"]),
        "anchor_alpha_early": float(v2_cfg["b2_alpha_prior_early"]),
        "anchor_alpha_mid": float(v2_cfg["b2_alpha_prior_mid"]),
        "anchor_alpha_late": float(v2_cfg["b2_alpha_prior_late"]),
        "anchor_alpha_ema": float(v2_cfg["b2_alpha_ema_decay"]),
        "anchor_blend_mid": float(v2_cfg["b2_blend_mid"]),
        "anchor_blend_late": float(v2_cfg["b2_blend_late"]),
        "anchor_diff_cap_mid": float(v2_cfg["b2_disagree_rel_cap_mid"]),
        "anchor_diff_cap_late": float(v2_cfg["b2_disagree_rel_cap_late"]),
    }


def _resolved_base_thresh(
    raw: dict[str, Any],
    *,
    label_prefix: str,
    default_thresh: float,
) -> float:
    value = _coerce_optional_float(
        raw.get("base_thresh", raw.get("cache_thresh", raw.get("thresh", raw.get("threshold")))),
        label=f"{label_prefix}.base_thresh",
    )
    resolved = float(default_thresh if value is None else value)
    if resolved < 0:
        raise ValueError(f"{label_prefix}.base_thresh must be non-negative. Received {resolved!r}.")
    return resolved


def _resolved_ret_steps(
    raw: dict[str, Any],
    *,
    label_prefix: str,
    default_ret_steps: int,
) -> int:
    value = _coerce_optional_int(
        raw.get("warmup_pairs", raw.get("cache_ret_steps", raw.get("ret_steps"))),
        label=f"{label_prefix}.ret_steps",
    )
    resolved = int(default_ret_steps if value is None else value)
    if resolved < 0:
        raise ValueError(f"{label_prefix}.ret_steps must be non-negative. Received {resolved!r}.")
    return resolved


def _resolve_phase_thresholds(
    *,
    base_thresh: float,
    explicit_thresholds: dict[str, Any] | None,
    explicit_scales: dict[str, Any] | None,
    default_scales: dict[str, Any] | None,
) -> tuple[dict[str, float], dict[str, float]]:
    resolved_scales = _resolve_phase_mapping(
        default_scales or {phase: 1.0 for phase in _PHASES},
        explicit_scales,
        float,
    )
    resolved_thresholds: dict[str, float] = {}
    explicit = explicit_thresholds or {}
    for phase in _PHASES:
        explicit_value = explicit.get(phase)
        if explicit_value is None:
            resolved_thresholds[phase] = float(base_thresh) * float(resolved_scales[phase])
        else:
            resolved_thresholds[phase] = float(explicit_value)
    return resolved_thresholds, resolved_scales


def _resolve_aligncache_config(
    *,
    method_config: dict[str, Any] | None,
    sample_steps: int,
    label_prefix: str,
    default_thresh: float,
    default_ret_steps: int = 10,
    default_phase_threshold_scales: dict[str, float] | None = None,
    default_phase_scale_profile: str | None = None,
    default_age_cap: dict[str, int] | None = None,
    default_scale_init: dict[str, float] | None = None,
    default_scale_low: dict[str, float] | None = None,
    default_scale_high: dict[str, float] | None = None,
    default_v2_cfg_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    raw = _normalize_aligncache_method_config(method_config)
    base_thresh = _resolved_base_thresh(raw, label_prefix=label_prefix, default_thresh=default_thresh)
    ret_steps = _resolved_ret_steps(raw, label_prefix=label_prefix, default_ret_steps=default_ret_steps)

    explicit_thresholds = {}
    explicit_thresholds.update(
        _phase_mapping_from_flat_keys(
            raw,
            mapping_keys={
                "early": "phase_thresh_early",
                "mid": "phase_thresh_mid",
                "late": "phase_thresh_late",
            },
            label_prefix=label_prefix,
            caster=float,
        )
    )
    if isinstance(raw.get("phase_thresholds"), dict):
        explicit_thresholds.update(dict(raw["phase_thresholds"]))

    explicit_scales = {}
    explicit_scales.update(
        _phase_mapping_from_flat_keys(
            raw,
            mapping_keys={
                "early": "phase_scale_early",
                "mid": "phase_scale_mid",
                "late": "phase_scale_late",
            },
            label_prefix=label_prefix,
            caster=float,
        )
    )
    if isinstance(raw.get("phase_threshold_scales"), dict):
        explicit_scales.update(dict(raw["phase_threshold_scales"]))

    phase_thresholds, phase_threshold_scales = _resolve_phase_thresholds(
        base_thresh=base_thresh,
        explicit_thresholds=explicit_thresholds or None,
        explicit_scales=explicit_scales or None,
        default_scales=default_phase_threshold_scales,
    )

    scale_init = _resolve_phase_mapping(
        default_scale_init or _ALIGNCACHE_DEFAULT_SCALE_INIT,
        _resolve_phase_override_mapping(
            raw,
            nested_key="scale_init",
            flat_mapping_keys={
                "early": "fb_scale_init_early",
                "mid": "fb_scale_init_mid",
                "late": "fb_scale_init_late",
            },
            label_prefix=label_prefix,
            caster=float,
        ),
        float,
    )
    scale_low = _resolve_phase_mapping(
        default_scale_low or _ALIGNCACHE_DEFAULT_SCALE_LOW,
        _resolve_phase_override_mapping(
            raw,
            nested_key="scale_low",
            flat_mapping_keys={
                "early": "fb_scale_low_early",
                "mid": "fb_scale_low_mid",
                "late": "fb_scale_low_late",
            },
            label_prefix=label_prefix,
            caster=float,
        ),
        float,
    )
    scale_high = _resolve_phase_mapping(
        default_scale_high or _ALIGNCACHE_DEFAULT_SCALE_HIGH,
        _resolve_phase_override_mapping(
            raw,
            nested_key="scale_high",
            flat_mapping_keys={
                "early": "fb_scale_high_early",
                "mid": "fb_scale_high_mid",
                "late": "fb_scale_high_late",
            },
            label_prefix=label_prefix,
            caster=float,
        ),
        float,
    )
    age_cap = _resolve_phase_mapping(
        default_age_cap or _ALIGNCACHE_DEFAULT_AGE_CAP,
        _resolve_phase_override_mapping(
            raw,
            nested_key="age_cap",
            flat_mapping_keys={
                "early": "fb_age_cap_early",
                "mid": "fb_age_cap_mid",
                "late": "fb_age_cap_late",
            },
            label_prefix=label_prefix,
            caster=int,
        ),
        int,
    )
    feedback_ratio_ema_decay = float(
        _coerce_optional_float(raw.get("fb_ratio_ema_decay", raw.get("ratio_ema_decay")), label=f"{label_prefix}.ratio_ema_decay")
        or 0.90
    )
    feedback_ratio_clip_low = float(
        _coerce_optional_float(raw.get("fb_ratio_clip_low", raw.get("ratio_clip_low")), label=f"{label_prefix}.ratio_clip_low")
        or 0.85
    )
    feedback_ratio_clip_high = float(
        _coerce_optional_float(raw.get("fb_ratio_clip_high", raw.get("ratio_clip_high")), label=f"{label_prefix}.ratio_clip_high")
        or 1.25
    )
    phase_scale_profile = _coerce_optional_str(
        raw.get("phase_scale_profile", raw.get("base_phase_scale_profile")),
        label=f"{label_prefix}.phase_scale_profile",
    )
    v2_input = raw.get("v2_cfg") or raw.get("pair_cfg") or raw.get("predictor") or raw
    if default_v2_cfg_overrides:
        merged_v2_input = dict(default_v2_cfg_overrides)
        if isinstance(v2_input, dict):
            merged_v2_input.update(v2_input)
        v2_input = merged_v2_input
    v2_cfg = _coerce_aligncache_v2_cfg_input(v2_input)
    pair_cfg = _pair_cfg_from_v2_cfg(v2_cfg)
    resolved = {
        "method": "aligncache",
        "scheduler": "aligncache-base-plus-fb",
        "cache_thresh": float(base_thresh),
        "cache_ret_steps": int(ret_steps),
        "base_thresh": float(base_thresh),
        "ret_steps": int(ret_steps),
        "warmup_pairs": int(ret_steps),
        "sample_steps": int(sample_steps),
        "guide_scale": None,
        "phase_thresholds": {phase: float(value) for phase, value in phase_thresholds.items()},
        "phase_threshold_scales": {phase: float(value) for phase, value in phase_threshold_scales.items()},
        "phase_scale_profile": phase_scale_profile if phase_scale_profile is not None else default_phase_scale_profile,
        "base_scheduler": {
            "method": "aligncache_preserved_accumulated_error",
            "base_thresh": float(base_thresh),
            "ret_steps": int(ret_steps),
            "warmup_pairs": int(ret_steps),
            "phase_thresholds": {phase: float(value) for phase, value in phase_thresholds.items()},
            "phase_threshold_scales": {phase: float(value) for phase, value in phase_threshold_scales.items()},
        },
        "age_cap": {phase: int(value) for phase, value in age_cap.items()},
        "scale_init": {phase: float(value) for phase, value in scale_init.items()},
        "scale_low": {phase: float(value) for phase, value in scale_low.items()},
        "scale_high": {phase: float(value) for phase, value in scale_high.items()},
        "ratio_ema_decay": float(feedback_ratio_ema_decay),
        "ratio_clip_low": float(feedback_ratio_clip_low),
        "ratio_clip_high": float(feedback_ratio_clip_high),
        "score_eps": 1e-6,
        "v2_cfg": dict(v2_cfg),
        "pair_cfg": {
            "pair_enable": bool(pair_cfg["enable"]),
            "enable": bool(pair_cfg["enable"]),
            "pair_aggr": float(v2_cfg["aggressiveness"]),
            "pair_second_order": bool(pair_cfg["second_order"]),
            "pair_online_scale": bool(pair_cfg["online_scale"]),
            "pair_stride": int(pair_cfg["stride"]),
            "pair_ratio_ema": float(pair_cfg["ratio_ema"]),
            "pair_dir_ema": float(pair_cfg["dir_ema"]),
            "pair_beta_mid": float(pair_cfg["beta_mid"]),
            "pair_beta_late": float(pair_cfg["beta_late"]),
            "pair_ratio_tol_mid": float(pair_cfg["ratio_tol_mid"]),
            "pair_ratio_tol_late": float(pair_cfg["ratio_tol_late"]),
            "pair_dir_min_mid": float(pair_cfg["dir_min_mid"]),
            "pair_dir_min_late": float(pair_cfg["dir_min_late"]),
            "pair_scale_ema": float(pair_cfg["scale_ema"]),
            "pair_scale_low": float(pair_cfg["scale_low"]),
            "pair_scale_high": float(pair_cfg["scale_high"]),
            "pair_early_end_ratio": float(pair_cfg["early_end_ratio"]),
            "pair_late_start_ratio": float(pair_cfg["late_start_ratio"]),
            "anchor_enable": bool(pair_cfg["anchor_enable"]),
            "anchor_exact_early": bool(pair_cfg["anchor_exact_early"]),
            "anchor_age_mid": int(pair_cfg["anchor_age_mid"]),
            "anchor_age_late": int(pair_cfg["anchor_age_late"]),
            "anchor_cond_cap_mid": float(pair_cfg["anchor_cond_cap_mid"]),
            "anchor_cond_cap_late": float(pair_cfg["anchor_cond_cap_late"]),
            "anchor_alpha_early": float(pair_cfg["anchor_alpha_early"]),
            "anchor_alpha_mid": float(pair_cfg["anchor_alpha_mid"]),
            "anchor_alpha_late": float(pair_cfg["anchor_alpha_late"]),
            "anchor_alpha_ema": float(pair_cfg["anchor_alpha_ema"]),
            "anchor_blend_mid": float(pair_cfg["anchor_blend_mid"]),
            "anchor_blend_late": float(pair_cfg["anchor_blend_late"]),
            "anchor_diff_cap_mid": float(pair_cfg["anchor_diff_cap_mid"]),
            "anchor_diff_cap_late": float(pair_cfg["anchor_diff_cap_late"]),
        },
        "predictor": dict(v2_cfg),
    }
    return resolved


def resolve_wan_aligncache_config(
    *,
    task: str,
    method_config: dict[str, Any] | None,
    sample_steps: int,
) -> dict[str, Any]:
    resolved_task = str(task)
    if resolved_task not in _WAN_SUPPORTED_TASKS:
        supported = ", ".join(sorted(_WAN_SUPPORTED_TASKS))
        raise ValueError(
            "AlignCache for Wan currently supports the vendored T2V tasks only. "
            f"Received task={resolved_task!r}. Supported tasks: {supported}"
        )
    return _resolve_aligncache_config(
        method_config=method_config,
        sample_steps=int(sample_steps),
        label_prefix="wan.aligncache",
        default_thresh=0.065,
        default_ret_steps=12,
        default_phase_threshold_scales=_ALIGNCACHE_BALANCED_PHASE_THRESHOLD_SCALES,
        default_phase_scale_profile="balanced",
    )


def resolve_hunyuan_aligncache_config(
    method_config: dict[str, Any] | None,
    *,
    sample_steps: int,
) -> dict[str, Any]:
    resolved = _resolve_aligncache_config(
        method_config=method_config,
        sample_steps=int(sample_steps),
        label_prefix="hunyuan.aligncache",
        default_thresh=0.030,
        default_ret_steps=8,
        default_phase_threshold_scales=_HUNYUAN_ALIGNCACHE_PHASE_THRESHOLD_SCALES,
        default_phase_scale_profile="hunyuan",
        default_age_cap=_HUNYUAN_ALIGNCACHE_AGE_CAP,
        default_scale_init=_HUNYUAN_ALIGNCACHE_SCALE_INIT,
        default_scale_low=_HUNYUAN_ALIGNCACHE_SCALE_LOW,
        default_scale_high=_HUNYUAN_ALIGNCACHE_SCALE_HIGH,
        default_v2_cfg_overrides=_HUNYUAN_ALIGNCACHE_V2_DEFAULTS,
    )
    raw = _normalize_aligncache_method_config(method_config)
    force_cfg_pair = _coerce_optional_bool(
        raw.get("force_cfg_pair", raw.get("cfg_pair")),
        label="hunyuan.aligncache.force_cfg_pair",
    )
    resolved["force_cfg_pair"] = True if force_cfg_pair is None else bool(force_cfg_pair)
    return resolved


def resolve_opensora_aligncache_config(
    method_config: dict[str, Any] | None,
    *,
    sample_steps: int,
) -> dict[str, Any]:
    return _resolve_aligncache_config(
        method_config=method_config,
        sample_steps=int(sample_steps),
        label_prefix="opensora.aligncache",
        default_thresh=0.11,
        default_ret_steps=8,
        default_phase_threshold_scales=_OPENSORA_ALIGNCACHE_PHASE_THRESHOLD_SCALES,
        default_age_cap=_OPENSORA_ALIGNCACHE_AGE_CAP,
        default_scale_high=_OPENSORA_ALIGNCACHE_SCALE_HIGH,
    )


def _tensor_list(values: torch.Tensor | Sequence[torch.Tensor] | None) -> list[torch.Tensor] | None:
    if values is None:
        return None
    if isinstance(values, torch.Tensor):
        return [values]
    return list(values)


def _hunyuan_has_cfg_pair(raw_input: torch.Tensor | Sequence[torch.Tensor]) -> bool:
    if not isinstance(raw_input, torch.Tensor):
        return False
    if raw_input.ndim < 1:
        return False
    batch_size = int(raw_input.shape[0])
    if batch_size <= 1 or batch_size % 2 != 0:
        return False
    half_batch = batch_size // 2
    try:
        return bool(torch.equal(raw_input[:half_batch], raw_input[half_batch : half_batch * 2]))
    except Exception:
        return False


def _clone_tensor_list(values: Sequence[torch.Tensor] | None) -> list[torch.Tensor] | None:
    if values is None:
        return None
    return [value.detach().clone() for value in values]


def _tensor_list_subsample(values: Sequence[torch.Tensor] | None, stride: int) -> torch.Tensor | None:
    if values is None:
        return None
    stride = max(int(stride), 1)
    pieces = [value.detach().reshape(-1)[::stride].to(torch.float32).clone() for value in values]
    return torch.cat(pieces, dim=0) if pieces else None


def _tensor_list_cosine(first: torch.Tensor | None, second: torch.Tensor | None) -> float | None:
    if first is None or second is None:
        return None
    first_vec = first.reshape(-1).float()
    second_vec = second.reshape(-1).float()
    denom = torch.norm(first_vec, p=2) * torch.norm(second_vec, p=2)
    if float(denom.item()) <= 1e-12:
        return None
    return float((first_vec.dot(second_vec) / denom).clamp(min=-1.0, max=1.0).item())


def _tensor_list_mean_abs(values: Sequence[torch.Tensor] | None) -> float | None:
    if values is None:
        return None
    flat = torch.cat([value.detach().to(torch.float32).flatten() for value in values], dim=0)
    if flat.numel() == 0:
        return 0.0
    return float(flat.abs().mean().item())


def _tensor_list_rel_mse(prediction: Sequence[torch.Tensor] | None, reference: Sequence[torch.Tensor] | None) -> float | None:
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


def _fit_affine_scale_list(target: Sequence[torch.Tensor], basis: Sequence[torch.Tensor], default_alpha: float = 1.0) -> float:
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


def _blend_tensor_lists(base: Sequence[torch.Tensor], target: Sequence[torch.Tensor], weight: float) -> list[torch.Tensor]:
    weight = float(weight)
    if weight <= 1e-6:
        return [value.detach().clone() for value in base]
    if weight >= 1.0:
        return [value.detach().to(dtype=base_value.dtype) for base_value, value in zip(base, target)]
    return [
        base_value + (value.detach().to(dtype=base_value.dtype) - base_value) * weight
        for base_value, value in zip(base, target)
    ]


def _update_running_ema(previous: float | None, current: float | None, decay: float) -> float | None:
    if current is None:
        return previous
    if previous is None:
        return current
    return float(decay) * previous + (1.0 - float(decay)) * current


def _get_phase(step_idx: int, total_steps: int, cfg: dict[str, Any]) -> str:
    ratio = float(step_idx) / max(int(total_steps) - 1, 1)
    if ratio < float(cfg["early_end_ratio"]):
        return "early"
    if ratio < float(cfg["late_start_ratio"]):
        return "mid"
    return "late"


def _correction_params(cfg: dict[str, Any], phase: str) -> tuple[float, float, float]:
    if phase == "mid":
        return float(cfg["beta_mid"]), float(cfg["ratio_tol_mid"]), float(cfg["dir_min_mid"])
    if phase == "late":
        return float(cfg["beta_late"]), float(cfg["ratio_tol_late"]), float(cfg["dir_min_late"])
    return 0.0, 0.0, 1.0


def _init_phase_ema(value: float | None) -> dict[str, float | None]:
    stored = None if value is None else float(value)
    return {phase: stored for phase in _PHASES}


def _ratio_gate(ratio_ema: float | None, ratio_tol: float) -> float:
    if ratio_ema is None:
        return 0.0
    drift = abs(float(ratio_ema) - 1.0)
    return max(0.0, 1.0 - drift / max(float(ratio_tol), 1e-6))


def _dir_gate(direction: float | None, dir_min: float) -> float:
    if direction is None:
        return 0.0
    return max(0.0, min(1.0, (float(direction) - float(dir_min)) / max(1.0 - float(dir_min), 1e-6)))


def _branch_name(is_cond: bool) -> str:
    return "cond" if bool(is_cond) else "uncond"


def _reset_shadow_pair_state(runtime: Any) -> None:
    runtime.cond_last = None
    runtime.uncond_last = None
    runtime.cond_prev = None
    runtime.uncond_prev = None
    runtime.cond_last_sub = None
    runtime.uncond_last_sub = None
    runtime.cond_ratio_ema = None
    runtime.uncond_ratio_ema = None
    runtime.cond_dir_ema = None
    runtime.uncond_dir_ema = None
    runtime.guided_sub = None
    runtime.guided_ratio_ema = None
    runtime.guided_dir_ema = None
    runtime.guided_ready = False
    runtime.cond_scale_ema = _init_phase_ema(1.0)
    runtime.uncond_scale_ema = _init_phase_ema(1.0)
    runtime.single_cache = None
    runtime.single_last = None
    runtime.single_prev = None
    runtime.single_last_sub = None
    runtime.single_ratio_ema = None
    runtime.single_dir_ema = None
    runtime.single_scale_ema = _init_phase_ema(1.0)


def _init_anchor_state(cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "anchor_cond": None,
        "anchor_uncond": None,
        "last_exact_pair": -1,
        "alpha_ema": {
            "early": float(cfg["anchor_alpha_early"]),
            "mid": float(cfg["anchor_alpha_mid"]),
            "late": float(cfg["anchor_alpha_late"]),
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


def _record_anchor_event(runtime: Any, key: str, *, enabled: bool = True) -> None:
    if not enabled:
        return
    state = getattr(runtime, "anchor_state", None)
    if state is None:
        return
    counts = state["counts"]
    if key in counts:
        counts[key] += 1


def _clear_v2_pending_cond_state(runtime: Any) -> None:
    runtime.pending_cond = None
    runtime.pending_cond_exact = False


def _set_v2_pending_cond_state(runtime: Any, output: Sequence[torch.Tensor], exact_pair: bool) -> None:
    runtime.pending_cond = _clone_tensor_list(output)
    runtime.pending_cond_exact = bool(exact_pair)


def _store_pair_branch_history(runtime: Any, residual: Sequence[torch.Tensor], is_cond: bool) -> None:
    residual_detached = _clone_tensor_list(residual)
    residual_sub = _tensor_list_subsample(residual_detached, runtime.pair_cfg["stride"])
    if is_cond:
        last = runtime.cond_last
        last_sub = runtime.cond_last_sub
        ratio_ema = runtime.cond_ratio_ema
        dir_ema = runtime.cond_dir_ema
    else:
        last = runtime.uncond_last
        last_sub = runtime.uncond_last_sub
        ratio_ema = runtime.uncond_ratio_ema
        dir_ema = runtime.uncond_dir_ema

    if last_sub is not None and residual_sub is not None:
        prev_norm = torch.norm(last_sub, p=2).clamp_(min=1e-12)
        curr_norm = torch.norm(residual_sub, p=2)
        ratio = float((curr_norm / prev_norm).item())
        direction = _tensor_list_cosine(residual_sub, last_sub)
        ratio_ema = _update_running_ema(ratio_ema, ratio, runtime.pair_cfg["ratio_ema"])
        dir_ema = _update_running_ema(dir_ema, direction, runtime.pair_cfg["dir_ema"])

    if is_cond:
        runtime.cond_prev = last
        runtime.cond_last = residual_detached
        runtime.cond_last_sub = residual_sub
        runtime.cond_ratio_ema = ratio_ema
        runtime.cond_dir_ema = dir_ema
    else:
        runtime.uncond_prev = last
        runtime.uncond_last = residual_detached
        runtime.uncond_last_sub = residual_sub
        runtime.uncond_ratio_ema = ratio_ema
        runtime.uncond_dir_ema = dir_ema


def _compose_guided_pair_sub(runtime: Any, cond_sub: torch.Tensor | None, uncond_sub: torch.Tensor | None) -> torch.Tensor | None:
    if cond_sub is None or uncond_sub is None:
        return None
    guide_scale = float(getattr(runtime, "sample_guide_scale", 5.0))
    return uncond_sub + (cond_sub - uncond_sub) * guide_scale


def _update_guided_pair_history(runtime: Any) -> None:
    guided_sub = _compose_guided_pair_sub(runtime, runtime.cond_last_sub, runtime.uncond_last_sub)
    if guided_sub is None:
        runtime.guided_ready = False
        return
    prev_guided = runtime.guided_sub
    if prev_guided is not None:
        prev_norm = torch.norm(prev_guided, p=2).clamp_(min=1e-12)
        curr_norm = torch.norm(guided_sub, p=2)
        ratio = float((curr_norm / prev_norm).item())
        direction = _tensor_list_cosine(guided_sub, prev_guided)
        runtime.guided_ratio_ema = _update_running_ema(runtime.guided_ratio_ema, ratio, runtime.pair_cfg["ratio_ema"])
        runtime.guided_dir_ema = _update_running_ema(runtime.guided_dir_ema, direction, runtime.pair_cfg["dir_ema"])
    runtime.guided_sub = guided_sub.clone()
    runtime.guided_ready = True


def _pair_scale_state(runtime: Any, is_cond: bool) -> dict[str, float | None]:
    return runtime.cond_scale_ema if bool(is_cond) else runtime.uncond_scale_ema


def _pair_scale_value(runtime: Any, phase: str, is_cond: bool) -> float:
    if not runtime.pair_cfg.get("online_scale", True):
        return 1.0
    scale_state = _pair_scale_state(runtime, is_cond)
    scale = 1.0 if scale_state is None else float(scale_state[phase])
    return _clamp_float(scale, runtime.pair_cfg["scale_low"], runtime.pair_cfg["scale_high"])


def _pair_blend_coeff(runtime: Any, phase: str, is_cond: bool) -> tuple[float, float]:
    beta, ratio_tol, dir_min = _correction_params(runtime.pair_cfg, phase)
    if beta <= 0.0:
        return 0.0, 0.0
    if is_cond:
        branch_ratio_ema = runtime.cond_ratio_ema
        branch_dir_ema = runtime.cond_dir_ema
    else:
        branch_ratio_ema = runtime.uncond_ratio_ema
        branch_dir_ema = runtime.uncond_dir_ema
    gate = min(
        _ratio_gate(runtime.guided_ratio_ema, ratio_tol),
        _dir_gate(runtime.guided_dir_ema, dir_min),
        _ratio_gate(branch_ratio_ema, ratio_tol),
        _dir_gate(branch_dir_ema, dir_min),
    )
    beta_eff = beta * gate
    return beta_eff, gate


def _predict_pair_residual(runtime: Any, phase: str, is_cond: bool) -> list[torch.Tensor] | None:
    last = runtime.cond_last if bool(is_cond) else runtime.uncond_last
    prev = runtime.cond_prev if bool(is_cond) else runtime.uncond_prev
    if last is None:
        return None
    base_scale = _pair_scale_value(runtime, phase, is_cond)
    base_pred = [tensor.detach().to(torch.float32) * base_scale for tensor in last]
    if not runtime.pair_cfg.get("second_order", True) or prev is None:
        return base_pred
    beta_eff, blend_gate = _pair_blend_coeff(runtime, phase, is_cond)
    if beta_eff <= 1e-4 or blend_gate <= 1e-4:
        return base_pred
    return [
        base + (current.detach().to(torch.float32) - previous.detach().to(torch.float32)) * beta_eff
        for base, current, previous in zip(base_pred, last, prev)
    ]


def _update_pair_scale_from_exact(runtime: Any, residual: Sequence[torch.Tensor], phase: str, is_cond: bool) -> None:
    if not runtime.pair_cfg.get("online_scale", True):
        return
    last = runtime.cond_last if bool(is_cond) else runtime.uncond_last
    if last is None:
        return
    scale_state = _pair_scale_state(runtime, is_cond)
    fitted = _fit_affine_scale_list(residual, last, default_alpha=scale_state[phase])
    fitted = _clamp_float(fitted, runtime.pair_cfg["scale_low"], runtime.pair_cfg["scale_high"])
    scale_state[phase] = _update_running_ema(scale_state[phase], fitted, runtime.pair_cfg["scale_ema"])


def _learn_pair_from_exact(runtime: Any, residual: Sequence[torch.Tensor], phase: str, is_cond: bool) -> None:
    _update_pair_scale_from_exact(runtime, residual, phase, is_cond)


def _predict_pair_output(runtime: Any, raw_input: Sequence[torch.Tensor], phase: str, is_cond: bool) -> list[torch.Tensor] | None:
    cache = runtime.cond_cache if bool(is_cond) else runtime.uncond_cache
    if cache is None:
        return None
    base_output = [value + delta.to(dtype=value.dtype) for value, delta in zip(raw_input, cache)]
    if not runtime.pair_cfg.get("enable", True) or not runtime.guided_ready:
        return base_output
    pred_residual = _predict_pair_residual(runtime, phase, is_cond)
    if pred_residual is None:
        return base_output
    return [value + delta.to(dtype=value.dtype) for value, delta in zip(raw_input, pred_residual)]


def _write_pair_branch_result(
    runtime: Any,
    raw_input: Sequence[torch.Tensor],
    output: Sequence[torch.Tensor],
    is_cond: bool,
    phase: str,
    learn_from_exact: bool,
) -> None:
    residual = [(u - v).detach() for u, v in zip(output, raw_input)]
    if learn_from_exact:
        _learn_pair_from_exact(runtime, residual, phase, is_cond)
    _store_pair_branch_history(runtime, residual, is_cond)
    if not bool(is_cond):
        _update_guided_pair_history(runtime)


def _single_scale_value(runtime: Any, phase: str) -> float:
    if not runtime.pair_cfg.get("online_scale", True):
        return 1.0
    scale_state = getattr(runtime, "single_scale_ema", None)
    if not isinstance(scale_state, dict):
        runtime.single_scale_ema = _init_phase_ema(1.0)
        scale_state = runtime.single_scale_ema
    scale = 1.0 if scale_state.get(phase) is None else float(scale_state[phase])
    return _clamp_float(scale, runtime.pair_cfg["scale_low"], runtime.pair_cfg["scale_high"])


def _single_blend_coeff(runtime: Any, phase: str) -> tuple[float, float]:
    beta, ratio_tol, dir_min = _correction_params(runtime.pair_cfg, phase)
    if beta <= 0.0:
        return 0.0, 0.0
    gate = min(
        _ratio_gate(getattr(runtime, "single_ratio_ema", None), ratio_tol),
        _dir_gate(getattr(runtime, "single_dir_ema", None), dir_min),
    )
    beta_eff = beta * gate
    return beta_eff, gate


def _predict_single_residual(runtime: Any, phase: str) -> list[torch.Tensor] | None:
    last = getattr(runtime, "single_last", None)
    prev = getattr(runtime, "single_prev", None)
    if last is None:
        return None
    base_scale = _single_scale_value(runtime, phase)
    base_pred = [tensor.detach().to(torch.float32) * base_scale for tensor in last]
    if not runtime.pair_cfg.get("second_order", True) or prev is None:
        return base_pred
    beta_eff, blend_gate = _single_blend_coeff(runtime, phase)
    if beta_eff <= 1e-4 or blend_gate <= 1e-4:
        return base_pred
    return [
        base + (current.detach().to(torch.float32) - previous.detach().to(torch.float32)) * beta_eff
        for base, current, previous in zip(base_pred, last, prev)
    ]


def _predict_single_output(
    runtime: Any,
    raw_input: Sequence[torch.Tensor],
    phase: str,
) -> list[torch.Tensor] | None:
    cache = getattr(runtime, "single_cache", None)
    if cache is None:
        return None
    base_output = [value + delta.to(dtype=value.dtype) for value, delta in zip(raw_input, cache)]
    if not runtime.pair_cfg.get("enable", True):
        return base_output
    pred_residual = _predict_single_residual(runtime, phase)
    if pred_residual is None:
        return base_output
    return [value + delta.to(dtype=value.dtype) for value, delta in zip(raw_input, pred_residual)]


def _update_single_scale_from_exact(runtime: Any, residual: Sequence[torch.Tensor], phase: str) -> None:
    if not runtime.pair_cfg.get("online_scale", True):
        return
    last = getattr(runtime, "single_last", None)
    if last is None:
        return
    scale_state = getattr(runtime, "single_scale_ema", None)
    if not isinstance(scale_state, dict):
        runtime.single_scale_ema = _init_phase_ema(1.0)
        scale_state = runtime.single_scale_ema
    fitted = _fit_affine_scale_list(residual, last, default_alpha=scale_state[phase])
    fitted = _clamp_float(fitted, runtime.pair_cfg["scale_low"], runtime.pair_cfg["scale_high"])
    scale_state[phase] = _update_running_ema(scale_state[phase], fitted, runtime.pair_cfg["scale_ema"])


def _store_single_branch_history(runtime: Any, residual: Sequence[torch.Tensor]) -> None:
    residual_detached = _clone_tensor_list(residual)
    residual_sub = _tensor_list_subsample(residual_detached, runtime.pair_cfg["stride"])
    last = getattr(runtime, "single_last", None)
    last_sub = getattr(runtime, "single_last_sub", None)
    ratio_ema = getattr(runtime, "single_ratio_ema", None)
    dir_ema = getattr(runtime, "single_dir_ema", None)

    if last_sub is not None and residual_sub is not None:
        prev_norm = torch.norm(last_sub, p=2).clamp_(min=1e-12)
        curr_norm = torch.norm(residual_sub, p=2)
        ratio = float((curr_norm / prev_norm).item())
        direction = _tensor_list_cosine(residual_sub, last_sub)
        ratio_ema = _update_running_ema(ratio_ema, ratio, runtime.pair_cfg["ratio_ema"])
        dir_ema = _update_running_ema(dir_ema, direction, runtime.pair_cfg["dir_ema"])

    runtime.single_prev = last
    runtime.single_last = residual_detached
    runtime.single_last_sub = residual_sub
    runtime.single_ratio_ema = ratio_ema
    runtime.single_dir_ema = dir_ema


def _write_single_branch_result(
    runtime: Any,
    raw_input: Sequence[torch.Tensor],
    output: Sequence[torch.Tensor],
    phase: str,
    learn_from_exact: bool,
) -> list[torch.Tensor]:
    residual = [(u - v).detach() for u, v in zip(output, raw_input)]
    runtime.single_cache = _clone_tensor_list(residual)
    if learn_from_exact:
        _update_single_scale_from_exact(runtime, residual, phase)
    _store_single_branch_history(runtime, residual)
    return residual


def _predict_anchor_uncond(
    runtime: Any,
    phase: str,
    step_idx: int,
    cond_output: Sequence[torch.Tensor] | None,
    base_output: Sequence[torch.Tensor] | None,
    *,
    record_event: bool = True,
) -> list[torch.Tensor] | None:
    cfg = runtime.pair_cfg
    if not cfg.get("enable", True) or not cfg.get("anchor_enable", True):
        return None
    state = getattr(runtime, "anchor_state", None)
    if state is None or cond_output is None or base_output is None:
        _record_anchor_event(runtime, "reject_no_anchor", enabled=record_event)
        return None
    if state["anchor_cond"] is None or state["anchor_uncond"] is None:
        _record_anchor_event(runtime, "reject_no_anchor", enabled=record_event)
        return None
    if cfg.get("anchor_exact_early", True) and phase == "early":
        _record_anchor_event(runtime, "reject_early_exact", enabled=record_event)
        return None
    max_pred_age = {"early": 0, "mid": int(cfg["anchor_age_mid"]), "late": int(cfg["anchor_age_late"])}
    anchor_age = int(step_idx) - int(state["last_exact_pair"])
    if anchor_age > max_pred_age[phase]:
        _record_anchor_event(runtime, "reject_max_age", enabled=record_event)
        return None
    cond_rel_cap = {"early": 0.0, "mid": float(cfg["anchor_cond_cap_mid"]), "late": float(cfg["anchor_cond_cap_late"])}
    cond_rel = _tensor_list_rel_mse(cond_output, state["anchor_cond"])
    cond_rel = 0.0 if cond_rel is None else cond_rel
    if cond_rel > cond_rel_cap[phase]:
        _record_anchor_event(runtime, "reject_cond_gate", enabled=record_event)
        return None
    alpha = float(state["alpha_ema"][phase])
    anchor_output = [
        anchor_uncond.detach().to(torch.float32)
        + (cond_value.detach().to(torch.float32) - anchor_cond.detach().to(torch.float32)) * alpha
        for cond_value, anchor_cond, anchor_uncond in zip(cond_output, state["anchor_cond"], state["anchor_uncond"])
    ]
    disagree_cap = {"early": 0.0, "mid": float(cfg["anchor_diff_cap_mid"]), "late": float(cfg["anchor_diff_cap_late"])}
    disagree_rel = _tensor_list_rel_mse(anchor_output, base_output)
    disagree_rel = 0.0 if disagree_rel is None else disagree_rel
    if disagree_rel > disagree_cap[phase]:
        _record_anchor_event(runtime, "reject_disagree", enabled=record_event)
        return None
    blend_base = {"early": 0.0, "mid": float(cfg["anchor_blend_mid"]), "late": float(cfg["anchor_blend_late"])}
    blend_weight = blend_base[phase]
    if blend_weight <= 1e-6:
        _record_anchor_event(runtime, "reject_weak_gate", enabled=record_event)
        return None
    cond_gate = 1.0 if cond_rel_cap[phase] <= 0.0 else max(0.0, 1.0 - cond_rel / max(cond_rel_cap[phase], 1e-6))
    disagree_gate = 1.0 if disagree_cap[phase] <= 0.0 else max(0.0, 1.0 - disagree_rel / max(disagree_cap[phase], 1e-6))
    if max_pred_age[phase] <= 1:
        age_gate = 1.0
    else:
        age_gate = max(0.0, 1.0 - max(float(anchor_age) - 1.0, 0.0) / max(float(max_pred_age[phase] - 1), 1.0))
    blend_weight *= min(cond_gate, disagree_gate, age_gate)
    if blend_weight <= 1e-4:
        _record_anchor_event(runtime, "reject_weak_gate", enabled=record_event)
        return None
    _record_anchor_event(runtime, "applied_steps", enabled=record_event)
    return _blend_tensor_lists(base_output, anchor_output, blend_weight)


def _update_anchor_state(
    runtime: Any,
    phase: str,
    step_idx: int,
    cond_output: Sequence[torch.Tensor] | None,
    uncond_output: Sequence[torch.Tensor] | None,
    learn: bool,
) -> None:
    cfg = runtime.pair_cfg
    if not cfg.get("enable", True) or not cfg.get("anchor_enable", True):
        return
    state = getattr(runtime, "anchor_state", None)
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
        alpha = _fit_affine_scale_list(uncond_delta, cond_delta, default_alpha=state["alpha_ema"][phase])
        decay = float(cfg["anchor_alpha_ema"])
        state["alpha_ema"][phase] = decay * state["alpha_ema"][phase] + (1.0 - decay) * alpha
    state["anchor_cond"] = _clone_tensor_list(cond_output)
    state["anchor_uncond"] = _clone_tensor_list(uncond_output)
    state["last_exact_pair"] = int(step_idx)
    state["counts"]["anchor_updates"] += 1


def _guided_output(
    cond_output: Sequence[torch.Tensor],
    uncond_output: Sequence[torch.Tensor],
    guide_scale: float,
) -> list[torch.Tensor]:
    return [
        uncond.detach().to(torch.float32)
        + (cond.detach().to(torch.float32) - uncond.detach().to(torch.float32)) * float(guide_scale)
        for cond, uncond in zip(cond_output, uncond_output)
    ]


def _init_aligncache_fb2_state(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    config = dict(cfg or {})
    scale_init = _resolve_phase_mapping(_ALIGNCACHE_DEFAULT_SCALE_INIT, config.get("scale_init"), float)
    ratio_ema = {phase: 1.0 / max(float(scale_init[phase]), 1e-6) for phase in _PHASES}
    return {
        "scale": dict(scale_init),
        "ratio_ema": dict(ratio_ema),
        "last_exact_pair_idx": -1,
        "current_exec_mode": None,
        "last_probe": {},
        "pending_phase": None,
        "pending_budget": None,
        "pending_feedback_eligible": False,
        "pending_cond_draft": None,
        "pending_uncond_draft": None,
        "summary": {
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


def _reset_aligncache_fb2_pending_feedback(state: dict[str, Any]) -> None:
    state["pending_phase"] = None
    state["pending_budget"] = None
    state["pending_feedback_eligible"] = False
    state["pending_cond_draft"] = None
    state["pending_uncond_draft"] = None


def _fb_pair_age(current_pair_idx: int, last_pair_idx: int | None) -> int:
    if last_pair_idx is None or int(last_pair_idx) < 0:
        return int(current_pair_idx) + 1
    return int(current_pair_idx) - int(last_pair_idx)


def _fb_should_collect_feedback(reason: str, budget_next: float | None) -> bool:
    return budget_next is not None and str(reason) == "budget_exact"


def _fb_route_pair(
    state: dict[str, Any],
    phase: str,
    pair_idx: int,
    budget_next: float | None,
    phase_threshold: float,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    age = _fb_pair_age(pair_idx, state["last_exact_pair_idx"])
    if budget_next is None:
        return {"exec_mode": "exact_pair", "reason": "bootstrap", "age": int(age)}
    if age > int(cfg["age_cap"][phase]):
        return {"exec_mode": "exact_pair", "reason": "max_age", "age": int(age)}
    if float(budget_next) < float(phase_threshold):
        return {"exec_mode": "skip_pair", "reason": "budget_skip", "age": int(age)}
    return {"exec_mode": "exact_pair", "reason": "budget_exact", "age": int(age)}


def _fb_register_pair_decision(
    state: dict[str, Any],
    phase: str,
    *,
    exec_mode: str,
    reason: str,
    budget_next: float | None,
    pred_change_base: float | None,
    base_phase_threshold: float,
    phase_threshold: float,
    threshold_scale: float,
    ratio_ema: float,
    anchor_age: int,
    feedback_eligible: bool,
) -> None:
    state["current_exec_mode"] = str(exec_mode)
    state["pending_phase"] = str(phase)
    state["pending_budget"] = None if budget_next is None else float(budget_next)
    state["pending_feedback_eligible"] = bool(feedback_eligible)
    state["pending_cond_draft"] = None
    state["pending_uncond_draft"] = None
    state["last_probe"] = {
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
    summary_key = _ALIGNCACHE_REASON_SUMMARY_KEYS.get(str(reason))
    if summary_key is not None:
        state["summary"][summary_key] += 1


def _fb_complete_pair(state: dict[str, Any], pair_idx: int, exec_mode: str) -> None:
    anchor_age = _fb_pair_age(pair_idx, state["last_exact_pair_idx"])
    state["summary"]["max_anchor_age"] = max(int(state["summary"]["max_anchor_age"]), int(anchor_age))
    if str(exec_mode) == "exact_pair":
        state["last_exact_pair_idx"] = int(pair_idx)
        state["summary"]["exact_pairs"] += 1
    else:
        state["summary"]["skipped_pairs"] += 1
    state["current_exec_mode"] = str(exec_mode)
    if isinstance(state.get("last_probe"), dict):
        state["last_probe"]["exec_mode"] = str(exec_mode)


def _fb_predict_cond_feedback_draft(runtime: Any, raw_input: Sequence[torch.Tensor], phase: str) -> list[torch.Tensor] | None:
    draft_output = _predict_pair_output(runtime, raw_input, phase, True)
    if draft_output is None and getattr(runtime, "cond_cache", None) is not None:
        draft_output = [(current + delta).detach() for current, delta in zip(raw_input, runtime.cond_cache)]
    return None if draft_output is None else _clone_tensor_list(draft_output)


def _fb_predict_uncond_feedback_draft(
    runtime: Any,
    raw_input: Sequence[torch.Tensor],
    phase: str,
    pair_idx: int,
    cond_draft: Sequence[torch.Tensor] | None,
) -> list[torch.Tensor] | None:
    draft_output = _predict_pair_output(runtime, raw_input, phase, False)
    if draft_output is None and getattr(runtime, "uncond_cache", None) is not None:
        draft_output = [(current + delta).detach() for current, delta in zip(raw_input, runtime.uncond_cache)]
    if draft_output is None:
        return None
    anchor_output = _predict_anchor_uncond(runtime, phase, pair_idx, cond_draft, draft_output, record_event=False)
    if anchor_output is not None:
        draft_output = anchor_output
    return _clone_tensor_list(draft_output)


def _fb_update_from_exact_feedback(
    state: dict[str, Any],
    cfg: dict[str, Any],
    cond_output: Sequence[torch.Tensor] | None,
    uncond_output: Sequence[torch.Tensor] | None,
    guide_scale: float,
) -> None:
    if not bool(state.get("pending_feedback_eligible")):
        _reset_aligncache_fb2_pending_feedback(state)
        return
    phase = state.get("pending_phase")
    budget = state.get("pending_budget")
    cond_draft = state.get("pending_cond_draft")
    uncond_draft = state.get("pending_uncond_draft")
    if (
        phase is None
        or budget is None
        or cond_draft is None
        or uncond_draft is None
        or cond_output is None
        or uncond_output is None
    ):
        state["summary"]["feedback_skipped_pairs"] += 1
        _reset_aligncache_fb2_pending_feedback(state)
        return
    guided_draft = _guided_output(cond_draft, uncond_draft, guide_scale)
    guided_exact = _guided_output(cond_output, uncond_output, guide_scale)
    obs_err = _tensor_list_rel_mse(guided_draft, guided_exact)
    if obs_err is None:
        state["summary"]["feedback_skipped_pairs"] += 1
        _reset_aligncache_fb2_pending_feedback(state)
        return
    err_ratio = float(obs_err) / max(float(budget), float(cfg["score_eps"]))
    clipped_ratio = _clamp_float(err_ratio, float(cfg["ratio_clip_low"]), float(cfg["ratio_clip_high"]))
    prev_ratio_ema = float(state["ratio_ema"][phase])
    updated_ratio_ema = float(cfg["ratio_ema_decay"]) * prev_ratio_ema + (1.0 - float(cfg["ratio_ema_decay"])) * float(clipped_ratio)
    state["ratio_ema"][phase] = float(updated_ratio_ema)
    state["scale"][phase] = _clamp_float(
        1.0 / max(float(updated_ratio_ema), float(cfg["score_eps"])),
        float(cfg["scale_low"][phase]),
        float(cfg["scale_high"][phase]),
    )
    state["summary"]["feedback_updates"] += 1
    _reset_aligncache_fb2_pending_feedback(state)


def _init_pair_predictor_state(cfg: dict[str, Any], *, guide_scale: float) -> Any:
    runtime = types.SimpleNamespace()
    runtime.sample_guide_scale = float(guide_scale)
    runtime.pair_cfg = dict(cfg)
    runtime.fb_cfg = None
    runtime.fb_state = None
    runtime.cond_cache = None
    runtime.uncond_cache = None
    runtime.pending_cond = None
    runtime.pending_cond_exact = False
    runtime.anchor_state = _init_anchor_state(runtime.pair_cfg)
    runtime.aux_output = {"cond": None, "uncond": None}
    _reset_shadow_pair_state(runtime)
    return runtime


def _advance_aligncache_counter(module: Any, *, step_limit: int) -> None:
    module.cnt = (int(getattr(module, "cnt", 0)) + 1) % max(int(step_limit), 1)


def _split_cfg_batch(
    value: torch.Tensor | Sequence[torch.Tensor],
    branch_order: Sequence[str],
) -> dict[str, torch.Tensor] | None:
    if not branch_order:
        return None
    if isinstance(value, torch.Tensor):
        branch_count = len(branch_order)
        if value.shape[0] < branch_count or value.shape[0] % branch_count != 0:
            return None
        chunk = value.shape[0] // branch_count
        pieces = list(torch.split(value, chunk, dim=0))
        if len(pieces) != branch_count:
            return None
        return {branch: piece for branch, piece in zip(branch_order, pieces)}
    if isinstance(value, Sequence) and len(value) == len(branch_order):
        branch_values = {}
        for branch, item in zip(branch_order, value):
            if not isinstance(item, torch.Tensor):
                return None
            branch_values[str(branch)] = item
        return branch_values
    return None


def _merge_cfg_batch(branches: dict[str, torch.Tensor], branch_order: Sequence[str]) -> torch.Tensor | None:
    tensors = [branches.get(branch) for branch in branch_order]
    if any(tensor is None for tensor in tensors):
        return None
    return torch.cat([tensor for tensor in tensors if tensor is not None], dim=0)


def _split_opensora_primary_aux(
    output: torch.Tensor | Sequence[torch.Tensor],
    *,
    raw_input: torch.Tensor | Sequence[torch.Tensor],
) -> tuple[list[torch.Tensor], list[torch.Tensor] | None]:
    primary = _tensor_list(output) or []
    raw = _tensor_list(raw_input) or []
    if len(primary) != len(raw):
        return primary, None
    resolved_primary: list[torch.Tensor] = []
    resolved_aux: list[torch.Tensor] = []
    has_aux = False
    for value, raw_value in zip(primary, raw):
        raw_channels = int(raw_value.shape[1])
        output_channels = int(value.shape[1])
        if output_channels == raw_channels * 2:
            resolved_primary.append(value[:, :raw_channels])
            resolved_aux.append(value[:, raw_channels:])
            has_aux = True
        else:
            resolved_primary.append(value)
    return resolved_primary, resolved_aux if has_aux else None


def _compose_opensora_pred_sigma_output(
    primary_output: Sequence[torch.Tensor],
    aux_output: Sequence[torch.Tensor] | None,
) -> torch.Tensor:
    merged: list[torch.Tensor] = []
    for index, primary in enumerate(primary_output):
        if aux_output is None or index >= len(aux_output) or aux_output[index] is None:
            merged.append(primary)
            continue
        aux = aux_output[index]
        if aux.shape[0] != primary.shape[0] or aux.shape[2:] != primary.shape[2:]:
            aux = primary.new_zeros((primary.shape[0], aux.shape[1], *primary.shape[2:]))
        merged.append(torch.cat([primary, aux.to(dtype=primary.dtype)], dim=1))
    if len(merged) != 1:
        return torch.cat(merged, dim=0)
    return merged[0]


def _normalize_opensora_pair_output_for_cache(
    runtime: Any,
    *,
    branch: str,
    raw_input: torch.Tensor | Sequence[torch.Tensor],
    output: torch.Tensor | Sequence[torch.Tensor],
) -> list[torch.Tensor]:
    primary_output, aux_output = _split_opensora_primary_aux(output, raw_input=raw_input)
    runtime.aux_output[str(branch)] = _clone_tensor_list(aux_output)
    return primary_output


def reset_single_aligncache_state(module: Any, *, sample_steps: int) -> None:
    cfg = dict(getattr(module, "aligncache_cfg", {}) or {})
    pair_cfg = _pair_cfg_from_v2_cfg(dict(cfg.get("v2_cfg") or _ALIGNCACHE_V2_DEFAULTS))
    guide_scale = float(getattr(module, "sample_guide_scale", cfg.get("guide_scale") or 1.0))
    runtime = _init_pair_predictor_state(pair_cfg, guide_scale=guide_scale)
    runtime.fb_cfg = cfg
    runtime.fb_state = _init_aligncache_fb2_state(cfg)
    runtime.anchor_state = _init_anchor_state(pair_cfg)
    module.aligncache_state = runtime.fb_state
    module.aligncache_pair_state = runtime
    module.aligncache_previous_raw_input_cond = None
    module.aligncache_previous_raw_output_cond = None
    module.aligncache_prev_prev_raw_input_cond = None
    module.aligncache_k = None
    module.accumulated_error = 0.0
    module.cnt = 0
    module.num_steps = max(int(sample_steps), 1)
    module.v2_current_phase_threshold = float(cfg.get("base_thresh", 0.0) or 0.0)


def prepare_single_aligncache_step(
    module: Any,
    *,
    raw_input: torch.Tensor | Sequence[torch.Tensor],
) -> dict[str, Any] | None:
    if not bool(getattr(module, "enable_aligncache", False)):
        return None
    branch_order = tuple(getattr(module, "aligncache_branch_order", ()) or ())
    if len(branch_order) != 2:
        return {
            "kind": "noop",
            "phase": "mid",
            "step_idx": int(getattr(module, "cnt", 0)),
            "exec_mode": "exact",
            "pair_exec_mode": "exact_pair",
        }
    branches = _split_cfg_batch(raw_input, branch_order)
    if branches is None or "cond" not in branches or "uncond" not in branches:
        return {
            "kind": "noop",
            "phase": "mid",
            "step_idx": int(getattr(module, "cnt", 0)),
            "exec_mode": "exact",
            "pair_exec_mode": "exact_pair",
        }

    cfg = dict(getattr(module, "aligncache_cfg", {}) or {})
    if not cfg:
        reset_single_aligncache_state(module, sample_steps=max(int(getattr(module, "num_steps", 1) or 1), 1))
        cfg = dict(getattr(module, "aligncache_cfg", {}) or {})
    state = getattr(module, "aligncache_state")
    runtime = getattr(module, "aligncache_pair_state")
    runtime.sample_guide_scale = float(getattr(module, "sample_guide_scale", cfg.get("guide_scale") or 5.0))

    step_idx = int(getattr(module, "cnt", 0))
    total_steps = max(int(getattr(module, "num_steps", 1) or 1), 1)
    pair_cfg = _pair_cfg_from_v2_cfg(dict(cfg.get("v2_cfg") or _ALIGNCACHE_V2_DEFAULTS))
    phase = _get_phase(step_idx, total_steps, pair_cfg)
    base_phase_threshold = float(cfg["phase_thresholds"][phase])
    threshold_scale = float(state["scale"][phase])
    phase_threshold = float(base_phase_threshold) * float(threshold_scale)
    current_ratio_ema = float(state["ratio_ema"][phase])
    module.v2_current_phase_threshold = float(base_phase_threshold)

    cond_raw_tensor = branches["cond"]
    uncond_raw_tensor = branches["uncond"]
    cond_raw = _tensor_list(cond_raw_tensor) or []
    uncond_raw = _tensor_list(uncond_raw_tensor) or []

    budget_next = None
    pred_change_base = None
    if step_idx < int(cfg["cache_ret_steps"]) or step_idx >= max(total_steps - 1, 0):
        exec_mode = "exact_pair"
        reason = "warmup_tail"
        module.accumulated_error = 0.0
        anchor_age = _fb_pair_age(step_idx, state["last_exact_pair_idx"])
        feedback_eligible = False
    else:
        previous_input = getattr(module, "aligncache_previous_raw_input_cond", None)
        previous_output = getattr(module, "aligncache_previous_raw_output_cond", None)
        if previous_input is not None and previous_output is not None and getattr(module, "aligncache_k", None) is not None:
            raw_input_change = (cond_raw_tensor - previous_input).abs().mean()
            output_norm = _tensor_list_mean_abs(_tensor_list(previous_output))
            if output_norm is not None:
                pred_change_base = float(getattr(module, "aligncache_k", 0.0)) * float(raw_input_change.item()) / max(
                    float(output_norm),
                    float(cfg["score_eps"]),
                )
                budget_next = float(getattr(module, "accumulated_error", 0.0)) + float(pred_change_base)
            route = _fb_route_pair(state, phase, step_idx, budget_next, phase_threshold, cfg)
        else:
            route = {"exec_mode": "exact_pair", "reason": "bootstrap", "age": _fb_pair_age(step_idx, state["last_exact_pair_idx"])}
        anchor_age = int(route["age"])
        exec_mode = str(route["exec_mode"])
        reason = str(route["reason"])
        feedback_eligible = bool(exec_mode == "exact_pair" and _fb_should_collect_feedback(reason, budget_next))
        if exec_mode == "exact_pair":
            module.accumulated_error = 0.0
        else:
            module.accumulated_error = 0.0 if budget_next is None else float(budget_next)

    if exec_mode == "skip_pair" and (runtime.cond_cache is None or runtime.uncond_cache is None):
        exec_mode = "exact_pair"
        reason = "cache_miss_exact"
        feedback_eligible = False
        module.accumulated_error = 0.0

    _fb_register_pair_decision(
        state,
        phase,
        exec_mode=exec_mode,
        reason=reason,
        budget_next=budget_next,
        pred_change_base=pred_change_base,
        base_phase_threshold=base_phase_threshold,
        phase_threshold=phase_threshold,
        threshold_scale=threshold_scale,
        ratio_ema=current_ratio_ema,
        anchor_age=anchor_age,
        feedback_eligible=feedback_eligible,
    )

    if feedback_eligible:
        state["pending_cond_draft"] = _fb_predict_cond_feedback_draft(runtime, cond_raw, phase)
        state["pending_uncond_draft"] = _fb_predict_uncond_feedback_draft(
            runtime,
            uncond_raw,
            phase,
            step_idx,
            state["pending_cond_draft"],
        )

    module.aligncache_previous_raw_input_cond = cond_raw_tensor.detach().clone()

    if exec_mode != "skip_pair":
        return {
            "kind": "pair",
            "phase": phase,
            "step_idx": step_idx,
            "exec_mode": "exact",
            "pair_exec_mode": "exact_pair",
        }

    cond_output = _predict_pair_output(runtime, cond_raw, phase, True)
    if cond_output is None and runtime.cond_cache is not None:
        cond_output = [(value + delta).detach() for value, delta in zip(cond_raw, runtime.cond_cache)]
    uncond_output = _predict_pair_output(runtime, uncond_raw, phase, False)
    if uncond_output is None and runtime.uncond_cache is not None:
        uncond_output = [(value + delta).detach() for value, delta in zip(uncond_raw, runtime.uncond_cache)]
    if cond_output is None or uncond_output is None:
        return {
            "kind": "pair",
            "phase": phase,
            "step_idx": step_idx,
            "exec_mode": "exact",
            "pair_exec_mode": "exact_pair",
        }
    anchor_output = _predict_anchor_uncond(runtime, phase, step_idx, cond_output, uncond_output)
    if anchor_output is not None:
        uncond_output = anchor_output
    _set_v2_pending_cond_state(runtime, cond_output, False)
    predicted_output = _merge_cfg_batch(
        {
            "cond": _compose_opensora_pred_sigma_output(cond_output, runtime.aux_output.get("cond")),
            "uncond": _compose_opensora_pred_sigma_output(uncond_output, runtime.aux_output.get("uncond")),
        },
        branch_order,
    )
    if predicted_output is None:
        _clear_v2_pending_cond_state(runtime)
        return {
            "kind": "pair",
            "phase": phase,
            "step_idx": step_idx,
            "exec_mode": "exact",
            "pair_exec_mode": "exact_pair",
        }
    return {
        "kind": "pair",
        "phase": phase,
        "step_idx": step_idx,
        "exec_mode": "skip",
        "pair_exec_mode": "skip_pair",
        "predicted_output": predicted_output,
    }


def complete_single_aligncache_skip(
    module: Any,
    *,
    raw_input: torch.Tensor | Sequence[torch.Tensor],
    predicted_output: torch.Tensor | Sequence[torch.Tensor],
    decision: dict[str, Any],
) -> list[torch.Tensor]:
    if str(decision.get("kind")) != "pair":
        _advance_aligncache_counter(module, step_limit=max(int(getattr(module, "num_steps", 1) or 1), 1))
        return [predicted_output] if isinstance(predicted_output, torch.Tensor) else list(predicted_output)

    branch_order = tuple(getattr(module, "aligncache_branch_order", ()) or ())
    raw_branches = _split_cfg_batch(raw_input, branch_order)
    output_branches = _split_cfg_batch(predicted_output, branch_order)
    if raw_branches is None or output_branches is None:
        _advance_aligncache_counter(module, step_limit=max(int(getattr(module, "num_steps", 1) or 1), 1))
        return [predicted_output] if isinstance(predicted_output, torch.Tensor) else list(predicted_output)

    runtime = getattr(module, "aligncache_pair_state")
    runtime.sample_guide_scale = float(getattr(module, "sample_guide_scale", 5.0))
    cond_raw = _tensor_list(raw_branches["cond"]) or []
    uncond_raw = _tensor_list(raw_branches["uncond"]) or []
    cond_output = _normalize_opensora_pair_output_for_cache(
        runtime,
        branch="cond",
        raw_input=raw_branches["cond"],
        output=output_branches["cond"],
    )
    uncond_output = _normalize_opensora_pair_output_for_cache(
        runtime,
        branch="uncond",
        raw_input=raw_branches["uncond"],
        output=output_branches["uncond"],
    )
    runtime.cond_cache = [(u - v).detach() for u, v in zip(cond_output, cond_raw)]
    runtime.uncond_cache = [(u - v).detach() for u, v in zip(uncond_output, uncond_raw)]
    _write_pair_branch_result(runtime, cond_raw, cond_output, True, str(decision["phase"]), False)
    _write_pair_branch_result(runtime, uncond_raw, uncond_output, False, str(decision["phase"]), False)
    _clear_v2_pending_cond_state(runtime)
    _reset_aligncache_fb2_pending_feedback(getattr(module, "aligncache_state"))
    _fb_complete_pair(getattr(module, "aligncache_state"), int(decision["step_idx"]), "skip_pair")
    _advance_aligncache_counter(module, step_limit=max(int(getattr(module, "num_steps", 1) or 1), 1))
    return [predicted_output] if isinstance(predicted_output, torch.Tensor) else list(predicted_output)


def complete_single_aligncache_exact(
    module: Any,
    *,
    raw_input: torch.Tensor | Sequence[torch.Tensor],
    exact_output: torch.Tensor | Sequence[torch.Tensor],
    decision: dict[str, Any],
) -> list[torch.Tensor]:
    if str(decision.get("kind")) != "pair":
        _advance_aligncache_counter(module, step_limit=max(int(getattr(module, "num_steps", 1) or 1), 1))
        return [exact_output] if isinstance(exact_output, torch.Tensor) else list(exact_output)

    branch_order = tuple(getattr(module, "aligncache_branch_order", ()) or ())
    raw_branches = _split_cfg_batch(raw_input, branch_order)
    output_branches = _split_cfg_batch(exact_output, branch_order)
    if raw_branches is None or output_branches is None:
        _advance_aligncache_counter(module, step_limit=max(int(getattr(module, "num_steps", 1) or 1), 1))
        return [exact_output] if isinstance(exact_output, torch.Tensor) else list(exact_output)

    runtime = getattr(module, "aligncache_pair_state")
    cfg = dict(getattr(module, "aligncache_cfg", {}) or {})
    runtime.sample_guide_scale = float(getattr(module, "sample_guide_scale", cfg.get("guide_scale") or 5.0))

    cond_output = _normalize_opensora_pair_output_for_cache(
        runtime,
        branch="cond",
        raw_input=raw_branches["cond"],
        output=output_branches["cond"],
    )
    uncond_output = _normalize_opensora_pair_output_for_cache(
        runtime,
        branch="uncond",
        raw_input=raw_branches["uncond"],
        output=output_branches["uncond"],
    )
    cond_output_tensor = cond_output[0]

    previous_output = getattr(module, "aligncache_previous_raw_output_cond", None)
    previous_input = getattr(module, "aligncache_previous_raw_input_cond", None)
    prev_prev_input = getattr(module, "aligncache_prev_prev_raw_input_cond", None)
    if previous_output is not None and previous_input is not None and prev_prev_input is not None:
        output_change = (cond_output_tensor - previous_output).abs().mean()
        input_change = (previous_input - prev_prev_input).abs().mean()
        if float(input_change.item()) > 0.0:
            module.aligncache_k = float(output_change.item()) / max(float(input_change.item()), float(cfg["score_eps"]))

    module.aligncache_prev_prev_raw_input_cond = None if previous_input is None else previous_input.detach().clone()
    module.aligncache_previous_raw_output_cond = cond_output_tensor.detach().clone()

    cond_raw = _tensor_list(raw_branches["cond"]) or []
    uncond_raw = _tensor_list(raw_branches["uncond"]) or []
    runtime.cond_cache = [(u - v).detach() for u, v in zip(cond_output, cond_raw)]
    runtime.uncond_cache = [(u - v).detach() for u, v in zip(uncond_output, uncond_raw)]

    _write_pair_branch_result(runtime, cond_raw, cond_output, True, str(decision["phase"]), True)
    _write_pair_branch_result(runtime, uncond_raw, uncond_output, False, str(decision["phase"]), True)
    _update_anchor_state(runtime, str(decision["phase"]), int(decision["step_idx"]), cond_output, uncond_output, True)
    _fb_update_from_exact_feedback(
        getattr(module, "aligncache_state"),
        cfg,
        cond_output,
        uncond_output,
        runtime.sample_guide_scale,
    )
    _clear_v2_pending_cond_state(runtime)
    _fb_complete_pair(getattr(module, "aligncache_state"), int(decision["step_idx"]), "exact_pair")
    _advance_aligncache_counter(module, step_limit=max(int(getattr(module, "num_steps", 1) or 1), 1))
    return [exact_output] if isinstance(exact_output, torch.Tensor) else list(exact_output)


def _reset_hunyuan_aligncache_runtime_state(module: Any, *, sample_steps: int) -> None:
    reset_single_aligncache_state(module, sample_steps=max(int(sample_steps), 1))
    module.aligncache_previous_raw_input = None
    module.aligncache_previous_output = None
    module.aligncache_prev_prev_raw_input = None
    module.aligncache_cache = None
    module.aligncache_k = None
    module.aligncache_accumulated_error = 0.0
    module.aligncache_predicted_output = None


def _prepare_hunyuan_aligncache_single_step(
    module: Any,
    *,
    raw_input: torch.Tensor,
) -> dict[str, Any]:
    cfg = dict(getattr(module, "aligncache_cfg", {}) or {})
    if not cfg or not isinstance(getattr(module, "aligncache_state", None), dict):
        _reset_hunyuan_aligncache_runtime_state(
            module,
            sample_steps=max(int(getattr(module, "num_steps", 1) or 1), 1),
        )
        cfg = dict(getattr(module, "aligncache_cfg", {}) or {})
    state = getattr(module, "aligncache_state")

    step_idx = int(getattr(module, "cnt", 0))
    total_steps = max(int(getattr(module, "num_steps", 1) or 1), 1)
    pair_cfg = _pair_cfg_from_v2_cfg(dict(cfg.get("v2_cfg") or _ALIGNCACHE_V2_DEFAULTS))
    phase = _get_phase(step_idx, total_steps, pair_cfg)
    base_phase_threshold = float(cfg["phase_thresholds"][phase])
    threshold_scale = float(state["scale"][phase])
    phase_threshold = float(base_phase_threshold) * float(threshold_scale)
    current_ratio_ema = float(state["ratio_ema"][phase])
    module.v2_current_phase_threshold = float(base_phase_threshold)

    budget_next = None
    pred_change_base = None
    if step_idx < int(cfg["cache_ret_steps"]) or step_idx >= max(total_steps - 1, 0):
        exec_mode = "exact_pair"
        reason = "warmup_tail"
        module.aligncache_accumulated_error = 0.0
        anchor_age = _fb_pair_age(step_idx, state["last_exact_pair_idx"])
        feedback_eligible = False
    else:
        previous_input = getattr(module, "aligncache_previous_raw_input", None)
        previous_output = getattr(module, "aligncache_previous_output", None)
        if previous_input is not None and previous_output is not None and getattr(module, "aligncache_k", None) is not None:
            raw_input_change = (raw_input - previous_input).abs().mean()
            output_norm = previous_output.abs().mean()
            if float(output_norm.item()) > 0.0:
                pred_change_base = float(getattr(module, "aligncache_k", 0.0)) * float(raw_input_change.item()) / max(
                    float(output_norm.item()),
                    float(cfg["score_eps"]),
                )
                budget_next = float(getattr(module, "aligncache_accumulated_error", 0.0)) + float(pred_change_base)
            route = _fb_route_pair(state, phase, step_idx, budget_next, phase_threshold, cfg)
        else:
            route = {
                "exec_mode": "exact_pair",
                "reason": "bootstrap",
                "age": _fb_pair_age(step_idx, state["last_exact_pair_idx"]),
            }
        anchor_age = int(route["age"])
        exec_mode = str(route["exec_mode"])
        reason = str(route["reason"])
        feedback_eligible = bool(exec_mode == "exact_pair" and _fb_should_collect_feedback(reason, budget_next))
        if exec_mode == "exact_pair":
            module.aligncache_accumulated_error = 0.0
        else:
            module.aligncache_accumulated_error = 0.0 if budget_next is None else float(budget_next)

    if exec_mode == "skip_pair" and getattr(module, "aligncache_cache", None) is None:
        exec_mode = "exact_pair"
        reason = "cache_miss_exact"
        feedback_eligible = False
        module.aligncache_accumulated_error = 0.0

    _fb_register_pair_decision(
        state,
        phase,
        exec_mode=exec_mode,
        reason=reason,
        budget_next=budget_next,
        pred_change_base=pred_change_base,
        base_phase_threshold=base_phase_threshold,
        phase_threshold=phase_threshold,
        threshold_scale=threshold_scale,
        ratio_ema=current_ratio_ema,
        anchor_age=anchor_age,
        feedback_eligible=feedback_eligible,
    )

    runtime = getattr(module, "aligncache_pair_state", None)
    raw_values = [raw_input]
    module.aligncache_previous_raw_input = raw_input.detach().clone()
    module.aligncache_predicted_output = None
    if feedback_eligible and runtime is not None:
        predicted_values = _predict_single_output(runtime, raw_values, phase)
        if predicted_values is not None:
            predicted_output = predicted_values[0]
            state["pending_cond_draft"] = [predicted_output.detach().clone()]
            state["pending_uncond_draft"] = [predicted_output.detach().clone()]
            module.aligncache_predicted_output = predicted_output.detach().clone()

    if exec_mode != "skip_pair" or runtime is None:
        return {
            "kind": "single",
            "phase": phase,
            "step_idx": step_idx,
            "exec_mode": "exact",
            "pair_exec_mode": "exact_pair",
        }

    predicted_values = _predict_single_output(runtime, raw_values, phase)
    if predicted_values is None:
        return {
            "kind": "single",
            "phase": phase,
            "step_idx": step_idx,
            "exec_mode": "exact",
            "pair_exec_mode": "exact_pair",
        }
    predicted_output = predicted_values[0]
    module.aligncache_predicted_output = predicted_output.detach().clone()
    return {
        "kind": "single",
        "phase": phase,
        "step_idx": step_idx,
        "exec_mode": "skip",
        "pair_exec_mode": "skip_pair",
        "predicted_output": predicted_output,
    }


def _complete_hunyuan_aligncache_single_skip(
    module: Any,
    *,
    predicted_output: torch.Tensor,
    decision: dict[str, Any],
) -> list[torch.Tensor]:
    _reset_aligncache_fb2_pending_feedback(getattr(module, "aligncache_state"))
    _fb_complete_pair(getattr(module, "aligncache_state"), int(decision["step_idx"]), "skip_pair")
    module.aligncache_predicted_output = None
    _advance_aligncache_counter(module, step_limit=max(int(getattr(module, "num_steps", 1) or 1), 1))
    return [predicted_output]


def _complete_hunyuan_aligncache_single_exact(
    module: Any,
    *,
    raw_input: torch.Tensor,
    exact_output: torch.Tensor,
    decision: dict[str, Any],
) -> list[torch.Tensor]:
    cfg = dict(getattr(module, "aligncache_cfg", {}) or {})
    previous_output = getattr(module, "aligncache_previous_output", None)
    previous_input = getattr(module, "aligncache_previous_raw_input", None)
    prev_prev_input = getattr(module, "aligncache_prev_prev_raw_input", None)
    if previous_output is not None and previous_input is not None and prev_prev_input is not None:
        output_change = (exact_output - previous_output).abs().mean()
        input_change = (previous_input - prev_prev_input).abs().mean()
        if float(input_change.item()) > 0.0:
            module.aligncache_k = float(output_change.item()) / max(float(input_change.item()), float(cfg["score_eps"]))

    module.aligncache_prev_prev_raw_input = None if previous_input is None else previous_input.detach().clone()
    module.aligncache_previous_output = exact_output.detach().clone()
    module.aligncache_cache = (exact_output - raw_input).detach()
    runtime = getattr(module, "aligncache_pair_state", None)
    if runtime is not None:
        _write_single_branch_result(
            runtime,
            [raw_input],
            [exact_output],
            str(decision.get("phase", "mid")),
            True,
        )
    _fb_update_from_exact_feedback(
        getattr(module, "aligncache_state"),
        cfg,
        [exact_output],
        [exact_output],
        1.0,
    )
    module.aligncache_predicted_output = None
    _fb_complete_pair(getattr(module, "aligncache_state"), int(decision["step_idx"]), "exact_pair")
    _advance_aligncache_counter(module, step_limit=max(int(getattr(module, "num_steps", 1) or 1), 1))
    return [exact_output]


def prepare_hunyuan_aligncache_step(
    module: Any,
    *,
    raw_input: torch.Tensor,
) -> dict[str, Any] | None:
    if not bool(getattr(module, "enable_aligncache", False)):
        return None
    if _hunyuan_has_cfg_pair(raw_input):
        return prepare_single_aligncache_step(module, raw_input=raw_input)
    return _prepare_hunyuan_aligncache_single_step(module, raw_input=raw_input)


def complete_hunyuan_aligncache_skip(
    module: Any,
    *,
    raw_input: torch.Tensor,
    predicted_output: torch.Tensor,
    decision: dict[str, Any],
) -> list[torch.Tensor]:
    if _hunyuan_has_cfg_pair(raw_input):
        return complete_single_aligncache_skip(
            module,
            raw_input=raw_input,
            predicted_output=predicted_output,
            decision=decision,
        )
    return _complete_hunyuan_aligncache_single_skip(
        module,
        predicted_output=predicted_output,
        decision=decision,
    )


def complete_hunyuan_aligncache_exact(
    module: Any,
    *,
    raw_input: torch.Tensor,
    exact_output: torch.Tensor,
    decision: dict[str, Any],
) -> list[torch.Tensor]:
    if _hunyuan_has_cfg_pair(raw_input):
        return complete_single_aligncache_exact(
            module,
            raw_input=raw_input,
            exact_output=exact_output,
            decision=decision,
        )
    return _complete_hunyuan_aligncache_single_exact(
        module,
        raw_input=raw_input,
        exact_output=exact_output,
        decision=decision,
    )


def configure_hunyuan_aligncache(
    sampler: Any,
    *,
    method_name: str,
    method_config: dict[str, Any] | None,
    sample_steps: int,
) -> dict[str, Any] | None:
    from . import teacache

    pipeline = sampler.pipeline
    transformer = pipeline.transformer
    if not getattr(transformer, "_vbench_teacache_patch_applied", False):
        transformer.forward = types.MethodType(teacache._hunyuan_teacache_forward, transformer)
        transformer._vbench_teacache_patch_applied = True

    enabled = str(method_name) == "aligncache"
    transformer.enable_aligncache = bool(enabled)
    if enabled:
        transformer.enable_teacache = False
        transformer.enable_easycache = False
        transformer.aligncache_cfg = resolve_hunyuan_aligncache_config(
            method_config,
            sample_steps=int(sample_steps),
        )
    else:
        transformer.aligncache_cfg = dict(getattr(transformer, "aligncache_cfg", {}) or {})
    pipeline._vbench_force_cfg_pair = bool(enabled and transformer.aligncache_cfg.get("force_cfg_pair", True))
    transformer.aligncache_branch_order = ("uncond", "cond")
    _reset_hunyuan_aligncache_runtime_state(transformer, sample_steps=max(int(sample_steps), 1))
    if not enabled:
        return None
    return dict(transformer.aligncache_cfg)


def restore_hunyuan_aligncache_state(sampler: Any, *, sample_steps: int) -> None:
    pipeline = sampler.pipeline
    transformer = pipeline.transformer
    transformer.enable_aligncache = False
    transformer.aligncache_cfg = dict(getattr(transformer, "aligncache_cfg", {}) or {})
    pipeline._vbench_force_cfg_pair = False
    transformer.aligncache_branch_order = ("uncond", "cond")
    _reset_hunyuan_aligncache_runtime_state(transformer, sample_steps=max(int(sample_steps), 1))


def configure_opensora_aligncache(
    model: Any,
    *,
    enabled: bool,
    method_config: dict[str, Any] | None,
    sample_steps: int,
) -> dict[str, Any] | None:
    model.enable_aligncache = bool(enabled)
    if enabled:
        model.aligncache_cfg = resolve_opensora_aligncache_config(method_config, sample_steps=int(sample_steps))
    else:
        model.aligncache_cfg = dict(getattr(model, "aligncache_cfg", {}) or {})
    model.aligncache_branch_order = ("cond", "uncond")
    reset_single_aligncache_state(model, sample_steps=max(int(sample_steps), 1))
    if not enabled:
        return None
    return dict(model.aligncache_cfg)


def _reset_wan_pair_state(model: Any) -> None:
    model.call_idx = 0
    model.skip_budget = 0.0
    model.run_exact_pair = True
    model.motion_gain = None
    model.is_cond = True
    model.last_cond_input = None
    model.last_cond_output = None
    model.prev_cond_input = None
    model.cond_cache = None
    model.uncond_cache = None
    model.phase_base_thresh = None
    _reset_shadow_pair_state(model)
    model.anchor_state = _init_anchor_state(model.pair_cfg)
    _clear_v2_pending_cond_state(model)
    model.fb_state = _init_aligncache_fb2_state(model.fb_cfg)


def _advance_wan_counter(model: Any) -> None:
    model.call_idx += 1
    if model.call_idx >= model.total_calls:
        model.call_idx = 0
        _clear_v2_pending_cond_state(model)
        model.fb_state = _init_aligncache_fb2_state(model.fb_cfg)


def configure_wan_aligncache(
    pipeline: Any,
    *,
    method_name: str,
    method_config: dict[str, Any] | None,
    task: str,
    sample_steps: int,
) -> dict[str, Any] | None:
    model = pipeline.model
    if not hasattr(model, "_vbench_aligncache_original_forward"):
        model._vbench_aligncache_original_forward = model.forward
    enabled = str(method_name) == "aligncache"
    if enabled:
        model.forward = types.MethodType(_wan_aligncache_forward, model)
    else:
        original_forward = getattr(model, "_vbench_aligncache_original_forward", None)
        if original_forward is not None:
            model.forward = original_forward
    model.enable_aligncache = bool(enabled)
    requested_config = dict(method_config or {})
    resolved_guide_scale = float(
        requested_config.get(
            "sample_guide_scale",
            requested_config.get("guide_scale", getattr(model, "sample_guide_scale", 5.0)),
        )
    )
    resolved = resolve_wan_aligncache_config(
        task=str(task),
        method_config=method_config,
        sample_steps=int(sample_steps),
    )
    resolved["guide_scale"] = float(resolved_guide_scale)
    model.aligncache_cfg = dict(resolved)
    model.total_calls = max(int(sample_steps) * 2, 1)
    model.warmup_calls = int(resolved["warmup_pairs"]) * 2
    model.tail_call_start = max(int(sample_steps) * 2 - 2, 0)
    model.base_thresh = float(resolved["base_thresh"])
    model.phase_skip_thresholds = dict(resolved["phase_thresholds"])
    model.sample_guide_scale = float(resolved_guide_scale)
    model.fb_cfg = dict(resolved)
    model.pair_cfg = _pair_cfg_from_v2_cfg(dict(resolved["v2_cfg"]))
    _reset_wan_pair_state(model)
    if not enabled:
        return None
    return dict(resolved)


def restore_wan_aligncache_state(pipeline: Any) -> None:
    model = pipeline.model
    original_forward = getattr(model, "_vbench_aligncache_original_forward", None)
    if original_forward is not None:
        model.forward = original_forward
    model.enable_aligncache = False
    model.aligncache_cfg = dict(getattr(model, "aligncache_cfg", {}) or {})
    model.total_calls = max(int(getattr(model, "total_calls", 1) or 1), 1)
    model.warmup_calls = 0
    model.tail_call_start = max(model.total_calls - 2, 0)
    model.base_thresh = 0.0
    model.phase_skip_thresholds = {phase: 0.0 for phase in _PHASES}
    model.pair_cfg = _pair_cfg_from_v2_cfg(dict(_ALIGNCACHE_V2_DEFAULTS))
    model.fb_cfg = {
        "score_eps": 1e-6,
        "ratio_ema_decay": 0.90,
        "ratio_clip_low": 0.85,
        "ratio_clip_high": 1.25,
        "scale_init": dict(_ALIGNCACHE_DEFAULT_SCALE_INIT),
        "scale_low": dict(_ALIGNCACHE_DEFAULT_SCALE_LOW),
        "scale_high": dict(_ALIGNCACHE_DEFAULT_SCALE_HIGH),
        "age_cap": dict(_ALIGNCACHE_DEFAULT_AGE_CAP),
    }
    _reset_wan_pair_state(model)


def _wan_aligncache_forward(
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

    raw_input = _clone_tensor_list(x)
    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)
    if y is not None:
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

    self.is_cond = bool(int(getattr(self, "call_idx", 0)) % 2 == 0)
    pair_idx = int(getattr(self, "call_idx", 0)) // 2
    total_pairs = max(int(getattr(self, "total_calls", 1)) // 2, 1)
    phase = _get_phase(pair_idx, total_pairs, self.pair_cfg)
    state = self.fb_state
    base_phase_threshold = float(self.phase_skip_thresholds[phase])
    threshold_scale = float(state["scale"][phase])
    current_phase_threshold = float(base_phase_threshold) * float(threshold_scale)
    current_ratio_ema = float(state["ratio_ema"][phase])
    self.phase_base_thresh = float(base_phase_threshold)

    if self.is_cond:
        pred_change_base = None
        budget_next = None
        anchor_age = _fb_pair_age(pair_idx, state["last_exact_pair_idx"])
        exec_mode = "exact_pair"
        feedback_eligible = False
        if self.call_idx < self.warmup_calls or self.call_idx >= self.tail_call_start:
            self.run_exact_pair = True
            self.skip_budget = 0.0
            decision_reason = "warmup_tail"
        else:
            if self.last_cond_input is not None and self.last_cond_output is not None:
                raw_input_change = torch.cat(
                    [(current - previous).flatten() for current, previous in zip(raw_input, self.last_cond_input)],
                    dim=0,
                ).abs().mean()
                output_norm = _tensor_list_mean_abs(self.last_cond_output)
                if output_norm is not None and self.motion_gain is not None:
                    motion = float(raw_input_change.item()) / max(float(output_norm), float(self.fb_cfg["score_eps"]))
                    pred_change_base = float(self.motion_gain) * float(motion)
                    budget_next = float(self.skip_budget) + float(pred_change_base)
            route = _fb_route_pair(state, phase, pair_idx, budget_next, current_phase_threshold, self.fb_cfg)
            anchor_age = int(route["age"])
            exec_mode = str(route["exec_mode"])
            decision_reason = str(route["reason"])
            self.run_exact_pair = exec_mode == "exact_pair"
            feedback_eligible = bool(self.run_exact_pair and _fb_should_collect_feedback(decision_reason, budget_next))
            if self.run_exact_pair:
                self.skip_budget = 0.0
            else:
                self.skip_budget = 0.0 if budget_next is None else float(budget_next)

        if exec_mode == "skip_pair" and (self.cond_cache is None or self.uncond_cache is None):
            self.run_exact_pair = True
            exec_mode = "exact_pair"
            feedback_eligible = False
            decision_reason = "cache_miss_exact"
            self.skip_budget = 0.0

        _fb_register_pair_decision(
            state,
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
            state["pending_cond_draft"] = _fb_predict_cond_feedback_draft(self, raw_input, phase)
        self.last_cond_input = _clone_tensor_list(raw_input)

    if self.is_cond and not self.run_exact_pair and self.cond_cache is not None:
        pred_output = _predict_pair_output(self, raw_input, phase, True)
        if pred_output is None:
            pred_output = [(value + delta).detach() for value, delta in zip(raw_input, self.cond_cache)]
        _set_v2_pending_cond_state(self, pred_output, False)
        _write_pair_branch_result(self, raw_input, pred_output, True, phase, False)
        _advance_wan_counter(self)
        return [value.float() for value in pred_output]

    pair_mode = state.get("current_exec_mode", "exact_pair")
    if (not self.is_cond) and pair_mode == "skip_pair" and self.uncond_cache is not None:
        pred_output = _predict_pair_output(self, raw_input, phase, False)
        if pred_output is None:
            pred_output = [(value + delta).detach() for value, delta in zip(raw_input, self.uncond_cache)]
        anchor_output = _predict_anchor_uncond(self, phase, pair_idx, getattr(self, "pending_cond", None), pred_output)
        if anchor_output is not None:
            pred_output = anchor_output
        _write_pair_branch_result(self, raw_input, pred_output, False, phase, False)
        _clear_v2_pending_cond_state(self)
        _fb_complete_pair(state, pair_idx, "skip_pair")
        _reset_aligncache_fb2_pending_feedback(state)
        _advance_wan_counter(self)
        return [value.float() for value in pred_output]

    if (not self.is_cond) and self.run_exact_pair and bool(state.get("pending_feedback_eligible")) and state["pending_uncond_draft"] is None:
        state["pending_uncond_draft"] = _fb_predict_uncond_feedback_draft(
            self,
            raw_input,
            phase,
            pair_idx,
            state.get("pending_cond_draft"),
        )

    x = [self.patch_embedding(value.unsqueeze(0)) for value in x]
    grid_sizes = torch.stack([torch.tensor(value.shape[2:], dtype=torch.long) for value in x])
    x = [value.flatten(2).transpose(1, 2) for value in x]
    seq_lens = torch.tensor([value.size(1) for value in x], dtype=torch.long)
    assert seq_lens.max() <= seq_len
    x = torch.cat([torch.cat([value, value.new_zeros(1, seq_len - value.size(1), value.size(2))], dim=1) for value in x], dim=0)

    with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).float())
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))
        assert e.dtype == torch.float32 and e0.dtype == torch.float32

    context_lens = None
    context = self.text_embedding(
        torch.stack([torch.cat([value, value.new_zeros(self.text_len - value.size(0), value.size(1))]) for value in context])
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

    if self.is_cond:
        if self.last_cond_output is not None:
            output_change = torch.cat(
                [(current - previous).flatten() for current, previous in zip(output, self.last_cond_output)],
                dim=0,
            ).abs().mean()
            if self.prev_cond_input is not None:
                input_change = torch.cat(
                    [(current - previous).flatten() for current, previous in zip(self.last_cond_input, self.prev_cond_input)],
                    dim=0,
                ).abs().mean()
                self.motion_gain = float(output_change.item()) / max(float(input_change.item()), float(self.fb_cfg["score_eps"]))
        self.prev_cond_input = getattr(self, "last_cond_input", None)
        self.last_cond_output = _clone_tensor_list(output)
        self.cond_cache = [current.detach() - previous.detach() for current, previous in zip(output, raw_input)]
        _set_v2_pending_cond_state(self, output, True)
        _write_pair_branch_result(self, raw_input, output, True, phase, True)
    else:
        self.uncond_cache = [current.detach() - previous.detach() for current, previous in zip(output, raw_input)]
        _write_pair_branch_result(self, raw_input, output, False, phase, True)
        _update_anchor_state(self, phase, pair_idx, getattr(self, "pending_cond", None), output, getattr(self, "pending_cond_exact", False))
        _fb_update_from_exact_feedback(state, self.fb_cfg, getattr(self, "pending_cond", None), output, self.sample_guide_scale)
        _clear_v2_pending_cond_state(self)
        _fb_complete_pair(state, pair_idx, "exact_pair")

    _advance_wan_counter(self)
    return [value.float() for value in output]


def _method_config_from_args(args):
    return {
        "base_thresh": args.base_thresh,
        "warmup_pairs": args.warmup_pairs,
        "phase_thresh_early": args.phase_thresh_early,
        "phase_thresh_mid": args.phase_thresh_mid,
        "phase_thresh_late": args.phase_thresh_late,
        "phase_scale_early": args.phase_scale_early,
        "phase_scale_mid": args.phase_scale_mid,
        "phase_scale_late": args.phase_scale_late,
        "fb_ratio_ema_decay": args.fb_ratio_ema_decay,
        "fb_ratio_clip_low": args.fb_ratio_clip_low,
        "fb_ratio_clip_high": args.fb_ratio_clip_high,
        "fb_age_cap_early": args.fb_age_cap_early,
        "fb_age_cap_mid": args.fb_age_cap_mid,
        "fb_age_cap_late": args.fb_age_cap_late,
        "fb_scale_init_early": args.fb_scale_init_early,
        "fb_scale_init_mid": args.fb_scale_init_mid,
        "fb_scale_init_late": args.fb_scale_init_late,
        "fb_scale_low_early": args.fb_scale_low_early,
        "fb_scale_low_mid": args.fb_scale_low_mid,
        "fb_scale_low_late": args.fb_scale_low_late,
        "fb_scale_high_early": args.fb_scale_high_early,
        "fb_scale_high_mid": args.fb_scale_high_mid,
        "fb_scale_high_late": args.fb_scale_high_late,
        "pair_enable": args.pair_enable,
        "pair_aggr": args.pair_aggr,
        "pair_second_order": args.pair_second_order,
        "pair_online_scale": args.pair_online_scale,
        "pair_stride": args.pair_stride,
        "pair_ratio_ema": args.pair_ratio_ema,
        "pair_dir_ema": args.pair_dir_ema,
        "pair_beta_mid": args.pair_beta_mid,
        "pair_beta_late": args.pair_beta_late,
        "pair_ratio_tol_mid": args.pair_ratio_tol_mid,
        "pair_ratio_tol_late": args.pair_ratio_tol_late,
        "pair_dir_min_mid": args.pair_dir_min_mid,
        "pair_dir_min_late": args.pair_dir_min_late,
        "pair_scale_ema": args.pair_scale_ema,
        "pair_scale_low": args.pair_scale_low,
        "pair_scale_high": args.pair_scale_high,
        "anchor_enable": args.anchor_enable,
        "anchor_exact_early": args.anchor_exact_early,
        "anchor_age_mid": args.anchor_age_mid,
        "anchor_age_late": args.anchor_age_late,
        "anchor_cond_cap_mid": args.anchor_cond_cap_mid,
        "anchor_cond_cap_late": args.anchor_cond_cap_late,
        "anchor_alpha_early": args.anchor_alpha_early,
        "anchor_alpha_mid": args.anchor_alpha_mid,
        "anchor_alpha_late": args.anchor_alpha_late,
        "anchor_alpha_ema": args.anchor_alpha_ema,
        "anchor_blend_mid": args.anchor_blend_mid,
        "anchor_blend_late": args.anchor_blend_late,
        "anchor_diff_cap_mid": args.anchor_diff_cap_mid,
        "anchor_diff_cap_late": args.anchor_diff_cap_late,
        "sample_guide_scale": args.sample_guide_scale,
    }


def _setup_aligncache(pipeline, args):
    resolved = configure_wan_aligncache(
        pipeline,
        method_name="aligncache",
        method_config=_method_config_from_args(args),
        task=args.task,
        sample_steps=int(args.sample_steps),
    )
    logging.info(
        "AlignCache: phase_thresholds=%s age_cap=%s ratio_ema_decay=%.3f ratio_clip=[%.2f, %.2f]",
        resolved["phase_thresholds"],
        resolved["age_cap"],
        float(resolved["ratio_ema_decay"]),
        float(resolved["ratio_clip_low"]),
        float(resolved["ratio_clip_high"]),
    )
    logging.info(
        "AlignCache predictor: beta_mid=%.2f beta_late=%.2f online_scale=%s anchor=%s",
        resolved["pair_cfg"]["pair_beta_mid"],
        resolved["pair_cfg"]["pair_beta_late"],
        resolved["pair_cfg"]["pair_online_scale"],
        resolved["pair_cfg"]["anchor_enable"],
    )
    return resolved


def _log_aligncache_summary(model):
    state = getattr(model, "fb_state", None)
    if state is None:
        return
    logging.info("AlignCache stats: %s", dict(state.get("stats", {})))
    logging.info("AlignCache threshold scale: %s", dict(state.get("scale", {})))
    logging.info("AlignCache ratio EMA: %s", dict(state.get("ratio_ema", {})))


def _log_anchor_summary(model):
    state = getattr(model, "anchor_state", None)
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
        type=lambda value: parse_task_name_arg(value, allowed_tasks=_WAN_SUPPORTED_TASKS),
        default="t2v-1.3B",
        choices=sorted(_WAN_SUPPORTED_TASKS),
        help="The task to run.",
    )
    parser.add_argument("--size", type=str, default="832*480", choices=list(SIZE_CONFIGS.keys()), help="Output size.")
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
    parser.add_argument("--base_thresh", "--thresh", dest="base_thresh", type=float, default=0.065, help="Base threshold used by AlignCache.")
    parser.add_argument("--warmup_pairs", "--ret_steps", dest="warmup_pairs", default=12, type=int, help="Warm-up exact pairs before AlignCache can skip.")
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
        _setup_aligncache(pipeline, args)
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
        _log_aligncache_summary(pipeline.model)
        _log_anchor_summary(pipeline.model)

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
        _setup_aligncache(pipeline, args)
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
        _log_aligncache_summary(pipeline.model)
        _log_anchor_summary(pipeline.model)

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
