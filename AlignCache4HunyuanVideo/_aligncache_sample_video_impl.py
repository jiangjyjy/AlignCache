from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from loguru import logger


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aligncache_runtime import aligncache  # noqa: E402
from hyvideo.config import parse_args as parse_hunyuan_args  # noqa: E402
from hyvideo.inference import HunyuanVideoSampler  # noqa: E402
from hyvideo.utils.file_utils import save_videos_grid  # noqa: E402


def _str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value!r}")


def _parse_args(*, enable_feedback: bool) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--base-thresh", type=float, default=0.025, help="AlignCache base threshold.")
    parser.add_argument("--warmup-pairs", type=int, default=8, help="Warm-up exact pairs before skipping.")
    parser.add_argument(
        "--phase-scale-profile",
        type=str,
        default=None,
        choices=["alignbase", "balanced"],
        help="Optional preset for phase threshold scaling.",
    )
    parser.add_argument("--pair-enable", type=_str2bool, default=True, help="Enable pair predictor.")
    parser.add_argument("--pair-aggr", type=float, default=0.0, help="Pair predictor aggressiveness.")
    parser.add_argument("--pair-stride", type=int, default=64, help="Pair predictor subsample stride.")
    parser.add_argument("--anchor-enable", type=_str2bool, default=True, help="Enable anchor predictor.")
    parser.add_argument(
        "--summary-json",
        type=str,
        default=None,
        help="Optional path to dump the resolved AlignCache configuration as JSON.",
    )
    if enable_feedback:
        parser.add_argument("--fb-ratio-ema-decay", type=float, default=0.90, help="Feedback EMA decay.")
        parser.add_argument("--fb-ratio-clip-low", type=float, default=0.85, help="Feedback ratio clip low.")
        parser.add_argument("--fb-ratio-clip-high", type=float, default=1.25, help="Feedback ratio clip high.")

    aligncache_args, remaining = parser.parse_known_args()
    args = parse_hunyuan_args(args=remaining)
    args.aligncache_args = aligncache_args
    return args


def _build_method_config(parsed: argparse.Namespace, *, enable_feedback: bool) -> dict[str, object]:
    method_config: dict[str, object] = {
        "feedback_enabled": bool(enable_feedback),
        "base_thresh": float(parsed.base_thresh),
        "warmup_pairs": int(parsed.warmup_pairs),
        "pair_enable": bool(parsed.pair_enable),
        "pair_aggr": float(parsed.pair_aggr),
        "pair_stride": int(parsed.pair_stride),
        "anchor_enable": bool(parsed.anchor_enable),
    }
    if parsed.phase_scale_profile:
        method_config["phase_scale_profile"] = str(parsed.phase_scale_profile)
    if enable_feedback:
        method_config["fb_ratio_ema_decay"] = float(parsed.fb_ratio_ema_decay)
        method_config["fb_ratio_clip_low"] = float(parsed.fb_ratio_clip_low)
        method_config["fb_ratio_clip_high"] = float(parsed.fb_ratio_clip_high)
    return method_config


def _save_outputs(
    *,
    outputs: dict,
    save_path: str,
    method_tag: str,
) -> None:
    if "LOCAL_RANK" in os.environ and int(os.environ["LOCAL_RANK"]) != 0:
        return
    os.makedirs(save_path, exist_ok=True)
    samples = outputs["samples"]
    for index, sample in enumerate(samples):
        sample = sample.unsqueeze(0)
        timestamp = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H-%M-%S")
        prompt_stub = outputs["prompts"][index][:100].replace("/", "").replace("\n", " ").strip()
        prompt_stub = "_".join(prompt_stub.split()) or "sample"
        output_path = (
            Path(save_path)
            / f"{timestamp}_{method_tag}_seed{outputs['seeds'][index]}_{prompt_stub}.mp4"
        )
        save_videos_grid(sample, str(output_path), fps=24)
        logger.info("Sample saved to {}", output_path)


def main(*, enable_feedback: bool) -> None:
    args = _parse_args(enable_feedback=enable_feedback)
    aligncache_args = args.aligncache_args
    method_config = _build_method_config(aligncache_args, enable_feedback=enable_feedback)
    method_tag = "aligncache_fb" if enable_feedback else "aligncache"

    models_root_path = Path(args.model_base)
    if not models_root_path.exists():
        raise ValueError(f"`model_base` does not exist: {models_root_path}")

    os.makedirs(args.save_path, exist_ok=True)
    sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=args)
    resolved_config = aligncache.configure_hunyuan_aligncache(
        sampler,
        method_name="aligncache",
        method_config=method_config,
        sample_steps=int(args.infer_steps),
    )
    sampler.pipeline.transformer.sample_guide_scale = float(args.cfg_scale)
    logger.info("{} resolved config: {}", method_tag, resolved_config)

    if aligncache_args.summary_json:
        import json

        summary_path = Path(aligncache_args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(resolved_config, indent=2), encoding="utf-8")

    outputs = sampler.predict(
        prompt=args.prompt,
        height=args.video_size[0],
        width=args.video_size[1],
        video_length=args.video_length,
        seed=args.seed,
        negative_prompt=args.neg_prompt,
        infer_steps=args.infer_steps,
        guidance_scale=args.cfg_scale,
        num_videos_per_prompt=args.num_videos,
        flow_shift=args.flow_shift,
        batch_size=args.batch_size,
        embedded_guidance_scale=args.embedded_cfg_scale,
    )

    _save_outputs(
        outputs=outputs,
        save_path=args.save_path,
        method_tag=method_tag,
    )
