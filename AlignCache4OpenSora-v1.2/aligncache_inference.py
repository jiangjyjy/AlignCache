from __future__ import annotations

import types
from collections.abc import Sequence
from typing import Any

import torch


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
_HUNYUAN_ALIGNCACHE_PREDICTOR_DEFAULTS = {"early_end_ratio": 0.18, "late_start_ratio": 0.72}
_OPENSORA_ALIGNCACHE_PHASE_THRESHOLD_SCALES = {"early": 0.95, "mid": 1.15, "late": 1.00}
_OPENSORA_ALIGNCACHE_AGE_CAP = {"early": 5, "mid": 5, "late": 4}
_OPENSORA_ALIGNCACHE_SCALE_HIGH = {"early": 1.03, "mid": 1.10, "late": 1.05}
_OPENSORA_TEACACHE_COEFFICIENTS = [2.17546007e02, -1.18329252e02, 2.68662585e01, -4.59364272e-02, 4.84426240e-02]

_ALIGNCACHE_PREDICTOR_DEFAULTS = {
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
    "anchor_enable": True,
    "anchor_exact_early": True,
    "anchor_age_mid": 1,
    "anchor_age_late": 2,
    "anchor_cond_cap_mid": 0.0050,
    "anchor_cond_cap_late": 0.0200,
    "anchor_alpha_early": 1.279282,
    "anchor_alpha_mid": 1.018750,
    "anchor_alpha_late": 0.972800,
    "anchor_alpha_ema": 0.85,
    "anchor_blend_mid": 0.35,
    "anchor_blend_late": 0.20,
    "anchor_diff_cap_mid": 0.0200,
    "anchor_diff_cap_late": 0.0600,
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


def _require_non_negative_float(value: float | None, *, label: str) -> None:
    if value is not None and value < 0:
        raise ValueError(f"{label} must be non-negative when provided. Received {value!r}.")


def _require_non_negative_int(value: int | None, *, label: str) -> None:
    if value is not None and value < 0:
        raise ValueError(f"{label} must be non-negative when provided. Received {value!r}.")


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


def _apply_predictor_aggressiveness(cfg: dict[str, Any]) -> dict[str, Any]:
    resolved = dict(cfg)
    aggressiveness = float(resolved.get("aggressiveness") or 0.0)
    if abs(aggressiveness) <= 1e-8:
        return resolved
    beta_scale = _clamp_float(1.0 + 0.60 * aggressiveness, 0.50, 1.35)
    blend_scale = _clamp_float(1.0 + 0.50 * aggressiveness, 0.50, 1.40)
    clamp_delta = 0.08 * aggressiveness
    resolved["beta_mid"] = float(resolved["beta_mid"]) * beta_scale
    resolved["beta_late"] = float(resolved["beta_late"]) * beta_scale
    resolved["anchor_blend_mid"] = float(resolved["anchor_blend_mid"]) * blend_scale
    resolved["anchor_blend_late"] = float(resolved["anchor_blend_late"]) * blend_scale
    resolved["scale_clamp_low"] = max(0.60, float(resolved["scale_clamp_low"]) - clamp_delta)
    resolved["scale_clamp_high"] = max(
        float(resolved["scale_clamp_low"]),
        float(resolved["scale_clamp_high"]) + clamp_delta,
    )
    return resolved


def _coerce_aligncache_predictor_cfg_input(value: Any) -> dict[str, Any]:
    source = dict(value or {})
    resolved = dict(_ALIGNCACHE_PREDICTOR_DEFAULTS)
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
        "anchor_enable": ("anchor_enable",),
        "anchor_exact_early": ("anchor_exact_early",),
        "anchor_age_mid": ("anchor_age_mid",),
        "anchor_age_late": ("anchor_age_late",),
        "anchor_cond_cap_mid": ("anchor_cond_cap_mid",),
        "anchor_cond_cap_late": ("anchor_cond_cap_late",),
        "anchor_alpha_early": ("anchor_alpha_early",),
        "anchor_alpha_mid": ("anchor_alpha_mid",),
        "anchor_alpha_late": ("anchor_alpha_late",),
        "anchor_alpha_ema": ("anchor_alpha_ema",),
        "anchor_blend_mid": ("anchor_blend_mid",),
        "anchor_blend_late": ("anchor_blend_late",),
        "anchor_diff_cap_mid": ("anchor_diff_cap_mid",),
        "anchor_diff_cap_late": ("anchor_diff_cap_late",),
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
        if isinstance(_ALIGNCACHE_PREDICTOR_DEFAULTS[key], bool):
            resolved[key] = bool(_coerce_optional_bool(value_found, label=f"aligncache.predictor.{key}"))
        elif isinstance(_ALIGNCACHE_PREDICTOR_DEFAULTS[key], int):
            resolved[key] = int(_coerce_optional_int(value_found, label=f"aligncache.predictor.{key}"))
        else:
            resolved[key] = float(_coerce_optional_float(value_found, label=f"aligncache.predictor.{key}"))
    return _apply_predictor_aggressiveness(resolved)


def _pair_cfg_from_predictor_cfg(predictor_cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "enable": bool(predictor_cfg["enable"]),
        "early_end_ratio": float(predictor_cfg["early_end_ratio"]),
        "late_start_ratio": float(predictor_cfg["late_start_ratio"]),
        "stride": int(predictor_cfg["subsample_stride"]),
        "ratio_ema": float(predictor_cfg["qc_ratio_ema_decay"]),
        "dir_ema": float(predictor_cfg["qc_dir_ema_decay"]),
        "beta_mid": float(predictor_cfg["beta_mid"]),
        "beta_late": float(predictor_cfg["beta_late"]),
        "ratio_tol_mid": float(predictor_cfg["ratio_tol_mid"]),
        "ratio_tol_late": float(predictor_cfg["ratio_tol_late"]),
        "dir_min_mid": float(predictor_cfg["dir_min_mid"]),
        "dir_min_late": float(predictor_cfg["dir_min_late"]),
        "second_order": bool(predictor_cfg["use_second_order"]),
        "online_scale": bool(predictor_cfg["use_online_scale"]),
        "scale_ema": float(predictor_cfg["scale_ema_decay"]),
        "scale_low": float(predictor_cfg["scale_clamp_low"]),
        "scale_high": float(predictor_cfg["scale_clamp_high"]),
        "anchor_enable": bool(predictor_cfg["anchor_enable"]),
        "anchor_exact_early": bool(predictor_cfg["anchor_exact_early"]),
        "anchor_age_mid": int(predictor_cfg["anchor_age_mid"]),
        "anchor_age_late": int(predictor_cfg["anchor_age_late"]),
        "anchor_cond_cap_mid": float(predictor_cfg["anchor_cond_cap_mid"]),
        "anchor_cond_cap_late": float(predictor_cfg["anchor_cond_cap_late"]),
        "anchor_alpha_early": float(predictor_cfg["anchor_alpha_early"]),
        "anchor_alpha_mid": float(predictor_cfg["anchor_alpha_mid"]),
        "anchor_alpha_late": float(predictor_cfg["anchor_alpha_late"]),
        "anchor_alpha_ema": float(predictor_cfg["anchor_alpha_ema"]),
        "anchor_blend_mid": float(predictor_cfg["anchor_blend_mid"]),
        "anchor_blend_late": float(predictor_cfg["anchor_blend_late"]),
        "anchor_diff_cap_mid": float(predictor_cfg["anchor_diff_cap_mid"]),
        "anchor_diff_cap_late": float(predictor_cfg["anchor_diff_cap_late"]),
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
    default_predictor_cfg_overrides: dict[str, Any] | None = None,
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
    predictor_input = raw.get("predictor_cfg") or raw.get("pair_cfg") or raw.get("predictor") or raw
    if default_predictor_cfg_overrides:
        merged_predictor_input = dict(default_predictor_cfg_overrides)
        if isinstance(predictor_input, dict):
            merged_predictor_input.update(predictor_input)
        predictor_input = merged_predictor_input
    predictor_cfg = _coerce_aligncache_predictor_cfg_input(predictor_input)
    pair_cfg = _pair_cfg_from_predictor_cfg(predictor_cfg)
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
        "predictor_cfg": dict(predictor_cfg),
        "pair_cfg": {
            "pair_enable": bool(pair_cfg["enable"]),
            "enable": bool(pair_cfg["enable"]),
            "pair_aggr": float(predictor_cfg["aggressiveness"]),
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
        "predictor": dict(predictor_cfg),
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
        default_predictor_cfg_overrides=_HUNYUAN_ALIGNCACHE_PREDICTOR_DEFAULTS,
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


def _clear_pair_pending_cond_state(runtime: Any) -> None:
    runtime.pending_cond = None
    runtime.pending_cond_exact = False


def _set_pair_pending_cond_state(runtime: Any, output: Sequence[torch.Tensor], exact_pair: bool) -> None:
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


def _init_aligncache_feedback_state(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
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


def _reset_aligncache_pending_feedback(state: dict[str, Any]) -> None:
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
        _reset_aligncache_pending_feedback(state)
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
        _reset_aligncache_pending_feedback(state)
        return
    guided_draft = _guided_output(cond_draft, uncond_draft, guide_scale)
    guided_exact = _guided_output(cond_output, uncond_output, guide_scale)
    obs_err = _tensor_list_rel_mse(guided_draft, guided_exact)
    if obs_err is None:
        state["summary"]["feedback_skipped_pairs"] += 1
        _reset_aligncache_pending_feedback(state)
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
    _reset_aligncache_pending_feedback(state)


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
    pair_cfg = _pair_cfg_from_predictor_cfg(dict(cfg.get("predictor_cfg") or _ALIGNCACHE_PREDICTOR_DEFAULTS))
    guide_scale = float(getattr(module, "sample_guide_scale", cfg.get("guide_scale") or 1.0))
    runtime = _init_pair_predictor_state(pair_cfg, guide_scale=guide_scale)
    runtime.fb_cfg = cfg
    runtime.fb_state = _init_aligncache_feedback_state(cfg)
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
    module.aligncache_current_phase_threshold = float(cfg.get("base_thresh", 0.0) or 0.0)


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
    pair_cfg = _pair_cfg_from_predictor_cfg(dict(cfg.get("predictor_cfg") or _ALIGNCACHE_PREDICTOR_DEFAULTS))
    phase = _get_phase(step_idx, total_steps, pair_cfg)
    base_phase_threshold = float(cfg["phase_thresholds"][phase])
    threshold_scale = float(state["scale"][phase])
    phase_threshold = float(base_phase_threshold) * float(threshold_scale)
    current_ratio_ema = float(state["ratio_ema"][phase])
    module.aligncache_current_phase_threshold = float(base_phase_threshold)

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
    _set_pair_pending_cond_state(runtime, cond_output, False)
    predicted_output = _merge_cfg_batch(
        {
            "cond": _compose_opensora_pred_sigma_output(cond_output, runtime.aux_output.get("cond")),
            "uncond": _compose_opensora_pred_sigma_output(uncond_output, runtime.aux_output.get("uncond")),
        },
        branch_order,
    )
    if predicted_output is None:
        _clear_pair_pending_cond_state(runtime)
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
    _clear_pair_pending_cond_state(runtime)
    _reset_aligncache_pending_feedback(getattr(module, "aligncache_state"))
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
    _clear_pair_pending_cond_state(runtime)
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
    pair_cfg = _pair_cfg_from_predictor_cfg(dict(cfg.get("predictor_cfg") or _ALIGNCACHE_PREDICTOR_DEFAULTS))
    phase = _get_phase(step_idx, total_steps, pair_cfg)
    base_phase_threshold = float(cfg["phase_thresholds"][phase])
    threshold_scale = float(state["scale"][phase])
    phase_threshold = float(base_phase_threshold) * float(threshold_scale)
    current_ratio_ema = float(state["ratio_ema"][phase])
    module.aligncache_current_phase_threshold = float(base_phase_threshold)

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
    _reset_aligncache_pending_feedback(getattr(module, "aligncache_state"))
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


aligncache = types.SimpleNamespace(
    configure_opensora_aligncache=configure_opensora_aligncache,
    prepare_single_aligncache_step=prepare_single_aligncache_step,
    complete_single_aligncache_skip=complete_single_aligncache_skip,
    complete_single_aligncache_exact=complete_single_aligncache_exact,
    reset_single_aligncache_state=reset_single_aligncache_state,
)


import gc
import types

import numpy as np
import torch.distributed as dist
from einops import rearrange


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


def resolve_opensora_teacache_config(method_config: dict[str, Any] | None) -> dict[str, Any]:
    raw = dict(method_config or {})
    rel_l1_thresh = _coerce_optional_float(
        raw.get("rel_l1_thresh", raw.get("teacache_rel_l1_thresh", raw.get("teacache_thresh"))),
        label="opensora.teacache.rel_l1_thresh",
    )
    _require_non_negative_float(rel_l1_thresh, label="opensora.teacache.rel_l1_thresh")
    return {
        "rel_l1_thresh": float(0.10 if rel_l1_thresh is None else rel_l1_thresh),
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
        "easycache_thresh": float(0.08 if thresh is None else thresh),
        "ret_steps": int(5 if ret_steps is None else ret_steps),
    }


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
        resolved = resolve_opensora_teacache_config(method_config)
        model.enable_teacache = True
        model.rel_l1_thresh = float(resolved["rel_l1_thresh"])
        model.enable_pab = False
        return dict(resolved)

    if resolved_method_name == "easycache":
        resolved = resolve_opensora_easycache_config(method_config)
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


def _prepare_aligncache_state_for_new_sampling_sequence(
    model: Any,
    *,
    timestep: Any,
    all_timesteps: Any,
) -> None:
    if not getattr(model, "enable_aligncache", False):
        return
    if int(getattr(model, "cnt", 0)) != 0:
        return

    current_timestep = _extract_timestep_value(timestep)
    timestep_sequence = _normalize_timestep_sequence(all_timesteps)
    if timestep_sequence and current_timestep is not None and current_timestep != timestep_sequence[0]:
        return

    aligncache.reset_single_aligncache_state(
        model,
        sample_steps=max(int(getattr(model, "num_steps", 1) or 1), 1),
    )


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
    _prepare_aligncache_state_for_new_sampling_sequence(
        self,
        timestep=timestep,
        all_timesteps=all_timesteps,
    )
    dtype = self.x_embedder.proj.weight.dtype
    aligncache_enabled = bool(getattr(self, "enable_aligncache", False))
    easycache_enabled = bool(getattr(self, "enable_easycache", False))
    raw_input = x.clone() if (easycache_enabled or aligncache_enabled) else None
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
    if aligncache_enabled and raw_input is not None:
        aligncache_decision = aligncache.prepare_single_aligncache_step(
            self,
            raw_input=raw_input,
        )
        if (
            aligncache_decision is not None
            and str(aligncache_decision.get("exec_mode")) == "skip"
            and aligncache_decision.get("predicted_output") is not None
        ):
            predicted_output = aligncache.complete_single_aligncache_skip(
                self,
                raw_input=raw_input,
                predicted_output=aligncache_decision["predicted_output"],
                decision=aligncache_decision,
            )[0]
            return predicted_output.to(torch.float32)

    should_calc = True
    if easycache_enabled:
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
            rescale_func = np.poly1d(_OPENSORA_TEACACHE_COEFFICIENTS)
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
    if aligncache_enabled and raw_input is not None and aligncache_decision is not None:
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



import argparse
import json
import os
import sys
import time
from pathlib import Path
from pprint import pformat

_SCRIPT_DIR = Path(__file__).resolve().parent
if (_SCRIPT_DIR / "opensora").is_dir() and str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import colossalai
from colossalai.cluster import DistCoordinator
from mmengine.config import Config
from mmengine.runner import set_random_seed
from tqdm import tqdm

from opensora.acceleration.parallel_states import set_sequence_parallel_group
from opensora.datasets import save_sample
from opensora.datasets.aspect import get_image_size, get_num_frames
from opensora.models.text_encoder.t5 import text_preprocessing
from opensora.registry import MODELS, SCHEDULERS, build_module
from opensora.utils.config_utils import merge_args, parse_args as parse_opensora_args
from opensora.utils.inference_utils import (
    add_watermark,
    append_generated,
    append_score_to_prompts,
    apply_mask_strategy,
    collect_references_batch,
    dframe_to_frame,
    extract_json_from_prompts,
    extract_prompts_loop,
    get_save_path_name,
    load_prompts,
    merge_prompt,
    prepare_multi_resolution_info,
    refine_prompts_by_openai,
    split_prompt,
)
from opensora.utils.misc import all_exists, create_logger, is_distributed, is_main_process, to_torch_dtype


def _str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value!r}")


def _parse_aligncache_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--base_thresh", "--base-thresh", "--thresh", dest="base_thresh", type=float, default=0.11)
    parser.add_argument("--warmup_pairs", "--warmup-pairs", "--ret_steps", dest="warmup_pairs", type=int, default=8)
    parser.add_argument(
        "--phase_scale_profile",
        "--phase-scale-profile",
        dest="phase_scale_profile",
        type=str,
        default=None,
        choices=["alignbase", "balanced"],
    )
    parser.add_argument("--phase_thresh_early", type=float, default=None)
    parser.add_argument("--phase_thresh_mid", type=float, default=None)
    parser.add_argument("--phase_thresh_late", type=float, default=None)
    parser.add_argument("--phase_scale_early", type=float, default=_OPENSORA_ALIGNCACHE_PHASE_THRESHOLD_SCALES["early"])
    parser.add_argument("--phase_scale_mid", type=float, default=_OPENSORA_ALIGNCACHE_PHASE_THRESHOLD_SCALES["mid"])
    parser.add_argument("--phase_scale_late", type=float, default=_OPENSORA_ALIGNCACHE_PHASE_THRESHOLD_SCALES["late"])
    parser.add_argument("--fb_ratio_ema_decay", "--ratio-ema-decay", dest="fb_ratio_ema_decay", type=float, default=0.90)
    parser.add_argument("--fb_ratio_clip_low", "--ratio-clip-low", dest="fb_ratio_clip_low", type=float, default=0.85)
    parser.add_argument("--fb_ratio_clip_high", "--ratio-clip-high", dest="fb_ratio_clip_high", type=float, default=1.25)
    parser.add_argument("--fb_age_cap_early", type=int, default=_OPENSORA_ALIGNCACHE_AGE_CAP["early"])
    parser.add_argument("--fb_age_cap_mid", type=int, default=_OPENSORA_ALIGNCACHE_AGE_CAP["mid"])
    parser.add_argument("--fb_age_cap_late", type=int, default=_OPENSORA_ALIGNCACHE_AGE_CAP["late"])
    parser.add_argument("--fb_scale_init_early", type=float, default=_ALIGNCACHE_DEFAULT_SCALE_INIT["early"])
    parser.add_argument("--fb_scale_init_mid", type=float, default=_ALIGNCACHE_DEFAULT_SCALE_INIT["mid"])
    parser.add_argument("--fb_scale_init_late", type=float, default=_ALIGNCACHE_DEFAULT_SCALE_INIT["late"])
    parser.add_argument("--fb_scale_low_early", type=float, default=_ALIGNCACHE_DEFAULT_SCALE_LOW["early"])
    parser.add_argument("--fb_scale_low_mid", type=float, default=_ALIGNCACHE_DEFAULT_SCALE_LOW["mid"])
    parser.add_argument("--fb_scale_low_late", type=float, default=_ALIGNCACHE_DEFAULT_SCALE_LOW["late"])
    parser.add_argument("--fb_scale_high_early", type=float, default=_OPENSORA_ALIGNCACHE_SCALE_HIGH["early"])
    parser.add_argument("--fb_scale_high_mid", type=float, default=_OPENSORA_ALIGNCACHE_SCALE_HIGH["mid"])
    parser.add_argument("--fb_scale_high_late", type=float, default=_OPENSORA_ALIGNCACHE_SCALE_HIGH["late"])
    parser.add_argument("--pair_enable", "--pair-enable", dest="pair_enable", type=_str2bool, default=True)
    parser.add_argument("--pair_aggr", "--pair-aggr", dest="pair_aggr", type=float, default=0.0)
    parser.add_argument("--pair_second_order", dest="pair_second_order", type=_str2bool, default=True)
    parser.add_argument("--pair_online_scale", dest="pair_online_scale", type=_str2bool, default=True)
    parser.add_argument("--pair_stride", "--pair-stride", dest="pair_stride", type=int, default=64)
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
    parser.add_argument("--anchor_enable", "--anchor-enable", dest="anchor_enable", type=_str2bool, default=True)
    parser.add_argument("--anchor_exact_early", dest="anchor_exact_early", type=_str2bool, default=True)
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
    parser.add_argument(
        "--summary_json",
        "--summary-json",
        dest="summary_json",
        type=str,
        default=None,
    )
    return parser.parse_known_args(sys.argv[1:])


def _load_cfg(remaining_argv: list[str]):
    original_argv = list(sys.argv)
    sys.argv = [sys.argv[0], *remaining_argv]
    try:
        args = parse_opensora_args(training=False)
        cfg = Config.fromfile(args.config, lazy_import=False)
        return merge_args(cfg, args, training=False)
    finally:
        sys.argv = original_argv


def _allow_local_t5_pretrained(cfg) -> None:
    text_encoder_cfg = cfg.get("text_encoder", {})
    if not hasattr(text_encoder_cfg, "get"):
        return
    source = text_encoder_cfg.get("from_pretrained")
    if source in (None, ""):
        return

    source_text = str(source)
    source_path = Path(source_text).expanduser()
    if not source_path.is_absolute():
        source_path = (Path.cwd() / source_path).resolve()
    if not source_path.exists():
        return

    from opensora.models.text_encoder import t5 as t5_module

    available_models = getattr(t5_module.T5Embedder, "available_models", None)
    if not isinstance(available_models, list):
        return
    for candidate in (source_text, str(source_path), str(source_path.resolve())):
        if candidate not in available_models:
            available_models.append(candidate)


def _find_local_pixart_vae(cfg) -> str | None:
    candidates: list[Path] = []
    for env_var in ("ALIGNCACHE_OPENSORA_PIXART_VAE_PATH", "VBENCH_OPENSORA_PIXART_VAE_PATH"):
        env_path = os.getenv(env_var)
        if env_path:
            candidates.append(Path(env_path).expanduser())

    for env_var in ("ALIGNCACHE_CHECKPOINTS_DIR", "VBENCH_CHECKPOINTS_DIR"):
        checkpoints_root = os.getenv(env_var)
        if checkpoints_root:
            candidates.append(Path(checkpoints_root).expanduser() / "pixart_sigma_sdxlvae_T5_diffusers")

    vae_cfg = cfg.get("vae", {})
    if hasattr(vae_cfg, "get"):
        vae_source = vae_cfg.get("from_pretrained")
        if vae_source not in (None, ""):
            source_path = Path(str(vae_source)).expanduser()
            if source_path.exists():
                candidates.extend(
                    [
                        source_path / "pixart_sigma_sdxlvae_T5_diffusers",
                        source_path.parent / "pixart_sigma_sdxlvae_T5_diffusers",
                    ]
                )

    candidates.append(_SCRIPT_DIR / "checkpoints" / "pixart_sigma_sdxlvae_T5_diffusers")
    candidates.append(Path.cwd() / "checkpoints" / "pixart_sigma_sdxlvae_T5_diffusers")

    for candidate in candidates:
        resolved = candidate.resolve()
        if (resolved / "vae" / "config.json").is_file():
            return str(resolved)
    return None


def _allow_local_pixart_vae(cfg) -> None:
    local_pixart_vae = _find_local_pixart_vae(cfg)
    if local_pixart_vae is None:
        return

    from opensora.models.vae import vae as vae_module

    target_cls = getattr(vae_module, "VideoAutoencoderKL", None)
    if target_cls is None or getattr(target_cls, "_aligncache_local_pixart_patch", False):
        return

    original_init = target_cls.__init__

    def _init_with_local_pixart(self, from_pretrained=None, *args, **kwargs):
        if from_pretrained == "PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers":
            from_pretrained = local_pixart_vae
        return original_init(self, from_pretrained=from_pretrained, *args, **kwargs)

    target_cls.__init__ = _init_with_local_pixart
    target_cls._aligncache_local_pixart_patch = True


def _build_method_config(parsed: argparse.Namespace) -> dict[str, object]:
    method_config: dict[str, object] = {
        "feedback_enabled": True,
        "base_thresh": float(parsed.base_thresh),
        "warmup_pairs": int(parsed.warmup_pairs),
        "phase_thresh_early": parsed.phase_thresh_early,
        "phase_thresh_mid": parsed.phase_thresh_mid,
        "phase_thresh_late": parsed.phase_thresh_late,
        "phase_scale_early": float(parsed.phase_scale_early),
        "phase_scale_mid": float(parsed.phase_scale_mid),
        "phase_scale_late": float(parsed.phase_scale_late),
        "fb_ratio_ema_decay": float(parsed.fb_ratio_ema_decay),
        "fb_ratio_clip_low": float(parsed.fb_ratio_clip_low),
        "fb_ratio_clip_high": float(parsed.fb_ratio_clip_high),
        "fb_age_cap_early": int(parsed.fb_age_cap_early),
        "fb_age_cap_mid": int(parsed.fb_age_cap_mid),
        "fb_age_cap_late": int(parsed.fb_age_cap_late),
        "fb_scale_init_early": float(parsed.fb_scale_init_early),
        "fb_scale_init_mid": float(parsed.fb_scale_init_mid),
        "fb_scale_init_late": float(parsed.fb_scale_init_late),
        "fb_scale_low_early": float(parsed.fb_scale_low_early),
        "fb_scale_low_mid": float(parsed.fb_scale_low_mid),
        "fb_scale_low_late": float(parsed.fb_scale_low_late),
        "fb_scale_high_early": float(parsed.fb_scale_high_early),
        "fb_scale_high_mid": float(parsed.fb_scale_high_mid),
        "fb_scale_high_late": float(parsed.fb_scale_high_late),
        "pair_enable": bool(parsed.pair_enable),
        "pair_aggr": float(parsed.pair_aggr),
        "pair_second_order": bool(parsed.pair_second_order),
        "pair_online_scale": bool(parsed.pair_online_scale),
        "pair_stride": int(parsed.pair_stride),
        "pair_ratio_ema": float(parsed.pair_ratio_ema),
        "pair_dir_ema": float(parsed.pair_dir_ema),
        "pair_beta_mid": float(parsed.pair_beta_mid),
        "pair_beta_late": float(parsed.pair_beta_late),
        "pair_ratio_tol_mid": float(parsed.pair_ratio_tol_mid),
        "pair_ratio_tol_late": float(parsed.pair_ratio_tol_late),
        "pair_dir_min_mid": float(parsed.pair_dir_min_mid),
        "pair_dir_min_late": float(parsed.pair_dir_min_late),
        "pair_scale_ema": float(parsed.pair_scale_ema),
        "pair_scale_low": float(parsed.pair_scale_low),
        "pair_scale_high": float(parsed.pair_scale_high),
        "anchor_enable": bool(parsed.anchor_enable),
        "anchor_exact_early": bool(parsed.anchor_exact_early),
        "anchor_age_mid": int(parsed.anchor_age_mid),
        "anchor_age_late": int(parsed.anchor_age_late),
        "anchor_cond_cap_mid": float(parsed.anchor_cond_cap_mid),
        "anchor_cond_cap_late": float(parsed.anchor_cond_cap_late),
        "anchor_alpha_early": float(parsed.anchor_alpha_early),
        "anchor_alpha_mid": float(parsed.anchor_alpha_mid),
        "anchor_alpha_late": float(parsed.anchor_alpha_late),
        "anchor_alpha_ema": float(parsed.anchor_alpha_ema),
        "anchor_blend_mid": float(parsed.anchor_blend_mid),
        "anchor_blend_late": float(parsed.anchor_blend_late),
        "anchor_diff_cap_mid": float(parsed.anchor_diff_cap_mid),
        "anchor_diff_cap_late": float(parsed.anchor_diff_cap_late),
    }
    if parsed.phase_scale_profile:
        method_config["phase_scale_profile"] = str(parsed.phase_scale_profile)
    return method_config


def main() -> None:
    torch.set_grad_enabled(False)
    aligncache_args, remaining_argv = _parse_aligncache_args()
    cfg = _load_cfg(remaining_argv)
    _allow_local_t5_pretrained(cfg)
    _allow_local_pixart_vae(cfg)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg_dtype = cfg.get("dtype", "fp32")
    assert cfg_dtype in ["fp16", "bf16", "fp32"], f"Unknown mixed precision {cfg_dtype}"
    dtype = to_torch_dtype(cfg.get("dtype", "bf16"))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if is_distributed():
        colossalai.launch_from_torch({})
        coordinator = DistCoordinator()
        enable_sequence_parallelism = coordinator.world_size > 1
        if enable_sequence_parallelism:
            set_sequence_parallel_group(dist.group.WORLD)
    else:
        coordinator = None
        enable_sequence_parallelism = False
    set_random_seed(seed=cfg.get("seed", 1024))

    logger = create_logger()
    logger.info("Inference configuration:\\n %s", pformat(cfg.to_dict()))
    verbose = cfg.get("verbose", 1)
    progress_wrap = tqdm if verbose == 1 else (lambda x: x)

    logger.info("Building models...")
    text_encoder_dtype = dtype if device == "cuda" else torch.float32
    logger.info("Building text encoder (dtype=%s)...", text_encoder_dtype)
    text_encoder = build_module(cfg.text_encoder, MODELS, device=device, dtype=text_encoder_dtype)
    logger.info("Text encoder is ready.")
    logger.info("Building VAE (dtype=%s)...", dtype)
    vae = build_module(cfg.vae, MODELS).to(device, dtype).eval()
    logger.info("VAE is ready.")

    image_size = cfg.get("image_size", None)
    if image_size is None:
        resolution = cfg.get("resolution", None)
        aspect_ratio = cfg.get("aspect_ratio", None)
        assert resolution is not None and aspect_ratio is not None, (
            "resolution and aspect_ratio must be provided if image_size is not provided"
        )
        image_size = get_image_size(resolution, aspect_ratio)
    num_frames = get_num_frames(cfg.num_frames)

    input_size = (num_frames, *image_size)
    latent_size = vae.get_latent_size(input_size)
    logger.info("Building diffusion model (dtype=%s)...", dtype)
    model = (
        build_module(
            cfg.model,
            MODELS,
            input_size=latent_size,
            in_channels=vae.out_channels,
            caption_channels=text_encoder.output_dim,
            model_max_length=text_encoder.model_max_length,
            enable_sequence_parallelism=enable_sequence_parallelism,
        )
        .to(device, dtype)
        .eval()
    )
    logger.info("Diffusion model is ready.")
    text_encoder.y_embedder = model.y_embedder

    logger.info("Building scheduler...")
    scheduler = build_module(cfg.scheduler, SCHEDULERS)
    logger.info("Scheduler is ready.")
    method_config = _build_method_config(aligncache_args)
    resolved_method_config = configure_opensora_method(
        model,
        method_name="aligncache",
        method_config=method_config,
        sample_steps=int(cfg.scheduler.get("num_sampling_steps", 1)),
    )
    logger.info("AlignCache enabled for OpenSora: %s", resolved_method_config)
    if aligncache_args.summary_json:
        summary_path = Path(aligncache_args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(resolved_method_config, indent=2), encoding="utf-8")

    prompts = cfg.get("prompt", None)
    start_idx = cfg.get("start_index", 0)
    if prompts is None:
        if cfg.get("prompt_path", None) is not None:
            prompts = load_prompts(cfg.prompt_path, start_idx, cfg.get("end_index", None))
        else:
            prompts = [cfg.get("prompt_generator", "")] * 1_000_000

    reference_path = cfg.get("reference_path", [""] * len(prompts))
    mask_strategy = cfg.get("mask_strategy", [""] * len(prompts))
    assert len(reference_path) == len(prompts)
    assert len(mask_strategy) == len(prompts)

    fps = cfg.fps
    save_fps = cfg.get("save_fps", fps // cfg.get("frame_interval", 1))
    multi_resolution = cfg.get("multi_resolution", None)
    batch_size = cfg.get("batch_size", 1)
    num_sample = cfg.get("num_sample", 1)
    loop = cfg.get("loop", 1)
    condition_frame_length = cfg.get("condition_frame_length", 5)
    condition_frame_edit = cfg.get("condition_frame_edit", 0.0)
    align = cfg.get("align", None)

    save_dir = cfg.save_dir
    os.makedirs(save_dir, exist_ok=True)
    sample_name = cfg.get("sample_name", None)
    prompt_as_path = cfg.get("prompt_as_path", False)

    for i in progress_wrap(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[i : i + batch_size]
        ms = mask_strategy[i : i + batch_size]
        refs = reference_path[i : i + batch_size]

        batch_prompts, refs, ms = extract_json_from_prompts(batch_prompts, refs, ms)
        original_batch_prompts = batch_prompts
        refs = collect_references_batch(refs, vae, image_size)
        model_args = prepare_multi_resolution_info(
            multi_resolution, len(batch_prompts), image_size, num_frames, fps, device, dtype
        )

        for k in range(num_sample):
            save_paths = [
                get_save_path_name(
                    save_dir,
                    sample_name=sample_name,
                    sample_idx=start_idx + idx,
                    prompt=original_batch_prompts[idx],
                    prompt_as_path=prompt_as_path,
                    num_sample=num_sample,
                    k=k,
                )
                for idx in range(len(batch_prompts))
            ]
            if prompt_as_path and all_exists(save_paths):
                continue

            batched_prompt_segment_list = []
            batched_loop_idx_list = []
            for prompt in batch_prompts:
                prompt_segment_list, loop_idx_list = split_prompt(prompt)
                batched_prompt_segment_list.append(prompt_segment_list)
                batched_loop_idx_list.append(loop_idx_list)

            if cfg.get("llm_refine", False):
                if not enable_sequence_parallelism or (enable_sequence_parallelism and is_main_process()):
                    for idx, prompt_segment_list in enumerate(batched_prompt_segment_list):
                        batched_prompt_segment_list[idx] = refine_prompts_by_openai(prompt_segment_list)
                if enable_sequence_parallelism:
                    coordinator.block_all()
                    prompt_segment_length = [len(prompt_segment_list) for prompt_segment_list in batched_prompt_segment_list]
                    batched_prompt_segment_list = [
                        prompt_segment
                        for prompt_segment_list in batched_prompt_segment_list
                        for prompt_segment in prompt_segment_list
                    ]
                    broadcast_obj_list = [batched_prompt_segment_list] * coordinator.world_size
                    dist.broadcast_object_list(broadcast_obj_list, 0)
                    batched_prompt_segment_list = []
                    segment_start_idx = 0
                    all_prompts = broadcast_obj_list[0]
                    for num_segment in prompt_segment_length:
                        batched_prompt_segment_list.append(all_prompts[segment_start_idx : segment_start_idx + num_segment])
                        segment_start_idx += num_segment

            for idx, prompt_segment_list in enumerate(batched_prompt_segment_list):
                batched_prompt_segment_list[idx] = append_score_to_prompts(
                    prompt_segment_list,
                    aes=cfg.get("aes", None),
                    flow=cfg.get("flow", None),
                    camera_motion=cfg.get("camera_motion", None),
                )
            for idx, prompt_segment_list in enumerate(batched_prompt_segment_list):
                batched_prompt_segment_list[idx] = [text_preprocessing(prompt) for prompt in prompt_segment_list]

            batch_prompts = []
            for prompt_segment_list, loop_idx_list in zip(batched_prompt_segment_list, batched_loop_idx_list):
                batch_prompts.append(merge_prompt(prompt_segment_list, loop_idx_list))

            video_clips = []
            for loop_i in range(loop):
                batch_prompts_loop = extract_prompts_loop(batch_prompts, loop_i)
                if loop_i > 0:
                    refs, ms = append_generated(
                        vae, video_clips[-1], refs, ms, loop_i, condition_frame_length, condition_frame_edit
                    )

                z = torch.randn(len(batch_prompts), vae.out_channels, *latent_size, device=device, dtype=dtype)
                masks = apply_mask_strategy(z, refs, ms, loop_i, align=align)
                samples = scheduler.sample(
                    model,
                    text_encoder,
                    z=z,
                    prompts=batch_prompts_loop,
                    device=device,
                    additional_args=model_args,
                    progress=verbose >= 2,
                    mask=masks,
                )
                samples = vae.decode(samples.to(dtype), num_frames=num_frames)
                video_clips.append(samples)

            if is_main_process():
                for idx, batch_prompt in enumerate(batch_prompts):
                    if verbose >= 2:
                        logger.info("Prompt: %s", batch_prompt)
                    save_path = save_paths[idx]
                    video = [video_clips[i][idx] for i in range(loop)]
                    for loop_index in range(1, loop):
                        video[loop_index] = video[loop_index][:, dframe_to_frame(condition_frame_length) :]
                    video = torch.cat(video, dim=1)
                    save_path = save_sample(video, fps=save_fps, save_path=save_path, verbose=verbose >= 2)
                    if save_path.endswith(".mp4") and cfg.get("watermark", False):
                        time.sleep(1)
                        add_watermark(save_path)
        start_idx += len(batch_prompts)

    logger.info("Inference finished.")
    logger.info("Saved %s samples to %s", start_idx, save_dir)


if __name__ == "__main__":
    main()
