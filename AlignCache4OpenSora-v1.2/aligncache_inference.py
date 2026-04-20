from __future__ import annotations

import types
from typing import Any, Sequence

import torch


_PHASES = ("early", "mid", "late")

_ALIGNCACHE_DEFAULT_PHASE_THRESHOLD_SCALES = {
    "early": 1.0,
    "mid": 1.0,
    "late": 1.0,
}
_ALIGNCACHE_BALANCED_PHASE_THRESHOLD_SCALES = {
    "early": 1.15,
    "mid": 1.0,
    "late": 0.8,
}
_ALIGNCACHE_DEFAULT_AGE_CAP = {
    "early": 5,
    "mid": 4,
    "late": 3,
}
_ALIGNCACHE_DEFAULT_SCALE_INIT = {
    "early": 1.0,
    "mid": 1.0,
    "late": 1.0,
}
_ALIGNCACHE_DEFAULT_SCALE_LOW = {
    "early": 0.95,
    "mid": 0.90,
    "late": 0.85,
}
_ALIGNCACHE_DEFAULT_SCALE_HIGH = {
    "early": 1.03,
    "mid": 1.06,
    "late": 1.0,
}
_ALIGNCACHE_DEFAULTS = {
    "ratio_ema_decay": 0.90,
    "ratio_clip_low": 0.85,
    "ratio_clip_high": 1.25,
    "score_eps": 1e-6,
}
_ALIGNCACHE_PHASE_SCALE_PROFILES = {
    "alignbase": dict(_ALIGNCACHE_DEFAULT_PHASE_THRESHOLD_SCALES),
    "balanced": dict(_ALIGNCACHE_BALANCED_PHASE_THRESHOLD_SCALES),
}
_ALIGNCACHE_REASON_SUMMARY_KEYS = {
    "warmup_tail": "warmup_tail_pairs",
    "bootstrap": "bootstrap_pairs",
    "max_age": "max_age_pairs",
    "budget_exact": "budget_exact_pairs",
    "budget_skip": "budget_skip_pairs",
    "cache_miss_exact": "cache_miss_exact_pairs",
}
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
}
_WAN_SUPPORTED_TASKS = frozenset({"t2v-1.3B", "t2v-14B"})


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


def _clamp_float(value: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(value)))


def _coerce_optional_str(value: Any, *, label: str) -> str | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    if not text:
        raise ValueError(f"{label} must be a non-empty string when provided.")
    return text


def _resolve_phase_mapping(
    defaults: dict[str, Any],
    override: dict[str, Any] | None,
    caster,
) -> dict[str, Any]:
    resolved: dict[str, Any] = {}
    source = dict(override or {})
    for phase in _PHASES:
        value = source.get(phase, defaults[phase])
        resolved[phase] = caster(value)
    return resolved


def _phase_mapping_from_flat_keys(raw: dict[str, Any], prefix: str, caster) -> dict[str, Any] | None:
    resolved: dict[str, Any] = {}
    for phase in _PHASES:
        key = f"{prefix}_{phase}"
        if raw.get(key) is None:
            continue
        resolved[phase] = caster(raw[key])
    return resolved or None


def _normalize_aligncache_method_config(raw: dict[str, Any] | None) -> dict[str, Any]:
    normalized = dict(raw or {})

    def _alias(preferred: str, legacy: str) -> None:
        value = normalized.get(preferred)
        if value not in (None, ""):
            normalized[legacy] = value

    _alias("base_thresh", "cache_thresh")
    _alias("warmup_pairs", "cache_ret_steps")
    _alias("base_phase_scale_profile", "phase_scale_profile")
    if normalized.get("phase_threshold_scales") in (None, "") and isinstance(
        normalized.get("base_phase_threshold_scales"), dict
    ):
        normalized["phase_threshold_scales"] = dict(normalized["base_phase_threshold_scales"])

    for label in ("ratio_ema_decay", "ratio_clip_low", "ratio_clip_high"):
        _alias(f"fb_{label}", f"fb2_{label}")

    for phase in _PHASES:
        _alias(f"fb_age_cap_{phase}", f"fb2_age_cap_{phase}")
        _alias(f"fb_scale_init_{phase}", f"fb2_scale_init_{phase}")
        _alias(f"fb_scale_low_{phase}", f"fb2_scale_low_{phase}")
        _alias(f"fb_scale_high_{phase}", f"fb2_scale_high_{phase}")

    for label in ("age_cap", "scale_init", "scale_low", "scale_high"):
        _alias(f"fb_{label}", f"fb2_{label}")

    pair_aliases = {
        "pair_enable": "v2_enable",
        "pair_aggr": "v2_aggressiveness",
        "pair_early_end_ratio": "v2_early_end_ratio",
        "pair_late_start_ratio": "v2_late_start_ratio",
        "pair_second_order": "v2_use_second_order",
        "pair_online_scale": "v2_use_online_scale",
        "pair_stride": "v2_subsample_stride",
        "pair_ratio_ema": "v2_ratio_ema_decay",
        "pair_dir_ema": "v2_dir_ema_decay",
        "pair_beta_mid": "v2_beta_mid",
        "pair_beta_late": "v2_beta_late",
        "pair_ratio_tol_mid": "v2_ratio_tol_mid",
        "pair_ratio_tol_late": "v2_ratio_tol_late",
        "pair_dir_min_mid": "v2_dir_min_mid",
        "pair_dir_min_late": "v2_dir_min_late",
        "pair_scale_ema": "v2_scale_ema_decay",
        "pair_scale_low": "v2_scale_clamp_low",
        "pair_scale_high": "v2_scale_clamp_high",
        "anchor_enable": "v2_b2_enable",
        "anchor_exact_early": "v2_b2_exact_early",
        "anchor_age_mid": "v2_b2_max_pred_age_mid",
        "anchor_age_late": "v2_b2_max_pred_age_late",
        "anchor_cond_cap_mid": "v2_b2_cond_rel_cap_mid",
        "anchor_cond_cap_late": "v2_b2_cond_rel_cap_late",
        "anchor_alpha_early": "v2_b2_alpha_prior_early",
        "anchor_alpha_mid": "v2_b2_alpha_prior_mid",
        "anchor_alpha_late": "v2_b2_alpha_prior_late",
        "anchor_alpha_ema": "v2_b2_alpha_ema_decay",
        "anchor_blend_mid": "v2_b2_blend_mid",
        "anchor_blend_late": "v2_b2_blend_late",
        "anchor_diff_cap_mid": "v2_b2_disagree_rel_cap_mid",
        "anchor_diff_cap_late": "v2_b2_disagree_rel_cap_late",
    }
    for preferred, legacy in pair_aliases.items():
        _alias(preferred, legacy)
    return normalized


def _resolved_ret_steps(
    raw: dict[str, Any],
    *,
    label_prefix: str,
    sample_steps: int,
    default_ret_steps: int | None = None,
) -> int:
    ret_steps = _coerce_optional_int(
        raw.get("warmup_pairs", raw.get("cache_ret_steps", raw.get("ret_steps"))),
        label=f"{label_prefix}.warmup_pairs",
    )
    if ret_steps is not None:
        if ret_steps < 0:
            raise ValueError(f"{label_prefix}.warmup_pairs must be non-negative. Received {ret_steps!r}.")
        return int(ret_steps)
    if default_ret_steps is not None:
        return int(default_ret_steps)
    return 8 if int(sample_steps) <= 30 else 10


def _resolved_base_thresh(raw: dict[str, Any], *, label_prefix: str, default: float) -> float:
    thresh = _coerce_optional_float(
        raw.get("base_thresh", raw.get("cache_thresh", raw.get("aligncache_thresh", raw.get("thresh", raw.get("threshold"))))),
        label=f"{label_prefix}.base_thresh",
    )
    resolved = float(default if thresh is None else thresh)
    if resolved < 0.0:
        raise ValueError(f"{label_prefix}.base_thresh must be non-negative. Received {resolved!r}.")
    return resolved


def _resolve_phase_thresholds(
    *,
    base_thresh: float,
    raw: dict[str, Any],
    label_prefix: str,
    default_phase_threshold_scales: dict[str, float] | None = None,
    default_phase_scale_profile: str | None = None,
) -> tuple[dict[str, float], dict[str, float], str | None]:
    explicit_thresholds = raw.get("phase_thresholds")
    if not isinstance(explicit_thresholds, dict):
        explicit_thresholds = _phase_mapping_from_flat_keys(raw, "phase_thresh", float)
    phase_scale_profile = _coerce_optional_str(
        raw.get("phase_scale_profile"),
        label=f"{label_prefix}.phase_scale_profile",
    )
    resolved_profile = None if phase_scale_profile is None else phase_scale_profile.lower()
    if resolved_profile is None and default_phase_scale_profile not in (None, ""):
        resolved_profile = str(default_phase_scale_profile).strip().lower()
    profile_scales = None
    if resolved_profile is not None:
        profile_scales = _ALIGNCACHE_PHASE_SCALE_PROFILES.get(resolved_profile)
        if profile_scales is None:
            supported = ", ".join(sorted(_ALIGNCACHE_PHASE_SCALE_PROFILES))
            raise ValueError(
                f"{label_prefix}.phase_scale_profile must be one of {supported}. "
                f"Received {phase_scale_profile!r}."
            )
    explicit_scales = raw.get("phase_threshold_scales")
    if not isinstance(explicit_scales, dict):
        explicit_scales = raw.get("base_phase_threshold_scales")
    if not isinstance(explicit_scales, dict):
        explicit_scales = _phase_mapping_from_flat_keys(raw, "phase_scale", float)
    if explicit_scales is None and profile_scales is not None:
        explicit_scales = dict(profile_scales)
    resolved_scales = _resolve_phase_mapping(
        dict(default_phase_threshold_scales or _ALIGNCACHE_DEFAULT_PHASE_THRESHOLD_SCALES),
        explicit_scales,
        float,
    )
    resolved_thresholds: dict[str, float] = {}
    for phase in _PHASES:
        explicit_value = None if explicit_thresholds is None else explicit_thresholds.get(phase)
        if explicit_value is None:
            explicit_value = float(base_thresh) * float(resolved_scales[phase])
        resolved_thresholds[phase] = float(explicit_value)
    return resolved_thresholds, resolved_scales, resolved_profile


def _pair_cfg_from_v2_cfg(v2_cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "pair_enable": bool(v2_cfg["enable"]),
        "enable": bool(v2_cfg["enable"]),
        "pair_aggr": float(v2_cfg["aggressiveness"]),
        "pair_second_order": bool(v2_cfg["use_second_order"]),
        "pair_online_scale": bool(v2_cfg["use_online_scale"]),
        "pair_stride": int(v2_cfg["subsample_stride"]),
        "pair_ratio_ema": float(v2_cfg["qc_ratio_ema_decay"]),
        "pair_dir_ema": float(v2_cfg["qc_dir_ema_decay"]),
        "pair_beta_mid": float(v2_cfg["beta_mid"]),
        "pair_beta_late": float(v2_cfg["beta_late"]),
        "pair_ratio_tol_mid": float(v2_cfg["ratio_tol_mid"]),
        "pair_ratio_tol_late": float(v2_cfg["ratio_tol_late"]),
        "pair_dir_min_mid": float(v2_cfg["dir_min_mid"]),
        "pair_dir_min_late": float(v2_cfg["dir_min_late"]),
        "pair_scale_ema": float(v2_cfg["scale_ema_decay"]),
        "pair_scale_low": float(v2_cfg["scale_clamp_low"]),
        "pair_scale_high": float(v2_cfg["scale_clamp_high"]),
        "pair_early_end_ratio": float(v2_cfg["early_end_ratio"]),
        "pair_late_start_ratio": float(v2_cfg["late_start_ratio"]),
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


def _coerce_aligncache_v2_cfg_input(cfg: dict[str, Any] | None) -> dict[str, Any]:
    raw = _normalize_aligncache_method_config(dict(cfg or {}))
    if "subsample_stride" in raw:
        return raw
    return _resolve_aligncache_v2_cfg(raw, label_prefix="aligncache.pair_cfg")


def _apply_v2_aggressiveness(cfg: dict[str, Any], aggressiveness: float) -> dict[str, Any]:
    tuned = dict(cfg)
    aggressiveness = float(aggressiveness)
    beta_scale = _clamp_float(1.0 + 0.60 * aggressiveness, 0.50, 1.35)
    blend_scale = _clamp_float(1.0 + 0.50 * aggressiveness, 0.50, 1.40)
    clamp_delta = 0.08 * aggressiveness
    tuned["beta_mid"] = float(tuned["beta_mid"]) * beta_scale
    tuned["beta_late"] = float(tuned["beta_late"]) * beta_scale
    tuned["b2_blend_mid"] = float(tuned["b2_blend_mid"]) * blend_scale
    tuned["b2_blend_late"] = float(tuned["b2_blend_late"]) * blend_scale
    tuned["scale_clamp_low"] = max(0.60, float(tuned["scale_clamp_low"]) - clamp_delta)
    tuned["scale_clamp_high"] = max(float(tuned["scale_clamp_low"]), float(tuned["scale_clamp_high"]) + clamp_delta)
    return tuned


def _resolve_aligncache_v2_cfg(raw: dict[str, Any], *, label_prefix: str) -> dict[str, Any]:
    resolved = dict(_ALIGNCACHE_V2_DEFAULTS)
    bool_keys = {
        "pair_enable": ("enable", "v2_enable"),
        "pair_second_order": ("use_second_order", "v2_use_second_order"),
        "pair_online_scale": ("use_online_scale", "v2_use_online_scale"),
        "anchor_enable": ("b2_enable", "v2_b2_enable"),
        "anchor_exact_early": ("b2_exact_early", "v2_b2_exact_early"),
    }
    int_keys = {
        "pair_stride": ("subsample_stride", "v2_subsample_stride"),
        "anchor_age_mid": ("b2_max_pred_age_mid", "v2_b2_max_pred_age_mid"),
        "anchor_age_late": ("b2_max_pred_age_late", "v2_b2_max_pred_age_late"),
    }
    float_keys = {
        "pair_aggr": ("aggressiveness", "v2_aggressiveness"),
        "pair_early_end_ratio": ("early_end_ratio", "v2_early_end_ratio"),
        "pair_late_start_ratio": ("late_start_ratio", "v2_late_start_ratio"),
        "pair_ratio_ema": ("qc_ratio_ema_decay", "v2_ratio_ema_decay"),
        "pair_dir_ema": ("qc_dir_ema_decay", "v2_dir_ema_decay"),
        "pair_beta_mid": ("beta_mid", "v2_beta_mid"),
        "pair_beta_late": ("beta_late", "v2_beta_late"),
        "pair_ratio_tol_mid": ("ratio_tol_mid", "v2_ratio_tol_mid"),
        "pair_ratio_tol_late": ("ratio_tol_late", "v2_ratio_tol_late"),
        "pair_dir_min_mid": ("dir_min_mid", "v2_dir_min_mid"),
        "pair_dir_min_late": ("dir_min_late", "v2_dir_min_late"),
        "pair_scale_ema": ("scale_ema_decay", "v2_scale_ema_decay"),
        "pair_scale_low": ("scale_clamp_low", "v2_scale_clamp_low"),
        "pair_scale_high": ("scale_clamp_high", "v2_scale_clamp_high"),
        "anchor_cond_cap_mid": ("b2_cond_rel_cap_mid", "v2_b2_cond_rel_cap_mid"),
        "anchor_cond_cap_late": ("b2_cond_rel_cap_late", "v2_b2_cond_rel_cap_late"),
        "anchor_alpha_early": ("b2_alpha_prior_early", "v2_b2_alpha_prior_early"),
        "anchor_alpha_mid": ("b2_alpha_prior_mid", "v2_b2_alpha_prior_mid"),
        "anchor_alpha_late": ("b2_alpha_prior_late", "v2_b2_alpha_prior_late"),
        "anchor_alpha_ema": ("b2_alpha_ema_decay", "v2_b2_alpha_ema_decay"),
        "anchor_blend_mid": ("b2_blend_mid", "v2_b2_blend_mid"),
        "anchor_blend_late": ("b2_blend_late", "v2_b2_blend_late"),
        "anchor_diff_cap_mid": ("b2_disagree_rel_cap_mid", "v2_b2_disagree_rel_cap_mid"),
        "anchor_diff_cap_late": ("b2_disagree_rel_cap_late", "v2_b2_disagree_rel_cap_late"),
    }
    for label_key, (target_key, legacy_key) in bool_keys.items():
        value = _coerce_optional_bool(raw.get(label_key, raw.get(legacy_key)), label=f"{label_prefix}.{label_key}")
        if value is not None:
            resolved[target_key] = bool(value)
    for label_key, (target_key, legacy_key) in int_keys.items():
        value = _coerce_optional_int(raw.get(label_key, raw.get(legacy_key)), label=f"{label_prefix}.{label_key}")
        if value is not None:
            resolved[target_key] = int(value)
    aggressiveness = float(resolved.pop("aggressiveness", 0.0))
    for label_key, (target_key, legacy_key) in float_keys.items():
        value = _coerce_optional_float(raw.get(label_key, raw.get(legacy_key)), label=f"{label_prefix}.{label_key}")
        if value is None:
            continue
        if target_key == "aggressiveness":
            aggressiveness = float(value)
        else:
            resolved[target_key] = float(value)
    if int(resolved["subsample_stride"]) < 1:
        raise ValueError(f"{label_prefix}.pair_stride must be >= 1.")
    if not 0.0 <= float(resolved["qc_ratio_ema_decay"]) <= 1.0:
        raise ValueError(f"{label_prefix}.pair_ratio_ema must be in [0, 1].")
    if not 0.0 <= float(resolved["qc_dir_ema_decay"]) <= 1.0:
        raise ValueError(f"{label_prefix}.pair_dir_ema must be in [0, 1].")
    if not 0.0 <= float(resolved["scale_ema_decay"]) <= 1.0:
        raise ValueError(f"{label_prefix}.pair_scale_ema must be in [0, 1].")
    if float(resolved["scale_clamp_low"]) <= 0.0:
        raise ValueError(f"{label_prefix}.pair_scale_low must be positive.")
    if float(resolved["scale_clamp_high"]) < float(resolved["scale_clamp_low"]):
        raise ValueError(f"{label_prefix}.pair_scale_high must be >= pair_scale_low.")
    if float(resolved["early_end_ratio"]) < 0.0 or float(resolved["early_end_ratio"]) > 1.0:
        raise ValueError(f"{label_prefix}.pair_early_end_ratio must be in [0, 1].")
    if float(resolved["late_start_ratio"]) < float(resolved["early_end_ratio"]) or float(resolved["late_start_ratio"]) > 1.0:
        raise ValueError(f"{label_prefix}.pair_late_start_ratio must be in [pair_early_end_ratio, 1].")
    resolved = _apply_v2_aggressiveness(resolved, aggressiveness)
    resolved["aggressiveness"] = float(aggressiveness)
    return resolved


def _resolve_aligncache_config(
    *,
    method_config: dict[str, Any] | None,
    sample_steps: int,
    label_prefix: str,
    default_thresh: float,
    default_ret_steps: int | None = None,
    default_phase_threshold_scales: dict[str, float] | None = None,
    default_phase_scale_profile: str | None = None,
) -> dict[str, Any]:
    raw = _normalize_aligncache_method_config(method_config)
    base_thresh = _resolved_base_thresh(raw, label_prefix=label_prefix, default=default_thresh)
    ret_steps = _resolved_ret_steps(
        raw,
        label_prefix=label_prefix,
        sample_steps=int(sample_steps),
        default_ret_steps=default_ret_steps,
    )
    phase_thresholds, phase_threshold_scales, phase_scale_profile = _resolve_phase_thresholds(
        base_thresh=base_thresh,
        raw=raw,
        label_prefix=label_prefix,
        default_phase_threshold_scales=default_phase_threshold_scales,
        default_phase_scale_profile=default_phase_scale_profile,
    )

    age_cap = raw.get("fb_age_cap", raw.get("fb2_age_cap"))
    if not isinstance(age_cap, dict):
        age_cap = _phase_mapping_from_flat_keys(raw, "fb_age_cap", int) or _phase_mapping_from_flat_keys(raw, "fb2_age_cap", int)
    scale_init = raw.get("fb_scale_init", raw.get("fb2_scale_init"))
    if not isinstance(scale_init, dict):
        scale_init = _phase_mapping_from_flat_keys(raw, "fb_scale_init", float) or _phase_mapping_from_flat_keys(raw, "fb2_scale_init", float)
    scale_low = raw.get("fb_scale_low", raw.get("fb2_scale_low"))
    if not isinstance(scale_low, dict):
        scale_low = _phase_mapping_from_flat_keys(raw, "fb_scale_low", float) or _phase_mapping_from_flat_keys(raw, "fb2_scale_low", float)
    scale_high = raw.get("fb_scale_high", raw.get("fb2_scale_high"))
    if not isinstance(scale_high, dict):
        scale_high = _phase_mapping_from_flat_keys(raw, "fb_scale_high", float) or _phase_mapping_from_flat_keys(raw, "fb2_scale_high", float)

    resolved_age_cap = _resolve_phase_mapping(_ALIGNCACHE_DEFAULT_AGE_CAP, age_cap, int)
    resolved_scale_init = _resolve_phase_mapping(_ALIGNCACHE_DEFAULT_SCALE_INIT, scale_init, float)
    resolved_scale_low = _resolve_phase_mapping(_ALIGNCACHE_DEFAULT_SCALE_LOW, scale_low, float)
    resolved_scale_high = _resolve_phase_mapping(_ALIGNCACHE_DEFAULT_SCALE_HIGH, scale_high, float)

    for phase in _PHASES:
        if float(resolved_scale_low[phase]) <= 0.0:
            raise ValueError(f"{label_prefix}.fb_scale_low[{phase}] must be positive.")
        if float(resolved_scale_high[phase]) < float(resolved_scale_low[phase]):
            raise ValueError(f"{label_prefix}.fb_scale_high[{phase}] must be >= fb_scale_low[{phase}].")
        if not (float(resolved_scale_low[phase]) <= float(resolved_scale_init[phase]) <= float(resolved_scale_high[phase])):
            raise ValueError(
                f"{label_prefix}.fb_scale_init[{phase}] must lie within "
                f"[fb_scale_low[{phase}], fb_scale_high[{phase}]]."
            )

    ratio_ema_decay = _coerce_optional_float(
        raw.get("fb_ratio_ema_decay", raw.get("fb2_ratio_ema_decay")),
        label=f"{label_prefix}.fb_ratio_ema_decay",
    )
    ratio_clip_low = _coerce_optional_float(
        raw.get("fb_ratio_clip_low", raw.get("fb2_ratio_clip_low")),
        label=f"{label_prefix}.fb_ratio_clip_low",
    )
    ratio_clip_high = _coerce_optional_float(
        raw.get("fb_ratio_clip_high", raw.get("fb2_ratio_clip_high")),
        label=f"{label_prefix}.fb_ratio_clip_high",
    )
    resolved_ratio_ema_decay = float(_ALIGNCACHE_DEFAULTS["ratio_ema_decay"] if ratio_ema_decay is None else ratio_ema_decay)
    resolved_ratio_clip_low = float(_ALIGNCACHE_DEFAULTS["ratio_clip_low"] if ratio_clip_low is None else ratio_clip_low)
    resolved_ratio_clip_high = float(_ALIGNCACHE_DEFAULTS["ratio_clip_high"] if ratio_clip_high is None else ratio_clip_high)
    if not 0.0 <= resolved_ratio_ema_decay <= 1.0:
        raise ValueError(f"{label_prefix}.fb_ratio_ema_decay must be in [0, 1].")
    if resolved_ratio_clip_low <= 0.0:
        raise ValueError(f"{label_prefix}.fb_ratio_clip_low must be positive.")
    if resolved_ratio_clip_high < resolved_ratio_clip_low:
        raise ValueError(f"{label_prefix}.fb_ratio_clip_high must be >= fb_ratio_clip_low.")

    v2_cfg = _resolve_aligncache_v2_cfg(raw, label_prefix=label_prefix)
    pair_cfg = _pair_cfg_from_v2_cfg(v2_cfg)
    guide_scale = _coerce_optional_float(raw.get("guide_scale", raw.get("sample_guide_scale")), label=f"{label_prefix}.guide_scale")
    return {
        "method": "aligncache",
        "scheduler": "aligncache-base-plus-fb",
        "cache_thresh": float(base_thresh),
        "cache_ret_steps": int(ret_steps),
        "base_thresh": float(base_thresh),
        "ret_steps": int(ret_steps),
        "warmup_pairs": int(ret_steps),
        "sample_steps": int(sample_steps),
        "guide_scale": None if guide_scale is None else float(guide_scale),
        "phase_thresholds": dict(phase_thresholds),
        "phase_threshold_scales": dict(phase_threshold_scales),
        "phase_scale_profile": phase_scale_profile,
        "base_scheduler": {
            "method": "aligncache_preserved_accumulated_error",
            "base_thresh": float(base_thresh),
            "ret_steps": int(ret_steps),
            "warmup_pairs": int(ret_steps),
            "phase_thresholds": dict(phase_thresholds),
            "phase_threshold_scales": dict(phase_threshold_scales),
        },
        "age_cap": dict(resolved_age_cap),
        "scale_init": dict(resolved_scale_init),
        "scale_low": dict(resolved_scale_low),
        "scale_high": dict(resolved_scale_high),
        "threshold_controller": {
            "scale_init": dict(resolved_scale_init),
            "scale_low": dict(resolved_scale_low),
            "scale_high": dict(resolved_scale_high),
        },
        "fb_age_cap": dict(resolved_age_cap),
        "fb_scale_init": dict(resolved_scale_init),
        "fb_scale_low": dict(resolved_scale_low),
        "fb_scale_high": dict(resolved_scale_high),
        "feedback": {
            "ratio_ema_decay": float(resolved_ratio_ema_decay),
            "ratio_clip_low": float(resolved_ratio_clip_low),
            "ratio_clip_high": float(resolved_ratio_clip_high),
            "update_reasons": ["budget_exact"],
        },
        "fb_ratio_ema_decay": float(resolved_ratio_ema_decay),
        "fb_ratio_clip_low": float(resolved_ratio_clip_low),
        "fb_ratio_clip_high": float(resolved_ratio_clip_high),
        "ratio_ema_decay": float(resolved_ratio_ema_decay),
        "ratio_clip_low": float(resolved_ratio_clip_low),
        "ratio_clip_high": float(resolved_ratio_clip_high),
        "score_eps": float(_ALIGNCACHE_DEFAULTS["score_eps"]),
        "v2_cfg": dict(v2_cfg),
        "pair_cfg": dict(pair_cfg),
        "predictor": dict(pair_cfg),
    }


def resolve_wan_aligncache_config(*, task: str, method_config: dict[str, Any] | None, sample_steps: int) -> dict[str, Any]:
    if str(task) not in _WAN_SUPPORTED_TASKS:
        supported = ", ".join(sorted(_WAN_SUPPORTED_TASKS))
        raise ValueError(
            "AlignCache for Wan currently supports the vendored T2V tasks only. "
            f"Received task={task!r}. Supported tasks: {supported}"
        )
    return _resolve_aligncache_config(
        method_config=method_config,
        sample_steps=int(sample_steps),
        label_prefix="wan.aligncache",
        default_thresh=0.0625,
        default_ret_steps=8,
        default_phase_threshold_scales=_ALIGNCACHE_BALANCED_PHASE_THRESHOLD_SCALES,
        default_phase_scale_profile="balanced",
    )


def resolve_hunyuan_aligncache_config(method_config: dict[str, Any] | None, *, sample_steps: int) -> dict[str, Any]:
    return _resolve_aligncache_config(
        method_config=method_config,
        sample_steps=int(sample_steps),
        label_prefix="hunyuan.aligncache",
        default_thresh=0.025,
    )


def resolve_opensora_aligncache_config(method_config: dict[str, Any] | None, *, sample_steps: int) -> dict[str, Any]:
    return _resolve_aligncache_config(
        method_config=method_config,
        sample_steps=int(sample_steps),
        label_prefix="opensora.aligncache",
        default_thresh=0.03,
    )


def _tensor_list(values: torch.Tensor | Sequence[torch.Tensor] | None) -> list[torch.Tensor] | None:
    if values is None:
        return None
    if isinstance(values, torch.Tensor):
        return [values]
    return [value for value in values]


def _clone_tensor_list(values: Sequence[torch.Tensor] | None) -> list[torch.Tensor] | None:
    if values is None:
        return None
    return [value.detach().clone() for value in values]


def _tensor_list_subsample(values: Sequence[torch.Tensor] | None, stride: int) -> torch.Tensor | None:
    if values is None:
        return None
    stride = max(int(stride), 1)
    chunks: list[torch.Tensor] = []
    for value in values:
        chunks.append(value.detach().reshape(-1)[::stride].to(torch.float32).clone())
    return torch.cat(chunks, dim=0) if chunks else None


def _tensor_list_cosine(first: torch.Tensor | None, second: torch.Tensor | None) -> float | None:
    if first is None or second is None:
        return None
    first_vec = first.reshape(-1).float()
    second_vec = second.reshape(-1).float()
    denom = torch.norm(first_vec, p=2) * torch.norm(second_vec, p=2)
    if float(denom.item()) <= 1e-12:
        return None
    value = (first_vec.dot(second_vec) / denom).clamp(min=-1.0, max=1.0)
    return float(value.item())


def _tensor_list_mean_abs(values: Sequence[torch.Tensor] | None) -> float | None:
    if not values:
        return None
    total = 0.0
    count = 0
    for value in values:
        current = value.detach().to(torch.float32)
        total += float(current.abs().sum().item())
        count += int(current.numel())
    if count <= 0:
        return None
    return float(total / count)


def _tensor_list_rel_mse(prediction: Sequence[torch.Tensor] | None, reference: Sequence[torch.Tensor] | None) -> float | None:
    if prediction is None or reference is None:
        return None
    numer = None
    denom = None
    for predicted, expected in zip(prediction, reference):
        predicted_f = predicted.detach().to(torch.float32)
        expected_f = expected.detach().to(torch.float32)
        current_numer = torch.sum((predicted_f - expected_f) ** 2, dtype=torch.float32)
        current_denom = torch.sum(expected_f * expected_f, dtype=torch.float32)
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
        target_f = target_value.detach().to(torch.float32)
        basis_f = basis_value.detach().to(torch.float32)
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
    return float(decay) * float(previous) + (1.0 - float(decay)) * float(current)


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
    return {"early": stored, "mid": stored, "late": stored}


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


def _init_v2_b2_state(cfg: dict[str, Any]) -> dict[str, Any]:
    state = {
        "anchor_cond": None,
        "anchor_uncond": None,
        "last_exact_step": -1,
        "phase_alpha_ema": {
            "early": float(cfg["b2_alpha_prior_early"]),
            "mid": float(cfg["b2_alpha_prior_mid"]),
            "late": float(cfg["b2_alpha_prior_late"]),
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
    state["last_exact_pair"] = state["last_exact_step"]
    state["alpha_ema"] = state["phase_alpha_ema"]
    return state


def _init_pair_predictor_state(v2_cfg: dict[str, Any]) -> dict[str, Any]:
    resolved_v2_cfg = _coerce_aligncache_v2_cfg_input(v2_cfg)
    state = {
        "cache": {"cond": None, "uncond": None},
        "shadow_cache": {"cond": None, "uncond": None},
        "shadow_prev_cache": {"cond": None, "uncond": None},
        "shadow_cache_sub": {"cond": None, "uncond": None},
        "shadow_pair_ratio_ema": {"cond": None, "uncond": None},
        "shadow_pair_dir_ema": {"cond": None, "uncond": None},
        "shadow_guided_pair_sub": None,
        "shadow_guided_pair_ratio_ema": None,
        "shadow_guided_pair_dir_ema": None,
        "shadow_pair_valid": False,
        "shadow_pair_scale_ema": {"cond": _init_phase_ema(1.0), "uncond": _init_phase_ema(1.0)},
        "pending_cond_output": None,
        "pending_cond_exact": False,
        "b2_state": _init_v2_b2_state(resolved_v2_cfg),
    }
    state["pending_cond"] = state["pending_cond_output"]
    state["anchor_state"] = state["b2_state"]
    return state


def _store_pair_branch_history(pair_state: dict[str, Any], residual: Sequence[torch.Tensor], is_cond: bool, v2_cfg: dict[str, Any]) -> None:
    branch = _branch_name(is_cond)
    residual_detached = _clone_tensor_list(residual)
    residual_sub = _tensor_list_subsample(residual_detached, int(v2_cfg["subsample_stride"]))
    last = pair_state["shadow_cache"][branch]
    last_sub = pair_state["shadow_cache_sub"][branch]
    ratio_ema = pair_state["shadow_pair_ratio_ema"][branch]
    dir_ema = pair_state["shadow_pair_dir_ema"][branch]
    if last_sub is not None and residual_sub is not None:
        prev_norm = torch.norm(last_sub, p=2).clamp_(min=1e-12)
        curr_norm = torch.norm(residual_sub, p=2)
        ratio = float((curr_norm / prev_norm).item())
        direction = _tensor_list_cosine(residual_sub, last_sub)
        ratio_ema = _update_running_ema(ratio_ema, ratio, float(v2_cfg["qc_ratio_ema_decay"]))
        dir_ema = _update_running_ema(dir_ema, direction, float(v2_cfg["qc_dir_ema_decay"]))
    pair_state["shadow_prev_cache"][branch] = last
    pair_state["shadow_cache"][branch] = residual_detached
    pair_state["shadow_cache_sub"][branch] = residual_sub
    pair_state["shadow_pair_ratio_ema"][branch] = ratio_ema
    pair_state["shadow_pair_dir_ema"][branch] = dir_ema


def _compose_guided_pair_sub(cond_sub: torch.Tensor | None, uncond_sub: torch.Tensor | None, guide_scale: float) -> torch.Tensor | None:
    if cond_sub is None or uncond_sub is None:
        return None
    return uncond_sub + (cond_sub - uncond_sub) * float(guide_scale)


def _update_guided_pair_history(pair_state: dict[str, Any], v2_cfg: dict[str, Any], guide_scale: float) -> None:
    guided_sub = _compose_guided_pair_sub(
        pair_state["shadow_cache_sub"]["cond"],
        pair_state["shadow_cache_sub"]["uncond"],
        guide_scale,
    )
    if guided_sub is None:
        pair_state["shadow_pair_valid"] = False
        return
    prev_guided = pair_state["shadow_guided_pair_sub"]
    if prev_guided is not None:
        prev_norm = torch.norm(prev_guided, p=2).clamp_(min=1e-12)
        curr_norm = torch.norm(guided_sub, p=2)
        ratio = float((curr_norm / prev_norm).item())
        direction = _tensor_list_cosine(guided_sub, prev_guided)
        pair_state["shadow_guided_pair_ratio_ema"] = _update_running_ema(
            pair_state["shadow_guided_pair_ratio_ema"],
            ratio,
            float(v2_cfg["qc_ratio_ema_decay"]),
        )
        pair_state["shadow_guided_pair_dir_ema"] = _update_running_ema(
            pair_state["shadow_guided_pair_dir_ema"],
            direction,
            float(v2_cfg["qc_dir_ema_decay"]),
        )
    pair_state["shadow_guided_pair_sub"] = guided_sub.clone()
    pair_state["shadow_pair_valid"] = True


def _pair_scale_value(pair_state: dict[str, Any], phase: str, is_cond: bool, v2_cfg: dict[str, Any]) -> float:
    if not bool(v2_cfg.get("use_online_scale", True)):
        return 1.0
    branch = _branch_name(is_cond)
    scale_state = pair_state["shadow_pair_scale_ema"][branch]
    scale = 1.0 if scale_state is None else float(scale_state[phase])
    return _clamp_float(scale, float(v2_cfg["scale_clamp_low"]), float(v2_cfg["scale_clamp_high"]))


def _pair_blend_coeff(pair_state: dict[str, Any], phase: str, is_cond: bool, v2_cfg: dict[str, Any]) -> tuple[float, float]:
    beta, ratio_tol, dir_min = _correction_params(v2_cfg, phase)
    if beta <= 0.0:
        return 0.0, 0.0
    branch = _branch_name(is_cond)
    gate = min(
        _ratio_gate(pair_state["shadow_guided_pair_ratio_ema"], ratio_tol),
        _dir_gate(pair_state["shadow_guided_pair_dir_ema"], dir_min),
        _ratio_gate(pair_state["shadow_pair_ratio_ema"][branch], ratio_tol),
        _dir_gate(pair_state["shadow_pair_dir_ema"][branch], dir_min),
    )
    beta_eff = beta * gate
    return float(beta_eff), float(gate)


def _predict_pair_residual(pair_state: dict[str, Any], phase: str, is_cond: bool, v2_cfg: dict[str, Any]) -> list[torch.Tensor] | None:
    branch = _branch_name(is_cond)
    last = pair_state["shadow_cache"][branch]
    prev = pair_state["shadow_prev_cache"][branch]
    if last is None:
        return None
    base_scale = _pair_scale_value(pair_state, phase, is_cond, v2_cfg)
    base_pred = [tensor.detach().to(torch.float32) * float(base_scale) for tensor in last]
    if not bool(v2_cfg.get("use_second_order", True)) or prev is None:
        return base_pred
    beta_eff, blend_gate = _pair_blend_coeff(pair_state, phase, is_cond, v2_cfg)
    if beta_eff <= 1e-4 or blend_gate <= 1e-4:
        return base_pred
    return [
        base + (current.detach().to(torch.float32) - previous.detach().to(torch.float32)) * beta_eff
        for base, current, previous in zip(base_pred, last, prev)
    ]


def _update_pair_scale_from_exact(pair_state: dict[str, Any], residual: Sequence[torch.Tensor], phase: str, is_cond: bool, v2_cfg: dict[str, Any]) -> None:
    if not bool(v2_cfg.get("use_online_scale", True)):
        return
    branch = _branch_name(is_cond)
    last = pair_state["shadow_cache"][branch]
    if last is None:
        return
    scale_state = pair_state["shadow_pair_scale_ema"][branch]
    fitted = _fit_affine_scale_list(residual, last, default_alpha=float(scale_state[phase]))
    fitted = _clamp_float(fitted, float(v2_cfg["scale_clamp_low"]), float(v2_cfg["scale_clamp_high"]))
    scale_state[phase] = _update_running_ema(scale_state[phase], fitted, float(v2_cfg["scale_ema_decay"]))


def _predict_pair_output(
    raw_input: Sequence[torch.Tensor],
    pair_state: dict[str, Any],
    phase: str,
    is_cond: bool,
    v2_cfg: dict[str, Any],
) -> list[torch.Tensor] | None:
    branch = _branch_name(is_cond)
    cached = pair_state["cache"][branch]
    if cached is None:
        return None
    base_output = [value + delta.to(dtype=value.dtype) for value, delta in zip(raw_input, cached)]
    if not bool(v2_cfg.get("enable", True)) or not bool(pair_state["shadow_pair_valid"]):
        return base_output
    pred_residual = _predict_pair_residual(pair_state, phase, is_cond, v2_cfg)
    if pred_residual is None:
        return base_output
    return [value + delta.to(dtype=value.dtype) for value, delta in zip(raw_input, pred_residual)]


def _write_pair_branch_result(
    pair_state: dict[str, Any],
    raw_input: Sequence[torch.Tensor],
    output: Sequence[torch.Tensor],
    is_cond: bool,
    phase: str,
    learn_from_exact: bool,
    v2_cfg: dict[str, Any],
    guide_scale: float,
) -> None:
    residual = [(u - v).detach() for u, v in zip(output, raw_input)]
    if learn_from_exact:
        _update_pair_scale_from_exact(pair_state, residual, phase, is_cond, v2_cfg)
    _store_pair_branch_history(pair_state, residual, is_cond, v2_cfg)
    if not bool(is_cond):
        _update_guided_pair_history(pair_state, v2_cfg, guide_scale)


def _record_v2_b2_event(pair_state: dict[str, Any], key: str, *, enabled: bool = True) -> None:
    if not enabled:
        return
    counts = pair_state["b2_state"]["counts"]
    if key in counts:
        counts[key] += 1


def _clear_v2_pending_cond_state(pair_state: dict[str, Any]) -> None:
    pair_state["pending_cond_output"] = None
    pair_state["pending_cond_exact"] = False
    pair_state["pending_cond"] = None


def _set_v2_pending_cond_state(pair_state: dict[str, Any], output: Sequence[torch.Tensor], exact_pair: bool) -> None:
    pair_state["pending_cond_output"] = _clone_tensor_list(output)
    pair_state["pending_cond_exact"] = bool(exact_pair)
    pair_state["pending_cond"] = pair_state["pending_cond_output"]


def _predict_v2_b2_uncond(
    pair_state: dict[str, Any],
    phase: str,
    step_idx: int,
    cond_output: Sequence[torch.Tensor] | None,
    base_output: Sequence[torch.Tensor] | None,
    v2_cfg: dict[str, Any],
    *,
    record_event: bool = True,
) -> list[torch.Tensor] | None:
    if not bool(v2_cfg.get("enable", True)) or not bool(v2_cfg.get("b2_enable", True)):
        return None
    state = pair_state["b2_state"]
    if cond_output is None or base_output is None:
        _record_v2_b2_event(pair_state, "reject_no_anchor", enabled=record_event)
        return None
    if state["anchor_cond"] is None or state["anchor_uncond"] is None:
        _record_v2_b2_event(pair_state, "reject_no_anchor", enabled=record_event)
        return None
    if bool(v2_cfg.get("b2_exact_early", True)) and phase == "early":
        _record_v2_b2_event(pair_state, "reject_early_exact", enabled=record_event)
        return None
    max_pred_age = {
        "early": 0,
        "mid": int(v2_cfg["b2_max_pred_age_mid"]),
        "late": int(v2_cfg["b2_max_pred_age_late"]),
    }
    last_exact_step = state.get("last_exact_pair", state.get("last_exact_step", -1))
    pred_age = int(step_idx) - int(last_exact_step)
    if pred_age > max_pred_age[phase]:
        _record_v2_b2_event(pair_state, "reject_max_age", enabled=record_event)
        return None
    cond_rel_cap = {
        "early": 0.0,
        "mid": float(v2_cfg["b2_cond_rel_cap_mid"]),
        "late": float(v2_cfg["b2_cond_rel_cap_late"]),
    }
    cond_rel = _tensor_list_rel_mse(cond_output, state["anchor_cond"])
    cond_rel = 0.0 if cond_rel is None else cond_rel
    if cond_rel > cond_rel_cap[phase]:
        _record_v2_b2_event(pair_state, "reject_cond_gate", enabled=record_event)
        return None
    alpha = float(state["phase_alpha_ema"][phase])
    b2_output = [
        anchor_uncond.detach().to(torch.float32)
        + (cond_value.detach().to(torch.float32) - anchor_cond.detach().to(torch.float32)) * alpha
        for cond_value, anchor_cond, anchor_uncond in zip(cond_output, state["anchor_cond"], state["anchor_uncond"])
    ]
    disagree_cap = {
        "early": 0.0,
        "mid": float(v2_cfg["b2_disagree_rel_cap_mid"]),
        "late": float(v2_cfg["b2_disagree_rel_cap_late"]),
    }
    disagree_rel = _tensor_list_rel_mse(b2_output, base_output)
    disagree_rel = 0.0 if disagree_rel is None else disagree_rel
    if disagree_rel > disagree_cap[phase]:
        _record_v2_b2_event(pair_state, "reject_disagree", enabled=record_event)
        return None
    blend_base = {
        "early": 0.0,
        "mid": float(v2_cfg["b2_blend_mid"]),
        "late": float(v2_cfg["b2_blend_late"]),
    }
    blend_weight = blend_base[phase]
    if blend_weight <= 1e-6:
        _record_v2_b2_event(pair_state, "reject_weak_gate", enabled=record_event)
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
            1.0 - max(float(pred_age) - 1.0, 0.0) / max(float(max_pred_age[phase] - 1), 1.0),
        )
    blend_weight *= min(cond_gate, disagree_gate, age_gate)
    if blend_weight <= 1e-4:
        _record_v2_b2_event(pair_state, "reject_weak_gate", enabled=record_event)
        return None
    _record_v2_b2_event(pair_state, "applied_steps", enabled=record_event)
    return _blend_tensor_lists(base_output, b2_output, blend_weight)


def _update_v2_b2_exact_state(
    pair_state: dict[str, Any],
    phase: str,
    step_idx: int,
    cond_output: Sequence[torch.Tensor] | None,
    uncond_output: Sequence[torch.Tensor] | None,
    learn: bool,
    v2_cfg: dict[str, Any],
) -> None:
    if not bool(v2_cfg.get("enable", True)) or not bool(v2_cfg.get("b2_enable", True)):
        return
    state = pair_state["b2_state"]
    if cond_output is None or uncond_output is None:
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
        alpha = _fit_affine_scale_list(uncond_delta, cond_delta, default_alpha=float(state["phase_alpha_ema"][phase]))
        decay = float(v2_cfg["b2_alpha_ema_decay"])
        state["phase_alpha_ema"][phase] = decay * float(state["phase_alpha_ema"][phase]) + (1.0 - decay) * alpha
    state["anchor_cond"] = _clone_tensor_list(cond_output)
    state["anchor_uncond"] = _clone_tensor_list(uncond_output)
    state["last_exact_step"] = int(step_idx)
    state["last_exact_pair"] = int(step_idx)
    state["counts"]["anchor_updates"] += 1


def _init_anchor_state(pair_cfg: dict[str, Any]) -> dict[str, Any]:
    return _init_v2_b2_state(_coerce_aligncache_v2_cfg_input(pair_cfg))


def _predict_anchor_uncond(
    pair_state: dict[str, Any],
    phase: str,
    step_idx: int,
    cond_output: Sequence[torch.Tensor] | None,
    base_output: Sequence[torch.Tensor] | None,
    pair_cfg: dict[str, Any],
    *,
    record_event: bool = True,
) -> list[torch.Tensor] | None:
    return _predict_v2_b2_uncond(
        pair_state,
        phase,
        step_idx,
        cond_output,
        base_output,
        _coerce_aligncache_v2_cfg_input(pair_cfg),
        record_event=record_event,
    )


def _update_anchor_state(
    pair_state: dict[str, Any],
    phase: str,
    step_idx: int,
    cond_output: Sequence[torch.Tensor] | None,
    uncond_output: Sequence[torch.Tensor] | None,
    learn: bool,
    pair_cfg: dict[str, Any],
) -> None:
    _update_v2_b2_exact_state(
        pair_state,
        phase,
        step_idx,
        cond_output,
        uncond_output,
        learn,
        _coerce_aligncache_v2_cfg_input(pair_cfg),
    )


def _init_aligncache_feedback_state(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = dict(cfg or {})
    scale_init = _resolve_phase_mapping(_ALIGNCACHE_DEFAULT_SCALE_INIT, cfg.get("scale_init"), float)
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


def _init_fb_state(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    return _init_aligncache_feedback_state(cfg)


def _reset_aligncache_feedback_pending_feedback(state: dict[str, Any]) -> None:
    state["pending_phase"] = None
    state["pending_budget"] = None
    state["pending_feedback_eligible"] = False
    state["pending_cond_draft"] = None
    state["pending_uncond_draft"] = None


def _reset_fb_state(state: dict[str, Any]) -> None:
    _reset_aligncache_feedback_pending_feedback(state)


def _fb_should_collect_feedback(reason: str, budget_next: float | None) -> bool:
    return budget_next is not None and str(reason) == "budget_exact"


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


def _fb_pair_age(current_pair_idx: int, last_pair_idx: int | None) -> int:
    if last_pair_idx is None or int(last_pair_idx) < 0:
        return int(current_pair_idx) + 1
    return int(current_pair_idx) - int(last_pair_idx)


def _fb_complete_pair(state: dict[str, Any], pair_idx: int, exec_mode: str) -> None:
    anchor_age = _fb_pair_age(pair_idx, state["last_exact_pair_idx"])
    state["summary"]["max_anchor_age"] = max(int(state["summary"]["max_anchor_age"]), int(anchor_age))
    if exec_mode == "exact_pair":
        state["last_exact_pair_idx"] = int(pair_idx)
        state["summary"]["exact_pairs"] += 1
    else:
        state["summary"]["skipped_pairs"] += 1
    state["current_exec_mode"] = str(exec_mode)
    probe = state.get("last_probe")
    if isinstance(probe, dict):
        probe["exec_mode"] = str(exec_mode)


def _fb_guided_output(
    cond_output: Sequence[torch.Tensor],
    uncond_output: Sequence[torch.Tensor],
    guide_scale: float,
) -> list[torch.Tensor]:
    return [
        uncond.detach().to(torch.float32)
        + (cond.detach().to(torch.float32) - uncond.detach().to(torch.float32)) * float(guide_scale)
        for cond, uncond in zip(cond_output, uncond_output)
    ]


def _fb_predict_cond_feedback_draft(
    raw_input: Sequence[torch.Tensor],
    phase: str,
    pair_state: dict[str, Any],
    v2_cfg: dict[str, Any],
) -> list[torch.Tensor] | None:
    draft_output = _predict_pair_output(raw_input, pair_state, phase, True, v2_cfg)
    return None if draft_output is None else _clone_tensor_list(draft_output)


def _fb_predict_uncond_feedback_draft(
    raw_input: Sequence[torch.Tensor],
    phase: str,
    pair_idx: int,
    cond_draft: Sequence[torch.Tensor] | None,
    pair_state: dict[str, Any],
    v2_cfg: dict[str, Any],
) -> list[torch.Tensor] | None:
    draft_output = _predict_pair_output(raw_input, pair_state, phase, False, v2_cfg)
    if draft_output is None:
        return None
    b2_output = _predict_v2_b2_uncond(
        pair_state,
        phase,
        pair_idx,
        cond_draft,
        draft_output,
        v2_cfg,
        record_event=False,
    )
    if b2_output is not None:
        draft_output = b2_output
    return _clone_tensor_list(draft_output)


def _fb_update_from_exact_feedback(
    state: dict[str, Any],
    cfg: dict[str, Any],
    cond_output: Sequence[torch.Tensor] | None,
    uncond_output: Sequence[torch.Tensor] | None,
    guide_scale: float,
) -> None:
    if not bool(state.get("pending_feedback_eligible")):
        _reset_aligncache_feedback_pending_feedback(state)
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
        _reset_aligncache_feedback_pending_feedback(state)
        return
    guided_draft = _fb_guided_output(cond_draft, uncond_draft, guide_scale)
    guided_exact = _fb_guided_output(cond_output, uncond_output, guide_scale)
    obs_err = _tensor_list_rel_mse(guided_draft, guided_exact)
    if obs_err is None:
        state["summary"]["feedback_skipped_pairs"] += 1
        _reset_aligncache_feedback_pending_feedback(state)
        return
    err_ratio = float(obs_err) / max(float(budget), float(cfg["score_eps"]))
    clipped_ratio = _clamp_float(err_ratio, float(cfg["ratio_clip_low"]), float(cfg["ratio_clip_high"]))
    prev_ratio_ema = float(state["ratio_ema"][phase])
    updated_ratio_ema = float(cfg["ratio_ema_decay"]) * prev_ratio_ema + (1.0 - float(cfg["ratio_ema_decay"])) * clipped_ratio
    state["ratio_ema"][phase] = float(updated_ratio_ema)
    state["scale"][phase] = _clamp_float(
        1.0 / max(float(updated_ratio_ema), float(cfg["score_eps"])),
        float(cfg["scale_low"][phase]),
        float(cfg["scale_high"][phase]),
    )
    state["summary"]["feedback_updates"] += 1
    _reset_aligncache_feedback_pending_feedback(state)


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


def _advance_aligncache_counter(module: Any, *, step_limit: int) -> None:
    module.cnt = int(getattr(module, "cnt", 0)) + 1
    if int(getattr(module, "cnt", 0)) >= max(int(step_limit), 1):
        module.cnt = 0
        pair_state = getattr(module, "aligncache_pair_state", None)
        if isinstance(pair_state, dict):
            _clear_v2_pending_cond_state(pair_state)
        state = getattr(module, "aligncache_state", None)
        if isinstance(state, dict):
            _reset_aligncache_feedback_pending_feedback(state)


def _split_cfg_batch(raw_input: torch.Tensor, branch_order: Sequence[str]) -> dict[str, torch.Tensor] | None:
    if not isinstance(raw_input, torch.Tensor):
        return None
    if raw_input.ndim < 1 or int(raw_input.shape[0]) < 2 or int(raw_input.shape[0]) % 2 != 0:
        return None
    first, second = raw_input.chunk(2, dim=0)
    return {str(branch_order[0]): first, str(branch_order[1]): second}


def _merge_cfg_batch(branch_values: dict[str, Sequence[torch.Tensor]], branch_order: Sequence[str]) -> torch.Tensor:
    return torch.cat(
        [branch_values[str(branch_order[0])][0], branch_values[str(branch_order[1])][0]],
        dim=0,
    )


def reset_single_aligncache_state(module: Any, *, sample_steps: int) -> None:
    cfg = dict(getattr(module, "aligncache_cfg", {}) or {})
    module.aligncache_state = _init_aligncache_feedback_state(cfg)
    module.aligncache_pair_state = _init_pair_predictor_state(
        dict(cfg.get("pair_cfg") or cfg.get("v2_cfg") or _ALIGNCACHE_V2_DEFAULTS)
    )
    module.aligncache_previous_raw_input_cond = None
    module.aligncache_previous_raw_output_cond = None
    module.aligncache_prev_prev_raw_input_cond = None
    module.aligncache_k = None
    module.accumulated_error = 0.0
    module.cnt = 0
    module.num_steps = max(int(sample_steps), 1)
    module.v2_current_phase_threshold = None


def prepare_single_aligncache_step(
    module: Any,
    *,
    raw_input: torch.Tensor | Sequence[torch.Tensor],
) -> dict[str, Any] | None:
    if not bool(getattr(module, "enable_aligncache", False)):
        return None
    branch_order = tuple(getattr(module, "aligncache_branch_order", ()) or ())
    if len(branch_order) != 2 or not isinstance(raw_input, torch.Tensor):
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
        reset_single_aligncache_state(module, sample_steps=int(getattr(module, "num_steps", 1)))
        cfg = dict(getattr(module, "aligncache_cfg", {}) or {})
    state = getattr(module, "aligncache_state")
    pair_state = getattr(module, "aligncache_pair_state")
    v2_cfg = dict(cfg.get("v2_cfg") or _coerce_aligncache_v2_cfg_input(cfg.get("pair_cfg")))
    step_idx = int(getattr(module, "cnt", 0))
    total_steps = max(int(getattr(module, "num_steps", 1)), 1)
    phase = _get_phase(step_idx, total_steps, v2_cfg)
    base_phase_threshold = float(cfg["phase_thresholds"][phase])
    threshold_scale = float(state["scale"][phase])
    phase_threshold = float(base_phase_threshold) * float(threshold_scale)
    current_ratio_ema = float(state["ratio_ema"][phase])
    module.v2_current_phase_threshold = float(base_phase_threshold)

    cond_raw_tensor = branches["cond"]
    uncond_raw_tensor = branches["uncond"]
    cond_raw = _tensor_list(cond_raw_tensor)
    uncond_raw = _tensor_list(uncond_raw_tensor)
    assert cond_raw is not None and uncond_raw is not None

    budget_next = None
    pred_change_base = None
    if step_idx < int(cfg["cache_ret_steps"]) or step_idx >= max(total_steps - 1, 0):
        exec_mode = "exact_pair"
        reason = "warmup_tail"
        module.accumulated_error = 0.0
        anchor_age = _fb_pair_age(step_idx, int(state["last_exact_pair_idx"]))
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
        anchor_age = int(route["age"])
        exec_mode = str(route["exec_mode"])
        reason = str(route["reason"])
        feedback_eligible = bool(exec_mode == "exact_pair" and _fb_should_collect_feedback(reason, budget_next))
        if exec_mode == "exact_pair":
            module.accumulated_error = 0.0
        else:
            module.accumulated_error = float(budget_next or 0.0)

    predicted_output = None
    if exec_mode == "skip_pair":
        cond_pred = _predict_pair_output(cond_raw, pair_state, phase, True, v2_cfg)
        if cond_pred is not None:
            _set_v2_pending_cond_state(pair_state, cond_pred, False)
        uncond_pred = _predict_pair_output(uncond_raw, pair_state, phase, False, v2_cfg)
        if cond_pred is not None and uncond_pred is not None:
            b2_output = _predict_v2_b2_uncond(
                pair_state,
                phase,
                step_idx,
                pair_state["pending_cond_output"],
                uncond_pred,
                v2_cfg,
            )
            if b2_output is not None:
                uncond_pred = b2_output
            predicted_output = _merge_cfg_batch(
                {"cond": cond_pred, "uncond": uncond_pred},
                branch_order,
            )
        if predicted_output is None:
            exec_mode = "exact_pair"
            reason = "cache_miss_exact"
            feedback_eligible = False
            module.accumulated_error = 0.0
            _clear_v2_pending_cond_state(pair_state)

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
    if exec_mode == "exact_pair" and feedback_eligible:
        state["pending_cond_draft"] = _fb_predict_cond_feedback_draft(cond_raw, phase, pair_state, v2_cfg)
        if state["pending_cond_draft"] is not None:
            state["pending_uncond_draft"] = _fb_predict_uncond_feedback_draft(
                uncond_raw,
                phase,
                step_idx,
                state["pending_cond_draft"],
                pair_state,
                v2_cfg,
            )
    module.aligncache_previous_raw_input_cond = cond_raw_tensor.detach().clone()
    return {
        "kind": "pair",
        "phase": phase,
        "step_idx": int(step_idx),
        "exec_mode": "skip" if exec_mode == "skip_pair" else "exact",
        "pair_exec_mode": exec_mode,
        "reason": reason,
        "predicted_output": predicted_output,
    }


def complete_single_aligncache_skip(
    module: Any,
    *,
    raw_input: torch.Tensor | Sequence[torch.Tensor],
    predicted_output: torch.Tensor | Sequence[torch.Tensor],
    decision: dict[str, Any],
) -> list[torch.Tensor]:
    if str(decision.get("kind")) != "pair" or not isinstance(raw_input, torch.Tensor) or not isinstance(predicted_output, torch.Tensor):
        return _tensor_list(predicted_output) or []
    branch_order = tuple(getattr(module, "aligncache_branch_order", ()) or ())
    raw_branches = _split_cfg_batch(raw_input, branch_order)
    pred_branches = _split_cfg_batch(predicted_output, branch_order)
    assert raw_branches is not None and pred_branches is not None
    pair_state = getattr(module, "aligncache_pair_state")
    v2_cfg = dict(getattr(module, "aligncache_cfg", {})["v2_cfg"])
    guide_scale = float(getattr(module, "sample_guide_scale", getattr(module, "aligncache_cfg", {}).get("guide_scale") or 5.0))
    _write_pair_branch_result(
        pair_state,
        _tensor_list(raw_branches["cond"]) or [],
        _tensor_list(pred_branches["cond"]) or [],
        True,
        str(decision["phase"]),
        False,
        v2_cfg,
        guide_scale,
    )
    _write_pair_branch_result(
        pair_state,
        _tensor_list(raw_branches["uncond"]) or [],
        _tensor_list(pred_branches["uncond"]) or [],
        False,
        str(decision["phase"]),
        False,
        v2_cfg,
        guide_scale,
    )
    _clear_v2_pending_cond_state(pair_state)
    _fb_complete_pair(getattr(module, "aligncache_state"), int(decision["step_idx"]), "skip_pair")
    _reset_aligncache_feedback_pending_feedback(getattr(module, "aligncache_state"))
    _advance_aligncache_counter(module, step_limit=int(getattr(module, "num_steps", 1)))
    return [predicted_output]


def complete_single_aligncache_exact(
    module: Any,
    *,
    raw_input: torch.Tensor | Sequence[torch.Tensor],
    exact_output: torch.Tensor | Sequence[torch.Tensor],
    decision: dict[str, Any],
) -> list[torch.Tensor]:
    if str(decision.get("kind")) != "pair" or not isinstance(raw_input, torch.Tensor) or not isinstance(exact_output, torch.Tensor):
        _advance_aligncache_counter(module, step_limit=int(getattr(module, "num_steps", 1)))
        return _tensor_list(exact_output) or []
    branch_order = tuple(getattr(module, "aligncache_branch_order", ()) or ())
    raw_branches = _split_cfg_batch(raw_input, branch_order)
    output_branches = _split_cfg_batch(exact_output, branch_order)
    assert raw_branches is not None and output_branches is not None
    cfg = dict(getattr(module, "aligncache_cfg", {}) or {})
    pair_state = getattr(module, "aligncache_pair_state")
    state = getattr(module, "aligncache_state")
    v2_cfg = dict(cfg.get("v2_cfg") or _coerce_aligncache_v2_cfg_input(cfg.get("pair_cfg")))
    guide_scale = float(getattr(module, "sample_guide_scale", cfg.get("guide_scale") or 5.0))

    cond_raw_tensor = raw_branches["cond"]
    cond_output_tensor = output_branches["cond"]
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

    cond_raw = _tensor_list(cond_raw_tensor) or []
    uncond_raw = _tensor_list(raw_branches["uncond"]) or []
    cond_output = _tensor_list(cond_output_tensor) or []
    uncond_output = _tensor_list(output_branches["uncond"]) or []
    pair_state["cache"]["cond"] = [(u - v).detach() for u, v in zip(cond_output, cond_raw)]
    pair_state["cache"]["uncond"] = [(u - v).detach() for u, v in zip(uncond_output, uncond_raw)]
    _write_pair_branch_result(pair_state, cond_raw, cond_output, True, str(decision["phase"]), True, v2_cfg, guide_scale)
    _write_pair_branch_result(pair_state, uncond_raw, uncond_output, False, str(decision["phase"]), True, v2_cfg, guide_scale)
    _update_v2_b2_exact_state(pair_state, str(decision["phase"]), int(decision["step_idx"]), cond_output, uncond_output, True, v2_cfg)
    _fb_update_from_exact_feedback(state, cfg, cond_output, uncond_output, guide_scale)
    _clear_v2_pending_cond_state(pair_state)
    _fb_complete_pair(state, int(decision["step_idx"]), "exact_pair")
    _advance_aligncache_counter(module, step_limit=int(getattr(module, "num_steps", 1)))
    return [exact_output]


def _reset_wan_pair_state(model: Any) -> None:
    cfg = dict(getattr(model, "aligncache_cfg", {}) or {})
    model.aligncache_state = _init_aligncache_feedback_state(cfg)
    model.aligncache_pair_state = _init_pair_predictor_state(
        dict(cfg.get("pair_cfg") or cfg.get("v2_cfg") or _ALIGNCACHE_V2_DEFAULTS)
    )
    model.aligncache_previous_raw_input_even = None
    model.aligncache_previous_raw_output_even = None
    model.aligncache_prev_prev_raw_input_even = None
    model.aligncache_k = None
    model.accumulated_error_even = 0.0
    model.should_calc_current_pair = True
    model.cnt = 0
    model.v2_current_phase_threshold = None


def _advance_wan_counter(model: Any) -> None:
    _advance_aligncache_counter(model, step_limit=int(getattr(model, "num_steps", 1)))


def restore_wan_aligncache_state(pipeline: Any) -> None:
    model = pipeline.model
    original_forward = getattr(model, "_vbench_aligncache_original_forward", None)
    if original_forward is not None:
        model.forward = original_forward
    model.enable_aligncache = False
    _reset_wan_pair_state(model)


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
    model.aligncache_cfg = resolve_wan_aligncache_config(
        task=str(task),
        method_config=method_config,
        sample_steps=int(sample_steps),
    )
    model.num_steps = max(int(sample_steps) * 2, 1)
    model.ret_steps = int(model.aligncache_cfg["cache_ret_steps"]) * 2
    model.cutoff_steps = max(int(sample_steps) * 2 - 2, 1)
    model.sample_guide_scale = float(
        getattr(pipeline, "sample_guide_scale", model.aligncache_cfg.get("guide_scale") or getattr(model, "sample_guide_scale", 5.0))
    )
    _reset_wan_pair_state(model)
    if not enabled:
        return None
    return dict(model.aligncache_cfg)


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
    assert raw_input is not None
    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)

    if y is not None:
        x = [torch.cat([current, extra], dim=0) for current, extra in zip(x, y)]

    cfg = dict(getattr(self, "aligncache_cfg", {}) or {})
    v2_cfg = dict(cfg.get("v2_cfg") or _coerce_aligncache_v2_cfg_input(cfg.get("pair_cfg")))
    state = getattr(self, "aligncache_state")
    pair_state = getattr(self, "aligncache_pair_state")
    is_cond = bool(int(getattr(self, "cnt", 0)) % 2 == 0)
    pair_idx = int(getattr(self, "cnt", 0)) // 2
    total_pairs = max(int(getattr(self, "num_steps", 1)) // 2, 1)
    phase = _get_phase(pair_idx, total_pairs, v2_cfg)
    base_phase_threshold = float(cfg["phase_thresholds"][phase])
    threshold_scale = float(state["scale"][phase])
    current_phase_threshold = float(base_phase_threshold) * float(threshold_scale)
    current_ratio_ema = float(state["ratio_ema"][phase])
    self.v2_current_phase_threshold = float(base_phase_threshold)

    if is_cond:
        pred_change_base = None
        budget_next = None
        anchor_age = _fb_pair_age(pair_idx, state["last_exact_pair_idx"])
        exec_mode = "exact_pair"
        feedback_eligible = False
        if self.cnt < self.ret_steps or self.cnt >= self.cutoff_steps:
            self.should_calc_current_pair = True
            self.accumulated_error_even = 0.0
            decision_reason = "warmup_tail"
        else:
            if self.aligncache_previous_raw_input_even is not None and self.aligncache_previous_raw_output_even is not None:
                raw_input_change = torch.cat(
                    [(u - v).flatten() for u, v in zip(raw_input, self.aligncache_previous_raw_input_even)]
                ).abs().mean()
                output_norm = _tensor_list_mean_abs(self.aligncache_previous_raw_output_even)
                if output_norm is not None and self.aligncache_k is not None:
                    motion = float(raw_input_change.item()) / max(float(output_norm), float(cfg["score_eps"]))
                    pred_change_base = float(self.aligncache_k) * float(motion)
                    budget_next = float(self.accumulated_error_even) + float(pred_change_base)
            route = _fb_route_pair(state, phase, pair_idx, budget_next, current_phase_threshold, cfg)
            anchor_age = int(route["age"])
            exec_mode = str(route["exec_mode"])
            decision_reason = str(route["reason"])
            self.should_calc_current_pair = exec_mode == "exact_pair"
            feedback_eligible = self.should_calc_current_pair and _fb_should_collect_feedback(decision_reason, budget_next)
            if self.should_calc_current_pair:
                self.accumulated_error_even = 0.0
            else:
                self.accumulated_error_even = float(budget_next)

        if exec_mode == "skip_pair" and (pair_state["cache"]["cond"] is None or pair_state["cache"]["uncond"] is None):
            self.should_calc_current_pair = True
            exec_mode = "exact_pair"
            feedback_eligible = False
            decision_reason = "cache_miss_exact"
            self.accumulated_error_even = 0.0

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
        if self.should_calc_current_pair and feedback_eligible:
            state["pending_cond_draft"] = _fb_predict_cond_feedback_draft(raw_input, phase, pair_state, v2_cfg)
        self.aligncache_previous_raw_input_even = _clone_tensor_list(raw_input)

    if is_cond and not bool(getattr(self, "should_calc_current_pair", True)) and pair_state["cache"]["cond"] is not None:
        pred_output = _predict_pair_output(raw_input, pair_state, phase, True, v2_cfg)
        if pred_output is None:
            pred_output = [(u + v).detach() for u, v in zip(raw_input, pair_state["cache"]["cond"])]
        _set_v2_pending_cond_state(pair_state, pred_output, False)
        if bool(v2_cfg.get("enable", True)):
            _write_pair_branch_result(pair_state, raw_input, pred_output, True, phase, False, v2_cfg, float(self.sample_guide_scale))
        _advance_wan_counter(self)
        return [value.float() for value in pred_output]

    current_exec_mode = state.get("current_exec_mode", "exact_pair")
    if (not is_cond) and current_exec_mode == "skip_pair" and pair_state["cache"]["uncond"] is not None:
        pred_output = _predict_pair_output(raw_input, pair_state, phase, False, v2_cfg)
        if pred_output is None:
            pred_output = [(u + v).detach() for u, v in zip(raw_input, pair_state["cache"]["uncond"])]
        b2_output = _predict_v2_b2_uncond(
            pair_state,
            phase,
            pair_idx,
            pair_state.get("pending_cond_output"),
            pred_output,
            v2_cfg,
        )
        if b2_output is not None:
            pred_output = b2_output
        if bool(v2_cfg.get("enable", True)):
            _write_pair_branch_result(pair_state, raw_input, pred_output, False, phase, False, v2_cfg, float(self.sample_guide_scale))
        _clear_v2_pending_cond_state(pair_state)
        _fb_complete_pair(state, pair_idx, "skip_pair")
        _reset_aligncache_feedback_pending_feedback(state)
        _advance_wan_counter(self)
        return [value.float() for value in pred_output]

    if (not is_cond) and bool(getattr(self, "should_calc_current_pair", True)) and bool(state.get("pending_feedback_eligible")) and state["pending_uncond_draft"] is None:
        state["pending_uncond_draft"] = _fb_predict_uncond_feedback_draft(
            raw_input,
            phase,
            pair_idx,
            state.get("pending_cond_draft"),
            pair_state,
            v2_cfg,
        )

    x = [self.patch_embedding(value.unsqueeze(0)) for value in x]
    grid_sizes = torch.stack([torch.tensor(value.shape[2:], dtype=torch.long) for value in x])
    x = [value.flatten(2).transpose(1, 2) for value in x]
    seq_lens = torch.tensor([value.size(1) for value in x], dtype=torch.long)
    assert seq_lens.max() <= seq_len
    x = torch.cat(
        [torch.cat([value, value.new_zeros(1, seq_len - value.size(1), value.size(2))], dim=1) for value in x]
    )

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

    if is_cond:
        if self.aligncache_previous_raw_output_even is not None and self.aligncache_prev_prev_raw_input_even is not None:
            output_change = torch.cat([(u - v).flatten() for u, v in zip(output, self.aligncache_previous_raw_output_even)]).abs().mean()
            input_change = torch.cat(
                [(u - v).flatten() for u, v in zip(self.aligncache_previous_raw_input_even, self.aligncache_prev_prev_raw_input_even)]
            ).abs().mean()
            if float(input_change.item()) > 0.0:
                self.aligncache_k = float(output_change.item()) / max(float(input_change.item()), float(cfg["score_eps"]))
        self.aligncache_prev_prev_raw_input_even = _clone_tensor_list(self.aligncache_previous_raw_input_even)
        self.aligncache_previous_raw_output_even = _clone_tensor_list(output)
        pair_state["cache"]["cond"] = [u.detach() - v.detach() for u, v in zip(output, raw_input)]
        _set_v2_pending_cond_state(pair_state, output, True)
        if bool(v2_cfg.get("enable", True)):
            _write_pair_branch_result(pair_state, raw_input, output, True, phase, True, v2_cfg, float(self.sample_guide_scale))
        _advance_wan_counter(self)
        return [value.float() for value in output]

    pair_state["cache"]["uncond"] = [u.detach() - v.detach() for u, v in zip(output, raw_input)]
    if bool(v2_cfg.get("enable", True)):
        _write_pair_branch_result(pair_state, raw_input, output, False, phase, True, v2_cfg, float(self.sample_guide_scale))
        _update_v2_b2_exact_state(
            pair_state,
            phase,
            pair_idx,
            pair_state.get("pending_cond_output"),
            output,
            bool(pair_state.get("pending_cond_exact", False)),
            v2_cfg,
        )
    _fb_update_from_exact_feedback(
        state,
        cfg,
        pair_state.get("pending_cond_output"),
        output,
        float(self.sample_guide_scale),
    )
    _clear_v2_pending_cond_state(pair_state)
    _fb_complete_pair(state, pair_idx, "exact_pair")
    _advance_wan_counter(self)
    return [value.float() for value in output]


def configure_hunyuan_aligncache(
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
    enabled = str(method_name) == "aligncache"
    transformer.enable_aligncache = bool(enabled)
    transformer.aligncache_cfg = resolve_hunyuan_aligncache_config(method_config, sample_steps=int(sample_steps))
    transformer.aligncache_branch_order = ("uncond", "cond")
    reset_single_aligncache_state(transformer, sample_steps=int(sample_steps))
    if not enabled:
        return None
    return dict(transformer.aligncache_cfg)


def configure_opensora_aligncache(
    model: Any,
    *,
    enabled: bool,
    method_config: dict[str, Any] | None,
    sample_steps: int,
) -> dict[str, Any] | None:
    model.enable_aligncache = bool(enabled)
    model.aligncache_cfg = resolve_opensora_aligncache_config(method_config, sample_steps=int(sample_steps))
    model.aligncache_branch_order = ("cond", "uncond")
    reset_single_aligncache_state(model, sample_steps=int(sample_steps))
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
from typing import Any

import numpy as np
import torch
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



import argparse
import json
import os
import sys
import time
from pathlib import Path
from pprint import pformat

import colossalai
from colossalai.cluster import DistCoordinator
from mmengine.runner import set_random_seed
from tqdm import tqdm

from opensora.acceleration.parallel_states import set_sequence_parallel_group
from opensora.datasets import save_sample
from opensora.datasets.aspect import get_image_size, get_num_frames
from opensora.models.text_encoder.t5 import text_preprocessing
from opensora.registry import MODELS, SCHEDULERS, build_module
from opensora.utils.config_utils import parse_configs
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
    parser.add_argument("--base-thresh", type=float, default=0.03, help="AlignCache base threshold.")
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
    parser.add_argument("--ratio-ema-decay", type=float, default=0.90, help="Threshold feedback EMA decay.")
    parser.add_argument("--ratio-clip-low", type=float, default=0.85, help="Threshold feedback ratio clip low.")
    parser.add_argument("--ratio-clip-high", type=float, default=1.25, help="Threshold feedback ratio clip high.")
    parser.add_argument(
        "--summary-json",
        type=str,
        default=None,
        help="Optional path to dump the resolved AlignCache configuration as JSON.",
    )
    return parser.parse_known_args(sys.argv[1:])


def _load_cfg(remaining_argv: list[str]):
    original_argv = list(sys.argv)
    sys.argv = [sys.argv[0], *remaining_argv]
    try:
        return parse_configs(training=False)
    finally:
        sys.argv = original_argv


def _build_method_config(parsed: argparse.Namespace) -> dict[str, object]:
    method_config: dict[str, object] = {
        "feedback_enabled": True,
        "base_thresh": float(parsed.base_thresh),
        "warmup_pairs": int(parsed.warmup_pairs),
        "pair_enable": bool(parsed.pair_enable),
        "pair_aggr": float(parsed.pair_aggr),
        "pair_stride": int(parsed.pair_stride),
        "anchor_enable": bool(parsed.anchor_enable),
        "fb_ratio_ema_decay": float(parsed.ratio_ema_decay),
        "fb_ratio_clip_low": float(parsed.ratio_clip_low),
        "fb_ratio_clip_high": float(parsed.ratio_clip_high),
    }
    if parsed.phase_scale_profile:
        method_config["phase_scale_profile"] = str(parsed.phase_scale_profile)
    return method_config


def main() -> None:
    torch.set_grad_enabled(False)
    aligncache_args, remaining_argv = _parse_aligncache_args()
    cfg = _load_cfg(remaining_argv)

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
