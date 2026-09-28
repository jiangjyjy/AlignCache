"""Qwen flow predictors and legacy/paper AlignCache controllers.

All cached anchors are exact outputs. CFG pairing is independent of physical
model batching; normalized denoising residuals are represented as velocities.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch

PREDICTORS = (
    "reuse",
    "scaled_output_momentum",
    "scaled_input_residual",
    "time_normalized_output",
)
SAMPLING_RULE = "stratified_hash_v1"


def sample(x: torch.Tensor, limit: int = 4096) -> torch.Tensor:
    flat = x.detach().reshape(-1)
    if limit < 1:
        raise ValueError("Sampling limit must be positive")
    if flat.numel() <= limit:
        return flat.float().clone()
    # One fixed hash-selected offset per disjoint stratum. A regular stride of
    # 64 on [1,64,64,64] observes only column zero, missing the image interior.
    # Hashing indices keeps spatial coverage without touching the noise RNG.
    index = torch.arange(limit, device=flat.device, dtype=torch.int64)
    starts = index * flat.numel() // limit
    widths = (index + 1) * flat.numel() // limit - starts
    hashed = ((index + 1) * 0x45D9F3B) & 0xFFFFFFFF
    hashed = ((hashed ^ (hashed >> 16)) * 0x45D9F3B) & 0xFFFFFFFF
    hashed = hashed ^ (hashed >> 16)
    selected = starts + hashed % widths
    return flat[selected].float().clone()


def predict(name, latest, previous, x, timestep, alpha=1.0, beta=0.0):
    """Anchors contain exact outputs only; returned tensor never aliases them."""
    a = latest.output.float()
    if name == "reuse":
        return latest.output.clone()
    b = previous.output.float()
    sa, sb = sample(a), sample(b)
    na, nb = sa.norm(), sb.norm()
    cosine = torch.dot(sa, sb) / (na * nb).clamp_min(1e-12)
    gate = cosine.clamp(0, 1) * (
        torch.minimum(na, nb) / torch.maximum(na, nb).clamp_min(1e-12)
    )
    multiplier = 1.0
    if name == "time_normalized_output":
        gap = latest.timestep - previous.timestep
        if abs(gap) < 1e-8:
            raise ValueError("near_zero_time_gap")
        multiplier = (timestep - latest.timestep) / gap
        if not 0 <= multiplier <= 2:
            raise ValueError("unsupported_time_extrapolation")
    if name == "scaled_input_residual":
        a = a - latest.input_full.float()
        b = b - previous.input_full.float()
        value = x.float() + alpha * a + beta * gate * (a - b)
    else:
        value = alpha * a + beta * gate * multiplier * (a - b)
    return value.to(dtype=latest.output.dtype, device=latest.output.device)


def errors(draft, exact):
    diff = draft.float() - exact.detach().float()
    norm = exact.detach().float().norm()
    l2 = float(diff.norm() / norm.clamp_min(1e-12))
    return {
        "relative_l2": l2,
        "relative_mse": l2 * l2,
        "absolute_rms": float(diff.square().mean().sqrt()),
        "zero_output_norm": bool(norm <= 1e-12),
    }


@dataclass(frozen=True)
class Config:
    mode: str = "trace"
    base_threshold: float = 0.0
    max_consecutive_skips: int = 1
    warmup_steps: int = -1
    tail_steps: int = -1
    predictor: str = "scaled_output_momentum"
    export_trace: bool = True
    beta: float = 0.0
    phase_scales: tuple = (1.0, 1.0, 1.0)
    max_time_gap: float = 0.2
    schema_version: int = 1
    run_nonce: str = ""
    policy: str = "legacy"
    phase_age_caps: tuple | None = None
    feedback_scale_min: tuple = (0.8, 0.8, 0.8)
    feedback_scale_max: tuple = (1.0, 1.0, 1.0)
    feedback_decay: float = 0.9
    feedback_ratio_bounds: tuple = (0.01, 100.0)
    phase_beta_cond: tuple | None = None
    phase_beta_uncond: tuple | None = None
    anchor_strength: tuple = (0.0, 0.25, 0.0)
    anchor_cond_cap: tuple = (0.02, 0.04, 0.02)
    anchor_disagreement_cap: tuple = (0.01, 0.02, 0.01)

    def __post_init__(self):
        object.__setattr__(self, "phase_scales", tuple(self.phase_scales))
        if self.policy not in {"legacy", "paper"}:
            raise ValueError("Unknown routing policy")
        for name in (
            "phase_age_caps",
            "feedback_scale_min",
            "feedback_scale_max",
            "phase_beta_cond",
            "phase_beta_uncond",
            "anchor_strength",
            "anchor_cond_cap",
            "anchor_disagreement_cap",
            "feedback_ratio_bounds",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, tuple(value))
        for name in (
            "feedback_scale_min",
            "feedback_scale_max",
            "anchor_cond_cap",
            "anchor_disagreement_cap",
        ):
            values = getattr(self, name)
            if len(values) != 3 or any(not math.isfinite(v) or v <= 0 for v in values):
                raise ValueError(f"{name} requires three positive finite values")
        for name in ("phase_beta_cond", "phase_beta_uncond", "anchor_strength"):
            values = getattr(self, name)
            if values is not None and (
                len(values) != 3 or any(not 0 <= v <= 1 for v in values)
            ):
                raise ValueError(f"{name} requires three values in [0,1]")
        if self.phase_age_caps is not None and (
            len(self.phase_age_caps) != 3
            or any(type(v) is not int or v < 0 for v in self.phase_age_caps)
        ):
            raise ValueError("phase_age_caps requires three nonnegative integers")
        if any(a > b for a, b in zip(self.feedback_scale_min, self.feedback_scale_max)):
            raise ValueError("feedback scale lower bound exceeds upper bound")
        if not 0 <= self.feedback_decay < 1:
            raise ValueError("feedback_decay must be in [0,1)")
        if (
            len(self.feedback_ratio_bounds) != 2
            or not 0
            < self.feedback_ratio_bounds[0]
            <= self.feedback_ratio_bounds[1]
            < math.inf
        ):
            raise ValueError("Invalid feedback ratio bounds")
        if self.policy == "paper" and self.predictor == "scaled_input_residual":
            raise ValueError(
                "Paper flow policy predicts normalized denoising residuals (velocity), not velocity minus latent"
            )
        if self.mode not in ("off", "trace", "on") or self.predictor not in PREDICTORS:
            raise ValueError("Unsupported AlignCache mode or predictor")
        if not math.isfinite(self.base_threshold) or self.base_threshold < 0:
            raise ValueError("base_threshold must be finite and nonnegative")
        if (
            self.max_consecutive_skips < 0
            or min(self.warmup_steps, self.tail_steps) < -1
        ):
            raise ValueError(
                "Invalid skip/warmup/tail limits (-1 means automatic warmup/tail)"
            )
        if (
            not 0 <= self.beta <= 1
            or not math.isfinite(self.max_time_gap)
            or self.max_time_gap <= 0
        ):
            raise ValueError("Invalid beta or time gap")
        if len(self.phase_scales) != 3 or any(
            not math.isfinite(v) or v <= 0 for v in self.phase_scales
        ):
            raise ValueError("Three positive finite phase scales required")


@dataclass
class Anchor:
    output: torch.Tensor
    input_sample: torch.Tensor
    timestep: float
    input_full: torch.Tensor | None = None


@dataclass(frozen=True)
class Decision:
    step: int
    phase: int
    action: str
    reason: str
    budget: float
    threshold: float
    age: int
    timestep: float
    feedback_eligible: bool = False


class Controller:
    def __init__(self, config: Config, planned_steps: int):
        self.config = config
        self.steps = planned_steps
        self.warmup = min(
            planned_steps,
            max(2, math.ceil(0.15 * planned_steps))
            if config.warmup_steps == -1
            else config.warmup_steps,
        )
        self.tail = min(
            planned_steps,
            max(1, math.ceil(0.10 * planned_steps))
            if config.tail_steps == -1
            else config.tail_steps,
        )
        self.anchors = []
        self.previous_input = None
        self.budget = 0.0
        self.age = 0
        self.alpha = [1.0] * 3
        self.ratio_ema = [1.0] * 3
        self.peak_bytes = 0

    def clear(self):
        self.anchors.clear()
        self.previous_input = None
        self.budget = 0.0
        self.age = 0

    def decide(self, x, step, timestep, disabled=None):
        phase = (
            0
            if step / max(self.steps - 1, 1) < 0.2
            else (1 if step / max(self.steps - 1, 1) < 0.8 else 2)
        )
        threshold = (
            self.config.base_threshold
            * self.config.phase_scales[phase]
            * self.feedback_scale(phase)
        )
        reason, budget = disabled, self.budget
        if not reason:
            if (
                self.config.base_threshold == 0
                or self.config.max_consecutive_skips == 0
            ):
                reason = "disabled_by_config"
            elif self.warmup + self.tail >= self.steps:
                reason = "no_skip_window"
            elif step < self.warmup:
                reason = "warmup"
            elif step >= self.steps - self.tail:
                reason = "tail"
            elif len(self.anchors) < 2:
                reason = "history_not_ready"
            elif (
                self.config.phase_age_caps is not None
                and self.age >= self.config.phase_age_caps[phase]
            ):
                reason = "phase_age_cap"
            elif self.age >= self.config.max_consecutive_skips:
                reason = "max_consecutive_skips"
        if not reason:
            b, a = self.anchors
            xs = sample(x)
            dx = float((a.input_sample - b.input_sample).abs().mean())
            sy, py = sample(a.output), sample(b.output)
            norm = float(sy.abs().mean())
            if dx <= 1e-8 or norm <= 1e-8:
                reason = "unreliable_gain"
            elif abs(timestep - a.timestep) > self.config.max_time_gap:
                reason = "unsupported_time_gap"
            else:
                gain = float((sy - py).abs().mean()) / dx
                budget += gain * float((xs - self.previous_input).abs().mean()) / norm
                if not math.isfinite(budget):
                    reason = "nonfinite_risk"
                elif budget >= threshold:
                    reason = "budget"
        return Decision(
            step,
            phase,
            "exact" if reason else "skip",
            reason or "within_budget",
            budget,
            threshold,
            self.age,
            timestep,
            reason == "budget",
        )

    def predict(self, decision, x):
        if len(self.anchors) < 2:
            return None
        b, a = self.anchors
        return predict(
            self.config.predictor,
            a,
            b,
            x,
            decision.timestep,
            self.alpha[decision.phase],
            self.config.phase_beta_cond[decision.phase]
            if self.config.phase_beta_cond is not None
            else self.config.beta,
        )

    def feedback_scale(self, phase):
        return min(
            self.config.feedback_scale_max[phase],
            max(self.config.feedback_scale_min[phase], 1.0 / self.ratio_ema[phase]),
        )

    def update_feedback(self, decision, metrics):
        if decision.feedback_eligible and metrics:
            key = "relative_mse" if self.config.policy == "paper" else "relative_l2"
            low, high = self.config.feedback_ratio_bounds
            ratio = min(high, max(low, metrics[key] / max(decision.budget, 1e-12)))
            decay = self.config.feedback_decay
            p = decision.phase
            self.ratio_ema[p] = decay * self.ratio_ema[p] + (1 - decay) * ratio

    def commit_exact(self, decision, x, output, draft=None):
        if not torch.isfinite(output).all():
            raise FloatingPointError("QwenImage21 exact model output contains NaN/Inf")
        metrics = errors(draft, output) if draft is not None else {}
        p = decision.phase
        self.update_feedback(decision, metrics)
        if self.anchors:
            a, y = sample(self.anchors[-1].output), sample(output)
            denom = float(torch.dot(a, a))
            if denom > 1e-8:
                fitted = min(1.1, max(0.9, float(torch.dot(a, y)) / denom))
                self.alpha[p] = 0.9 * self.alpha[p] + 0.1 * fitted
        self.anchors.append(
            Anchor(
                output.detach().clone(),
                sample(x),
                decision.timestep,
                x.detach().clone()
                if self.config.predictor == "scaled_input_residual"
                else None,
            )
        )
        self.anchors = self.anchors[-2:]
        self.previous_input = sample(x)
        self.budget, self.age = 0.0, 0
        tensors = [
            t
            for a in self.anchors
            for t in (a.output, a.input_sample, a.input_full)
            if t is not None
        ]
        self.peak_bytes = max(
            self.peak_bytes,
            sum(t.numel() * t.element_size() for t in tensors)
            + self.previous_input.numel() * 4,
        )
        return metrics

    def commit_skip(self, decision, x):
        self.previous_input = sample(x)
        self.budget, self.age = decision.budget, self.age + 1


def guided(outputs, scale):
    return (
        outputs[0].float()
        if len(outputs) == 1
        else outputs[1].float() + scale * (outputs[0].float() - outputs[1].float())
    )


class PairController(Controller):
    def __init__(self, config, steps, cfg):
        super().__init__(config, steps)
        self.cfg = cfg
        self.uncond = (
            Controller(replace(config, phase_beta_cond=config.phase_beta_uncond), steps)
            if cfg != 1
            else None
        )
        self.transfer_alpha = [1.0] * 3
        self.anchor_weight = 0.0

    def clear(self):
        super().clear()
        if self.uncond is not None:
            self.uncond.clear()

    def predict_pair(self, decision, x):
        cond = self.predict(decision, x)
        self.anchor_weight = 0.0
        if cond is None:
            return None
        if self.uncond is None:
            return [cond]
        uncond = self.uncond.predict(decision, x)
        if uncond is None:
            return None
        p = decision.phase
        ca, ua = self.anchors[-1].output.float(), self.uncond.anchors[-1].output.float()
        candidate = ua + self.transfer_alpha[p] * (cond.float() - ca)
        cond_error = errors(cond, ca)["relative_mse"]
        disagreement = errors(candidate, uncond)["relative_mse"]
        cap = (
            self.config.phase_age_caps[p]
            if self.config.phase_age_caps is not None
            else self.config.max_consecutive_skips
        )
        freshness = max(0.0, 1.0 - self.age / max(cap, 1))
        confidence = max(0.0, 1.0 - cond_error / self.config.anchor_cond_cap[p])
        agreement = max(
            0.0, 1.0 - disagreement / self.config.anchor_disagreement_cap[p]
        )
        weight = self.config.anchor_strength[p] * min(freshness, confidence, agreement)
        self.anchor_weight = weight
        return [cond, ((1 - weight) * uncond.float() + weight * candidate).to(uncond)]

    def commit_pair(self, decision, x, outputs, drafts):
        metrics = (
            errors(guided(drafts, self.cfg), guided(outputs, self.cfg))
            if drafts is not None
            else {}
        )
        # One guided feedback update, before either exact anchor is replaced.
        self.update_feedback(decision, metrics)
        no_feedback = replace(decision, feedback_eligible=False)
        if self.uncond is not None:
            if self.anchors and self.uncond.anchors:
                dc = sample(outputs[0]) - sample(self.anchors[-1].output)
                du = sample(outputs[1]) - sample(self.uncond.anchors[-1].output)
                denom = float(torch.dot(dc, dc))
                if denom > 1e-8:
                    fitted = min(2.0, max(0.0, float(torch.dot(dc, du)) / denom))
                    p = decision.phase
                    self.transfer_alpha[p] = 0.9 * self.transfer_alpha[p] + 0.1 * fitted
            self.uncond.commit_exact(no_feedback, x, outputs[1])
        super().commit_exact(no_feedback, x, outputs[0])
        self.peak_bytes = max(self.peak_bytes, self.anchor_bytes())
        return metrics

    def anchor_bytes(self):
        return sum(
            t.numel() * t.element_size()
            for c in [self, self.uncond]
            if c is not None
            for a in c.anchors
            for t in [a.output, a.input_sample]
        )
