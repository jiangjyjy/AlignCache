from dataclasses import replace

import pytest
import torch

from AlignCache4QwenImage21.aligncache_accelerate import (
    Config,
    Controller,
    Decision,
    PairController,
    guided,
)


def config(**kwargs):
    return Config(
        policy="paper",
        base_threshold=0.12,
        warmup_steps=2,
        tail_steps=0,
        max_consecutive_skips=4,
        predictor="scaled_output_momentum",
        **kwargs,
    )


def decision(step, phase=1, feedback=False):
    return Decision(
        step, phase, "exact", "budget", 0.1, 0.12, 0, 1 - step * 0.04, feedback
    )


def test_guided_mse_feedback_not_branch_l2_and_only_budget_events():
    c = PairController(config(), 25, 2)
    x = torch.ones(1, 4, 2, 2)
    exact = [x * 2, x]
    # Exact guided velocity=3, draft=2.4 -> relative MSE=.04, ratio=.4.
    draft = [x * 1.6, x * 0.8]
    metrics = c.commit_pair(decision(0, feedback=True), x, exact, draft)
    assert metrics["relative_mse"] == pytest.approx(0.04)
    assert c.ratio_ema[1] == pytest.approx(0.94)
    old = c.ratio_ema[:]
    c.commit_pair(decision(1), x + 0.1, exact, draft)
    assert c.ratio_ema == old
    assert c.uncond.ratio_ema == [1, 1, 1]


def test_guided_cancellation_uses_composed_error():
    c = PairController(config(), 25, 2)
    x = torch.ones(1, 1, 2, 2)
    # Both branches differ, but guided output is exactly unchanged.
    m = c.commit_pair(decision(0, feedback=True), x, [x * 2, x], [x * 3, x * 3])
    assert m["relative_mse"] == 0
    assert c.ratio_ema[1] == pytest.approx(0.901)


def test_phase_caps_and_independent_feedback_relaxation():
    c = Controller(config(phase_age_caps=(0, 3, 1), feedback_scale_max=(1, 1.2, 1)), 25)
    x = torch.ones(1, 1, 2, 2)
    for i in range(2):
        c.commit_exact(decision(i), x + i * 0.01, x * 2 + i * 0.01)
    assert c.decide(x, 2, 0.96).reason == "phase_age_cap"
    c.age = 3
    assert c.decide(x, 10, 0.92).reason == "phase_age_cap"
    c.ratio_ema = [0.5, 0.5, 0.5]
    assert [c.feedback_scale(p) for p in range(3)] == [1, 1.2, 1]


def test_anchor_transfer_gates_and_exact_only_branch_history():
    c = PairController(
        config(
            beta=0.4,
            anchor_strength=(0, 0.5, 0),
            anchor_cond_cap=(1, 1, 1),
            anchor_disagreement_cap=(1, 1, 1),
        ),
        25,
        2,
    )
    x = torch.ones(1, 1, 2, 2)
    for i in range(2):
        c.commit_pair(
            decision(i), x + i * 0.01, [x * (10 + i * 0.1), x * (10 + i * 0.2)], None
        )
    assert c.transfer_alpha[1] == pytest.approx(1.1, abs=1e-5)
    before = c.anchors[-1].output.clone(), c.uncond.anchors[-1].output.clone()
    d = replace(decision(2), action="skip")
    draft = c.predict_pair(d, x)
    assert 0 < c.anchor_weight <= 0.5
    assert len(draft) == 2
    c.commit_skip(d, x)
    assert torch.equal(before[0], c.anchors[-1].output)
    assert torch.equal(before[1], c.uncond.anchors[-1].output)
    c.age = 4
    c.predict_pair(d, x)
    assert c.anchor_weight == 0
    c.age = 0
    c.predict_pair(replace(d, phase=0), x)
    assert c.anchor_weight == 0
    c.clear()
    assert not c.anchors and not c.uncond.anchors


def test_flow_residual_adaptation_and_cfg1_reduction():
    x = torch.ones(1, 1, 2, 2)
    c = PairController(config(beta=0.4), 25, 1)
    velocity = x * 2
    # x0-x=-sigma*v. A constant vector field has constant normalized residual,
    # even as x and sigma change; subtracting x from v would not.
    for i, sigma in enumerate([0.8, 0.6]):
        latent = x * (i + 1)
        x0 = latent - sigma * velocity
        recovered = (latent - x0) / sigma
        c.commit_pair(decision(i), latent, [recovered], None)
    draft = c.predict_pair(decision(2), x * 4)
    assert c.uncond is None and len(draft) == 1
    assert torch.allclose(guided(draft, 1), velocity, atol=1e-6)


def test_independent_branch_momentum_and_disagreement_rejection():
    c = PairController(
        config(
            phase_beta_cond=(0, 0, 0),
            phase_beta_uncond=(0, 0.5, 0),
            anchor_strength=(0, 0.5, 0),
            anchor_disagreement_cap=(1e-12, 1e-12, 1e-12),
        ),
        25,
        2,
    )
    x = torch.ones(1, 1, 2, 2)
    for i in range(2):
        c.commit_pair(decision(i), x + i * 0.01, [x * (10 + i), x * (10 + i)], None)
    draft = c.predict_pair(decision(2), x)
    assert torch.all(draft[1] > draft[0])
    assert c.anchor_weight == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"phase_age_caps": (1, -1, 2)},
        {"phase_age_caps": (1.0, 2, 3)},
        {"feedback_scale_max": (0.5, 1, 1)},
        {"phase_beta_uncond": (0, 2, 0)},
        {"anchor_cond_cap": (0, 1, 1)},
        {"feedback_ratio_bounds": (2, 1)},
    ],
)
def test_reject_invalid_paper_parameters(kwargs):
    with pytest.raises(ValueError):
        config(**kwargs)
