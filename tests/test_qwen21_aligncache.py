import math

import pytest
import torch

from AlignCache4QwenImage21.aligncache_accelerate import Config, Controller, errors


def feed(c, step, force=None):
    x = torch.full((1, 4, 2, 2), 1 + step * 0.01)
    d = c.decide(x, step, 1 - step * 0.04, force)
    draft = c.predict(d, x)
    y = (2 + x * 0.01).to(torch.bfloat16)
    if d.action == "skip":
        c.commit_skip(d, x)
    else:
        c.commit_exact(d, x, y, draft)
    return d, x, y


def test_exact_anchors_budget_and_aliases():
    c = Controller(
        Config(mode="on", base_threshold=0.1, warmup_steps=2, tail_steps=1), 25
    )
    for step in range(2):
        d, x, y = feed(c, step)
    original = c.anchors[-1].output.clone()
    x.add_(10)
    y.add_(10)
    assert torch.equal(original, c.anchors[-1].output)
    anchors = list(c.anchors)
    d, _, _ = feed(c, 2)
    assert d.action == "skip"
    assert c.anchors == anchors and c.age == 1
    d, _, _ = feed(c, 3)
    assert d.reason == "max_consecutive_skips" and c.budget == 0 and c.age == 0


@pytest.mark.parametrize(
    "config",
    [Config(base_threshold=0), Config(base_threshold=1, max_consecutive_skips=0)],
)
def test_zero_controls_never_skip(config):
    c = Controller(config, 25)
    assert all(feed(c, k)[0].action == "exact" for k in range(25))


def test_phases_auto_windows_and_short_schedule():
    c = Controller(Config(base_threshold=0.1), 25)
    assert (c.warmup, c.tail) == (4, 3)
    decisions = [feed(c, k)[0] for k in range(25)]
    assert decisions[0].reason == "warmup"
    assert decisions[22].reason == "tail"
    assert [decisions[k].phase for k in (0, 10, 24)] == [0, 1, 2]
    assert (
        feed(Controller(Config(base_threshold=0.1), 2), 0)[0].reason == "no_skip_window"
    )


def test_feedback_only_budget_and_uses_frozen_draft():
    c = Controller(Config(base_threshold=1e-7, warmup_steps=2, tail_steps=0), 25)
    x = torch.ones(1, 4, 2, 2)
    for k in range(2):
        d = c.decide(x + k, k, 1 - k * 0.04)
        c.commit_exact(d, x + k, x + k)
    d = c.decide(x + 2, 2, 0.92)
    assert d.reason == "budget" and d.feedback_eligible
    draft = c.predict(d, x + 2)
    frozen = draft.clone()
    old_alpha = c.alpha[d.phase]
    m = c.commit_exact(d, x + 2, (x + 2) * 10, draft)
    assert torch.equal(draft, frozen)
    assert m["relative_mse"] == pytest.approx(m["relative_l2"] ** 2)
    assert c.ratio_ema[d.phase] > 1 and c.alpha[d.phase] != old_alpha
    old_ratio = c.ratio_ema[:]
    d = c.decide(x + 3, 3, 0.88, "signature_changed")
    c.commit_exact(d, x + 3, (x + 3) * 10, c.predict(d, x + 3))
    assert c.ratio_ema == old_ratio


@pytest.mark.parametrize(
    "predictor",
    [
        "reuse",
        "scaled_output_momentum",
        "scaled_input_residual",
        "time_normalized_output",
    ],
)
def test_predictor_dtype_shape_device(predictor):
    c = Controller(
        Config(predictor=predictor, base_threshold=0.1, warmup_steps=2, tail_steps=0),
        25,
    )
    feed(c, 0)
    feed(c, 1)
    x = torch.ones(1, 4, 2, 2)
    out = c.predict(c.decide(x, 2, 0.92), x)
    assert (
        out.dtype == torch.bfloat16 and out.device == x.device and out.shape == x.shape
    )


def test_nonfinite_and_near_zero_denominators():
    c = Controller(Config(base_threshold=0.1, warmup_steps=2, tail_steps=0), 25)
    x = torch.ones(1, 4, 2, 2)
    for k in range(2):
        d = c.decide(x, k, 1 - k * 0.04)
        c.commit_exact(d, x, x)
    assert c.decide(x, 2, 0.92).reason == "unreliable_gain"
    with pytest.raises(FloatingPointError):
        c.commit_exact(d, x, x * math.nan)
    for value in [math.nan, math.inf, -1]:
        with pytest.raises(ValueError):
            Config(base_threshold=value)
    assert errors(x, x * 0)["zero_output_norm"]


def test_signed_time_prediction_and_stability_gate():
    from AlignCache4QwenImage21.aligncache_accelerate import Anchor, predict

    x = torch.ones(1, 1, 2, 2)
    b = Anchor(x.clone(), x.flatten().clone(), 0.8)
    a = Anchor(x.clone() * 2, x.flatten().clone(), 0.6)
    # g=1/2, signed time ratio=(-.1)/(-.2)=1/2.
    result = predict("time_normalized_output", a, b, x, 0.5, alpha=1, beta=0.2)
    assert torch.allclose(result, x * 2.05)
    b.output = -b.output
    result = predict("scaled_output_momentum", a, b, x, 0.5, alpha=1, beta=0.2)
    assert torch.equal(result, a.output)
    b.timestep = a.timestep
    with pytest.raises(ValueError, match="near_zero"):
        predict("time_normalized_output", a, b, x, 0.5)


def test_statistic_sample_covers_latent_interior_without_using_rng():
    from AlignCache4QwenImage21.aligncache_accelerate import sample

    tensor = torch.arange(64 * 64 * 64).reshape(1, 64, 64, 64)
    rng = torch.random.get_rng_state().clone()
    indices = sample(tensor).long()
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert len(indices) == len(indices.unique()) == 4096
    assert len((indices % 64).unique()) == 64
    assert len(((indices // 64) % 64).unique()) == 64
    assert torch.equal(indices, sample(tensor.clone()).long())
    interior = torch.zeros(1, 64, 64, 64)
    interior[:, :, 16:48, 16:48] = 1
    assert 0.20 < float(sample(interior).mean()) < 0.30
