from types import SimpleNamespace

import pytest
import torch

from AlignCache4QwenImage21.aligncache_accelerate import Config
from AlignCache4QwenImage21.comfyui_nodes import RunState, qwen21_diffusion_wrapper
from AlignCache4QwenImage21.runtime import RUN_STATE_KEY


def state(mode="on", threshold=0.1):
    model = SimpleNamespace(
        diffusion_model=object(), model_sampling=SimpleNamespace(timestep=lambda t: t)
    )
    guider = SimpleNamespace(
        model_patcher=SimpleNamespace(model=model, patches_uuid="fixed")
    )
    return RunState(
        Config(mode=mode, base_threshold=threshold, warmup_steps=2, tail_steps=1),
        torch.linspace(1, 0, 26),
        guider,
    )


class Executor:
    def __init__(self):
        self.calls = []
        self.outputs = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        output = (2 + args[0] * 0.01).to(torch.bfloat16)
        self.outputs.append(output)
        return output


def invoke(run, executor, k, context=None, refs=None, slots=None, **kwargs):
    options = {RUN_STATE_KEY: run, "cond_or_uncond": [0]}
    x = torch.full((1, 4, 2, 2), 1 + k * 0.01)
    t = run.sigmas[min(k, 25)].reshape(1)
    context = torch.ones(1, 2, 4) if context is None else context
    out = qwen21_diffusion_wrapper(
        executor, x, t, context, refs, slots, options, **kwargs
    )
    return out


@pytest.mark.parametrize("mode,threshold", [("trace", 0.1), ("on", 0)])
def test_trace_and_zero_threshold_return_original_object(mode, threshold):
    run, ex = state(mode, threshold), Executor()
    for k in range(25):
        assert invoke(run, ex, k) is ex.outputs[-1]
    assert len(ex.calls) == 25 and run.skip_count == 0
    if mode == "trace":
        assert any(r["shadow_action"] == "skip" for r in run.rows)


def test_six_positional_args_and_kwargs_and_skip_call_count():
    ex = Executor()
    args = [
        torch.ones(1, 4, 2, 2),
        torch.ones(1),
        torch.ones(1, 2, 4),
        [torch.zeros(1)],
        [1],
        {},
    ]
    qwen21_diffusion_wrapper(ex, *args, control="sentinel", other=3)
    assert all(a is b for a, b in zip(args, ex.calls[0][0]))
    assert ex.calls[0][1] == {"control": "sentinel", "other": 3}
    run, ex = state(), Executor()
    for k in range(25):
        invoke(run, ex, k)
    assert (
        run.skip_count > 0 and len(ex.calls) == run.exact_count == 25 - run.skip_count
    )


def test_trace_keeps_shadow_anchors():
    run, ex = state("trace"), Executor()
    invoke(run, ex, 0)
    invoke(run, ex, 1)
    anchor = run.controller.anchors[-1]
    invoke(run, ex, 2)
    assert run.rows[-1]["shadow_action"] == "skip"
    assert run.controller.anchors[-1] is anchor


@pytest.mark.parametrize(
    "change", ["context", "refs", "slots", "layout", "time", "patch", "branch"]
)
def test_signature_and_contract_changes_disable_rest_of_run(change):
    run, ex = state(), Executor()
    invoke(run, ex, 0)
    invoke(run, ex, 1)
    opts = {RUN_STATE_KEY: run, "cond_or_uncond": [0]}
    x, t, ctx, refs, slots = (
        torch.ones(1, 4, 2, 2) * 1.02,
        run.sigmas[2].reshape(1),
        torch.ones(1, 2, 4),
        None,
        None,
    )
    if change == "context":
        ctx = ctx + 1
    if change == "refs":
        refs = [torch.ones(1, 4, 2, 2)]
    if change == "slots":
        slots = [1]
    if change == "layout":
        x = x.double()
    if change == "time":
        t = run.sigmas[1].reshape(1)
    if change == "patch":
        run.guider.model_patcher.patches_uuid = "changed"
    if change == "branch":
        opts["cond_or_uncond"] = [0, 1]
    out = qwen21_diffusion_wrapper(ex, x, t, ctx, refs, slots, opts)
    assert out is ex.outputs[-1] and run.reasons
    for k in range(3, 6):
        invoke(run, ex, k)
    assert len(ex.calls) == 6 and not run.controller.anchors


def test_same_content_new_objects_allowed_and_inplace_change_detected():
    run, ex = state(), Executor()
    ctx = torch.ones(1, 2, 4)
    invoke(run, ex, 0, ctx)
    invoke(run, ex, 1, ctx.clone())
    invoke(run, ex, 2, ctx.clone())
    assert run.skip_count == 1 and not run.reasons
    ctx.add_(1)
    invoke(run, ex, 3, ctx)
    assert "condition_signature_changed" in run.reasons


def test_nonfinite_exact_is_not_masked():
    run = state()
    with pytest.raises(FloatingPointError):
        invoke(run, lambda *a, **k: a[0] * float("nan"), 0)


def test_same_shape_reference_replacement_and_order_in_trace():
    for mutate in ("content", "order"):
        run, ex = state("trace"), Executor()
        refs = [torch.ones(1, 4, 2, 2), torch.zeros(1, 4, 2, 2)]
        invoke(run, ex, 0, refs=refs)
        invoke(run, ex, 1, refs=[r.clone() for r in refs])
        assert not run.reasons
        if mutate == "content":
            refs[0].add_(1)
        else:
            refs.reverse()
        invoke(run, ex, 2, refs=refs)
        assert "condition_signature_changed" in run.reasons and len(ex.calls) == 3


def test_edit_attention_mask_is_a_condition_signature_and_on_remains_exact():
    run, ex = state("on"), Executor()
    options = {RUN_STATE_KEY: run, "cond_or_uncond": [0]}
    mask = torch.ones(1, 4)
    x = torch.ones(1, 4, 2, 2)
    qwen21_diffusion_wrapper(
        ex,
        x,
        run.sigmas[0].reshape(1),
        torch.ones(1, 2, 4),
        [torch.ones(1, 4, 2, 2)],
        None,
        options,
        attention_mask=mask,
    )
    assert "editing_not_calibrated" in run.reasons and run.skip_count == 0
    assert run.signature is None


def test_nonfinite_failure_metadata_is_written_as_valid_json(tmp_path):
    import json

    from AlignCache4QwenImage21.runtime import write_run

    run = state()
    run.guider.model_patcher.load_device = torch.device("cpu")
    run.manifest["sigmas"] = [float("nan"), 0.0]
    run.failure = {"type": "FloatingPointError", "message": "nonfinite output"}
    run.rows.append({"budget": float("inf"), "reason": "nonfinite_risk"})
    write_run(tmp_path, run, "failed", 0.1)
    directory = tmp_path / "aligncache_qwen21" / run.run_id
    summary = json.loads((directory / "summary.json").read_text())
    row = json.loads((directory / "trace.jsonl").read_text())
    assert summary["status"] == "failed" and summary["manifest"]["sigmas"][0] is None
    assert row["budget"] is None and row["reason"] == "nonfinite_risk"


def test_consecutive_skip_log_matches_completed_route():
    run, executor = state(), Executor()
    for step in range(4):
        invoke(run, executor, step)
    assert [row["consecutive_skips"] for row in run.rows] == [0, 0, 1, 0]


def test_native_default_kv_policy_is_explicit_in_signature_manifest():
    run, ex = state("trace"), Executor()
    invoke(run, ex, 0)
    assert run.manifest["kv_config"] == {
        "device": "auto",
        "dtype": "default",
        "source": "native_default",
    }
