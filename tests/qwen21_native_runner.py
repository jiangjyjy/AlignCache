"""Isolated CPU contract runner, called by test_qwen21_native_contract.py."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import comfy.latent_formats
import comfy.model_base
import comfy.model_patcher
import comfy.sampler_helpers
import comfy.samplers
import comfy.supported_models_base
import folder_paths
import nodes
import torch
from comfy.patcher_extension import WrapperExecutor, get_all_wrappers
from comfy_api.latest import _io
from comfy_extras.nodes_images import SaveImageAdvanced
from comfy_extras.nodes_qwen import QwenImage21Cache, TextEncodeQwenImage21

from AlignCache4QwenImage21.benchmark import (
    BENCHMARK_KEY,
    BENCHMARK_STATE_KEY,
    AlignCacheQwenImage21Benchmark,
)
from AlignCache4QwenImage21.comfyui_nodes import AlignCacheQwenImage21
from AlignCache4QwenImage21.runtime import CONFIG_KEY, RUN_STATE_KEY, WRAPPER_KEY


class TinyConfig(comfy.supported_models_base.BASE):
    unet_extra_config = {}
    latent_format = comfy.latent_formats.QwenImage21
    sampling_settings = {"shift": 0.69, "multiplier": 1.0}


torch.manual_seed(123)
cfg = TinyConfig(
    dict(
        in_channels=64,
        out_channels=64,
        num_layers=1,
        num_attention_heads=2,
        attention_head_dim=16,
        context_in_dim=16,
        axes_dims_rope=(4, 6, 6),
        dtype=torch.float32,
    )
)
base = comfy.model_base.QwenImage21(cfg, device=torch.device("cpu"))
with torch.no_grad():
    for p in base.parameters():
        p.uniform_(-0.05, 0.05)
patcher = comfy.model_patcher.ModelPatcher(
    base, torch.device("cpu"), torch.device("cpu")
)
node = AlignCacheQwenImage21()

with tempfile.TemporaryDirectory() as tmp:
    folder_paths.set_output_directory(tmp)
    patched = node.attach(
        patcher, mode="trace", base_threshold=0.5, warmup_steps=2, tail_steps=1
    )[0]
    patched = node.attach(
        patched, mode="trace", base_threshold=0.5, warmup_steps=2, tail_steps=1
    )[0]
    assert len(patched.get_wrappers("diffusion_model", WRAPPER_KEY)) == 1
    assert not patcher.wrappers
    assert patched.model is patcher.model
    off = node.attach(patched, mode="off")[0]
    assert not off.get_wrappers("diffusion_model", WRAPPER_KEY)
    assert CONFIG_KEY not in off.model_options["transformer_options"]

    # Real CFGGuider.sample -> prepare_model_patcher -> OUTER_SAMPLE ->
    # inner_sample -> BaseModel -> native Qwen.forward -> DIFFUSION_MODEL.
    noise = torch.randn(1, 64, 2, 2)
    latent = torch.zeros_like(noise)
    cond = [[torch.randn(1, 3, 16), {}]]
    sampler = comfy.samplers.sampler_object("euler")
    sigmas = torch.linspace(1, 0, 10)
    outputs = []
    for model in (
        off,
        patched,
        patched,
        node.attach(patched, mode="on", base_threshold=0)[0],
    ):
        guider = comfy.samplers.CFGGuider(model)
        guider.set_conds(cond, cond)
        original = guider.model_options
        out = guider.sample(
            noise.clone(), latent.clone(), sampler, sigmas, disable_pbar=True, seed=7
        )
        outputs.append(out)
        assert guider.model_options is original
        assert RUN_STATE_KEY not in original["transformer_options"]
        assert base.diffusion_model.prefix_cache is None
        assert not base.diffusion_model.prefix_cache_enabled
    assert all(torch.equal(outputs[0], v) for v in outputs[1:])
    summaries = [
        json.loads(p.read_text())
        for p in Path(tmp).glob("aligncache_qwen21/*/summary.json")
    ]
    assert len(summaries) == 3
    assert len({s["run_id"] for s in summaries}) == 3
    assert all(
        s["status"] == "completed" and s["actual_model_evaluations"] == 9
        for s in summaries
    )
    assert all(not s["degradation_reasons"] for s in summaries), summaries

    # A real skip must bypass the native forward; threshold here is synthetic.
    accelerated = node.attach(
        patcher, mode="on", base_threshold=100.0, warmup_steps=2, tail_steps=1
    )[0]
    guider = comfy.samplers.CFGGuider(accelerated)
    guider.set_conds(cond, cond)
    guider.sample(
        noise.clone(), latent.clone(), sampler, sigmas, disable_pbar=True, seed=7
    )
    latest = max(
        Path(tmp).glob("aligncache_qwen21/*/summary.json"),
        key=lambda p: p.stat().st_mtime_ns,
    )
    summary = json.loads(latest.read_text())
    assert summary["skip_count"] > 0, summary

    observer = AlignCacheQwenImage21Benchmark()
    measurement_ids = []
    for i, model in enumerate(
        (
            off,
            patched,
            node.attach(patcher, mode="on", base_threshold=0)[0],
            accelerated,
        )
    ):
        measured = observer.attach(model, run_nonce=f"native-{i}", profile=(i == 3))[0]
        measured = observer.attach(measured, run_nonce=f"native-{i}", profile=(i == 3))[
            0
        ]
        assert len(measured.get_wrappers("outer_sample", BENCHMARK_KEY)) == 1
        assert len(measured.get_wrappers("diffusion_model", BENCHMARK_KEY)) == 1
        assert not model.get_wrappers("diffusion_model", BENCHMARK_KEY)
        guider = comfy.samplers.CFGGuider(measured)
        guider.set_conds(cond, cond)
        original = guider.model_options
        result = guider.sample(
            noise.clone(), latent.clone(), sampler, sigmas, disable_pbar=True, seed=7
        )
        assert (
            guider.model_options is original
            and BENCHMARK_STATE_KEY not in original["transformer_options"]
        )
        assert base.diffusion_model.prefix_cache is None
        latest_measurement = max(
            Path(tmp).glob("aligncache_qwen21_benchmark/*/measurement.json"),
            key=lambda p: p.stat().st_mtime_ns,
        )
        measurement = json.loads(latest_measurement.read_text())
        measurement_ids.append(measurement["run_id"])
        assert (
            measurement["status"] == "completed" and measurement["sampling_seconds"] > 0
        )
        assert measurement["run_nonce"] == f"native-{i}"
        if i < 3:
            assert torch.equal(result, outputs[0])
            assert measurement["forward_completed"] == 9 and not measurement["forwards"]
        else:
            latest = max(
                Path(tmp).glob("aligncache_qwen21/*/summary.json"),
                key=lambda p: p.stat().st_mtime_ns,
            )
            summary = json.loads(latest.read_text())
            assert summary["skip_count"] > 0 and not summary["degradation_reasons"], (
                summary
            )
            assert measurement["forward_completed"] == summary["exact_count"]
            assert len(measurement["forwards"]) == summary["exact_count"] < 9
            assert not measurement["performance_sample"]
    assert len(set(measurement_ids)) == 4

    # Validate keyword binding at the real outer boundary, independently of
    # CFGGuider.sample's current positional invocation.
    guider = comfy.samplers.CFGGuider(patched)
    guider.set_conds(cond, cond)
    guider.conds = {
        key: [value.copy() for value in values]
        for key, values in guider.original_conds.items()
    }
    guider.model_options = comfy.model_patcher.create_model_options_clone(
        guider.model_options
    )
    comfy.sampler_helpers.prepare_model_patcher(
        patched, guider.conds, guider.model_options
    )
    original = guider.model_options
    executor = WrapperExecutor.new_class_executor(
        guider.outer_sample,
        guider,
        get_all_wrappers("outer_sample", guider.model_options, is_model_options=True),
    )
    executor.execute(
        noise=noise.clone(),
        latent_image=latent.clone(),
        sampler=sampler,
        sigmas=sigmas,
        disable_pbar=True,
    )
    assert guider.model_options is original

    for cfg_value, test_sampler, reason in [
        (2.0, sampler, "unsupported_cfg"),
        (1.0, comfy.samplers.sampler_object("heun"), "unsupported_sampler"),
        (
            1.0,
            comfy.samplers.KSAMPLER(sampler.sampler_function, {"s_churn": 0.1}),
            "unsupported_sampler_options",
        ),
    ]:
        guider = comfy.samplers.CFGGuider(accelerated)
        guider.set_conds(cond, cond)
        guider.set_cfg(cfg_value)
        guider.sample(
            noise.clone(), latent.clone(), test_sampler, sigmas, disable_pbar=True
        )
        latest = max(
            Path(tmp).glob("aligncache_qwen21/*/summary.json"),
            key=lambda p: p.stat().st_mtime_ns,
        )
        summary = json.loads(latest.read_text())
        assert (
            summary["skip_count"] == 0 and reason in summary["degradation_reasons"]
        ), summary

    # Native pre_run/cleanup must still clear prefix KV when a wrapper raises.
    def failure(*args, **kwargs):
        raise RuntimeError("native-contract-failure")

    broken = node.attach(patcher, mode="trace")[0]
    broken.add_wrapper_with_key("diffusion_model", "failure", failure)
    guider = comfy.samplers.CFGGuider(broken)
    guider.set_conds(cond, cond)
    original = guider.model_options
    try:
        guider.sample(noise, latent, sampler, sigmas, disable_pbar=True)
        raise AssertionError("failure not raised")
    except RuntimeError as error:
        assert str(error) == "native-contract-failure"
    assert (
        guider.model_options is original and base.diffusion_model.prefix_cache is None
    )
    measured_failure = observer.attach(broken, run_nonce="native-failure")[0]
    guider = comfy.samplers.CFGGuider(measured_failure)
    guider.set_conds(cond, cond)
    original = guider.model_options
    try:
        guider.sample(noise, latent, sampler, sigmas, disable_pbar=True)
        raise AssertionError("failure not raised")
    except RuntimeError as error:
        assert str(error) == "native-contract-failure"
    latest_measurement = max(
        Path(tmp).glob("aligncache_qwen21_benchmark/*/measurement.json"),
        key=lambda p: p.stat().st_mtime_ns,
    )
    measurement = json.loads(latest_measurement.read_text())
    assert (
        measurement["status"] == "failed"
        and measurement["failure"]["type"] == "RuntimeError"
    )
    assert guider.model_options is original
    with patch(
        "AlignCache4QwenImage21.benchmark.write_measurement",
        side_effect=OSError("measurement export failure"),
    ):
        try:
            guider.sample(noise, latent, sampler, sigmas, disable_pbar=True)
            raise AssertionError("failure not raised")
        except RuntimeError as error:
            assert str(error) == "native-contract-failure"
    assert (
        guider.model_options is original and base.diffusion_model.prefix_cache is None
    )
    latest = max(
        Path(tmp).glob("aligncache_qwen21/*/summary.json"),
        key=lambda p: p.stat().st_mtime_ns,
    )
    assert json.loads(latest.read_text())["status"] == "failed"
    with patch(
        "AlignCache4QwenImage21.runtime.write_run", side_effect=OSError("log failure")
    ):
        try:
            guider.sample(noise, latent, sampler, sigmas, disable_pbar=True)
            raise AssertionError("failure not raised")
        except RuntimeError as error:
            assert str(error) == "native-contract-failure"
    assert (
        guider.model_options is original and base.diffusion_model.prefix_cache is None
    )
# Paper pair routing must tolerate Comfy's CFG batching choices and preserve
# native exact outputs. This uses the real random Qwen model, not a stub.
with tempfile.TemporaryDirectory() as paper_tmp:
    folder_paths.set_output_directory(paper_tmp)
    for cfg_value in (1.0, 2.0):
        for split in (False, True):
            reference = None
            for mode, threshold in (("off", 0), ("trace", 100), ("on", 0), ("on", 100)):
                model = node.attach(
                    patcher,
                    mode=mode,
                    base_threshold=threshold,
                    warmup_steps=2,
                    tail_steps=1,
                    max_consecutive_skips=4,
                    predictor="scaled_output_momentum",
                    advanced_config="preset:paper_qwen",
                )[0]
                model = observer.attach(
                    model,
                    run_nonce=f"paper-{cfg_value}-{split}-{mode}-{threshold}",
                )[0]
                guider = comfy.samplers.CFGGuider(model)
                guider.set_conds(cond, [[cond[0][0] * 0.7, {}]])
                guider.set_cfg(cfg_value)
                from contextlib import nullcontext

                with (
                    patch(
                        "comfy.samplers.can_concat_cond",
                        side_effect=lambda a, b: a is b,
                    )
                    if split
                    else nullcontext()
                ):
                    out = guider.sample(
                        noise.clone(),
                        latent.clone(),
                        sampler,
                        sigmas,
                        disable_pbar=True,
                        seed=7,
                    )
                if mode == "off":
                    reference = out
                    continue
                summary = json.loads(
                    max(
                        Path(paper_tmp).glob("aligncache_qwen21/*/summary.json"),
                        key=lambda p: p.stat().st_mtime_ns,
                    ).read_text()
                )
                measurement = json.loads(
                    max(
                        Path(paper_tmp).glob(
                            "aligncache_qwen21_benchmark/*/measurement.json"
                        ),
                        key=lambda p: p.stat().st_mtime_ns,
                    ).read_text()
                )
                assert not summary["degradation_reasons"], summary
                assert summary["exact_count"] + summary["skip_count"] == 9
                assert (
                    summary["native_forward_count"] == measurement["forward_completed"]
                ), summary
                assert summary["native_forward_count"] == summary["exact_count"] * (
                    2 if split and cfg_value > 1 else 1
                )
                if mode == "trace" or threshold == 0:
                    assert torch.equal(reference, out), (cfg_value, split, mode)
                else:
                    assert summary["skip_count"] > 0, summary
                assert base.diffusion_model.prefix_cache is None
                assert RUN_STATE_KEY not in guider.model_options["transformer_options"]
    # Switching back to off also removes pair wrappers.
    assert not node.attach(model, mode="off")[0].get_wrappers(
        "calc_cond_batch", WRAPPER_KEY
    )
    # A same-shape negative condition change must invalidate both branches before
    # the first possible skip; positive-only signatures would silently miss it.
    model = node.attach(
        patcher,
        mode="on",
        base_threshold=100,
        warmup_steps=2,
        tail_steps=1,
        advanced_config="preset:paper_qwen",
    )[0]
    # The pinned Comfy prefix-cache slot eviction has a tensor-equality bug on
    # in-place condition mutation. Disable KV only for this invalidation test;
    # normal CFG batching/equivalence tests above use native default KV.
    model.model_options["transformer_options"]["qwen_image21_cache"] = {
        "device": "off",
        "dtype": "default",
    }
    guider = comfy.samplers.CFGGuider(model)
    guider.set_conds(cond, [[cond[0][0].clone() * 0.7, {}]])
    guider.set_cfg(2)

    def mutate_negative(step, *args):
        if step == 1:
            guider.conds["negative"][0]["model_conds"]["c_crossattn"].cond.add_(0.1)

    guider.sample(
        noise.clone(),
        latent.clone(),
        sampler,
        sigmas,
        callback=mutate_negative,
        disable_pbar=True,
        seed=7,
    )
    summary = json.loads(
        max(
            Path(paper_tmp).glob("aligncache_qwen21/*/summary.json"),
            key=lambda p: p.stat().st_mtime_ns,
        ).read_text()
    )
    assert "condition_signature_changed" in summary["degradation_reasons"], summary
    assert summary["skip_count"] == 0

print("Native Qwen/CFGGuider/ModelPatcher/wrapper lifecycle contract passed")

# Native dynamic-input expansion verifies API field names without loading weights.
mapping = {
    **nodes.NODE_CLASS_MAPPINGS,
    "TextEncodeQwenImage21": TextEncodeQwenImage21,
    "QwenImage21Cache": QwenImage21Cache,
    "SaveImageAdvanced": SaveImageAdvanced,
    "AlignCacheQwenImage21": AlignCacheQwenImage21,
    "AlignCacheQwenImage21Benchmark": AlignCacheQwenImage21Benchmark,
}
workflow_dir = Path(__file__).resolve().parents[1] / "AlignCache4QwenImage21/workflows"
for path in workflow_dir.glob("*_api.json"):
    graph = json.loads(path.read_text())
    if path.name == "qwen21_t2i_aligncache_api.json":
        inputs = dict(graph["900"]["inputs"])
        inputs["model"] = patcher
        fixed = node.attach(**inputs)[0]
        config = fixed.model_options["transformer_options"][CONFIG_KEY]
        assert config.mode == "on" and config.base_threshold == 0.12
        assert config.predictor == "time_normalized_output" and config.beta == 0.4
        assert (
            config.max_consecutive_skips,
            config.warmup_steps,
            config.tail_steps,
        ) == (4, 5, 2)
    for key, entry in graph.items():
        cls = mapping[entry["class_type"]]
        schema = cls.INPUT_TYPES()
        if issubclass(cls, _io.ComfyNode):
            schema, _, _ = _io.get_finalized_class_inputs(schema, entry["inputs"])
        required = set(schema.get("required", {}))
        supplied = set(entry["inputs"])
        available = required | set(schema.get("optional", {}))
        assert not required - supplied, (key, required - supplied)
        assert not supplied - available, (key, supplied - available)
print("Native workflow schemas passed")
