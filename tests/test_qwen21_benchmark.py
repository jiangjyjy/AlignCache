import pytest
import torch

from AlignCache4QwenImage21.benchmark import (
    BENCHMARK_KEY,
    BENCHMARK_STATE_KEY,
    MeasurementConfig,
    MeasurementState,
    benchmark_model_wrapper,
    benchmark_outer_wrapper,
)
from AlignCache4QwenImage21.runtime import WRAPPER_KEY, wrapper_reason


def test_measurement_wrapper_preserves_output_and_arguments():
    state = MeasurementState(MeasurementConfig(".", "nonce", profile=True))
    args = [
        torch.ones(1, 64, 2, 2),
        torch.ones(1),
        torch.ones(1, 2, 4),
        None,
        None,
        {BENCHMARK_STATE_KEY: state},
    ]
    output = torch.ones_like(args[0])
    calls = []

    def executor(*given, **kwargs):
        calls.append((given, kwargs))
        return output

    assert benchmark_model_wrapper(executor, *args, control=None) is output
    assert all(a is b for a, b in zip(args, calls[0][0]))
    assert calls[0][1] == {"control": None}
    assert state.forward_attempts == state.forward_completed == 1
    assert len(state.rows) == 1 and state.rows[0]["host_seconds"] >= 0
    assert state.rows[0]["cuda_seconds"] is None


def test_normal_measurement_does_not_create_cuda_events(monkeypatch):
    state = MeasurementState(MeasurementConfig(".", "nonce", profile=False))

    def unexpected(*args, **kwargs):
        raise AssertionError("Normal measurement must not construct per-step events")

    monkeypatch.setattr(torch.cuda, "Event", unexpected)
    value = torch.ones(1, 64, 2, 2)
    assert (
        benchmark_model_wrapper(
            lambda *args, **kwargs: value,
            value,
            torch.ones(1),
            value,
            transformer_options={BENCHMARK_STATE_KEY: state},
        )
        is value
    )
    assert not state.events and not state.rows


def test_only_known_observer_functions_are_trusted():
    known = {
        "outer_sample": {BENCHMARK_KEY: [benchmark_outer_wrapper]},
        "diffusion_model": {
            WRAPPER_KEY: [object()],
            BENCHMARK_KEY: [benchmark_model_wrapper],
        },
    }
    assert wrapper_reason(known) is None
    known["diffusion_model"][BENCHMARK_KEY] = [lambda *args: None]
    assert wrapper_reason(known) == "external_wrapper"
    known["diffusion_model"] = {
        BENCHMARK_KEY: [benchmark_model_wrapper],
        WRAPPER_KEY: [object()],
    }
    assert wrapper_reason(known) == "measurement_must_follow_routing"


def test_failed_forward_count_and_profile_row():
    state = MeasurementState(MeasurementConfig(".", "nonce", profile=True))

    def fail(*args, **kwargs):
        raise RuntimeError("model error")

    x = torch.ones(1, 64, 2, 2)
    with pytest.raises(RuntimeError, match="model error"):
        benchmark_model_wrapper(
            fail, x, torch.ones(1), x, transformer_options={BENCHMARK_STATE_KEY: state}
        )
    assert state.forward_attempts == 1 and state.forward_completed == 0
    assert len(state.rows) == 1


def test_cuda_profile_records_events_without_step_synchronization(monkeypatch):
    from types import SimpleNamespace

    order = []
    stream = object()

    class Event:
        def __init__(self, enable_timing):
            assert enable_timing

        def record(self, selected_stream):
            assert selected_stream is stream
            order.append("event")

        def elapsed_time(self, other):
            raise AssertionError(
                "Events must be resolved only after the outer boundary"
            )

    def unexpected_sync(*args, **kwargs):
        raise AssertionError("Profiler must not synchronize at each diffusion call")

    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: stream)
    monkeypatch.setattr(torch.cuda, "synchronize", unexpected_sync)
    state = MeasurementState(MeasurementConfig(".", "nonce", profile=True))
    x = SimpleNamespace(device=torch.device("cuda:0"))
    output = object()

    def native(*args, **kwargs):
        order.append("native")
        return output

    result = benchmark_model_wrapper(
        native, x, None, None, transformer_options={BENCHMARK_STATE_KEY: state}
    )
    assert result is output
    assert order == ["event", "native", "event"]
    assert len(state.events) == len(state.rows) == 1
    assert state.rows[0]["cuda_seconds"] is None
