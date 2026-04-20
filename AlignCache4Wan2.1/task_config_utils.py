from __future__ import annotations

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
