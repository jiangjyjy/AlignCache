import os
from pathlib import Path


HF_WEIGHT_FILENAMES = (
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
)
CHECKPOINT_GLOB_PATTERNS = (
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
    "OpenSora-v1-HQ-16x512x512.pth",
    "OpenSora-v1-HQ-16x256x256.pth",
    "OpenSora-v1-16x256x256.pth",
    "*.safetensors",
    "*.bin",
    "*.pt",
    "*.pth",
)


def find_local_checkpoint_file(path: str | None) -> str | None:
    if not path:
        return None
    directory = Path(os.path.abspath(os.path.expanduser(str(path))))
    if not directory.is_dir():
        return None
    seen: set[str] = set()
    for pattern in CHECKPOINT_GLOB_PATTERNS:
        for candidate in sorted(directory.glob(pattern)):
            if not candidate.is_file():
                continue
            resolved = str(candidate.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            return resolved
    return None


def is_legacy_opensora_v1_checkpoint(path: str | None) -> bool:
    candidate = os.path.basename(str(path or ""))
    return candidate.startswith("OpenSora-v1-") and candidate.endswith((".pt", ".pth"))


def resolve_pretrained_source(path: str | None) -> tuple[str | None, str | None]:
    if path is None:
        return None, None
    text = str(path)
    expanded = os.path.expanduser(text)
    resolved = os.path.abspath(expanded)
    if os.path.isabs(text) or text.startswith((".", "~")) or os.path.exists(resolved):
        if os.path.isdir(resolved) and not is_hf_pretrained_directory(resolved):
            checkpoint_file = find_local_checkpoint_file(resolved)
            if checkpoint_file is not None:
                return checkpoint_file, checkpoint_file
        return resolved, resolved
    return text, None


def should_use_hf_from_pretrained(source: str | None, local_path: str | None) -> bool:
    if source is None:
        return False
    if local_path is None:
        return True

    resolved_local = os.path.abspath(os.path.expanduser(str(local_path)))
    if os.path.isfile(resolved_local):
        return False
    if os.path.isdir(resolved_local):
        return is_hf_pretrained_directory(resolved_local)
    return False


def is_hf_pretrained_directory(path: str | None) -> bool:
    if not path:
        return False
    target = os.path.abspath(os.path.expanduser(str(path)))
    if not os.path.isdir(target):
        return False
    if os.path.isfile(os.path.join(target, "model_index.json")):
        return True
    if not os.path.isfile(os.path.join(target, "config.json")):
        return False
    return any(os.path.isfile(os.path.join(target, filename)) for filename in HF_WEIGHT_FILENAMES)
