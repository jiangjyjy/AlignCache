import os
from pathlib import Path


def _default_checkpoints_root() -> Path:
    config_file = globals().get("__file__")
    if config_file:
        return Path(config_file).resolve().parents[3] / "checkpoints"
    return Path.cwd() / "checkpoints"


def _first_env_path(*env_vars: str) -> str | None:
    for env_var in env_vars:
        env_value = os.getenv(env_var)
        if env_value:
            return str(Path(env_value).expanduser())
    return None


def _resolve_local_pretrained(default_repo_id: str, *dir_names: str, env_vars: tuple[str, ...] = ()) -> str:
    env_value = _first_env_path(*env_vars)
    if env_value:
        return env_value

    checkpoints_root = Path(
        _first_env_path("ALIGNCACHE_CHECKPOINTS_DIR") or _default_checkpoints_root()
    ).expanduser()
    for dirname in dir_names:
        candidate = checkpoints_root / dirname
        if candidate.exists():
            return str(candidate.resolve())
    return default_repo_id


resolution = "240p"
aspect_ratio = "9:16"
num_frames = 51
fps = 24
frame_interval = 1
save_fps = 24

save_dir = "./samples/samples/"
seed = 42
batch_size = 1
multi_resolution = "STDiT2"
dtype = "bf16"
condition_frame_length = 5
align = 5

model = dict(
    type="STDiT3-XL/2",
    from_pretrained=_resolve_local_pretrained(
        "hpcai-tech/OpenSora-STDiT-v3",
        "OpenSora-STDiT-v3",
        env_vars=("ALIGNCACHE_OPENSORA_MODEL_PATH",),
    ),
    qk_norm=True,
    enable_flash_attn=True,
    enable_layernorm_kernel=True,
)
vae = dict(
    type="OpenSoraVAE_V1_2",
    from_pretrained=_resolve_local_pretrained(
        "hpcai-tech/OpenSora-VAE-v1.2",
        "OpenSora-VAE-v1.2",
        env_vars=("ALIGNCACHE_OPENSORA_VAE_PATH",),
    ),
    micro_frame_size=17,
    micro_batch_size=4,
)
text_encoder = dict(
    type="t5",
    from_pretrained=_resolve_local_pretrained(
        "DeepFloyd/t5-v1_1-xxl",
        "t5-v1_1-xxl",
        env_vars=("ALIGNCACHE_OPENSORA_T5_PATH",),
    ),
    model_max_length=300,
)
scheduler = dict(
    type="rflow",
    use_timestep_transform=True,
    num_sampling_steps=30,
    cfg_scale=7.0,
)

aes = 6.5
flow = None
