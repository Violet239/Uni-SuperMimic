from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from unilab.base.config_adapter import create_env
from unilab.training import ensure_registries
from unisupermimic.scripts.train_supermimic import build_env_override

config_dir = str(Path("src/unisupermimic/conf/supermimic").resolve())
with initialize_config_dir(version_base="1.3", config_dir=config_dir):
    cfg = compose(
        config_name="config",
        overrides=["task=dex_hand2_reorient/mujoco", "algo.num_envs=2"],
    )
ensure_registries()
env = create_env(
    cfg,
    num_envs=2,
    env_cfg_override=build_env_override(cfg, eval_mode=False, learner_device="cpu"),
)
try:
    obs, info = env.reset()
    assert tuple(obs["obs"].shape) == (2, 69), obs["obs"].shape
    assert tuple(obs["critic"].shape) == (2, 9), obs["critic"].shape
    action_dim = env.action_manager.total_action_dim
    assert action_dim == 20, action_dim
    action = torch.zeros((2, action_dim), dtype=torch.float32, device=env.device)
    state = env.step(action)
    assert "supermimic" in state.info
    assert tuple(state.info["supermimic"]["s"].shape) == (2, 53)
    assert tuple(state.info["supermimic"]["xi"].shape) == (2, 46)
    assert torch.isfinite(state.reward).all()
    print("hand2 SuperMimic env smoke: PASS")
    print("obs:", {key: tuple(value.shape) for key, value in obs.items()})
    print("transition:", {key: tuple(value.shape) for key, value in state.info["supermimic"].items()})
finally:
    env.close()
    env.backend.close()
