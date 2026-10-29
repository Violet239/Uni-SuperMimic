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
    obs, _ = env.reset()
    action_dim = env.action_manager.total_action_dim
    assert action_dim == 20, action_dim
    # The policy receives normalized joints, target error, cube pose and
    # action history (20 + 20 + 3 + 6 + 20 = 69).  The critic group is the
    # SuperMimic goal frame (one 3x3 rotation = 9 values).
    assert tuple(obs["obs"].shape) == (2, 69), obs["obs"].shape
    assert tuple(obs["critic"].shape) == (2, 9), obs["critic"].shape
    contact = env.command_manager.get_term("hand2").tip_contact()
    assert tuple(contact.shape) == (2, 5), contact.shape
    assert torch.isfinite(contact).all()
    action = torch.zeros((2, action_dim), dtype=torch.float32, device=env.device)
    state = env.step(action)
    assert torch.isfinite(state.reward).all()
    print("hand2 env smoke: PASS")
    print("obs:", {key: tuple(value.shape) for key, value in obs.items()})
    print("action_dim:", action_dim)
    print("reward:", state.reward.tolist())
finally:
    env.close()
    env.backend.close()
