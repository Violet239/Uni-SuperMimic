"""Register the minimal Wuji Hand 2 reorientation environment."""

from unilab.base import registry
from unilab.envs import ManagerBasedRlEnvCfg

from .env import SUPPORTED_BACKENDS, make_hand2_env

TASK_NAME = "DexHand2Reorient"
registry.register_env_config(TASK_NAME, ManagerBasedRlEnvCfg)
for backend in SUPPORTED_BACKENDS:
    registry.register_env(TASK_NAME, make_hand2_env, sim_backend=backend)

__all__ = ["TASK_NAME"]
