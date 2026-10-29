"""Wuji Hand 2 SuperMimic environment and MuJoCo backend factory."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import torch
from unilab.base.backend_factory import create_backend, env_backend_kwargs
from unilab.base.cpu_runtime import apply_cpu_confinement
from unilab.base.entity import EntityCfg
from unilab.base.reset_selectors import resolve_reset_env_ids
from unilab.base.torch_env import TorchEnvState
from unilab.envs.manager_based_rl_env import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from unisim.backend.base import SimBackend

from unisupermimic.rl.contracts.supermimic import (
    SUPERMIMIC_STATE_KEY,
    SUPERMIMIC_TRANSITION_KEY,
    SuperMimicMdpProtocol,
)
from unisupermimic.tasks.motion_tracking.g1.supermimic.manager_terms import (
    make_current_info,
    make_transition_info,
    merge_current_info,
)

from .manager_terms import DexHand2Command
from .robot import HAND2_BODY_NAMES

SUPPORTED_BACKENDS = ("mujoco",)


class DexHand2SuperMimicEnv(ManagerBasedRlEnv):
    def __init__(self, cfg: ManagerBasedRlEnvCfg, backend: SimBackend, num_envs: int):
        if not math.isclose(cfg.ctrl_dt, 0.02, rel_tol=0.0, abs_tol=1.0e-9):
            raise ValueError("Dex Hand 2 SuperMimic requires ctrl_dt=0.02")
        super().__init__(cfg, backend, num_envs)
        commands = [
            self.command_manager.get_term(name)
            for name in self.command_manager.active_terms
            if isinstance(self.command_manager.get_term(name), DexHand2Command)
        ]
        if len(commands) != 1:
            raise ValueError("Dex Hand 2 SuperMimic requires exactly one DexHand2Command")
        self._supermimic_command = commands[0]
        self._pre_action: dict[str, Any] | None = None

    @property
    def supermimic_mdp(self) -> SuperMimicMdpProtocol:
        return self._supermimic_command.mdp

    def apply_action(self, actions: torch.Tensor, state: TorchEnvState) -> torch.Tensor:
        command = self._supermimic_command
        self._pre_action = {
            "state": tuple(value.detach().clone() for value in command.state()),
            "frames": command.f.detach().clone(),
            "context": command.xi.detach().clone(),
            "previous_action": command.a_prev.detach().clone(),
            "ep_step": command.ep_step.detach().clone(),
            "action": actions.detach().clone(),
        }
        return super().apply_action(actions, state)

    def _update_state_in_read_phase(self, state: TorchEnvState) -> TorchEnvState:
        updated = super()._update_state_in_read_phase(state)
        if self._pre_action is None:
            raise RuntimeError("Dex Hand 2 update_state requires a preceding apply_action snapshot")
        command = self._supermimic_command
        transition = make_transition_info(
            self._pre_action["state"],
            command.state(),
            self._pre_action["action"],
            self._pre_action["previous_action"],
            self._pre_action["context"],
            updated.reward,
            self._pre_action["frames"],
            self._pre_action["ep_step"],
            updated.terminated,
            updated.truncated,
        )
        # Store post-step fingertip contact labels for the optional contact
        # head in Uni-SuperMimic's world model.  The touch sensors are read
        # from the same packed phase as ``command.state()``.
        transition["contact"] = (command.tip_contact() > 1.0e-6).to(torch.float32)
        updated.info[SUPERMIMIC_TRANSITION_KEY] = transition
        updated.info[SUPERMIMIC_STATE_KEY] = self._current_info()
        self._pre_action = None
        return updated

    def _current_info(self) -> dict[str, torch.Tensor]:
        command = self._supermimic_command
        return make_current_info(
            command.state(), command.f, command.xi, command.a_prev, command.ep_step
        )

    def reset(
        self,
        env_ids: torch.Tensor | None = None,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        if env_ids is None:
            reset_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        else:
            ids = resolve_reset_env_ids(
                env_ids,
                num_envs=self.num_envs,
                device=self.device,
                owner="DexHand2SuperMimicEnv.reset",
            )
            reset_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            reset_mask[ids] = True
        obs, info = super().reset(env_ids, seed=seed, options=options)
        assert self._state is not None
        current = self._current_info()
        previous = self._state.info.get(SUPERMIMIC_STATE_KEY)
        if previous is not None:
            current = merge_current_info(previous, current, reset_mask)
        self._state.info[SUPERMIMIC_STATE_KEY] = current
        info[SUPERMIMIC_STATE_KEY] = current
        return obs, info


def make_hand2_env(
    cfg: ManagerBasedRlEnvCfg,
    num_envs: int = 1,
    backend_type: str = "mujoco",
) -> DexHand2SuperMimicEnv:
    if backend_type not in SUPPORTED_BACKENDS:
        raise ValueError(f"Hand 2 SuperMimic supports {SUPPORTED_BACKENDS}, got {backend_type!r}")
    if cfg.scene is None:
        raise ValueError("Hand 2 task requires a scene")
    robot_cfg = cfg.scene.entities.get("robot")
    if not isinstance(robot_cfg, EntityCfg):
        raise TypeError("Hand 2 robot scene entity must be an EntityCfg")
    if robot_cfg.root_body_name != "right_palm_link":
        raise ValueError("Hand 2 robot root_body_name must be right_palm_link")
    if robot_cfg.body_names is None:
        cfg.scene.entities["robot"] = replace(robot_cfg, body_names=HAND2_BODY_NAMES)
    cfg.validate()
    apply_cpu_confinement(cfg.cpu_ids)
    backend = create_backend(
        backend_type,
        cfg.scene,
        num_envs,
        cfg.sim_dt,
        body_state_required=True,
        base_name=robot_cfg.root_body_name,
        **env_backend_kwargs(cfg),
    )
    try:
        return DexHand2SuperMimicEnv(cfg, backend, num_envs)
    except Exception:
        backend.cleanup_scene_assets()
        raise


__all__ = ["DexHand2SuperMimicEnv", "SUPPORTED_BACKENDS", "make_hand2_env"]
