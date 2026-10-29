"""Hand 2 command and transition terms for Uni-SuperMimic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from unilab.managers import CommandTerm, CommandTermCfg
from unisupermimic.rl.algos.supermimic.geometry import matmul, matvec, quat_to_mat
from unisupermimic.rl.contracts.supermimic import State

from .mdp import CONTEXT_DIM, DexHand2Mdp, NUM_J
from unilab.base.reset_selectors import resolve_reset_env_ids

if TYPE_CHECKING:
    from unilab.managers.types import ManagerBasedRlEnv


@dataclass(kw_only=True)
class DexHand2CommandCfg(CommandTermCfg):
    entity_name: str = "robot"
    object_name: str = "object"
    context_dim: int = CONTEXT_DIM

    def build(self, env: ManagerBasedRlEnv) -> DexHand2Command:
        return DexHand2Command(self, env)


class DexHand2Command(CommandTerm):
    cfg: DexHand2CommandCfg

    def __init__(self, cfg: DexHand2CommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.entity_name]
        self.object = env.scene[cfg.object_name]
        self.default_q = self.robot.data.default_joint_pos(self._device)[0].clone()
        self.mdp = DexHand2Mdp(self.default_q)
        self.f = torch.zeros(self.num_envs, dtype=torch.int64, device=self._device)
        self.xi = torch.zeros((self.num_envs, cfg.context_dim), dtype=torch.float32, device=self._device)
        self.xi[:, :NUM_J] = self.default_q
        self.xi[:, NUM_J : NUM_J + 9] = self.mdp.goal_frame
        self.a_prev = torch.zeros((self.num_envs, NUM_J), dtype=torch.float32, device=self._device)
        self.ep_step = torch.zeros(self.num_envs, dtype=torch.int64, device=self._device)

    @property
    def command(self) -> torch.Tensor:
        return self.mdp.goal_frame.expand(self.num_envs, -1)

    @property
    def tensor_body_names(self) -> tuple[str, ...]:
        return ("right_palm_link",)

    def state(self) -> State:
        read_plan = getattr(self._env.scene, "read_plan", None)
        if read_plan is None or not read_plan.ready:
            zeros = torch.zeros((self.num_envs, 3), device=self._device)
            rotation = torch.eye(3, device=self._device).expand(self.num_envs, 3, 3)
            joint_pos = self.default_q.expand(self.num_envs, -1)
            joint_vel = torch.zeros_like(joint_pos)
            return zeros, rotation, zeros, zeros, joint_pos, joint_vel
        palm = read_plan.body_tensor_view(self.robot, ("right_palm_link",))
        obj = read_plan.body_tensor_view(self.object, ("cube",))
        joints = read_plan.joint_tensor_view(self.robot)
        palm_rot = quat_to_mat(palm.quat_w[:, 0, :])
        cube_rot = quat_to_mat(obj.quat_w[:, 0, :])
        inv_palm = palm_rot.transpose(-1, -2)
        position = matvec(inv_palm, obj.pos_w[:, 0, :] - palm.pos_w[:, 0, :])
        rotation = matmul(inv_palm, cube_rot)
        lin_vel = matvec(inv_palm, obj.lin_vel_w[:, 0, :] - palm.lin_vel_w[:, 0, :])
        ang_vel = matvec(inv_palm, obj.ang_vel_w[:, 0, :])
        return position, rotation, lin_vel, ang_vel, joints.joint_pos, joints.joint_vel

    def observation(self) -> torch.Tensor:
        return self.mdp.obs(self.state(), self.a_prev, self.xi, self.f)

    def goal(self, num_frames: int) -> torch.Tensor:
        return self.mdp.goal(self.f, num_frames)

    def reset(self, env_ids=None, *, publish_metrics: bool = True):
        extras = super().reset(env_ids, publish_metrics=publish_metrics)
        ids = resolve_reset_env_ids(
            env_ids, num_envs=self.num_envs, device=self._device, owner="DexHand2Command.reset"
        )
        self.f[ids] = 0
        self.ep_step[ids] = 0
        self.a_prev[ids] = 0.0
        return extras

    def _update_metrics(self) -> None:
        return

    def _resample_command(self, reset_mask: torch.Tensor) -> None:
        del reset_mask

    def _update_command(self) -> None:
        self.f.add_(1)
        self.ep_step.add_(1)
        self.a_prev.copy_(self._env.action_manager.action)

    def refresh_reset(self, reset_mask: torch.Tensor) -> None:
        del reset_mask

    def post_compute_reset(self) -> None:
        return


def hand2_obs(env: ManagerBasedRlEnv, command_name: str = "hand2") -> torch.Tensor:
    return env.command_manager.get_term(command_name).observation()


def hand2_goal(
    env: ManagerBasedRlEnv,
    command_name: str = "hand2",
    num_frames: int = 1,
    entity_name: str = "object",
) -> torch.Tensor:
    del entity_name
    return env.command_manager.get_term(command_name).goal(num_frames)


hand2_goal.tensor_body_names = ("cube",)
