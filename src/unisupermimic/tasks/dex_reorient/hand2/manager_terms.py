"""Hand 2 command and transition terms for Uni-SuperMimic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from unilab.base.reset_selectors import resolve_reset_env_ids
from unilab.envs.mdp.actions.actions import BaseAction, BaseActionCfg
from unilab.managers import ActionTerm, ActionTermCfg, CommandTerm, CommandTermCfg

from unisupermimic.rl.algos.supermimic.geometry import matmul, matvec, quat_to_mat, so3_exp
from unisupermimic.rl.contracts.supermimic import State

from .mdp import CONTEXT_DIM, NUM_J, DexHand2Mdp

if TYPE_CHECKING:
    from unilab.managers.types import ManagerBasedRlEnv



@dataclass(kw_only=True)
class DexHand2ActionCfg(BaseActionCfg):
    """Absolute joint target with the Dex-Supermimic EMA controller."""

    action_scale: float = 0.5
    ema_alpha: float = 0.3
    warmup_steps: int = 20

    def build(self, env: ManagerBasedRlEnv) -> DexHand2Action:
        return DexHand2Action(self, env)


class DexHand2Action(BaseAction):
    cfg: DexHand2ActionCfg

    def __init__(self, cfg: DexHand2ActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._default = self._entity.data.default_joint_pos(self._device)[:, self._target_index].clone()
        limits = self._entity.data.soft_joint_pos_limits(self._device)[self._target_index]
        self._lower = limits[:, 0].clone()
        self._upper = limits[:, 1].clone()
        self._previous_target = self._default.clone()
        self._processed_actions.zero_()

    def process_actions(self, actions: torch.Tensor) -> None:
        if actions.shape != self._raw_actions.shape:
            raise ValueError(f"DexHand2Action expected {self._raw_actions.shape}, got {actions.shape}")
        self._raw_actions.copy_(actions)
        raw_target = self._default + self.cfg.action_scale * actions.clamp(-1.0, 1.0)
        raw_target = raw_target.clamp(self._lower, self._upper)
        target = self.cfg.ema_alpha * raw_target + (1.0 - self.cfg.ema_alpha) * self._previous_target
        warmup = (self._env.episode_length_buf < self.cfg.warmup_steps).unsqueeze(-1)
        target = torch.where(warmup, self._default, target)
        self._processed_actions.copy_(target)
        self._previous_target.copy_(target)

    def apply_actions(self) -> None:
        self._entity.set_full_width_joint_position_target(self._processed_actions)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            self._previous_target.copy_(self._default)
            self._processed_actions.copy_(self._default)
        else:
            self._previous_target[env_ids] = self._default[env_ids]
            self._processed_actions[env_ids] = self._default[env_ids]

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
        limits = self.robot.data.soft_joint_pos_limits(self._device)
        self.mdp = DexHand2Mdp(self.default_q, limits[:, 0], limits[:, 1])
        self.f = torch.zeros(self.num_envs, dtype=torch.int64, device=self._device)
        self.xi = torch.zeros((self.num_envs, cfg.context_dim), dtype=torch.float32, device=self._device)
        self.xi[:, :NUM_J] = self.default_q
        self.xi[:, NUM_J : NUM_J + 9] = self.mdp.goal_frame
        self.a_prev = torch.zeros((self.num_envs, NUM_J), dtype=torch.float32, device=self._device)
        self.ep_step = torch.zeros(self.num_envs, dtype=torch.int64, device=self._device)
        self.goal_rotation = torch.eye(3, device=self._device).expand(self.num_envs, 3, 3).clone()
        self.xi[:, NUM_J : NUM_J + 9] = self.goal_rotation.flatten(-2)

    @property
    def command(self) -> torch.Tensor:
        return self.goal_rotation.flatten(-2)

    @property
    def tensor_body_names(self) -> tuple[str, ...]:
        return ("right_palm_link", "right_wrist_tag_frame")

    @property
    def tensor_sensor_names(self) -> tuple[str, ...]:
        """Tip touch sensors requested by the command's packed read phase.

        The MuJoCo scene exposes one scalar touch sensor per fingertip.  The
        command owns the read because rewards and diagnostics need the same
        phase-consistent values as the state tuple.
        """
        return HAND2_TOUCH_SENSOR_NAMES

    def state(self) -> State:
        read_plan = getattr(self._env.scene, "read_plan", None)
        if read_plan is None or not read_plan.ready:
            zeros = torch.zeros((self.num_envs, 3), device=self._device)
            rotation = torch.eye(3, device=self._device).expand(self.num_envs, 3, 3)
            joint_pos = self.default_q.expand(self.num_envs, -1)
            joint_vel = torch.zeros_like(joint_pos)
            return zeros, rotation, zeros, zeros, joint_pos, joint_vel
        tag = read_plan.body_tensor_view(self.robot, ("right_wrist_tag_frame",))
        obj = read_plan.body_tensor_view(self.object, ("cube",))
        joints = read_plan.joint_tensor_view(self.robot)
        tag_rot = quat_to_mat(tag.quat_w[:, 0, :])
        cube_rot = quat_to_mat(obj.quat_w[:, 0, :])
        inv_tag = tag_rot.transpose(-1, -2)
        position = matvec(inv_tag, obj.pos_w[:, 0, :] - tag.pos_w[:, 0, :])
        rotation = matmul(inv_tag, cube_rot)
        lin_vel = matvec(inv_tag, obj.lin_vel_w[:, 0, :] - tag.lin_vel_w[:, 0, :])
        ang_vel = matvec(inv_tag, obj.ang_vel_w[:, 0, :])
        return position, rotation, lin_vel, ang_vel, joints.joint_pos, joints.joint_vel

    def tip_contact(self) -> torch.Tensor:
        """Return the five fingertip touch values for every environment.

        ``SceneTensorReadPlan`` is refreshed once per manager read phase, so
        this accessor never crosses the backend boundary itself.  A zero
        fallback keeps cold-path construction and reset diagnostics safe
        before the first packed read is published.
        """
        read_plan = getattr(self._env.scene, "read_plan", None)
        if read_plan is None or not read_plan.ready:
            return torch.zeros((self.num_envs, len(HAND2_TOUCH_SENSOR_NAMES)), device=self._device)
        views = read_plan.sensor_tensor_views(self.robot, HAND2_TOUCH_SENSOR_NAMES)
        values = []
        for name in HAND2_TOUCH_SENSOR_NAMES:
            value = views.values[name]
            values.append(value.reshape(self.num_envs, -1)[:, 0])
        return torch.stack(values, dim=-1)

    def observation(self) -> torch.Tensor:
        return self.mdp.obs(self.state(), self.a_prev, self.xi, self.f)

    def goal(self, num_frames: int) -> torch.Tensor:
        return self.mdp.goal_with_context(self.f, self.xi, num_frames)

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
        count = int(reset_mask.sum().item())
        if count == 0:
            return
        axis = self._rng.uniform(-1.0, 1.0, (count, 3))
        axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
        angle = self._rng.uniform(-3.14159265, 3.14159265, (count,))
        rotations = so3_exp(axis * angle.unsqueeze(-1))
        self.goal_rotation[reset_mask] = rotations
        self.xi[reset_mask, NUM_J : NUM_J + 9] = rotations.flatten(-2)

    def _update_command(self) -> None:
        self.f.add_(1)
        self.ep_step.add_(1)
        action_term = self._env.action_manager.get_term("joint_pos")
        self.xi[:, :NUM_J] = action_term.processed_action
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


def hand2_orientation_reward(env: ManagerBasedRlEnv, command_name: str = "hand2") -> torch.Tensor:
    command = env.command_manager.get_term(command_name)
    state = command.state()
    goal = command.xi[:, NUM_J : NUM_J + 9].reshape(-1, 3, 3)
    angle = torch.sqrt(torch.clamp(
        (torch.diagonal(state[1].transpose(-1, -2) @ goal, dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5,
        min=0.0,
    ))
    return torch.where(angle <= 0.2, torch.ones_like(angle), (1.0 - (angle - 0.2) / 3.14159265).clamp_min(0.0))


def hand2_pose_penalty(env: ManagerBasedRlEnv, command_name: str = "hand2") -> torch.Tensor:
    command = env.command_manager.get_term(command_name)
    return torch.sum((command.state()[4] - command.mdp.default_q).square(), dim=-1)


def hand2_cage_escape(env: ManagerBasedRlEnv, command_name: str = "hand2", margin: float = 0.15) -> torch.Tensor:
    command = env.command_manager.get_term(command_name)
    position = command.state()[0]
    return torch.relu(torch.linalg.norm(position, dim=-1) - margin)


def hand2_cage_drop(env: ManagerBasedRlEnv, command_name: str = "hand2", margin: float = 0.15) -> torch.Tensor:
    command = env.command_manager.get_term(command_name)
    return torch.linalg.norm(command.state()[0], dim=-1) > margin


HAND2_TOUCH_SENSOR_NAMES = tuple(
    f"right_finger{finger}_tip_touch" for finger in range(1, 6)
)


def hand2_tip_contact_reward(
    env: ManagerBasedRlEnv,
    command_name: str = "hand2",
    sensor_scale: float = 1.0e-3,
) -> torch.Tensor:
    """Smoothly reward simultaneous fingertip contacts with the object.

    MuJoCo touch sensors report non-negative scalar forces.  Normalising by a
    small force scale and applying ``tanh`` keeps the term bounded while still
    providing a useful gradient when contact is first established.
    """
    command = env.command_manager.get_term(command_name)
    contact = command.tip_contact().clamp_min(0.0)
    return torch.tanh(contact / max(float(sensor_scale), 1.0e-6)).mean(dim=-1)


hand2_tip_contact_reward.tensor_sensor_names = HAND2_TOUCH_SENSOR_NAMES
