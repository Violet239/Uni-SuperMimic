"""Differentiable MDP for the first Wuji Hand 2 SuperMimic port."""

from __future__ import annotations

import copy

import torch

from unisupermimic.rl.algos.supermimic.geometry import (
    invariant_features,
    mat_to_6d,
    rot_angle_sq,
    yaw_mat,
)

from unisupermimic.rl.contracts.supermimic import State

NUM_J = 20
CONTEXT_DIM = 46
OBS_DIM = 69
GOAL_FRAME_DIM = 9
MAX_STEP_REWARD = 1.0


class DexHand2Mdp:
    num_joints = NUM_J
    context_dim = CONTEXT_DIM
    obs_dim = OBS_DIM
    goal_frame_dim = GOAL_FRAME_DIM
    max_step_reward = MAX_STEP_REWARD

    def __init__(self, default_q: torch.Tensor):
        if tuple(default_q.shape) != (NUM_J,):
            raise ValueError(f"Expected default_q shape {(NUM_J,)}, got {default_q.shape}")
        self.default_q = default_q.float()
        self.goal_rotation = torch.eye(3, device=default_q.device, dtype=torch.float32)
        self.goal_frame = self.goal_rotation.reshape(1, 9)

    def to(self, device: str | torch.device) -> DexHand2Mdp:
        device = torch.device(device)
        if device == self.default_q.device:
            return self
        new = copy.copy(self)
        new.default_q = self.default_q.to(device)
        new.goal_rotation = self.goal_rotation.to(device)
        new.goal_frame = self.goal_frame.to(device)
        return new

    def obs(
        self,
        state: State,
        previous_action: torch.Tensor,
        context: torch.Tensor,
        frames: torch.Tensor,
    ) -> torch.Tensor:
        del frames
        _, rotation, _, _, joint_pos, joint_vel = state
        target = context[..., :NUM_J]
        relative_rotation = mat_to_6d(rotation)
        orientation_error = relative_rotation
        return torch.cat(
            (
                joint_pos - self.default_q,
                joint_pos - target,
                state[0],
                orientation_error,
                previous_action,
            ),
            -1,
        )

    def add_noise(self, obs: torch.Tensor) -> torch.Tensor:
        return obs

    def goal(self, frames: torch.Tensor, num_frames: int) -> torch.Tensor:
        batch = frames.shape[0]
        return self.goal_frame.expand(batch, num_frames, GOAL_FRAME_DIM).reshape(
            batch, num_frames * GOAL_FRAME_DIM
        )

    def reward(
        self,
        state: State,
        frames: torch.Tensor,
        action: torch.Tensor,
        previous_action: torch.Tensor,
    ) -> torch.Tensor:
        del frames
        position, rotation = state[0], state[1]
        orientation = torch.exp(-rot_angle_sq(rotation, self.goal_rotation) / 0.25)
        position_term = torch.exp(-torch.sum(position.square(), dim=-1) / 0.01)
        smoothness = torch.sum((action - previous_action).square(), dim=-1)
        return orientation * 0.8 + position_term * 0.2 - 0.001 * smoothness

    def soft_termination(self, state: State, frames: torch.Tensor) -> torch.Tensor:
        del state
        return torch.zeros(frames.shape[0], device=self.default_q.device)

    def critic_features(
        self,
        state: State,
        frames: torch.Tensor,
        normalized_context: torch.Tensor,
    ) -> torch.Tensor:
        del frames
        return torch.cat((invariant_features(state), normalized_context), -1)

    def normalize_context(self, context: torch.Tensor) -> torch.Tensor:
        return context

    def valid_imagination_start(self, frames: torch.Tensor, horizon: int) -> torch.Tensor:
        del horizon
        return torch.ones_like(frames, dtype=torch.bool)
