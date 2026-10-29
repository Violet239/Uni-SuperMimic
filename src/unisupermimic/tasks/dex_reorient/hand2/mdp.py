"""Differentiable MDP for the first Wuji Hand 2 SuperMimic port."""

from __future__ import annotations

import copy

import torch

from unisupermimic.rl.algos.supermimic.geometry import (
    invariant_features,
    mat_to_6d,
    rot_angle_sq,
)
from unisupermimic.rl.contracts.supermimic import State

NUM_J = 20
CONTEXT_DIM = 46
OBS_DIM = 69
GOAL_FRAME_DIM = 9
CONTACT_DIM = 5
MAX_STEP_REWARD = 1.0


class DexHand2Mdp:
    num_joints = NUM_J
    context_dim = CONTEXT_DIM
    obs_dim = OBS_DIM
    goal_frame_dim = GOAL_FRAME_DIM
    contact_dim = CONTACT_DIM
    max_step_reward = MAX_STEP_REWARD

    def __init__(self, default_q: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor):
        if tuple(default_q.shape) != (NUM_J,):
            raise ValueError(f"Expected default_q shape {(NUM_J,)}, got {default_q.shape}")
        if tuple(lower.shape) != (NUM_J,) or tuple(upper.shape) != (NUM_J,):
            raise ValueError("Expected 20 lower and upper joint limits")
        self.default_q = default_q.float()
        self.lower = lower.float()
        self.upper = upper.float()
        self.goal_rotation = torch.eye(3, device=default_q.device, dtype=torch.float32)
        self.goal_frame = self.goal_rotation.reshape(1, 9)
        self.position_noise = 0.008
        self.orientation_noise = 0.05
        self.joint_noise = 0.06

    def to(self, device: str | torch.device) -> DexHand2Mdp:
        device = torch.device(device)
        if device == self.default_q.device:
            return self
        new = copy.copy(self)
        new.default_q = self.default_q.to(device)
        new.lower = self.lower.to(device)
        new.upper = self.upper.to(device)
        new.goal_rotation = self.goal_rotation.to(device)
        new.position_noise = self.position_noise
        new.orientation_noise = self.orientation_noise
        new.joint_noise = self.joint_noise
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
        _, rotation, _, _, joint_pos, _ = state
        target = context[..., :NUM_J]
        center = 0.5 * (self.lower + self.upper)
        half = 0.5 * (self.upper - self.lower).clamp_min(1.0e-6)
        normalized_q = ((joint_pos - center) / half).clamp(-1.0, 1.0)
        normalized_target = ((target - center) / half).clamp(-1.0, 1.0)
        goal_rotation = context[..., NUM_J : NUM_J + 9].reshape(*rotation.shape[:-2], 3, 3)
        cube_ori_error = (rotation @ goal_rotation.transpose(-1, -2))[..., 1:, :].reshape(
            *rotation.shape[:-2], 6
        )
        return torch.cat(
            (
                normalized_q,
                normalized_q - normalized_target,
                state[0],
                cube_ori_error,
                previous_action,
            ),
            -1,
        )

    def add_noise(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.shape[-1] != OBS_DIM:
            raise ValueError(f"Expected observation width {OBS_DIM}, got {obs.shape[-1]}")
        noisy = obs.clone()
        noisy[..., 0:20].add_(torch.empty_like(noisy[..., 0:20]).uniform_(-self.joint_noise, self.joint_noise))
        noisy[..., 20:40].add_(torch.empty_like(noisy[..., 20:40]).uniform_(-self.joint_noise, self.joint_noise))
        noisy[..., 40:43].add_(torch.empty_like(noisy[..., 40:43]).uniform_(-self.position_noise, self.position_noise))
        noisy[..., 43:49].add_(torch.randn_like(noisy[..., 43:49]) * self.orientation_noise)
        return noisy

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
        context = torch.zeros((frames.shape[0], CONTEXT_DIM), device=state[0].device)
        context[:, NUM_J : NUM_J + 9] = self.goal_frame
        return self.reward_with_context(state, frames, action, previous_action, context)

    def reward_with_context(
        self,
        state: State,
        frames: torch.Tensor,
        action: torch.Tensor,
        previous_action: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        del frames
        position, rotation = state[0], state[1]
        goal_rotation = context[..., NUM_J : NUM_J + 9].reshape(*rotation.shape[:-2], 3, 3)
        orientation = torch.exp(-rot_angle_sq(rotation, goal_rotation) / 0.25)
        position_term = torch.exp(-torch.sum(position.square(), dim=-1) / 0.01)
        smoothness = torch.sum((action - previous_action).square(), dim=-1)
        return orientation * 0.8 + position_term * 0.2 - 0.001 * smoothness

    def reward_with_context_and_contact(
        self,
        state: State,
        frames: torch.Tensor,
        action: torch.Tensor,
        previous_action: torch.Tensor,
        context: torch.Tensor,
        contact_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Add the differentiable fingertip-contact term used in imagination.

        The simulator provides binary labels from five MuJoCo touch sensors;
        the world model predicts logits.  Centering the sigmoid at its
        no-contact prior keeps an untrained contact head reward-neutral while
        preserving a gradient toward establishing contact.
        """
        reward = self.reward_with_context(state, frames, action, previous_action, context)
        if contact_logits.shape[-1] != CONTACT_DIM:
            raise ValueError(
                f"Expected {CONTACT_DIM} contact logits, got {contact_logits.shape[-1]}"
            )
        contact_bonus = 0.5 * (torch.sigmoid(contact_logits) - 0.5).mean(dim=-1)
        return reward + contact_bonus

    def goal_with_context(
        self, frames: torch.Tensor, context: torch.Tensor, num_frames: int
    ) -> torch.Tensor:
        goal = context[..., NUM_J : NUM_J + 9]
        return goal.unsqueeze(1).expand(-1, num_frames, -1).reshape(
            frames.shape[0], num_frames * GOAL_FRAME_DIM
        )

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
