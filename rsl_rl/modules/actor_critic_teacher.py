# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation


class ActorCriticTeacher(nn.Module):
    """Actor-Critic module with an additional teacher policy head.

    - Actor: maps student/actor observations to action distribution.
    - Critic: maps critic/privileged observations to value.
    - Teacher: maps teacher/privileged observations to mean actions used for distillation.
    """

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_teacher_obs: int,
        num_actions: int,
        actor_hidden_dims: list[int] = [256, 256, 256],
        critic_hidden_dims: list[int] = [256, 256, 256],
        teacher_hidden_dims: list[int] = [256, 256, 256],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticTeacher.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = resolve_nn_activation(activation)

        # Policy (actor)
        actor_layers: list[nn.Module] = []
        actor_layers.append(nn.Linear(num_actor_obs, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for i in range(len(actor_hidden_dims)):
            if i == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[i], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[i], actor_hidden_dims[i + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function (critic)
        critic_layers: list[nn.Module] = []
        critic_layers.append(nn.Linear(num_critic_obs, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for i in range(len(critic_hidden_dims)):
            if i == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[i], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[i], critic_hidden_dims[i + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        # Teacher policy
        teacher_layers: list[nn.Module] = []
        teacher_layers.append(nn.Linear(num_teacher_obs, teacher_hidden_dims[0]))
        teacher_layers.append(activation)
        for i in range(len(teacher_hidden_dims)):
            if i == len(teacher_hidden_dims) - 1:
                teacher_layers.append(nn.Linear(teacher_hidden_dims[i], num_actions))
            else:
                teacher_layers.append(nn.Linear(teacher_hidden_dims[i], teacher_hidden_dims[i + 1]))
                teacher_layers.append(activation)
        self.teacher = nn.Sequential(*teacher_layers)
        self.teacher.eval()
        self.loaded_teacher = False

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")
        print(f"Teacher MLP: {self.teacher}")

        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution (populated in update_distribution)
        self.distribution: Normal | None = None
        Normal.set_default_validate_args(False)

    # not used at the moment
    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean  # type: ignore

    @property
    def action_std(self):
        return self.distribution.stddev  # type: ignore

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)  # type: ignore

    def update_distribution(self, observations: torch.Tensor):
        # compute mean
        mean = self.actor(observations)
        # compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # create distribution
        self.distribution = Normal(mean, std)

    def act(self, observations: torch.Tensor, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()  # type: ignore

    def get_actions_log_prob(self, actions: torch.Tensor):
        return self.distribution.log_prob(actions).sum(dim=-1)  # type: ignore

    def act_inference(self, observations: torch.Tensor):
        actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(self, critic_observations: torch.Tensor, **kwargs):
        value = self.critic(critic_observations)
        return value

    def evaluate_teacher(self, teacher_observations: torch.Tensor):
        with torch.no_grad():
            actions = self.teacher(teacher_observations)
        return actions

    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the actor-critic-teacher model.

        Returns:
            bool: Whether this training resumes a previous training (True) or loads teacher from PPO (False).
        """
        # check if state_dict contains actor.* (rl training) and not teacher.*
        if any("actor." in key for key in state_dict.keys()) and not any(
            key.startswith("teacher.") for key in state_dict.keys()
        ):
            # rename keys to match teacher and remove critic parameters
            teacher_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith("actor."):
                    teacher_state_dict[key.replace("actor.", "")] = value
            self.teacher.load_state_dict(teacher_state_dict, strict=strict)
            # set flag for successfully loading the parameters
            self.loaded_teacher = True
            self.teacher.eval()
            # also try to load actor/critic (non-strict)
            try:
                super().load_state_dict(state_dict, strict=False)
            except Exception:
                pass
            return False
        # otherwise load full checkpoint
        super().load_state_dict(state_dict, strict=strict)
        self.teacher.eval()
        if any(key.startswith("teacher.") for key in state_dict.keys()) or any(
            key.startswith("actor.") for key in state_dict.keys()
        ):
            self.loaded_teacher = True
        return True
