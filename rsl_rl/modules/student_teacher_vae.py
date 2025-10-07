# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation


class StudentTeacher(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_student_obs,
        num_teacher_obs,
        num_actions,
        num_vae_obs,
        latent_dim=32,
        student_encoder_dims=[256, 256, 256],
        student_decoder_dims=[256, 256, 256],
        teacher_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=0.1,
        **kwargs,
    ):
        if kwargs:
            print(
                "StudentTeacher.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = resolve_nn_activation(activation)
        self.loaded_teacher = False  # indicates if teacher has been loaded

        mlp_input_dim_s = num_student_obs
        mlp_input_dim_t = num_teacher_obs

        vae_obs_shape = num_vae_obs
        self.latent_dim = latent_dim
        self.vae_obs_shape = vae_obs_shape

        # student
        student_layers = []
        student_layers.append(nn.Linear(vae_obs_shape, student_encoder_dims[0]))
        student_layers.append(activation)
        for layer_index in range(len(student_encoder_dims)):
            if layer_index == len(student_encoder_dims) - 1:
                student_layers.append(nn.Linear(student_encoder_dims[layer_index], latent_dim * 2))
            else:
                student_layers.append(nn.Linear(student_encoder_dims[layer_index], student_encoder_dims[layer_index + 1]))
                student_layers.append(activation)
        self.student_encoder = nn.Sequential(*student_layers)

        # student_layers = []
        # student_layers.append(nn.Linear(mlp_input_dim_s - vae_obs_shape, student_decoder_dims[0]))
        # # student_layers.append(nn.Linear(mlp_input_dim_s, student_decoder_dims[0]))
        # student_layers.append(activation)
        # for layer_index in range(len(student_decoder_dims)):
        #     if layer_index == len(student_decoder_dims) - 1:
        #         student_layers.append(nn.Linear(student_decoder_dims[layer_index], latent_dim * 2))
        #     else:
        #         student_layers.append(nn.Linear(student_decoder_dims[layer_index], student_decoder_dims[layer_index + 1]))
        #         student_layers.append(activation)
        # self.student_prior = nn.Sequential(*student_layers)

        # # Initialize prior weights with Xavier uniform
        # for layer in self.student_prior:
        #     if isinstance(layer, nn.Linear):
        #         nn.init.xavier_uniform_(layer.weight, gain=0.01)
        #     if layer.bias is not None:
        #         nn.init.zeros_(layer.bias)


        student_layers = []
        student_layers.append(nn.Linear(latent_dim + mlp_input_dim_s - vae_obs_shape, student_decoder_dims[0]))
        # student_layers.append(nn.Linear(mlp_input_dim_s, student_decoder_dims[0]))
        student_layers.append(activation)
        for layer_index in range(len(student_decoder_dims)):
            if layer_index == len(student_decoder_dims) - 1:
                student_layers.append(nn.Linear(student_decoder_dims[layer_index], num_actions))
            else:
                student_layers.append(nn.Linear(student_decoder_dims[layer_index], student_decoder_dims[layer_index + 1]))
                student_layers.append(activation)
        self.student_decoder = nn.Sequential(*student_layers)

        # teacher
        teacher_layers = []
        teacher_layers.append(nn.Linear(mlp_input_dim_t - 1, teacher_hidden_dims[0]))
        teacher_layers.append(activation)
        for layer_index in range(len(teacher_hidden_dims)):
            if layer_index == len(teacher_hidden_dims) - 1:
                teacher_layers.append(nn.Linear(teacher_hidden_dims[layer_index], num_actions))
            else:
                teacher_layers.append(nn.Linear(teacher_hidden_dims[layer_index], teacher_hidden_dims[layer_index + 1]))
                teacher_layers.append(activation)
        self.teacher = nn.Sequential(*teacher_layers)
        self.teacher.eval()

        print(f"Student MLP: {self.student_encoder} + {self.student_decoder}")
        print(f"Teacher MLP: {self.teacher}")

        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        self._last_u = None

    def reset(self, dones=None, hidden_states=None):
        pass

    def forward(self):
        raise NotImplementedError

    def clear_last_u(self):
        self._last_u = None

    @property
    def last_u(self):
        last_u = self._last_u
        self._last_u = self.action_mean
        return last_u

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def prior_mean(self):
        return self.prior_distribution.mean
    
    @property
    def prior_std(self):
        return self.prior_distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        vae_obs = observations[..., :self.vae_obs_shape]
        mean_logvar = self.student_encoder(vae_obs)
        mean, logvar = torch.split(mean_logvar, [self.latent_dim, self.latent_dim], dim=-1)

        # prior_obs = observations[..., self.vae_obs_shape:]
        # prior_mean_logvar = self.student_prior(prior_obs)
        # prior_mean, prior_logvar = torch.split(prior_mean_logvar, [self.latent_dim, self.latent_dim], dim=-1)

        std = torch.exp(0.5 * logvar)
        self.distribution = Normal(mean, std)

        # prior_std = torch.exp(0.5 * prior_logvar)
        # self.prior_distribution = Normal(prior_mean, prior_std)

    def act(self, observations):
        self.update_distribution(observations)
        z = self.distribution.rsample()
        # z = self.distribution.mean
        decoder_obs = observations[..., self.vae_obs_shape:]
        z_obs = torch.cat([z, decoder_obs], dim=-1)
        action = self.student_decoder(z_obs)
        # action = self.student_decoder(observations)
        return action

    def act_inference(self, observations):
        self.update_distribution(observations)
        # z = self.distribution.rsample()
        z = self.distribution.mean
        decoder_obs = observations[..., self.vae_obs_shape:]
        z_obs = torch.cat([z, decoder_obs], dim=-1)
        action = self.student_decoder(z_obs)
        self.z = z
        # action = self.student_decoder(observations)
        return action

    def evaluate(self, teacher_observations):
        with torch.no_grad():
            actions = self.teacher(teacher_observations)
        return actions

    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the student and teacher networks.

        Args:
            state_dict (dict): State dictionary of the model.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                           module's state_dict() function.

        Returns:
            bool: Whether this training resumes a previous training. This flag is used by the `load()` function of
                  `OnPolicyRunner` to determine how to load further parameters.
        """
        # check if state_dict contains teacher and student or just teacher parameters
        if any("actor" in key and "teacher" not in key for key in state_dict.keys()):  # loading parameters from rl training
            # rename keys to match teacher and remove critic parameters
            teacher_state_dict = {}
            for key, value in state_dict.items():
                if "actor." in key:
                    teacher_state_dict[key.replace("actor.", "")] = value
            self.teacher.load_state_dict(teacher_state_dict, strict=strict)
            # also load recurrent memory if teacher is recurrent
            if self.is_recurrent and self.teacher_recurrent:
                raise NotImplementedError("Loading recurrent memory for the teacher is not implemented yet")  # TODO
            # set flag for successfully loading the parameters
            self.loaded_teacher = True
            self.teacher.eval()
            return False
        if any("student" in key for key in state_dict.keys()):  # loading parameters from distillation training
            super().load_state_dict(state_dict, strict=False)
            # set flag for successfully loading the parameters
            self.loaded_teacher = True
            self.teacher.eval()
            return True
        else:
            raise ValueError("state_dict does not contain student or teacher parameters")

    def get_hidden_states(self):
        return None

    def detach_hidden_states(self, dones=None):
        pass
