#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


from .state_estimator import VAE


class ActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        num_latent,
        history_length,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        encoder_hidden_dims=[256, 128, 64],
        decoder_hidden_dims=[64, 128, 256],
        activation="elu",
        init_noise_std=1.0,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = get_activation(activation)

        mlp_input_dim_a = (num_actor_obs - 3) // (history_length + 1)
        mlp_input_dim_c = num_critic_obs

        self.num_obs_history = (history_length + 1) * mlp_input_dim_a
        self.num_single_obs = mlp_input_dim_a   

        # estimator
        self.estimator = VAE(
            num_single_obs=mlp_input_dim_a,
            num_obs_history=self.num_obs_history,
            num_latent=num_latent,
            encoder_hidden_dims=encoder_hidden_dims,
            decoder_hidden_dims=decoder_hidden_dims,
        )

        self.p_sample = torch.zeros(1)

        # Policy
        actor_layers = []
        actor_layers.append(
            nn.Linear(mlp_input_dim_a + num_latent + 3, actor_hidden_dims[0])
        )
        actor_layers.append(activation)
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                actor_layers.append(
                    nn.Linear(actor_hidden_dims[layer_index], num_actions)
                )
            else:
                actor_layers.append(
                    nn.Linear(
                        actor_hidden_dims[layer_index],
                        actor_hidden_dims[layer_index + 1],
                    )
                )
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(
                    nn.Linear(
                        critic_hidden_dims[layer_index],
                        critic_hidden_dims[layer_index + 1],
                    )
                )
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.logstd = nn.Parameter(np.log(init_noise_std) * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        # Normal.set_default_validate_args = False

        self.estimator.apply(init_weights)
        # self.actor.apply(init_weights)
        # self.critic.apply(init_weights)
        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(
                mod for mod in sequential if isinstance(mod, nn.Linear)
            )
        ]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.actor(observations)
        self.distribution = Normal(mean, torch.exp(self.logstd) + 1e-3)

    def act(self, observations, **kwargs):
        obs_history, curr_obs = (
            observations[:, : -3],
            observations[:, self.num_obs_history - self.num_single_obs : -3],
        )

        gt_vel = observations[:, -3:]

        batch_size = curr_obs.size(0)
        p_sample = torch.rand(batch_size) < self.p_sample

        latent, vel = self.estimator.sample(obs_history)
        latent_mu, vel_mu = self.estimator.inference(obs_history)
        latent[~p_sample] = latent_mu[~p_sample]
        vel[~p_sample] = vel_mu[~p_sample]
        # vel[~p_sample] = gt_vel[~p_sample]

        latent = latent.detach()
        vel = vel.detach()
        self.update_distribution(torch.cat([curr_obs, latent, vel], dim=-1))
        try:
            act = self.distribution.sample()
        except:
            breakpoint()
        return act

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        obs_history, curr_obs = (
            observations[:, : -3],
            observations[:, self.num_obs_history - self.num_single_obs : -3],
        )
        latent, vel = self.estimator.inference(obs_history)
        latent = latent.detach()
        vel = vel.detach()
        actions_mean = self.actor(torch.cat([curr_obs, latent, vel], dim=-1))
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.CReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
