# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.distributions import Categorical
from torch.nn.modules import rnn

class ActorCriticRedQ(nn.Module):
    is_recurrent = False
    def __init__(self,  num_actor_obs,
                        num_critic_obs,
                        num_actions,
                        num_critics=10,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        activation='elu',
                        init_noise_std=0.1,
                        target_critic_stepside=0.005,
                        **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super().__init__()

        activation = get_activation(activation)

        mlp_input_dim_a = num_actor_obs
        mlp_input_dim_c = num_critic_obs

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                if isinstance(num_actions, int):
                    actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions, bias=True))
                    torch.nn.init.normal_(actor_layers[-1].weight, mean=0.0, std=0.1)
                elif isinstance(num_actions, list):
                    class MultiHeadActor(nn.Module):
                        def __init__(self, in_dim, out_dim):
                            super().__init__()
                            self.heads = nn.ModuleList([nn.Linear(in_dim, i) for i in out_dim])
                            torch.nn.init.normal_(self.heads[0].weight, mean=0.0, std=0.01)
                            torch.nn.init.normal_(self.heads[1].weight, mean=0.0, std=0.01)
                        def forward(self, x):
                            return [self.heads[i](x) for i in range(len(self.heads))]
                    actor_layers.append(MultiHeadActor(actor_hidden_dims[l], num_actions))
                    self.num_actions = [num_actions[0], 1]

            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        def build_critic(mlp_input_dim_c, critic_hidden_dims, activation):
            critic_layers = []
            critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
            critic_layers.append(activation)
            for l in range(len(critic_hidden_dims)):
                if l == len(critic_hidden_dims) - 1:
                    critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
                else:
                    critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                    critic_layers.append(activation)
            return nn.Sequential(*critic_layers)
        
        self.critic = []
        self.num_critics = num_critics
        self.target_critic_stepside = target_critic_stepside
        self.critic = nn.ModuleList([build_critic(mlp_input_dim_c + sum(num_actions), critic_hidden_dims, activation) for i in range(self.num_critics)])
        self.target_critic = nn.ModuleList([build_critic(mlp_input_dim_c + sum(num_actions), critic_hidden_dims, activation) for i in range(self.num_critics)])

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        if isinstance(num_actions, int):
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
            self.log_alpha = nn.Parameter(torch.zeros(num_actions))
        elif isinstance(num_actions, list):
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions[0]), requires_grad=False)
            self.log_alpha = nn.Parameter(torch.zeros(num_actions[0]))

        self.target_entropy = -12
        
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        Categorical.set_default_validate_args = False

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


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
        if isinstance(mean, list):
            if len(mean) > 2:
                raise NotImplementedError
            class MultiHeadDistribution:
                def __init__(self, dists):
                    self.dists = dists
                def sample(self):
                    mean = self.dists[0].rsample()
                    mean = torch.tanh(mean)
                    return torch.cat([mean, self.dists[1].sample()], dim=-1)
                def log_prob(self, actions):
                    actions = torch.split(actions, [12, 1], dim=-1)
                    u = torch.atanh(actions[0])
                    log_prob = self.dists[0].log_prob(u) - torch.log(1 - torch.square(u)).sum(-1)

                    return torch.cat([log_prob, self.dists[1].log_prob(actions[1])], dim=-1)
                def entropy(self):
                    return self.dists[1].entropy().sum(dim=-1, keepdim=True)
                @property
                def mean(self):
                    return self.dists[0].mean
                @property
                def stddev(self):
                    return self.dists[0].stddev
                
            self.distribution = MultiHeadDistribution([
                                        Normal(mean[0], self.std), 
                                        Categorical(torch.softmax(mean[1], dim=-1).unsqueeze(1))
                                        ])
        else:
            self.distribution = Normal(mean, mean*0. + self.std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        actions_mean = self.actor(observations)
        if isinstance(actions_mean, list):
            print(torch.softmax(actions_mean[1], dim=-1))
            return torch.cat([actions_mean[0], torch.argmax(actions_mean[1], dim=-1, keepdim=True)], dim=-1)
        return actions_mean

    def evaluate(self, critic_idx, critic_observations, actions, **kwargs):
        return self.critic[critic_idx](torch.cat([critic_observations, actions], dim=-1))

    def evaluate_target(self, critic_idx, critic_observations, actions, **kwargs):
        return self.target_critic[critic_idx](torch.cat([critic_observations, actions], dim=-1))

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None

def soft_update(self, critic_idx):
    """Soft update model parameters.
    θ_target = τ*θ_local + (1 - τ)*θ_target
    Params
    ======
        local_model: PyTorch model (weights will be copied from)
        target_model: PyTorch model (weights will be copied to)
        tau (float): interpolation parameter 
    """
    for target_param, local_param in zip(self.target_critic[critic_idx].parameters(), self.critic[critic_idx].parameters()):
        target_param.data.copy_(self.target_critic_stepside*local_param.data + (1.0-self.target_critic_stepside)*target_param.data)