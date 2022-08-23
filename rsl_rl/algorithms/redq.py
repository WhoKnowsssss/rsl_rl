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

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import ActorCritic
from rsl_rl.storage import ReplayBuffer

class REDQ:
    actor_critic: ActorCritic
    def __init__(self,
                 actor_critic,
                 num_learning_epochs=10,
                 mini_batch_size=256,
                 num_critics_update=2,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=[3e-4, 3e-4, 3e-4],
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu',
                 ):

        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None # initialized later
        
        self.actor_optimizer = optim.Adam(self.actor_critic.actor.parameters(), lr=learning_rate[0])
        self.actor_optimizer.add_param_group({"params": self.actor_critic.std})
        
        self.critic_optimizer = []
        for i in range(self.actor_critic.num_critics):
            self.critic_optimizer.append(optim.Adam(self.actor_critic.critic[i].parameters(), lr=learning_rate[1]))
        
        self.temperature_optimizer = optim.Adam(self.actor_critic.log_alpha, lr=learning_rate[2])
        self.transition = ReplayBuffer.Transition()

        # RED-Q parameters
        self.num_learning_epochs = num_learning_epochs
        self.num_critics_update = num_critics_update
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.mini_batch_size = mini_batch_size


    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = ReplayBuffer(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.device)

    def test_mode(self):
        self.actor_critic.test()
    
    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs):
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs).detach()        
        self.transition.observations = obs
        return self.transition.actions
    
    def process_env_step(self, next_obs, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        self.transition.next_observations = next_obs
        
        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()

    def update(self):
        obs, actions, rewards, next_obs, dones = self.storage.mini_batch_generator(self.mini_batch_size)
        
        mean_critic_loss = []
        for i in range(self.num_learning_epochs):

            indices = torch.randperm(self.actor_critic.num_critics, device=self.device)[:self.num_critics_update]
            # ---------------------------- update critic ---------------------------- #

            with torch.no_grad():
                # Get predicted next-state actions and Q values from target models
                next_action = self.actor_critic.sample(next_obs)
                next_log_prob = self.actor_critic.get_actions_log_prob(next_action)
                Q_target_next = torch.stack([self.actor_critic.evaluate_target(j, next_obs, next_action) for j in indices])
                Q_target_next = torch.min(Q_target_next, dim=0) - self.actor_critic.log_alpha.exp() * next_log_prob

            Q_targets = rewards + (self.gamma * (1 - dones) * Q_target_next)

            # Compute critic losses and update critics 
            for j in range(self.actor_critic.num_critics):
            # for critic, optim, target in zip(self.critics, self.optims, self.target_critics):
                Q = self.actor_critic.evaluate(j, obs, actions).cpu()
                Q_loss = torch.nn.functional.mse_loss(Q, Q_targets)
            
                # Update critic
                self.critic_optimizer[j].zero_grad()
                Q_loss.backward()
                mean_critic_loss.append(Q_loss.detach().cpu().item())
                self.critic_optimizer[j].step()
                # soft update of the targe
                self.actor_critic.soft_update(j)
            
            # ---------------------------- update actor ---------------------------- #
            if i == self.num_learning_epochs-1:
                actions = self.actor_critic.sample(obs)
                log_prob = self.actor_critic.get_actions_log_prob(obs)        
                
                Q = torch.stack([self.actor_critic.evaluate(j, obs, actions) for j in indices])
                Q = torch.min(Q, dim=0)

                actor_loss = (self.actor_critic.log_alpha.exp() * log_prob - Q ).mean()
                # Optimize the actor loss
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Compute alpha loss 
                alpha_loss = - (self.actor_critic.log_alpha.exp() * (log_prob + self.actor_critic.target_entropy)).mean()

                self.temperature_optimizer.zero_grad()
                alpha_loss.backward()
                self.temperature_optimizer.step()

            mean_critic_loss = sum(mean_critic_loss)/len(mean_critic_loss)

        return mean_critic_loss, actor_loss


    
