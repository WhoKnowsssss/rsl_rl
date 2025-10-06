# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# torch
import torch
import torch.nn as nn
import torch.optim as optim

# rsl-rl
from rsl_rl.modules import StudentTeacher, StudentTeacherRecurrent
from rsl_rl.storage import RolloutStorage


class Distillation:
    """Distillation algorithm for training a student model to mimic a teacher model."""

    policy: StudentTeacher | StudentTeacherRecurrent
    """The student teacher model."""

    def __init__(
        self,
        policy,
        num_learning_epochs=1,
        gradient_length=15,
        learning_rate=5e-4,
        kl_coeff_start=1e-3,
        kl_coeff_end=1e-4,
        consistency_coeff=0.005,
        loss_type="mse",
        device="cpu",
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
    ):
        # device-related parameters
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        # Multi-GPU parameters
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        self.rnd = None  # TODO: remove when runner has a proper base class

        # distillation components
        self.policy = policy
        self.policy.to(self.device)
        self.storage = None  # initialized later
        self.optimizer = optim.AdamW(
            list(self.policy.student_encoder.parameters()) + list(self.policy.student_decoder.parameters()), 
            lr=learning_rate
        )
        self.transition = RolloutStorage.Transition()
        self.last_hidden_states = None

        # distillation parameters
        self.num_learning_epochs = num_learning_epochs
        self.gradient_length = gradient_length
        self.learning_rate = learning_rate

        # initialize the loss function
        if loss_type == "mse":
            self.loss_fn = nn.functional.mse_loss
        elif loss_type == "huber":
            self.loss_fn = nn.functional.huber_loss
        else:
            raise ValueError(f"Unknown loss type: {loss_type}. Supported types are: mse, huber")

        self.num_updates = 0
        self.kl_coeff_start = kl_coeff_start
        self.kl_coeff_end = kl_coeff_end
        self.consistency_coeff = consistency_coeff

        self.obs_reflect_op, self.action_reflect_op = self.get_reflection_ops()

    def get_reflection_ops(self):
        """Get reflection operations for symmetry augmentation"""
        from rsl_rl.algorithms.symm_utils import get_reflect_op, get_reflect_reps, BODY_NAMES, JOINT_NAMES
        Q, Rd, Rd_pseudo, Q_Rd, Q_Rd_pseudo, num_bodies = get_reflect_reps(BODY_NAMES, JOINT_NAMES)

        Q_Rd_pseudo = Q_Rd_pseudo.view(num_bodies, 3, num_bodies, 3)
        Q_Rd = Q_Rd.view(num_bodies, 3, num_bodies, 3)
        
        # obs_reflect_reps = [Q] * 10 + [Rd] * 1 + [Rd, Rd_pseudo] * 1 + [Rd] + [Rd_pseudo] * 2 + [Q] * 3
        # # first line: encoder, second line: decoder and prior
        Q_Rd_pseudo_rot6d = torch.zeros(6 * num_bodies, 6 * num_bodies)
        for i in range(num_bodies):
            for j in range(num_bodies):
                Q_Rd_pseudo_rot6d[6 * i : 6 * i + 3, 6 * j : 6 * j + 3] = Q_Rd[i, :, j, :]
                Q_Rd_pseudo_rot6d[6 * i + 3 : 6 * i + 6, 6 * j + 3 : 6 * j + 6] = Q_Rd_pseudo[i, :, j, :]

        Q_Rd = Q_Rd.view(num_bodies* 3, num_bodies* 3)
        obs_reflect_reps = [Q] * 10 + [Rd] * 1 + [Rd, Rd_pseudo] * 1 + [Q_Rd] + [Q_Rd_pseudo_rot6d] + [Rd] + [Rd_pseudo] + [Q] * 3 \
                         + [Rd] + [Rd_pseudo] * 2 + [Q] * 3
        # note: for rot6d, use [Rd, Rd_pseudo] to reflect
        action_reflect_reps = [Q]

        obs_reflect_op = get_reflect_op(obs_reflect_reps).to(torch.float32).to(self.device)
        action_reflect_op = get_reflect_op(action_reflect_reps).to(torch.float32).to(self.device)
        return obs_reflect_op, action_reflect_op

    def init_storage(
        self, training_type, num_envs, num_transitions_per_env, student_obs_shape, teacher_obs_shape, actions_shape
    ):
        # create rollout storage
        self.storage = RolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            student_obs_shape,
            teacher_obs_shape,
            actions_shape,
            None,
            self.device,
        )

    def act(self, obs, teacher_obs):
        # compute the actions
        self.transition.actions = self.policy.act(obs).detach()
        self.transition.privileged_actions = self.policy.evaluate(teacher_obs).detach()
        # record the observations
        self.transition.observations = obs
        self.transition.privileged_observations = teacher_obs
        return self.transition.actions
    
    def update_kl_coeff(self, current_learning_iteration, total_iterations):
        """Update the KL coefficient based on the current iteration."""
        progress = current_learning_iteration / total_iterations
        if progress >= 0.25 and progress <= 0.5:
            # Linearly interpolate between kl_coeff_start and kl_coeff_end
            alpha = (progress - 0.25) / 0.25
            self.kl_coeff = self.kl_coeff_start * (1 - alpha) + self.kl_coeff_end * alpha
        elif progress > 0.5:
            # Keep it at kl_coeff_end after 50% of total iterations
            self.kl_coeff = self.kl_coeff_end
        else:
            # Before 25% of total iterations, keep it at kl_coeff_start
            self.kl_coeff = self.kl_coeff_start

    def process_env_step(self, rewards, dones, infos):
        # record the rewards and dones
        self.transition.rewards = rewards
        self.transition.dones = dones
        # record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def vae_losses(
        self,
        mu, logvar
    ):
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return kl_loss

    def consistency_losses(
        self,
        last_u, current_u
    ):  
        if last_u is None:
            return torch.tensor(0.0, device=self.device)
        consistency_loss = nn.functional.mse_loss(current_u, last_u)
        return consistency_loss
    
    def symmetric_augment(self, obs, action):
        obs = torch.cat([obs, obs @ self.obs_reflect_op], dim=0)
        action = torch.cat([action, action @ self.action_reflect_op], dim=0)
        return obs, action
    
    def update(self, current_learning_iteration, total_iterations):  # noqa: C901
        
        self.num_updates += 1
        mean_behavior_loss = 0
        loss = 0
        cnt = 0

        self.update_kl_coeff(current_learning_iteration, total_iterations)

        for epoch in range(self.num_learning_epochs):
            self.policy.reset(hidden_states=self.last_hidden_states)
            self.policy.detach_hidden_states()
            self.policy.clear_last_u()
            for obs, _, _, privileged_actions, dones in self.storage.generator():
                
                # -------------------------- write symmetry augmentation here --------------------------
                obs, privileged_actions = self.symmetric_augment(obs, privileged_actions)
                # inference the student for gradient computation
                actions = self.policy.act(obs)

                vae_loss = self.vae_losses(self.policy.action_mean, 2 * torch.log(self.policy.action_std + 1e-6))
                consistency_loss = self.consistency_losses(self.policy.last_u, self.policy.action_mean)

                # behavior cloning loss
                behavior_loss = self.loss_fn(actions, privileged_actions)

                # total loss
                loss = loss + behavior_loss + self.kl_coeff * vae_loss + self.consistency_coeff * consistency_loss
                mean_behavior_loss += behavior_loss.item()
                vae_loss += vae_loss.item()
                consistency_loss += consistency_loss.item()
                cnt += 1

                # gradient step
                if cnt % self.gradient_length == 0:
                    self.optimizer.zero_grad()
                    loss.backward()
                    if self.is_multi_gpu:
                        self.reduce_parameters()
                    self.optimizer.step()
                    self.policy.detach_hidden_states()
                    loss = 0

                # reset dones
                self.policy.reset(dones.view(-1))
                self.policy.detach_hidden_states(dones.view(-1))

        mean_behavior_loss /= cnt
        self.storage.clear()
        self.last_hidden_states = self.policy.get_hidden_states()
        self.policy.detach_hidden_states()

        # construct the loss dictionary
        loss_dict = {"behavior": mean_behavior_loss, 
                     "kl": vae_loss / cnt, 
                     "consistency": consistency_loss / cnt,
                     "kl_coeff": self.kl_coeff
                    }

        return loss_dict

    """
    Helper functions
    """

    def broadcast_parameters(self):
        """Broadcast model parameters to all GPUs."""
        # obtain the model parameters on current GPU
        model_params = [self.policy.state_dict()]
        # broadcast the model parameters
        torch.distributed.broadcast_object_list(model_params, src=0)
        # load the model parameters on all GPUs from source GPU
        self.policy.load_state_dict(model_params[0])

    def reduce_parameters(self):
        """Collect gradients from all GPUs and average them.

        This function is called after the backward pass to synchronize the gradients across all GPUs.
        """
        # Create a tensor to store the gradients
        grads = [param.grad.view(-1) for param in self.policy.parameters() if param.grad is not None]
        all_grads = torch.cat(grads)
        # Average the gradients across all GPUs
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        # Update the gradients for all parameters with the reduced gradients
        offset = 0
        for param in self.policy.parameters():
            if param.grad is not None:
                numel = param.numel()
                # copy data back from shared buffer
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                # update the offset for the next parameter
                offset += numel
