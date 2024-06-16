import torch.nn as nn
import torch
from torch.distributions import Normal
from torch.nn import functional as F

from .actor_critic import get_activation


class VAE(nn.Module):
    def __init__(
        self,
        num_single_obs,
        num_obs_history,
        num_latent,
        activation="elu",
        encoder_hidden_dims=[128, 64],
        decoder_hidden_dims=[512, 256, 128],
    ):
        super(VAE, self).__init__()
        self.num_latent = num_latent

        # Build Encoder
        activation = get_activation(activation)

        # Adaptation module
        modules = []
        modules.append(nn.Linear(num_obs_history, encoder_hidden_dims[0]))
        modules.append(activation)
        for l in range(len(encoder_hidden_dims)):
            if l == len(encoder_hidden_dims) - 2:
                pass
            else:
                modules.append(
                    nn.Linear(encoder_hidden_dims[l], encoder_hidden_dims[l + 1])
                )
                modules.append(activation)
        self.encoder = nn.Sequential(*modules)

        # variational parameters
        self.latent_mu = nn.Linear(encoder_hidden_dims[-1], num_latent)
        self.latent_var = nn.Linear(encoder_hidden_dims[-1], num_latent)
        self.vel_mu = nn.Linear(encoder_hidden_dims[-1], 3)
        self.vel_var = nn.Linear(encoder_hidden_dims[-1] * 4, 3)

        # Build Decoder
        modules = []

        decoder_input_dim = num_latent + 3
        modules.append(nn.Linear(decoder_input_dim, decoder_input_dim[0]))
        modules.append(activation)
        for l in range(len(decoder_hidden_dims)):
            if l == len(decoder_hidden_dims) - 1:
                modules.append(nn.Linear(decoder_hidden_dims[l], num_single_obs))
            else:
                modules.append(
                    nn.Linear(decoder_hidden_dims[l], decoder_hidden_dims[l + 1])
                )
                modules.append(activation)
        self.decoder = nn.Sequential(*modules)

    def encode(self, obs_history):
        encoded = self.encoder(obs_history)
        latent_mu = self.latent_mu(encoded)
        latent_var = self.latent_var(encoded)
        vel_mu = self.vel_mu(encoded)
        vel_var = self.vel_var(encoded)
        return latent_mu, latent_var, vel_mu, vel_var

    def decode(self, latent, vel):
        input = torch.cat([latent, vel], dim=-1)
        return self.decoder(input)

    def forward(self, obs_history):
        latent_mu, latent_var, vel_mu, vel_var = self.encode(obs_history)
        latent = self.reparameterize(latent_mu, latent_var)
        vel = self.reparameterize(vel_mu, vel_var)
        return latent, vel, latent_mu, latent_var

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def loss_fn(self, obs_history, next_single_obs, gt_vel, kld_weight=1.0):
        latent, vel, latent_mu, latent_var = self.forward(obs_history)
        # Reconstruction loss
        recons_next_obs = self.decode(latent, vel)
        recons_loss = F.mse_loss(recons_next_obs, next_single_obs, reduction="none").mean(-1)
        # Supervised loss
        vel_loss = F.mse_loss(vel, gt_vel, reduction="none").mean(-1)

        kld_loss = -0.5 * torch.sum(
            1 + latent_var - latent_mu**2 - latent_var.exp(), dim=1
        )

        loss = recons_loss + vel_loss + kld_weight * kld_loss
        
        return {
            "loss": loss,
            "recons_loss": recons_loss,
            "vel_loss": vel_loss,
            "kld_loss": kld_loss,
        }

    def sample(self, obs_history):
        latent, vel, _, _ = self.forward(obs_history)
        return latent, vel

    def inference(self, obs_history):
        latent_mu, latent_var, vel_mu, vel_var = self.encode(obs_history)
        return latent_mu, vel_mu
