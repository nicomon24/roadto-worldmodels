'''
    This file creates a joint policy and VAE network. This can be trained
    in both directions. This is specifically designed for 32x32 black/White
    images (convs and deconvs are specific) and for the Car-Racing-v0 environment.
    If the method works, we can try to extend it.
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as dist

class VAEPolicy(nn.Module):

    def __init__(self, latent_size=32, vae_beta=1.0, dropout_proba=0.0, kl_tolerance=0.5, avoidance='none', optimizer=None):
        super(VAEPolicy, self).__init__()
        self.vae_beta = vae_beta
        self.latent_size = latent_size
        self.dropout_proba = dropout_proba
        self.kl_tolerance = kl_tolerance
        self.n_centroids = 16
        # Convolutional layers
        self.conv1 = nn.Sequential(nn.Conv2d(1, 32, 4, stride=2), nn.ReLU(), nn.Dropout(self.dropout_proba))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(), nn.Dropout(self.dropout_proba))
        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, 4, stride=2), nn.ReLU(), nn.Dropout(self.dropout_proba))
        # Deconvolutional layers
        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(512, 128, 5, stride=2), nn.ReLU(), nn.Dropout(self.dropout_proba))
        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(128, 64, 6, stride=2), nn.ReLU(), nn.Dropout(self.dropout_proba))
        self.deconv3 = nn.Sequential(nn.ConvTranspose2d(64, 1, 6, stride=2), nn.Sigmoid(), nn.Dropout(self.dropout_proba))
        # Linear layers
        self.fc11 = nn.Linear(512, latent_size)
        self.fc12 = nn.Linear(512, latent_size)
        self.fc2 = nn.Linear(latent_size, 512)
        # Policy layers
        self.linear1 = nn.Linear(self.latent_size, 16)
        self.linear2 = nn.Linear(16, 3)
        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                #nn.init.xavier_normal_(m.weight, gain=1)
                nn.init.normal_(m.weight, mean=0, std=0.1)
        if optimizer is None:
            self.optimizer = optim.Adam(self.parameters(), lr=1e-03)
        else:
            raise Exception("Not now buddy.")
        # Dynamic VAE
        self.vae_batch = []
        self.max_batch_size = 100
        self.avoidance = avoidance
        self.avoidance_threshold = 0.01
        if self.avoidance == 'centroid':
            self.generate_centroids()

    def encode(self, x):
        # Convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # Flatten and return mu and log_sigma
        x = x.view(-1, 512)
        return self.fc11(x), self.fc12(x)

    def decode(self, z):
        z = self.fc2(z).view(-1, 512, 1, 1)
        # Deconvolutional layers
        z = self.deconv1(z)
        z = self.deconv2(z)
        z = self.deconv3(z)
        return z

    def encode_decode(self, x):
        mu, log_sigma = self.encode(x)
        z = self.reparameterize(mu, log_sigma)
        rebuild = self.decode(z)
        return rebuild

    def reparameterize(self, mu, log_sigma):
        if self.training:
            std = torch.exp(0.5 * log_sigma)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def losses(self, x):
        # Forward pass
        mu, log_sigma = self.encode(x)
        z = self.reparameterize(mu, log_sigma)
        rebuild = self.decode(z)
        # Reconstruction loss
        mse = ((x-rebuild)**2)
        mse = mse.view(mse.size(0), -1)
        reco_loss = torch.mean(torch.sum(mse, dim=1))
        # Normalization loss
        norm_loss = -0.5 * torch.sum(1 + log_sigma - mu**2 - log_sigma.exp(), dim=1)
        norm_loss = torch.clamp(norm_loss, 0, self.kl_tolerance * self.latent_size)
        # KL tolerance
        norm_loss = torch.mean(norm_loss)
        # Total loss
        total_loss = reco_loss + self.vae_beta * norm_loss
        return reco_loss, norm_loss, total_loss

    def optimize_vae(self, x):
        # Append observation to batch
        is_added_to_batch = 0
        if self.avoidance == 'centroid':
            avoidance_score = self.centroids_distance(x).item()
        elif self.avoidance == 'self':
            avoidance_score = self.self_distance(x).item()
        else:
            avoidance_score = 1
        if avoidance_score > self.avoidance_threshold:
            self.vae_batch.append(x)
            is_added_to_batch = 1
        # Check if the batch is full
        if len(self.vae_batch) == self.max_batch_size:
            batch = torch.cat(self.vae_batch, dim=0)
            # Optimize
            self.optimizer.zero_grad()
            reco_loss, norm_loss, total_loss = self.losses(batch)
            total_loss.backward()
            self.optimizer.step()
            # Clear current batch
            self.vae_batch = []
            if self.avoidance == 'centroid':
                self.generate_centroids()
        else:
            reco_loss, norm_loss, total_loss = self.losses(x)
        return reco_loss.detach().numpy(), norm_loss.detach().numpy(), total_loss.detach().numpy(), is_added_to_batch, avoidance_score

    def generate_centroids(self):
        # Sample gaussian noise to feed the latent vector
        gaussian_noise = torch.randn((self.n_centroids, self.latent_size))
        self.centroids = self.decode(gaussian_noise)

    def centroids_distance(self, observation):
        if not hasattr(self, 'centroids'):
            return 0
        else:
            return torch.mean(torch.mean(((self.centroids - observation)**2).view(-1, 32 * 32 * 1), dim=1)) #NOTE: is mean right?

    def self_distance(self, observation):
        mse = ((observation-self.encode_decode(observation))**2)
        mse = mse.view(mse.size(0), -1)
        reco_loss = torch.mean(torch.sum(mse, dim=1))
        return reco_loss

    def act(self, x):
        VARIANCE = 0.1
        # Set a 4D shape
        x = x.view(1, 1, 32, 32)
        # First get the hidden representation
        mu, log_sigma = self.encode(x)
        # Compute alpha and beta for the beta distribution
        z = F.relu(self.linear1(mu))
        out = F.relu(self.linear2(z))
        # Sample the beta distribution
        a_dist = dist.MultivariateNormal(out, covariance_matrix=VARIANCE*torch.eye(3))
        actions = a_dist.sample()
        log_proba = a_dist.log_prob(actions)
        # Now move the 3 beta samples in the action space
        # Note: only the first action, steer, must be rescaled
        return actions[0].numpy(), log_proba

    def optimize_policy(self, J):
        raise Exception('To be implemented')
