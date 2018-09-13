'''
    Specification of the Variational AutoEncoder in PyTorch.
    Change this for changes on the architecture.
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class VAE(nn.Module):

    def __init__(self, arch='base_car_racing', latent_size=32, beta=1.0, dropout_proba=0.2):
        super(VAE, self).__init__()
        self.beta = beta
        self.latent_size = latent_size
        self.dropout_proba = dropout_proba
        # Convolutional layers
        self.conv1 = nn.Sequential(nn.BatchNorm2d(3), nn.Conv2d(3, 32, 4, stride=2), nn.ReLU(), nn.Dropout(self.dropout_proba))
        self.conv2 = nn.Sequential(nn.BatchNorm2d(32), nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(), nn.Dropout(self.dropout_proba))
        self.conv3 = nn.Sequential(nn.BatchNorm2d(64), nn.Conv2d(64, 128, 4, stride=2), nn.ReLU(), nn.Dropout(self.dropout_proba))
        self.conv4 = nn.Sequential(nn.BatchNorm2d(128), nn.Conv2d(128, 256, 4, stride=2), nn.ReLU(), nn.Dropout(self.dropout_proba))
        # Deconvolutional layers
        self.deconv1 = nn.Sequential(nn.BatchNorm2d(1024), nn.ConvTranspose2d(1024, 128, 5, stride=2), nn.ReLU(), nn.Dropout(self.dropout_proba))
        self.deconv2 = nn.Sequential(nn.BatchNorm2d(128), nn.ConvTranspose2d(128, 64, 5, stride=2), nn.ReLU(), nn.Dropout(self.dropout_proba))
        self.deconv3 = nn.Sequential(nn.BatchNorm2d(64), nn.ConvTranspose2d(64, 32, 6, stride=2), nn.ReLU(), nn.Dropout(self.dropout_proba))
        self.deconv4 = nn.Sequential(nn.BatchNorm2d(32), nn.ConvTranspose2d(32, 3, 6, stride=2), nn.ReLU(), nn.Dropout(self.dropout_proba))
        # Linear layers
        self.fc11 = nn.Linear(1024, latent_size)
        self.fc12 = nn.Linear(1024, latent_size)
        self.fc2 = nn.Linear(latent_size, 1024)

    def encode(self, x):
        # Convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # Flatten and return mu and logsigma
        x = x.view(-1, 1024)
        return self.fc11(x), self.fc12(x)

    def reparameterize(self, mu, log_sigma):
        if self.training:
            std = torch.exp(0.5 * log_sigma)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        z = self.fc2(z).view(-1, 1024, 1, 1)
        z = self.deconv1(z)
        z = self.deconv2(z)
        z = self.deconv3(z)
        z = self.deconv4(z)
        return z

    def forward(self, x):
        mu, log_sigma = self.encode(x)
        z = self.reparameterize(mu, log_sigma)
        rebuild = self.decode(z)
        assert x.shape == rebuild.shape, 'Image dimension not aligned.'
        return mu, log_sigma, z, rebuild

    def losses(self, x):
        # Forward pass
        mu, log_sigma, z, rebuild = self(x)
        # Reconstruction loss
        reco_loss = torch.mean((x - rebuild) ** 2)
        # Normalization loss
        norm_loss = -torch.mean(0.5 * (1 + log_sigma - mu**2 - log_sigma.exp()))
        # Total loss
        total_loss = reco_loss + self.beta * norm_loss
        return reco_loss, norm_loss, total_loss

    def optimize(self, x, optimizer):
        optimizer.zero_grad()
        reco_loss, norm_loss, total_loss = self.losses(x)
        total_loss.backward()
        optimizer.step()
        return reco_loss, norm_loss, total_loss
