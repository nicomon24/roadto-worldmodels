'''
    Define a class for the VAE/GAN model.
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class VAEGAN(nn.Module):

    def __init__(self, latent_size=32):
        super(VAEGAN, self).__init__()
        self.latent_size = latent_size
        self.beta = 1
        # Convolutional layers
        self.conv1 = nn.Sequential(nn.Conv2d(1, 32, 4, stride=2), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, 4, stride=1), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, 4, stride=1), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(128, 256, 4, stride=1), nn.ReLU())
        # Deconvolutional layers
        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(4096, 256, 4, stride=1), nn.ReLU())
        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, stride=1), nn.ReLU())
        self.deconv3 = nn.Sequential(nn.ConvTranspose2d(128, 64, 4, stride=1), nn.ReLU())
        self.deconv4 = nn.Sequential(nn.ConvTranspose2d(64, 32, 4, stride=1), nn.ReLU())
        self.deconv5 = nn.Sequential(nn.ConvTranspose2d(32, 1, 4, stride=2), nn.Sigmoid())
        # Linear layers
        self.fc11 = nn.Linear(4096, self.latent_size)
        self.fc12 = nn.Linear(4096, self.latent_size)
        self.fc2 = nn.Linear(self.latent_size, 4096)
        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                #nn.init.xavier_normal_(m.weight, gain=1)
                nn.init.normal_(m.weight, mean=0, std=0.1)

    def encode(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.shape[0], -1)
        return self.fc11(x), self.fc12(x)

    def reparameterize(self, mu, log_sigma):
        if self.training:
            std = torch.exp(0.5 * log_sigma)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        z = self.fc2(z)
        z = z.view(z.shape[0], z.shape[1], 1, 1)
        z = self.deconv1(z)
        z = self.deconv2(z)
        z = self.deconv3(z)
        z = self.deconv4(z)
        z = self.deconv5(z)
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
        mse = ((x-rebuild)**2)
        mse = mse.view(mse.size(0), -1)
        reco_loss = torch.mean(torch.sum(mse, dim=1))
        # Normalization loss
        norm_loss = -0.5 * torch.sum(1 + log_sigma - mu**2 - log_sigma.exp(), dim=1)
        # KL tolerance
        norm_loss = torch.mean(norm_loss)
        # Total loss
        total_loss = reco_loss + self.beta * norm_loss
        return reco_loss, norm_loss, total_loss

    def optimize(self, x, optimizer):
        optimizer.zero_grad()
        reco_loss, norm_loss, total_loss = self.losses(x)
        total_loss.backward()
        optimizer.step()
        return reco_loss, norm_loss, total_loss
