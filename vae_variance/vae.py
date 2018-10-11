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

    def __init__(self, kl_tolerance=0.5):
        super(VAE, self).__init__()
        self.kl_tolerance = kl_tolerance

    def encode(self, x):
        raise Exception("This should not be called.")

    def reparameterize(self, mu, log_sigma):
        raise Exception("This should not be called.")

    def decode(self, z):
        raise Exception("This should not be called.")

    '''
        Forward, losses and optimize functions are equal for each VAE
    '''
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

def VAEbyArch(arch):
    if arch == 'base_cr_64':
        return BaseCRVAE64
    elif arch == 'base_cr_32':
        return BaseCRVAE32
    else:
        raise Exception("Unrecognized architecture.")

class BaseCRVAE64(VAE):

    def __init__(self, latent_size=32, beta=1.0, dropout_proba=0.0):
        super(BaseCRVAE64, self).__init__()
        self.beta = beta
        self.latent_size = latent_size
        self.dropout_proba = dropout_proba
        # Convolutional layers
        self.conv1 = nn.Sequential(nn.Conv2d(1, 32, 4, stride=2), nn.ReLU(), nn.Dropout(self.dropout_proba))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(), nn.Dropout(self.dropout_proba))
        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, 4, stride=2), nn.ReLU(), nn.Dropout(self.dropout_proba))
        self.conv4 = nn.Sequential(nn.Conv2d(128, 256, 4, stride=2), nn.ReLU(), nn.Dropout(self.dropout_proba))
        # Deconvolutional layers
        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(1024, 128, 5, stride=2), nn.ReLU(), nn.Dropout(self.dropout_proba))
        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(128, 64, 5, stride=2), nn.ReLU(), nn.Dropout(self.dropout_proba))
        self.deconv3 = nn.Sequential(nn.ConvTranspose2d(64, 32, 6, stride=2), nn.ReLU(), nn.Dropout(self.dropout_proba))
        self.deconv4 = nn.Sequential(nn.ConvTranspose2d(32, 1, 6, stride=2), nn.Sigmoid(), nn.Dropout(self.dropout_proba))
        # Linear layers
        self.fc11 = nn.Linear(1024, latent_size)
        self.fc12 = nn.Linear(1024, latent_size)
        self.fc2 = nn.Linear(latent_size, 1024)
        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                #nn.init.xavier_normal_(m.weight, gain=1)
                nn.init.normal_(m.weight, mean=0, std=0.1)

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

class BaseCRVAE32(VAE):

    def __init__(self, latent_size=32, beta=1.0, dropout_proba=0.0, kl_tolerance=0.5):
        super(BaseCRVAE32, self).__init__(kl_tolerance=kl_tolerance)
        self.beta = beta
        self.latent_size = latent_size
        self.dropout_proba = dropout_proba
        self.kl_tolerance = kl_tolerance
        # Convolutional layers
        self.conv1 = nn.Sequential(nn.BatchNorm2d(1), nn.Conv2d(1, 32, 4, stride=2), nn.ReLU(), nn.Dropout(self.dropout_proba))
        self.conv2 = nn.Sequential(nn.BatchNorm2d(32), nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(), nn.Dropout(self.dropout_proba))
        self.conv3 = nn.Sequential(nn.BatchNorm2d(64), nn.Conv2d(64, 128, 4, stride=2), nn.ReLU(), nn.Dropout(self.dropout_proba))
        # Deconvolutional layers
        self.deconv1 = nn.Sequential(nn.BatchNorm2d(512), nn.ConvTranspose2d(512, 128, 5, stride=2), nn.ReLU(), nn.Dropout(self.dropout_proba))
        self.deconv2 = nn.Sequential(nn.BatchNorm2d(128), nn.ConvTranspose2d(128, 64, 6, stride=2), nn.ReLU(), nn.Dropout(self.dropout_proba))
        self.deconv3 = nn.Sequential(nn.BatchNorm2d(64), nn.ConvTranspose2d(64, 1, 6, stride=2), nn.Sigmoid(), nn.Dropout(self.dropout_proba))
        # Linear layers
        self.fc11 = nn.Linear(512, latent_size)
        self.fc12 = nn.Linear(512, latent_size)
        self.fc2 = nn.Linear(latent_size, 512)
        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                #nn.init.xavier_normal_(m.weight, gain=1)
                nn.init.normal_(m.weight, mean=0, std=0.1)

    def encode(self, x):
        # Convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # Flatten and return mu and logsigma
        x = x.view(-1, 512)
        return self.fc11(x), self.fc12(x)

    def reparameterize(self, mu, log_sigma):
        if self.training:
            std = torch.exp(0.5 * log_sigma)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        z = self.fc2(z).view(-1, 512, 1, 1)
        z = self.deconv1(z)
        z = self.deconv2(z)
        z = self.deconv3(z)
        return z
