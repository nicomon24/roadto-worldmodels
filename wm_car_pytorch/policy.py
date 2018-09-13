'''
    Specification of the policy. This policy is specifically created for
    the CarRacing environment, so the action space is taken as granted
    Action space has 3 float values: steer[-1,1], gas[0,1], brake[0,1].

    Future TODO:
    - environment independence check
    - beta distribution
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, MultivariateNormal

from gym.spaces import Box

class Policy(nn.Module):

    def __init__(self, env, fixed_variance=True):
        super(Policy, self).__init__()
        self.env = env
        self.fixed_variance = fixed_variance
        # Check the type of action space
        self.discrete = not (type(env.action_space) == Box)
        if self.discrete:
            # Not a box, discrete actions
            self.last_layer_dim = self.env.action_space.n
        else:
            # Box, continuous actions
            self.last_layer_dim = self.env.action_space.shape[0]
        # Declare layers
        self.linear1 = nn.Linear(np.prod(self.env.observation_space.shape), 16)
        self.linear2 = nn.Linear(16, self.last_layer_dim)

    def act(self, obs):
        VARIANCE = 0.25
        obs = torch.from_numpy(obs).float().unsqueeze(0) # Unsqueeze to flat observation
        out = self(obs) # Forward pass
        if self.discrete:
            a_dist = Categorical(out)
            a = a_dist.sample()
            return a.item(), a_dist.log_prob(a)
        else:
            a_dist = MultivariateNormal(out, covariance_matrix=VARIANCE*torch.eye(self.env.action_space.shape[0]))
            a = a_dist.sample()
            return np.clip(a[0].numpy(), self.env.action_space.low, self.env.action_space.high), a_dist.log_prob(a)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        if self.discrete:
            return F.softmax(x, dim=1)
        else:
            return x
