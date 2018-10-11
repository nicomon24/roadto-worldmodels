'''
    This script is used to render the policy in the environment.
    This requires both a trained VAE file and a trained policy file.
'''

import numpy as np
import torch
import gym, argparse
import matplotlib.pyplot as plt
from tqdm import trange

from wrappers import ResizeObservation, CropCarRacing, Scolorized, NormalizeRGB, VAEObservation
from policy import Policy

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vae', type=str, default=None, help="""VAE checkpoint to restore.""")
    parser.add_argument('--policy', type=str, default=None, help="""Policy checkpoint to restore.""")
    parser.add_argument('--latent_size', type=int, default=32, help="""Size of the latent vector.""")
    parser.add_argument('--arch', type=str, default='base_car_racing', help="""Model architecture.""")
    parser.add_argument('--seed', type=int, default=42, help="""Seed used in the environment initialization.""")
    parser.add_argument('--episodes', type=int, default=10, help="""Number of episodes to render.""")
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Enables CUDA training')
    parser.add_argument('--horizon', type=int, default=1000, help='horizon (default: 1000)')
    parser.add_argument('--env', type=str, default='CarRacing-v0', help='environment to train on (default: CartPole-v0)')
    args, unparsed = parser.parse_known_args()
    # Check cuda
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")

    assert args.vae, "You need to provide a VAE file."
    assert args.policy, "You need to provide a policy file."
    env = gym.make(args.env)
    env = CropCarRacing(env)
    env = ResizeObservation(env, (32, 32, 3))
    env = Scolorized(env, weights=[0.0, 1.0, 0.0])
    env = NormalizeRGB(env)
    env = VAEObservation(env, args.vae, arch=args.arch)

    policy = Policy(env)
    policy.load_state_dict(torch.load(args.policy))
    policy.eval()

    env.seed(args.seed)

    for i in trange(args.episodes):
        obs = env.reset()
        done = False
        i = 0
        rtotal = 0
        while not done and i < args.horizon:
            action, action_proba = policy.act(obs)
            obs, reward, done, info = env.step(action)
            env.render()
            rtotal += reward
            i += 1
        print(rtotal)
    env.close()
