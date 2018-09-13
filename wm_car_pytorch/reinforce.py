'''
    This script trains a policy using the REINFORCE algorithm. This is just a start.
    The environment observation are encoded using a trained VAE, for which a
    checkpoint must be specified.
'''

import argparse, gym, time, os
import numpy as np
from collections import deque
from tqdm import trange
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from tensorboardX import SummaryWriter

from policy import Policy
from sampler import Sampler
from wrappers import CropCarRacing, ResizeObservation, Scolorized, NormalizeRGB, VAEObservation

def finish_episode(trajectories, policy, args):
    H = trajectories['rewards'].shape[1] # Horizon
    N = trajectories['rewards'].shape[0] # Number of episodes
    # Compute cumulative discounted rewards
    future_disc_rewards = np.zeros((N, H))
    R = trajectories['rewards'][:,-1]
    future_disc_rewards[:, -1] = R
    for i in range(1, H):
        R = args.gamma * R + trajectories['rewards'][:, H-i-1]
        future_disc_rewards[:, H-i-1] = R
    future_disc_rewards = future_disc_rewards * trajectories['mask']
    # Use the baseline if specified
    baseline = 0
    if args.baseline:
        # Square log proba
        square_logp = (trajectories['logp'].sum(dim=1)**2)
        # Compute baseline
        baseline = ((square_logp * torch.from_numpy(future_disc_rewards[:,0])).mean(dim=0) / square_logp.mean(dim=0)).item()
    # Compute J estimate
    j_estimate = (trajectories['logp'] * torch.from_numpy(future_disc_rewards - baseline)).sum(dim=1).mean(dim=0)
    # Compute gradient
    policy.zero_grad()
    j_estimate.backward()
    for param in policy.parameters():
        param.data = param.data + args.lr * param.grad

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='REINFORCE using PyTorch')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor (default: 0.99)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--eb', type=int, default=1, help='episode batch (default: 1)')
    parser.add_argument('--episodes', type=int, default=10000, help='simulated episodes (default: 10000)')
    parser.add_argument('--policy', type=str, default=None, help="""Policy checkpoint to restore.""")
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--horizon', type=int, default=1000, help='horizon (default: 1000)')
    parser.add_argument('--render', action='store_true', help='render the environment')
    parser.add_argument('--baseline', action='store_true', help='use the baseline for the REINFORCE algorithm')
    parser.add_argument('--render_interval', type=int, default=100, help='interval between rendered epochs (default: 100)')
    parser.add_argument('--env', type=str, default='CarRacing-v0', help='environment to train on (default: CartPole-v0)')
    parser.add_argument('--vae', type=str, default=None, help='VAE checkpoint to load')
    args = parser.parse_args()
    # Initialize environment
    env = gym.make(args.env)
    env = CropCarRacing(env)
    env = ResizeObservation(env, (64, 64, 3))
    env = NormalizeRGB(env)
    env = VAEObservation(env, args.vae)
    print(env.observation_space)
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    print("Env final goal:", env.spec.reward_threshold)
    # Create the alias for the run
    alias = 'reinforce_lr=%s_eb=%s_seed=%s' % (args.lr, args.eb, args.seed)
    if args.baseline:
        alias += '_baseline'
    alias += '_%s' % (time.time())
    # Use alias for checkpoints
    checkpoint_best_filename = 'policy_weights/' + alias + '_best.torch'
    checkpoint_final_filename = 'policy_weights/' + alias + '_final.torch'
    if not os.path.exists('policy_weights/'):
        os.makedirs('policy_weights/')
    # Tensorboard writer
    writer = SummaryWriter('policy_logs/' + alias)
    # Declare policy
    policy = Policy(env)
    if args.policy:
        policy.load_state_dict(torch.load(args.policy))
        policy.eval()
    # Declare sampler
    sampler = Sampler(env, args.horizon)
    # Run episodes
    running_reward = deque(maxlen=100)
    best_reward = None
    for i_episode in trange(0, args.episodes, args.eb, desc="Episodes", unit_scale=args.eb):
        # Sample trajectories
        trajectories = sampler.sample(args.eb, policy, render=(i_episode%args.render_interval==0))
        # Update policy
        finish_episode(trajectories, policy, args)
        # Get quantities for summaries
        episode_rewards = np.sum(trajectories['rewards'], axis=1)
        mean_reward = np.mean(episode_rewards)
        episode_lens = np.sum(trajectories['mask'], axis=1)
        for sub_i in range(args.eb):
            # Summaries: mean episode reward for 100 episodes
            running_reward.append(episode_rewards[sub_i])
            writer.add_scalar('data/mean_100episode_reward', np.mean(running_reward), i_episode + sub_i)
            # Summaries: mean episode len
            writer.add_scalar('data/episode_len', episode_lens[sub_i], i_episode + sub_i)
            writer.add_scalar('data/episode_reward', episode_rewards[sub_i], i_episode + sub_i)
        # Save best model if needed
        if (best_reward is None) or (mean_reward > best_reward):
            best_reward = mean_reward
            print("Saving best model:", best_reward)
            torch.save(policy.state_dict(), checkpoint_best_filename)
        # Check if completed
        if np.mean(running_reward) > env.spec.reward_threshold:
            print("Solved, stopping. Mean reward:", np.mean(running_reward))
            break

    # Save final model
    torch.save(policy.state_dict(), checkpoint_final_filename)
    # Close env and writer
    env.close()
    writer.close()

if __name__ == '__main__':
    main()
