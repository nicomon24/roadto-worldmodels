'''
    Test the VAE reconstruction results on a specified dataset (not the environment as in render)
'''

'''
    This script, after loading a VAE checkpoints, shows a snek environment and
    its VAE reconstruction over time.
'''

import gym, argparse, pickle, time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import torch

from vae import VAEbyArch
from wrappers import CropCarRacing, ResizeObservation, Scolorized, NormalizeRGB, VAEObservation
from utils import side_by_side, NCHW, NHWC, imshow_bw_or_rgb
from policy import Policy

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vae', type=str, default='', help="""Path of a checkpoint to restore.""")
    parser.add_argument('--vae_old', type=str, default='', help="""VAE used for policy usage.""")
    parser.add_argument('--policy', type=str, default=None, help="""Policy checkpoint to restore.""")
    parser.add_argument('--latent_size', type=int, default=32, help="""Size of the latent vector.""")
    parser.add_argument('--sample', type=int, default=0, help="""Specify the sample to visualize.""")
    parser.add_argument('--dataset', type=str, default='', help="""Dataset file to load.""")
    parser.add_argument('--arch', type=str, default='base_car_racing', help="""Model architecture.""")
    parser.add_argument('--seed', type=int, default=42, help="""Seed used in the environment initialization.""")
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Enables CUDA training')
    args, unparsed = parser.parse_known_args()
    # Check cuda
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    # Loading the dataset
    if args.dataset:
        dataset = np.array(pickle.load(open(args.dataset, 'rb')))
        N_SAMPLES, W, H, CHANNELS = dataset.shape
        print("Dataset size:", N_SAMPLES)
        print("Channels:", CHANNELS)
        print("Image dim: (%d,%d)" % (W,H))
        dataset_torch = torch.from_numpy(NCHW(dataset)).float().to(device)
    else:
        print("Using gym environment directly.")
        env = gym.make('CarRacing-v0')
        env = CropCarRacing(env)
        env = ResizeObservation(env, (32, 32, 3))
        env = NormalizeRGB(env)
        env = Scolorized(env, weights=[0.0, 1.0, 0.0])
        env.seed(args.seed)

    # Network creation
    VAE_class = VAEbyArch(args.arch)
    vae = VAE_class(latent_size=args.latent_size).to(device)
    # Restore checkpoint
    assert args.vae, "No checkpoint provided."
    vae.load_state_dict(torch.load(args.vae))
    vae.eval()

    if args.dataset:
        # Single observation display
        mu, log_sigma, z, rebuild = vae(dataset_torch[args.sample:args.sample+1])
        rebuild = rebuild.detach().numpy()[0]
        imshow_bw_or_rgb(side_by_side(dataset[args.sample], NHWC(rebuild)))
        plt.show()
    else:
        # Check if we use a policy
        policy = None
        if args.policy and args.vae_old:
            policy_env = VAEObservation(env, args.vae_old, arch=args.arch)
            policy = Policy(policy_env)
            policy.load_state_dict(torch.load(args.policy))
            policy.eval()
            vae_old = VAE_class(latent_size=args.latent_size).to(device)
            vae_old.load_state_dict(torch.load(args.vae_old))
            vae_old.eval()
        # Animation of environment
        obs = env.reset()
        obs_torch = torch.from_numpy(NCHW([obs])).float().to(device)
        mu, log_sigma, z, rebuild = vae(obs_torch)
        rebuild = NHWC(rebuild.detach().numpy()[0])

        fig1 = plt.figure()
        if len(obs.shape) == 3 and (obs.shape[-1]==1):
            im = plt.imshow(side_by_side(obs, rebuild), cmap="Greys")
        else:
            im = plt.imshow(side_by_side(obs, rebuild))
        done = False

        # Setting animation update function
        def updatefig(*args):
            global done
            global obs
            if not done:
                if policy:
                    obs_torch = torch.from_numpy(NCHW([obs])).float().to(device)
                    mu, _ = vae.encode(obs_torch)
                    action, action_proba = policy.act(mu.detach().numpy())
                    action = action[0]
                else:
                    action = env.action_space.sample()
                    action = [action[0], 0.3, 0.0]
                obs, reward, done, info = env.step(action)
                env.render(mode='human')
            else:
                done = False
                obs = env.reset()
            obs_torch = torch.from_numpy(NCHW([obs])).float().to(device)
            mu, log_sigma, z, rebuild = vae(obs_torch)
            rebuild = NHWC(rebuild.detach().numpy()[0])
            im.set_array(side_by_side(obs, rebuild))
            time.sleep(0.01)
            return im,

        # Start animation
        ani = animation.FuncAnimation(fig1, updatefig, interval=50, blit=True)
        plt.show()
