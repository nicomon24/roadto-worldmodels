'''
    Dataset generator script.
    This script runs the car racing environment with a random policy.
    TODO: add non-random policy

    It produces a pickle file containing a dataset of resized frames, already
    shuffled.
'''

import gym, time, argparse, cv2
from gym import wrappers, logger
import numpy as np
from tqdm import trange
import pickle
import torch

from wrappers import CropCarRacing, ResizeObservation, NormalizeRGB, VAEObservation, Scolorized
from policy import Policy
from vae import VAEbyArch
from utils import NCHW, NHWC

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=1000, help="""Number of frames to generate.""")
    parser.add_argument('--output', type=str, default='car_dataset.pkl', help="""Output filename.""")
    parser.add_argument('--seed', type=int, default=42, help="""Output filename.""")
    parser.add_argument('--policy', type=str, default=None, help="""Policy checkpoint to restore.""")
    parser.add_argument('--vae', type=str, default=None, help="""VAE checkpoint to restore.""")
    parser.add_argument('--latent_size', type=int, default=32, help="""Size of the latent vector.""")
    parser.add_argument('--arch', type=str, default='base_car_racing', help="""Model architecture.""")
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Enables CUDA training')
    parser.add_argument('--horizon', type=int, default=1000, help='horizon (default: 1000)')
    args, unparsed = parser.parse_known_args()
    # Check cuda
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    # Create env
    env = gym.make('CarRacing-v0')
    env = CropCarRacing(env)
    env = ResizeObservation(env, (32, 32, 3))
    env = Scolorized(env, weights=[0.0, 1.0, 0.0])
    env = NormalizeRGB(env)
    env.seed(args.seed)
    np.random.seed(args.seed)
    # Init policy and VAE
    if args.policy:
        assert args.vae, "You need to provide also a VAE file for the policy."
        policy_env = VAEObservation(env, args.vae)
        policy = Policy(policy_env)
        policy.load_state_dict(torch.load(args.policy))
        policy.eval()
        VAE_class = VAEbyArch(args.arch)
        vae = VAE_class(latent_size=args.latent_size)
        vae.load_state_dict(torch.load(args.vae))
        vae.eval()
    # Data generation
    dataset = []
    obs = env.reset()
    step = 0
    for i in trange(args.size):
        if args.policy:
            obs_torch = torch.from_numpy(NCHW([obs])).float().to(device)
            mu, _ = vae.encode(obs_torch)
            action, action_proba = policy.act(mu.detach().numpy())
            action = action[0]
        else:
            action = env.action_space.sample()
            action = [action[0], 0.3, 0.0]
        obs, reward, done, info = env.step(action)
        step += 1
        #env.render()
        dataset.append(obs)
        if done or step >= args.horizon:
            env.seed(args.seed + i)
            obs = env.reset()
            step = 0

    env.close()
    np.random.shuffle(dataset)

    print("Generated dataset.")

    # Save to file
    with open(args.output, 'wb') as outfile:
        pickle.dump(dataset, outfile)

    print("Saved dataset to", args.output)
