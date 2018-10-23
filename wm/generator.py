'''
    Generate episodes to train the VAE and the model on, using a
    specified model or a random one.
    Parallelized for multi-core execution.
'''

from multiprocessing import Pool

import gym, time, argparse, cv2, uuid, os
from gym import wrappers, logger
import numpy as np
from tqdm import trange
import torch

from wrappers import CropCarRacing, ResizeObservation, NormalizeRGB
from utils import NCHW, NHWC, select_n_workers

def generate_episode(make_env, make_model, seed, horizon):
    # Make env
    env = make_env(seed)
    model = make_model()
    # Aggregators
    episode_observations = []
    episode_actions = []
    print("Here i am")
    obs = env.reset()
    print("Rocking like a hurricane")
    # Iteration
    for timestep in trange(horizon):
        # Registering observation
        episode_observations.append(obs)
        # Getting action and step
        a = env.action_space.sample()
        obs, reward, done, info = env.step(a)
        # Registering action
        episode_actions.append(a)
        # Check if done
        if done:
            break
    # Numpify
    episode_observations = np.array(episode_observations)
    episode_actions = np.array(episode_actions)
    return episode_observations, episode_actions

class ParallelGenerator(object):

    def __init__(self, make_env, make_model, horizon, n_workers=-1, seed=0):
        self.make_env = make_env
        self.make_model = make_model
        self.seed = seed
        self.horizon = horizon
        # Get the number of workers
        self.n_workers = select_n_workers(n_workers)
        print('Using', self.n_workers, 'workers.')
        # Define the pool
        self.pool = Pool(self.n_workers)

    def generate(self, size):
        # Apply async
        results = [self.pool.apply_async(generate_episode, args=(self.make_env, self.make_model, self.seed+i, self.horizon)) for i in range(size)]
        observations, actions = zip(*[r.get() for r in results])
        print("Done?")
        return np.array(observations, actions)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=10, help="""Number of episodes to generate.""")
    parser.add_argument('--seed', type=int, default=42, help="""Random seed to use.""")
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Enables CUDA training')
    parser.add_argument('--model', type=str, default=None, help="""Model checkpoint to load (optional).""")
    parser.add_argument('--horizon', type=int, default=1000, help='Horizon (default: 1000)')
    parser.add_argument('--dir', type=str, default='data', help="""Directory in which to save to generated files.""")
    parser.add_argument('--n_workers', type=int, default=-1, help='Number of workers (default: -1, use all cores)')
    args, unparsed = parser.parse_known_args()
    # Check cuda
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")

    # Env creation function
    def make_env(seed):
        env = gym.make('CarRacing-v0')
        env = CropCarRacing(env)
        env = ResizeObservation(env, (64, 64, 3))
        env = NormalizeRGB(env)
        print("what", seed)
        env.seed(seed)
        np.random.seed(seed)
        return env

    def make_model():
        return None

    # Create parallel generator
    generator = ParallelGenerator(make_env, make_model, horizon=args.horizon, n_workers=args.n_workers, seed=args.seed)
    observations, actions = generator.generate(args.size)

    print(observations.shape, actions.shape)

    '''
    # Numpify global
    observations = np.array(observations)
    actions = np.array(actions)
    # Check that dir exists
    if not os.path.exists(args.dir):
        os.makedirs(args.dir)
    # Generate random filename and save
    filename = args.dir + '/' + str(uuid.uuid4()) + '.npz'
    np.savez_compressed(filename, obs=observations, action=actions)

    # Close the env, we are done
    env.close()
    '''
