'''
    Generate episodes to train the VAE and the model on, using a
    specified model or a random one.
    Parallelized for multi-core execution.
'''

from multiprocessing import Process, Queue, Event

import gym, time, argparse, cv2, uuid, os
from gym import wrappers, logger
import numpy as np
from tqdm import trange
import torch

from wrappers import CropCarRacing, ResizeObservation, NormalizeRGB
from utils import NCHW, NHWC, select_n_workers



class Worker(Process):

    def __init__(self, input, output, event, make_env, make_model, horizon, seed):
        super(Worker, self).__init__()
        self.input = input
        self.output = output
        self.event = event
        self.make_env = make_env
        self.make_model = make_model
        self.horizon = horizon
        self.seed = seed

    def generate_episode(self):
        # Aggregators
        episode_observations = []
        episode_actions = []
        obs = self.env.reset()
        # Iteration
        for timestep in range(self.horizon):
            # Registering observation
            episode_observations.append(obs)
            # Getting action and step
            a = self.env.action_space.sample()
            obs, reward, done, info = self.env.step(a)
            # Registering action
            episode_actions.append(a)
            # Check if done
            if done:
                break
        # Numpify
        episode_observations = np.array(episode_observations)
        episode_actions = np.array(episode_actions)
        return episode_observations, episode_actions

    def run(self):
        # Make env and model
        self.env = self.make_env(self.seed)
        self.model = self.make_model()
        print("Worker %s - Running with seed %s" % (os.getpid(), self.seed))
        # Start continuous loop
        while True:
            self.event.wait()
            self.event.clear()
            command, worker_id = self.input.get()
            if command == 'generate':
                episode_observations, episode_actions = self.generate_episode()
                self.output.put((worker_id, episode_observations, episode_actions))
            elif command == 'exit':
                self.env.close()
                break

class ParallelGenerator(object):

    def __init__(self, make_env, make_model, horizon, n_workers=-1, seed=0):
        self.make_env = make_env
        self.make_model = make_model
        self.seed = seed
        self.horizon = horizon
        # Get the number of workers
        self.n_workers = select_n_workers(n_workers)
        print('Using', self.n_workers, 'workers.')
        # Define queues, events and the pool of workers
        self.output_queue = Queue()
        self.input_queues = [Queue() for _ in range(self.n_workers)]
        self.events = [Event() for _ in range(self.n_workers)]
        self.workers = [Worker(self.input_queues[i], self.output_queue, self.events[i],
                            self.make_env, self.make_model, self.horizon, self.seed+i) for i in range(self.n_workers)]
        # Start the workers
        for w in self.workers:
            w.start()

    def generate(self, size):
        total_steps = 0
        for i in range(self.n_workers):
            self.input_queues[i].put('generate')
            self.events[i].set()

        observations = []
        actions = []
        while total_steps < size:
            id, episode_observations, episode_actions = self.output_queue.get()
            observations.append(episode_observations)
            actions.append(episode_actions)
            total_steps += 1
            if total_steps < (size - self.n_workers+1):
                self.input_queues[id].put(('generate', id))
                self.events[id].set()

        return np.array(observations), np.array(actions)

    def close(self):
        for i in range(self.n_workers):
            self.input_queues[i].put('exit')
        for e in self.events:
            e.set()
        for w in self.workers:
            w.join()


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
        env = NormalizeRGB(env)
        env = CropCarRacing(env)
        env = ResizeObservation(env, (64, 64, 3))
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

    generator.close()

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
