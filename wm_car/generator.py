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

parser = argparse.ArgumentParser()
parser.add_argument('--size', type=int, default=1000,
  help="""\
  Number of frames to generate.
""")
parser.add_argument('--output', type=str, default='car_dataset.pkl',
  help="""\
  Output filename.
""")
parser.add_argument('--seed', type=int, default=42,
  help="""\
  Output filename.
""")
FLAGS, unparsed = parser.parse_known_args()

# We need a function to crop from 96x96 down to 84x96 and resize from 84x96 to 64x64
def resize(img):
    img = img[:84,:,:]
    img = cv2.resize(img, dsize=(64, 64))
    return img

def normalize(img):
    return img / 255

def scolorize(img):
    return np.reshape(np.mean(img, axis=2), (img.shape[0], img.shape[1], 1))

env = gym.make('CarRacing-v0')
env.seed(FLAGS.seed)
np.random.seed(FLAGS.seed)

dataset = []
BUFFER = 8
#env = wrappers.Monitor(env, directory='tmp', force=True)

obs = env.reset()
dataset.append(obs)
for i in trange(FLAGS.size + BUFFER):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()
    obs = resize(obs)
    obs = normalize(obs)
    #obs = scolorize(obs)
    dataset.append(obs)

dataset = dataset[BUFFER:]

env.close()

np.random.shuffle(dataset)
print("Generated dataset.")

# Save to file
with open(FLAGS.output, 'wb') as outfile:
    pickle.dump(dataset, outfile)

print("Saved dataset to", FLAGS.output)
