import numpy as np
import tensorflow as tf
import gym
import matplotlib.pyplot as plt
from tqdm import trange

from wrappers import ResizeObservation, CropCarRacing, Scolorized, NormalizeRGB

env = gym.make('CarRacing-v0')
env = CropCarRacing(env)
env = ResizeObservation(env, (64, 64, 3))
#env = Scolorized(env)
env = NormalizeRGB(env)

dataset = []
env.seed(42)
obs = env.reset()
done = False

print(env.observation_space)
print(env.action_space)

for i in trange(50):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    env.render()
    dataset.append(obs)
env.close()

plt.imshow(dataset[-1])
plt.show()
