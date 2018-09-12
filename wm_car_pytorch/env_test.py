import numpy as np
import tensorflow as tf
import gym
import matplotlib.pyplot as plt
from tqdm import trange

from wrappers import ResizeObservation, CropCarRacing, Scolorized, NormalizeRGB

'''
    Car Racing action space:
    Box(3) floats
    action[0]: steer, -1 to 1
    action[1]: gas. 0 to 1
    action[2]: brake, 0 to 1
'''

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
