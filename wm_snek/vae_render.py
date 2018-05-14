'''
    This script, after loading a VAE checkpoints, shows a snek environment and
    its VAE reconstruction over time.
'''

import gym, sneks, time, argparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tensorflow as tf
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, default='',
  help="""\
  Path of a checkpoint to restore.
""")
parser.add_argument('--latent_size', type=int, default=64,
  help="""\
  Size of the latent vector.
""")
FLAGS, unparsed = parser.parse_known_args()

tf.reset_default_graph()

# Input image
X = tf.placeholder(dtype=tf.float32, shape=(None, 80, 80, 3))
# Dropout keep proba
keep_prob = tf.placeholder(dtype=tf.float32, shape=(), name='keep_prob')
learning_rate = tf.placeholder(dtype=tf.float32, shape=(), name='learning_rate')
lr_summary = tf.summary.scalar('learning_rate', learning_rate)
batch_size = tf.placeholder(dtype=tf.int32, shape=(), name='batch_size')

# ENCODER GRAPH
with tf.variable_scope("encoder", reuse=None):
    conv1 = tf.layers.conv2d(X, filters=64, kernel_size=4, strides=2, padding='same', activation=tf.nn.relu)
    drop1 = tf.nn.dropout(conv1, keep_prob)
    conv2 = tf.layers.conv2d(drop1, filters=64, kernel_size=4, strides=2, padding='same', activation=tf.nn.relu)
    drop2 = tf.nn.dropout(conv2, keep_prob)
    conv3 = tf.layers.conv2d(drop2, filters=64, kernel_size=4, strides=1, padding='same', activation=tf.nn.relu)
    drop3 = tf.nn.dropout(conv3, keep_prob)
    flat = tf.layers.flatten(drop3)
    latent_means = tf.layers.dense(flat, units=FLAGS.latent_size)
    latent_std = tf.layers.dense(flat, units=FLAGS.latent_size)
    latent_noise = tf.random_normal(shape=(batch_size, FLAGS.latent_size))
    latent_vector = latent_means + tf.multiply(latent_std, latent_noise)

# DECODER GRAPH
with tf.variable_scope("decoder", reuse=None):
    deflat = tf.layers.dense(latent_vector, units=flat.shape[1])
    deflat4d = tf.reshape(deflat, shape=(-1, drop2.shape[1], drop2.shape[2], drop2.shape[3]))
    deconv1 = tf.layers.conv2d_transpose(deflat4d, filters=64, kernel_size=4, strides=1, padding='same', activation=tf.nn.relu)
    dedrop1 = tf.nn.dropout(deconv1, keep_prob)
    deconv2 = tf.layers.conv2d_transpose(dedrop1, filters=64, kernel_size=4, strides=2, padding='same', activation=tf.nn.relu)
    dedrop2 = tf.nn.dropout(deconv2, keep_prob)
    deconv3 = tf.layers.conv2d_transpose(dedrop2, filters=3, kernel_size=4, strides=2, padding='same', activation=tf.nn.relu)
    dedrop3 = tf.nn.dropout(deconv3, keep_prob)
    rebuild = tf.reshape(dedrop3, shape=(-1, 80, 80, 3))

# Losses
reconstruction_loss = tf.reduce_sum(tf.squared_difference(rebuild, X))
tf_mrl = tf.placeholder(tf.float32, ())
tf_mrl_summary = tf.summary.scalar('mean_reconstruction_loss', tf_mrl)

reg_loss = tf.reduce_sum(-tf.log(tf.abs(latent_std)) + 0.5 * (tf.square(latent_std) + tf.square(latent_means) - 1))
tf_mnl = tf.placeholder(tf.float32, ())
tf_mnl_summary = tf.summary.scalar('mean_normalization_loss', tf_mnl)

complete_loss = reconstruction_loss + reg_loss
tf_mcl = tf.placeholder(tf.float32, ())
tf_mcl_summary = tf.summary.scalar('mean_complete_loss', tf_mcl)

# Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(complete_loss)

# Session and init
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Restore checkpoint
if FLAGS.checkpoint:
    _saver = tf.train.Saver(tf.global_variables())
    _saver.restore(sess, FLAGS.checkpoint)
else:
    raise(Exception('No checkpoint provided.'))

# Env init
env = gym.make('snek-rgb-zoom5-v1')
env.seed(42)

def add_reco(obs):
    # Compute reconstruction
    reco = sess.run(rebuild, feed_dict={X: [obs], keep_prob:1.0, batch_size:1})[0]
    return np.concatenate([obs/255, reco], axis=1)

obs = env.reset()

fig1 = plt.figure()
im = plt.imshow(add_reco(obs))
done = False

def updatefig(*args):
    global done
    if not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
    else:
        done = False
        obs = env.reset()
    im.set_array(add_reco(obs))
    time.sleep(0.1)
    return im,

ani = animation.FuncAnimation(fig1, updatefig, interval=50, blit=True)
plt.show()
