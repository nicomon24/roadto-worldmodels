'''
    Training of a VAE on the MNIST dataset.
    Can be easily run in remote server.
    Uses refined regularization loss.
'''

import argparse, os
import tensorflow as tf
import numpy as np
from tqdm import trange

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=32,
  help="""\
  Size of the image batches.
""")
parser.add_argument('--latent_size', type=int, default=64,
  help="""\
  Size of the latent vector.
""")
parser.add_argument('--dropout', type=float, default=0.6,
  help="""\
  Dropout used during training.
""")
parser.add_argument('--epochs', type=str, default='1000',
  help="""\
  Comma separated epochs.
""")
parser.add_argument('--learning_rates', type=str, default='1e-05',
  help="""\
  Comma separated learning rates. (Must be of the same size of epochs)
""")
parser.add_argument('--log_dir', type=str, default='logs',
  help="""\
  Directory in which logs for tensorboard are saved.
""")
parser.add_argument('--alias', type=str, default='base',
  help="""\
  Alias of the model.
""")

FLAGS, unparsed = parser.parse_known_args()
LEARNING_RATES = list(map(float, FLAGS.learning_rates.split(',')))
EPOCHS = list(map(int, FLAGS.epochs.split(',')))
if len(LEARNING_RATES) != len(EPOCHS):
    raise(Exception('Epochs and Learning rates must be of the same size!'))

# Load the dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

print("Network creation...")

tf.reset_default_graph()

# Input image
X = tf.placeholder(dtype=tf.float32, shape=(None, 784))
X4D = tf.reshape(X, (-1, 28, 28, 1))
# Dropout keep proba
keep_prob = tf.placeholder(dtype=tf.float32, shape=(), name='keep_prob')
learning_rate = tf.placeholder(dtype=tf.float32, shape=(), name='learning_rate')
lr_summary = tf.summary.scalar('learning_rate', learning_rate)
batch_size = tf.placeholder(dtype=tf.int32, shape=(), name='batch_size')

# ENCODER GRAPH
with tf.variable_scope("encoder", reuse=None):
    conv1 = tf.layers.conv2d(X4D, filters=32, kernel_size=5, strides=1, padding='same', activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    conv2 = tf.layers.conv2d(pool1, filters=64, kernel_size=5, strides=1, padding='same', activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    flat = tf.layers.flatten(pool2)
    drop1 = tf.nn.dropout(flat, keep_prob)
    dense1 = tf.layers.dense(drop1, 1024)
    latent_means = tf.layers.dense(dense1, units=FLAGS.latent_size)
    latent_log_sigma = tf.layers.dense(flat, units=FLAGS.latent_size)
    latent_noise = tf.random_normal(shape=(batch_size, FLAGS.latent_size))
    latent_vector = latent_means + tf.multiply(tf.exp(latent_log_sigma / 2), latent_noise) # /2 because latent_log_sigma represent log(sigma^2)

# DECODER GRAPH
with tf.variable_scope("decoder", reuse=None):
    deflat = tf.layers.dense(latent_vector, units=drop1.shape[1])
    deflat_drop = tf.nn.dropout(deflat, keep_prob)
    dedense1 = tf.layers.dense(deflat_drop, units=flat.shape[1])
    deflat4d = tf.reshape(dedense1, shape=(-1, pool2.shape[1], pool2.shape[2], pool2.shape[3]))
    deconv1 = tf.layers.conv2d_transpose(deflat4d, filters=32, kernel_size=5, strides=2, padding='same', activation=tf.nn.relu)
    deconv2 = tf.layers.conv2d_transpose(deconv1, filters=1, kernel_size=5, strides=2, padding='same', activation=tf.nn.relu)
    rebuild = tf.reshape(deconv2, shape=(-1, 784))

# Losses
reconstruction_loss = tf.reduce_sum(tf.squared_difference(rebuild, X))
tf_mrl = tf.placeholder(tf.float32, ())
tf_mrl_summary = tf.summary.scalar('mean_reconstruction_loss', tf_mrl)

#reg_loss = tf.reduce_sum(-tf.log(tf.abs(latent_std)) + 0.5 * (tf.square(latent_std) + tf.square(latent_means) - 1))
reg_loss = 0.5 * tf.reduce_sum(tf.exp(latent_log_sigma) + tf.square(latent_means) - 1 + latent_log_sigma)
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

# Summaries
merged_summaries = tf.summary.merge([tf_mrl_summary, tf_mnl_summary, tf_mcl_summary, lr_summary])
train_writer = tf.summary.FileWriter(os.path.join(FLAGS.log_dir, FLAGS.alias + '_train'))

# Training
print("Start training...")

lr_index = 0
for epoch in trange(sum(EPOCHS)):
    if epoch > sum(EPOCHS[:lr_index+1]):
        lr_index += 1
    lr = LEARNING_RATES[lr_index]
    # Compute batch
    closses, rlosses, nlosses = [], [], []
    for bindex in range(0, mnist.train.num_examples, FLAGS.batch_size):
        batch, _ = mnist.train.next_batch(FLAGS.batch_size)
        _rloss, _nloss, _closs, _ = sess.run([reconstruction_loss, reg_loss, complete_loss, optimizer], feed_dict = {X: batch, keep_prob:FLAGS.dropout, learning_rate:lr, batch_size:len(batch)})
        closses.append(_closs)
        rlosses.append(_rloss)
        nlosses.append(_nloss)
    summary = sess.run(merged_summaries, feed_dict={
        tf_mrl: np.mean(rlosses),
        tf_mnl: np.mean(nlosses),
        tf_mcl: np.mean(closses),
        learning_rate: lr
    })
    train_writer.add_summary(summary, epoch)

print("Bye bye")
