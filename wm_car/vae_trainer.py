'''
    This script is used to train the VAE on a dataset generated from car-race game.
    To provide support for remote training, it provides logs for tensorboard and saves checkpoints.
'''

import numpy as np
import tensorflow as tf
import gym, argparse, os, pickle
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
parser.add_argument('--save_interval', type=int, default=10,
  help="""\
  Size of the latent vector.
""")
parser.add_argument('--dropout', type=float, default=0.9,
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
parser.add_argument('--save_dir', type=str, default='checkpoints',
  help="""\
  Directory in which checkpoints are saved.
""")
parser.add_argument('--log_dir', type=str, default='logs',
  help="""\
  Directory in which logs for tensorboard are saved.
""")
parser.add_argument('--checkpoint', type=str, default='',
  help="""\
  Path of a checkpoint to restore.
""")
parser.add_argument('--start_epoch', type=int, default=0,
  help="""\
  Start epoch when loading a checkpoint.
""")
parser.add_argument('--alias', type=str, default='base',
  help="""\
  Alias of the model.
""")
parser.add_argument('--dataset', type=str, default='',
  help="""\
  Dataset file to load.
""")

FLAGS, unparsed = parser.parse_known_args()
LEARNING_RATES = list(map(float, FLAGS.learning_rates.split(',')))
EPOCHS = list(map(int, FLAGS.epochs.split(',')))
if len(LEARNING_RATES) != len(EPOCHS):
    raise(Exception('Epochs and Learning rates must be of the same size!'))

# Loading the dataset
dataset = pickle.load(open(FLAGS.dataset, 'rb'))

print("Network creation...")

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
    conv1 = tf.layers.conv2d(X, filters=32, kernel_size=4, strides=2, padding='same', activation=tf.nn.relu)
    drop1 = tf.nn.dropout(conv1, keep_prob)
    conv2 = tf.layers.conv2d(drop1, filters=64, kernel_size=4, strides=2, padding='same', activation=tf.nn.relu)
    drop2 = tf.nn.dropout(conv2, keep_prob)
    conv3 = tf.layers.conv2d(drop2, filters=128, kernel_size=4, strides=2, padding='same', activation=tf.nn.relu)
    drop3 = tf.nn.dropout(conv3, keep_prob)
    conv4 = tf.layers.conv2d(drop3, filters=256, kernel_size=4, strides=2, padding='same', activation=tf.nn.relu)
    drop4 = tf.nn.dropout(conv4, keep_prob)
    flat = tf.layers.flatten(drop4)
    latent_means = tf.layers.dense(flat, units=FLAGS.latent_size)
    latent_std = tf.layers.dense(flat, units=FLAGS.latent_size)
    latent_noise = tf.random_normal(shape=(batch_size, FLAGS.latent_size))
    latent_vector = latent_means + tf.multiply(latent_std, latent_noise)

# DECODER GRAPH
with tf.variable_scope("decoder", reuse=None):
    deflat = tf.layers.dense(latent_vector, units=flat.shape[1])
    deflat4d = tf.reshape(deflat, shape=(-1, drop4.shape[1], drop4.shape[2], drop4.shape[3]))
    deconv1 = tf.layers.conv2d_transpose(deflat4d, filters=128, kernel_size=4, strides=2, padding='same', activation=tf.nn.relu)
    dedrop1 = tf.nn.dropout(deconv1, keep_prob)
    deconv2 = tf.layers.conv2d_transpose(dedrop1, filters=64, kernel_size=4, strides=2, padding='same', activation=tf.nn.relu)
    dedrop2 = tf.nn.dropout(deconv2, keep_prob)
    deconv3 = tf.layers.conv2d_transpose(dedrop2, filters=32, kernel_size=4, strides=2, padding='same', activation=tf.nn.relu)
    dedrop3 = tf.nn.dropout(deconv3, keep_prob)
    deconv4 = tf.layers.conv2d_transpose(dedrop3, filters=3, kernel_size=4, strides=2, padding='same', activation=tf.nn.relu)
    dedrop4 = tf.nn.dropout(deconv4, keep_prob)
    rebuild = tf.reshape(dedrop4, shape=(-1, 80, 80, 3))

# Losses
reconstruction_loss = tf.reduce_sum(tf.squared_difference(rebuild, X))
tf_mrl = tf.placeholder(tf.float32, ())
tf_mrl_summary = tf.summary.scalar('mean_reconstruction_loss', tf_mrl)

#reg_loss = tf.reduce_sum(-tf.log(tf.abs(latent_std)) + 0.5 * (tf.square(latent_std) + tf.square(latent_means) - 1))
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

# Saver and summaries
saver = tf.train.Saver(tf.global_variables(), max_to_keep=40)
merged_summaries = tf.summary.merge([tf_mrl_summary, tf_mnl_summary, tf_mcl_summary, lr_summary])
train_writer = tf.summary.FileWriter(os.path.join(FLAGS.log_dir, FLAGS.alias + '_train'))

# Check if we need to load a checkpoint
if FLAGS.checkpoint:
    _saver = tf.train.Saver(tf.global_variables())
    _saver.restore(sess, FLAGS.checkpoint)

# Training
print("Start training...")

lr_index = 0
for epoch in trange(FLAGS.start_epoch, sum(EPOCHS)):
    if epoch > sum(EPOCHS[:lr_index+1]):
        lr_index += 1
    lr = LEARNING_RATES[lr_index]
    # Compute batch
    closses, rlosses, nlosses = [], [], []
    for bindex in range(480, len(dataset), FLAGS.batch_size):
        batch = dataset[bindex:bindex+FLAGS.batch_size]
        for b in batch:
            print(b.shape)
        print(len(batch), batch[0].shape, bindex)
        _rloss, _nloss, _closs, _ = sess.run([reconstruction_loss, reg_loss, complete_loss, optimizer], feed_dict = {X: batch, keep_prob:0.0, learning_rate:0.0, batch_size:len(batch)})
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
    # Check if we need to save
    if epoch % FLAGS.save_interval == 0:
        checkpoint_path = os.path.join(FLAGS.save_dir, FLAGS.alias + '-' + str(epoch) + '.ckpt')
        tf.logging.info('Saving to "%s-%d"', checkpoint_path, epoch)
        saver.save(sess, checkpoint_path, write_meta_graph=True)

# Save the last model
checkpoint_path = os.path.join(FLAGS.save_dir, FLAGS.alias + '-' + str(sum(EPOCHS)) + '.ckpt')
tf.logging.info('Saving to "%s-%d"', checkpoint_path, epoch)
saver.save(sess, checkpoint_path, write_meta_graph=True)

print("Bye bye")
