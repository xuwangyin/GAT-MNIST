import argparse
import json
import sys
from datetime import datetime

import os

import cifar10_input
from models import Classifier
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', choices=['cifar10', 'mnist'], default='cifar10')
parser.add_argument('--test_model', type=str)

args = parser.parse_args()


with open('configs/cifar10.json') as config_file:
  config = json.load(config_file)

# seeding randomness
tf.set_random_seed(config['tf_random_seed'])
np.random.seed(config['np_random_seed'])

# Setting up training parameters
max_num_training_steps = config['max_num_training_steps']
num_output_steps = config['num_output_steps']
num_summary_steps = config['num_summary_steps']
num_checkpoint_steps = config['num_checkpoint_steps']
step_size_schedule = config['step_size_schedule']
weight_decay = config['weight_decay']
data_path = config['data_path']
momentum = config['momentum']
batch_size = config['training_batch_size']

# Setting up the data and the model
raw_cifar = cifar10_input.CIFAR10Data(data_path)
global_step = tf.contrib.framework.get_or_create_global_step()

if args.test_model is not None:
  classifier = Classifier(var_scope='classifier', dataset='CIFAR10', mode='eval')
  vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='classifier')
  saver = tf.train.Saver(var_list=vars)
  with tf.Session() as sess:
    cifar = cifar10_input.AugmentedCIFAR10Data(raw_cifar, sess, classifier)
    test_cifar = cifar10_input.CIFAR10Data(data_path)

    sess.run(tf.global_variables_initializer())

    saver.restore(sess, args.test_model)
    print('loaded {}'.format(args.test_model))

    num_corrects = 0.0
    num_test = test_cifar.eval_data.xs.shape[0]
    for i in range(0, num_test, batch_size):
      x_batch = test_cifar.eval_data.xs[i: i + batch_size]
      y_batch = test_cifar.eval_data.ys[i: i + batch_size]
      x_batch = np.reshape(x_batch, [x_batch.shape[0], -1])
      test_feed_dict = {classifier.x_input: x_batch, classifier.y_input: y_batch}
      acc, num_corrects_ = sess.run([classifier.accuracy, classifier.num_correct], feed_dict=test_feed_dict)
      num_corrects += num_corrects_
      print('test {} acc {:.4f}'.format(i, acc))

    print('test accuracy {:.4f}'.format(num_corrects / num_test))
    sys.exit(0)

classifier = Classifier(var_scope='classifier', dataset='CIFAR10', mode='train')
#
# use GLOBAL_VARIABLES to collect batchnorm moving mean and variance
vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='classifier')
saver = tf.train.Saver(var_list=vars, max_to_keep=3)
tf.summary.scalar('train accuracy', classifier.accuracy)
tf.summary.scalar('train loss', classifier.xent)
merged_summaries = tf.summary.merge_all()

# Setting up the optimizer
boundaries = [int(sss[0]) for sss in step_size_schedule]
boundaries = boundaries[1:]
values = [sss[1] for sss in step_size_schedule]
learning_rate = tf.train.piecewise_constant(tf.cast(global_step, tf.int32),
                                            boundaries, values)
total_loss = classifier.xent + weight_decay * classifier.weight_decay_loss
train_step = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(
    total_loss,
    global_step=global_step)

with tf.Session() as sess:
  cifar = cifar10_input.AugmentedCIFAR10Data(raw_cifar, sess, classifier)
  test_cifar = cifar10_input.CIFAR10Data(data_path)

  summary_writer = tf.summary.FileWriter('logs/classifier_{}/'.format(args.dataset), sess.graph)
  sess.run(tf.global_variables_initializer())

  # Main training loop
  for step in range(max_num_training_steps):
    x_batch, y_batch = cifar.train_data.get_next_batch(batch_size, multiple_passes=True)
    x_batch = np.reshape(x_batch, [x_batch.shape[0], -1])
    feed_dict = {classifier.x_input: x_batch, classifier.y_input: y_batch}

    if step % num_output_steps == 0:
      acc, loss = sess.run([classifier.accuracy, classifier.xent], feed_dict=feed_dict)
      print('Step {}: ({})'.format(step, datetime.now()), end=' ')
      print('training accuracy {:.4f}, loss {:.4f}'.format(acc * 100, loss), end='| ')
      test_x_batch = test_cifar.eval_data.xs[:2000, :].reshape([2000, -1])
      test_y_batch = test_cifar.eval_data.ys[:2000]
      test_feed_dict = {classifier.x_input: test_x_batch, classifier.y_input: test_y_batch}
      acc, loss = sess.run([classifier.accuracy, classifier.xent], feed_dict=test_feed_dict)
      print('test accuracy {:.4f}, loss {:.4f}'.format(acc * 100, loss))

    sess.run(train_step, feed_dict=feed_dict)

    # Tensorboard summaries
    if step % num_summary_steps == 0:
      summary = sess.run(merged_summaries, feed_dict=feed_dict)
      summary_writer.add_summary(summary, global_step.eval(sess))

    # Write a checkpoint
    if step % num_checkpoint_steps == 0:
      model_name = os.path.join('checkpoints/{}/classifier'.format(args.dataset))
      saver.save(sess, model_name, global_step=global_step)
      print('saved {}'.format(model_name))

