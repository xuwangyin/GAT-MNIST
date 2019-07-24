"""
Implementation of attack methods. Running this file as a program will
apply the attack to the model specified by the config file and store
the examples in an .npy file.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import sys
from tensorflow.keras.datasets import mnist


class LinfPGDAttack:
  def __init__(self, model, epsilon, k, a, random_start, loss_func):
    """Attack parameter initialization. The attack performs k steps of
       size a, while always staying within epsilon from the initial
       point."""
    self.model = model
    self.epsilon = epsilon
    self.k = k
    self.a = a
    self.rand = random_start

    if loss_func == 'xent':
      loss = model.xent
    elif loss_func == 'cw':
      label_mask = tf.one_hot(model.y_input,
                              10,
                              on_value=1.0,
                              off_value=0.0,
                              dtype=tf.float32)
      correct_logit = tf.reduce_sum(label_mask * model.logits, axis=1)
      wrong_logit = tf.reduce_max((1-label_mask) * model.logits, axis=1)
      loss = -tf.nn.relu(correct_logit - wrong_logit)
    else:
      print('Unknown loss function. Defaulting to cross-entropy')
      loss = model.xent

    self.grad = tf.gradients(loss, model.x_input)[0]

  def perturb(self, x_nat, y, sess):
    """Given a set of examples (x_nat, y), returns a set of adversarial
       examples within epsilon of x_nat in l_infinity norm."""
    if self.rand:
      x = x_nat + np.random.uniform(-self.epsilon, self.epsilon, x_nat.shape)
    else:
      x = np.copy(x_nat)

    for i in range(self.k):
      grad = sess.run(self.grad, feed_dict={self.model.x_input: x,
                                            self.model.y_input: y})

      x += self.a * np.sign(grad)

      x = np.clip(x, x_nat - self.epsilon, x_nat + self.epsilon)
      x = np.clip(x, 0, 1) # ensure valid pixel range

    return x


if __name__ == '__main__':
  import json
  import sys
  import math

  from tensorflow.examples.tutorials.mnist import input_data

  from models import Classifier

  with open('configs/mnist.json') as config_file:
    config = json.load(config_file)

  model_file = sys.argv[1]

  model  = Classifier(var_scope='classifier')
  vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='classifier')
  saver = tf.train.Saver(var_list=vars)
  attack = LinfPGDAttack(model,
                         config['epsilon'],
                         config['k'],
                         config['a'],
                         config['random_start'],
                         config['loss_func'])

 # mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  x_train = np.reshape(x_train, [x_train.shape[0], -1])
  x_test = np.reshape(x_test, [x_test.shape[0], -1])
  x_train = x_train.astype(np.float32) / 255.
  x_test = x_test.astype(np.float32) / 255.


  conf = tf.ConfigProto(device_count = {'GPU': 0})
  with tf.Session(config=conf) as sess:
    # Restore the checkpoint
    saver.restore(sess, model_file)

    # Iterate over the samples batch-by-batch
    num_eval_examples = config['num_eval_examples']
    eval_batch_size = config['eval_batch_size']
    num_batches = int(math.ceil(num_eval_examples / eval_batch_size))

    x_adv = [] # adv accumulator
    y_adv = []

    print('Iterating over {} batches'.format(num_batches))

    s = 0.0
    corrects = 0.0
    for ibatch in range(num_batches):
      bstart = ibatch * eval_batch_size
      bend = min(bstart + eval_batch_size, num_eval_examples)
      print('batch size: {}'.format(bend - bstart))

      x_batch = x_test[bstart:bend, :]
      y_batch = y_test[bstart:bend]

      x_batch_adv = attack.perturb(x_batch, y_batch, sess)

      correct = sess.run(model.num_correct, feed_dict={model.x_input: x_batch_adv, model.y_input: y_batch})
      corrects += correct
      s += bend - bstart
      print(corrects/s)

      x_adv.append(x_batch_adv)
      y_adv.append(y_batch)

    print('Storing examples')
    path = config['store_adv_path']
    x_adv = np.concatenate(x_adv, axis=0)
    y_adv = np.concatenate(y_adv)
    np.save('robust_attack.npy', x_adv)
    np.save('y_test.npy', y_adv)
    print('Examples stored in {}'.format(path))
    acc = sess.run(model.accuracy, feed_dict={model.x_input: x_adv, model.y_input: y_adv})
    print('x_adv.shape {}, acc {}'.format(x_adv.shape, acc))
