import os
import argparse
import numpy as np
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

from models import Classifier

parser = argparse.ArgumentParser()

args = parser.parse_args()

tf.set_random_seed(4557077)
np.random.seed(123)

batch_size = 32

mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = np.reshape(x_train, [x_train.shape[0], -1])
x_test = np.reshape(x_test, [x_test.shape[0], -1])

classifier = Classifier(var_scope='classifier', dataset='MNIST')
#
vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='classifier')
saver = tf.train.Saver(var_list=vars, max_to_keep=1)
train_step = tf.train.AdamOptimizer(5e-4, name='classifier_adam').minimize(classifier.xent)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  num_epoches = 10
  for epoch in range(num_epoches):
    perm = np.random.permutation(x_train.shape[0])
    x_train, y_train = x_train[perm], y_train[perm]
    for i in range(0, x_train.shape[0], batch_size):
      x_batch = x_train[i: i + batch_size]
      y_batch = y_train[i: i + batch_size]
      _, acc = sess.run([train_step, classifier.accuracy], feed_dict={classifier.x_input: x_batch, classifier.y_input: y_batch})

    test_acc = sess.run(classifier.accuracy, feed_dict={classifier.x_input: x_test, classifier.y_input: y_test})
    print('testing epoch {}, acc {:.3f}'.format(epoch, test_acc))

  model_name = os.path.join('checkpoints/mnist/classifier')
  saver.save(sess, model_name)
  print('saved ' + model_name)

