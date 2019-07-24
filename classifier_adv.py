import argparse
import json
import sys
import time
import tensorflow as tf
import os
from timeit import default_timer as timer

from models import Classifier, PGDAttack
from utils import *
import pathlib

parser = argparse.ArgumentParser()
parser.add_argument('-m', choices=['softmax'], default='softmax')
parser.add_argument('--dim', type=int, default=2)
parser.add_argument('-d', type=float, default=1.5)
parser.add_argument('--dataset', choices=['cifar10', 'mnist'], default='mnist')
parser.add_argument('--test_model', type=str)
parser.add_argument('--distance_type', choices=['L2', 'Linf'], default='Linf')
parser.add_argument('--test_steps', type=int, default=5000)
parser.add_argument('--test_step_size', type=float, default=0.001)

args = parser.parse_args()

args.d = {'L2': 1.5, 'Linf': 0.3}[args.distance_type]

with open('configs/mnist.json') as config_file:
  config = json.load(config_file)

tf.set_random_seed(config['random_seed'])
np.random.seed(123)

batch_size = config['training_batch_size']

x_min, x_max = 0.0, 1.0

x_train, y_train, x_test, y_test, num_classes = dataset(args)
#x_test, y_test = x_test[:1000], y_test[:1000]

classifier = Classifier(var_scope='classifier')
#
vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='classifier')
saver = tf.train.Saver(var_list=vars, max_to_keep=200)
train_step = tf.train.AdamOptimizer(5e-4, name='classifier_adam').minimize(classifier.xent)

train_attack = PGDAttack(classifier, detector=None,
                         max_distance=args.d, left_conf_th=0.0, right_conf_th=1.0,
                         num_steps=40, step_size=0.01, random_start=True,
                         x_min=x_min, x_max=x_max, batch_size=batch_size,
                         distance_type=args.distance_type, loss_fn='clf_adv')

test_attack = PGDAttack(classifier, detector=None,
                        max_distance=args.d, left_conf_th=0.0, right_conf_th=1.0,
                        num_steps=200, step_size=0.01, random_start=False,
                        x_min=x_min, x_max=x_max, loss_fn='clf_adv',
                        batch_size=x_test.shape[0], distance_type=args.distance_type)

config = tf.ConfigProto(device_count = {'GPU': 0})
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  if args.test_model is not None:
    saver.restore(sess, args.test_model)
    x_test, y_test = x_test[:1000], y_test[:1000]
    test_attack = PGDAttack(classifier, detector=None,
                              max_distance=args.d, left_conf_th=0.0, right_conf_th=1.0,
                              num_steps=args.test_steps, step_size=args.test_step_size, random_start=False,
                              x_min=x_min, x_max=x_max, loss_fn='clf_adv',
                              distance_type=args.distance_type, batch_size=x_test.shape[0])
    x_test_adv, _, _ = test_attack.perturb(x_test, y_test, sess)
    print('x_test_adv.shape: {}'.format(x_test_adv.shape))
    #x_test_adv = np.load('./robust_attack.npy')
    #y_test_adv = np.load('./y_test.npy')
    #x_test_adv, y_test_adv = test_attack.perturb(x_test, y_test, sess)
    test_acc = sess.run(classifier.accuracy, feed_dict={classifier.x_input: x_test_adv, classifier.y_input: y_test})
    print('adv test set acc {:.3f}'.format(test_acc))
    test_acc = sess.run(classifier.accuracy, feed_dict={classifier.x_input: x_test, classifier.y_input: y_test})
    print('nat test set acc {:.3f}'.format(test_acc))
    sys.exit(0)

  for epoch in range(1, 201):
    perm = np.random.permutation(x_train.shape[0])
    x_train, y_train = x_train[perm], y_train[perm]
    for i in range(0, x_train.shape[0], batch_size):
      x_batch = x_train[i: i + batch_size]
      y_batch = y_train[i: i + batch_size]
      if x_batch.shape[0] < batch_size:
        continue

      tic = timer()
      x_batch_adv, confidence, dist = train_attack.perturb(x_batch, y_batch, sess, verbose=False)
      toc = timer()

      feed_dict = {classifier.x_input: x_batch_adv, classifier.y_input: y_batch}
      _, acc = sess.run([train_step, classifier.accuracy], feed_dict)

      if (i // batch_size) % 10 == 0:
        print('epoch {} {}/{} acc {:.4f} time {:.2f}ms'.format(epoch, i, x_train.shape[0], acc, 1000 * (toc - tic)))

    # x_test_adv, y_test_adv = test_attack.perturb(x_train[:x_test.shape[0]], y_train[:y_test.shape[0]], sess)
    # test_acc = sess.run(classifier.accuracy, feed_dict={classifier.x_input: x_test_adv, classifier.y_input: y_test_adv})
    # print('testing epoch {}, train set acc {:.3f}'.format(epoch, test_acc))

    tic = timer()
    x_test_adv, confidence, dist = test_attack.perturb(x_test, y_test, sess)
    toc = timer()
    test_acc = sess.run(classifier.accuracy, feed_dict={classifier.x_input: x_test_adv, classifier.y_input: y_test})
    print('testing epoch {}, test set acc {:.3f}, time {:.2f}ms'.format(epoch, test_acc, 1000 * (toc - tic)))

    pathlib.Path('checkpoints/mnist/classifier').mkdir(parents=True, exist_ok=True)
    model_name = os.path.join('checkpoints/mnist/classifier/adv_trained-{}-{}-{}'.format(
      args.distance_type, args.d, args.dataset))
    saver.save(sess, model_name, global_step=epoch)
    print('saved ' + model_name)

