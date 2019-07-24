import argparse
import json
import sys
from datetime import datetime
import time
from sklearn.metrics import roc_curve, auc

import os

import cifar10_input
from models import Classifier, Detector, PGDAttack
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', choices=['cifar10', 'mnist'], default='cifar10')
parser.add_argument('-d', metavar='max-distance', type=float, default=8.0)
parser.add_argument('--target_class', type=int, required=True)
parser.add_argument('--distance_type', choices=['L2', 'Linf'], default='Linf')
parser.add_argument('--test_model', type=str)

args = parser.parse_args()

x_min, x_max = 0.0, 255.0

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
batch_size_others = batch_size * 9
test_batch_size_target = 30
test_batch_size_others = 270

# Setting up the data and the model
raw_cifar_target = cifar10_input.CIFAR10Data(data_path, target_class=args.target_class, select='target')
raw_cifar_others = cifar10_input.CIFAR10Data(data_path, target_class=args.target_class, select='others')
global_step = tf.contrib.framework.get_or_create_global_step()

if args.test_model is not None:
  classifier = Classifier(var_scope='classifier', dataset='CIFAR10', mode='eval')
  detector = Detector(var_scope='detector', dataset='CIFAR10', mode='eval')
  vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='detector')
  saver = tf.train.Saver(var_list=vars)

  test_attack = PGDAttack(classifier, detector,
                          max_distance=args.d, left_conf_th=0, right_conf_th=200,
                          num_steps=config['num_steps'] * 2, step_size=config['step_size'], random_start=False,
                          x_min=x_min, x_max=x_max, lambda_=1, loss_fn='one-vs-rest',
                          batch_size=test_batch_size_others, distance_type=args.distance_type)

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    saver.restore(sess, args.test_model)
    print('loaded {}'.format(args.test_model))

    x_test_target = raw_cifar_target.eval_data.xs[:test_batch_size_target]
    x_test_others = raw_cifar_others.eval_data.xs[:test_batch_size_others]
    x_test_target = np.reshape(x_test_target, [x_test_target.shape[0], -1])
    x_test_others = np.reshape(x_test_others, [x_test_others.shape[0], -1])

    tic = time.time()
    x_test_others_adv, test_dist, _ = test_attack.perturb(x_test_others, None, sess)
    toc = time.time()

    x_test_with_adv = np.concatenate([x_test_target, x_test_others_adv])
    y_test_with_adv = np.concatenate(
      [np.ones(x_test_target.shape[0], dtype=np.int64), np.zeros(x_test_others_adv.shape[0], dtype=np.int64)])

    test_detector_logits, y_pred_test, f_score, precision, recall, acc, balanced_acc, tpr, fpr = sess.run(
      [detector.logits, detector.predictions, detector.f_score, detector.precision, detector.recall,
       detector.accuracy,
       detector.balanced_accuracy, detector.true_positive_rate, detector.false_positive_rate],
      feed_dict={detector.x_input: x_test_with_adv, detector.y_input: y_test_with_adv})
    fpr_, tpr_, thresholds = roc_curve(y_test_with_adv, test_detector_logits)
    roc_auc = auc(fpr_, tpr_)
    print(
      '=== test auc {:.4f}, f-score {:.4f}, precision {:.4f}, recall {:.4f}, acc {:.4f}, balanced_acc {:.4f}, tpr {:.4f} fpr {:.4f} '.format(
        roc_auc, f_score, precision, recall, acc, balanced_acc, tpr, fpr), end='|')

    print('pos {:.4f}, true pos {:.4f}, target {}, others {}, {:.1f}ms'.format(np.sum(y_pred_test),
                                                                               np.sum(
                                                                                 np.bitwise_and(y_pred_test,
                                                                                                y_test_with_adv)),
                                                                               np.sum(y_test_with_adv),
                                                                               np.sum(1 - y_test_with_adv),
                                                                               1000 * (toc - tic)))

    sys.exit(0)

classifier = Classifier(var_scope='classifier', dataset='CIFAR10', mode='eval')
detector = Detector(var_scope='detector', dataset='CIFAR10', mode='train')
#
# use GLOBAL_VARIABLES to collect batchnorm moving mean and variance
vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='detector')
detector_saver = tf.train.Saver(var_list=vars, max_to_keep=30)
tf.summary.scalar('train accuracy', detector.accuracy)
tf.summary.scalar('train loss', detector.xent)
merged_summaries = tf.summary.merge_all()

# Setting up the optimizer
boundaries = [int(sss[0]) for sss in step_size_schedule]
boundaries = boundaries[1:]
values = [sss[1] for sss in step_size_schedule]
learning_rate = tf.train.piecewise_constant(tf.cast(global_step, tf.int32),
                                            boundaries, values)
total_loss = detector.xent + weight_decay * detector.weight_decay_loss
train_step = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(
    total_loss,
    global_step=global_step)


train_attack = PGDAttack(classifier, detector,
                         max_distance=args.d, left_conf_th=0, right_conf_th=200,
                         num_steps=config['num_steps'], step_size=config['step_size'], random_start=True,
                         x_min=x_min, x_max=x_max, lambda_=1, batch_size=batch_size_others,
                         loss_fn='one-vs-rest', distance_type=args.distance_type)

test_attack = PGDAttack(classifier, detector,
                        max_distance=args.d, left_conf_th=0, right_conf_th=200,
                        num_steps=config['num_steps']*2, step_size=config['step_size'], random_start=False,
                        x_min=x_min, x_max=x_max, lambda_=1, loss_fn='one-vs-rest',
                        batch_size=test_batch_size_others, distance_type=args.distance_type)

with tf.Session() as sess:
  cifar_target = cifar10_input.AugmentedCIFAR10Data(raw_cifar_target, sess, detector)
  cifar_others = cifar10_input.AugmentedCIFAR10Data(raw_cifar_others, sess, detector)

  summary_writer = tf.summary.FileWriter('logs/detector_{}/'.format(args.dataset), sess.graph)
  sess.run(tf.global_variables_initializer())

  # Main training loop
  for step in range(max_num_training_steps):
    x_batch_target, y_batch_target = cifar_target.train_data.get_next_batch(batch_size, multiple_passes=True)
    x_batch_others, y_batch_others = cifar_others.train_data.get_next_batch(batch_size_others, multiple_passes=True)
    x_batch_target = np.reshape(x_batch_target, [x_batch_target.shape[0], -1])
    x_batch_others = np.reshape(x_batch_others, [x_batch_others.shape[0], -1])

    if x_batch_target.shape[0] != batch_size or \
        x_batch_others.shape[0] != batch_size_others:
      continue

    tic = time.time()
    x_batch_others_adv, batch_dist, _ = train_attack.perturb(x_batch_others, None, sess, verbose=False)
    toc = time.time()

    x_batch_with_adv = np.concatenate([x_batch_target, x_batch_others_adv])
    y_batch_with_adv = np.concatenate(
      [np.ones(x_batch_target.shape[0], dtype=np.int64), np.zeros(x_batch_others_adv.shape[0], dtype=np.int64)])

    _, f_score, precision, recall, acc, balanced_acc, tpr, fpr = sess.run(
      [train_step, detector.f_score, detector.precision, detector.recall, detector.accuracy,
       detector.balanced_accuracy, detector.true_positive_rate, detector.false_positive_rate],
      feed_dict={detector.x_input: x_batch_with_adv, detector.y_input: y_batch_with_adv})

    print('step {}/{}, '
          'train f-score {:.4f}, precision {:.4f}, recall {:.4f}, '
          'acc {:.4f}, balanced_acc {:.4f} tpr {:.4f} fpr {:.4f} '
          'dist<={:.4f} {:.4f} {:.4f}/{:.4f} time {:.1f}'.format(
      step, max_num_training_steps, f_score, precision, recall,
      acc, balanced_acc, tpr, fpr, args.d,
      (batch_dist <= args.d + 1e-6).mean(), batch_dist.mean(), batch_dist.std(), 1000 * (toc - tic)))

    # Tensorboard summaries
    if step % num_summary_steps == 0:
      x_test_target = raw_cifar_target.eval_data.xs[:test_batch_size_target]
      x_test_others = raw_cifar_others.eval_data.xs[:test_batch_size_others]
      x_test_target = np.reshape(x_test_target, [x_test_target.shape[0], -1])
      x_test_others = np.reshape(x_test_others, [x_test_others.shape[0], -1])

      tic = time.time()
      x_test_others_adv, test_dist, _ = test_attack.perturb(x_test_others, None, sess)
      toc = time.time()

      x_test_with_adv = np.concatenate([x_test_target, x_test_others_adv])
      y_test_with_adv = np.concatenate(
        [np.ones(x_test_target.shape[0], dtype=np.int64), np.zeros(x_test_others_adv.shape[0], dtype=np.int64)])

      test_detector_logits, y_pred_test, f_score, precision, recall, acc, balanced_acc, tpr, fpr = sess.run(
        [detector.logits, detector.predictions, detector.f_score, detector.precision, detector.recall,
         detector.accuracy,
         detector.balanced_accuracy, detector.true_positive_rate, detector.false_positive_rate],
        feed_dict={detector.x_input: x_test_with_adv, detector.y_input: y_test_with_adv})
      fpr_, tpr_, thresholds = roc_curve(y_test_with_adv, test_detector_logits)
      roc_auc = auc(fpr_, tpr_)
      print(
        '=== test auc {:.4f}, f-score {:.4f}, precision {:.4f}, recall {:.4f}, acc {:.4f}, balanced_acc {:.4f}, tpr {:.4f} fpr {:.4f} '.format(
          roc_auc, f_score, precision, recall, acc, balanced_acc, tpr, fpr), end='|')

      print('pos {:.4f}, true pos {:.4f}, target {}, others {}, {:.1f}ms'.format(np.sum(y_pred_test),
                                                                       np.sum(
                                                                         np.bitwise_and(y_pred_test, y_test_with_adv)),
                                                                       np.sum(y_test_with_adv),
                                                                       np.sum(1 - y_test_with_adv), 1000 * (toc - tic)))

    # Write a checkpoint
    if step % num_checkpoint_steps == 0:
      model_name = 'checkpoints/{}/detector_ovr_class{}_{}_distance{}'.format(
        args.dataset, args.target_class, args.distance_type, args.d)
      detector_saver.save(sess, model_name, global_step=global_step)
      print('saved {}'.format(model_name))

