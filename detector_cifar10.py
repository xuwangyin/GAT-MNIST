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
batch_size_others = batch_size
test_batch_size_target = batch_size
test_batch_size_others = batch_size

# Setting up the data and the model
raw_cifar_target = cifar10_input.CIFAR10Data(data_path, target_class=args.target_class, select='target')
raw_cifar_others = cifar10_input.CIFAR10Data(data_path, target_class=args.target_class, select='others')
global_step = tf.contrib.framework.get_or_create_global_step()

class LinfPGDAttack:
  def __init__(self, model, epsilon, num_steps, step_size, random_start, loss_func):
    """Attack parameter initialization. The attack performs k steps of
       size a, while always staying within epsilon from the initial
       point."""
    self.model = model
    self.epsilon = epsilon
    self.num_steps = num_steps
    self.step_size = step_size
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
      wrong_logit = tf.reduce_max((1-label_mask) * model.logits - 1e4*label_mask, axis=1)
      loss = -tf.nn.relu(correct_logit - wrong_logit + 50)
    else:
      print('Unknown loss function. Defaulting to cross-entropy')
      loss = model.xent

    self.grad = tf.gradients(loss, model.x_input)[0]

  def perturb(self, x_nat, y, sess):
    """Given a set of examples (x_nat, y), returns a set of adversarial
       examples within epsilon of x_nat in l_infinity norm."""
    if self.rand:
      x = x_nat + np.random.uniform(-self.epsilon, self.epsilon, x_nat.shape)
      x = np.clip(x, 0, 255) # ensure valid pixel range
    else:
      x = np.copy(x_nat)

    for i in range(self.num_steps):
      print(i)
      grad = sess.run(self.grad, feed_dict={self.model.x_input: x,
                                            self.model.y_input: y})

      x = np.add(x, self.step_size * np.sign(grad), out=x, casting='unsafe')

      x = np.clip(x, x_nat - self.epsilon, x_nat + self.epsilon)
      x = np.clip(x, 0, 255) # ensure valid pixel range

    return x

if args.test_model is not None:
  classifier = Classifier(var_scope='classifier', dataset='CIFAR10', mode='eval')
  classifier_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='classifier')
  classifier_saver = tf.train.Saver(var_list=classifier_vars)

  detector = Detector(var_scope='detector', dataset='CIFAR10', mode='eval')
  detector_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='detector')
  detector_saver = tf.train.Saver(var_list=detector_vars)

  test_attack = PGDAttack(classifier, None,
                          max_distance=args.d, left_conf_th=0, right_conf_th=200,
                          num_steps=config['num_steps'] * 2, step_size=config['step_size'], random_start=False,
                          x_min=x_min, x_max=x_max, loss_fn='clf_adv',
                          batch_size=test_batch_size_others, distance_type=args.distance_type, target_class=args.target_class)

  madry_attack = LinfPGDAttack(classifier,
                         config['epsilon'],
                         config['num_steps']*2,
                         config['step_size'],
                         random_start=False,
                         loss_func='cw')

  config = tf.ConfigProto(device_count = {'GPU': 0})
  with tf.Session() as sess:
    # sess.run(tf.global_variables_initializer())

    classifier_saver.restore(sess, '../models/madry_advtrained_classifier/checkpoint-70000')
    i = 0
    x_test_target = raw_cifar_target.eval_data.xs[i * 128: (i + 1) * 128].astype(np.float32)
    x_test_others = raw_cifar_others.eval_data.xs[i * 128: (i + 1) * 128].astype(np.float32)
    x_test_target = np.reshape(x_test_target, [x_test_target.shape[0], -1])
    x_test_others = np.reshape(x_test_others, [x_test_others.shape[0], -1])

    y_test_target = raw_cifar_target.eval_data.ys[i * 128: (i + 1) * 128].astype(np.float32)
    y_test_others = raw_cifar_others.eval_data.ys[i * 128: (i + 1) * 128].astype(np.float32)

    test_feed_dict = {classifier.x_input: x_test_others, classifier.y_input: y_test_others}
    acc, loss = sess.run([classifier.accuracy, classifier.xent], feed_dict=test_feed_dict)
    print('test accuracy {:.4f}, loss {:.4f}'.format(acc * 100, loss))

    print(datetime.now())
    x_test_others_adv = madry_attack.perturb(x_test_others, y_test_others, sess)
    test_feed_dict = {classifier.x_input: x_test_others_adv, classifier.y_input: y_test_others}
    acc, loss = sess.run([classifier.accuracy, classifier.xent], feed_dict=test_feed_dict)
    print('madry adv test accuracy {:.4f}, loss {:.4f}'.format(acc * 100, loss))
    print(datetime.now())

    x_test_others_adv, test_dist, detector_logits = test_attack.perturb(x_test_others, y_test_others, sess)
    test_feed_dict = {classifier.x_input: x_test_others_adv, classifier.y_input: y_test_others}
    acc, loss = sess.run([classifier.accuracy, classifier.xent], feed_dict=test_feed_dict)
    print('adv test accuracy {:.4f}, loss {:.4f}'.format(acc * 100, loss))
    print(datetime.now())

    sys.exit(0)
    detector_saver.restore(sess, args.test_model)
    print('loaded {}'.format(args.test_model))

    madry_adv = np.load('/home/xy4cm/Projects/cifar10_challenge/attack.npy')[:1000]
    madry_adv = np.reshape(madry_adv, [madry_adv.shape[0], -1]).astype(np.float32)
    test_feed_dict = {detector.x_input: madry_adv, detector.y_input: np.zeros(madry_adv.shape[0])}
    acc, loss, detector_logits = sess.run([detector.accuracy, detector.xent, detector.logits], feed_dict=test_feed_dict)
    print('madry adv detection accuracy {:.4f}, loss {}, logits mean {} std {}'.format(acc * 100, loss, detector_logits.mean(), detector_logits.std()))


    x_test_target = raw_cifar_target.eval_data.xs[:1000].astype(np.float32)
    x_test_target = np.reshape(x_test_target, [x_test_target.shape[0], -1])
    test_feed_dict = {detector.x_input: x_test_target, detector.y_input: np.ones(x_test_target.shape[0])}
    acc, loss, detector_logits = sess.run([detector.accuracy, detector.xent, detector.logits], feed_dict=test_feed_dict)
    print('target detection accuracy {:.4f}, loss {}, logits mean {} std {}'.format(acc * 100, loss, detector_logits.mean(), detector_logits.std()))

    x_test_with_madry_adv = np.concatenate([x_test_target, madry_adv])
    y_test_with_madry_adv = np.concatenate(
      [np.ones(x_test_target.shape[0], dtype=np.int64), np.zeros(madry_adv.shape[0], dtype=np.int64)])
    test_feed_dict = {detector.x_input: x_test_with_madry_adv, detector.y_input: y_test_with_madry_adv}
    x_test_with_madry_adv_logits = sess.run(detector.logits, feed_dict=test_feed_dict)
    fpr_, tpr_, thresholds = roc_curve(y_test_with_madry_adv, x_test_with_madry_adv_logits)
    roc_auc = auc(fpr_, tpr_)
    print('x_test_with_madry_adv auc {}'.format(roc_auc))

    for i in range(10):
      x_test_target = raw_cifar_target.eval_data.xs[i * test_batch_size_target: (i + 1) * test_batch_size_target].astype(np.float32)
      x_test_others = raw_cifar_others.eval_data.xs[i * test_batch_size_others: (i + 1) * test_batch_size_others].astype(np.float32)
      x_test_target = np.reshape(x_test_target, [x_test_target.shape[0], -1])
      x_test_others = np.reshape(x_test_others, [x_test_others.shape[0], -1])

      y_test_target = raw_cifar_target.eval_data.ys[i * test_batch_size_target: (i + 1) * test_batch_size_target].astype(np.float32)
      y_test_others = raw_cifar_others.eval_data.ys[i * test_batch_size_others: (i + 1) * test_batch_size_others].astype(np.float32)

      test_feed_dict = {classifier.x_input: x_test_others, classifier.y_input: y_test_others}
      acc, loss = sess.run([classifier.accuracy, classifier.xent], feed_dict=test_feed_dict)
      print('test accuracy {:.4f}, loss {:.4f}'.format(acc * 100, loss))

      x_test_with_nat = np.concatenate([x_test_target, x_test_others])
      y_test_with_nat = np.concatenate(
        [np.ones(x_test_target.shape[0], dtype=np.int64), np.zeros(x_test_others.shape[0], dtype=np.int64)])
      test_feed_dict = {detector.x_input: x_test_with_nat, detector.y_input: y_test_with_nat}
      x_test_with_nat_logits = sess.run(detector.logits, feed_dict=test_feed_dict)
      fpr_, tpr_, thresholds = roc_curve(y_test_with_nat, x_test_with_nat_logits)
      roc_auc = auc(fpr_, tpr_)
      print('x_test_with_nat auc {}'.format(roc_auc))


      test_feed_dict = {detector.x_input: x_test_target, detector.y_input: np.ones(x_test_target.shape[0])}
      acc, loss, detector_logits = sess.run([detector.accuracy, detector.xent, detector.logits], feed_dict=test_feed_dict)
      print('target detection accuracy {:.4f}, loss {}, logits mean {} std {}'.format(acc * 100, loss, detector_logits.mean(), detector_logits.std()))

      test_feed_dict = {detector.x_input: x_test_others, detector.y_input: np.zeros(x_test_others.shape[0])}
      acc, loss, detector_logits = sess.run([detector.accuracy, detector.xent, detector.logits], feed_dict=test_feed_dict)
      print('others detection accuracy {:.4f}, loss {}, logits mean {} std {}'.format(acc * 100, loss, detector_logits.mean(), detector_logits.std()))
      np.save('detector_logits_nat.npy', detector_logits)
       
      sys.exit(0)

      tic = time.time()
      x_test_others_adv, test_dist, detector_logits, mask = test_attack.perturb(x_test_others, y_test_others, sess)
      toc = time.time()
      print('detector_logits.mean: {} std: {}'.format(detector_logits.mean(), detector_logits.std()))
      np.save('detector_logits_adv.npy', detector_logits)

      test_feed_dict = {detector.x_input: x_test_others_adv, detector.y_input: np.zeros(x_test_others_adv.shape[0])}
      acc, loss, detector_logits = sess.run([detector.accuracy, detector.xent, detector.logits], feed_dict=test_feed_dict)
      print('others adv detection accuracy {:.4f}, loss {}, logits mean {} std {}'.format(acc * 100, loss, detector_logits.mean(), detector_logits.std()))
    
      print('x_test_others_adv.shape {}'.format(x_test_others_adv.shape))

      test_feed_dict = {classifier.x_input: x_test_others_adv, classifier.y_input: y_test_others[mask]}
      preds, acc, loss = sess.run([classifier.predictions, classifier.accuracy, classifier.xent], feed_dict=test_feed_dict)
      print('adv others test accuracy {:.4f}, loss {:.4f}'.format(acc * 100, loss))
      print(preds)


      x_test_with_adv = np.concatenate([x_test_target, x_test_others_adv])
      y_test_with_adv = np.concatenate(
        [np.ones(x_test_target.shape[0], dtype=np.int64), np.zeros(x_test_others_adv.shape[0], dtype=np.int64)])

      test_feed_dict = {detector.x_input: x_test_with_adv, detector.y_input: y_test_with_adv}
      x_test_with_adv_logits = sess.run(detector.logits, feed_dict=test_feed_dict)
      fpr_, tpr_, thresholds = roc_curve(y_test_with_adv, x_test_with_adv_logits)
      roc_auc = auc(fpr_, tpr_)
      print('x_test_with_adv auc {}'.format(roc_auc))


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

      plt.plot(fpr_, tpr_, label='AUC = {:.4f}'.format(roc_auc))
      plt.xlabel('False Positive Rate')
      plt.ylabel('True Positive Rate')
      plt.title('Receiver operating characteristic example')
      plt.legend(loc="lower right")
      plt.show()

      fig, axes = plt.subplots(nrows=10, ncols=10, figsize=(10, 10))
      for img, ax, y, pred in zip(x_test_with_adv, axes.ravel(), y_test_with_adv, y_pred_test):
        ax.imshow(img.reshape([32, 32, 3]).astype(np.uint8))
        ax.set_title('{}->{}'.format(y, pred))
        ax.axis('off')
      plt.show()

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
                         x_min=x_min, x_max=x_max, batch_size=batch_size_others,
                         loss_fn='det_adv', distance_type=args.distance_type, target_class=args.target_class)

test_attack = PGDAttack(classifier, detector,
                        max_distance=args.d, left_conf_th=0, right_conf_th=200,
                        num_steps=config['num_steps']*2, step_size=config['step_size'], random_start=False,
                        x_min=x_min, x_max=x_max, loss_fn='det_adv',
                        batch_size=test_batch_size_others, distance_type=args.distance_type, target_class=args.target_class)

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

    _, detector_logits, f_score, precision, recall, acc, balanced_acc, tpr, fpr = sess.run(
      [train_step, detector.logits, detector.f_score, detector.precision, detector.recall, detector.accuracy,
       detector.balanced_accuracy, detector.true_positive_rate, detector.false_positive_rate],
      feed_dict={detector.x_input: x_batch_with_adv, detector.y_input: y_batch_with_adv})
    fpr_, tpr_, thresholds = roc_curve(y_batch_with_adv, detector_logits)
    roc_auc = auc(fpr_, tpr_)


    print('step {}/{}, '
          'train auc {:.4f}, f-score {:.4f}, precision {:.4f}, recall {:.4f}, '
          'acc {:.4f}, balanced_acc {:.4f} tpr {:.4f} fpr {:.4f} '
          'dist<={:.4f} {:.4f} {:.4f}/{:.4f} time {:.1f}'.format(
      step, max_num_training_steps, roc_auc, f_score, precision, recall,
      acc, balanced_acc, tpr, fpr, args.d,
      (batch_dist <= args.d + 1e-6).mean(), batch_dist.mean(), batch_dist.std(), 1000 * (toc - tic)))

    # x_batch_with_nat = np.concatenate([x_batch_target, x_batch_others])
    # y_batch_with_nat = np.concatenate(
    #   [np.ones(x_batch_target.shape[0], dtype=np.int64), np.zeros(x_batch_others.shape[0], dtype=np.int64)])
    # sess.run(train_step, feed_dict={detector.x_input: x_batch_with_nat, detector.y_input: y_batch_with_nat})

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

      x_test_with_nat = np.concatenate([x_test_target, x_test_others])
      y_test_with_nat = np.concatenate(
        [np.ones(x_test_target.shape[0], dtype=np.int64), np.zeros(x_test_others.shape[0], dtype=np.int64)])

      test_detector_logits = sess.run(detector.logits, feed_dict={detector.x_input: x_test_with_nat, detector.y_input: y_test_with_nat})
      fpr_, tpr_, thresholds = roc_curve(y_test_with_nat, test_detector_logits)
      roc_auc = auc(fpr_, tpr_)
      print('x_test_with_nat auc {}'.format(roc_auc))

    # Write a checkpoint
    if step % num_checkpoint_steps == 0:
      model_name = 'checkpoints/{}/detector_ovr_class{}_{}_distance{}'.format(
        args.dataset, args.target_class, args.distance_type, args.d)
      detector_saver.save(sess, model_name, global_step=global_step)
      print('saved {}'.format(model_name))

