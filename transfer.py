import argparse
import sys
import os
import pandas as pd
from cleverhans.attacks import FastGradientMethod
from cleverhans.attacks import CarliniWagnerL2
from cleverhans.attacks import SPSA
from cleverhans.model import Model


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from sklearn.metrics import roc_curve, auc

from models import Detector, Classifier
from models import PGDAttackDetector, PGDAttackClassifier, PGDAttackAda, MadryPGDAttackDetector, MadryPGDAttackClassifier
from utils import *
import matplotlib.pyplot as plt
import json

parser = argparse.ArgumentParser()
# parser.add_argument('--target_class', type=int, required=True)
parser.add_argument('--epsilon', metavar='max-distance', type=float, required=True)
parser.add_argument('--norm', choices=['L2', 'Linf'], default='Linf')
parser.add_argument('--optimizer', choices=['adam', 'normgrad'], default='adam')
parser.add_argument('--steps', type=int, default=500)
parser.add_argument('--step_size', type=float, default=0.01)
parser.add_argument('--method', choices=['ovr', 'nat', 'static-adv', 'detector-adv', 'adaptive-adv'], required=True)
parser.add_argument('--madry', action='store_true')


args = parser.parse_args()
print(args)
arg_str = '-'.join(['{}={}'.format(k, v) for k, v in vars(args).items()])

np.random.seed(123)

if args.norm == 'L2':
  pass
  # assert args.epsilon in [2.5, 5.0, 10.0]
  # assert args.step_size == 0.1
  # assert args.optimizer == 'normgrad'
if args.norm == 'Linf':
  pass
  #assert args.epsilon in [0.3, 0.5, 0.7, 1.0]
  # assert args.optimizer == 'adam'


class CleverhansModel(Model):
  def __init__(self, classifier):
    super().__init__(classifier.var_scope, classifier.output_size)
    self.classifier = classifier

  def fprop(self, x, **kwargs):
    logits = classifier.net.forward(x)
    return {self.O_LOGITS: logits, self.O_PROBS: tf.nn.softmax(logits=logits)}


mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = np.reshape(x_train, [x_train.shape[0], -1])
x_test = np.reshape(x_test, [x_test.shape[0], -1])

x_min, x_max = 0.0, 1.0

classifier = Classifier(var_scope='classifier', dataset='MNIST')

vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='classifier')
saver = tf.train.Saver(var_list=vars, max_to_keep=1)

num_classes = 10
if args.epsilon == 0.3 or (args.norm == 'Linf') or (args.norm == 'L2'):
  print('args.epsilon: {}'.format(args.epsilon))
  dir = 'checkpoints/mnist/detector_Linf_0.3/ovr-steps100-adam-noclip-balanced/'
  best = [91, 68, 76, 95, 71, 90, 89, 64, 54, 59]
  detector_models = [os.path.join(dir, 'ovr_class{}_{}_distance{}-{}'.format(i, 'Linf', 0.3, best[i])) for i in range(num_classes)]
  print('using linf 0.3 model')

  #dir = 'checkpoints/mnist/detector_Linf_0.5/ovr-steps100-adam-noclip-balanced/'
  #best = [97, 39, 30, 28, 28, 29, 72, 63, 67, 37]
  #detector_models = [os.path.join(dir, 'ovr_class{}_{}_distance{}-{}'.format(i, args.norm, 0.5, best[i])) for i in range(num_classes)]

  #dir = 'checkpoints/mnist/detector_L2_2.5/ovr-steps100-adam-noclip-balanced/'
  #best = [50,80,51,83,95,99,50,91,72,66]
  #detector_models = [os.path.join(dir, 'ovr_class{}_{}_distance{}-{}'.format(i, 'L2', 2.5, best[i])) for i in range(num_classes)]
  #dir = 'checkpoints/mnist/detector_L2_5.0/ovr-steps200-adam-noclip-balanced'
  #best = [57,99,77,100,43,93,80,98,88,59]
  #detector_models = [os.path.join(dir, 'ovr_class{}_{}_distance{}-{}'.format(i, 'L2', 5.0, best[i])) for i in range(num_classes)]
elif args.epsilon == 0.5:
  print('args.epsilon: {}'.format(args.epsilon))
  dir = 'checkpoints/mnist/detector_Linf_0.5/ovr-steps100-adam-noclip-balanced/'
  best = [97, 39, 30, 28, 28, 29, 72, 63, 67, 37]
  detector_models = [os.path.join(dir, 'ovr_class{}_{}_distance{}-{}'.format(i, 'Linf', 0.5, best[i])) for i in range(num_classes)]

  #dir = 'checkpoints/mnist/detector_L2_2.5/ovr-steps100-adam-noclip-balanced/'
  #best = [50,80,51,83,95,99,50,91,72,66]
  #detector_models = [os.path.join(dir, 'ovr_class{}_{}_distance{}-{}'.format(i, 'L2', 2.5, best[i])) for i in range(num_classes)]
  #dir = 'checkpoints/mnist/detector_L2_5.0/ovr-steps200-adam-noclip-balanced'
  #best = [57,99,77,100,43,93,80,98,88,59]
  #detector_models = [os.path.join(dir, 'ovr_class{}_{}_distance{}-{}'.format(i, 'L2', 5.0, best[i])) for i in range(num_classes)]

  #dir = 'checkpoints/mnist/detector_Linf_0.3/ovr-steps100-adam-noclip-balanced/'
  #best = [91, 68, 76, 95, 71, 90, 89, 64, 54, 59]
  #detector_models = [os.path.join(dir, 'ovr_class{}_{}_distance{}-{}'.format(i, 'Linf', 0.3, best[i])) for i in range(num_classes)]

  #dir = 'checkpoints/mnist/detector_Linf_0.5/ovr-steps200-adam-noclip-balanced/'
  #best = [86, 39, 30, 28, 28, 29, 72, 63, 67, 37]
elif args.epsilon == 2.5:
  print('args.epsilon: {}'.format(args.epsilon))
  dir = 'checkpoints/mnist/detector_L2_2.5/ovr-steps100-adam-noclip-balanced/'
  best = [50,80,51,83,95,99,50,91,72,66]
  detector_models = [os.path.join(dir, 'ovr_class{}_{}_distance{}-{}'.format(i, 'L2', 2.5, best[i])) for i in range(num_classes)]

  #dir = 'checkpoints/mnist/detector_L2_5.0/ovr-steps200-adam-noclip-balanced'
  #best = [57,99,77,100,43,93,80,98,88,59]
  #detector_models = [os.path.join(dir, 'ovr_class{}_{}_distance{}-{}'.format(i, 'L2', 5.0, best[i])) for i in range(num_classes)]

  #dir = 'checkpoints/mnist/detector_Linf_0.3/ovr-steps100-adam-noclip-balanced/'
  #best = [91, 68, 76, 95, 71, 90, 89, 64, 54, 59]
  #detector_models = [os.path.join(dir, 'ovr_class{}_{}_distance{}-{}'.format(i, 'Linf', 0.3, best[i])) for i in range(num_classes)]

  #dir = 'checkpoints/mnist/detector_Linf_0.5/ovr-steps100-adam-noclip-balanced/'
  #best = [97, 39, 30, 28, 28, 29, 72, 63, 67, 37]
  #detector_models = [os.path.join(dir, 'ovr_class{}_{}_distance{}-{}'.format(i, 'Linf', 0.5, best[i])) for i in range(num_classes)]
elif args.epsilon == 5.0:
  print('args.epsilon: {}'.format(args.epsilon))
  #dir = 'checkpoints/mnist/detector_L2_5.0/ovr-steps200-adam-noclip-balanced/'
  #best = [88, 49, 100, 98, 88, 94, 93, 62, 29, 56]
  dir = 'checkpoints/mnist/detector_L2_5.0/ovr-steps200-adam-noclip-balanced'
  best = [57,99,77,100,43,93,80,98,88,59]
  detector_models = [os.path.join(dir, 'ovr_class{}_{}_distance{}-{}'.format(i, 'L2', 5.0, best[i])) for i in range(num_classes)]

  ##dir = 'checkpoints/mnist/detector_Linf_0.3/ovr-steps100-adam-noclip-balanced/'
  ##best = [91, 68, 76, 95, 71, 90, 89, 64, 54, 59]
  ##detector_models = [os.path.join(dir, 'ovr_class{}_{}_distance{}-{}'.format(i, 'Linf', 0.3, best[i])) for i in range(num_classes)]
  #dir = 'checkpoints/mnist/detector_Linf_0.5/ovr-steps100-adam-noclip-balanced/'
  #best = [97, 39, 30, 28, 28, 29, 72, 63, 67, 37]
  #detector_models = [os.path.join(dir, 'ovr_class{}_{}_distance{}-{}'.format(i, 'Linf', 0.5, best[i])) for i in range(num_classes)]

  #dir = 'checkpoints/mnist/detector_L2_2.5/ovr-steps100-adam-noclip-balanced/'
  #best = [50,80,51,83,95,99,50,91,72,66]
  #detector_models = [os.path.join(dir, 'ovr_class{}_{}_distance{}-{}'.format(i, 'L2', 2.5, best[i])) for i in range(num_classes)]


#detector_models = [os.path.join(dir, 'ovr_class{}_{}_distance{}-{}'.format(i, args.norm, args.epsilon, best[i])) for i in range(num_classes)]

detectors, detector_savers = [], []
for i in range(num_classes):
  scope = 'detector-class{}'.format(i)
  detectors.append(Detector(var_scope=scope, dataset='MNIST'))
  detector_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
  detector_savers.append(tf.train.Saver(var_list=detector_vars))

if args.method == 'detector-adv':
  AUC = []
  best_adv = np.zeros_like(x_test)
  best_logit = np.zeros(x_test.shape[0]) 
  best_logit.fill(-np.inf)
  for attack_class in range(0, 10):
    x_test_target = x_test[y_test == attack_class]
    x_test_others = x_test[y_test != attack_class]
    y_test_others = y_test[y_test != attack_class]
    
    best_adv_others = best_adv[y_test != attack_class]
    best_logit_others = best_logit[y_test != attack_class]

    #mask = np.bitwise_and(y_test != 0, y_test != 1)
    #mask = np.bitwise_and(y_test != 2, mask).astype(np.bool)
    #x_test_others = x_test[mask]
    #y_test_others = y_test[mask]

    detector = detectors[attack_class]
    attack = PGDAttackDetector(detector=detector,
                               max_distance=args.epsilon,
                               num_steps=args.steps, step_size=args.step_size,
                               random_start=False, x_min=x_min, x_max=x_max,
                               batch_size=x_test_others.shape[0],
                               norm=args.norm, optimizer=args.optimizer)

    with tf.Session() as sess:
      saver.restore(sess, 'checkpoints/mnist/classifier')
      detector_savers[attack_class].restore(sess, detector_models[attack_class])

      x_test_others_adv = attack.perturb(x_test_others, y_test_others, sess)
      logits, y_pred = sess.run([detector.logits, detector.predictions], feed_dict={detector.x_input: x_test_others_adv})

      y_adv = np.zeros(x_test_others_adv.shape[0], np.int64)
      fpr_, tpr_, thresholds = roc_curve(y_adv, logits)
      roc_auc = auc(fpr_, tpr_)
      print('auc {}'.format(roc_auc))
      update_mask = logits > best_logit_others
      best_logit_others[update_mask] = logits[update_mask]
      best_adv_others[update_mask] = x_test_others_adv[update_mask]
      
      best_adv[y_test != attack_class] = best_adv_others
      best_logit[y_test != attack_class] = best_logit_others
     
      continue
  np.savez('best_adv.npz', best_adv=best_adv, best_logit=best_logit, y_test=y_test)
  print('saved')
else:
  data = np.load('best_adv.npz')
  x_test = data['best_adv']
  best_logit = data['best_logit']
  y_test = data['y_test']
  print(best_logit)
  if True:
    madry_attack = MadryPGDAttackDetector(target_class=0,
                               max_distance=args.epsilon,
                               num_steps=args.steps, step_size=args.step_size,
                               random_start=False, x_min=x_min, x_max=x_max,
                               batch_size=10000,
                               norm=args.norm, optimizer=args.optimizer)
    madry_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='madry')
    madry_saver = tf.train.Saver(var_list=madry_vars, max_to_keep=1)
    with tf.Session() as sess:
      madry_saver.restore(sess, '/home/xy4cm/Projects/tmp/mnist_challenge/models/adv_trained_madry/checkpoint-99900')
      sess.run([madry_attack.assign_delta, madry_attack.assign_x0, madry_attack.assign_y], feed_dict={
        madry_attack.delta_input: np.zeros_like(x_test), madry_attack.x0_input: x_test, madry_attack.y_input: y_test})
      acc = sess.run(madry_attack.accuracy)
      print('==================== loaded madry classifier, acc {}'.format(acc))

