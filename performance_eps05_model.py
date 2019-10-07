import argparse
import sys
import os
import pandas as pd
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report


from models import Detector, Classifier, MadryClassifier
from models import PGDAttackDetector, PGDAttackClassifier, PGDAttackAda, PGDAttackBayesianOVR, BayesClassifier, PGDAttackCombined, MadryPGDAttackClassifier
from utils import *
import matplotlib.pyplot as plt
import json

parser = argparse.ArgumentParser()
# parser.add_argument('--target_class', type=int, required=True)
parser.add_argument('--epsilon', metavar='max-distance', default=0.3, type=float)
parser.add_argument('--norm', choices=['L2', 'Linf'], default='Linf')
parser.add_argument('--optimizer', choices=['adam', 'normgrad'], default='adam')
parser.add_argument('--steps', type=int, default=100)
parser.add_argument('--step_size', type=float, default=0.01)
parser.add_argument('--method', choices=['two', 'bayes_classifier', 'ovr', 'bayesian-ovr', 'bayesian-ovr-adv', 'nat', 'static-adv', 'detector-adv', 'adaptive-adv'], default='two')
parser.add_argument('--tag', type=str, default='tag')
parser.add_argument('--c', type=float, default=0)
parser.add_argument('--gen', action='store_true')
parser.add_argument('--plot', action='store_true')
parser.add_argument('--test_madry', action='store_true')
parser.add_argument('--min_dist', action='store_true')
parser.add_argument('--adv_tiling', action='store_true')


args = parser.parse_args()
print(args)
arg_str = '-'.join(['{}={}'.format(k, v) for k, v in vars(args).items()])

np.random.seed(123)


mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = np.reshape(x_train, [x_train.shape[0], -1])
x_test = np.reshape(x_test, [x_test.shape[0], -1])

x_min, x_max = 0.0, 1.0

if args.test_madry or args.adv_tiling:
  classifier = MadryClassifier(var_scope='classifier')
else:
  classifier = Classifier(var_scope='classifier', dataset='MNIST')

vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='classifier')
saver = tf.train.Saver(var_list=vars, max_to_keep=1)

num_classes = 10

dir = 'checkpoints/mnist/detector_Linf_0.3/ovr-steps100-adam-noclip-balanced/'
best = [91, 68, 76, 95, 71, 90, 89, 64, 54, 59]
detector_models_eps03 = [os.path.join(dir, 'ovr_class{}_{}_distance{}-{}'.format(i, args.norm, args.epsilon, best[i])) for i in range(num_classes)]

dir = 'checkpoints/mnist/detector_Linf_0.5/ovr-steps100-adam-noclip-balanced/'
best = [97, 39, 30, 28, 28, 29, 72, 63, 67, 37]
detector_models_eps05 = [os.path.join(dir, 'ovr_class{}_{}_distance{}-{}'.format(i, 'Linf', 0.5, best[i])) for i in range(num_classes)]


detectors, detector_savers = [], []
for i in range(num_classes):
  scope = 'detector-class{}'.format(i)
  detectors.append(Detector(var_scope=scope, dataset='MNIST'))
  detector_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
  detector_savers.append(tf.train.Saver(var_list=detector_vars))

with tf.Session() as sess:
    
  if isinstance(classifier, MadryClassifier):
    saver.restore(sess, 'checkpoints/mnist/adv_trained_prefixed_classifier/checkpoint-99900')
  else:
    saver.restore(sess, 'checkpoints/mnist/classifier')

  nat_acc, y_pred_test = sess.run([classifier.accuracy, classifier.predictions],
                                  feed_dict={classifier.x_input: x_test,
                                             classifier.y_input: y_test})
  print('classifier nat acc {}'.format(nat_acc))


  bayes_classifier = BayesClassifier(detectors)

  attack_config = {'max_distance': args.epsilon, 'num_steps': args.steps, 'step_size': args.step_size, 'random_start': False,
          'x_min': 0, 'x_max': 1.0, 'batch_size': x_test.shape[0]//2, 'optimizer': args.optimizer, 'norm': args.norm}
  print(attack_config)

  # ======== Classification performance of the integrated classifier and generative classifier ============
  plt.figure(figsize=(3.5*1.7,2*1.7))

  for i in range(num_classes):
    detector_savers[i].restore(sess, detector_models_eps03[i])
  bayes_nat_accs = bayes_classifier.nat_accs(x_test, y_test, sess)
  # x_test_adv = np.load(os.path.join('performance_data/eps0.3', 'x_test_adv_detector.npy'))

  attack = PGDAttackClassifier(classifier=bayes_classifier, loss_fn='cw', **attack_config)
  x_test_adv = attack.batched_perturb(x_test, y_test, sess)
  bayes_adv_error = bayes_classifier.adv_error(x_test_adv, y_test, sess)
  plt.plot(bayes_adv_error, bayes_nat_accs, label='Generative classifier', color='tab:blue')

  for i in range(num_classes):
    detector_savers[i].restore(sess, detector_models_eps05[i])
  bayes_nat_accs = bayes_classifier.nat_accs(x_test, y_test, sess)

  # x_test_adv = np.load(os.path.join('performance_data/eps0.3_eps0.5model', 'x_test_adv_detector.npy'))
  attack = PGDAttackClassifier(classifier=bayes_classifier, loss_fn='cw', **attack_config)
  x_test_adv = attack.batched_perturb(x_test, y_test, sess)
  bayes_adv_error = bayes_classifier.adv_error(x_test_adv, y_test, sess)
  plt.plot(bayes_adv_error, bayes_nat_accs, label='Generative classifier (use eps=0.5 base detectors)', color='tab:red')

  plt.annotate('robust classifier', xy=(0.08, 0.984),  xycoords='data', xytext=(0.08, 0.97),
          textcoords='data', arrowprops=dict(facecolor='black', shrink=0.2, width=1.5, headwidth=5, headlength=5),
          horizontalalignment='center', verticalalignment='center',) 
  # plt.annotate('robust classifier (eps=0.4)', xy=(0.941, 0.984),  xycoords='data', xytext=(0.85, 0.97),
  #         textcoords='data', arrowprops=dict(facecolor='black', shrink=0.2, width=1.5, headwidth=5, headlength=5),
  #         horizontalalignment='right', verticalalignment='center',) 
  plt.plot([0.08], [0.984], 'rx')
  #plt.xlim([-0.01, 0.5])
  plt.ylim([0.95, 0.993])
  plt.xlabel('Error on perturbed MNIST test set')
  plt.ylabel('Accuracy on MNSIT test set')
  plt.legend()
  plt.grid(True, alpha=0.5)
  plt.savefig('clf_mnist_eps05_new.pdf', bbox_inches='tight')
  plt.show()
  sys.exit(0)

