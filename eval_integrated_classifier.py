import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from models import Classifier, BayesClassifier, PGDAttackClassifier, PGDAttackCombined
from eval_utils import BaseDetectorFactory, load_mnist_data
from eval_utils import get_adv_errors, get_nat_accs

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

np.random.seed(123)

(x_train, y_train), (x_test, y_test) = load_mnist_data()

classifier = Classifier(var_scope='classifier', dataset='MNIST')
classifier_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='classifier')
classifier_saver = tf.train.Saver(var_list=classifier_vars, max_to_keep=1)
factory = BaseDetectorFactory(eps=0.3)

attack_config = {'max_distance': 0.3, 'num_steps': 100, 'step_size': 0.01, 'random_start': False,
        'x_min': 0, 'x_max': 1.0, 'batch_size': x_test.shape[0]//2, 'optimizer': 'adam', 'norm': 'Linf'}
print('attack config: {}'.format(attack_config))

datadir = 'performance_data/eps0.3'
plt.figure(figsize=(3.5*1.7,2*1.7))

with tf.Session() as sess:
  # Restore variables
  classifier_saver.restore(sess, 'checkpoints/mnist/classifier')
  factory.restore_base_detectors(sess)
  base_detectors = factory.get_base_detectors()

  bayes_classifier = BayesClassifier(base_detectors)

  for loss_fn in ['yoyo', 'cw']:
    logit_ths = np.linspace(-250., 50.0, 1000)
    attack = PGDAttackCombined(classifier=classifier, bayes_classifier=bayes_classifier, loss_fn=loss_fn, **attack_config)
    x_test_adv = attack.batched_perturb(x_test, y_test, sess)
    nat_accs = get_nat_accs(x_test, y_test, logit_ths, classifier, base_detectors, sess)
    adv_errors = get_adv_errors(x_test_adv, y_test, logit_ths, classifier, base_detectors, sess)
    plt.plot(adv_errors, nat_accs, label='Integrated classifier ({} loss)'.format(loss_fn))

  plt.ylim([0.95, 0.993])
  plt.xlabel('Error on perturbed MNIST test set')
  plt.ylabel('Accuracy on MNSIT test set')
  plt.legend()
  plt.grid(True, alpha=0.5)
  plt.show()
