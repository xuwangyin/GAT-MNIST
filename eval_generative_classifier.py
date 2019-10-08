import sys
import os
import numpy as np
import tensorflow as tf
from models import MadryClassifier, BayesClassifier, PGDAttackClassifier
from eval_utils import BaseDetectorFactory, load_mnist_data
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

np.random.seed(123)

(x_train, y_train), (x_test, y_test) = load_mnist_data()

classifier = MadryClassifier(var_scope='classifier')
classifier_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='classifier')
classifier_saver = tf.train.Saver(var_list=classifier_vars, max_to_keep=1)
factory = BaseDetectorFactory(eps=0.3)

attack_config = {'max_distance': 0.3, 'num_steps': 100, 'step_size': 0.01, 'random_start': False,
        'x_min': 0, 'x_max': 1.0, 'batch_size': x_test.shape[0]//2, 'optimizer': 'adam', 'norm': 'Linf'}
print('attack config: {}'.format(attack_config))

datadir = 'performance_data/eps0.3'

with tf.Session() as sess:
  # Restore variables
  classifier_saver.restore(sess, 'checkpoints/mnist/adv_trained_prefixed_classifier/checkpoint-99900')
  factory.restore_base_detectors(sess)

  bayes_classifier = BayesClassifier(factory.get_base_detectors())

  # Evaluate robust classifier
  nat_acc = sess.run(classifier.accuracy,
                     feed_dict={classifier.x_input: x_test, classifier.y_input: y_test})
  print('robust classifier nat acc {}'.format(nat_acc))
  attack = PGDAttackClassifier(classifier=classifier, loss_fn='cw', **attack_config)
  x_test_adv = attack.batched_perturb(x_test, y_test, sess)
  adv_acc = sess.run(classifier.accuracy, 
                     feed_dict={classifier.x_input: x_test_adv, classifier.y_input: y_test})
  print('robust classifier adv acc {}'.format(adv_acc))

  plt.figure(figsize=(3.5*1.2,2*1.2))

  # Evaluate generative classifier
  attack = PGDAttackClassifier(classifier=bayes_classifier, loss_fn='cw', **attack_config)
  x_test_adv_bayes = attack.batched_perturb(x_test, y_test, sess)
  # x_test_adv_bayes = np.load(os.path.join(datadir, 'x_test_adv_detector.npy'))
  bayes_nat_accs = bayes_classifier.nat_accs(x_test, y_test, sess)
  bayes_adv_errors = bayes_classifier.adv_error(x_test_adv_bayes, y_test, sess)
  plt.plot(bayes_adv_errors, bayes_nat_accs, label='Generative classifier')

  # Evaluate generative classifier - use cross-entropy loss
  attack = PGDAttackClassifier(classifier=bayes_classifier, loss_fn='xent', **attack_config)
  x_test_adv_bayes = attack.batched_perturb(x_test, y_test, sess)
  # x_test_adv_bayes = np.load(os.path.join(datadir, 'x_test_adv_detector_xent.npy'))
  bayes_nat_accs = bayes_classifier.nat_accs(x_test, y_test, sess)
  bayes_adv_errors = bayes_classifier.adv_error(x_test_adv_bayes, y_test, sess)
  plt.plot(bayes_adv_errors, bayes_nat_accs, label='Generative classifier (xent loss)')

  plt.ylim([0.9, 1.0])
  plt.xlabel('Error on perturbed MNIST test set')
  plt.ylabel('Accuracy on MNSIT test set')
  plt.legend()
  plt.grid(True, alpha=0.5)
  plt.show()


