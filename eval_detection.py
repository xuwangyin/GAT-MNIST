import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from models import Classifier, BayesClassifier, PGDAttackClassifier, PGDAttackCombined
from eval_utils import BaseDetectorFactory, load_mnist_data
from eval_utils import get_tpr, get_fpr

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

np.random.seed(123)

(x_train, y_train), (x_test, y_test) = load_mnist_data()

classifier = Classifier(var_scope='classifier', dataset='MNIST')
classifier_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                    scope='classifier')
classifier_saver = tf.train.Saver(var_list=classifier_vars, max_to_keep=1)
factory = BaseDetectorFactory(eps=0.3)

attack_config = {
    'max_distance': 0.3,
    'num_steps': 100,
    'step_size': 0.1,
    'random_start': False,
    'x_min': 0,
    'x_max': 1.0,
    'batch_size': x_test.shape[0] // 2,
    'optimizer': 'adam',
    'norm': 'Linf'
}
print('attack config: {}'.format(attack_config))

datadir = 'performance_data/eps0.3'
plt.figure(figsize=(3.5 * 1.7, 2 * 1.7))
logit_ths = np.linspace(-250., 50.0, 1000)

with tf.Session() as sess:
    # Restore variables
    classifier_saver.restore(sess, 'checkpoints/mnist/classifier')
    factory.restore_base_detectors(sess)
    base_detectors = factory.get_base_detectors()
    bayes_classifier = BayesClassifier(base_detectors)

    def compute_fpr(attack, sess):
        x_test_adv = attack.batched_perturb(x_test, y_test, sess)
        return get_fpr(x_test_adv, y_test, logit_ths, classifier,
                       base_detectors, sess)

    # Integrated detection
    tpr = get_tpr(x_test, logit_ths, classifier, base_detectors, sess)

    fpr = compute_fpr(
        PGDAttackClassifier(classifier=classifier,
                            loss_fn='cw',
                            **attack_config), sess)
    plt.plot(fpr, tpr, label='Integrated detection (classifier attack)')

    fpr = compute_fpr(
        PGDAttackClassifier(classifier=bayes_classifier,
                            loss_fn='cw',
                            **attack_config), sess)
    plt.plot(fpr, tpr, label='Integrated detection (detector attack)')

    fpr = compute_fpr(
        PGDAttackClassifier(classifier=bayes_classifier,
                            loss_fn='xent',
                            **attack_config), sess)
    plt.plot(fpr,
             tpr,
             label='Integrated detection (detector attack xent loss)')

    fpr = compute_fpr(
        PGDAttackCombined(classifier=classifier,
                          bayes_classifier=bayes_classifier,
                          loss_fn='cw',
                          **attack_config), sess)
    plt.plot(fpr, tpr, label='Integrated detection (combined attack cw loss)')

    fpr = compute_fpr(
        PGDAttackCombined(classifier=classifier,
                          bayes_classifier=bayes_classifier,
                          loss_fn='default',
                          **attack_config), sess)
    plt.plot(fpr,
             tpr,
             label='Integrated detection (combined attack)',
             linewidth=2)

    # Generative detection
    bayes_tpr = bayes_classifier.nat_tpr(x_test, sess)
    attack = PGDAttackClassifier(classifier=bayes_classifier,
                                 loss_fn='cw',
                                 **attack_config)
    x_test_adv = attack.batched_perturb(x_test, y_test, sess)
    bayes_fpr = bayes_classifier.adv_fpr(x_test_adv, y_test, sess)
    plt.plot(bayes_fpr, bayes_tpr, label='Generative detection', linewidth=2)

    plt.ylim([0.9, 1.0])
    plt.xlim([0.0, 0.5])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.show()
