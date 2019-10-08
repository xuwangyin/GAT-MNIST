import os
import numpy as np
import tensorflow as tf
from models import Detector


def load_mnist_data():
  mnist = tf.keras.datasets.mnist
  
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  x_train, x_test = x_train / 255.0, x_test / 255.0
  x_train = np.reshape(x_train, [x_train.shape[0], -1])
  x_test = np.reshape(x_test, [x_test.shape[0], -1])
  return (x_train, y_train), (x_test, y_test)


def get_det_logits(x, x_preds, base_detectors, sess):
    """Compute detector logits for the input.

    First assign x to base detectors based on the classifier output (x_preds), 
    then computes detector logit outputs.  
    """
    det_logits = np.zeros_like(x_preds)
    for classidx in range(10):
      assign = x_preds == classidx
      feed_dict = {base_detectors[classidx].x_input: x[assign]}
      det_logits[assign] = sess.run(base_detectors[classidx].logits, feed_dict=feed_dict)
    return det_logits


def get_adv_errors(x_adv, y, logit_ths, classifier, base_detectors, sess):
  """With reject option, the naive classifier's error rate on perturbed test set.

  The error rate is computed as the portion of samples that are
  not rejected (det_logits > th) and at the same time
  causing misclassification (adv_preds != y)
  In other words, any samples that are rejected or
  corrected classified, are assumed to be properly handled.
  """
  adv_preds = sess.run(classifier.predictions, feed_dict={classifier.x_input: x_adv})
  det_logits = get_det_logits(x_adv, adv_preds, base_detectors, sess)
  errors = [np.logical_and(det_logits > th, adv_preds != y).mean() for th in logit_ths]
  return errors


def get_fpr(x_adv, y, logit_ths, classifier, base_detectors, sess):
  """The portion of perturbed data samples that are adversarial (adv_preds != y) and
  at the same time successfully fool the base detectors (det_logits > th)"""
  return get_adv_errors(x_adv, y, logit_ths, classifier, base_detectors, sess)


def get_nat_accs(x_nat, y, logit_ths, classifier, base_detectors, sess):
  """Accuracy on the natural data set"""
  nat_preds = sess.run(classifier.predictions, feed_dict={classifier.x_input: x_nat})
  det_logits = get_det_logits(x_nat, nat_preds, base_detectors, sess)
  accs = [(np.logical_and(det_logits > th, nat_preds == y)).mean() for th in logit_ths] 
  return accs


def get_tpr(x_nat, logit_ths, classifier, base_detectors, sess):
  """Recall on the set of original data set"""
  nat_preds = sess.run(classifier.predictions, feed_dict={classifier.x_input: x_nat})
  det_logits = get_det_logits(x_nat, nat_preds, base_detectors, sess)
  # print('nat logits min/max {}/{}'.format(det_logits.min(), det_logits.max()))
  tpr = [(det_logits > th).mean() for th in logit_ths]
  return tpr



def get_detector_ckpt(eps):
  assert eps in [0.3, 0.5, 2.5, 5.0]
  num_classes = 10
  if eps == 0.3:
    dir = 'checkpoints/mnist/detector_Linf_0.3/ovr-steps100-adam-noclip-balanced/'
    best = [91, 68, 76, 95, 71, 90, 89, 64, 54, 59]
    checkpoints = [os.path.join(dir, 'ovr_class{}_{}_distance{}-{}'.format(i, 'Linf', 0.3, best[i])) for i in range(num_classes)]
  elif eps == 0.5:
    dir = 'checkpoints/mnist/detector_Linf_0.5/ovr-steps100-adam-noclip-balanced/'
    best = [97, 39, 30, 28, 28, 29, 72, 63, 67, 37]
    checkpoints = [os.path.join(dir, 'ovr_class{}_{}_distance{}-{}'.format(i, 'Linf', 0.5, best[i])) for i in range(num_classes)]
  elif eps == 2.5:
    dir = 'checkpoints/mnist/detector_L2_2.5/ovr-steps100-adam-noclip-balanced/'
    best = [50,80,51,83,95,99,50,91,72,66]
    checkpoints = [os.path.join(dir, 'ovr_class{}_{}_distance{}-{}'.format(i, 'L2', 2.5, best[i])) for i in range(num_classes)]
  elif eps == 5.0:
    dir = 'checkpoints/mnist/detector_L2_5.0/ovr-steps200-adam-noclip-balanced'
    best = [57,99,77,100,43,93,80,98,88,59]
    checkpoints = [os.path.join(dir, 'ovr_class{}_{}_distance{}-{}'.format(i, 'L2', 5.0, best[i])) for i in range(num_classes)]
  return checkpoints


class BaseDetectorFactory:
  def __init__(self, eps):
    self.eps = eps
    self.__checkpoints = get_detector_ckpt(eps)
    self.__base_detectors = [] 
    self.__detector_savers = []
    self.num_classes = 10
    for i in range(self.num_classes):
      scope = 'detector-class{}'.format(i)
      self.__base_detectors.append(Detector(var_scope=scope, dataset='MNIST'))
      detector_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
      self.__detector_savers.append(tf.train.Saver(var_list=detector_vars))
    self.restored = False
  
  def restore_base_detectors(self, sess):
    for i in range(self.num_classes):
      self.__detector_savers[i].restore(sess, self.__checkpoints[i])
    self.restored = True

  def get_base_detectors(self):
    return self.__base_detectors

  
