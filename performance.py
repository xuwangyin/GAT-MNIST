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
if args.epsilon == 0.3:
  print('args.epsilon: {}'.format(args.epsilon))
  dir = 'checkpoints/mnist/detector_Linf_0.3/ovr-steps100-adam-noclip-balanced/'
  best = [91, 68, 76, 95, 71, 90, 89, 64, 54, 59]
  detector_models = [os.path.join(dir, 'ovr_class{}_{}_distance{}-{}'.format(i, args.norm, args.epsilon, best[i])) for i in range(num_classes)]
  print('using models {}'.format(dir))

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
  print('using models {}'.format(dir))

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
# print('args.epsilon: {}'.format(args.epsilon))
# dir = 'checkpoints/mnist/detector_Linf_0.5/ovr-steps100-adam-noclip-balanced/'
# best = [97, 39, 30, 28, 28, 29, 72, 63, 67, 37]
# detector_models = [os.path.join(dir, 'ovr_class{}_{}_distance{}-{}'.format(i, 'Linf', 0.5, best[i])) for i in range(num_classes)]

#dir = 'checkpoints/mnist/detector_Linf_0.3/ovr-steps100-adam-noclip-balanced/'
#best = [91, 68, 76, 95, 71, 90, 89, 64, 54, 59]
#detector_models = [os.path.join(dir, 'ovr_class{}_{}_distance{}-{}'.format(i, 'Linf', 0.3, best[i])) for i in range(num_classes)]
#print('using models {}'.format(dir))

# dir = 'checkpoints/mnist/detector_Linf_0.5/ovr-steps100-adam-noclip-balanced/'
# best = [97, 39, 30, 28, 28, 29, 72, 63, 67, 37]
# detector_models = [os.path.join(dir, 'ovr_class{}_{}_distance{}-{}'.format(i, 'Linf', 0.5, best[i])) for i in range(num_classes)]

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
  for i in range(num_classes):
    detector_savers[i].restore(sess, detector_models[i])
  nat_logits = sess.run(bayes_classifier.logits, feed_dict={bayes_classifier.x_input: x_test})

  attack_config = {'max_distance': args.epsilon, 'num_steps': args.steps, 'step_size': args.step_size, 'random_start': False,
          'x_min': 0, 'x_max': 1.0, 'batch_size': x_test.shape[0]//2, 'optimizer': args.optimizer, 'norm': args.norm}
  print(attack_config)

  if args.test_madry:
    attack = PGDAttackClassifier(classifier=classifier, loss_fn='cw', **attack_config)
    x_test_adv = attack.batched_perturb(x_test, y_test, sess)
    nat_acc, y_pred_test = sess.run([classifier.accuracy, classifier.predictions],
                                    feed_dict={classifier.x_input: x_test_adv,
                                               classifier.y_input: y_test})
    print('classifier adv acc {}'.format(nat_acc))

  def get_det_logits(x, x_preds):
      """Compute detector logits for the input.

      First assign x to detectors based on the classifier output (x_preds), 
      then computes detector logit outputs.  
      """
      det_logits = np.zeros_like(x_preds)
      for classidx in range(10):
        assign = x_preds == classidx
        feed_dict = {detectors[classidx].x_input: x[assign]}
        det_logits[assign] = sess.run(detectors[classidx].logits, feed_dict=feed_dict)
      return det_logits

  def get_fpr(x_adv, y, logit_ths):
    """The portion of perturbed data samples that are adversarial (adv_preds != y) and
    at the same time successfully fool the detectors (det_logits > th)"""
    adv_preds = sess.run(classifier.predictions, feed_dict={classifier.x_input: x_adv})
    det_logits = get_det_logits(x_adv, adv_preds)
    print('adv logits min/max {}/{}'.format(det_logits.min(), det_logits.max()))
    fpr = [np.logical_and(det_logits > th, adv_preds != y).mean() for th in logit_ths]
    return fpr

  def get_adv_errors(x_adv, y, logit_ths):
    """With reject option, the naive classifier's error rate on perturbed test set.

    The error rate is computed as the portion of samples that are
    not rejected (det_logits > th) and at the same time
    causing misclassification (adv_preds != y)
    In other words, any samples that are rejected or
    corrected classified, are assumed to be properly handled.
    """
    adv_preds = sess.run(classifier.predictions, feed_dict={classifier.x_input: x_adv})
    det_logits = get_det_logits(x_adv, adv_preds)
    errors = [np.logical_and(det_logits > th, adv_preds != y).mean() for th in logit_ths]
    return errors

  def get_tpr(x_nat, logit_ths):
    """Recall on the set of original data set"""
    nat_preds = sess.run(classifier.predictions, feed_dict={classifier.x_input: x_nat})
    det_logits = get_det_logits(x_nat, nat_preds)
    print('nat logits min/max {}/{}'.format(det_logits.min(), det_logits.max()))
    tpr = [(det_logits > th).mean() for th in logit_ths]
    return tpr

  def get_nat_accs(x_nat, y, logit_ths):
    """Accuracy on the natural data set"""
    nat_preds = sess.run(classifier.predictions, feed_dict={classifier.x_input: x_nat})
    det_logits = get_det_logits(x_nat, nat_preds)
    accs = [(np.logical_and(det_logits > th, nat_preds == y)).mean() for th in logit_ths] 
    return accs

  datadir = 'performance_data/eps{}'.format(args.epsilon)
  # datadir = 'performance_data/eps0.3_eps0.5model'
  Path(datadir).mkdir(parents=True, exist_ok=True)
  files = {'x_test_adv_combined.npy': 'Integrated detection (combined attack)',
           'x_test_adv_combined_cw.npy': 'Integrated detection (combined attack cw loss)',
           'x_test_adv_detector.npy': 'Integrated detection (detectors attack)',
           'x_test_adv_classifier.npy': 'Integrated detection (classifier attack)'}

  def update_bound(lower, upper, current, success):
    if success:
      lower = current
    else:
      upper = current
    return lower, upper, (lower + upper) * 0.5
      

  if args.min_dist:
    batch_size = 100
    attack_config = {'max_distance': 100, 'num_steps': 500, 'step_size': 0.1, 'random_start': False,
            'x_min': 0, 'x_max': 1.0, 'batch_size': batch_size, 'optimizer': args.optimizer, 'norm': 'L2'}
    attack = PGDAttackClassifier(classifier=bayes_classifier, loss_fn='cw', **attack_config)
    best_dists = []
    best_adv = []
    for b in range(0, x_test.shape[0], batch_size):
      print('processing batch {}-{}'.format(b, b+batch_size))
      x_batch, y_batch = x_test[b: b+batch_size], y_test[b: b+batch_size]
      lowers, uppers = np.zeros(batch_size), np.zeros(batch_size) + 8.0

      # # validate upper bound
      # c_constants = np.zeros(batch_size) + 8
      # x_batch_adv = attack.perturb(x_batch, y_batch, sess, c_constants=c_constants)
      # adv_logits = sess.run(bayes_classifier.logits, feed_dict={bayes_classifier.x_input: x_batch_adv})
      # success = np.logical_and(np.argmax(adv_logits, 1) != y_batch, np.max(adv_logits, 1) > 3.6)
      # print('sucess {}'.format(success.mean()))
      # continue


      c_constants = np.zeros(batch_size)
      batch_best_dists = np.zeros(batch_size) + 1e9
      batch_best_adv = np.zeros_like(x_batch)
      for depth in range(20):
        x_batch_adv = attack.perturb(x_batch, y_batch, sess, c_constants=c_constants)
        adv_logits = sess.run(bayes_classifier.logits, feed_dict={bayes_classifier.x_input: x_batch_adv})
        success = np.logical_and(np.argmax(adv_logits, 1) != y_batch, np.max(adv_logits, 1) > 3.6)
        dist = np.linalg.norm(x_batch_adv - x_batch, axis=1)
        for i in range(adv_logits.shape[0]):
          if i == 0:
            print('lower {}, upper {}, c_constant {} dist {}, sucess {}'.format(lowers[i], uppers[i], c_constants[i], dist[i], success[i]))
          lowers[i], uppers[i], c_constants[i] = update_bound(lowers[i], uppers[i], c_constants[i], success[i])
          if success[i] and dist[i] < batch_best_dists[i]:
            batch_best_dists[i] = dist[i]
            batch_best_adv[i] = x_batch_adv[i]
      print('sucess {} dist mean {}'.format((batch_best_dists<1e8).mean(), batch_best_dists[batch_best_dists<1e8].mean()))
      best_dists.append(batch_best_dists)
      best_adv.append(batch_best_adv)
    best_dists = np.concatenate(best_dists)
    np.savez(os.path.join(datadir, 'best_dists_eps0.5_model.npz'), best_dists=best_dists, best_adv=best_adv) 
    print('sucess {}, dist mean {}'.format((best_dists < 1e5).mean(), best_dists[best_dists<1e5].mean()))

      #print('c_constant: {}'.format(c_constant))
      #for i in range(dist.shape[0]):
      #  print('dist {} pred {} y{}'.format(dist[i], preds[i], y_subset[i]))


  if args.gen:
    # # classifier attack
    # attack = PGDAttackClassifier(classifier=classifier, loss_fn='cw', **attack_config)
    # x_test_adv = attack.batched_perturb(x_test, y_test, sess)
    # np.save(os.path.join(datadir, 'x_test_adv_classifier.npy'), x_test_adv)

    # detector attack
    attack = PGDAttackClassifier(classifier=bayes_classifier, loss_fn='cw', **attack_config)
    x_test_adv = attack.batched_perturb(x_test, y_test, sess)
    np.save(os.path.join(datadir, 'x_test_adv_detector.npy'), x_test_adv)

    # # detector attack
    # attack = PGDAttackClassifier(classifier=bayes_classifier, loss_fn='xent', **attack_config)
    # x_test_adv = attack.batched_perturb(x_test, y_test, sess)
    # np.save(os.path.join(datadir, 'x_test_adv_detector_xent.npy'), x_test_adv)

    # combined attack
    attack = PGDAttackCombined(classifier=classifier, bayes_classifier=bayes_classifier, loss_fn='yin', **attack_config)
    x_test_adv = attack.batched_perturb(x_test, y_test, sess)
    np.save(os.path.join(datadir, 'x_test_adv_combined.npy'), x_test_adv)

    # # combined attack using cw loss
    # attack = PGDAttackCombined(classifier=classifier, bayes_classifier=bayes_classifier, loss_fn='cw', **attack_config)
    # x_test_adv = attack.batched_perturb(x_test, y_test, sess)
    # np.save(os.path.join(datadir, 'x_test_adv_combined_cw.npy'), x_test_adv)


  if args.plot: 
    logit_ths = np.linspace(-250., 50.0, 1000)
    # logit_ths = np.linspace(0., 1.0, 200)
    combined_tpr = get_tpr(x_test, logit_ths)

    # ======== Detection performance of the combined system using different attack methods ============
    plt.figure(figsize=(3.5*1.7,2*1.7))
    for filename, label in files.items():
      x_test_adv = np.load(os.path.join(datadir, filename))
      fpr = get_fpr(x_test_adv, y_test, logit_ths)
      plt.plot(fpr, combined_tpr, label=label)

    # # roc of combined system
    # x_test_adv_combined = np.load(os.path.join(datadir, 'x_test_adv_combined.npy'))
    # combined_fpr = get_fpr(x_test_adv_combined, y_test, logit_ths)
    # plt.plot(combined_fpr, combined_tpr, label='Detection with combined system')

    # roc of Bayes classifier
    x_test_adv_bayes = np.load(os.path.join(datadir, 'x_test_adv_detector.npy'))
    bayes_tpr = bayes_classifier.nat_tpr(x_test, sess)
    bayes_fpr = bayes_classifier.adv_fpr(x_test_adv_bayes, y_test, sess)
    plt.plot(bayes_fpr, bayes_tpr, label='Generative detection')
    plt.ylim([0.9, 1.0])
    plt.xlim([0.0, 0.5])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.savefig('det_mnist.pdf', bbox_inches='tight')

    # ======== Generative detection using different attack methods ============
    plt.figure(figsize=(3.5*1.2,2*1.2))
    # roc of generative detection
    x_test_adv_bayes = np.load(os.path.join(datadir, 'x_test_adv_detector.npy'))
    bayes_tpr = bayes_classifier.nat_tpr(x_test, sess)
    bayes_fpr = bayes_classifier.adv_fpr(x_test_adv_bayes, y_test, sess)
    plt.plot(bayes_fpr, bayes_tpr, label='Generative detection')
    # roc of generative detection xent
    x_test_adv_bayes = np.load(os.path.join(datadir, 'x_test_adv_detector_xent.npy'))
    bayes_tpr = bayes_classifier.nat_tpr(x_test, sess)
    bayes_fpr = bayes_classifier.adv_fpr(x_test_adv_bayes, y_test, sess)
    plt.plot(bayes_fpr, bayes_tpr, label='Generative detection (xent loss)')
    # roc of generative detection cw
    x_test_adv_bayes = np.load(os.path.join(datadir, 'x_test_adv_detector_cw.npy'))
    bayes_tpr = bayes_classifier.nat_tpr(x_test, sess)
    bayes_fpr = bayes_classifier.adv_fpr(x_test_adv_bayes, y_test, sess)
    plt.plot(bayes_fpr, bayes_tpr, label='Generative detection (cw loss)')

    plt.ylim([0.9, 1.0])
    plt.xlabel('False positive rate on perturbed MNIST test set')
    plt.ylabel('True positive rate on MNIST test set')
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.savefig('det_bayes_compare_mnist.pdf', bbox_inches='tight')


    # ======== Generative classification using different attack methods ============
    plt.figure(figsize=(3.5*1.2,2*1.2))
    # roc of generative detection
    x_test_adv_bayes = np.load(os.path.join(datadir, 'x_test_adv_detector.npy'))
    bayes_nat_accs = bayes_classifier.nat_accs(x_test, y_test, sess)
    bayes_adv_errors = bayes_classifier.adv_error(x_test_adv_bayes, y_test, sess)
    plt.plot(bayes_adv_errors, bayes_nat_accs, label='Generative classifier')
    # roc of generative detection xent
    x_test_adv_bayes = np.load(os.path.join(datadir, 'x_test_adv_detector_xent.npy'))
    bayes_tpr = bayes_classifier.nat_accs(x_test, y_test, sess)
    bayes_adv_errors = bayes_classifier.adv_error(x_test_adv_bayes, y_test, sess)
    plt.plot(bayes_adv_errors, bayes_tpr, label='Generative classifier (xent loss)')
    # roc of generative detection cw
    x_test_adv_bayes = np.load(os.path.join(datadir, 'x_test_adv_detector_cw.npy'))
    bayes_tpr = bayes_classifier.nat_accs(x_test, y_test, sess)
    bayes_adv_errors = bayes_classifier.adv_error(x_test_adv_bayes, y_test, sess)
    plt.plot(bayes_adv_errors, bayes_tpr, label='Generative classifier (cw loss)')

    plt.ylim([0.9, 1.0])
    plt.xlabel('Error on perturbed MNIST test set')
    plt.ylabel('Accuracy on MNSIT test set')
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.savefig('clf_bayes_compare_mnist.pdf', bbox_inches='tight')


    # ======== Classification performance of the integrated classifier and generative classifier ============
    plt.figure(figsize=(3.5*1.7,2*1.7))
    combined_nat_accs = get_nat_accs(x_test, y_test, logit_ths)
    x_test_adv_combined = np.load(os.path.join(datadir, 'x_test_adv_combined.npy'))
    combined_adv_errors = get_adv_errors(x_test_adv_combined, y_test, logit_ths)
    plt.plot(combined_adv_errors, combined_nat_accs, label='Integrated classifier')

    bayes_nat_accs = bayes_classifier.nat_accs(x_test, y_test, sess)
    x_test_adv = np.load(os.path.join(datadir, 'x_test_adv_detector.npy'))
    bayes_adv_error = bayes_classifier.adv_error(x_test_adv, y_test, sess)
    plt.plot(bayes_adv_error, bayes_nat_accs, label='Generative classifier')

    bayes_nat_accs = bayes_classifier.nat_accs(x_test, y_test, sess)
    x_test_adv = np.load(os.path.join(datadir.replace('0.3', '0.4'), 'x_test_adv_detector.npy'))
    bayes_adv_error = bayes_classifier.adv_error(x_test_adv, y_test, sess)
    plt.plot(bayes_adv_error, bayes_nat_accs, label='Generative classifier (eps=0.4)')
 
    # x_test_adv_bayes = np.load(os.path.join('performance_data/eps0.3_eps0.5model', 'x_test_adv_detector.npy'))
    # # bayes_tpr = bayes_classifier.nat_tpr(x_test, sess)
    # # bayes_fpr = bayes_classifier.adv_fpr(x_test_adv_bayes, y_test, sess)
    # # np.savez('performance_data/bayes_0.5_model.npz', bayes_tpr=bayes_tpr, bayes_fpr=bayes_fpr)
    # data = np.load('performance_data/bayes_0.5_model.npz')
    # plt.plot(data['bayes_fpr'], data['bayes_tpr'], label='Detection with Bayes classifier (eps 0.5 trained)')

    plt.annotate('robust classifier', xy=(0.08, 0.984),  xycoords='data', xytext=(0.15, 0.984),
            textcoords='data', arrowprops=dict(facecolor='black', shrink=0.2, width=1.5, headwidth=5, headlength=5),
            horizontalalignment='left', verticalalignment='center',) 
    plt.annotate('robust classifier (eps=0.4)', xy=(0.941, 0.984),  xycoords='data', xytext=(0.85, 0.97),
            textcoords='data', arrowprops=dict(facecolor='black', shrink=0.2, width=1.5, headwidth=5, headlength=5),
            horizontalalignment='right', verticalalignment='center',) 
    plt.plot([0.08], [0.984], 'rx')
    plt.plot([0.941], [0.984], 'rx')
    #plt.xlim([-0.01, 0.5])
    plt.ylim([0.95, 0.993])
    plt.xlabel('Error on perturbed MNIST test set')
    plt.ylabel('Accuracy on MNSIT test set')
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.savefig('clf_mnist.pdf', bbox_inches='tight')

    plt.figure(figsize=(3.5*1.7,2*1.7))

    bayes_nat_accs = bayes_classifier.nat_accs(x_test, y_test, sess)
    x_test_adv = np.load(os.path.join(datadir.replace('0.3', '0.4'), 'x_test_adv_detector.npy'))
    bayes_adv_error = bayes_classifier.adv_error(x_test_adv, y_test, sess)
    plt.plot(bayes_adv_error, bayes_nat_accs, label='Generative classifier (eps=0.4)')

    bayes_nat_accs = bayes_classifier.nat_accs(x_test, y_test, sess)
    x_test_adv = np.load(os.path.join(datadir, 'x_test_adv_detector.npy'))
    bayes_adv_error = bayes_classifier.adv_error(x_test_adv, y_test, sess)
    plt.plot(bayes_adv_error, bayes_nat_accs, label='Generative classifier (eps=0.3)')

    bayes_nat_accs = bayes_classifier.nat_accs(x_test, y_test, sess)
    x_test_adv = np.load(os.path.join(datadir.replace('0.3', '0.5'), 'x_test_adv_detector.npy'))
    bayes_adv_error = bayes_classifier.adv_error(x_test_adv, y_test, sess)
    plt.plot(bayes_adv_error, bayes_nat_accs, label='Generative classifier (eps=0.5)')

    plt.annotate('Adversarial trained classifier (eps=0.4)', xy=(0.941, 0.984),  xycoords='data', xytext=(0.85, 0.97),
            textcoords='data', arrowprops=dict(facecolor='black', shrink=0.2, width=1.5, headwidth=5, headlength=5),
            horizontalalignment='right', verticalalignment='center',) 
    plt.plot([0.941], [0.984], 'rx')
    plt.xlim([0.00, 1.0])
    plt.ylim([0.95, 0.993])
    plt.xlabel('Error on perturbed MNIST test set')
    plt.ylabel('Accuracy on MNSIT test set')
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.savefig('clf_bayes_eps_mnist.pdf', bbox_inches='tight')
    plt.show()


  if args.adv_tiling:
    # plot deformation tiling
    EPS = 0.4
    x_test_adv_bayes = np.load(os.path.join(datadir.replace('0.3', str(EPS)), 'x_test_adv_detector.npy'))
    logits = sess.run(bayes_classifier.logits, feed_dict={bayes_classifier.x_input: x_test_adv_bayes})
    max_logits = np.max(logits, 1)
    preds = np.argmax(logits, 1)
    perturbed_samples = np.zeros((10, 10, 28*28), dtype=np.float32)
    original_samples = np.zeros((10, 10, 28*28), dtype=np.float32)
    for classidx in range(10):
      submask = np.logical_and(preds == classidx, y_test != classidx)
      original_sub_samples = x_test[submask]
      perturbed_sub_samples = x_test_adv_bayes[submask]
      sub_logits = max_logits[submask]
      top_indices = np.argsort(sub_logits)[::-1][:10]
      print(np.round(sub_logits[top_indices], 1))
      perturbed_samples[classidx] = np.take(perturbed_sub_samples, top_indices, axis=0)
      original_samples[classidx] = np.take(original_sub_samples, top_indices, axis=0)

    assert isinstance(classifier, MadryClassifier)
    attack_config['batch_size'] = 100
    attack_config['max_distance'] = EPS
    attack = PGDAttackClassifier(classifier=classifier, loss_fn='cw', targeted=True, **attack_config)
    targets = np.concatenate([np.zeros(10, dtype=np.int64) + i for i in range(10)])
    robust_perturbed_samples = attack.perturb(original_samples.reshape([100, 28*28]), targets, sess)
    robust_perturbed_samples = robust_perturbed_samples.reshape([10, 10, 28*28])

    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10, 4))
    bins = np.linspace(-20, 30, 100)

    c0_nat_logits_generative = sess.run(detectors[0].logits, feed_dict={detectors[0].x_input: x_test[y_test==0]})
    c0_adv_logits_generative = sess.run(detectors[0].logits, feed_dict={detectors[0].x_input: perturbed_samples[0]})
    axes[0].hist(c0_nat_logits_generative, bins=bins, color='tab:blue', label='logit outputs of natural samples of class 1')
    axes[0].hist(c0_adv_logits_generative, bins=bins, color='tab:red', label='logit outputs of generated samples')
    axes[0].set_title('generative classifier')
    axes[0].legend()

    c0_nat_logits_robust = sess.run(classifier.logits, feed_dict={classifier.x_input: x_test[y_test==0]})
    c0_adv_logits_robust = sess.run(classifier.logits, feed_dict={classifier.x_input: robust_perturbed_samples[0]})
    axes[1].hist(c0_nat_logits_robust[:,0], bins=bins, color='tab:blue', label='logit outputs of natural samples of class 1')
    axes[1].hist(c0_adv_logits_robust[:,0], bins=bins, color='tab:red', label='logit outputs of generated samples')
    axes[1].set_title('robust classifier')
    axes[1].legend()
    plt.savefig('adv_tiling_hist_mnist.pdf')
    plt.show()

    nrows = ncols = 10
    dim = 28
    pad = 1
    space = dim + pad

    fig, ax = plt.subplots(1, 3, figsize=(11, 4))
    tiling = np.ones((space * nrows, space * ncols), dtype=np.float32)
    for row in range(nrows):
      for col in range(ncols):
        tiling[row*space: row*space+dim, col*space: col*space+dim] = original_samples[row, col].reshape((28, 28))
    
    ax[0].imshow(tiling, cmap='gray')
    ax[0].axis('off')
    ax[0].set_title('Natural samples')

    tiling = np.ones((space * nrows, space * ncols), dtype=np.float32)
    for row in range(nrows):
      for col in range(ncols):
        tiling[row*space: row*space+dim, col*space: col*space+dim] = perturbed_samples[row, col].reshape((28, 28))
    
    ax[1].imshow(tiling, cmap='gray')
    ax[1].axis('off')
    ax[1].set_title('Perturbed samples (generative classifier)')

    tiling = np.ones((space * nrows, space * ncols), dtype=np.float32)
    for row in range(nrows):
      for col in range(ncols):
        tiling[row*space: row*space+dim, col*space: col*space+dim] = robust_perturbed_samples[row, col].reshape((28, 28))
    
    ax[2].imshow(tiling, cmap='gray')
    ax[2].axis('off')
    ax[2].set_title('Perturbed samples (robust classifier)')


    for axes in ax:
      axes.xaxis.set_major_locator(plt.NullLocator())
      axes.yaxis.set_major_locator(plt.NullLocator())
    # plt.subplots_adjust(top = 1, bottom = 0, right = 0, left = 1, 
    #                     hspace = 0, wspace = 0)
    plt.subplots_adjust(top = 1, bottom = 0, right=1, left=0, hspace = 0, wspace = 0.1)
    #plt.show()
    plt.savefig('mnist_tiling_eps.pdf'.format(EPS), dpi=500, bbox_inches='tight', pad_inches=0)
