import argparse
import sys
import os
import pandas as pd
#from cleverhans.attacks import FastGradientMethod
#from cleverhans.attacks import CarliniWagnerL2
#from cleverhans.attacks import SPSA
#from cleverhans.model import Model


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report


from models import Detector, Classifier, MadryClassifier
from models import PGDAttackDetector, PGDAttackClassifier, PGDAttackAda, PGDAttackBayesianOVR, BayesianOVR
from utils import *
import matplotlib.pyplot as plt
import json

parser = argparse.ArgumentParser()
# parser.add_argument('--target_class', type=int, required=True)
parser.add_argument('--epsilon', metavar='max-distance', default=0.3, type=float)
parser.add_argument('--norm', choices=['L2', 'Linf'], default='Linf')
parser.add_argument('--optimizer', choices=['adam', 'normgrad'], default='adam')
parser.add_argument('--steps', type=int, default=50)
parser.add_argument('--step_size', type=float, default=0.01)
parser.add_argument('--method', choices=['ovr', 'bayesian-ovr', 'bayesian-ovr-adv', 'nat', 'static-adv', 'detector-adv', 'adaptive-adv'], default='bayesian-ovr')
parser.add_argument('--tag', type=str, default='tag')


args = parser.parse_args()
print(args)
arg_str = '-'.join(['{}={}'.format(k, v) for k, v in vars(args).items()])

np.random.seed(123)

#if args.norm == 'L2':
#  assert args.epsilon in [2.5, 5.0]
#  # assert args.step_size == 0.1
#  # assert args.optimizer == 'normgrad'
#if args.norm == 'Linf':
#  assert args.epsilon in [0.3, 0.5]
#  # assert args.optimizer == 'adam'


# class CleverhansModel(Model):
#   def __init__(self, classifier):
#     super().__init__(classifier.var_scope, classifier.output_size)
#     self.classifier = classifier
# 
#   def fprop(self, x, **kwargs):
#     logits = classifier.net.forward(x)
#     return {self.O_LOGITS: logits, self.O_PROBS: tf.nn.softmax(logits=logits)}


mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = np.reshape(x_train, [x_train.shape[0], -1])
x_test = np.reshape(x_test, [x_test.shape[0], -1])

x_min, x_max = 0.0, 1.0

#classifier = Classifier(var_scope='classifier', dataset='MNIST')
classifier = MadryClassifier(var_scope='classifier')

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
print('args.epsilon: {}'.format(args.epsilon))
# dir = 'checkpoints/mnist/detector_Linf_0.5/ovr-steps100-adam-noclip-balanced/'
# best = [97, 39, 30, 28, 28, 29, 72, 63, 67, 37]
# detector_models = [os.path.join(dir, 'ovr_class{}_{}_distance{}-{}'.format(i, 'Linf', 0.5, best[i])) for i in range(num_classes)]

#dir = 'checkpoints/mnist/detector_Linf_0.3/ovr-steps100-adam-noclip-balanced/'
#best = [91, 68, 76, 95, 71, 90, 89, 64, 54, 59]
#detector_models = [os.path.join(dir, 'ovr_class{}_{}_distance{}-{}'.format(i, 'Linf', 0.3, best[i])) for i in range(num_classes)]
#print('using models {}'.format(dir))

dir = 'checkpoints/mnist/detector_Linf_0.5/ovr-steps100-adam-noclip-balanced/'
best = [97, 39, 30, 28, 28, 29, 72, 63, 67, 37]
detector_models = [os.path.join(dir, 'ovr_class{}_{}_distance{}-{}'.format(i, 'Linf', 0.5, best[i])) for i in range(num_classes)]

detectors, detector_savers = [], []
for i in range(num_classes):
  scope = 'detector-class{}'.format(i)
  detectors.append(Detector(var_scope=scope, dataset='MNIST'))
  detector_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
  detector_savers.append(tf.train.Saver(var_list=detector_vars))

if args.method == 'ovr':
  with tf.Session() as sess:
    saver.restore(sess, 'checkpoints/mnist/classifier')

    nat_acc, y_pred_test = sess.run([classifier.accuracy, classifier.predictions],
                                    feed_dict={classifier.x_input: x_test,
                                               classifier.y_input: y_test})
    print('nat acc {}'.format(nat_acc))

    all_logits = []
    for i in range(num_classes):
      detector_savers[i].restore(sess, detector_models[i])
      logits, y_pred = sess.run([detectors[i].logits, detectors[i].predictions], feed_dict={detectors[i].x_input: x_test})
      all_logits.append(logits)

    all_logits = np.stack(all_logits, axis=1)
    preds = np.argmax(all_logits, axis=1)
    print('detector committee acc {}'.format((preds == y_test).mean()))

elif args.method == 'bayesian-ovr':
  with tf.Session() as sess:
    # saver.restore(sess, 'checkpoints/mnist/classifier')
    saver.restore(sess, 'checkpoints/mnist/adv_trained_prefixed_classifier/checkpoint-99900')

    nat_acc, y_pred_test = sess.run([classifier.accuracy, classifier.predictions],
                                    feed_dict={classifier.x_input: x_test,
                                               classifier.y_input: y_test})
    print('nat acc {}'.format(nat_acc))


    classifier = BayesianOVR(detectors)
    for i in range(num_classes):
      detector_savers[i].restore(sess, detector_models[i])
    acc = sess.run(classifier.accuracy, feed_dict={classifier.x_input: x_test, classifier.y_input: y_test})
    print('bayesian ovr acc {}'.format(acc))

    x_test, y_test = x_test[:10000], y_test[:10000]

    original_logits, original_preds = sess.run([classifier.logits, classifier.predictions], feed_dict={classifier.x_input: x_test, classifier.y_input: y_test})
    original_max_logits = np.max(original_logits, axis=1)


    attack = PGDAttackClassifier(classifier=classifier, loss_fn='cw',
            max_distance=0.3, num_steps=100, step_size=0.01, 
            random_start=False, x_min=0, x_max=1.0, 
            batch_size=x_test.shape[0], 
            norm='Linf', optimizer='adam') 
    
    # attack = PGDAttackDetector(detector=detectors[0],
    #                            max_distance=10,
    #                            num_steps=100, step_size=0.1,
    #                            random_start=False, x_min=0, x_max=1.0,
    #                            batch_size=x_test.shape[0],
    #                            norm='L2', optimizer='adam')

    x_test_adv = attack.perturb(x_test, y_test, sess)
    logits, preds = sess.run([classifier.logits, classifier.predictions], feed_dict={classifier.x_input: x_test_adv, classifier.y_input: y_test})
    max_logits = np.max(logits, axis=1)
    error_mask = np.logical_and(preds != y_test, max_logits > 0)
    print(np.round(np.max(logits, axis=1), 1))
    print('adv_acc {}'.format(1.0 - error_mask.mean()))
    max_logits, preds = max_logits[error_mask], preds[error_mask]
    x_test_adv, x_test, y_test = x_test_adv[error_mask], x_test[error_mask], y_test[error_mask]

    idx = np.argsort(max_logits)[::-1]
    #idx = np.arange(x_test.shape[0])

    max_logits, preds = max_logits[idx], preds[idx]
    x_test_adv, x_test, y_test = x_test_adv[idx], x_test[idx], y_test[idx]

    fig, axes = plt.subplots(nrows=10, ncols=10, figsize=(10, 10))
    for im, y, y_pred, logit, ax in zip(x_test, y_test, original_preds[error_mask][idx], original_max_logits[error_mask][idx], axes.ravel()):
      ax.imshow(im.reshape([28, 28]), cmap='gray')
      ax.set_axis_off()
      ax.set_title('{}->{},{:.2f}'.format(y, y_pred, logit), fontsize=8)
    plt.suptitle('original')

    fig, axes = plt.subplots(nrows=10, ncols=10, figsize=(10, 10))
    for im, y, y_pred, logit, ax in zip(x_test_adv, y_test, preds, max_logits, axes.ravel()):
      ax.imshow(im.reshape([28, 28]), cmap='gray')
      ax.set_axis_off()
      ax.set_title('{}->{},{:.2f}'.format(y, y_pred, logit), fontsize=8)
    plt.suptitle('pred as adv')
    #plt.tight_layout()
    plt.show()



elif args.method == 'nat':
  AUC = []
  for attack_class in range(10):
    detector = detectors[attack_class]
    x_test_target = x_test[y_test == attack_class]
    x_test_others = x_test[y_test != attack_class]
    y_test_others = y_test[y_test != attack_class]
    with tf.Session() as sess:
      detector_savers[attack_class].restore(sess, detector_models[attack_class])

      x = np.concatenate([x_test_target, x_test_others])
      y = np.concatenate([np.ones(x_test_target.shape[0], np.int64), np.zeros(x_test_others.shape[0], np.int64)])

      logits, y_pred = sess.run([detector.logits, detector.predictions], feed_dict={detector.x_input: x})

      fpr_, tpr_, thresholds = roc_curve(y, logits)
      roc_auc = auc(fpr_, tpr_)
      print('nat base detector {} auc {}, pos {}, neg {}'.format(attack_class, roc_auc, y.sum(), (1-y).sum()))
      AUC.append(roc_auc)
  df = pd.DataFrame({'nat ($\epsilon={}$)'.format(args.epsilon): AUC}).round(5)
  print(df.T)
  print(df.T.to_latex(escape=False))

# elif args.method == 'static-adv':
elif args.method == 'bayesian-ovr-adv':

  attack = PGDAttackBayesianOVR(detectors, thresholds=np.array([0, 0, 0, -0, -0, -0, 0, 0, 0, 0]) - 10,
                               max_distance=args.epsilon,
                               num_steps=args.steps, step_size=args.step_size, random_start=False,
                               x_min=x_min, x_max=x_max, batch_size=x_test.shape[0],
                               norm=args.norm, optimizer=args.optimizer)

  with tf.Session() as sess:
    for i in range(num_classes):
      detector_savers[i].restore(sess, detector_models[i])
    saver.restore(sess, 'checkpoints/mnist/classifier')

    nat_acc, y_pred_test = sess.run([classifier.accuracy, classifier.predictions],
                                    feed_dict={classifier.x_input: x_test,
                                               classifier.y_input: y_test})
    print('nat acc {}'.format(nat_acc))

    sess.run([attack.assign_delta, attack.assign_x0, attack.assign_y], feed_dict={
      attack.delta_input: np.zeros_like(x_test), attack.x0_input: x_test, attack.y_input: y_test})
    y_pred = sess.run(attack.predictions) 
    print('rejection {}'.format((y_pred == 10).mean()))
    print('Bayesian ovr acc {}'.format((y_pred == y_test).mean()))
    print(classification_report(y_test, y_pred, digits=3))


    x_test_adv = attack.perturb(x_test, y_test, sess)


    sess.run([attack.assign_delta, attack.assign_x0, attack.assign_y], feed_dict={
      attack.delta_input: np.zeros_like(x_test), attack.x0_input: x_test_adv, attack.y_input: y_test})

    y_pred = sess.run(attack.predictions) 
    mask = np.logical_and(y_pred != 10, y_pred != y_test)

    y_pred0 = sess.run(classifier.predictions, feed_dict={classifier.x_input: x_test_adv})

    fig, axes = plt.subplots(nrows=6, ncols=10, figsize=(20, 20))
    for im, y, y_pred0, y_pred, ax in zip(x_test_adv[mask], y_test[mask], y_pred0[mask], y_pred[mask], axes.ravel()):
      ax.imshow(im.reshape([28, 28]), cmap='gray')
      ax.set_axis_off()
      ax.set_title('{}->{},{}'.format(y, y_pred, y_pred0))
    plt.suptitle('pred as adv')
    #plt.tight_layout()
    plt.show()

    print('success {}'.format(np.mean(mask)))
    sys.exit(0)

    # fgsm = FastGradientMethod(cleverhans_model, sess=sess)
    # fgsm_params = {'eps': 0.3, 'clip_min': 0., 'clip_max': 1., 'ord': np.inf}
    # x_test_adv = fgsm.generate_np(x_test, **fgsm_params)

    # spsa = SPSA(cleverhans_model, sess)
    # x_test_adv = spsa.generate_np(x_val=x_test, y=y_test.astype(np.int32), eps=0.3, clip_min=0.0, clip_max=1.0, nb_iter=100)
    # print(np.linalg.norm(x_test_adv - x_test, ord=np.inf, axis=1)[:10])

    adv_acc, y_pred_test_adv = sess.run([classifier.accuracy, classifier.predictions],
                                        feed_dict={classifier.x_input: x_test_adv,
                                                   classifier.y_input: y_test})
    print('adv acc {}'.format(adv_acc))

    x_test_adv = x_test_adv[y_pred_test_adv != y_test]

    x_all = np.concatenate([x_test, x_test_adv])
    y_det = np.concatenate([np.ones(x_test.shape[0], np.int64), np.zeros(x_test_adv.shape[0], np.int64)])
    y_pred = sess.run(classifier.predictions, feed_dict={classifier.x_input: x_all})

    subsets = [(x_all[y_pred == i], y_det[y_pred == i]) for i in range(num_classes)]
    AUC = []
    for i in range(num_classes):
      x, y = subsets[i]
      detector_savers[i].restore(sess, detector_models[i])
      logits, y_pred = sess.run([detectors[i].logits, detectors[i].predictions], feed_dict={detectors[i].x_input: x})

      fpr_, tpr_, thresholds = roc_curve(y, logits)
      roc_auc = auc(fpr_, tpr_)
      print('{} detector {} auc {}, pos {}, neg {}'.format(arg_str, i, np.round(roc_auc, 5), y.sum(), (1-y).sum()))
      AUC.append(roc_auc)
    df = pd.DataFrame({'static-adv ($\epsilon={}$)'.format(args.epsilon): AUC}).round(5)
    print(df.T)
    print(df.T.to_latex(escape=False))

      # fig, axes = plt.subplots(nrows=10, ncols=10, figsize=(10, 10))
      # for im, ax in zip(x[y_pred == 0], axes.ravel()):
      #   ax.imshow(im.reshape([28, 28]), cmap='gray')
      #   ax.set_axis_off()
      # plt.suptitle('pred as adv')
      #
      # fig, axes = plt.subplots(nrows=10, ncols=10, figsize=(10, 10))
      # for im, ax in zip(x[y_pred == 1], axes.ravel()):
      #   ax.imshow(im.reshape([28, 28]), cmap='gray')
      #   ax.set_axis_off()
      # plt.suptitle('pred as nat')
      # plt.show()
elif args.method == 'adaptive-adv':
  AUC = []
  for attack_class in range(10):
    x_test_target = x_test[y_test == attack_class]
    x_test_others = x_test[y_test != attack_class]
    y_test_others = y_test[y_test != attack_class]

    detector = detectors[attack_class]
    attack = PGDAttackAda(attack_class, classifier, detector, method='x',
                          max_distance=args.epsilon, num_steps=args.steps,
                          step_size=args.step_size, random_start=False,
                          x_min=x_min, x_max=x_max, batch_size=x_test_others.shape[0],
                          norm=args.norm, optimizer=args.optimizer)

    with tf.Session() as sess:
      saver.restore(sess, 'checkpoints/mnist/classifier')
      detector_savers[attack_class].restore(sess, detector_models[attack_class])

      x_test_others_adv = attack.perturb(x_test_others, None, sess)

      print('x_test_others.shape {}'.format(x_test_others.shape[0]))

      adv_acc, y_pred_test_others_adv = sess.run([classifier.accuracy, classifier.predictions],
                                                 feed_dict={classifier.x_input: x_test_others_adv,
                                                            classifier.y_input: y_test_others})

      # # print('x_test_others_adv acc {}'.format(adv_acc))
      # x_test_others_adv = x_test_others_adv[y_pred_test_others_adv == attack_class]
      # print('x_test_others_adv.shape {}'.format(x_test_others_adv.shape[0]))

      x = np.concatenate([x_test_target, x_test_others_adv])
      y = np.concatenate([np.ones(x_test_target.shape[0], np.int64), np.zeros(x_test_others_adv.shape[0], np.int64)])

      logits, y_pred = sess.run([detector.logits, detector.predictions], feed_dict={detector.x_input: x})

      fpr_, tpr_, thresholds = roc_curve(y, logits)
      roc_auc = auc(fpr_, tpr_)
      print('{} detector {} detection auc {}, pos {}, neg {}'.format(arg_str, attack_class, roc_auc, y.sum(), (1-y).sum()))
      AUC.append(roc_auc)
  df = pd.DataFrame({'adaptive-adv ($\epsilon={}$)'.format(args.epsilon): AUC}).round(5)
  print(df.T)
  print(df.T.to_latex(escape=False))

elif args.method == 'detector-adv':
  AUC = []
  for attack_class in range(0, 1):
    x_test_target = x_test[y_test == attack_class]
    x_test_others = x_test[y_test != attack_class]
    y_test_others = y_test[y_test != attack_class]

    #mask = np.bitwise_and(y_test != 0, y_test != 1)
    #mask = np.bitwise_and(y_test != 2, mask).astype(np.bool)
    #x_test_others = x_test[mask]
    #y_test_others = y_test[mask]

    detector = detectors[attack_class]
    attack = PGDAttackDetector(detector=detector,
                               max_distance=args.epsilon,
                               num_steps=args.steps, step_size=args.step_size,
                               random_start=True, x_min=x_min, x_max=x_max,
                               batch_size=x_test_others.shape[0],
                               norm=args.norm, optimizer=args.optimizer)

    config = tf.ConfigProto(device_count = {'GPU': 0})
    with tf.Session() as sess:
      saver.restore(sess, 'checkpoints/mnist/classifier')
      detector_savers[attack_class].restore(sess, detector_models[attack_class])

      x_test_others_adv = attack.perturb(x_test_others, y_test_others, sess)

      x = np.concatenate([x_test_target, x_test_others_adv])
      x_with_others = np.concatenate([x_test_target, x_test_others])
      y = np.concatenate([np.ones(x_test_target.shape[0], np.int64), np.zeros(x_test_others_adv.shape[0], np.int64)])

      y_clf_pred = sess.run(classifier.predictions, feed_dict={classifier.x_input: x_with_others})

      logits, y_pred = sess.run([detector.logits, detector.predictions], feed_dict={detector.x_input: x})
      fpr_, tpr_, thresholds = roc_curve(y, logits)
      roc_auc = auc(fpr_, tpr_)
      print('roc_auc: {}'.format(roc_auc))
      sample_filename = 'random_test/{}-class{}_samples_rand{}.npz'.format(arg_str, attack_class, args.tag)
      np.savez(sample_filename, logits=logits, y=y)

      #  false_pos = np.bitwise_and((1 - y), y_pred).astype(np.bool)
      #  false_pos = np.bitwise_and(false_pos, y_clf_pred != attack_class).astype(np.bool)
      #  #false_pos = (1 - y).astype(np.bool)
      #  fig, axes = plt.subplots(nrows=10, ncols=10, figsize=(10, 10))
      #  false_pos_logits = logits[false_pos]
      #  false_pos_x_adv = x[false_pos]
      #  false_pos_x_nat = x_with_others[false_pos]

      #  idx = np.argsort(false_pos_logits)[::-1]
      #  false_pos_logits = false_pos_logits[idx]
      #  false_pos_x_adv = false_pos_x_adv[idx]
      #  false_pos_x_nat = false_pos_x_nat[idx]

      #  sample_filename = 'adv_stats/{}-class{}_samples.npz'.format(arg_str, attack_class)
      #  #np.savez(sample_filename, adv=false_pos_x_adv, nat=false_pos_x_nat)

      #  for logit, im, ax in zip(false_pos_logits, false_pos_x_adv, axes.ravel()):
      #    ax.imshow(im.reshape([28,28]), cmap='gray')
      #    ax.set_title('{:.2f}'.format(logit))
      #    ax.set_axis_off()
      #  plt.savefig('false_pos_sample.pdf')

      #  fig, axes = plt.subplots(nrows=10, ncols=10, figsize=(10, 10))
      #  for im, ax in zip(false_pos_x_nat, axes.ravel()):
      #    ax.imshow(im.reshape([28,28]), cmap='gray')
      #    ax.set_axis_off()
      #  plt.savefig('false_pos_sample_original.pdf')
      #  # plt.show()

      #  fpr_, tpr_, thresholds = roc_curve(y, logits)
      #  roc_filename = 'adv_stats/{}-class{}_roc.npz'.format(arg_str, attack_class)
      #  #np.savez(roc_filename, tpr=tpr_, fpr=fpr_, thresholds=thresholds)
      #  print('saved {}'.format(roc_filename))

      #  fpr_, tpr_, thresholds = roc_curve(y, logits)
      #  roc_auc = auc(fpr_, tpr_)
      #  print('{} detector {} auc {}, pos {}, neg {}'.format(arg_str, attack_class, np.round(roc_auc, 5), y.sum(), (1-y).sum()))
      #  #with open('adv_stats/{}-class{}_test.log'.format(arg_str, attack_class), 'w') as f:
      #  #  f.write('{} detector {} auc {}, pos {}, neg {}\n'.format(arg_str, attack_class, np.round(roc_auc, 5), y.sum(), (1-y).sum()))

      AUC.append(roc_auc)
  # df = pd.DataFrame({'detector-adv ($\epsilon={}$)'.format(args.epsilon): AUC}).round(5)
  # print(df.T)
  # print(df.T.to_latex(escape=False))


