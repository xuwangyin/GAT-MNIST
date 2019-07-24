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

elif args.method == 'static-adv':

  cleverhans_model = CleverhansModel(classifier)
  config = tf.ConfigProto(device_count = {'GPU': 0})
  # with tf.Session(config=config) as sess:
  #   saver.restore(sess, 'checkpoints/mnist/classifier')
  #   # attack = CarliniWagnerL2(cleverhans_model, sess=sess)


  attack = PGDAttackClassifier(classifier,
                               max_distance=args.epsilon,
                               num_steps=args.steps, step_size=args.step_size, random_start=False,
                               x_min=x_min, x_max=x_max, batch_size=x_test.shape[0],
                               norm=args.norm, optimizer=args.optimizer, loss_fn='cw')

  with tf.Session() as sess:
    saver.restore(sess, 'checkpoints/mnist/classifier')

    nat_acc, y_pred_test = sess.run([classifier.accuracy, classifier.predictions],
                                    feed_dict={classifier.x_input: x_test,
                                               classifier.y_input: y_test})
    print('nat acc {}'.format(nat_acc))

    #x_test_adv = attack.perturb(x_test, y_test, sess)
    x_test_adv = np.load('/home/xy4cm/Projects/tmp/mnist_challenge/attack.npy')

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
  data = np.load('x_test_target.npz')
  x_test, y_test, y_wrong = data['x_test'], data['y_test'], data['y_wrong']
  x_test_adv = []
  y_test_adv = []

  madry_attack = MadryPGDAttackDetector(target_class=0,
                                        max_distance=args.epsilon,
                                        num_steps=args.steps, step_size=args.step_size,
                                        random_start=False, x_min=x_min, x_max=x_max,
                                        batch_size=10000,
                                        norm=args.norm, optimizer=args.optimizer)
    
  for attack_class in range(0, 10):

    x_test_others = x_test[y_wrong == attack_class]
    y_test_others = y_test[y_wrong == attack_class]
    print('x_test_others: {}'.format(x_test_others.shape))
    
    # x_test_target = x_test[y_test == attack_class]
    # x_test_others = x_test[y_test != attack_class]
    # y_test_others = y_test[y_test != attack_class]


    

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

    # madry_attack = MadryPGDAttackDetector(target_class=0,
    #                            max_distance=args.epsilon,
    #                            num_steps=args.steps, step_size=args.step_size,
    #                            random_start=False, x_min=x_min, x_max=x_max,
    #                            batch_size=np.sum(y_wrong<=attack_class),
    #                            norm=args.norm, optimizer=args.optimizer)

    # madry_attack = MadryPGDAttackClassifier(loss_fn='cw',
    #                           max_distance=args.epsilon,
    #                           num_steps=args.steps, step_size=args.step_size,
    #                           random_start=False, x_min=x_min, x_max=x_max,
    #                           batch_size=x_test.shape[0],
    #                           norm=args.norm, optimizer=args.optimizer)

    madry_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='madry')
    madry_saver = tf.train.Saver(var_list=madry_vars, max_to_keep=1)
    # with tf.Session() as sess:
    #   madry_saver.restore(sess, '/home/xy4cm/Projects/tmp/mnist_challenge/models/adv_trained_madry/checkpoint-99900')
    #   sess.run([madry_attack.assign_delta, madry_attack.assign_x0, madry_attack.assign_y], feed_dict={
    #     madry_attack.delta_input: np.zeros_like(x_test), madry_attack.x0_input: x_test, madry_attack.y_input: y_test})
    #   acc = sess.run(madry_attack.accuracy)
    #   print('==================== loaded madry classifier, acc {}'.format(acc))

    #   y_wrong = sess.run(madry_attack.wrong_class)
    #   np.savez('x_test_target.npz', x_test=x_test, y_test=y_test, y_wrong=y_wrong)
      
    #   x_test_adv = madry_attack.perturb(x_test, y_test, sess)
    #   sess.run([madry_attack.assign_delta, madry_attack.assign_x0, madry_attack.assign_y], feed_dict={
    #     madry_attack.delta_input: np.zeros_like(x_test), madry_attack.x0_input: x_test_adv, madry_attack.y_input: y_test})
      
    #   y_pred_adv = sess.run(madry_attack.y_pred)
    #   print('y_pred_adv: {}, madry adv acc {}'.format(y_pred_adv, (y_pred_adv == y_test).mean()))
    #   sys.exit(0)


    config = tf.ConfigProto(device_count = {'GPU': 0})
    with tf.Session() as sess:

      #madry_saver.restore(sess, '/home/xy4cm/Projects/tmp/mnist_challenge/models/natural_madry/checkpoint-24900')
      madry_saver.restore(sess, '/home/xy4cm/Projects/tmp/mnist_challenge/models/adv_trained_madry/checkpoint-99900')
      # sess.run([madry_attack.assign_delta, madry_attack.assign_x0, madry_attack.assign_y], feed_dict={
      #   madry_attack.delta_input: np.zeros_like(x_test_others), madry_attack.x0_input: x_test_others, madry_attack.y_input: y_test_others})
      # acc = sess.run(madry_attack.accuracy)
      # print('==================== loaded madry classifier, acc {}'.format(acc))

      saver.restore(sess, 'checkpoints/mnist/classifier')
      detector_savers[attack_class].restore(sess, detector_models[attack_class])

      if args.madry:
        x_test_others_adv = madry_attack.perturb(x_test_others, y_test_others, sess)
      else:
        x_test_others_adv = attack.perturb(x_test_others, y_test_others, sess)
      x_test_adv.append(x_test_others_adv)
      y_test_adv.append(y_test_others)

      continue


      # fig, axes = plt.subplots(nrows=10, ncols=10, figsize=(10, 10))
      # for im, ax in zip(x_test_others_adv, axes.ravel()):
      #  ax.imshow(im.reshape((28,28)), cmap='gray')
      #  ax.set_axis_off()
      # plt.show()
      
      sess.run([madry_attack.assign_delta, madry_attack.assign_x0, madry_attack.assign_y], feed_dict={
        madry_attack.delta_input: np.zeros_like(x_test_others), madry_attack.x0_input: x_test_others_adv, madry_attack.y_input: y_test_others})
      y_pred_adv = sess.run(madry_attack.y_pred)
      print('y_pred_adv: {}, madry adv acc {}'.format(y_pred_adv, (y_pred_adv == y_test_others).mean()))
      continue

      x = np.concatenate([x_test_target, x_test_others_adv])
      x_with_others = np.concatenate([x_test_target, x_test_others])
      y = np.concatenate([np.ones(x_test_target.shape[0], np.int64), np.zeros(x_test_others_adv.shape[0], np.int64)])

      y_clf_pred = sess.run(classifier.predictions, feed_dict={classifier.x_input: x_with_others})

      logits, y_pred = sess.run([detector.logits, detector.predictions], feed_dict={detector.x_input: x})
      fpr_, tpr_, thresholds = roc_curve(y, logits)
      roc_auc = auc(fpr_, tpr_)
      print('auc {}'.format(roc_auc))

      #false_pos = np.bitwise_and((1 - y), y_pred).astype(np.bool)
      #false_pos = np.bitwise_and(false_pos, y_clf_pred != attack_class).astype(np.bool)
      false_pos = (1 - y).astype(np.bool)
      fig, axes = plt.subplots(nrows=10, ncols=10, figsize=(10, 10))
      false_pos_logits = logits[false_pos]
      false_pos_x_adv = x[false_pos]
      false_pos_x_nat = x_with_others[false_pos]

      idx = np.argsort(false_pos_logits)[::-1]
      #idx = [28, 40, 73, 90, 45, 47, 16, 97, 10, 49, 59, 31, 13, 95, 33, 46, 42, 62, 68, 64, 1, 12, 20, 23, 15, 50, 91, 76, 75, 55, 89, 93, 48, 87, 72, 21, 65, 92, 53, 60, 17, 99, 86, 19, 8, 9, 27, 82, 58, 44, 85, 14, 52, 18, 57, 3, 69, 67, 7, 6, 71, 74, 43, 5, 94, 77, 84, 83, 96, 70, 61, 0, 25, 80, 56, 37, 30, 35, 79, 36, 98, 39, 63, 54, 22, 41, 24, 81, 51, 29, 32, 34, 78, 38, 11, 88, 2, 66, 26, 4]
      #print(', '.join(map(str, idx)))
      false_pos_logits = false_pos_logits[idx]
      false_pos_x_adv = false_pos_x_adv[idx]
      false_pos_x_nat = false_pos_x_nat[idx]


      topn = 10
      topn_adv = np.zeros(shape=(28, 28*topn))
      topn_nat = np.zeros(shape=(28, 28*topn))
      for i in range(topn):
        topn_adv[:, 28*i: 28*i+28] = false_pos_x_adv[i].reshape((28,28))
        topn_nat[:, 28*i: 28*i+28] = false_pos_x_nat[i].reshape((28,28))
      plt.figure(figsize=(20, 1))
      plt.imshow(topn_nat, cmap='gray')
      plt.savefig('top15_nat.pdf')
      plt.imshow(topn_adv, cmap='gray')
      plt.savefig('top15_adv.pdf')
      plt.show()
      sys.exit(0)

      sample_filename = 'adv_stats/{}-class{}_samples.npz'.format(arg_str, attack_class)
      #np.savez(sample_filename, adv=false_pos_x_adv, nat=false_pos_x_nat)

      for logit, im, ax in zip(false_pos_logits, false_pos_x_adv, axes.ravel()):
        ax.imshow(im.reshape([28,28]), cmap='gray')
        ax.set_title('{:.2f}'.format(logit))
        ax.set_axis_off()
      plt.savefig('false_pos_sample.pdf')

      fig, axes = plt.subplots(nrows=10, ncols=10, figsize=(10, 10))
      for im, ax in zip(false_pos_x_nat, axes.ravel()):
        ax.imshow(im.reshape([28,28]), cmap='gray')
        ax.set_axis_off()
      plt.savefig('false_pos_sample_original.pdf')
      plt.show()

      fpr_, tpr_, thresholds = roc_curve(y, logits)
      roc_filename = 'adv_stats/{}-class{}_roc.npz'.format(arg_str, attack_class)
      #np.savez(roc_filename, tpr=tpr_, fpr=fpr_, thresholds=thresholds)
      print('saved {}'.format(roc_filename))

      fpr_, tpr_, thresholds = roc_curve(y, logits)
      roc_auc = auc(fpr_, tpr_)
      print('{} detector {} auc {}, pos {}, neg {}'.format(arg_str, attack_class, np.round(roc_auc, 5), y.sum(), (1-y).sum()))
      #with open('adv_stats/{}-class{}_test.log'.format(arg_str, attack_class), 'w') as f:
      #  f.write('{} detector {} auc {}, pos {}, neg {}\n'.format(arg_str, attack_class, np.round(roc_auc, 5), y.sum(), (1-y).sum()))

      AUC.append(roc_auc)

  with tf.Session() as sess:
      #madry_saver.restore(sess, '/home/xy4cm/Projects/tmp/mnist_challenge/models/natural_madry/checkpoint-24900')
      madry_saver.restore(sess, '/home/xy4cm/Projects/tmp/mnist_challenge/models/adv_trained_madry/checkpoint-99900')      
      x_test_adv_tmp = np.concatenate(x_test_adv)
      y_test_adv_tmp = np.concatenate(y_test_adv)
      sess.run([madry_attack.assign_delta, madry_attack.assign_x0, madry_attack.assign_y], feed_dict={
        madry_attack.delta_input: np.zeros_like(x_test_adv_tmp),
        madry_attack.x0_input: x_test_adv_tmp,
        madry_attack.y_input: y_test_adv_tmp})
      acc = sess.run(madry_attack.accuracy)
      print('====================  madry classifier running acc {}'.format(acc))
  
  df = pd.DataFrame({'detector-adv ($\epsilon={}$)'.format(args.epsilon): AUC}).round(5)
  print(df.T)
  print(df.T.to_latex(escape=False))


