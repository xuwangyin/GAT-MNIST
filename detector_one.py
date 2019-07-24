import sys
import argparse
import os
import time
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc


from models import Detector, Classifier, PGDAttack
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('-m', choices=['sigmoid', 'softmax'], default='softmax')
parser.add_argument('-lc', metavar='left confidence threshold', type=float, default=0.0)
parser.add_argument('-rc', metavar='right confidence threshold', type=float, default=500.0)
parser.add_argument('-d', metavar='max-distance', type=float)
parser.add_argument('--dim', type=int, default=20)
parser.add_argument('--dataset', choices=['3class', 'mnist'], default='mnist')
parser.add_argument('--constrain', choices=['confidence', 'distance'], default='distance')
parser.add_argument('--conf_det_thresh', type=float, default=9.0)
parser.add_argument('--distance_type', choices=['L2', 'Linf'], required=True)


# 'checkpoints/detector_softmax_lconfidence-0.0_rconfidence-100.0_distance-1.0-55'
# 'checkpoints/detector_softmax_lconfidence-0.0_rconfidence-100.0_distance-1.0_dim-2-11'
parser.add_argument('--test_model', type=str)
parser.add_argument('--test_steps', type=int, default=5000)
parser.add_argument('--test_step_size', type=float, default=0.001)
parser.add_argument('--test_stats', action='store_true')



args = parser.parse_args()
print(args)

np.random.seed(123)

batch_size = 200

args.d = {'L2': 1.5, 'Linf': 0.3}[args.distance_type]

x_train, y_train, x_test, y_test, num_classes = dataset(args)

if args.dataset == 'mnist':
  perm = np.random.permutation(x_test.shape[0])
  x_test, y_test = x_test[perm][:1000], y_test[perm][:1000]

classifier = Classifier(var_scope='classifier')

vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='classifier')
saver = tf.train.Saver(var_list=vars, max_to_keep=1)

detector_var_scope ='detector'
detector = Detector(var_scope=detector_var_scope)
detector_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=detector_var_scope)
detector_saver = tf.train.Saver(var_list=detector_vars, max_to_keep=100)

train_step = tf.train.AdamOptimizer(5e-4, name='detector_adam').minimize(detector.xent)

model_name = 'detector_mnist_distance{}'.format(args.distance_type, args.d)

x_min, x_max = np.min(x_train), np.max(x_train)
print('x_min: {}, x_max: {}'.format(x_min, x_max))
train_attack = PGDAttack(classifier, detector,
                         max_distance=args.d,
                         num_steps=40, step_size=0.01, random_start=True,
                         x_min=x_min, x_max=x_max, batch_size=batch_size,
                         loss_fn='clf_det_adv', distance_type=args.distance_type)

test_attack = PGDAttack(classifier, detector,
                        max_distance=args.d,
                        num_steps=200, step_size=0.01, random_start=False,
                        x_min=x_min, x_max=x_max, loss_fn='clf_det_adv',
                        batch_size=x_test.shape[0], distance_type=args.distance_type)

config = tf.ConfigProto(device_count = {'GPU': 0})
#config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.2))
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  saver.restore(sess, os.path.join('checkpoints/classifier_{}_dim{}_{}'.format(args.m, args.dim, args.dataset)))
  print('loaded ' + 'checkpoints/classifier_{}'.format(args.m))

  # args.test_model = 'checkpoints/mnist-one-babbage/detector_mnist_distanceL2-105'
  if args.test_model is not None:
    detector_saver.restore(sess, args.test_model)
    print('loaded {}'.format(args.test_model))

    test_attack = PGDAttack(classifier, detector,
                            max_distance=args.d,
                            num_steps=5000, step_size=0.001, random_start=False,
                            x_min=x_min, x_max=x_max, batch_size=x_test.shape[0],
                            loss_fn='clf_det_adv', target_class=None, distance_type=args.distance_type)

    x_test_adv, test_dist, detector_logits, mask = test_attack.perturb(x_test, y_test, sess)
    x_test_with_adv = np.concatenate([x_test, x_test_adv])
    y_test_with_adv = np.concatenate(
      [np.ones(x_test.shape[0], dtype=np.int64), np.zeros(x_test_adv.shape[0], dtype=np.int64)])

    classifier_preds = sess.run(classifier.predictions, feed_dict={classifier.x_input: x_test_adv})
    print('missclassification {:.4f}'.format((classifier_preds != y_test[mask]).mean()))

    test_detector_logits, y_pred_test, f_score, precision, recall, acc, balanced_acc, tpr, fpr = sess.run(
      [detector.logits, detector.predictions, detector.f_score, detector.precision, detector.recall, detector.accuracy,
       detector.balanced_accuracy, detector.true_positive_rate, detector.false_positive_rate],
      feed_dict={detector.x_input: x_test_with_adv, detector.y_input: y_test_with_adv})
    fpr_, tpr_, thresholds = roc_curve(y_test_with_adv, test_detector_logits)
    roc_auc = auc(fpr_, tpr_)
    print('test auc {:.4f} f-score {:.4f}, precision {:.4f}, recall {:.4f}, acc {:.4f}, balanced_acc {:.4f}, tpr {:.4f} fpr {:.4f} '.format(
      roc_auc, f_score, precision, recall, acc, balanced_acc, tpr, fpr), end='|')

    print('pos {:.4f}, true pos {:.4f}, adv {}, nat {}'.format(np.sum(y_pred_test),
                                                               np.sum(np.bitwise_and(y_pred_test, y_test_with_adv)),
                                                               np.sum(1-y_test_with_adv), np.sum(y_test_with_adv)))


    # with np.printoptions(suppress=True, threshold=np.inf):
    #   mask = np.bitwise_and(false_pos_mask[x_test_target.shape[0]:], classifier_preds == args.target_class)
    #   print('false pos {}, neg {}'.format(np.sum(false_pos_mask), np.sum(1-y_test_with_adv)))
    #   print('targeted false pos {}'.format(np.sum(mask)))
    #
    fpr, tpr, thresholds = roc_curve(y_test_with_adv, test_detector_logits)
    roc_auc = auc(fpr, tpr)
    # for fpr_, tpr_, th in zip(fpr, tpr, thresholds):
    #   print('thresh {:.4f} tpr {:.4f} fpr {:.4f}'.format(th, tpr_, fpr_))
    plt.plot(fpr, tpr, label='AUC = {:.4f}'.format(roc_auc))
    plt.xlim([-0.05, 1.05])
    plt.ylim([0.9, 1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    plt.show()
    #
    #false_pos_mask = np.bitwise_and(y_pred_test, 1 - y_test_with_adv).astype(np.bool)
    #false_pos = x_test_with_adv[false_pos_mask]
    #fig, axes = plt.subplots(nrows=10, ncols=10, figsize=(10, 10))
    #for img, ax in zip(false_pos, axes.flatten()):
    #  ax.imshow(img.reshape((28, 28)), cmap='gray')
    #plt.show()
    #
    #false_neg_mask = np.bitwise_and(1 - y_pred_test, y_test_with_adv).astype(np.bool)
    #false_neg = x_test_with_adv[false_neg_mask]
    #print('false neg {}, pos {}'.format(np.sum(false_neg_mask), np.sum(y_test_with_adv)))
    #fig, axes = plt.subplots(nrows=10, ncols=10, figsize=(10, 10))
    #for img, ax in zip(false_neg, axes.flatten()):
    # ax.imshow(img.reshape((28, 28)), cmap='gray')
    #plt.show()
    


    #with np.printoptions(suppress=True, threshold=np.inf):
    #  tp = np.bitwise_and(y_pred_test[x_test.shape[0]:], y_test_with_adv[x_test.shape[0]:])
    #  print('dist \t detected')
    #  print(np.stack([test_dist, tp], axis=1))

    sys.exit(0)

  for epoch in range(1, 100):
    perm = np.random.permutation(x_train.shape[0])
    x_train, y_train = x_train[perm], y_train[perm]

    for i in range(0, x_train.shape[0], batch_size):
      x_batch = x_train[i: i + batch_size]
      y_batch = y_train[i: i + batch_size]
      # TODO fixed sized input
      if x_batch.shape[0] < batch_size:
        continue

      tic = time.time()
      x_batch_adv, batch_dist, detector_logits, mask = train_attack.perturb(x_batch, y_batch, sess, verbose=False)
      toc = time.time()

      x_batch_with_adv = np.concatenate([x_batch, x_batch_adv])
      y_batch_with_adv = np.concatenate(
        [np.ones(x_batch.shape[0], dtype=np.int64), np.zeros(x_batch_adv.shape[0], dtype=np.int64)])

      _, detector_logits, f_score, precision, recall, acc, balanced_acc, tpr, fpr = sess.run(
        [train_step, detector.logits, detector.f_score, detector.precision, detector.recall, detector.accuracy,
         detector.balanced_accuracy, detector.true_positive_rate, detector.false_positive_rate],
        feed_dict={detector.x_input: x_batch_with_adv, detector.y_input: y_batch_with_adv})

      fpr_, tpr_, thresholds = roc_curve(y_batch_with_adv, detector_logits)
      roc_auc = auc(fpr_, tpr_)

      print('epoch {}, iter {}/{}'.format(epoch, i, x_train.shape[0]), end=' ')
      print('train auc {:.4f} f-score {:.4f}, precision {:.4f}, recall {:.4f}, '.format(
        roc_auc, f_score, precision, recall), end='')
      print('acc {:.4f}, balanced_acc {:.4f} tpr {:.4f} fpr {:.4f} '.format(acc, balanced_acc, tpr, fpr), end='')
      print('dist<={:.4f} {:.4f} {:.4f}/{:.4f} time {:.1f}'.format(args.d,
        (batch_dist <= args.d + 1e-6).mean(), batch_dist.mean(), batch_dist.std(), 1000*(toc-tic)))

    x_train_adv, train_dist, detector_logits, mask = train_attack.perturb(x_train[:batch_size], y_train[:batch_size], sess)
    x_train_with_adv = np.concatenate([x_train[:batch_size], x_train_adv])
    y_train_with_adv = np.concatenate(
      [np.ones(x_train[:batch_size].shape[0], dtype=np.int64), np.zeros(x_train_adv.shape[0], dtype=np.int64)])

    classifier_preds = sess.run(classifier.predictions, feed_dict={classifier.x_input: x_train_adv})
    print('missclassification {:.4f}'.format((classifier_preds != y_train[:batch_size][mask]).mean()))

    train_detector_logits, y_pred_train, f_score, precision, recall, acc, balanced_acc, tpr, fpr = sess.run(
      [detector.logits, detector.predictions, detector.f_score, detector.precision, detector.recall, detector.accuracy,
       detector.balanced_accuracy, detector.true_positive_rate, detector.false_positive_rate],
      feed_dict={detector.x_input: x_train_with_adv, detector.y_input: y_train_with_adv})
    fpr_, tpr_, thresholds = roc_curve(y_train_with_adv, train_detector_logits)
    roc_auc = auc(fpr_, tpr_)
    print('===epoch {}, train auc {:.4f} f-score {:.4f}, precision {:.4f}, recall {:.4f}, acc {:.4f}, balanced_acc {:.4f}, tpr {:.4f} fpr {:.4f}'.format(
      epoch, roc_auc, f_score, precision, recall, acc, balanced_acc, tpr, fpr), end='|')
    print('pos {:.4f}, true pos {:.4f}, adv {}, nat {}'.format(np.sum(y_pred_train), np.sum(np.bitwise_and(y_pred_train, y_train_with_adv)),
                                                               np.sum(1-y_train_with_adv), np.sum(y_train_with_adv)))

    x_test_adv, test_dist, detector_logits, mask = test_attack.perturb(x_test, y_test, sess)
    x_test_with_adv = np.concatenate([x_test, x_test_adv])
    y_test_with_adv = np.concatenate(
      [np.ones(x_test.shape[0], dtype=np.int64), np.zeros(x_test_adv.shape[0], dtype=np.int64)])

    classifier_preds = sess.run(classifier.predictions, feed_dict={classifier.x_input: x_test_adv})
    print('missclassification {:.4f}'.format((classifier_preds != y_test[:1000][mask]).mean()))

    test_detector_logits, y_pred_test, f_score, precision, recall, acc, balanced_acc, tpr, fpr = sess.run(
      [detector.logits, detector.predictions, detector.f_score, detector.precision, detector.recall, detector.accuracy,
       detector.balanced_accuracy, detector.true_positive_rate, detector.false_positive_rate],
      feed_dict={detector.x_input: x_test_with_adv, detector.y_input: y_test_with_adv})
    fpr_, tpr_, thresholds = roc_curve(y_test_with_adv, test_detector_logits)
    roc_auc = auc(fpr_, tpr_)
    print('===epoch {}, test auc {:.4f} f-score {:.4f}, precision {:.4f}, recall {:.4f}, acc {:.4f}, balanced_acc {:.4f}, tpr {:.4f} fpr {:.4f} '.format(
      epoch, roc_auc, f_score, precision, recall, acc, balanced_acc, tpr, fpr), end='|')

    print('pos {:.4f}, true pos {:.4f}, adv {}, nat {}'.format(np.sum(y_pred_test),
                                                               np.sum(np.bitwise_and(y_pred_test, y_test_with_adv)),
                                                               np.sum(1-y_test_with_adv), np.sum(y_test_with_adv)))

    if epoch == 1 or epoch % 3 == 0:
      detector_saver.save(sess, os.path.join('checkpoints/mnist-one-steps40-normgrad/', model_name), global_step=epoch)
