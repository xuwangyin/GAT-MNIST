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
parser.add_argument('--target_class', type=int, required=True)
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

batch_size = 100

args.d = {'L2': 1.5, 'Linf': 0.3}[args.distance_type]

x_train, y_train, x_test, y_test, num_classes = dataset(args)

# if args.dataset == 'mnist':
#   perm = np.random.permutation(x_test.shape[0])
#   x_test, y_test = x_test[perm][:1000], y_test[perm][:1000]

x_train_target = x_train[y_train == args.target_class]
y_train_target = y_train[y_train == args.target_class]
x_train_others = x_train[y_train != args.target_class]
y_train_others = y_train[y_train != args.target_class]

x_test_target = x_test[y_test == args.target_class]
y_test_target = y_test[y_test == args.target_class]
x_test_others = x_test[y_test != args.target_class]
y_test_others = y_test[y_test != args.target_class]

x_train_others = x_train_target
y_train_others = y_train_target
x_test_others = x_test_target
y_test_others = y_test_target

num_target = x_train_target.shape[0]
num_others = x_train_others.shape[0]

target_batch_size = int(batch_size * num_target / x_train.shape[0])
others_batch_size = batch_size - target_batch_size
target_batch_size = others_batch_size = batch_size


if args.dataset == '3class':
  args.d = 1.0

plot = (x_train.shape[1] == 2)

classifier = Classifier(var_scope='classifier', dataset='MNIST')

vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='classifier')
saver = tf.train.Saver(var_list=vars, max_to_keep=1)

detector_var_scope ='detector-class{}'.format(args.target_class)
detector = Detector(var_scope=detector_var_scope)
detector_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=detector_var_scope)
detector_saver = tf.train.Saver(var_list=detector_vars, max_to_keep=100)

train_step = tf.train.AdamOptimizer(5e-4, name='detector_adam').minimize(detector.xent)

model_name = 'detector_ovr_mnist_class{}_{}_distance{}'.format(args.target_class, args.distance_type, args.d)

# adversarial training of detector
# x_min, x_max = np.min(x_train, axis=0), np.max(x_train, axis=0)
# x_min, x_max = np.min(x_train), np.max(x_train)
x_min, x_max = 0.0, 1.0
print('x_min: {}, x_max: {}'.format(x_min, x_max))
train_attack = PGDAttack(classifier, detector,
                         max_distance=args.d,
                         num_steps=40, step_size=0.01, random_start=True,
                         x_min=x_min, x_max=x_max, batch_size=others_batch_size,
                         loss_fn='clf_det_adv_bypass10', distance_type=args.distance_type)

test_attack = PGDAttack(classifier, detector,
                        max_distance=args.d,
                        num_steps=100, step_size=0.01, random_start=False,
                        x_min=x_min, x_max=x_max, loss_fn='clf_det_adv',
                        batch_size=x_test_others.shape[0], distance_type=args.distance_type)

# config = tf.ConfigProto(device_count = {'GPU': 0})
config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.2))
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  saver.restore(sess, os.path.join('checkpoints/classifier_{}_dim{}_{}'.format(args.m, args.dim, args.dataset)))
  print('loaded ' + 'checkpoints/classifier_{}'.format(args.m))

  # detector_saver.restore(sess, 'checkpoints/detector_softmax_lconfidence-0.0_rconfidence-500.0_distance-1.5_dim-784_mnist-999')
  # print('loaded detector')

  if plot:
    attack = PGDAttack(classifier, detector,
                       max_distance=args.d,
                       num_steps=100, step_size=0.01, random_start=True,
                       x_min=x_min, x_max=x_max, batch_size=x_train.shape[0],
                       loss_fn='det_adv', distance_type=args.distance_type)
    x_train_adv, y_train_adv, detector_logits = attack.perturb(x_train, y_train, sess)
    plot_classifier(x_train, y_train, None, None, classifier, classifier.logits, sess, x_train_adv, y_train_adv, save_name='plots/classifier.pdf')

  if args.test_model is not None:
    detector_saver.restore(sess, args.test_model)
    print('loaded {}'.format(args.test_model))

    # x_test_target = x_train_target[:1000]
    # x_test_others = x_train_others[:1000]

    #x_test_target = x_test_target[:1000]
    x_test_others = x_test_others[:1000]
    y_test_others = y_test_others[:1000]

    #x_test_target = x_test_target[:1000]
    #x_test_others = x_test_target
    #y_test_others = y_test_target

    if args.test_stats:
      sample_idx = 0
      x_test_others = np.repeat(x_test_others[sample_idx:sample_idx+1], 100, axis=0)
      y_test_others = np.repeat(y_test_others[sample_idx:sample_idx+1], 100, axis=0)

      test_attack = PGDAttack(classifier, detector,
                              max_distance=args.d,
                              num_steps=args.test_steps, step_size=args.test_step_size, random_start=True,
                              x_min=x_min, x_max=x_max, batch_size=x_test_others.shape[0],
                              loss_fn='clf_det_adv', target_class=None, distance_type=args.distance_type)

      steps = np.array([1] + list(range(10, 1000, 10)))
      detector_stats = []
      for s in steps:
        print('computing step {}'.format(s))
        test_attack.num_steps = s
        x_test_others_adv, test_dist, detector_logits = test_attack.perturb(x_test_others, y_test_others, sess)
        detector_stats.append(detector_logits)
      np.savez('detector_logits_stepsize{}'.format(args.test_step_size),
               steps=steps, detector_logits=np.array(detector_stats))
    else:
      test_attack = PGDAttack(classifier, detector,
                              max_distance=args.d,
                              num_steps=args.test_steps, step_size=args.test_step_size, random_start=False,
                              x_min=x_min, x_max=x_max, batch_size=x_test_others.shape[0],
                              loss_fn='clf_det_adv', target_class=None, distance_type=args.distance_type)

      # attack = np.load('/home/xy4cm/Projects/ICML2019/mnist_challenge_origin/natural_attack.npy')
      # x_test_others_adv = attack
      # test_dist = np.zeros(x_test_others_adv.shape[0])
      x_test_others_adv, test_dist, detector_logits = test_attack.perturb(x_test_others, y_test_others, sess)

    print('x_test_others_adv.shape {}'.format(x_test_others_adv.shape))
    assert x_test_others_adv.size > 0
   
    classifier_preds = sess.run(classifier.predictions, feed_dict={classifier.x_input: x_test_others_adv})
    # x_test_others_adv = x_test_others_adv[classifier_preds==args.target_class]
    # print(classifier_preds[classifier_preds==args.target_class])
    # print('misclassified as 9 x_test_others_adv.shape {}'.format(x_test_others_adv.shape))


    # x_test_others_adv, test_dist = x_test_others, np.zeros(x_test_others.shape[0])

    x_test_with_adv = np.concatenate([x_test_target, x_test_others_adv])
    y_test_with_adv = np.concatenate(
      [np.ones(x_test_target.shape[0], dtype=np.int64), np.zeros(x_test_others_adv.shape[0], dtype=np.int64)])

    tic = time.time()
    detector_logits, y_pred_test, f_score, precision, recall, acc, balanced_acc, tpr, fpr = sess.run(
      [detector.logits, detector.predictions, detector.f_score, detector.precision, detector.recall, detector.accuracy,
       detector.balanced_accuracy, detector.true_positive_rate, detector.false_positive_rate],
      feed_dict={detector.x_input: x_test_with_adv, detector.y_input: y_test_with_adv})
    toc = time.time()
    print(
      'test  f-score {:.4f}, precision {:.4f}, recall {:.4f}, acc {:.4f}, balanced acc {:.4f}, tpr {:.4f} fpr {:.4f} dist<={:.4f} {:.4f} {:.4f}/{:.4f}'.format(
        f_score, precision, recall, acc, balanced_acc, tpr, fpr,
        args.d, (test_dist <= args.d + 1e-6).mean(), test_dist.mean(), test_dist.std()), end='|')
    pos = np.sum(y_test_with_adv)
    true_pos = np.sum(np.bitwise_and(y_pred_test, y_test_with_adv))
    # print('pos {:.4f}, true pos {:.4f}, adv {}, miss {}, nat {}, time {:.2f}ms'.format(np.sum(y_pred_test),
    #                                                            true_pos, pos, (pos - true_pos), np.sum(1 - y_test_with_adv),
    #                                                            1000*(toc - tic)))


    false_pos_mask = np.bitwise_and(y_pred_test, 1 - y_test_with_adv).astype(np.bool)
    false_pos = x_test_with_adv[false_pos_mask]

    # with np.printoptions(suppress=True, threshold=np.inf):
    #   mask = np.bitwise_and(false_pos_mask[x_test_target.shape[0]:], classifier_preds == args.target_class)
    #   print('false pos {}, neg {}'.format(np.sum(false_pos_mask), np.sum(1-y_test_with_adv)))
    #   print('targeted false pos {}'.format(np.sum(mask)))

    fpr, tpr, thresholds = roc_curve(y_test_with_adv, detector_logits)
    roc_auc = auc(fpr, tpr)
    print('\nroc_auc {:.4f}'.format(roc_auc))

    plt.plot(fpr, tpr, label='class {} (AUC = {:.4f})'.format(args.target_class, roc_auc))
    plt.xlim([-0.05, 1.05])
    plt.ylim([0.9, 1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    plt.show()

    #fig, axes = plt.subplots(nrows=10, ncols=10, figsize=(10, 10))
    #for img, ax in zip(false_pos, axes.flatten()):
    #  ax.imshow(img.reshape((28, 28)), cmap='gray')
    #plt.show()

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

  for epoch in range(1, 51):
    x_train_target = x_train_target[np.random.permutation(num_target)]
    x_train_others = x_train_others[np.random.permutation(num_others)]

    target_seq = range(0, num_target, target_batch_size)
    others_seq = range(0, num_others, others_batch_size)

    x_train_with_adv, y_train_with_adv = [], []

    for target_i, others_i in zip(target_seq, others_seq):
      x_batch_target = x_train_target[target_i: target_i + target_batch_size]
      x_batch_others = x_train_others[others_i: others_i + others_batch_size]
      y_batch_others = y_train_others[others_i: others_i + others_batch_size]

      if x_batch_target.shape[0] != target_batch_size or \
          x_batch_others.shape[0] != others_batch_size:
        continue

      tic = time.time()
      x_batch_others_adv, batch_dist, detector_logits = train_attack.perturb(x_batch_others, y_batch_others, sess, verbose=False)
      toc = time.time()

      classifier_preds = sess.run(classifier.predictions, feed_dict={classifier.x_input: x_batch_others_adv})

      x_batch_with_adv = np.concatenate([x_batch_target, x_batch_others_adv])
      y_batch_with_adv = np.concatenate(
        [np.ones(x_batch_target.shape[0], dtype=np.int64), np.zeros(x_batch_others_adv.shape[0], dtype=np.int64)])

      _, f_score, precision, recall, acc, balanced_acc, tpr, fpr = sess.run(
        [train_step, detector.f_score, detector.precision, detector.recall, detector.accuracy,
         detector.balanced_accuracy, detector.true_positive_rate, detector.false_positive_rate],
        feed_dict={detector.x_input: x_batch_with_adv, detector.y_input: y_batch_with_adv})

      print('epoch {}, target {}/{}, '
            'train f-score {:.4f}, precision {:.4f}, recall {:.4f}, '
            'acc {:.4f}, balanced_acc {:.4f} tpr {:.4f} fpr {:.4f} '
            'dist<={:.4f} {:.4f} {:.4f}/{:.4f} time {:.1f}'.format(
        epoch, target_i, num_target,
        f_score, precision, recall,
        acc, balanced_acc, tpr, fpr, args.d,
        (batch_dist <= args.d + 1e-6).mean(), batch_dist.mean(), batch_dist.std(), 1000*(toc-tic)))

      x_train_with_adv.append(x_batch_with_adv)
      y_train_with_adv.append(y_batch_with_adv)

    x_train_with_adv = np.concatenate(x_train_with_adv)
    y_train_with_adv = np.concatenate(y_train_with_adv)

    train_detector_logits, y_pred_train, f_score, precision, recall, acc, balanced_acc, tpr, fpr = sess.run(
      [detector.logits, detector.predictions, detector.f_score, detector.precision, detector.recall, detector.accuracy,
       detector.balanced_accuracy, detector.true_positive_rate, detector.false_positive_rate],
      feed_dict={detector.x_input: x_train_with_adv, detector.y_input: y_train_with_adv})
    fpr_, tpr_, thresholds = roc_curve(y_train_with_adv, train_detector_logits)
    roc_auc = auc(fpr_, tpr_)
    print('===epoch {}, train auc {:4f}, f-score {:.4f}, precision {:.4f}, recall {:.4f}, acc {:.4f}, balanced_acc {:.4f}, tpr {:.4f} fpr {:.4f}'.format(
      epoch, roc_auc, f_score, precision, recall, acc, balanced_acc, tpr, fpr), end='|')
    print('pos {:.4f}, true pos {:.4f}, target {}, others {}'.format(np.sum(y_pred_train), np.sum(np.bitwise_and(y_pred_train, y_train_with_adv)),
                                                               np.sum(y_train_with_adv), np.sum(1-y_train_with_adv)))

    x_test_others_adv, test_dist, detector_logits = test_attack.perturb(x_test_others, y_test_others, sess)

    x_test_with_adv = np.concatenate([x_test_target, x_test_others_adv])
    y_test_with_adv = np.concatenate(
      [np.ones(x_test_target.shape[0], dtype=np.int64), np.zeros(x_test_others_adv.shape[0], dtype=np.int64)])

    test_detector_logits, y_pred_test, f_score, precision, recall, acc, balanced_acc, tpr, fpr = sess.run(
      [detector.logits, detector.predictions, detector.f_score, detector.precision, detector.recall, detector.accuracy,
       detector.balanced_accuracy, detector.true_positive_rate, detector.false_positive_rate],
      feed_dict={detector.x_input: x_test_with_adv, detector.y_input: y_test_with_adv})
    fpr_, tpr_, thresholds = roc_curve(y_test_with_adv, test_detector_logits)
    roc_auc = auc(fpr_, tpr_)
    print('===epoch {}, test auc {:.4f}, f-score {:.4f}, precision {:.4f}, recall {:.4f}, acc {:.4f}, balanced_acc {:.4f}, tpr {:.4f} fpr {:.4f} '.format(
      epoch, roc_auc, f_score, precision, recall, acc, balanced_acc, tpr, fpr), end='|')

    print('pos {:.4f}, true pos {:.4f}, target {}, others {}'.format(np.sum(y_pred_test),
                                                               np.sum(np.bitwise_and(y_pred_test, y_test_with_adv)),
                                                               np.sum(y_test_with_adv), np.sum(1 - y_test_with_adv)))


    #confidence_bounded_detection(x_test_with_adv, y_test_with_adv, classifier, sess, args.conf_det_thresh)

    if plot:
      plot_confidence_bounded_detection(x_train_with_adv, y_train_with_adv, classifier, sess, args.conf_det_thresh,
                                        save_name='plots/confidence_bounded_detection.pdf')

      plot_detector(x_train, y_train, np.concatenate(x_train_adv), np.concatenate(y_train_adv),
                    detector, sess, save_name='plots/{}_epoch-{}.pdf'.format(model_name, epoch))
      plot_detector(x_test, y_test, x_test_adv, y_test_adv, detector,
                    sess, save_name='plots/{}_epoch-{}-attack.pdf'.format(model_name, epoch), use_noise=False)

      # plot_attack_loss(x_train, y_train, x_train_adv, attack, sess,
      #                  save_name='plots/{}_attack_loss_epoch-{}_attack-noise.pdf'.format(model_name, epoch), use_noise=True)
    detector_saver.save(sess, os.path.join('checkpoints/mnist-one-single_class/', model_name), global_step=epoch)
