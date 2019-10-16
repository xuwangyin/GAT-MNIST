import sys
import argparse
import os
import pathlib
import time
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from models import Detector, PGDAttackDetector
from eval_utils import load_mnist_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

parser = argparse.ArgumentParser()
parser.add_argument('--target_class',
                    metavar='class of base detector; start from 0',
                    type=int,
                    required=True)
parser.add_argument('--epsilon',
                    metavar='max-distance',
                    type=float,
                    default='0.3')
parser.add_argument('--norm', choices=['L2', 'Linf'], default='Linf')
parser.add_argument('--train_optimizer',
                    choices=['adam', 'normgrad'],
                    default='adam')
parser.add_argument('--test_optimizer',
                    choices=['adam', 'normgrad'],
                    default='adam')
parser.add_argument('--train_steps', type=int, default=100)
parser.add_argument('--step_size', type=float, default=0.01)

args = parser.parse_args()
print(args)

np.random.seed(123)

batch_size = 32

if args.norm == 'L2':
    assert args.epsilon in [2.5, 5.0]
    assert args.step_size == 0.1
if args.norm == 'Linf':
    assert args.epsilon in [0.3, 0.5]
    # assert args.step_size == 0.01

(x_train, y_train), (x_test, y_test) = load_mnist_data()

# use 20% training data to do validation
x_train, x_val, y_train, y_val = train_test_split(x_train,
                                                  y_train,
                                                  test_size=1 / 6.,
                                                  random_state=123)

print('training {}, validation {}'.format(x_train.shape[0], x_val.shape[0]))

x_val_target = x_val[y_val == args.target_class]
x_val_others = x_val[y_val != args.target_class]

x_test_target = x_test[y_test == args.target_class]
x_test_others = x_test[y_test != args.target_class]
y_test_others = y_test[y_test != args.target_class]

x_min, x_max = 0.0, 1.0

detector_var_scope = 'detector-class{}'.format(args.target_class)
detector = Detector(var_scope=detector_var_scope)
detector_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                  scope=detector_var_scope)
detector_saver = tf.train.Saver(var_list=detector_vars, max_to_keep=100)

train_step = tf.train.AdamOptimizer(5e-4, name='detector_adam').minimize(
    detector.xent)

model_name = 'ovr_class{}_{}_distance{}'.format(args.target_class, args.norm,
                                                args.epsilon)

train_attack = PGDAttackDetector(detector,
                                 max_distance=args.epsilon,
                                 num_steps=args.train_steps,
                                 step_size=args.step_size,
                                 random_start=True,
                                 x_min=x_min,
                                 x_max=x_max,
                                 batch_size=batch_size,
                                 norm=args.norm,
                                 optimizer=args.train_optimizer)

test_attack = PGDAttackDetector(detector,
                                max_distance=args.epsilon,
                                num_steps=200,
                                step_size=args.step_size,
                                random_start=False,
                                x_min=x_min,
                                x_max=x_max,
                                batch_size=x_val_others.shape[0],
                                norm=args.norm,
                                optimizer=args.test_optimizer)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(1, 101):
        perm = np.random.permutation(x_train.shape[0])
        x_train, y_train = x_train[perm], y_train[perm]

        for i in range(0, x_train.shape[0], batch_size * 10):
            x_batch = x_train[i:i + batch_size * 10]
            y_batch = y_train[i:i + batch_size * 10]
            x_batch_target = x_batch[y_batch == args.target_class]
            x_batch_others = x_batch[y_batch != args.target_class][:batch_size]
            if x_batch_others.shape[0] != batch_size:
                continue

            tic = time.time()
            x_batch_others_adv = train_attack.perturb(x_batch_others, None,
                                                      sess)
            toc = time.time()

            x_batch_with_adv = np.concatenate(
                [x_batch_target, x_batch_others_adv])
            y_batch_with_adv = np.concatenate([
                np.ones(x_batch_target.shape[0], dtype=np.int64),
                np.zeros(x_batch_others_adv.shape[0], dtype=np.int64)
            ])

            _, batch_detector_logits, f_score, precision, recall, acc, balanced_acc, tpr, fpr = sess.run(
                [
                    train_step, detector.logits, detector.f_score,
                    detector.precision, detector.recall, detector.accuracy,
                    detector.balanced_accuracy, detector.true_positive_rate,
                    detector.false_positive_rate
                ],
                feed_dict={
                    detector.x_input: x_batch_with_adv,
                    detector.y_input: y_batch_with_adv
                })

            fpr_, tpr_, thresholds = roc_curve(y_batch_with_adv,
                                               batch_detector_logits)
            roc_auc = auc(fpr_, tpr_)

            print('epoch {}, {}/{}'.format(epoch, i, x_train.shape[0]),
                  end='| ')

            print(
                'train auc {:.4f}, f-score {:.4f}, precision {:.4f}, recall {:.4f}, '
                .format(roc_auc, f_score, precision, recall),
                end='')
            print('acc {:.4f}, balanced_acc {:.4f} tpr {:.4f} fpr {:.4f},'.
                  format(acc, balanced_acc, tpr, fpr),
                  end='')
            print(' pos {}, neg {}'.format(x_batch_target.shape[0],
                                           x_batch_others_adv.shape[0]))
            # print('dist<={:.4f} {:.4f} {:.4f}/{:.4f} time {:.1f}'.format(args.epsilon,
            # (batch_dist <= args.epsilon + 1e-6).mean(), batch_dist.mean(), batch_dist.std(), 1000*(toc-tic)))

        x_val_others_adv = test_attack.perturb(x_val_others, None, sess)

        x_test_with_adv = np.concatenate([x_val_target, x_val_others_adv])
        y_test_with_adv = np.concatenate([
            np.ones(x_val_target.shape[0], dtype=np.int64),
            np.zeros(x_val_others_adv.shape[0], dtype=np.int64)
        ])

        test_detector_logits, y_pred_test, f_score, precision, recall, acc, balanced_acc, tpr, fpr = sess.run(
            [
                detector.logits, detector.predictions, detector.f_score,
                detector.precision, detector.recall, detector.accuracy,
                detector.balanced_accuracy, detector.true_positive_rate,
                detector.false_positive_rate
            ],
            feed_dict={
                detector.x_input: x_test_with_adv,
                detector.y_input: y_test_with_adv
            })
        fpr_, tpr_, thresholds = roc_curve(y_test_with_adv,
                                           test_detector_logits)
        roc_auc = auc(fpr_, tpr_)
        print(
            '===epoch {}, test auc {:.6f}, f-score {:.4f}, precision {:.4f}, recall {:.4f}, acc {:.4f}, balanced_acc {:.4f}, tpr {:.4f} fpr {:.4f} '
            .format(epoch, roc_auc, f_score, precision, recall, acc,
                    balanced_acc, tpr, fpr),
            end='|')

        print('pos {:.4f}, true pos {:.4f}, target {}, others {}'.format(
            np.sum(y_pred_test),
            np.sum(np.bitwise_and(y_pred_test, y_test_with_adv)),
            np.sum(y_test_with_adv), np.sum(1 - y_test_with_adv)))

        savedir = 'checkpoints/mnist/detector_{}_{}/ovr-steps{}-{}-noclip-balanced/'.format(
            args.norm, args.epsilon, args.train_steps, args.train_optimizer)
        pathlib.Path(savedir).mkdir(parents=True, exist_ok=True)
        detector_saver.save(sess,
                            os.path.join(savedir, model_name),
                            global_step=epoch)
