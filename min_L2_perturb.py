import sys
import os
import numpy as np
import tensorflow as tf
from models import BayesClassifier, PGDAttackClassifier, PGDAttackDetector
from eval_utils import BaseDetectorFactory, load_mnist_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)


def update_bound(lower, upper, current, success):
    if success:
        lower = current
    else:
        upper = current
    return lower, upper, (lower + upper) * 0.5


np.random.seed(123)

(x_train, y_train), (x_test, y_test) = load_mnist_data()

# Use eps=0.3 to reproduce esp0.3 result
factory = BaseDetectorFactory(eps=0.5)

batch_size = 100
attack_config = {
    'max_distance': 1000,
    'num_steps': 1000,
    'step_size': 1.0,
    'random_start': False,
    'x_min': 0,
    'x_max': 1.0,
    'batch_size': batch_size,
    'optimizer': 'adam',
    'norm': 'L2'
}
print('attack config: {}'.format(attack_config))

with tf.Session() as sess:
    # Restore variables
    factory.restore_base_detectors(sess)

    base_detectors = factory.get_base_detectors()

    bayes_classifier = BayesClassifier(factory.get_base_detectors())

    # Assign attack target for each sample
    targets = np.zeros(x_test.shape[0], dtype=np.int64)
    for i in range(x_test.shape[0]):
        targets[i] = np.random.choice(np.delete(np.arange(10), y_test[i]))

    all_targets_best_dists = []
    for target in range(0, 10):
        detector = base_detectors[target]
        x_test_sub = x_test[targets == target]
        y_test_sub = y_test[targets == target]
        attack = PGDAttackDetector(detector=detector, **attack_config)

        best_dists = []
        best_adv = []
        for b in range(0, x_test_sub.shape[0], batch_size):
            print('processing batch {}-{}'.format(b, b + batch_size))
            x_batch, y_batch = x_test_sub[b:b + batch_size], y_test_sub[b:b + batch_size]
            if x_batch.shape[0] != batch_size:
                continue
            lowers, uppers = np.zeros(batch_size), np.zeros(batch_size) + 8.0

            # # validate upper bound; to show a lower bound succeeds all while a upper bound fails all
            # c_constants = np.zeros(batch_size) + 0  # replace 8 with 0 to test the lower bound
            # x_batch_adv = attack.perturb(x_batch, y_batch, sess, c_constants=c_constants)
            # adv_logits = sess.run(bayes_classifier.logits, feed_dict={bayes_classifier.x_input: x_batch_adv})
            # success = np.logical_and(np.argmax(adv_logits, 1) != y_batch, np.max(adv_logits, 1) > 3.6)
            # print('sucess {}'.format(success.mean()))
            # continue

            c_constants = np.zeros(batch_size)
            batch_best_dists = np.zeros(batch_size) + 1e9
            batch_best_adv = np.zeros_like(x_batch)
            for depth in range(20):
                x_batch_adv = attack.perturb(x_batch,
                                             y_batch,
                                             sess,
                                             c_constants=c_constants)
                adv_logits = sess.run(
                    bayes_classifier.logits,
                    feed_dict={bayes_classifier.x_input: x_batch_adv})
                success = np.logical_and(
                    np.argmax(adv_logits, 1) != y_batch,
                    np.max(adv_logits, 1) > 3.6)
                dist = np.linalg.norm(x_batch_adv - x_batch, axis=1)
                for i in range(adv_logits.shape[0]):
                    if i == 0:
                        print(
                            'sample {}: lower {}, upper {}, c_constant {}, dist {}, sucess {}'
                            .format(i, lowers[i], uppers[i], c_constants[i],
                                    dist[i], success[i]))
                    lowers[i], uppers[i], c_constants[i] = update_bound(
                        lowers[i], uppers[i], c_constants[i], success[i])
                    if success[i] and dist[i] < batch_best_dists[i]:
                        batch_best_dists[i] = dist[i]
                        batch_best_adv[i] = x_batch_adv[i]
            print('sucess {} dist mean {}'.format(
                (batch_best_dists < 1e8).mean(),
                batch_best_dists[batch_best_dists < 1e8].mean()))
            best_dists.append(batch_best_dists)
            best_adv.append(batch_best_adv)
        best_dists = np.concatenate(best_dists)
        all_targets_best_dists.append(best_dists)
    all_targets_best_dists = np.concatenate(all_targets_best_dists)
    print('sucess {}, dist mean {}'.format(
        (all_targets_best_dists < 1e5).mean(), all_targets_best_dists[all_targets_best_dists < 1e5].mean()))
