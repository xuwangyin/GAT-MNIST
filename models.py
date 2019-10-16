import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class MNISTMLP:
    def __init__(self, hidden_layer_sizes, output_size, var_scope):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.output_size = output_size
        self.var_scope = var_scope

    def forward(self, x):
        with tf.variable_scope(self.var_scope, reuse=tf.AUTO_REUSE):
            for i, hidden_layer_size in enumerate(self.hidden_layer_sizes):
                x = tf.layers.dense(inputs=x,
                                    name='hidden_{}'.format(i),
                                    units=hidden_layer_size,
                                    activation=tf.nn.relu)

            logits = tf.layers.dense(inputs=x,
                                     name='output',
                                     units=self.output_size)
            if self.output_size == 1:
                logits = tf.squeeze(logits)

            return logits


class MNISTConvNet:
    def __init__(self, output_size, var_scope):
        self.output_size = output_size
        self.var_scope = var_scope

    def forward(self, x):
        x = tf.reshape(x, [-1, 28, 28, 1])
        with tf.variable_scope(self.var_scope, reuse=tf.AUTO_REUSE):
            # https://github.com/MadryLab/mnist_challenge/blob/master/model.py
            x = tf.layers.conv2d(inputs=x,
                                 filters=32,
                                 kernel_size=(5, 5),
                                 activation='relu',
                                 padding='same')
            x = tf.nn.max_pool(x,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')
            # x = tf.layers.max_pooling2d(inputs=x, pool_size=(2, 2))
            x = tf.layers.conv2d(inputs=x,
                                 filters=64,
                                 kernel_size=(5, 5),
                                 activation='relu',
                                 padding='same')
            # x = tf.layers.max_pooling2d(inputs=x, pool_size=(2, 2))
            x = tf.nn.max_pool(x,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

            x = tf.reshape(x, [-1, 7 * 7 * 64])
            x = tf.layers.dense(inputs=x, units=1024, activation='relu')
            logits = tf.layers.dense(inputs=x, units=self.output_size)

            if self.output_size == 1:
                logits = tf.squeeze(logits)

            return logits


class Detector(object):
    def __init__(self, var_scope, mode=None, dataset='MNIST'):
        self.output_size = 1
        self.var_scope = var_scope

        self.y_input = tf.placeholder(tf.int64, shape=[None])

        self.input_size = 28 * 28 * 1
        self.x_input = tf.placeholder(tf.float32,
                                      shape=[None, self.input_size])
        self.net = MNISTConvNet(output_size=1, var_scope=var_scope)
        self.logits = self.forward(self.x_input)

        self.y_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.cast(self.y_input, tf.float32), logits=self.logits)

        self.xent = tf.reduce_mean(self.y_xent)

        self.predictions = tf.cast(self.logits > 0, tf.int64)

        correct_prediction = tf.equal(self.predictions, self.y_input)

        self.num_correct = tf.reduce_sum(tf.cast(correct_prediction, tf.int64))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        true_positives = tf.bitwise.bitwise_and(self.y_input, self.predictions)
        self.true_positive_rate = tf.reduce_sum(
            true_positives) / tf.reduce_sum(self.y_input)

        false_positives = tf.bitwise.bitwise_and(1 - self.y_input,
                                                 self.predictions)
        self.false_positive_rate = tf.reduce_sum(
            false_positives) / tf.reduce_sum(1 - self.y_input)

        self.recall = self.true_positive_rate
        self.precision = tf.reduce_sum(true_positives) / tf.reduce_sum(
            self.predictions)

        self.f_score = 2 * (self.precision * self.recall) / (self.precision +
                                                             self.recall)

        # TODO validate formulation
        self.balanced_accuracy = 0.5 * (self.true_positive_rate +
                                        (1.0 - self.false_positive_rate))

        # self.x_input_nat = tf.boolean_mask(self.x_input, tf.equal(self.y_input, 0))
        # self.x_input_adv = tf.boolean_mask(self.x_input, tf.equal(self.y_input, 1))

    def forward(self, x):
        return self.net.forward(x)


class Classifier(object):
    def __init__(self, var_scope, mode=None, dataset='MNIST'):
        self.var_scope = var_scope
        self.y_input = tf.placeholder(tf.int64, shape=[None])
        self.output_size = 10

        self.input_size = 28 * 28 * 1
        self.x_input = tf.placeholder(tf.float32,
                                      shape=[None, self.input_size])
        self.net = MNISTConvNet(output_size=self.output_size,
                                var_scope=var_scope)
        self.logits = self.forward(self.x_input)

        self.y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.y_input, logits=self.logits)

        self.xent = tf.reduce_mean(self.y_xent)

        self.predictions = tf.argmax(self.logits, 1)

        correct_prediction = tf.equal(self.predictions, self.y_input)

        self.num_correct = tf.reduce_sum(tf.cast(correct_prediction, tf.int64))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def forward(self, x):
        return self.net.forward(x)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x.astype(np.float64)))


class BayesClassifier(object):
    def __init__(self, detectors):
        self.y_input = tf.placeholder(tf.int64, shape=[None])
        self.output_size = 10
        self.detectors = detectors

        self.input_size = 28 * 28 * 1
        self.x_input = tf.placeholder(tf.float32,
                                      shape=[None, self.input_size])

        self.logits = self.forward(self.x_input)

        self.y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.y_input, logits=self.logits)

        self.xent = tf.reduce_mean(self.y_xent)

        self.predictions = tf.argmax(self.logits, 1)

        correct_prediction = tf.equal(self.predictions, self.y_input)

        self.num_correct = tf.reduce_sum(tf.cast(correct_prediction, tf.int64))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.ths = np.linspace(1e-6, 1, 1000)
        self.logit_ths = np.linspace(-250.0, 50.0, 1000)

    def forward(self, x):
        # shape: [batch, num_classes]
        return tf.stack([d.net.forward(x) for d in self.detectors], axis=1)

    def nat_accs(self, x_nat, y, sess):
        """Accuracies on natural dataset."""
        nat_logits, nat_preds = sess.run([self.logits, self.predictions],
                                         feed_dict={self.x_input: x_nat})
        # p_x = np.mean(sigmoid(nat_logits), axis=1)
        p_x = np.max(nat_logits, axis=1)
        nat_accs = [
            np.logical_and(p_x > th, nat_preds == y).mean()
            for th in self.logit_ths
        ]
        return nat_accs

    def nat_tpr(self, x_nat, sess):
        """True positive rates on natural dataset."""
        nat_logits, nat_preds = sess.run([self.logits, self.predictions],
                                         feed_dict={self.x_input: x_nat})
        # p_x = np.mean(sigmoid(nat_logits), axis=1)
        print('nat logits min/max {}/{}'.format(nat_logits.min(),
                                                nat_logits.max()))
        p_x = np.max(nat_logits, axis=1)
        nat_tpr = [(p_x > th).mean() for th in self.logit_ths]
        for th in self.logit_ths:
            if (p_x > th).mean() < 0.95:
                print('threshold at 0.95 tpr: {}'.format(th))
                break
        return nat_tpr

    def adv_error(self, x_adv, y, sess):
        """The error on perturbed dataset."""
        adv_logits, adv_preds = sess.run([self.logits, self.predictions],
                                         feed_dict={self.x_input: x_adv})
        # p_x = np.mean(sigmoid(adv_logits), axis=1)
        # print('adv logits min/max {}/{}'.format(adv_logits.min(), adv_logits.max()))
        p_x = np.max(adv_logits, axis=1)
        adv_error = [
            np.logical_and(p_x > th, adv_preds != y).mean()
            for th in self.logit_ths
        ]
        return adv_error

    def adv_fpr(self, x_adv, y, sess):
        """False positive rates on perturbed dataset."""
        return self.adv_error(x_adv, y, sess)


class PGDAttack:
    """Base class for various attack methods"""
    def __init__(self, max_distance, num_steps, step_size, random_start, x_min,
                 x_max, batch_size, norm, optimizer):
        self.max_distance = max_distance
        self.num_steps = num_steps
        self.step_size = step_size
        self.rand = random_start
        self.x_min = x_min
        self.x_max = x_max
        self.norm = norm
        self.optimizer = optimizer
        self.batch_size = batch_size

        input_size = 28 * 28
        self.delta = tf.Variable(np.zeros((batch_size, input_size)),
                                 dtype=tf.float32,
                                 name='delta')
        self.x0 = tf.Variable(np.zeros((batch_size, input_size)),
                              dtype=tf.float32,
                              name='x0')
        self.y = tf.Variable(np.zeros(batch_size), dtype=tf.int64, name='y')
        self.c_constants = tf.Variable(np.zeros(batch_size),
                                       dtype=tf.float32,
                                       name='c_constants')

        self.delta_input = tf.placeholder(dtype=tf.float32,
                                          shape=[batch_size, input_size],
                                          name='delta_input')
        self.x0_input = tf.placeholder(dtype=tf.float32,
                                       shape=[batch_size, input_size],
                                       name='x0_input')
        self.y_input = tf.placeholder(dtype=tf.int64,
                                      shape=[batch_size],
                                      name='delta_input')
        self.c_constants_input = tf.placeholder(dtype=tf.float32,
                                                shape=[batch_size],
                                                name='c_constants_input')

        self.assign_delta = self.delta.assign(self.delta_input)
        self.assign_x0 = self.x0.assign(self.x0_input)
        self.assign_y = self.y.assign(self.y_input)
        self.assign_c_constants = self.c_constants.assign(
            self.c_constants_input)

        self.x = self.x0 + self.delta
        ord = {'L2': 2, 'Linf': np.inf}[norm]
        self.dist = tf.norm(self.x - self.x0, ord=ord, axis=1)

    def setup_optimizer(self):
        if self.optimizer == 'adam':
            # Setup the adam optimizer and keep track of created variables
            start_vars = set(x.name for x in tf.global_variables())
            optimizer = tf.train.AdamOptimizer(learning_rate=self.step_size,
                                               name='attack_adam')
            # This term measures the perturbation size.
            # It's supposed to be used when computing minimum perturbation adversarial examples.
            # When performing norm-constrained attacks, self.c_constant should be set to 0.
            dist_term = tf.reduce_sum(
                self.c_constants *
                tf.reduce_sum(tf.square(self.delta), axis=1))
            self.train_step = optimizer.minimize(self.loss + dist_term,
                                                 var_list=[self.delta])
            end_vars = tf.global_variables()
            new_vars = [x for x in end_vars if x.name not in start_vars]
            self.init = tf.variables_initializer(new_vars)
        elif self.optimizer == 'normgrad':
            # Note the minimum pertubation objective is not implemented here
            if self.norm == 'Linf':
                self.train_step = self.delta.assign(
                    self.delta + self.step_size *
                    tf.sign(tf.gradients(-self.loss, self.delta)[0]))
            else:
                grad = tf.gradients(-self.loss, self.delta)[0]
                grad_norm = tf.norm(grad, axis=1, keepdims=True)
                grad_norm = tf.clip_by_value(grad_norm,
                                             np.finfo(float).eps, np.inf)
                self.train_step = self.delta.assign(self.delta +
                                                    self.step_size * grad /
                                                    grad_norm)

        with tf.control_dependencies([self.train_step]):
            # following https://adversarial-ml-tutorial.org/adversarial_examples/
            delta_ = tf.minimum(tf.maximum(self.delta, self.x_min - self.x0),
                                self.x_max - self.x0)
            if self.norm == 'L2':
                norm = tf.norm(delta_, axis=1, keepdims=True)
                # TODO use np.inf instead of tf.reduce_max(norm)
                # delta_ = delta_ * self.max_distance / tf.clip_by_value(norm, clip_value_min=self.max_distance,
                #                                                        clip_value_max=tf.reduce_max(norm))
                bound_norm = tf.clip_by_value(
                    norm,
                    clip_value_min=np.finfo(float).eps,
                    clip_value_max=self.max_distance)
                delta_ = delta_ * bound_norm / tf.clip_by_value(
                    norm,
                    clip_value_min=np.finfo(float).eps,
                    clip_value_max=np.inf)
            else:
                delta_ = tf.clip_by_value(delta_, -self.max_distance,
                                          self.max_distance)
            self.calibrate_delta = self.delta.assign(delta_)

    def perturb(self, x_nat, y, sess, c_constants=None, verbose=False):
        delta = np.zeros_like(x_nat)
        if self.rand:
            if self.norm == 'L2':
                delta = np.random.randn(*x_nat.shape)
                scale = np.random.uniform(low=0.0,
                                          high=self.max_distance,
                                          size=[delta.shape[0], 1])
                delta = scale * delta / np.linalg.norm(
                    delta, axis=1, keepdims=True)
            else:
                delta = np.random.uniform(-self.max_distance,
                                          self.max_distance, x_nat.shape)
            # # This clips (x_nat+delta) to (x_min, x_max), but in practise I found it not neccessary
            # delta = np.minimum(np.maximum(delta, self.x_min - x_nat), self.x_max - x_nat)

        if self.optimizer == 'adam':
            sess.run(self.init)

        if c_constants is None:
            c_constants = np.zeros(x_nat.shape[0])

        if y is None:
            sess.run(
                [self.assign_delta, self.assign_x0, self.assign_c_constants],
                feed_dict={
                    self.delta_input: delta,
                    self.x0_input: x_nat,
                    self.c_constants_input: c_constants
                })
        else:
            sess.run(
                [
                    self.assign_delta, self.assign_x0, self.assign_y,
                    self.assign_c_constants
                ],
                feed_dict={
                    self.delta_input: delta,
                    self.x0_input: x_nat,
                    self.y_input: y,
                    self.c_constants_input: c_constants
                })

        for i in range(self.num_steps):
            sess.run([self.train_step, self.calibrate_delta])

        return sess.run(self.x)

    def batched_perturb(self, x, y, sess):
        adv = []
        for i in range(0, x.shape[0], self.batch_size):
            adv.append(
                self.perturb(x[i:i + self.batch_size],
                             y[i:i + self.batch_size], sess))
        return np.concatenate(adv)


class PGDAttackDetector(PGDAttack):
    def __init__(self, detector, **kwargs):
        super().__init__(**kwargs)
        self.detector_logits = detector.forward(self.x)
        if kwargs['optimizer'] == 'normgrad':
            # normgrad optimizes cross-entropy loss (optimize logit outputs makes no difference)
            labels = tf.zeros_like(self.detector_logits)
            self.loss = -tf.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=labels, logits=self.detector_logits))
        else:
            self.loss = tf.reduce_sum(-self.detector_logits)
        self.setup_optimizer()


class PGDAttackClassifier(PGDAttack):
    def __init__(self, classifier, loss_fn, targeted=False, **kwargs):
        super().__init__(**kwargs)
        if loss_fn == 'xent':
            logits = classifier.forward(self.x)
            y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.y, logits=logits)
            self.loss = -tf.reduce_sum(y_xent)
            if targeted:
                self.loss = tf.reduce_sum(y_xent)
        elif loss_fn == 'cw':
            logits = classifier.forward(self.x)
            label_mask = tf.one_hot(self.y,
                                    classifier.output_size,
                                    dtype=tf.float32)
            correct_logit = tf.reduce_sum(label_mask * logits, axis=1)
            wrong_logit = tf.reduce_max(
                (1 - label_mask) * logits - 1e4 * label_mask, axis=1)
            if isinstance(classifier, BayesClassifier):
                assert not targeted
                self.loss = tf.reduce_sum(-wrong_logit)
            else:
                self.loss = tf.reduce_sum(correct_logit - wrong_logit)
                if targeted:
                    self.loss = tf.reduce_sum(-correct_logit)
        self.setup_optimizer()


class PGDAttackCombined(PGDAttack):
    def __init__(self, classifier, bayes_classifier, loss_fn, **kwargs):
        super().__init__(**kwargs)
        assert isinstance(classifier, Classifier)
        assert isinstance(bayes_classifier, BayesClassifier)
        clf_logits = classifier.forward(self.x)
        det_logits = bayes_classifier.forward(self.x)

        label_mask = tf.one_hot(self.y,
                                classifier.output_size,
                                dtype=tf.float32)
        clf_correct_logit = tf.reduce_sum(label_mask * clf_logits, axis=1)
        clf_wrong_logit = tf.reduce_max(
            (1 - label_mask) * clf_logits - 1e4 * label_mask, axis=1)
        det_wrong_logit = tf.reduce_max(
            (1 - label_mask) * det_logits - 1e4 * label_mask, axis=1)

        if loss_fn == 'cw':
            with_det_logits = (-det_wrong_logit + 1) * tf.reduce_max(
                clf_logits, axis=1)
            correct_logit_with_det = tf.maximum(clf_correct_logit,
                                                with_det_logits)
            self.loss = tf.reduce_sum(correct_logit_with_det - clf_wrong_logit)
        else:
            mask = tf.cast(tf.greater(clf_wrong_logit, clf_correct_logit),
                           tf.float32)
            self.loss = tf.reduce_sum(mask * (-det_wrong_logit) +
                                      (1.0 - mask) *
                                      (clf_correct_logit - clf_wrong_logit))

        self.setup_optimizer()


# The followings are adapted or copied from https://github.com/MadryLab/mnist_challenge

# MIT License
#
# Copyright (c) 2017 Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


class MadryLinfPGDAttackDetector:
    def __init__(self, detector, epsilon, num_steps, step_size, random_start,
                 x_min, x_max):
        """Attack parameter initialization. The attack performs k steps of
       size a, while always staying within epsilon from the initial
       point."""
        self.detector = detector
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.rand = random_start
        self.x_min = x_min
        self.x_max = x_max

        # https://github.com/MadryLab/mnist_challenge/blob/master/model.py#L48
        # https://github.com/MadryLab/mnist_challenge/blob/master/pgd_attack.py#L38
        loss = tf.reduce_sum(detector.y_xent)
        self.grad = tf.gradients(loss, detector.x_input)[0]

    def perturb(self, x_nat, sess):
        """Given a set of examples (x_nat, y), returns a set of adversarial
       examples within epsilon of x_nat in l_infinity norm."""
        if self.rand:
            x = x_nat + np.random.uniform(-self.epsilon, self.epsilon,
                                          x_nat.shape)
            #x = np.clip(x, self.x_min, self.x_max) # ensure valid pixel range
        else:
            x = np.copy(x_nat)
        y = np.zeros(x.shape[0], dtype=np.int64)

        for i in range(self.num_steps):
            grad = sess.run(self.grad,
                            feed_dict={
                                self.detector.x_input: x,
                                self.detector.y_input: y
                            })

            x = np.add(x,
                       self.step_size * np.sign(grad),
                       out=x,
                       casting='unsafe')

            x = np.clip(x, x_nat - self.epsilon, x_nat + self.epsilon)
            x = np.clip(x, self.x_min, self.x_max)  # ensure valid pixel range

        return x, np.linalg.norm(x - x_nat, ord=np.inf, axis=1), sess.run(
            self.detector.logits, feed_dict={self.detector.x_input: x})


class MadryLinfPGDAttackClassifier:
    def __init__(self, classifier, epsilon, num_steps, step_size, random_start,
                 loss_func, x_min, x_max):
        """Attack parameter initialization. The attack performs k steps of
       size a, while always staying within epsilon from the initial
       point."""
        self.classifier = classifier
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.rand = random_start
        self.x_min = x_min
        self.x_max = x_max

        if loss_func == 'xent':
            loss = classifier.xent
        elif loss_func == 'cw':
            label_mask = tf.one_hot(classifier.y_input,
                                    10,
                                    on_value=1.0,
                                    off_value=0.0,
                                    dtype=tf.float32)
            correct_logit = tf.reduce_sum(label_mask * classifier.logits,
                                          axis=1)
            wrong_logit = tf.reduce_max(
                (1 - label_mask) * classifier.logits - 1e4 * label_mask,
                axis=1)
            loss = -tf.nn.relu(correct_logit - wrong_logit + 50)
        else:
            print('Unknown loss function. Defaulting to cross-entropy')
            loss = classifier.xent

        self.grad = tf.gradients(loss, classifier.x_input)[0]

    def perturb(self, x_nat, y, sess):
        """Given a set of examples (x_nat, y), returns a set of adversarial
       examples within epsilon of x_nat in l_infinity norm."""
        if self.rand:
            x = x_nat + np.random.uniform(-self.epsilon, self.epsilon,
                                          x_nat.shape)
            #x = np.clip(x, self.x_min, self.x_max) # ensure valid pixel range
        else:
            x = np.copy(x_nat)

        for i in range(self.num_steps):
            grad = sess.run(self.grad,
                            feed_dict={
                                self.classifier.x_input: x,
                                self.classifier.y_input: y
                            })

            x = np.add(x,
                       self.step_size * np.sign(grad),
                       out=x,
                       casting='unsafe')

            x = np.clip(x, x_nat - self.epsilon, x_nat + self.epsilon)
            x = np.clip(x, self.x_min, self.x_max)  # ensure valid pixel range

        return x


class MadryPGDAttackDetector(PGDAttack):
    def __init__(self, target_class, **kwargs):
        super().__init__(**kwargs)

        self.x_image = tf.reshape(self.x, [-1, 28, 28, 1])

        with tf.variable_scope('madry', reuse=tf.AUTO_REUSE):
            # first convolutional layer
            W_conv1 = self._weight_variable([5, 5, 1, 32])
            b_conv1 = self._bias_variable([32])

            h_conv1 = tf.nn.relu(self._conv2d(self.x_image, W_conv1) + b_conv1)
            h_pool1 = self._max_pool_2x2(h_conv1)

            # second convolutional layer
            W_conv2 = self._weight_variable([5, 5, 32, 64])
            b_conv2 = self._bias_variable([64])

            h_conv2 = tf.nn.relu(self._conv2d(h_pool1, W_conv2) + b_conv2)
            h_pool2 = self._max_pool_2x2(h_conv2)

            # first fully connected layer
            W_fc1 = self._weight_variable([7 * 7 * 64, 1024])
            b_fc1 = self._bias_variable([1024])

            h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

            # output layer
            W_fc2 = self._weight_variable([1024, 10])
            b_fc2 = self._bias_variable([10])

            self.pre_softmax = tf.matmul(h_fc1, W_fc2) + b_fc2

            self.y_pred = tf.argmax(self.pre_softmax, 1)

            correct_prediction = tf.equal(self.y_pred, self.y)

            self.num_correct = tf.reduce_sum(
                tf.cast(correct_prediction, tf.int64))
            self.accuracy = tf.reduce_mean(
                tf.cast(correct_prediction, tf.float32))

        self.detector_logits = self.pre_softmax[:, target_class]
        if kwargs['optimizer'] == 'normgrad':
            labels = tf.zeros_like(self.detector_logits)
            self.loss = -tf.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=labels, logits=self.detector_logits))
        else:
            self.loss = tf.reduce_sum(-self.detector_logits)
        self.setup_optimizer()

    @staticmethod
    def _weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    @staticmethod
    def _bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    @staticmethod
    def _conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    @staticmethod
    def _max_pool_2x2(x):
        return tf.nn.max_pool(x,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')


class MadryPGDAttackClassifier(PGDAttack):
    def __init__(self, loss_fn, **kwargs):
        super().__init__(**kwargs)

        self.x_image = tf.reshape(self.x, [-1, 28, 28, 1])

        with tf.variable_scope('madry', reuse=tf.AUTO_REUSE):
            # first convolutional layer
            W_conv1 = self._weight_variable([5, 5, 1, 32])
            b_conv1 = self._bias_variable([32])

            h_conv1 = tf.nn.relu(self._conv2d(self.x_image, W_conv1) + b_conv1)
            h_pool1 = self._max_pool_2x2(h_conv1)

            # second convolutional layer
            W_conv2 = self._weight_variable([5, 5, 32, 64])
            b_conv2 = self._bias_variable([64])

            h_conv2 = tf.nn.relu(self._conv2d(h_pool1, W_conv2) + b_conv2)
            h_pool2 = self._max_pool_2x2(h_conv2)

            # first fully connected layer
            W_fc1 = self._weight_variable([7 * 7 * 64, 1024])
            b_fc1 = self._bias_variable([1024])

            h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

            # output layer
            W_fc2 = self._weight_variable([1024, 10])
            b_fc2 = self._bias_variable([10])

            self.pre_softmax = tf.matmul(h_fc1, W_fc2) + b_fc2

            self.y_pred = tf.argmax(self.pre_softmax, 1)

            correct_prediction = tf.equal(self.y_pred, self.y)

            self.num_correct = tf.reduce_sum(
                tf.cast(correct_prediction, tf.int64))
            self.accuracy = tf.reduce_mean(
                tf.cast(correct_prediction, tf.float32))

        if loss_fn == 'xent':
            assert False
            self.loss = -tf.reduce_sum(classifier.y_xent)
        elif loss_fn == 'cw':
            label_mask = tf.one_hot(self.y, 10, dtype=tf.float32)
            correct_logit = tf.reduce_sum(label_mask * self.pre_softmax,
                                          axis=1)
            self.wrong_class = tf.argmax(
                (1 - label_mask) * self.pre_softmax - 1e4 * label_mask, axis=1)
            wrong_logit = tf.reduce_max(
                (1 - label_mask) * self.pre_softmax - 1e4 * label_mask, axis=1)
            self.loss = tf.reduce_sum(correct_logit - wrong_logit)
        self.setup_optimizer()

    @staticmethod
    def _weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    @staticmethod
    def _bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    @staticmethod
    def _conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    @staticmethod
    def _max_pool_2x2(x):
        return tf.nn.max_pool(x,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')


class MadryClassifier(object):
    def __init__(self, var_scope):
        self.var_scope = var_scope
        self.output_size = 10
        self.x_input = tf.placeholder(tf.float32, shape=[None, 784])
        self.y_input = tf.placeholder(tf.int64, shape=[None])

        self.pre_softmax = self.forward(self.x_input)
        self.logits = self.pre_softmax

        y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.y_input, logits=self.pre_softmax)

        self.xent = tf.reduce_sum(y_xent)

        self.y_pred = tf.argmax(self.pre_softmax, 1)
        self.predictions = self.y_pred

        correct_prediction = tf.equal(self.y_pred, self.y_input)

        self.num_correct = tf.reduce_sum(tf.cast(correct_prediction, tf.int64))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def forward(self, x):
        with tf.variable_scope(self.var_scope, reuse=tf.AUTO_REUSE):
            # first convolutional layer
            W_conv1 = tf.get_variable('Variable', [5, 5, 1, 32])
            b_conv1 = tf.get_variable('Variable_1', [32])

            x_image = tf.reshape(x, [-1, 28, 28, 1])
            h_conv1 = tf.nn.relu(self._conv2d(x_image, W_conv1) + b_conv1)
            h_pool1 = self._max_pool_2x2(h_conv1)

            # second convolutional layer
            W_conv2 = tf.get_variable('Variable_2', [5, 5, 32, 64])
            b_conv2 = tf.get_variable('Variable_3', [64])

            h_conv2 = tf.nn.relu(self._conv2d(h_pool1, W_conv2) + b_conv2)
            h_pool2 = self._max_pool_2x2(h_conv2)

            # first fully connected layer
            W_fc1 = tf.get_variable('Variable_4', [7 * 7 * 64, 1024])
            b_fc1 = tf.get_variable('Variable_5', [1024])

            h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

            # output layer
            W_fc2 = tf.get_variable('Variable_6', [1024, 10])
            b_fc2 = tf.get_variable('Variable_7', [10])

            pre_softmax = tf.matmul(h_fc1, W_fc2) + b_fc2
            return pre_softmax

    @staticmethod
    def _conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    @staticmethod
    def _max_pool_2x2(x):
        return tf.nn.max_pool(x,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')
