import numpy as np
from sklearn.datasets import make_moons, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import tensorflow as tf

h = .02  # step size in the mesh
colors = ['#FF0000', '#00FF00', '#0000FF']
cm_bright = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])


def two_moons_dataset():
  x, y = make_moons(noise=0.1, random_state=0, n_samples=300)
  x = StandardScaler().fit_transform(x)
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3, random_state=123)
  # scaler = StandardScaler()
  # x_train = scaler.fit_transform(x_train)
  # x_test = scaler.transform(x_test)
  return x_train, y_train, x_test, y_test


def multiclass_dataset(features):
  assert features >= 2
  from sklearn.datasets import make_blobs
  std = 1.0 if features == 2 else 4.5
  # yellow, blue, red
  centers = [[1, 1], [1, -1], [-1, -1]]
  x, y = make_blobs(n_samples=2000, n_features=2, centers=centers,
                    cluster_std=0.3, random_state=42)
  if features > 2:
    noise = np.random.randn(2000, features - 2) * 0.1
    x = np.concatenate([x, noise], axis=1)
  assert x.shape[1] == features
  x = StandardScaler().fit_transform(x)
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
  return x_train, y_train, x_test, y_test, 3

  # x, y = make_classification(n_classes=3, n_samples=1000, n_features=2, random_state=1,
  #                            n_informative=2, n_redundant=0, n_repeated=0,
  #                            n_clusters_per_class=1, class_sep=2.0,
  #                            hypercube=True)
  # x = StandardScaler().fit_transform(x)
  # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3)
  # return x_train, y_train, x_test, y_test


def get_grid(x):
  x_min, x_max = x[:, 0].min() - .5, x[:, 0].max() + .5
  y_min, y_max = x[:, 1].min() - .5, x[:, 1].max() + .5
  xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                       np.arange(y_min, y_max, h))
  return xx, yy


def plot_fn(xx, yy, ax, model, fn, sess, plot_contour=False, plot_contourf=True):
  # TODO remove y_input
  # zz = sess.run(fn, feed_dict={model.x_input: np.c_[xx.ravel(), yy.ravel()],
  #                              model.y_input: np.zeros_like(xx.ravel(), dtype=np.int64)})
  zz = sess.run(fn, feed_dict={model.x_input: np.c_[xx.ravel(), yy.ravel()]})
  zz = zz.reshape(xx.shape)

  if plot_contour:
    # CS = ax.contour(xx, yy, zz, 30, colors='k', alpha=.6, linestyles='solid')
    # ax.clabel(CS, inline=1, fontsize=12)
    # plot the decision boundary
    ax.contour(xx, yy, zz, [0], colors='r', alpha=0.8, linestyles='solid')
  if plot_contourf:
    ax.contourf(xx, yy, zz, 30, cmap='viridis', alpha=.8)


def plot_adversaries(x_train, y_train, x_train_adv, y_train_adv, classifier, sess, save_name=None):
  # xx, yy = get_grid(np.concatenate([x_train, x_test]))
  xx, yy = get_grid(x_train)
  fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(30, 40))
  for class_idx in range(3):
    class_logits = classifier.logits[:, class_idx]
    fns = [classifier.logits[:, o] - class_logits for o in {0, 1, 2} - {class_idx}]
    for fn_idx, fn in enumerate(fns):
      ax = axes[class_idx, fn_idx]

      # plot adverary-optimization function
      plot_fn(xx, yy, ax, classifier, fn, sess, plot_contour=True)

      # plot all training data
      ax.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=ListedColormap(colors),
                 edgecolors='black', s=25)

      # plot adversaries for current class
      x_train_adv_plot = x_train_adv[y_train_adv == class_idx]
      y_train_adv_plot = y_train_adv[y_train_adv == class_idx]
      ax.scatter(x_train_adv_plot[:, 0], x_train_adv_plot[:, 1], c=y_train_adv_plot, cmap=ListedColormap([colors[class_idx]]), marker='x',
                 edgecolors='black', s=25)

      ax.set_xlim(-2.5, 1.8)
      ax.set_ylim(-2.0, 2.5)

    # for i in range(x_train.shape[0]):
    #     delta = x_train_adv[i] - x_train[i]
    #     ax.arrow(*x_train[i], *delta, alpha=0.5, color='white')

  if save_name:
    plt.savefig(save_name)
    print('written {}'.format(save_name))

  # plt.show()


def plot_detector(x_train, y_train, x_train_adv, y_train_adv, detector, sess, save_name=None, use_noise=False):
  figure = plt.figure(figsize=(16, 12))
  ax = plt.subplot(1, 1, 1)

  xx, yy = get_grid(x_train)
  plot_fn(xx, yy, ax, detector, detector.logits, sess, plot_contour=True)

  ax.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=cm_bright,
             edgecolors='black', s=25)

  ax.scatter(x_train_adv[:, 0], x_train_adv[:, 1], c=y_train_adv, cmap=cm_bright, marker='x',
             edgecolors='black', s=25)
  # for i in range(x_train.shape[0]):
  #     delta = x_train_adv[i] - x_train[i]
  #     ax.arrow(*x_train[i], *delta, alpha=0.5, color='white')

  ax.set_xlim(-2.5, 1.8)
  ax.set_ylim(-2.0, 2.5)
  if save_name:
    plt.savefig(save_name)
    print('written {}'.format(save_name))

  # plt.show()


def plot_attack_loss(x_train, y_train, x_train_adv, attack, sess, save_name=None, use_noise=False):
  figure = plt.figure(figsize=(16, 12))
  ax = plt.subplot(1, 1, 1)

  xx, yy = get_grid(x_train)
  plot_fn(xx, yy, ax, attack, attack.loss, sess, plot_contour=False)

  ax.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=cm_bright,
             edgecolors='black', s=25)

  # ax.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap=cm_bright,
  #            alpha=0.6, edgecolors='black', s=25)
  if use_noise:
    ax.scatter(x_train_adv[:, 0] + np.random.uniform(-0.05, 0.05, x_train_adv.shape[0]),
               x_train_adv[:, 1] + np.random.uniform(-0.05, 0.05, x_train_adv.shape[0]),
               c=y_train, cmap=cm_bright, marker='x', edgecolors='black', s=25)
  else:
    ax.scatter(x_train_adv[:, 0], x_train_adv[:, 1], c=y_train, cmap=cm_bright, marker='x',
               edgecolors='black', s=25)
  # for i in range(x_train.shape[0]):
  #     delta = x_train_adv[i] - x_train[i]
  #     ax.arrow(*x_train[i], *delta, alpha=0.5, color='white')

  ax.set_xlim(-2.5, 1.8)
  ax.set_ylim(-2.0, 2.5)
  if save_name:
    plt.savefig(save_name)
    print('written {}'.format(save_name))

  # plt.show()


def plot_classifier(x_train, y_train, x_test, y_test, model, model_fn, sess, x_train_adv=None, y_train_adv=None, save_name=None):
  figure = plt.figure(figsize=(16, 12))
  ax = plt.subplot(1, 1, 1)

  #xx, yy = get_grid(np.concatenate([x_train, x_test]))
  xx, yy = get_grid(x_train)
  plot_fn(xx, yy, ax, model, model.predictions, sess, plot_contour=False, plot_contourf=True)

  ax.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=cm_bright,
             edgecolors='black', s=25)

  if x_test is not None:
    ax.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap=cm_bright,
               alpha=0.5, edgecolors='black', s=25)

  if x_train_adv is not None:
    ax.scatter(x_train_adv[:, 0], x_train_adv[:, 1], c=y_train_adv, cmap=cm_bright, marker='x',
               edgecolors='black', s=25)
    # for i in range(x_train.shape[0]):
    #     delta = x_train_adv[i] - x_train[i]
    #     ax.arrow(*x_train[i], *delta, alpha=0.5, color='white')

  ax.set_xlim(-2.5, 1.8)
  ax.set_ylim(-2.0, 2.5)
  if save_name:
    plt.savefig(save_name)
    print('written {}'.format(save_name))


def plot_confidence_bounded_detection(x, y, classifier, sess, right_conf_thresh, save_name=None):
  figure = plt.figure(figsize=(16, 12))
  ax = plt.subplot(1, 1, 1)

  xx, yy = get_grid(x)
  zz = confidence_bounded_detection(np.c_[xx.ravel(), yy.ravel()], None, classifier, sess, right_conf_thresh)
  zz = np.reshape(zz, xx.shape)
  ax.contourf(xx, yy, zz, 30, cmap='viridis', alpha=.8)

  ax.scatter(x[:, 0], x[:, 1], c=y, cmap=cm_bright,
             edgecolors='black', s=25)

  ax.set_xlim(-2.5, 1.8)
  ax.set_ylim(-2.0, 2.5)
  if save_name:
    plt.savefig(save_name)
    print('written {}'.format(save_name))

  # plt.show()


def confidence_bounded_detection(x, y, classifier, sess, right_conf_thresh):
  logits = sess.run(classifier.logits, feed_dict={classifier.x_input: x})
  sorted = np.sort(logits, axis=1)
  confidence = sorted[:, -1] - sorted[:, -2]
  y_pred = confidence < right_conf_thresh
  if y is not None:
    acc = (y_pred == y).mean()
    tpr = np.sum(np.bitwise_and(y_pred, y)) / np.sum(y)
    fpr = np.sum(np.bitwise_and(y_pred, 1 - y)) / np.sum(1-y)
    recall = tpr
    precision = np.sum(np.bitwise_and(y_pred, y)) / np.sum(y_pred)
    f_score = 2 * (precision * recall) / (precision + recall)
    print('===confidence bounded detection, f-score {:.5f}, precision {:.5f}, recall {:.5f} tpr {:.5f}, fpr {:.5f}, acc {:.5f}'.format(
      f_score, precision, recall, tpr, fpr, acc))
  return y_pred


def confidence_clean(classifier, x, y, left_confidence, right_confidence, sess):

  label_mask = tf.one_hot(classifier.y_input, classifier.output_size, dtype=tf.float32)

  correct_logit = tf.reduce_sum(label_mask * classifier.logits, axis=1)
  wrong_logit = tf.reduce_max((1 - label_mask) * classifier.logits - 1e4 * label_mask, axis=1)
  relative_confidence = correct_logit - wrong_logit

  conf = sess.run(relative_confidence,
                  feed_dict={classifier.x_input: x, classifier.y_input: y})

  mask = conf > right_confidence

  return x[mask], y[mask]


def dataset(args):
  from tensorflow.keras.datasets import mnist
  from sklearn.preprocessing import StandardScaler

  if args.dataset == '3class':
    print('================ using 3 class toy dataset ==================')
    # x_train, y_train, x_test, y_test = two_moons_dataset()
    x_train, y_train, x_test, y_test, num_classes = multiclass_dataset(args.dim)
  else:
    print('================ using mnist dataset ==================')
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    num_classes = 10
    x_train = np.reshape(x_train, [x_train.shape[0], -1])
    x_test = np.reshape(x_test, [x_test.shape[0], -1])
    x_train = x_train.astype(np.float32) / 255.
    x_test = x_test.astype(np.float32) / 255.
    # scaler = StandardScaler()
    # x_train = scaler.fit_transform(x_train.astype(np.float32))
    # x_test = scaler.transform(x_test.astype(np.float32))
    # print(x_train.mean(), x_train.std())
    # print(x_test.mean(), x_test.std())
    args.dim = x_train.shape[1]

  return x_train, y_train, x_test, y_test, num_classes


if __name__ == '__main__':
  X, y, x_test, y_test, num_classes = multiclass_dataset(2)
  # Plot data
  plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], "yo", marker='.')
  plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "bs", marker='.')
  plt.plot(X[:, 0][y == 2], X[:, 1][y == 2], "rd", marker='.')
  plt.xlabel(r"$x_1$", fontsize=20)
  plt.ylabel(r"$x_2$", fontsize=20, rotation=0)
  plt.title("Toy data set\n", fontsize=16)
  plt.show()
