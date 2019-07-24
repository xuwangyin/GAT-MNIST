import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

def make_generator_model():
    with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
      model = tf.keras.Sequential()
      model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
      model.add(layers.BatchNormalization())
      model.add(layers.LeakyReLU())

      model.add(layers.Reshape((7, 7, 256)))
      assert model.output_shape == (None, 7, 7, 256) # Note: None is the batch size

      model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
      assert model.output_shape == (None, 7, 7, 128)
      model.add(layers.BatchNormalization())
      model.add(layers.LeakyReLU())

      model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
      assert model.output_shape == (None, 14, 14, 64)
      model.add(layers.BatchNormalization())
      model.add(layers.LeakyReLU())

      model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='sigmoid'))
      assert model.output_shape == (None, 28, 28, 1)

      return model

class MNISTConvNet:
  def __init__(self, var_scope):
    self.var_scope = var_scope

  def forward(self, x):
    with tf.variable_scope(self.var_scope, reuse=tf.AUTO_REUSE):
      # https://github.com/MadryLab/mnist_challenge/blob/master/model.py
      x = tf.layers.conv2d(inputs=x, filters=32, kernel_size=(5, 5), activation='relu', padding='same')
      x = tf.nn.max_pool(x, ksize = [1,2,2,1], strides=[1,2,2,1], padding='SAME')
      x = tf.layers.conv2d(inputs=x, filters=64, kernel_size=(5, 5), activation='relu', padding='same')
      x = tf.nn.max_pool(x, ksize = [1,2,2,1], strides=[1,2,2,1], padding='SAME')

      x = tf.reshape(x, [-1, 7 * 7 * 64])
      x = tf.layers.dense(inputs=x, units=1024, activation='relu')
      logits = tf.layers.dense(inputs=x, units=1)

      logits = tf.squeeze(logits)

      return logits

generator = make_generator_model()
noise_input = tf.placeholder(tf.float32, shape=[None, 100])
fake = generator(noise_input)
target = 8
scope = 'detector-class{}'.format(target)
detector = MNISTConvNet(var_scope=scope)
output = tf.reduce_mean(detector.forward(fake))

detector_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
saver = tf.train.Saver(var_list=detector_vars)

generator_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

train_step = tf.train.AdamOptimizer(5e-4).minimize(-output, var_list=generator_vars)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  saver.restore(sess, 'checkpoints/mnist/detector_Linf_0.3/ovr-steps100-adam-noclip-balanced/ovr_class{}_Linf_distance0.3-54'.format(target))
  #saver.restore(sess, 'checkpoints/mnist/detector_Linf_0.3/ovr-steps100-adam-noclip-balanced/ovr_class{}_Linf_distance0.3-64'.format(target))
  for i in range(6000):
    noise = np.random.randn(64, 100)
    pred_before = sess.run(output, feed_dict={noise_input: noise})
    sess.run(train_step, feed_dict={noise_input: noise})
    pred_after = sess.run(output, feed_dict={noise_input: noise})
    if i % 10 == 0:
      print('before {}, after {}'.format(pred_before.mean(), pred_after.mean()))

  noise = np.random.randn(100, 100)
  fake = sess.run(fake, feed_dict={noise_input: noise})
  print(fake.shape)
  print(fake.min(), fake.max())
  fig, axes = plt.subplots(nrows=10, ncols=10)
  for im, ax in zip(fake, axes.ravel()):
    ax.imshow(im.reshape((28, 28)), cmap='gray')
    ax.set_axis_off()
  plt.show()
  
