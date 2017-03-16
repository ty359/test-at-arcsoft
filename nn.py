import os
import sys

import numpy as np
import tensorflow as tf

class NN:
  def __init__(self, sess, train=0.8):
    self.loss = 0
    self.TRAIN = train
    self.count = 0

  def _get_variable(self, name, shape, wd=.0):
    stddev = 1.0
    _tmp = 1
    for t in shape:
      stddev /= t
      _tmp *= t
    self.count += _tmp
    stddev = stddev ** 0.5
    ret = tf.get_variable(name=name, shape=shape, initializer=tf.truncated_normal_initializer(dtype=tf.float32, mean=0.0, stddev=stddev))
    if wd != .0:
      self.loss += wd * tf.nn.l2_loss(ret)
    return ret

  def fc(self, x, o_size=1, name='fc'):
    with tf.variable_scope(name):
      shape = x.get_shape().as_list()
      batch = shape[0]
      chans = 1
      for i in shape[1:]:
        chans *= i
      x = tf.reshape(x, [batch, chans])
      k = _get_variable('weights', [chans, o_size])
      b = _get_variable('biases', [o_size])
      return tf.nn.relu(tf.matmul(x, k) + b)

  def pool(self, x, name='pool'):
    with tf.variable_scope(name):
      return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

  def unpool(self, x, name='unpool', shape=None):
    with tf.variable_scope(name):
      if not shape:
        shape = x.get_shape().as_list()
        shape = [2*shape[1], 2*shape[2]]
      else:
        shape = [shape[1], shape[2]]
      return tf.image.resize_nearest_neighbor(x, shape, name=name)

  def conv(self, x, o_size, name='conv', ksize=3):
    with tf.variable_scope(name):
      shape = x.get_shape().as_list()
      k = self._get_variable('weights', [ksize, ksize, shape[3], o_size])
      b = self._get_variable('biases', [1, 1, 1, o_size])
      return tf.nn.relu(tf.nn.conv2d(x, k, [1, 1, 1 ,1], 'SAME') + b)

  def deconv(self, x, o_shape, name='deconv', ksize=4, stride=2):
    pass

  def norm(self, x, name='norm'):
    return x

  def gen_loss(self, x, y):
    self.loss += tf.nn.l2_loss(x - y)

  def dropout(self, x, name='dropout'):
    with tf.variable_scope(name):
      if self.TRAIN:
        return tf.nn.dropout(x, self.TRAIN)
      else:
        return x

  def layer_add(self, x, y, name='layer_add'):
    with tf.variable_scope(name):
      shape = x.get_shape().as_list()
      ch = shape[-1]
      kx = self._get_variable('x_weights', [1, 1, 1, ch])
      ky = self._get_variable('y_weights', [1, 1, 1, ch])
      b = self._get_variable('biases', [1, 1, 1, ch])
      return tf.nn.relu(kx * x + ky * y + b)