import os
import sys

import numpy as np
import tensorflow as tf

class NN:
  def __init__(sess, train=0.8):
    self.loss = 0
    self.TRAIN = train

  def _get_variable(name, shape, wd=.0):
    stddev = 1.0
    for t in shape:
      stddev /= t
    stddev = stddev ** 0.5
    ret = tf.get_variable(name=name, shape=shape, initializer=tf.truncated_normal_initializer(dtype=tf.float32, mean=0.0, stddev=stddev))
    if wd != .0:
      self.loss += wd * tf.nn.l2_loss(ret)
    return ret

  def fc(x, o_size=1, name='fc'):
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

  def pool(x, name='pool'):
    with tf.variable_scope(name):
      return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

  def unpool(x, name='unpool'):
    with tf.variable_scope(name):
      shape = x.get_shape().as_list()
      shape = [2*shape[1], 2*shape[2]]
      return tf.image.resize_nearest_neighbor(x, shape, name=name)

  def conv(x, o_size, name='conv', ksize=3):
    with tf.variable_scope(name):
      shape = x.get_shape().as_list()
      k = _get_variable('weights', [ksize, ksize, shape[3], o_size])
      b = _get_variable('biases', [shape[0], shape[1], shape[2], o_size])
      return tf.nn.relu(tf.nn.conv2d(x, k, [1, 1, 1 ,1], 'SAME') + b)

  def deconv(x, o_shape, name='deconv', ksize=4, stride=2):
    pass

  def norm(x, name='norm'):
    return x

  def gen_loss(x, y):
    self.loss += tf.nn.softmax_cross_entropy_with_logits(logits=x, labels=y)

  def dropout(x, name='dropout'):
    with tf.variable_scope(name):
      if self.TRAIN:
        return tf.nn.dropout(x, self.TRAIN)
      else:
        return x

  def layer_add(x, y, name='layer_add'):
    with tf.variable_scope(name):
      shape = x.get_shape().as_list()
      kx = _get_variable('x_weights', shape)
      ky = _get_variable('y_weights', shape)
      return tf.nn.relu(kx * x + ky * y)