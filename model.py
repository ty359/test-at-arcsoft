import os
import sys
import cv2
import random

import numpy as np
import tensorflow as tf

import nn
from predo import *

class model256:
  def __init__(self):
    self.step = 0
    pass

  def init(self, filename=None, folder=None):

    self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)
    self.sess = tf.InteractiveSession()

    if filename:
      self.saver.restore(filename)
    elif folder and tf.train.latest_checkpoint(folder):
      self.saver.restore(self.sess, tf.train.latest_checkpoint(folder))
    else:
      self.sess.run(tf.global_variables_initializer())

    self.summary = tf.summary.merge_all()
    self.summary_writer = tf.summary.FileWriter('logs/', self.sess.graph)


  def build(self, batch_size=1, train_rate=1e-3):
    global SHAPE

    self.x = tf.placeholder(dtype=tf.float32, shape=[batch_size] + SHAPE)
    self.y = tf.placeholder(dtype=tf.float32, shape=[batch_size] + SHAPE)

    with tf.variable_scope('model0'):

      c1 = nn.conv(self.x, 4, 'conv1')
      p1 = nn.pool(c1, 'pool1')

      c2 = nn.conv(p1, 8, 'conv2')
      p2 = nn.pool(c2, 'pool2')

      c3 = nn.conv(p2, 16, 'conv3')
      p3 = nn.pool(c3, 'pool3')

      c4 = nn.conv(p3, 32, 'conv4')
      p4 = nn.pool(c4, 'pool4')

      c5_1 = nn.conv(p4, 64, 'conv5_1')
      c5_2 = nn.conv(c5_1, 64, 'conv5_2')
      p5 = nn.pool(c5_2, 'pool5')

      c6_1 = nn.conv(p5, 128, 'conv6_1')
      c6_2 = nn.conv(c6_1, 96, 'conv6_2')
      dc6_1 = nn.conv(c6_2, 96, 'deconv6_1')
      dc6_2 = nn.conv(dc6_1, 64, 'deconv6_2')

      up5 = nn.layer_add(nn.unpool(dc6_2, 'unpool5_tmp', shape=c5_2.get_shape().as_list()), c5_2, 'unpool5')
      dc5_1 = nn.conv(up5, 32, 'deconv5_1')
      dc5_2 = nn.conv(dc5_1, 32, 'deconv5_2')

      up4 = nn.layer_add(nn.unpool(dc5_2, 'unpool4_tmp', shape=c4.get_shape().as_list()), c4, 'unpool4')
      dc4 = nn.conv(up4, 16, 'deconv4')

      up3 = nn.layer_add(nn.unpool(dc4, 'unpool3_tmp', shape=c3.get_shape().as_list()), c3, 'unpool3')
      dc3 = nn.conv(up3, 8, 'deconv3')

      up2 = nn.layer_add(nn.unpool(dc3, 'unpool2_tmp', shape=c2.get_shape().as_list()), c2, 'unpool2')
      dc2 = nn.conv(up2, 4, 'deconv2')

      up1 = nn.layer_add(nn.unpool(dc2, 'unpool1_tmp', shape=c1.get_shape().as_list()), c1, 'unpool1')
      dc1 = nn.conv(up1, 3, 'deconv1')


      self.out = dc1
      self.loss = nn.LOSS + tf.nn.l2_loss(self.out - self.y)
      self.opt = tf.train.AdamOptimizer(train_rate).minimize(self.loss)

# for tensorboard
      tf.summary.scalar('loss', self.loss)

  
  def save(self, filename):
    self.saver.save(self.sess, filename)

  def train(self, x, y):
    self.step += 1
    self.sess.run([self.opt], feed_dict={self.x: x, self.y: y})

  def eval(self, x):
    o = self.out.eval(feed_dict={self.x: x})
    return o

  def log(self, x, y):
    s = self.summary.eval(feed_dict={self.x: x, self.y: y})
    self.summary_writer.add_summary(s, self.step)