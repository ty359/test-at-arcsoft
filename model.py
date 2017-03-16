import os
import sys
import cv2
import random

import predo

from nn import *
import numpy as np
import tensorflow as tf

def build(nn, reuse=False):
  with tf.variable_scope('test1', reuse=reuse):
    c1 = nn.conv(x, 4, 'conv1')
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
    c6 = nn.conv(p5, 128, 'conv6')
    dc6 = nn.conv(c6, 64, 'deconv6')
    up5 = nn.layer_add(nn.unpool(dc6, 'unpool5_tmp', shape=c5_2.get_shape().as_list()), c5_2, 'unpool5')
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
    nn.gen_loss(dc1, y)
    return dc1