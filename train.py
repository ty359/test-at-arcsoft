import os
import sys
import cv2
from pathlib import PurePath
from random import sample as gendata

from model import *
import numpy as np
import tensorflow as tf

src = 'data/src/'
dst = 'data/dst/'

BATCH_SIZE = 10
SHAPE = [96, 160] # [32 * 3, 32 * 5], 32 = 2**5

_src = {x.split('.')[0]:x for x in os.listdir(src)}
_dst = {x.split('.')[0]:x for x in os.listdir(dst)}
DATA = [(_src[x], _dst[x]) for x in _src if _dst.get(x)]

_x = tf.placeholder(tf.float32, [BATCH_SIZE] + SHAPE + [3])
_y = tf.placeholder(tf.float32, [BATCH_SIZE] + SHAPE + [1])

def build(x=_x, y=_y):
  with tf.variable_scope('test1'):
    c1 = conv(x, 4, 'conv1')
    p1 = pool(c1, 'pool1')
    c2 = conv(p1, 8, 'conv2')
    p2 = pool(c2, 'pool2')
    c3 = conv(p2, 16, 'conv3')
    p3 = pool(c3, 'pool3')
    c4 = conv(p3, 32, 'conv4')
    p4 = pool(c4, 'pool4')
    c5_1 = conv(p4, 64, 'conv5_1')
    c5_2 = conv(c5_1, 64, 'conv5_2')
    p5 = pool(c5_2, 'pool5')
    c6 = conv(p5, 128, 'conv6')
    dc6 = conv(c6, 64, 'deconv6')
    up5 = layer_add(unpool(dc6, 'unpool5_tmp'), c5_2, 'unpool5')
    dc5_1 = conv(up5, 32, 'deconv5_1')
    dc5_2 = conv(dc5_1, 32, 'deconv5_2')
    up4 = layer_add(unpool(dc5_2, 'unpool4_tmp'), c4, 'unpool4')
    dc4 = conv(up4, 16, 'deconv4')
    up3 = layer_add(unpool(dc4, 'unpool3_tmp'), c3, 'unpool3')
    dc3 = conv(up3, 8, 'deconv3')
    up2 = layer_add(unpool(dc3, 'unpool2_tmp'), c2, 'unpool2')
    dc2 = conv(up2, 4, 'deconv2')
    up1 = layer_add(unpool(dc2, 'unpool1_tmp'), c1, 'unpool1')
    dc1 = conv(up1, 1, 'deconv1')
    gen_loss(dc1, y)
    return dc1

build()