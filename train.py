import os
import sys
import cv2
from pathlib import PurePath
import random
from model import build

from nn import *
import numpy as np
import tensorflow as tf

src = 'data/src/'
dst = 'data/dst/'
out = 'out.jpg'

BATCH_SIZE = 10
SHAPE = [96, 160] # [32 * 3, 32 * 5], 32 = 2**5

_src = {x.split('.')[0]:x for x in os.listdir(src)}
_dst = {x.split('.')[0]:x for x in os.listdir(dst)}
DATA = [(_src[x], _dst[x]) for x in _src if _dst.get(x)]

def gendata(size=BATCH_SIZE):
  s = random.sample(DATA, size)
  return (np.stack([cv2.imread(src + x[0]) for x in s]), np.stack([cv2.imread(dst + x[1]) * 255 for x in s]))

x, y = gendata()

sess = tf.InteractiveSession()

_x = tf.placeholder(tf.float32, [BATCH_SIZE] + SHAPE + [3])
_y = tf.placeholder(tf.float32, [BATCH_SIZE] + SHAPE + [3])

nn = NN(sess)

o = build(nn, x=_x, y=_y)

train_step = tf.train.AdamOptimizer(1e-3).minimize(nn.loss)
accuracy = tf.reduce_mean(tf.cast((tf.abs(_y - o) < 128), tf.float32))

saver = tf.train.Saver(keep_checkpoint_every_n_hours=0.1)
sess.run(tf.global_variables_initializer())
saver.restore(sess, "training_models/model.ckpt")

for _ in range(0, 1000000):
  x, y = gendata()
  feed_dict = {_x: x, _y: y}
  train_step.run(feed_dict=feed_dict)
  if _ % 50 == 0:
    saver.save(sess, "training_models/model.ckpt")
    print("loss = %f, accuracy = %f" % (nn.loss.eval(feed_dict=feed_dict), accuracy.eval(feed_dict=feed_dict)))
    cv2.imwrite(out, o.eval(feed_dict=feed_dict)[0])