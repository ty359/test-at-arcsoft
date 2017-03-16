import os
import sys
import cv2
from pathlib import PurePath
import random

from nn import *
import numpy as np
import tensorflow as tf
from model import build

imgfolder = 'data/fff/dst/'

SHAPE = [96, 160]

def train(x, y):
  x = tf.image.resize_image_with_crop_or_pad(x, SHAPE[0], SHAPE[1])
  y = tf.image.resize_image_with_crop_or_pad(y, SHAPE[0], SHAPE[1])
  feed_dict = {_x: x, _y: y}

  print(x)

  print(y)

  train_step.run(feed_dict=feed_dict)

  saver.save(sess, "training_models/model.ckpt")

  return o.eval(feed_dict=feed_dict)[0]

sess = tf.InteractiveSession()

_x = tf.placeholder(tf.float32, [1] + SHAPE + [3])
_y = tf.placeholder(tf.float32, [1] + SHAPE + [3])

nn = NN(sess)

o = build(nn, _x, _y)

train_step = tf.train.AdamOptimizer(1e-3).minimize(nn.loss)

saver = tf.train.Saver()

saver.restore(sess, "training_models/model.ckpt")

for imgname in open(imgfolder + 'namelist.txt', "r"):
  imgname = imgname.strip()
  x = cv2.imread(imgfolder + imgname + '.jpg')
  y = cv2.imread(imgfolder + "tmp_" + imgname + '.png')
  print(imgfolder + imgname.strip() + '.jpg')
  cv2.imwrite('out.png', train(x, y)[0])
