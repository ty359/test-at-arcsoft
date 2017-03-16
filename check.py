import os
import sys
import cv2
import random

from nn import *
import numpy as np
import tensorflow as tf
from model import build

imgfolder = 'data/src/'

def check(img):
	shape = img.shape
	with tf.Graph().as_default():
		with tf.Session() as sess:
			_x = tf.placeholder(tf.float32, list(shape))
			_y = tf.placeholder(tf.float32, list(shape))

			nn = NN(sess)

			o = build(nn, _x, _y)

			saver = tf.train.Saver()

			saver.restore(sess, "training_models_orz/model.ckpt")

			return o.eval(feed_dict={_x:img, _y:img})[0]

for imgname in os.listdir(imgfolder):
	ps = imgname.split('.')
	if len(ps) == 2 and ps[1] == 'jpg':
		o = check(np.stack([cv2.imread(imgfolder + imgname)]))
		cv2.imwrite(imgfolder + ps[0] + '_out.png', o)