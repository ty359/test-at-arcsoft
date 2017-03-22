import os
import sys
import cv2
import random

import numpy as np
import tensorflow as tf

from predo import *
import model

'''
  will trans all jpg files from this folder
'''
imgfolder = 'data/test/'

def check_sub(img, w1, w2, h1, h2):
  shape = img.shape
  img = img[w1: -w2, h1: -h2]
  shape = img.shape
  _w1 = 4
  _w2 = SHAPE[0] - shape[0] - _w1
  _h1 = 4
  _h2 = SHAPE[1] - shape[1] - _h1
  img = cv2.copyMakeBorder(img, _w1, _w2, _h1, _h2, cv2.BORDER_REPLICATE)
  img = M.eval(np.stack([img]))[0]
  img = img[_w1: -_w2, _h1: -_h2]
  img = cv2.copyMakeBorder(img, w1, w2, h1, h2, cv2.BORDER_CONSTANT)
  return img

def check(img):
  shape = img.shape
  ret = np.zeros(shape=shape, dtype=np.float)
  for i in range(0, max(1, shape[0] - 200), 50):
    for j in range(0, max(1, shape[1] - 200), 50):
      w1 = i
      h1 = j
      w2 = max(0, shape[0] - w1 - 200)
      h2 = max(0, shape[1] - h1 - 200)
      imx = check_sub(img, w1, w2, h1, h2)
      ret = np.amax(np.stack([ret, imx]), axis=0)
  return img, ret

M = model.model256()

M.build(1)

M.init(folder = 'training_models')

for imgname in os.listdir(imgfolder):
  ps = imgname.split('.')
  if len(ps) == 2 and ps[1] == 'jpg':
    i, o = check(cv2.imread(imgfolder + imgname))
    cv2.imwrite(imgfolder + ps[0] + '_out.png', (i + (256 - o)) * 0.8 + i * 0.2)
