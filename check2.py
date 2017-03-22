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

DATA = []

def data_init1():
  namelist = open('data/all/dstleft/namelist.txt', 'r')
  src = 'data/all/dstleft/'
  dst = 'data/all/dstleft/tmp_'
  return [(src + name.strip() + '.jpg', dst + name.strip() + '.png') for name in namelist]

DATA += data_init1()

def check(img):
  shape = img.shape
  w0 = 4
  w1 = SHAPE[0] - shape[0] - w0
  h0 = 4
  h1 = SHAPE[1] - shape[1] - h0
  img = cv2.copyMakeBorder(img, w0, w1, h0, h1, cv2.BORDER_REPLICATE)
  return img, M.eval(np.stack([img]))[0]


def gendata(size=1):
  s = random.sample(DATA, size)
  x = []
  y = []
  for _ in range(0, size):
    imx = cv2.imread(s[_][0])
    imy = (cv2.imread(s[_][1]) > 0) * np.uint8(255)
    t = random.random() + 0.3
    shape = imx.shape
    while shape[0] * t >= 230 or shape[1] * t >= 230 or shape[0] * t <= 20 or shape[1] * t <= 20:      
      t = random.random() + 0.3
    imx = cv2.resize(imx, dsize=None, fx=t, fy=t, interpolation=cv2.INTER_CUBIC)
    imy = cv2.resize(imy, dsize=None, fx=t, fy=t, interpolation=cv2.INTER_CUBIC)
    shape = imx.shape
    w0 = random.randint(2, SHAPE[0] - shape[0] - 2)
    w1 = SHAPE[0] - shape[0] - w0
    h0 = random.randint(2, SHAPE[1] - shape[1] - 2)
    h1 = SHAPE[1] - shape[1] - h0
    imx = cv2.copyMakeBorder(imx, w0, w1, h0, h1, cv2.BORDER_REPLICATE)
    imy = cv2.copyMakeBorder(imy, w0, w1, h0, h1, cv2.BORDER_REPLICATE)
    x.append(imx)
    y.append(imy)
  return (np.stack(x), np.stack(y), s)

M = model.model256()

M.build(1)

M.init(folder = 'training_models')

TP = PP = PN = .0001

for _ in range(0, 100):
  TP = PP = PN = .0001

  x, im2, names = gendata()
  im2 = im2[0]
  im1 = M.eval(x)[0]
  shape = im1.shape[:2]
  for i in range(0, shape[0]):
    for j in range(0, shape[1]):
      if int(im2[i][j][0]) + im2[i][j][1] + im2[i][j][2] >= 128*3:
        if int(im1[i][j][0]) + im1[i][j][1] + im1[i][j][2] >= 128*3:
          TP += 1
        PN += 1
      if int(im1[i][j][0]) + im1[i][j][1] + im1[i][j][2] >= 128*3:
        PP += 1
  print("Precision = %f, Recall = %f, IOU = %f, file = %s" % (TP / PP, TP / PN, TP / (PP + PN - TP), names[0]))
