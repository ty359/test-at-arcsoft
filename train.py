import os
import sys
import cv2
import random

import numpy as np
import tensorflow as tf

from predo import *
import model

BATCH_SIZE = 1

DATA = []

def data_init0():
  src = 'data/src/'
  dst = 'data/dst/'
  _src = {x.split('.')[0]:x for x in os.listdir(src)}
  _dst = {x.split('.')[0]:x for x in os.listdir(dst)}

  return [(src + _src[x], dst + _dst[x]) for x in _src if _dst.get(x)]

DATA += data_init0()


def data_init1():
  namelist = open('data/fff/dst/namelist.txt', 'r')
  src = 'data/fff/dst/'
  dst = 'data/fff/dst/tmp_'
  return [(src + name.strip() + '.jpg', dst + name.strip() + '.png') for name in namelist]

DATA += data_init1()



def gendata(size=BATCH_SIZE):
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
  return (np.stack(x), np.stack(y))



M = model.model256()

M.build(BATCH_SIZE)

M.init()


TP = .1
PP = .1
PN = .1

for _ in range(0, 100000000):
  x, y = gendata()
  M.train(x, y)
  if _ % 50 == 0:
    print("epoch %d" % _)
    im1 = M.eval(x)[0]
    im2 = y[0]
    shape = im1.shape[:2]
    for i in range(0, shape[0]):
      for j in range(0, shape[1]):
        if im2[i][j][0] + im2[i][j][1] + im2[i][j][2] >= 128:
          if im1[i][j][0] + im1[i][j][1] + im1[i][j][2] >= 128:
            TP += 1
          PN += 1
        if im1[i][j][0] + im1[i][j][1] + im1[i][j][2] >= 128:
          PP += 1
    print("Precision = %f, Recall = %f" % (TP / PP, TP / PN))


  if _ % 1000 == 0:
    cv2.imwrite("out.jpg", M.eval(x)[0])
    cv2.imwrite("in.jpg", x[0])
    cv2.imwrite("_out.jpg", y[0])
    M.save('training_models/model%d.ckpt' % (_ / 1000))
    print("Precision = %f, Recall = %f" % (TP / PP, TP / PN))
    TP = PP = PN = .0