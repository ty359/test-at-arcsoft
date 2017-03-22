import os
import sys
import cv2
import random

import numpy as np
import tensorflow as tf


BATCH_SIZE = 1
SHAPE = [256, 256, 3] # [32 * 3, 32 * 5], 32 = 2**5

def data_init0():
  src = 'data/src/'
  dst = 'data/dst/'
  _src = {x.split('.')[0]:x for x in os.listdir(src)}
  _dst = {x.split('.')[0]:x for x in os.listdir(dst)}

  return [(src + _src[x], dst + _dst[x]) for x in _src if _dst.get(x)]

def data_init1():
  namelist = open('data/fff/dst/namelist.txt', 'r')
  src = 'data/fff/dst/'
  dst = 'data/fff/dst/tmp_'
  return [(src + name.strip() + '.jpg', dst + name.strip() + '.png') for name in namelist]

def data_init2():
  namelist = open('data/fff2/dst/namelist.txt', 'r')
  src = 'data/fff2/dst/'
  dst = 'data/fff2/dst/tmp_'
  return [(src + name.strip() + '.jpg', dst + name.strip() + '.png') for name in namelist]


def data_init3():
  namelist = open('data/all/dstleft/namelist.txt', 'r')
  src = 'data/all/dstleft/'
  dst = 'data/all/dstleft/tmp_'
  return [(src + name.strip() + '.jpg', dst + name.strip() + '.png') for name in namelist]


def data_init4():
  namelist = open('data/test/namelist.txt', 'r')
  src = 'data/test/'
  dst = 'data/test/tmp_'
  return [(src + name.strip() + '.jpg', dst + name.strip() + '.png') for name in namelist]

def gendata(DATA, size=BATCH_SIZE):
  s = random.sample(DATA, size)
  x = []
  y = []
  for _ in range(0, size):
    imx = cv2.imread(s[_][0])
    imy = (cv2.imread(s[_][1]) > 0) * np.uint8(255)
    t = random.random() + 0.2
    shape = imx.shape
    while shape[0] * t >= 230 or shape[1] * t >= 230 or shape[0] * t <= 20 or shape[1] * t <= 20 or random.random() > 0.5:
      t = random.random() + 0.2
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
