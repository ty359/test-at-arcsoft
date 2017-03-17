import os
import sys
import cv2
import random

import numpy as np
import tensorflow as tf

from predo import *
import model

imgfolder = 'data/fff2/src/'

def check(img):
  shape = img.shape
  w0 = 4
  w1 = SHAPE[0] - shape[0] - w0
  h0 = 4
  h1 = SHAPE[1] - shape[1] - h0
  img = cv2.copyMakeBorder(img, w0, w1, h0, h1, cv2.BORDER_REPLICATE)
  return img, M.eval(np.stack([img]))[0]

M = model.model256()

M.build(1)

M.init()

for imgname in os.listdir(imgfolder):
  ps = imgname.split('.')
  if len(ps) == 2 and ps[1] == 'jpg':
    i, o = check(cv2.imread(imgfolder + imgname))
    cv2.imwrite(imgfolder + ps[0] + '_out.png', i + (256 - o))
