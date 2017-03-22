import os
import sys
import cv2
import random

import numpy as np
import tensorflow as tf

from predo import *
import model


data = data_init1() + data_init2() + data_init3()

M = model.model256()

M.build(BATCH_SIZE)

M.init(folder = 'training_models')


TP = PP = PN = .0001

for _ in range(0, 100000000):
  x, y = gendata(data)
  M.train(x, y)

  if _ % 50 == 0:
    print("epoch %d" % _)
    im1 = M.eval(x)[0]
    im2 = y[0]
    shape = im1.shape[:2]
    for i in range(0, shape[0]):
      for j in range(0, shape[1]):
        if int(im2[i][j][0]) + im2[i][j][1] + im2[i][j][2] >= 128*3:
          if int(im1[i][j][0]) + im1[i][j][1] + im1[i][j][2] >= 128*3:
            TP += 1
          PN += 1
        if int(im1[i][j][0]) + im1[i][j][1] + im1[i][j][2] >= 128*3:
          PP += 1
    print("Precision = %f, Recall = %f, IOU = %f" % (TP / PP, TP / PN, TP / (PP + PN - TP)))


  if _ % 1000 == 0:
    M.save('training_models/model%d.ckpt' % (_ / 1000))
    cv2.imwrite("out.jpg", M.eval(x)[0])
    cv2.imwrite("in.jpg", x[0])
    cv2.imwrite("_out.jpg", y[0])
    TP = PP = PN = .0001