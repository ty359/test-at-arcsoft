import os
import sys
import cv2
import random


SHAPE = [256, 256, 3] # [32 * 3, 32 * 5], 32 = 2**5


def stdshape(img):
  global SHAPE
  shape = img.shape
  x = random.randint(2, SHAPE[0] - shape[0] - 2)
  y = random.randint(2, SHAPE[1] - shape[1] - 2)
  return cv2.copyMakeBorder(img, x, SHAPE[0] - shape[0] - x, y, SHAPE[1] - shape[1] - y, cv2.BORDER_REPLICATE)
