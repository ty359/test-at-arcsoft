import os
import sys
import cv2
import random

SHAPE = [256, 256, 3]

def reshape(img):
  shape = img.shape
  return cv2.copyMakeBorder(img, 4, SHAPE[0] - shape[0] - 4, 4, SHAPE[1] - shape[1] - 4, cv2.BORDER_REPLICATE)