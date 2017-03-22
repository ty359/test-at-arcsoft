import os
import sys

import numpy as np
import cv2

cnt = 0

names = set(os.listdir())
for name in names:
	xxname = name.split('.')
	if len(xxname) == 2:
		pref, suff = xxname
	axx = pref + '.txt'
	if suff == 'jpg' and axx in names:
		pfile = open(axx, "r")
		SHAPE = cv2.imread(name).shape
		if 'tmp_' + pref + '.png' in names:
			img = cv2.imread('tmp_' + pref + '.png')
		else:
			img = np.zeros(shape=SHAPE, dtype=np.uint8)
		pts = []
		for lines in pfile:
			pts += lines.split()
		pts = [(int(pts[i]), int(pts[i + 1])) for i in range(0, len(pts), 2)]
		if len(pts) > 2:
			cv2.fillConvexPoly(img, np.array(pts), (255, 255, 255))
		cv2.imwrite('tmp_' + pref + '.png', img)

