import os
import sys

from PIL import Image, ImageDraw

cnt = 0

names = set(os.listdir())
fout = open('namelist.txt', "w")
for name in names:
	xxname = name.split('.')
	if len(xxname) == 2:
		pref, suff = xxname
	if suff == 'jpg' and ('tmp_' + pref + '.png') in names:
		fout.write(pref + '\n')