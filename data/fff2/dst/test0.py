import os
import sys

from PIL import Image, ImageDraw

cnt = 0

names = set(os.listdir())
for name in names:
	xxname = name.split('.')
	if len(xxname) == 2:
		pref, suff = xxname
	axx = pref + '.txt'
	if suff == 'jpg' and axx in names:
		pfile = open(axx, "r")
		savefile = open("tmp_" + pref + ".png", "bw")
		img = Image.open(name)
		draw = ImageDraw.Draw(img)
		x = []
		for line in pfile:
			x += line.split(' ')
		y = [(float(x[i]), float(x[i + 1])) for i in range(0, len(x), 2)]
		help(draw.polygon)
		draw.polygon([(0, 0), (0, 1000), (1000, 1000), (1000, 0)], fill = (0, 0, 0))
		draw.polygon(y, fill = (255, 255, 255))
		img.save(savefile, "PNG")