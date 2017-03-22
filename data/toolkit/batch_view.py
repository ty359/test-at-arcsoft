import os
import sys

cnt = 0

names = set(os.listdir())
for name in names:
	xxname = name.split('.')
	if len(xxname) == 2:
		pref, suff = xxname
	axx = pref + '.txt'
	if suff == 'jpg' and axx in names:
		os.system("VisualizeOutline.exe " + name)
		cnt += 1
	print(cnt)