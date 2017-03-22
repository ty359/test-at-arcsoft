import os
import sys

cnt = 0

names = set(os.listdir())
for name in names:
	xxname = name.split('.')
	if len(xxname) == 2:
		pref, suff = xxname
		if suff == 'jpg':
			os.system("OutlineAnnotation.exe " + name)
			cnt += 1
			print(cnt)