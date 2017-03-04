import os

files = os.listdir('contours')
contours = []
for f in files:
	if f.endswith('bad'):
		reader = open('contours/' + f)
		ctr = []
		l = reader.readline()
		while l != ''