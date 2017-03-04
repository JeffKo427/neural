import os
import pickle

data_dir = 'contours/'
files = os.listdir(data_dir)
print(files)
contours = []
for f in files:
    with open(data_dir + f,'rb') as fp:
        if f.endswith('_bad'):
            loaded = pickle.load(fp, encoding='latin1')
            for l in loaded:
                contours.append(l)
        if f.endswith('_good'):
            loaded = pickle.load(fp, encoding='latin1')
            for l in loaded:
                contours.append(l)
for c in contours:
    print(c)
