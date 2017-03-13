import numpy as np
import os
import pickle
import cv2
import random

# Make a list of all the files we'll be working with.
files1 = os.listdir('data/images_true')
files0 = os.listdir('data/images_false')
files_all = []

for f in files1:
    files_all.append(f)
for f in files0:
    files_all.append(f)

# Randomize it.
files_all.shuffle()

# Randomly sort it into training and testing data.
test = []
train = []
for f in files_all:

