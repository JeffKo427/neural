from __future__ import print_function
import numpy as np
np.random.seed(1337)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

import os
import pickle
import time
import cv2

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

# Normalize the data to be shift-invariant by moving the contour to the top-left.
def makeShiftInvariant(contour):
     lowest_x = 1e9
     lowest_y = 1e9
     for pixel in contour:
         px = pixel[0]
         if px[0] < lowest_x:
             lowest_x = px[0]
         if px[1] < lowest_y:
             lowest_y = px[1]
     for pixel in contour:
         px = pixel[0]
         px[0] = px[0] - lowest_x
         px[1] = px[1] - lowest_y

#TODO: Normalize the data to be scale-invariant by drawing the contour on a black image and resizing that image to 256x256 pixels.
def imagize(contour):
    highest_x = 0
    highest_y = 0
    for pixel in contour:
        px = pixel[0]
        if px[0] > highest_x:
            highest_x = px[0]
        if px[1] > highest_y:
            highest_y = px[1]
    dim = max(highest_x, highest_y)

    size = dim,dim,1
    img = np.zeros(size, dtype=np.uint8)
    cv2.drawContours(img, [contour], -1, 255, -1)

    datum = cv2.resize( img, (256,256) )
    return datum

#TODO: Double the size of our data by mirroring every contour. (Don't flip them, though. It's not rotation-invariant.)

# HYPERPARAMETERS
batch_size = 128
nb_classes = 2
nb_epoch = 12

#TODO: Shuffle the data and split them into training and test sets.
for c in contours:
    makeShiftInvariant(c)
    img = imagize(c)
    cv2.imshow('Contour', img)
    cv2.waitKey(0)
