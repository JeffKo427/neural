from __future__ import print_function
import numpy as np
np.random.seed(1337)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.callbacks import TensorBoard, ModelCheckpoint

import os
import pickle
import time
import cv2
import random

data_dir = 'contours/'
files = os.listdir(data_dir)
contours = []
for f in files:
    with open(data_dir + f,'rb') as fp:
        if f.endswith('_bad'):
            loaded = pickle.load(fp, encoding='latin1')
            for l in loaded:
                contours.append( (l, 0) )
        if f.endswith('_good'):
            loaded = pickle.load(fp, encoding='latin1')
            for l in loaded:
                contours.append( (l, 1) )

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

    datum = cv2.resize( img, (128,128) )
    return datum

#TODO: Double the size of our data by mirroring every contour. (Don't flip them, though. It's not rotation-invariant.)

# HYPERPARAMETERS
batch_size = 32
nb_classes = 2
nb_epoch = 12

# input image dimensions
img_rows, img_cols = 128,128
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

# the data, shuffled and split between train and test sets
#(X_train, y_train), (X_test, y_test) = mnist.load_data()
print( len(contours) )
all_data = []
for c in contours:
    makeShiftInvariant(c[0])
    img = imagize(c[0])
    all_data.append( (img, c[1]) )
    #cv2.imshow('Contour', img)
    #if 0xFF & cv2.waitKey(0) == 27:
    #    break

xtr = []
ytr = []
xte = []
yte = []

for datum in all_data:
    if random.random() < 0.2:
        xte.append( datum[0] )
        yte.append( datum[1] )
    else:
        xtr.append( datum[0] )
        ytr.append( datum[1] )

print(xtr)
print(ytr)

X_train = np.array(xtr)
y_train = np.array(ytr)
X_test = np.array(xte)
y_test = np.array(yte)

if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()

model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

checkpoint = ModelCheckpoint('contour_model.h5', monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
tb = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, callbacks=[tb, checkpoint], validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

def viewResults():
    results = model.predict(X_test, batch_size=32, verbose=0)
    pickle.dump( results, open('results/predictions','wb') )
    pickle.dump( xte, open('results/inputs', 'wb') )

viewResults()
