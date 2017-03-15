import numpy as np
import os

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, Reshape
from keras.utils import np_utils
from keras import backend as K
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

# HYPERPARAMETERS
batch_size = 128
nb_classes = 2
nb_epoch = 5
nb_filters = 32
pool_size = (2, 2)
kernel_size = (3, 3)
image_size = (180,320)
input_shape = (image_size[0], image_size[1], 3)

def buildPartialModel(full_model):
    model = Sequential()

    model.add(Reshape((image_size[1], image_size[0], 3), input_shape=input_shape))

    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                            border_mode='valid',
                            weights=full_model.layers[1].get_weights()))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                            weights=full_model.layers[3].get_weights()))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))

    return model

datagen = ImageDataGenerator(
        #featurewise_center=True,
        #featurewise_std_normalization=True,
        #rotation_range=20,
        #horizontal_flip=True,
        rescale=1./255)

training_generator = datagen.flow_from_directory(
        'data/training/',
        target_size=image_size,
        batch_size=batch_size)

validation_generator = datagen.flow_from_directory(
        'data/validation/',
        target_size=image_size,
        batch_size=batch_size)

modelName = 'lane_identifier.h5'
if modelName in os.listdir():
    model = load_model(modelName)

partial_model = buildPartialModel(model)
for X_batch, Y_batch in validation_generator:
    break
features = partial_model.predict(X_batch)

def overlayFeaturemap(fmaps):
    m = np.zeros(fmaps[0].shape)
    for f in fmaps:
        m = m + f

    return m


import cv2
'''
for f in features:
    print(f.shape)
    g = f.swapaxes(0,2)
    print(g.shape)
    cv2.imshow('help',g[0])
    cv2.waitKey(0)
    '''
for i in range(128):
    cv2.imshow('I got this.', X_batch[i])
    print(X_batch[i].shape)
    print(X_batch[i])
    cv2.imshow("You're doing great, keep it up!", overlayFeaturemap(features[i].swapaxes(0,2)))
    cv2.waitKey(0)
