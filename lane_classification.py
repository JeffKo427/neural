import numpy as np
import os

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, Reshape, BatchNormalization, ZeroPadding2D
from keras.utils import np_utils, plot_model
from keras import backend as K
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

from keras.applications.vgg16 import VGG16

# HYPERPARAMETERS
batch_size = 128
nb_classes = 2
nb_epoch = 20
nb_filters = 32
pool_size = (2, 2)
kernel_size = (3, 3)
image_size = (180,320)
#image_size = (224,224)
input_shape = (image_size[0], image_size[1], 3)

def buildKerasSampleModel():
    model = Sequential()

    #model.add(Reshape((image_size[1], image_size[0], 3), input_shape=input_shape))

    model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=input_shape,
                name='conv1-1', activation='relu'))
    model.add(Convolution2D(32, 3, 3, border_mode='same',
                name='conv1-2', activation='relu'))
    model.add(MaxPooling2D(pool_size=pool_size))


    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    '''
    model.add(Dropout(0.25))
    model.add(Convolution2D(256, 90, 160, activation='relu', name='conv7'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(2, 1, 1))
    model.add(Flatten())
    model.add(Activation('softmax'))
    '''

    return model

def buildVGGlike():
    model = Sequential()

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Dropout(0.25))
    model.add(Convolution2D(256, 80, 45, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(2, 1, 1))
    model.add(Flatten())
    model.add(Activation('softmax'))

    return model



def buildSegNetEncoder():
    model = Sequential()

    model.add(Reshape((image_size[1], image_size[0], 3), input_shape=input_shape))

    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], border_mode='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))

    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], border_mode='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))

    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], border_mode='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))

    model.add(Flatten())
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    return model

datagen = ImageDataGenerator(
        #featurewise_center=True,
        #featurewise_std_normalization=True,
        #rotation_range=20,
        horizontal_flip=True,
        rescale=1./255)

training_generator = datagen.flow_from_directory(
        'data/training/',
        target_size=image_size,
        batch_size=batch_size)

validation_generator = datagen.flow_from_directory(
        'data/validation/',
        target_size=image_size,
        batch_size=batch_size)
modelName = 'wtf.h5'

if modelName in os.listdir():
    model = load_model(modelName)
else:
    model = buildKerasSampleModel()
    model.compile(loss='categorical_crossentropy',
            optimizer='adadelta',
            metrics=(['accuracy']))

plot_model(model, 'CNN.png', show_shapes=True)

checkpoint = ModelCheckpoint(
        modelName,
        monitor='val_acc',
        verbose=0,
        save_best_only=True,
        save_weights_only=False,
        mode='auto',
        period=1)
tb = TensorBoard(
        log_dir='./logs',
        histogram_freq=0,
        write_graph=True,
        write_images=True)

model.fit_generator(
        training_generator,
        samples_per_epoch=58501,
        nb_epoch=nb_epoch,
        callbacks=[checkpoint,tb],
        validation_data=validation_generator,
        nb_val_samples=14742)

