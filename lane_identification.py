import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.callbacks import TensorBoard, ModelCheckpoint

# HYPERPARAMETERS
batch_size = 128
nb_classes = 2
nb_epoch = 5
nb_filters = 32
pool_size = (2, 2)
kernel_size = (3, 3)
image_shape = (320, 180, 1)
image_size = image_shape[:1]

def buildModel():
    model = Sequential()

    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                            border_mode='valid', input_shape=input_shape))
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

    return model

datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        horizontal_flip=True,
        rescale=1./255)

train_generator = datagen.flow_from_directory(
        'data/training/',
        target_size=image_size,
        batch_size=batch_size,
        class_mode='binary')

validation_generator = datagen.flow_from_directory(
        'data/validation/',
        target_size=image_size,
        batch_size=batch_size,
        class_mode='binary')

model = buildModel()
model.compile(loss='categorical_crossentropy',
        optimizer='adadelta',
        metrics=(['accuracy']))

checkpoint = ModelCheckpoint(
        'lane_identifier.h5',
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
        generator,
        samples_per_epoch=20000, #TODO
        nb_epoch=nb_epoch,
        callbacks=[checkpoint,tb],
        validation_data=validation_generator,
        nb_val_samples=8000) #TODO

