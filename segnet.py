import numpy as np
import os

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, Reshape, BatchNormalization #TODO: upsampling etc
from keras.utils import np_utils
from keras import backend as K
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

# Hyperparameters
num_enc_dec_blocks = 1
nb_epoch = 20
nb_filters = 32
kernel_size = (3,3)
pool_size = (2,2)
padding = (1,1)
data_gen_args = dict(
        #featurewise_center=True,
        #featurewise_std_normalization=True,
        rotation_range = 20.,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True)
model_name = 'SegNet.h5'
img_width = 640
img_height = 360
input_shape = (img_height, img_width, 3)

# Add an encoder block of three conv + batch norm + ReLU blocks followed by a pooling layer.
def addEncoderBlock(model):
    model.add(ZeroPadding2D(padding=padding)),
    model.add(Convolution2D(nb_filters, kernel[0], kernel[1], border_mode='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))


# Add a decoder block of an upsampling layer followed by three blocks of conv + batch norm + ReLU.
def addDecoderBlock(model, upsample=True):
    if upsample:
        model.add(UpSampling2D(size=pool_size))
    model.add(ZeroPadding2D(padding=padding))
    model.add(Convolution2D(nb_filters, kernel[0], kernel[1], border_mode='valid'))
    model.add(BatchNormalization())

# TODO: Build the model.
def buildModel():
    model = Sequential()

    model.add(Reshape(img_width, img_height, 3), input_shape=input_shape)


    #model.add(Activation('softmax'))

    #model.compile()

    return model

def buildDataGenerator(path):
    # Create two instances with the same arguments.
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    # Pass the same seed to both, on both the fit and flow methods.
    seed = 27
    #TODO: load some sample data with rank 4 for passing to fit
    #image_datagen.fit(images, augment=True, seed=seed)
    #mask_datagen.fit(masks, augment=True, seed=seed)

    image_generator = image_datagen.flow_from_directory(
            path + '/images',
            class_mode=None,
            seed=seed)
    mask_generator = mask_datagen.flow_from_directory(
            path + 'masks',
            class_mode=None,
            seed=seed)

    # Combine generators into one which yields images and masks.
    generator = zip(image_generator, mask_generator)

    return generator

model = buildModel()
training_generator = buildDataGenerator('data/training')
validation_generator = buildDataGenerator('data/validation')

checkpoint = ModelCheckpoint(
        model_name,
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
        samples_per_epoch=   ,
        nb_epoch=nb_epoch,
        callbacks=[checkpoint,tb]
        validation_data=validation_generator,
        nb_val_samples=   )
