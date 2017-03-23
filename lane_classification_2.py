import numpy as np
import os

from keras.models import Model, load_model
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

def buildModel():
    main_input = main_input = Input(shape=input_shape, name='main_input')

    conv1_1 = Convolution2D(32, 3, 3, border_mode='same', name='conv1-1', activation='relu')
    conv1 = Convolution2D(32, 3, 3, border_mode='same', name='conv1-2', activation='relu')(conv1_1)
    pool1 = MaxPooling2D((2,2), strides=(2,2), name='pool1')(conv1)

    conv2_1 = Convolution2D(64, 3, 3, border_mode='same', name='conv2-1', activation='relu')(pool1)
    conv2 = Convolution2D(64, 3, 3, border_mode='same', name='conv2-2', activation='relu')(conv2_1)
    pool2 = MaxPooling2D((2,2), strides=(2,2), name='pool2')(conv1)

    conv7 = Convolution2D(256, 90, 160, activation='relu', name='conv7')(Dropout(0.25)(pool2))
    predict = Convolution2d(2,1,1)(Dropout(0.5))

    classification = Activation('softmax')(Flatten()(predict(conv7)))
    FCN_4s = Upsampling2D(4)(predict(conv7))
    FCN_2s = Upsampling2D(2)(predict(pool1))
    semantic_labels = Add([FCN_2s, FCN_4s])

    model = Model(inputs=[main_input], outputs=[classification]) #, FCN_4s, FCN_2s, semantic_labels])
    model.load_weights(load_model('wtf.h5').get_weights(), by_name=True)

    model.compile(optimizer='adadelta', loss='categorical_crossentropy')  #, loss_weights=[1, 0, 0, 0])

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


modelName = 'VGG_lane_classifier.h5'

vgg = VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
#for layer in vgg.layers:
 #   layer.trainable=False
flat = Flatten()(vgg.output)
fc1 = Dense(1024, activation='relu')(flat)
#fc2 = Dense(1024, activation='relu')(fc1)
pred = Dense(2, activation='softmax')(fc1)

model = Model(vgg.input, pred)
plot_model(model, 'vgg.png', show_shapes=True)
model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])

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

