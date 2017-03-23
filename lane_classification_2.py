import numpy as np
import os

from keras.models import Model, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, Reshape, BatchNormalization, ZeroPadding2D
from keras.utils import np_utils#, plot_model
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

    conv1_1 = Conv2D(32, 3, 3, border_mode='same', name='conv1-1', activation='relu')(main_input)
    conv1 = Conv2D(32, 3, 3, border_mode='same', name='conv1-2', activation='relu')(conv1_1)
    pool1 = MaxPooling2D((2,2), strides=(2,2), name='pool1')(conv1)

    conv2_1 = Conv2D(64, 3, 3, border_mode='same', name='conv2-1', activation='relu')(pool1)
    conv2 = Conv2D(64, 3, 3, border_mode='same', name='conv2-2', activation='relu')(conv2_1)
    pool2 = MaxPooling2D((2,2), strides=(2,2), name='pool2')(conv2)

    flat = Dropout(0.25)(Flatten()(pool2))

    fc6 = Dense(256, activation='relu', name='fc7')(flat)
    fc7 = Dense(256, activation='relu')(fc6)

    classification = Dense(2, activation='softmax')(fc6)

    model = Model(main_input, classification) #, FCN_4s, FCN_2s, semantic_labels])

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


modelName = 'functional_lane_classifier.h5'

model = buildModel()
'''
vgg = VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
weights = vgg.get_weights()
for i in range(6):
    model.layers[i].set_weights(weights[i])
'''
#plot_model(model, 'flc.png', show_shapes=True)
model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])

for x,y in training_generator:
    print(x)
    break

checkpoint = ModelCheckpoint(
        modelName,
        monitor='val_acc',
        verbose=0,
        save_best_only=True,
        save_weights_only=False,
        mode='auto')
tb = TensorBoard(
        log_dir='./logs',
        histogram_freq=0,
        write_graph=True,
        write_images=True)

model.fit_generator(
        training_generator,
        samples_per_epoch=35759,
        nb_epoch=nb_epoch,
        callbacks=[checkpoint,tb],
        validation_data=validation_generator,
        nb_val_samples=9033)

