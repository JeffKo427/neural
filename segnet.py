import numpy as np
import os

from keras.models import Sequential, load_model
from keras.layers import Activation, Permute
from keras.layers import Convolution2D, MaxPooling2D, Reshape
from keras.layers import ZeroPadding2D, BatchNormalization, UpSampling2D
from keras.utils import np_utils
from keras import backend as K
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

# Hyperparameters
n_labels = 12
num_enc_dec_blocks = 1
nb_epoch = 1
batch_size = 4
nb_filters = 32
kernel = (3,3)
pool_size = (2,2)
padding = (1,1)
data_gen_args = dict(
        #featurewise_center=True,
        #featurewise_std_normalization=True,
        #rotation_range = 20.,
        #width_shift_range=0.1,
        #height_shift_range=0.1,
        #zoom_range=0.2,
        horizontal_flip=True)
model_name = 'SegNet.h5'
img_width = 480
img_height = 360
input_shape = (img_height, img_width, 3)

# Add an encoder block of three conv + batch norm + ReLU blocks followed by a pooling layer.
def addEncoderBlock(model, num_conv_layers, num_neurons):
    model.add(ZeroPadding2D(padding=padding)),
    for i in range(num_conv_layers):
        model.add(Convolution2D(num_neurons, kernel[0], kernel[1], border_mode='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))


# Add a decoder block of an upsampling layer followed by three blocks of conv + batch norm + ReLU.
def addDecoderBlock(model, num_conv_layers, num_neurons):
    model.add(UpSampling2D(size=pool_size))
    model.add(ZeroPadding2D(padding=padding))
    for i in range(num_conv_layers):
        model.add(Convolution2D(num_neurons, kernel[0], kernel[1], border_mode='valid'))
    model.add(BatchNormalization())

# Build the model.
def buildModel():
    model = Sequential()

    model.add(Reshape((img_width, img_height, 3), input_shape=input_shape))

    addEncoderBlock(model, 1, 32)
    addEncoderBlock(model, 1, 64)
    addEncoderBlock(model, 1, 128)

    addDecoderBlock(model, 1, 128)
    addDecoderBlock(model, 1, 64)
    addDecoderBlock(model, 1, 32)

    model.add(Convolution2D(n_labels, 1,1, border_mode='valid'))
    model.add(Reshape((n_labels, img_height * img_width)))
    model.add(Permute((2,1)))
    model.add(Activation('softmax'))

    model.compile(
            loss='categorical_crossentropy',
            optimizer='adadelta',
            metrics=['accuracy'])

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
            path + 'images/',
            class_mode=None,
            target_size=(img_height, img_width),
            seed=seed,
            batch_size=batch_size)
    mask_generator = mask_datagen.flow_from_directory(
            path + 'masks/',
            color_mode='grayscale',
            class_mode=None,
            target_size=(img_height, img_width),
            seed=seed,
            batch_size=batch_size)

    # Combine generators into one which yields images and masks.
    generator = zip(image_generator, mask_generator)

    return generator

model = buildModel()
training_generator = buildDataGenerator('data/training/')
validation_generator = buildDataGenerator('data/validation/')

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

def reshapeY(y_raw):
    y_train = np.reshape(y_raw, (batch_size, img_height*img_width, 1))
    y_train = y_train.swapaxes(0,1)
    y_train = y_train.swapaxes(1,2)
    Y_list = []
    for y in y_train:
        Y_list.append(np_utils.to_categorical(y,n_labels))
    Y_train = np.array(Y_list)
    Y_train = Y_train.swapaxes(0,1)

    return Y_train

def unshapeY(y_pred):
    y = np.zeros((y_pred.shape[0]))
    for a in range(y_pred.shape[0]):
        y[a] = y_pred[a].argmax()

    Y = y.reshape((360,480,1))
    return Y



for e in range(nb_epoch):
    batches = 0
    print("Epoch: " + str(e+1))
    for X_train, y_raw in training_generator:
        Y_train = reshapeY(y_raw)

        print(model.train_on_batch(X_train, Y_train))

        batches += 1
        print("Batch " + str(batches) + " of 150.")
        if batches >= 150:
            break
    model.save('segnet.h5')

    for X_val, y_raw in validation_generator:
        Y_val = reshapeY(y_raw)
        print(model.test_on_batch(X_val, Y_val))
        if batches >= 30:
            break

model = load_model('segnet.h5')
import cv2
for X_vis, y_raw in validation_generator:
    Y_pred = model.predict(X_vis, batch_size=batch_size)
    print(X_vis.shape)
    for i in range(batch_size):
        cv2.imshow('Original', X_vis[i]/255)
        cv2.imshow('Ground Truth', y_raw[i]/10)
        cv2.imshow('Prediction', unshapeY(Y_pred[i])/10)
        cv2.waitKey(0)
