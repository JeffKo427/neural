from keras.applications.vgg16 import VGG16
from keras.layers import Convolution2D
from keras.models import Sequential

image_height = 244
image_width = 244
batch_size = 32

model = VGG16(weights='imagenet', include_top=False)
model.add(Convolution2D(4096,16,9,activation='relu'))
model.add(Convolution2D(4096,1,1,activation='relu'))
model.add(Dense(2, activation='softmax'))


datagen = ImageDataGenerator(
         #featurewise_center=True,
         #featurewise_std_normalization=True,
         #rotation_range=20,
         horizontal_flip=True,
         rescale=1./255)

training_generator = datagen.flow_from_directory(
        'data/training/',
        target_size=(image_height, image_width),
        batch_size=batch_size)

validation_generator = datagen.flow_from_directory(
        'data/validation/',
        target_size=(image_height, image_width),
        batch_size=batch_size)

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
