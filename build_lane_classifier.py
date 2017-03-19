from keras.applications.vgg16 import VGG16

model = VGG16(weights='imagenet', include_top=False, input_shape=(3,320,180), pooling=None)

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

model.fit_generator(
        training_generator,
        samples_per_epoch=58501,
        nb_epoch=nb_epoch,
        callbacks=[checkpoint,tb],
        validation_data=validation_generator,
        nb_val_samples=14742)
