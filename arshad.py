# Run these commands to install dependencies, assuming you're on Ubuntu:
# sudo apt-get install pip
# pip install git+git://github.com/fchollet/keras
# pip install tensorflow

import keras
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras import backend as K
import numpy as np

model = Sequential()

# This is the model you sent me. Comment it out if you uncomment the other one.
model.add(Dense(1, activation='sigmoid', input_shape=(3,)))

# This is the new model I sent you a picture of. (That picture is slightly different, this one would actually have three red circles.)
#model.add(Dense(3, activation='sigmoid', input_shape=(3,)))
#model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adadelta')

x_train = np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0],[0,0,1],[0,1,1],[1,1,1],[1,0,1]])
y_train = training_set_outputs = np.array([[0,1,1,0,1,0,1,0]]).T
x_test = np.array([[0,0,0], [1,0,0], [1,0,1]])

model.fit(x_train, y_train, epochs=10000)

print("Training Inputs:")
print(x_train)
print("Training Outputs:")
print(y_train)
print("Test Inputs:")
print(x_test)
print("Test Outputs:")
print(model.predict(x_test))
