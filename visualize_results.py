import cv2
import numpy as np
import sys
import string
import random
import pickle

inputs = pickle.load( open('results/inputs') )
predictions = pickle.load( open('results/predictions') )
true_class = pickle.load( open('results/true') )

for i in range(len(inputs)):
    cv2.imshow('Contour', inputs[i])
    print "Prediction: " + predictions[i]
    print "True: " + true[i]
    if 0xFF & cv2.waitKey(0) == 27:
        break
