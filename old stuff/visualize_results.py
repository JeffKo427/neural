import cv2
import numpy as np
import sys
import string
import random
import pickle

inputs = pickle.load( open('results/inputs') )
predictions = pickle.load( open('results/predictions') )
true_class = pickle.load( open('results/true') )

def isRight(prob, true_class):
    if prob[0] < prob[1]:
        prediction = True
    else:
        prediction = False

    return prediction == true_class

for i in range(len(inputs)):
    if max(predictions[i]) < 0.7 or not isRight(predictions[i], true_class[i]):
        cv2.imshow('Contour', inputs[i])
        # predictions: [bad|good]
        # true_class: 1 is good, 0 is bad
        if predictions[i][0] < predictions[i][1]:
            p = 'is a lane line'
            c = str(predictions[i][1])
        else:
            p = 'not a lane line'
            c = str(predictions[i][0])
        if true_class[i]:
            t = 'is a lane line'
        else:
            t = 'not a lane line'
        print "\n"
        print "Predicted: " + p
        print "Actually: " + t
        print "Confidence: " + c

        if 0xFF & cv2.waitKey(0) == 27:
            break
