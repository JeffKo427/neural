#import numpy as np
import os
import pickle
import cv2
import random
import string

def getID(size=6, chars=string.ascii_lowercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

# Get a list of videos we will be using as data.
data_dir = 'data/videos/'
files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(data_dir)) for f in fn]

# Sort these videos by whether or not they contain a lane line.
vids0 = []
vids1 = []
for f in files:
    if f.endswith('0'):
        vids0.append(f)
    elif f.endswith('1'):
        vids1.append(f)
    else:
        print("File " + f + " not properly formatted, should end in either 0 or 1. Ignoring.")

# Write each frame of these videos to file.
def writeFrames(video, train_directory, test_directory, size=(320,180)):
    cap = cv2.VideoCapture(video)
    name = video.rsplit('/',1)[-1]
    while cap.get(1) != cap.get(7): # frame position != frame count
        ret, full_frame = cap.read()
        if not ret:
            print("Failed to get frame, aborting split.")
            break
        #frame = cv2.resize(full_frame, size)
        if random.random() < 0.2:
            full_name = test_directory + name + '-' + str(cap.get(1)) + '.png'
        else:
            full_name = train_directory + name + '-' + str(cap.get(1)) + '.png'
        cv2.imwrite(full_name, full_frame)


train_true = 'data/training/1/'
train_false = 'data/training/0/'
val_true = 'data/validation/1/'
val_false = 'data/validation/0/'
#TODO: assert that these dirs are empty
for v in vids0:
    print("Splitting " + v + "...")
    writeFrames(v, train_false, val_false)
for v in vids1:
    print("Splitting " + v + "...")
    writeFrames(v, train_true, val_true)

