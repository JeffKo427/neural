import cv2
import sys, os
import random

path = sys.argv[1]

raw_imgs = os.listdir(sys.argv[1])
random.shuffle(raw_imgs)

block_size = 37
C = 10
for img in raw_imgs:
    cv2.imread(path + img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)

    while True:
        mask = cv2.adaptiveThreshold(
                src=saturation,
                maxValue=255,
                adaptive_method=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                threshold_type=cv2.THRESH_BINARY_INV,
                block_size=block_size,
                C=C)
        cv2.imshow('Original', img)
        cv2.imshow('mask', mask)

        press = cv2.waitKey(0)
        if press == 65362: #up arrow
            #threshold up
        elif press == 65364: #down arrow
            #threshold down
        elif press == 119: #'w'
            #write
        else:
            #skip
