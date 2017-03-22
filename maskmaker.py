import cv2
import sys, os
import random

path = sys.argv[1]

raw_imgs = os.listdir(sys.argv[1])
random.shuffle(raw_imgs)

block_size = 37
C = 10
for i in raw_imgs:
    img = cv2.imread(path + i)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)

    while True:
        athresh = cv2.adaptiveThreshold(
                src=saturation,
                maxValue=255,
                adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                thresholdType=cv2.THRESH_BINARY_INV,
                blockSize=block_size,
                C=C)
        contours, hierarchy = cv2.findContours(athresh, cv2.RETR_EXTERNAL,     cv2.CHAIN_APPROX_NONE)
        bigContours = []
        for c in contours:
             if cv2.contourArea(c) > 200:
                 bigContours.append(c)

        con = athresh.copy() * 0
        cv2.drawContours(con, bigContours, -1, 255, -1)

        cv2.imshow('Original', img)
        cv2.imshow('Mask', con)

        press = cv2.waitKey(0)
        print press
        if 0xFF & press == 65362: #up arrow
            C -= 1
        elif 0xFF & press == 65364: #down arrow
            C += 1
        elif 0xFF & press == 119: #'w'
            #write
            break
        elif 0xFF & press == 27: #Esc
            sys.exit()
        else:
            break
