import cv2
import numpy as np
import sys
import skvideo.io

from keras.models import load_model
model = load_model('contour_model.h5')

cap = skvideo.io.vreader(sys.argv[1])
counter = 0

for full_frame in cap:
    print(full_frame)
    cv2.imshow('img',full_frame)
    #ret, full_frame = cap.read()
    '''
    height, width = full_frame.shape[:2]
    cropped_frame = full_frame[0:height - 200, 0:width]
    height, width = cropped_frame.shape[:2]
    reframe = cv2.resize(cropped_frame, (width/2, height/2))
    frame = cv2.GaussianBlur(reframe, ksize=(3,3), sigmaX=10)

    intensity = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)

    canny = cv2.Canny(saturation, 128, 255)
    sat_thresh = thresh( saturation, (30,80), (0,100) )

    athresh = cv2.adaptiveThreshold(saturation, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,37,10)
    contours, hierarchy = cv2.findContours(athresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    bigContours = []
    for c in contours:
        if cv2.contourArea(c) > 200:
            bigContours.append(c)
    inputContours = np.array(bigContours)
    predictions = model.predict(self, inputContours, batch_size=32, verbose=0)
    goodContours = []
    for i in range(len(predictions)):
        if predictions[i]:
            goodContours.append(inputContours[i])

    con = athresh.copy() * 0
    cv2.drawContours(con, goodContours, -1, 255, -1)
    cv2.imshow( 'Original', full_frame )
    cv2.imshow( 'Saturation', saturation)
    cv2.imshow( 'Contoured', con )

    #if cap.get(1) == cap.get(7): # Enums are broken, 1 is frame position, 7 is frame count
     #   cap.set(1, 0)
'''
    press = cv2.waitKey(5)
    if 0xFF & press == 27:
        break
    if 0xFF & press == 32:
        tmpimg = con.copy()
        getBadCountours(tmpimg, bigContours)
    press = None

cap.release()
cv2.destroyAllWindows()
