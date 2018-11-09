import numpy as np
import cv2
import dlib
import math
import sys
import pickle
import argparse
import os
import skvideo.io
import imutils
from imutils import face_utils

cap = cv2.VideoCapture(r'H:\MEGH\NITK\Third Year - B.Tech NITK\Fifth Semester\CMCK Mini Project\test.mp4')
i = 0;
# Read until video is completed
while(cap.isOpened()):
  
    ret, gray = cap.read()
    width_crop_max = 0;
    height_crop_max = 0;
    predictor_path = r"C:\Users\Megh Bhalerao\AppData\Local\Programs\Python\Python36\Lib\site-packages\dlib\shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    rects = detector(gray, 1)
    #d = enumerate(detections)
    (h,w,c) = gray.shape;
    detections = detector(gray, 1)

    # 20 mark for mouth
    marks = np.zeros((2, 20))

    # All unnormalized face features.
    Features_Abnormal = np.zeros((190, 1))

    # If the face is detected.
    print(len(detections))
    if len(detections) > 0:
        
        for k, d in enumerate(detections):

            # Shape of the face.
            shape = predictor(gray, d)

            co = 0
            # Specific for the mouth.
            for ii in range(48, 68):
                """
                This for loop is going over all mouth-related features.
                X and Y coordinates are extracted and stored separately.
                """
                X = shape.part(ii)
                A = (X.x, X.y)
                marks[0, co] = X.x
                marks[1, co] = X.y
                co += 1

            # Get the extreme points(top-left & bottom-right)
            X_left, Y_left, X_right, Y_right = [int(np.amin(marks, axis=1)[0]), int(np.amin(marks, axis=1)[1]),
                                                int(np.amax(marks, axis=1)[0]),
                                                int(np.amax(marks, axis=1)[1])]

            # Find the center of the mouth.
            X_center = (X_left + X_right) / 2.0
            Y_center = (Y_left + Y_right) / 2.0

            # Make a boarder for cropping.
            border = 30
            X_left_new = X_left - border
            Y_left_new = Y_left - border
            X_right_new = X_right + border
            Y_right_new = Y_right + border

            # Width and height for cropping(before and after considering the border).
            width_new = X_right_new - X_left_new
            height_new = Y_right_new - Y_left_new
            width_current = X_right - X_left
            height_current = Y_right - Y_left

            # Determine the cropping rectangle dimensions(the main purpose is to have a fixed area).
            if width_crop_max == 0 and height_crop_max == 0:
                width_crop_max = width_new
                height_crop_max = height_new
            else:
                width_crop_max += 1.5 * np.maximum(width_current - width_crop_max, 0)
                height_crop_max += 1.5 * np.maximum(height_current - height_crop_max, 0)

            # # # Uncomment if the lip area is desired to be rectangular # # # #
            #########################################################
            # Find the cropping points(top-left and bottom-right).
            X_left_crop = int(X_center - width_crop_max / 2.0)
            X_right_crop = int(X_center + width_crop_max / 2.0)
            Y_left_crop = int(Y_center - height_crop_max / 2.0)
            Y_right_crop = int(Y_center + height_crop_max / 2.0)
            if X_left_crop >= 0 and Y_left_crop >= 0 and X_right_crop < w and Y_right_crop < h:
                    mouth = gray[Y_left_crop:Y_right_crop, X_left_crop:X_right_crop, :]

        cv2.imshow("Mouth" + str(i),mouth);
        i = i+1;
cap.release()
 

