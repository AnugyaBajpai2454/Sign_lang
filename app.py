import streamlit as st
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import tempfile
import time

# Load classifier
classifier = Classifier("converted_keras/keras_model.h5", "converted_keras/labels.txt")
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300
labels = ["Hello", "I love you", "No", "Please", "Sorry", "Thank you", "yes"]

st.title("Real-time Hand Gesture Classifier")

run = st.checkbox('Start Camera')

if run:
    cap = cv2.VideoCapture(0)
    FRAME_WINDOW = st.image([])

    while True:
        success, img = cap.read()
        imgOutput = img.copy()
        hands, img = detector.findHands(img)
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255
            imgCrop = img[y-offset:y + h + offset, x-offset:x + w + offset]

            aspectRatio = h / w
            if aspectRatio > 1:
                k = imgSize / h
                wCal = round(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = (imgSize - wCal) // 2
                imgWhite[:, wGap:wGap + wCal] = imgResize
            else:
                k = imgSize / w
                hCal = round(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = (imgSize - hCal) // 2
                imgWhite[hGap:hGap + hCal, :] = imgResize

            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            cv2.putText(imgOutput, labels[index], (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

        FRAME_WINDOW.image(imgOutput, channels="BGR")
        if not run:
            break

    cap.release()
