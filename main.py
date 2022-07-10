import cv2 as cv
import mediapipe as mp
import time 
import os

video_cap = cv.VideoCapture('InputVideo/input.mp4')

if not video_cap.isOpened():
    print('No abre el video')
    exit()
while True:
    ret, frame = video_cap.read()

    if not ret:
        print('No se recibio streaming')
        break
    
    cv.imshow('Entrevistas', frame)
    if cv.waitKey(25) == ord('q'):
        break

video_cap.release()
cv.destroyAllWindows()