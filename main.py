import numpy as np

import cv2 as cv
from cv2 import COLOR_RGB2BGR
import mediapipe as mp
#importando LocationData para verificar formato bounding_box
from mediapipe.framework.formats.location_data_pb2 import LocationData

import math
import time 
import os

from numpy import dtype

#Capturando video
video_cap = cv.VideoCapture('InputVideo/input.mp4')
#Inicializando detector y drawing de mediapipe
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1,circle_radius=1)

#Verificando streaming del video
if not video_cap.isOpened():
    print('No abre el video')
    exit()

#Obteniendo resolucion video
_, frame = video_cap.read()
HEIGHT, WIDTH, _ = frame.shape

while True:
    #Capturando frame
    ret, frame = video_cap.read()
    #Verificacion captura frame exitosa
    if not ret:
        print('No se recibio streaming')
        break

    frame.flags.writeable = False
    with mp_face_detection.FaceDetection(min_detection_confidence = 0.75,
                                        model_selection=0) as face_detection:
        #Detectando caras en el frame
        results = face_detection.process(cv.cvtColor(frame,cv.COLOR_BGR2RGB))
        #Dibujando detections
        if results.detections:
            
            frame.flags.writeable = True
            cv.cvtColor(frame,cv.COLOR_RGB2BGR)
            #annotated_image = frame.copy()

            for detection in results.detections:
                #Obteniendo localizacion del boundingbox (https://stackoverflow.com/questions/69810210/mediapipe-solutionfacedetection)
                location_data = detection.location_data
                if location_data.format == LocationData.RELATIVE_BOUNDING_BOX:
                    bb = location_data.relative_bounding_box
                    bb_box = [
                        bb.xmin, bb.ymin,
                        bb.width, bb.height,
                    ]
                    print(f"RBBox: {bb_box}")
                    #Llevando localizacion a pixels
                    pixel_x = math.floor(bb.xmin * WIDTH)
                    pixel_y = math.floor(bb.ymin * HEIGHT)
                    pixel_width = math.floor(bb.width * WIDTH)
                    pixel_height = math.floor(bb.height * HEIGHT)
                    #Obteninedo las posiciones de inicio y fin del rectangulo de la cara
                    start_point=(pixel_x,pixel_y)
                    end_point=(pixel_x + pixel_width,pixel_y + pixel_height)
                    print(start_point)
                    print(end_point)
                    #cv.rectangle(annotated_image,start_point,end_point,(0,0,255)

                    #Drawing manual
                    #cv.rectangle(frame,start_point,end_point,(0,0,255),2)
                    #Se genera mascara con todos los valores en 0
                    mask = np.zeros(frame.shape,dtype=np.uint8)
                    mask = cv.rectangle(mask,start_point,end_point,(255,255,255),-1) #Se el -1 rellenan los valores del rectangulo a 255

                    #Aplicando blur a la imagen original
                    frame_blur = cv.blur(frame, (15, 15))
                    #Se aplica la mascara donde los pixel en los 3 canales sean igual a 255
                    anonymize_frame = np.where(mask!=np.array([255,255,255]),frame,frame_blur)



                #Drawing de mediapipe (incluye keypoints)
                #mp_drawing.draw_detection(annotated_image,detection)
                
    #Mostrando imagen anotada
    #cv.imshow('Entrevistas', annotated_image)
    #cv.imshow('Entrevistas', frame_blur)
    cv.imshow('Entrevistas', anonymize_frame)
    #Condicion para terminar ciclo
    if cv.waitKey(30) == ord('q'): #Frame rate = 1000/n  donde cv.waitkey(n)
        break

#Liberado recursos
video_cap.release()
cv.destroyAllWindows()