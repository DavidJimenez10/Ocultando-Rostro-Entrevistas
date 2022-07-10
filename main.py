import cv2 as cv
import mediapipe as mp
#importando LocationData para verificar formato bounding_box
from mediapipe.framework.formats.location_data_pb2 import LocationData

import math
import time 
import os

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

    
    with mp_face_detection.FaceDetection(min_detection_confidence = 0.8,
                                        model_selection=0) as face_detection:
        #Detectando caras en el frame
        results = face_detection.process(cv.cvtColor(frame,cv.COLOR_BGR2RGB))
        #Dibujando detections
        if results.detections:
            
            annotated_image = frame.copy()

            for detection in results.detections:
                #Obteniendo localizacion del boundingbox
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

                    start_point=(pixel_x,pixel_y)
                    end_point=(pixel_x + pixel_width,pixel_y + pixel_height)
                    print(start_point)
                    print(end_point)
                    #Drawing manual
                    cv.rectangle(annotated_image,start_point,end_point,(0,0,255),2)

                #Drawing de mediapipe (incluye keypoints)
                #mp_drawing.draw_detection(annotated_image,detection)
                
    #Mostrando imagen anotada
    cv.imshow('Entrevistas', annotated_image)
    #Condicion para terminar ciclo
    if cv.waitKey(30) == ord('q'):
        break

#Liberado recursos
video_cap.release()
cv.destroyAllWindows()