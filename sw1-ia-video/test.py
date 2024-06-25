import time
import cv2
import numpy as np
from cvzone.PoseModule import PoseDetector
from datetime import datetime
import torch
import requests

# Leemos el modelo
model = torch.hub.load('ultralytics/yolov5', 'custom', path='D:/Todo/iA/SW1-Laura-Proyecto/pythonProject/acoso.pt')
cap = cv2.VideoCapture(0)  # Cámara
detector = PoseDetector()
last_time = time.time() - 60
nimg = 0

while True:
    ret, frame = cap.read()  # Capturamos el fotograma

    # Realizamos detecciones
    detect = model(frame)
    img = detector.findPose(frame)
    lmlist, bboxInfo = detector.findPosition(img, bboxWithHands=False, draw=False)

    # Filtrar resultados para personas y acoso
    info = detect.pandas().xyxy[0]

    # Imprimir resultados
    if not info.empty and lmlist and time.time() - last_time > 60:
        print("¡Se ha detectado un posible acoso!")
        cv2.imwrite(f'nimg_{nimg}.png', frame)
        print(f'Registro insertado')

        

        nimg += 1
        last_time = time.time()

    # Mostrar los FPS
    cv2.imshow('Detector de Acoso', np.squeeze(detect.render()))

    # Leer por teclado
    t = cv2.waitKey(5)
    if t == 27:
        break

cap.release()
cv2.destroyAllWindows()
