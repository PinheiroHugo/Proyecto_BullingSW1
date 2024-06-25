#importamos librerias
import time
import torch
import numpy as np

########################################HTTP#################################
import http_data

########################################VIDEO CAPTURE#################################
import cv2
from collections import deque

# Define la duración de grabación deseada en segundos
record_duration = 15

# Define la velocidad de cuadros por segundo
frame_rate = 5

# Calcula el número máximo de frames a mantener en el video
max_frames = int(record_duration * frame_rate)

# Abre un archivo de video para escribir el video grabado con la velocidad de cuadros deseada
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('video_grabado.avi', fourcc, frame_rate, (640, 480),
                      isColor=True)  # Ajusta el formato y la resolución según tus necesidades

# Inicia la captura de la cámara
cap = cv2.VideoCapture(0)

# Inicializa el búfer de frames con una cola de longitud máxima max_frames
frame_buffer = deque(maxlen=max_frames)

########################################VIDEO CAPTURE#################################


#Leemos el modelo
model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path='D:/Hugo/Proyecto Final/Software bullyn/sw1-ia-video/acoso.pt')

#realizamos videocaptura
#detector = PoseDetector()
last_time = time.time() - 15
send_report = False

while len(frame_buffer) < max_frames:
    ret, frame = cap.read()  # Captura el fotograma
    # Graba el fotograma en el archivo de video
    out.write(frame)
    frame_buffer.append(frame)
    cv2.imshow('Detector', frame)
    # Leer por teclado
    t = cv2.waitKey(5)

#empezamos
while True:
    #realzar videocaptura
    ret, frame = cap.read()
    out.write(frame)

    # Realizamos detecciones
    detect = model(frame)
    #img = detector.findPose(frame)
    #lmlist, bboxInfo = detector.findPosition(img, bboxWithHands=False, draw=False)
    # Filtrar resultados para personas y el acoso
    info = detect.pandas().xyxy[0]

    # Imprimir resultados
    if (not info.empty) and time.time() - last_time > 15:
        print("¡Se ha detectado un posible acoso!, se enviara luego de 10 segundos")
        send_report = True
        last_time = time.time()

    if (send_report and time.time() - last_time > 10):
        send_report = False
        evento = http_data.crear_evento(descripcion='Video de posible acoso')

        if (evento != False):
            print(evento['data']['id'])
            id_evento = evento['data']['id']
            http_data.crear_evidencia_video(video_path='D:/Hugo/Proyecto Final/Software bullyn/sw1-ia-video/video_grabado.avi',
                                            id_evento=id_evento)
            print(f'Registro insertado')

    # Agrega el fotograma al búfer
    frame_buffer.append(frame)

    # Elimina los fotogramas más antiguos
    frame_buffer.popleft()

    # Abre un archivo de video para escribir el video grabado con la velocidad de cuadros deseada
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('video_grabado.avi', fourcc, frame_rate,
                          (640, 480))  # Ajusta el formato y la resolución según tus necesidades
    for f in frame_buffer:
        out.write(f)

    #Mostramos los FPS
    cv2.imshow('Detector', np.squeeze(detect.render()))

    #Leer por teclado
    t = cv2.waitKey(5)
    if t == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
