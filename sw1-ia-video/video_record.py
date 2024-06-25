import cv2
from collections import deque

import torch
print(torch.__version__)
print("CUDA available: ", torch.cuda.is_available())

# Define la duración de grabación deseada en segundos
record_duration = 5

# Define la velocidad de cuadros por segundo
frame_rate = 10

# Calcula el número máximo de frames a mantener en el video
max_frames = int(record_duration * frame_rate)

# Abre un archivo de video para escribir el video grabado con la velocidad de cuadros deseada
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('video_grabado.avi', fourcc, frame_rate, (640, 480),isColor=True)  # Ajusta el formato y la resolución según tus necesidades

# Inicia la captura de la cámara
cap = cv2.VideoCapture(0)

# Inicializa el búfer de frames con una cola de longitud máxima max_frames
frame_buffer = deque(maxlen=max_frames)

while len(frame_buffer) < max_frames:
    ret, frame = cap.read()  # Captura el fotograma
    # Graba el fotograma en el archivo de video
    out.write(frame)
    frame_buffer.append(frame)
    cv2.imshow('Video Grabado', frame)
    # Leer por teclado
    t = cv2.waitKey(5)

while True:
    ret, frame = cap.read()  # Captura el fotograma

    # Graba el fotograma en el archivo de video
    out.write(frame)

    # Agrega el fotograma al búfer
    frame_buffer.append(frame)

    cv2.imshow('Video Grabado', frame)

    # Elimina los fotogramas más antiguos
    frame_buffer.popleft()

    # Abre un archivo de video para escribir el video grabado con la velocidad de cuadros deseada
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('video_grabado.avi', fourcc, frame_rate, (640, 480))  # Ajusta el formato y la resolución según tus necesidades
    for f in frame_buffer:
        out.write(f)

    # Leer por teclado
    t = cv2.waitKey(5)
    if t == 27:
        break

# Libera el recurso de la cámara y cierra el archivo de video
cap.release()
out.release()
cv2.destroyAllWindows()
