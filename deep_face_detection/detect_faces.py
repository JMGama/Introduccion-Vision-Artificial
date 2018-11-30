import numpy as np
import cv2
import time

# Cargar arquitectura y modelo entrenado de Deep Neural Network en el formato de Caffe Framework.
video = cv2.VideoCapture('video.mp4')
# video = cv2.VideoCapture(1)
prototxt = "deploy.prototxt"
model = "cnn.caffemodel"

# Crear nuestra rerd neuronal con el modelo previamente cargado.
print("[INFO] cargando modelo...")
net = cv2.dnn.readNetFromCaffe(prototxt, model)
time.sleep(2.0)

# Loop hasta que finaliza el video
while video.isOpened():
    # Cargamos el frame actual del video
    ret, frame = video.read()

    # Tomamos las dimensiones del frame
    (h, w) = frame.shape[:2]

    # Transformamos el frame en un blob
    blob = cv2.dnn.blobFromImage(cv2.resize(
        frame, (300, 300)), 1.0, (300, 300), (104.04, 177.0, 123.0))

    # Pasamos el blob a la red neuronal y obtenemos las detecciones y predicciones
    net.setInput(blob)
    detecciones = net.forward()

    # Recorremos las detecciones
    for i in range(0, detecciones.shape[2]):
        # Obtenemos la probabilidad de la deteccion
        probabilidad = detecciones[0, 0, i, 2]

        # descarta las detecciones que sean menores al 50% de probabilidad
        if probabilidad < 0.5:
            continue

        # Procesamos las cordenadas de lrango para la caja de los objetos detectados
        box = detecciones[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # Formato del texto sobre la probabilidad del objeto
        text = "{:.2f}%".format(probabilidad * 100)
        # Posicion del texto en caso de que la caja quede al tope del frame
        y = startY - 10 if startY - 10 > 10 else startY + 10
        # Dibujamos el cuadro alrededor de la cara junto con la probabilidad
        cv2.rectangle(frame, (startX, startY), (endX, endY),
                      (0, 0, 255), 2)
        cv2.putText(frame, text, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    # Mostrar el frame resultante
    cv2.imshow('video', frame)
    key = cv2.waitKey(30) & 0xFF

    # Si la tecla q es precionada, termina el programa
    if key == ord("q"):
        break

# Destruye las ventanas y cierra el video
cv2.destroyAllWindows()
video.stop()
