import cv2
import numpy

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface.xml')

cam = cv2.VideoCapture(1)

while True:
    # Capturar imagen de la camara.
    ret, img = cam.read()

    # Transformar a escala de grises.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detectar rostros dentro de las imagen (video) mediante el haarcascade.
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # Dibujar rectangulo en las caras detectadas.
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

    # Mostrar video 
    cv2.imshow('img', img)

    # Salir al precionar la tecla Esc.
    k = cv2.waitKey(30)
    if k == 27:
        break

# Al finalizar libera la camara y destruye las ventanas.
cam.release()
cv2.destroyAllWindows()