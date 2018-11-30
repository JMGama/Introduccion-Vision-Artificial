import cv2
import numpy as np

# Cargar la imagen a ser analizada y ajustando tamaño.
img = cv2.imread('imagenes/monedas.jpg')
cv2.imshow('original', img)

# Convertimos la imagen a escala de grises.
img_gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('grises', img_gris)

# Quitamos el ruido de la imagen por medio de un filtro Gaussiano
img_gaussiana =  cv2.GaussianBlur(img_gris, (5,5), 0)
cv2.imshow('gaussiana', img_gaussiana)

# Realizamos la deteccion de bordes por medio del algoritmo Canny
img_canny = cv2.Canny(img_gaussiana, 50, 100)
cv2.imshow('canny', img_canny)

# Deteccion de contornos
(_,contornos,_) = cv2.findContours(img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
img_contornos = img.copy()
cv2.drawContours(img_contornos,contornos,-1,(0,255,0), 2)
cv2.imshow("contornos", img_contornos)

# Realizar los rectangulos para delimitar la region de interes (ROI)
objetos = 0
img_roi = img.copy()
for c in contornos:
    (x,y,w,h) = cv2.boundingRect(c)
    color = (0,255,0)
    cv2.rectangle(img_roi, (x,y),(x+w,y+h), color, 2)
    objetos += 1
cv2.imshow("ROI", img_roi)

print("Se encontraron {} objetos".format(objetos))

# Esperar presion de alguna tecla para finalizar
cv2.waitKey(0)
cv2.destroyAllWindows()