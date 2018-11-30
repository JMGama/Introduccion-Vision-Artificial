import cv2
import numpy as np
from matplotlib import pyplot as plt

# IMPRIMI VERSION DE OPENCV
print(cv2.__version__)

# LECTURA DE IMAGENES

img_color = cv2.imread('imagenes/flash.png', 1) # Imagen con color y se elimina la transparencia.
img_grayscale = cv2.imread('imagenes/flash.png', 0) # Imagen en escala de grises.
img_unchanged = cv2.imread('imagenes/flash.png', -1) # Imagen tal como es sin eliminar la transparencia.


# TAMAÑO DE LA IMAGEN Y NUMERO DE CANALAES
print(img_color.shape)
print(img_grayscale.shape)
print(img_unchanged.shape)

# MOSTRAR IMAGENES
cv2.imshow('color_image',img_color)
cv2.imshow('grayscale_image',img_grayscale)
cv2.imshow('unchanged_image', img_unchanged)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Mostrar imagen con matplotlib
plt.imshow(cv2.cvtColor(img_unchanged, cv2.COLOR_BGRA2RGBA))
plt.show()

# ACCEDER A LOS VALORES DE LOS PIXELES

print(img_color.item(50, 600, 0), img_color.item(
    50, 600, 1), img_color.item(50, 600, 2)) # Imagen a color
print(img_grayscale.item(50, 600)) # Imagn en escala de grises
print(img_unchanged.item(50, 600, 0), img_unchanged.item(50, 600, 1),
      img_unchanged.item(50, 600, 2), img_unchanged.item(50, 600, 3)) # Imagne tal como es (con el canal alpha)
