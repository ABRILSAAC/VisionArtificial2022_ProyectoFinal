import cv2
import numpy as np

imagen = cv2.imread('1.jpeg')
gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY) #Escala de grises
edged = cv2.Canny(gray, 30, 200)#imagen binarizada2 si funciona

contornos, herencia = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #RETR_LIST(todos) - 

#------------cv2.drawContours(imagen, contornos, 2, (0,0,255),3) #Dibujarlos en pantalla (imagen, contorno, -1 todos, color, grosor)
cv2.drawContours(imagen, contornos, 1, (0, 255, 0), 2)

print('len(contornos)= ' , len(contornos[1]))


cv2.imshow('Binarizada', edged)
cv2.waitKey(0)
cv2.imshow('ima ffff', imagen)
cv2.waitKey(0)



cv2.destroyAllWindows()