import cv2 
import numpy as np

img = cv2.imread('base.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#gray = cv2.cvtColor(imagen,cv2.COLOR_BGR2GRAY)

ret,thresh = cv2.threshold(gray,150,255,0)
carac = [] #LISTA DE VARIABLE
features = []
label = 0

contours,_= cv2.findContours(thresh, cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

cnt = contours[0]
#M = cv2.moments(cnt)

Hm = cv2.HuMoments(cv2.moments(cnt)).flatten()

carac = [ Hm[0], Hm[1], Hm[2], Hm[3], Hm[4], Hm[5], Hm[6] , label ] 
features.append(carac)
#[ ].join(Hm)
print(features)
##print("Ver la variable aqui abajo")
#print(carac.flatten() , "," , label)