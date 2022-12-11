#NUESTRO CODIGO ANDRÉS STEVEN Y JAIME

import cv2
import os
import numpy as np

#dataPath = 'C:/Users/Gaby/Desktop/Reconocimiento Facial/Data' #Cambia a la ruta donde hayas almacenado Data
dataPath ='D:/felpe/Documents/FINAL/Data'
peopleList = os.listdir(dataPath)
print('Lista de números: ', peopleList)

labels = []
facesData = []
carac = [] #LISTA DE VARIABLE
label = 0


for nameDir in peopleList:
	personPath = dataPath + '/' + nameDir
	print('Leyendo las imágenes')

	for fileName in os.listdir(personPath):
		print('Números: ', nameDir + '/' + fileName)
		labels.append(label)
		facesData.append(cv2.imread(personPath+'/'+fileName,0))
  
		image = cv2.imread(personPath+'/'+fileName) #acá se crea la imagen iterada
		#cv2.imshow('image',image)
		#cv2.waitKey(10)
  
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		ret,thresh = cv2.threshold(gray,150,255,0)
		contours,_= cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
		cnt = contours[0]
    
		#SE DA INICIO A LA CARACTERIZACIÓN
		M = cv2.moments(cnt) # 
		Hm = cv2.HuMoments(M).flatten() # Identifica los 7 Momentos de la imagen iterada
		carac =np.array( [ Hm[0], Hm[1], Hm[2], Hm[3], Hm[4], Hm[5], Hm[6] ] )
		
	label = label + 1
	


#print('labels= ',labels)
print('Caracteristicas= ',carac)
print(Hm)

print('Número de etiquetas 0: ',np.count_nonzero(np.array(labels)==0))
print('Número de etiquetas 1: ',np.count_nonzero(np.array(labels)==1))
print('Número de etiquetas 2: ',np.count_nonzero(np.array(labels)==2))
print('Número de etiquetas 3: ',np.count_nonzero(np.array(labels)==3))
print('Número de etiquetas 4: ',np.count_nonzero(np.array(labels)==4))
print('Número de etiquetas 5: ',np.count_nonzero(np.array(labels)==5))
print('Número de etiquetas 6: ',np.count_nonzero(np.array(labels)==6))
print('Número de etiquetas 7: ',np.count_nonzero(np.array(labels)==7))
print('Número de etiquetas 8: ',np.count_nonzero(np.array(labels)==8))
print('Número de etiquetas 9: ',np.count_nonzero(np.array(labels)==9))


mi_path = "../fichero2.txt"
f = open(mi_path, 'a+')

for i in labels:
    f.write( str(i) + ',')

f.close()