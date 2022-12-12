import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm

dataset = np.loadtxt('fichero1.txt', delimiter=',')
np.random.shuffle(dataset)
data, labels = dataset[:, 0:9], dataset[:, 9]

# Split dataset
data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.20, random_state=42)

# Train classifier
clf = svm.SVC(kernel='rbf')
clf.fit(data_train, labels_train)
print("Porcentaje de efectividad: ", clf.score(data_test, labels_test))
features = []

imagen = cv2.imread('9.jpg')
gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY) #Escala de grises
edged = cv2.Canny(gray, 30, 200)#imagen binarizada2 si funciona
contornos, herencia = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #RETR_LIST(todos) - 
threshold1, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
humm = cv2.HuMoments(cv2.moments(thresh1)).flatten()
prom = np.mean(thresh1, dtype=np.float32)
s1 = thresh1[0:68, 0:34]
s2 = thresh1[0:68, 34:68]
proms1 = np.mean(s1, dtype=np.float32)
proms2 = np.mean(s2, dtype=np.float32)
features = np.array([humm[0], humm[1], humm[2], humm[3], humm[4], humm[5], humm[6], prom, proms1])
features = np.array([features])
features = features.reshape(1, -1)
signal = clf.predict(features)
if signal == 0:
    print("El número corresponde a cero")
elif signal == 1:
    print("El número corresponde a uno")
elif signal == 2:
    print("El número corresponde a dos")
elif signal == 3:
    print("El número corresponde a tres")
elif signal == 4:
    print("El número corresponde a cuatro")
elif signal == 5:
    print("El número corresponde a cinco")
elif signal == 6:
    print("El número corresponde a seis")
elif signal == 7:
    print("El número corresponde a siete")
elif signal == 8:
    print("El número corresponde a ocho")
elif signal == 9:
    print("El número corresponde a nueve")

