import pickle
import numpy as np
from sklearn import svm
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
#pip install scikit-learn

# Funcion de prediccion creada / Toma como parametros el vector de caracteristicas y el calsificador

def test(vector, classifier):
    result = classifier.predict(vector)  # Se almacena en una variable el resultado de la prediccion
    # Se evaluan las posibles etiquetas
    if result == 0:
        print("El numero es cero")
    elif result == 1:
        print("El numero es uno")
    elif result == 2:
        print("El numero es dos")  
    elif result == 3:
        print("El numero es tres")      
    elif result == 4:
        print("El numero es cuatro")
    elif result == 5:
        print("El numero es cinco")
    elif result == 6:
        print("El numero es seis")  
    elif result == 7:
        print("El numero es siete")      
    elif result == 8:
        print("El numero es ocho")
    elif result == 9:
        print("El numero es nueve")
        
        
        
        
# Dataset de Imágenes
dataset = np.loadtxt('fichero1.txt', delimiter= ',')
np.random.shuffle(dataset)
print(dataset)
data, labels = dataset[:, 0:9], dataset[:, 9]
# Divison del dataset entre entrenamiento y prueba
data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.20, random_state=42)
labels_train2 = labels_train.T

# Vector de caracteristicas para prueba de prediccion
caracteristicas = np.array([0.3232598689224207, 0.030812110986665156, 0.02604666693749557, 0.013210946174004897, 0.00023907405799071226, 0.0021463879672540286, -5.3844705253261465e-05, 35.127552, 35.127552])  # Vector de caraceristicas de prueba
caracteristicas = caracteristicas.reshape(1, -1)  # Se reorganizan los datos

#####################################################################
#************************CLACIFICADORES*****************************#
#####################################################################

print('*************************************************')
print('*****************CLASIFICADORES******************' + '\n')
print('*************************************************')

#####################
# Arbol de decision #
#####################
clfTree = tree.DecisionTreeClassifier(max_depth=2)  # Clasificador
clfTree = clfTree.fit(data_train, labels_train)  # Entrenamiento
testTree = clfTree.score(data_test, labels_test)  # Prueba
filename = 'clfTreeModel.sav'
pickle.dump(clfTree, open(filename, 'wb'))
print('Puntuacion Arbol de decision:', testTree)  # Puntaje
test(caracteristicas, clfTree)  # Se invoca funcion para prediccion
print('*************************************************')

#################################################################
#################################################################

###############
# Naive Bayes #
###############
clfNB = GaussianNB()  # Clasificador
clfTNB = clfNB.fit(data_train, labels_train)  # Entrenamiento
testNB = clfNB.score(data_test, labels_test)  # Prueba
print('Puntuacion Naive Bayes:', testNB)  # Puntaje
test(caracteristicas, clfNB)  # Se invoca funcion para predicción
print('*************************************************')

#################################################################
#################################################################

########################
# K Vecino mas cercano #
########################
clfNN = KNeighborsClassifier(n_neighbors=2, algorithm='ball_tree')  # Clasificador
clfTNN = clfNN.fit(data_train, labels_train)  # Entrenamiento
testNN = clfNN.score(data_test, labels_test)  # Prueba
print('Puntuacion K Vecino mas cercano:', testNN)  # Puntaje
test(caracteristicas, clfNN)  # Se invoca funcion para predicción
print('*************************************************')

#################################################################
#################################################################

################################
# Maquina de soporte vectorial #
################################
clfSVM = svm.SVC(kernel='rbf', gamma=0.7, C=1.0)  # Clasificador
clfSVM = clfSVM.fit(data_train, labels_train)  # Entrenamiento
testSVM = clfSVM.score(data_test, labels_test)  # Prueba
print('Puntuacion Maquina de soporte vectorial:', testSVM)  # Puntaje
test(caracteristicas, clfSVM)  # Se invoca funcion para predicción
print('*************************************************')

###################################################################
###################################################################

###########################
# Red neuronal artificial #
###########################
clfANN = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(50, 1000), random_state=1)  # Clasificador
clfANN = clfANN.fit(data_train, labels_train)  # Entrenamiento
testANN = clfANN.score(data_test, labels_test)  # Prueba
print('Puntuacion Red neuronal artificial:', testANN)  # Puntaje
test(caracteristicas, clfANN)  # Se invoca funcion para prediccion
print('*************************************************')



