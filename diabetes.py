#Importamos librerias a utilizar
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


#leemos csv
diabetes = pd.read_csv('diabetes.csv')

#columnas
print(diabetes.columns )
print('\n')
#Encabezado 
diabetes.head()
print('\n')
#información
print(diabetes.info())
print('\n')
#dimensión
print('dimensión diabetes: ' + str(diabetes.shape) + '\n')
#Agrupamos datos
print(diabetes.groupby('Outcome').size())

#representación datos
sn.countplot(diabetes['Outcome'], label= 'Count')


#Conexión entre la complejidad del modelo y la precisión de datos.
X_train, X_test, y_train, y_test = train_test_split(diabetes.loc[:, diabetes.columns != 'Outcome'], diabetes['Outcome'], stratify=diabetes['Outcome'], random_state=66)

training_accuracy = []
test_accuracy = []
# probamos n_neighbors de 1 a 10
neighbors_settings = range(1, 11)

for n_neighbors in neighbors_settings:
    # construimos el modelo
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    # récord de precisión del conjunto de entrenamiento
    training_accuracy.append(knn.score(X_train, y_train))
    # récord de precisión del conjunto de pruebas
    test_accuracy.append(knn.score(X_test, y_test))

plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
plt.savefig('knn_compare_model')

#El gráfico muestra el entrenamiento y la precisión del conjunto de pruebas en el eje y contra la configuración de n_vecinos en el eje x. Considerando que si elegimos un solo vecino más cercano, la predicción sobre el conjunto de entrenamiento es perfecta. Pero cuando se consideran más vecinos, la precisión del entrenamiento disminuye, lo que indica que el uso del vecino más cercano conduce a un modelo demasiado complejo.
#El mejor rendimiento está en algún lugar alrededor de 9 vecinos.
#El gráfico anterior sugiere que deberíamos elegir n_neighbors=9.

knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(X_train, y_train)

print('Accuracy of K-NN classifier on training set: {:.2f}'.format(knn.score(X_train, y_train)))
print('Accuracy of K-NN classifier on test set: {:.2f}'.format(knn.score(X_test, y_test)))

#Regresión logística
#La regresión logística es uno de los algoritmos de clasificación más comunes.

logreg = LogisticRegression().fit(X_train, y_train)
print("Training set accuracy: {:.3f}".format(logreg.score(X_train, y_train)))
print("Test set accuracy: {:.3f}".format(logreg.score(X_test, y_test)))

#El valor predeterminado de C=1 proporciona una precisión del 78% en el entrenamiento y del 77% en el conjunto de pruebas.
logreg001 = LogisticRegression(C=0.01).fit(X_train, y_train)
print("Training set accuracy: {:.3f}".format(logreg001.score(X_train, y_train)))
print("Test set accuracy: {:.3f}".format(logreg001.score(X_test, y_test)))

#El uso de C=0,01 da lugar a una menor precisión tanto en el entrenamiento como en los sets de pruebas.

logreg100 = LogisticRegression(C=100).fit(X_train, y_train)
print("Training set accuracy: {:.3f}".format(logreg100.score(X_train, y_train)))
print("Test set accuracy: {:.3f}".format(logreg100.score(X_test, y_test)))