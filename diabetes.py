#Importamos librerias a utilizar
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler

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


### Conexión entre la complejidad del modelo y la precisión de datos.
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

### Regresión logística
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


#El uso de C=100 da como resultado una precisión un poco mayor en el conjunto de entrenamiento y un poco menor precisión en el conjunto de pruebas, confirmando que menos regularización y un modelo más complejo puede no generalizar mejor que la configuración predeterminada. 

#Por lo tanto, debemos elegir el valor predeterminado C=1.
#Finalmente, echemos un vistazo a los coeficientes aprendidos por los modelos con las tres configuraciones diferentes del parámetro de regularización C.

#Una mayor regularización (C=0,001) empuja los coeficientes cada vez más hacia cero. Al inspeccionar la trama más de cerca, también podemos ver que la característica «DiabetesPedigreeFunction», para C=100, C=1 y C=0,001, el coeficiente es positivo. Esto indica que la función «DiabetesPedigreeFunction» está relacionada con una muestra que es «diabetes», independientemente del modelo que veamos.


diabetes_features = [x for i,x in enumerate(diabetes.columns) if i!=8]

plt.figure(figsize=(8,6))
plt.plot(logreg.coef_.T, 'o', label="C=1")
plt.plot(logreg100.coef_.T, '^', label="C=100")
plt.plot(logreg001.coef_.T, 'v', label="C=0.001")
plt.xticks(range(diabetes.shape[1]), diabetes_features, rotation=90)
plt.hlines(0, 0, diabetes.shape[1])
plt.ylim(-5, 5)
plt.xlabel("Feature")
plt.ylabel("Coefficient magnitude")
plt.legend()
plt.savefig('log_coef')

### Arbol de decision

tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))

tree = DecisionTreeClassifier(max_depth=3, random_state=0)
tree.fit(X_train, y_train)

print("Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))

#La precisión del set de entrenamiento es del 100%, mientras que la precisión del set de prueba es mucho peor. Esto indica que el árbol se está sobreadaptando y no está generalizando bien a los nuevos datos. Por lo tanto, tenemos que aplicar la pre poda al árbol.

#Establecemos max_depth=3, limitando la profundidad del árbol disminuye el exceso de equipamiento. Esto conduce a una menor precisión en el equipo de entrenamiento, pero una mejora en el equipo de pruebas.

#Importancia de la característica en los árboles de Decisión
#La importancia de la característica determina cuán importante es cada característica para la decisión que toma un árbol. Es un número entre 0 y 1 para cada función, donde 0 significa «no se utiliza en absoluto» y 1 significa «predecir perfectamente el objetivo.» La importancia de las características siempre suman 1:
print("Feature importances:\n{}".format(tree.feature_importances_))

def plot_feature_importances_diabetes(model):
    plt.figure(figsize=(8,6))
    n_features = 8
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), diabetes_features)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)

plot_feature_importances_diabetes(tree)
plt.savefig('feature_importance')
# La «Glucosa» es, con mucha diferencia, la característica más importante.


### Bosque random
rf = RandomForestClassifier(n_estimators=100, random_state=0)
rf.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(rf.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(rf.score(X_test, y_test)))

rf1 = RandomForestClassifier(max_depth=3, n_estimators=100, random_state=0)
rf1.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(rf1.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(rf1.score(X_test, y_test)))

#El bosque aleatorio nos da una precisión del 78,6%, mejor que el modelo de regresión logística o un solo árbol de decisión, sin ajustar ningún parámetro. Sin embargo, podemos ajustar la configuración max_features, para ver si se puede mejorar el resultado.

### Lo importante en el Bosque random

#Al igual que el árbol de decisión único, el bosque aleatorio también da mucha importancia a la función «Glucosa», pero también elige «IMC» para ser la segunda característica más informativa en general. La aleatoriedad en la construcción del bosque aleatorio obliga al algoritmo a considerar muchas explicaciones posibles, el resultado es que el bosque aleatorio captura una imagen mucho más amplia de los datos que un solo árbol.


plot_feature_importances_diabetes(rf)

### Aumento de gradiente

gb = GradientBoostingClassifier(random_state=0)
gb.fit(X_train, y_train)

print("Accuracy on training set: {:.3f}".format(gb.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(gb.score(X_test, y_test)))



#Es probable que estemos sobreequipando. Para reducir el sobreequipamiento, podríamos aplicar una pre-pudadura más fuerte limitando la profundidad máxima o reducir la tasa de aprendizaje:

gb1 = GradientBoostingClassifier(random_state=0, max_depth=1)
gb1.fit(X_train, y_train)

print("Accuracy on training set: {:.3f}".format(gb1.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(gb1.score(X_test, y_test)))

gb2 = GradientBoostingClassifier(random_state=0, learning_rate=0.01)
gb2.fit(X_train, y_train)

print("Accuracy on training set: {:.3f}".format(gb2.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(gb2.score(X_test, y_test)))

#Ambos métodos de disminución de la complejidad del modelo redujeron la precisión del conjunto de entrenamiento, como se esperaba. En este caso, ninguno de estos métodos incrementó el rendimiento de generalización del equipo de ensayo.
#Podemos visualizar las características importantes para obtener más información sobre nuestro modelo a pesar de que no estamos realmente contentos con el modelo.

plot_feature_importances_diabetes(gb1)

#Podemos ver que las importancias de características de los árboles impulsados por gradiente son algo similares a las importancias de características de los bosques aleatorios, da peso a todas las características en este caso.

### Apoyo Vector Machine

svc = SVC()
svc.fit(X_train, y_train)

print("Accuracy on training set: {:.2f}".format(svc.score(X_train, y_train)))
print("Accuracy on test set: {:.2f}".format(svc.score(X_test, y_test)))

#El modelo sobresale bastante, con una puntuación perfecta en el set de entrenamiento y una precisión del 65% en el set de pruebas.
#SVM requiere que todas las funciones varíen en una escala similar. Tendremos que volver a analizar nuestros datos que todas las funciones están aproximadamente en la misma escala:

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

svc = SVC()
svc.fit(X_train_scaled, y_train)

print("Accuracy on training set: {:.2f}".format(svc.score(X_train_scaled, y_train)))
print("Accuracy on test set: {:.2f}".format(svc.score(X_test_scaled, y_test)))

#Escalar los datos es una gran diferencia. Ahora estamos en un régimen de infraadaptación, donde el entrenamiento y el rendimiento de los conjuntos de pruebas son bastante similares pero menos cercanos al 100% de precisión. Desde aquí, podemos intentar incrementar C o gamma para encajar en un modelo más complejo.

svc = SVC(C=1000)
svc.fit(X_train_scaled, y_train)

print("Accuracy on training set: {:.3f}".format(
    svc.score(X_train_scaled, y_train)))
print("Accuracy on test set: {:.3f}".format(svc.score(X_test_scaled, y_test)))
