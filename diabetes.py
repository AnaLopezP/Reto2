#Importamos librerias a utilizar
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

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
X_train, X_test, y_train, y_test = train_test_split(diabetes.loc[:, diabetes.columns != 'Outcome