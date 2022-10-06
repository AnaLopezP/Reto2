#Importamos librerias a utilizar
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#leemos csv
diabetes = pd.read_csv('diabetes.csv')

#columnas
print(diabetes.columns )
print('\n')
#dimensión
print('dimensión diabetes: ' + str(diabetes.shape) + '\n')
#Agrupamos datos
print(diabetes.groupby('Datos agrupados').size())
