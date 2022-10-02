import timeit
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold,train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from pandas import read_csv
from sklearn.model_selection import StratifiedKFold
np.random.seed(7)


def normalizar(d):
    d = d - np.min(d, axis= 0)
    d = d / np.ptp(d, axis = 0)
    return d

def rango(d):
    d = np.multiply(d, 0.89) 
    d = np.add(d, 0.01)
    return d




def modelo_constructor_1(input_dim,hidden_dim,output_dim):
    model = Sequential()
    model.add(Dense_Co(output_dim,hidden_dim=hidden_dim ,input_dim=input_dim))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    return model
def modelo_constructor(input_dim,hidden_dim,output_dim):
    model = Sequential()
    model.add(Dense_Co(output_dim,hidden_dim=hidden_dim ,input_dim=input_dim,  kernel_initializer=RandomNormal(
            mean=0.0, stddev=0.04), bias_initializer=RandomNormal(mean=0.0, stddev=0.04)))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    return model

epochs=5
batch_size=1
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
cvscores = []