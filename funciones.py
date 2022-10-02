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