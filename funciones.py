import timeit
import numpy as np
import pandas as pd
from keras.models import Sequiential
from sklearn.model_selection import KFold,train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from pandas import read_csv
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.initializers import RandomNormal, glorot_normal
from sklearn.model_selection import StratifiedKFold
np.random.seed(7)