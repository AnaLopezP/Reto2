import timeit
from keras.models import Model,Sequential
from keras.layers import Input, Embedding, LSTM, Dense,concatenate,  Dropout, Flatten, Conv2D, MaxPool2D, Activation,MaxPooling2D
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.optimizers import RMSprop, adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.datasets import mnist
import tensorflow as tf


import numpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

numpy.random.seed(7)


def X_1(x):
    return (K.pow(x,1))
get_custom_objects().update({'X_1': Activation(X_1)})

def X_2(x):
    return (K.pow(x,2))/2
get_custom_objects().update({'X_2': Activation(X_2)})

def X_3(x):
    return (K.pow(x,3))/6
get_custom_objects().update({'X_3': Activation(X_3)})

def X_4(x):
    return (K.pow(x,4))/24
get_custom_objects().update({'X_4': Activation(X_4)})

def X_5(x):
    return (K.pow(x,5))/120
get_custom_objects().update({'X_5': Activation(X_5)})



#Creates plot for loss and trainning functions
def plot_(history):
    training_loss1 = history.history['loss']
    test_loss1 = history.history['val_loss']
    epoch_count = range(1, len(training_loss1) + 1)
    plt.plot(epoch_count, training_loss1, 'r--')
    plt.plot(epoch_count, test_loss1, 'b-')
    plt.legend(['Training Loss', 'Test Loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

number_train=50000
number_test=10000
X_train=X_train[0:number_train,:,:,:]
y_train=y_train[0:number_train]
X_test=X_test[0:number_test,:,:,:]
y_test=y_test[0:number_test]

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 300.0
X_test = X_test / 300.0

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

epochs=4


input_img = Input(shape = (32, 32, 3))
tower_1 = Conv2D(64, (1,1), padding='same', activation='relu')(input_img)
tower_3 = MaxPooling2D((3,3), strides=(1,1), padding='same')(tower_1)
output = Flatten()(tower_3)

out    = Dense(10, activation='softmax')(output)

model = Model(inputs = input_img, outputs = out)

lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
history=model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=32)
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
plot_(history)

input_img = Input(shape = (32, 32, 3))
tower_1 = Conv2D(64, (1,1), padding='same', activation='X_1')(input_img)
tower_2 = Conv2D(64, (1,1), padding='same', activation='X_2')(input_img)
tower_3 = Conv2D(64, (1,1), padding='same', activation='X_3')(input_img)
concatenate_3_Layers= concatenate([tower_1,tower_2,tower_3])
Out_put_SWAG = Dense(10, activation='linear')(concatenate_3_Layers)

tower_3 = MaxPooling2D((3,3), strides=(1,1), padding='same')(Out_put_SWAG)
output = Flatten()(tower_3)
out    = Dense(10, activation='softmax')(output)
model = Model(inputs = input_img, outputs = out)

lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.summary()
history=model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=32)
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
plot_(history)