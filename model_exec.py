# model_exec.py>
# kaggle model

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPool2D, Dropout
from tensorflow.keras.utils import to_categorical 
from audio_functions import *


# functions
def model_exec1(X,Y, test_size =0.25, random_state = 1, epochs=3, batch_size=50):
    
    Y = to_categorical(Y)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = test_size, random_state = random_state)
    
    X_train, X_test = repeat_and_reshape(X_train, X_test, 16)
    input_dim = X_test.shape[1:]

    model = Sequential()
    model.add(Conv2D(64, (3, 3), padding = "same", activation = "tanh", input_shape = input_dim))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), padding = "same", activation = "tanh"))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))
    model.add(Flatten())
    model.add(Dense(1024, activation = "tanh"))
    model.add(Dense(10, activation = "softmax"))

    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

    model.fit(X_train, Y_train, epochs = epochs, batch_size = batch_size, validation_data = (X_test, Y_test))

    print(model.summary())



def model_exec2(X,Y, test_size =0.25, random_state = 1, epochs=3, batch_size=50):
    # model
    Y = to_categorical(Y)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = test_size, random_state = random_state)
    
    X_train, X_test = repeat_and_reshape(X_train, X_test, 16)
    input_dim = X_test.shape[1:]

    model = Sequential()
    model.add(Conv2D(64, (5, 5), padding = "same", activation = "tanh", input_shape = input_dim))
    model.add(MaxPool2D(pool_size=(1, 1)))
    model.add(Conv2D(64, (5, 5), padding = "same", activation = "tanh", input_shape = input_dim))
    model.add(MaxPool2D(pool_size=(1, 1)))    
    model.add(Conv2D(128, (3, 3), padding = "same", activation = "tanh"))
    model.add(MaxPool2D(pool_size=(1, 1)))    
    model.add(Conv2D(256, (3, 3), padding = "same", activation = "tanh"))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))
    model.add(Flatten())
    model.add(Dense(1024, activation = "tanh"))
    model.add(Dense(10, activation = "softmax"))

    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

    model.fit(X_train, Y_train, epochs = epochs, batch_size = batch_size, validation_data = (X_test, Y_test))

    print(model.summary())