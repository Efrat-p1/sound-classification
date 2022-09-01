import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from model_exec import *

#-------------
print('1. upload data')
#-------------

loaded = np.load('data4000-a.npz')
X= loaded['X']
y= loaded['y']


print('input_dim', X.shape)
print(len(y))


num_classes = len(np.unique(y))
print('num_classes: ', num_classes)

model_exec2(np.mean(X, axis =2), y, test_size=0.25, random_state=1)