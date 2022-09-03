import numpy as np
from model_exec import model_exec1, model_exec2
from audio_functions import mean_and_repeate, mean_and_fold, spectrogram_matrix
# from audio_functions import get_spectrogram_data, signal_plotting, spec_plotting, repeat_and_reshape
import time
import tensorflow as tf
from tensorflow.keras.utils import to_categorical 
from sklearn.model_selection import train_test_split

# -------------------
# 1. user parameters
# -------------------
processing_spect_data = 'SM'   
# VD-	Vector Duplicated 
# SM- 	Spectrogram Matrix
# F- 	Folding (Kaggle method)


start_time = time.time()

upload_file =False
upload_file_name = 'data8732_256_2048_0.25.npz'


# 1. upload data file 
loaded = np.load(upload_file_name)
X_= loaded['X']
y= loaded['y']

# 2. split to train and test
Y = to_categorical(y)
X_train_, X_test_, Y_train, Y_test = train_test_split(X_, Y, random_state = 1)  #6000,256,57


# 3. Processing spectrogram data before modeling

if processing_spect_data == 'VD': 
    X_train, X_test = mean_and_repeate(X_train_, X_test_, dup_times = 16)   # (6549, 256, 16, 1)

elif processing_spect_data == "SM": 
    X_train, X_test = spectrogram_matrix(X_train_, X_test_)                 # (6549, 256, 57, 1)

elif processing_spect_data == "F": 
    X_train, X_test = mean_and_fold(X_train_, X_test_, ratio="1:1")         # (6549, 16, 16, 1)


# 4. NN modeling
with tf.device('/GPU:0'):
    model_exec1(X_train, X_test, Y_train, Y_test, epochs=20, batch_size=100)



print('model: ', "our_model")
print(round((time.time() - start_time),0), " sec")
print(round((time.time() - start_time)/60,1), " min")