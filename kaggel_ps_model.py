# https://www.kaggle.com/code/prabhavsingh/urbansound8k-classification
# https://towardsdatascience.com/audio-deep-learning-made-simple-part-2-why-mel-spectrograms-perform-better-aad889a93505

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from model_exec import *
from audio_functions import *

import librosa
import librosa.display

import time



start_time = time.time()

df = get_paths_lables_df('files_urban_sound')


dat1, sampling_rate1 = librosa.load(df['relative_path'][150])
dat2, sampling_rate2 = librosa.load(df['relative_path'][1500])

row_num = 500 #  df.shape[0]

# melspectrogram
dat1, sampling_rate1 = librosa.load(df['relative_path'][0])
arr = librosa.feature.melspectrogram(y=dat1, sr=sampling_rate1)
print(arr.shape)
plt.imshow(arr, cmap=plt.cm.binary)
# plt.show()

feature = []
label = []

def parser(df,row_num):
    # Function to load files and extract features
    for i in range(row_num):
        
        file_name = df['relative_path'][i]
        # Here kaiser_fast is a technique used for faster extraction
        X, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
        # We extract mfcc feature from data
        mels = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T,axis=0)        
        feature.append(mels)
        label.append(df["classID"][i])
    return [feature, label]

temp = parser(df, row_num)
print(len(temp[0]))

print('done2')

temp = np.array(temp)
data = temp.transpose()

X_ = data[:, 0]
Y = data[:, 1]
print(X_.shape, Y.shape)
X = np.empty([row_num, 128])

for i in range(row_num):
    X[i] = (X_[i])


model_exec1(X,Y, test_size=0.25, random_state=1)

print('model: ', "kaggel-PS")
print('rows: ', row_num)
print(round((time.time() - start_time),0), " sec")
print(round((time.time() - start_time)/60,1), " min")