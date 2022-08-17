
from struct import pack_into
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPool2D, Dropout
from tensorflow.keras.utils import to_categorical 


#https://www.kaggle.com/code/prabhavsingh/urbansound8k-classification

my_path = r'.\test\UrbanSound8K.csv'
df = pd.read_csv(my_path)
# print(df)

'''Using random samples to observe difference in waveforms.'''

arr = np.array(df["slice_file_name"])
fold = np.array(df["fold"])
cla = np.array(df["class"])

for i in range(5):
    path = r'.\test\fold' + str(fold[i]) + '/' + arr[i]
    data, sampling_rate = librosa.load(path)
    
    plt.subplot(5, 1, i+1)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(data)), ref=np.max)
    
    librosa.display.specshow(D, y_axis='linear')
    plt.colorbar(format='%+2.0f dB')
    plt.title(cla[i])

plt.show()

'''EXAMPLE'''

dat1, sampling_rate1 = librosa.load(r'test\fold1\7061-6-0-0.wav')
arr = librosa.feature.melspectrogram(y=dat1, sr=sampling_rate1)
print(arr.shape)

feature = []
label = []

def parser(row):
    # Function to load files and extract features
    for i in range(df.shape[0]):
        file_name = r'.\test\fold' + str(np.array(df["fold"])[i]) + '/' +  np.array(df["slice_file_name"])[i]
        
        # Here kaiser_fast is a technique used for faster extraction
        X, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
        # We extract mfcc feature from data
        mels = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T,axis=0)        
        feature.append(mels)
        label.append(df["classID"][i])
    return [feature, label]
print("finish")
temp = parser(df)
temp = np.array(temp)
data = temp.transpose()
X_ = data[:, 0]
Y = data[:, 1]
print(X_.shape, Y.shape)
X = np.empty([df.shape[0], 128])
print('X:')
print(X)

Y = to_categorical(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state = 1)

'''Final Data'''
print('X_train:')
print(X_train.shape)

print('X_test:')
print(X_test.shape)



X_train = X_train.reshape(X_train.shape[0], 16, 8, 1)
X_test = X_test.reshape(X_test.shape[0], 16, 8, 1)
input_dim = (16, 8, 1)
print("finish_EDA")


#model
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
model.fit(X_train, Y_train, epochs = 90, batch_size = 50, validation_data = (X_test, Y_test))
model.summary()

predictions = model.predict(X_test)
score = model.evaluate(X_test, Y_test)
print(score)

preds = np.argmax(predictions, axis = 1)

result = pd.DataFrame(preds)

print(result)
print("finish_prediction")