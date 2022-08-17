import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
from scipy import signal
from scipy.io import wavfile

# from scipy.fft import fftshift
# rng   = np.random.default_rng()

files_name = 'files_urban_sound'

# Construct feature by concatenating fold and file name
my_path = './'+ files_name + r'/UrbanSound8K.csv'
df = pd.read_csv(my_path)
df['relative_path'] =  './'+ files_name +'/fold' + df['fold'].astype(str) + '/' + df['slice_file_name'].astype(str)
df = df[['relative_path', 'classID']]


def get_spectrogram(path, plot=True, restrict_freq=(0,1000000)):

    samplerate, data  = wavfile.read(path)               #44100
    length = data.shape[0] / samplerate                  #176400/44100

    if len(data.shape) == 1:
        x_l =  data     
        x_r =  data 
    else: 
        x_l= data[:, 0]
        x_r= data[:, 1]
    
    results=[]
    for x in [x_l, x_r]:
        N = len(x)                                                   #176400
        n = np.arange(N)
        T = N/samplerate
        freq = n/T                                                              #4.0

        y= np.fft.fft(x)

        f_raw, t, Sxx_raw = signal.spectrogram(x, fs=samplerate, nperseg=1024)
        
        
        restrict_freq=(100,200000)
        f = f_raw[(f_raw>restrict_freq[0]) & (f_raw<restrict_freq[1])]
        Sxx= Sxx_raw[(f_raw>restrict_freq[0]) & (f_raw<restrict_freq[1]),:]

        Sxx_log = np.log10(Sxx)*10
    
        if plot:
            # plt.pcolormesh(t, f, Sxx, shading='gouraud')
            plt.pcolormesh(t, np.log10(f+1), Sxx_log, shading='gouraud')

            plt.ylabel('10*log10(Frequency)')
            plt.xlabel('Time [sec]')
            plt.show()
        
        results.append([samplerate , length, len(x) ,f.shape[0], t.shape[0], Sxx_log.shape[0],Sxx_log.shape[1]]) 

    return results


f = open('./output.csv', 'w', newline='')
writer = csv.writer(f)


for i, path in enumerate(df['relative_path'][0:100]):
    print(i)
    # path, samplerate , length, len_x_l, len_x_r ,f_shape, t_shape, Sxx_log_shape = 
    # print(f'samplerate; {samplerate} ,length: {length:.1f} ,len_x_l: {len_x_l:8} ,len_x_r: {len_x_r:8} ,f_shape: {f_shape} ,t_shape: {t_shape}  ,Sxx_log_shape: {Sxx_log_shape}')
    row = get_spectrogram(path,plot=False)
    row[0].append(i)
    row[1].append(i)
    writer.writerow(row[0])
    writer.writerow(row[1])
f.close()
print("done")



