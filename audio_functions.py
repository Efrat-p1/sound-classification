# audio_functions.py>
 
import numpy as np
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
import pandas as pd
from scipy.fft import fft, fftfreq

def get_spectrogram_data(path, fft_size=1024,overlap_ratio=0, restrict_freq=(0,11025)):

    samplerate, audiodata  = wavfile.read(path)               #44100
    data = audiodata.astype(float)
    
    #if "mono" the data is duplicated else the data is "streo"
    if len(data.shape) > 1:
        data = audiodata.sum(axis=1) / 2

    N = len(data)  
    time = N / samplerate                   #176400/44100 =time sec                  

    samplerate_fixed = 22050                        #22050*4 = 88200  #should get 22.05kHz
    data = signal.resample(data, int(samplerate_fixed*time))  
    time = N/samplerate_fixed
    samplerate = samplerate_fixed


    # resize data. all files should get the same length of time
    if len(data) < 88200:                          #176400
        data = np.append(data,np.zeros(88200-len(data)))
    else:
        data = data[:88200]

    
    # print('len(x_l): ', len(x_l))
    # resample thr data to 22.05 kHZ. 
    # input parameter: samples. therfore fixed_sample_rate * audio time
                          
    
    # print('time:', round(time,2))             #4.0

    f_raw, time, Sxx_raw = signal.spectrogram(data, fs=samplerate, nperseg=fft_size, window="hann", noverlap=overlap_ratio*fft_size)

    # Sxx_raw = 100*Sxx_raw / np.array(Sxx_raw).sum()

    freq = f_raw[(f_raw>=restrict_freq[0]) & (f_raw<=restrict_freq[1])]
    Sxx= Sxx_raw[(f_raw>=restrict_freq[0]) & (f_raw<=restrict_freq[1]),:]
    
    
    total = np.sum(Sxx)
    Sxx= Sxx/total
    
    Sxx_log = 10*np.log10(Sxx + 0.00001)
    Sxx_log = Sxx_log - Sxx_log.min()
    # Sxx_log = Sxx_log.T

    return Sxx_log, time, freq

def signal_plotting(path):
    
    samplerate, audiodata  = wavfile.read(path)               #44100
    data = audiodata.astype(float)
    
    #if "mono" the data is duplicated else the data is "streo"
    if len(data.shape) > 1:
        data = audiodata.sum(axis=1) / 2
    data = data
    N= len(data)
    yf = fft(data)
    xf = fftfreq(N, samplerate)[:N//2]

    plt.subplot(2,1,1) 
    plt.plot(data)
    plt.ylabel('amplitude')
    plt.xlabel('Time')
    plt.title('audio signal')

    plt.subplot(2,1,2) 
    plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
    # plt.ylabel('amplitude?')
    plt.xlabel('frequncy')
    plt.title('audio signal')

    plt.tight_layout()
    plt.grid()
    plt.show()

    
def spec_plotting(path, fft_size=1024,overlap_ratio=0.25):
    
    # Sxx_log= list contaon two np.arrays: left and right
    Sxx_log, time, freq = get_spectrogram_data(path ,fft_size=fft_size,overlap_ratio=overlap_ratio)


    plt.subplot(2,1,1) 
    plt.pcolormesh(time, freq,Sxx_log, shading='gouraud')
    plt.ylabel('Frequency')
    plt.xlabel('Time [sec]')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Linear-frequency power spectrogram')


    plt.subplot(2,1,2) 
    plt.pcolormesh(time, np.log10(freq+1), Sxx_log, shading='gouraud')
    plt.ylabel('10*log10(f)')
    plt.xlabel('Time [sec]')
    plt.colorbar(format='%+2.0f dB')
    plt.title('logarithmic -frequency power spectrogram')

    plt.tight_layout()
    plt.show() 

def get_df_spec_data(df, fft_size=1024,overlap_ratio=0.25):

    results = []
    for i, path in enumerate(df):
        if i% 50==0:
            print (round(i/df.shape[0]*100,0),"%" )

        # print(i, " ", path)
        Sxx_log, time, freq = get_spectrogram_data(path ,fft_size=fft_size,overlap_ratio=overlap_ratio)
        results.append(Sxx_log)
    
    # return np.array(results)
    return results

def get_paths_lables_df(folder_name):
    if folder_name == 'files_urban_sound':
        path = f'./{folder_name}/UrbanSound8K.csv'   #f_string
        df = pd.read_csv(path)
        df['relative_path'] =  './' + folder_name+ '/fold'+  df['fold'].astype(str) + '/' + df['slice_file_name'].astype(str)
        df = df[['relative_path', 'classID','class']]
        print('count: ', df.shape[0])
    else:
        print("error")
    return df


def repeat_and_reshape(X_train, X_test,dup_size):
    input_size = X_train.shape[1]

    X_train_len = X_train.shape[0]
    X_test_len =X_test.shape[0]

    X_train = np.repeat(X_train, dup_size, axis=1)
    X_train = X_train.reshape(X_train_len, input_size, dup_size, 1)


    X_test = np.repeat(X_test, dup_size, axis=1)
    X_test = X_test.reshape(X_test_len, input_size, dup_size, 1)


    return (X_train, X_test)