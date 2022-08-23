import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
from scipy import signal
from scipy.io import wavfile
# from numpy import save

def get_spectrogram_data(path, plot=False, fft_size=1024,overlap_ratio=0, restrict_freq=(0,11025)):

    samplerate, data  = wavfile.read(path)               #44100
    
    #if "mono" the data is duplicated else the data is "streo"
    if len(data.shape) == 1:
        x_l =  data     
        x_r =  data 
    else: 
        x_l= data[:, 0]
        x_r= data[:, 1]

    length = len(x_r) / samplerate                  #176400/44100 =time sec

    samplerate_fixed = 22050                        #22050*4 = 88200
    x_l = signal.resample(x_l, int(samplerate_fixed*length))  
    x_r = signal.resample(x_r, int(samplerate_fixed*length))    

    #resize data. all files should get the same length of time
    if len(x_l) <= 88200:                          #176400
        x_l = np.append(x_l,np.zeros(88200-len(x_l)))
        x_r = np.append(x_r,np.zeros(88200-len(x_r)))
    else:
        x_l = x_l[:88200]
        x_r = x_r[:88200]

    # print('len(x_l): ', len(x_l))
    # resample thr data to 22.05 kHZ. 
    # input parameter: samples. therfore fixed_sample_rate * ausio time
 

    results =[]
    for i, x in enumerate([x_l, x_r]):

        N = len(x)                                  #176400
        n = np.arange(N)
        time = N/samplerate_fixed
        freq = N/time                               #should get 22.05kHz
        
        # print('time:', round(time,2))             #4.0

        f_raw, time, Sxx_raw = signal.spectrogram(x, fs=samplerate_fixed, nperseg=fft_size, window=('tukey', 0.25), noverlap=overlap_ratio*fft_size)

        # Sxx_raw = 100*Sxx_raw / np.array(Sxx_raw).sum()

        freq = f_raw[(f_raw>restrict_freq[0]) & (f_raw<restrict_freq[1])]
        Sxx= Sxx_raw[(f_raw>restrict_freq[0]) & (f_raw<restrict_freq[1]),:]

        Sxx_log = np.log10(Sxx + 0.00001)*10

        results.append(Sxx_log)

    return results[0], results[1], time, freq

def get_audio_spec_plot(path, plot = False, fft_size=1024,overlap_ratio=0.25):
    
    # Sxx_log= list contaon two np.arrays: left and right
    print(path)
    Sxx_log_l,Sxx_log_r, time, freq = get_spectrogram_data(path, plot ,fft_size=fft_size,overlap_ratio=overlap_ratio)
    

    
    if plot:

        plt.subplot(4,1,1) 
        plt.pcolormesh(time, np.log10(freq+1), Sxx_log_l, shading='gouraud')
        plt.ylabel('10*log10(Frequency)')
        plt.xlabel('Time [sec]')

        plt.subplot(4,1,2) 
        plt.pcolormesh(time, freq,Sxx_log_l, shading='gouraud')
        plt.ylabel('Frequency')
        plt.xlabel('Time [sec]')

        plt.subplot(4,1,3) 
        plt.pcolormesh(time, np.log10(freq+1), Sxx_log_r, shading='gouraud')
        plt.ylabel('10*log10(Frequency)')
        plt.xlabel('Time [sec]')

        plt.subplot(4,1,4) 
        plt.pcolormesh(time, freq, Sxx_log_r, shading='gouraud')
        plt.ylabel('Frequency')
        plt.xlabel('Time [sec]')

        plt.show() 

def get_audio_df_data(df, fft_size=1024,overlap_ratio=0.25):
    f = open('./output.csv', 'w', newline='')
    writer = csv.writer(f)

    results = []
    for i, path in enumerate(df):
        print(i, " ", path)
        Sxx_log_l,Sxx_log_r, time, freq = get_spectrogram_data(path ,fft_size=fft_size,overlap_ratio=overlap_ratio)

        writer.writerow([i,Sxx_log_l.shape[0],Sxx_log_l.shape[1]])
        # writer.writerow([i,Sxx_log_r.shape[0],Sxx_log_r.shape[1]])
        results.append(Sxx_log_l)
        # results.append(Sxx_log_r)

    f.close()
    print("done")
    return np.array(results)
    # return results

# Construct feature by concatenating fold and file name
files_name = 'files_urban_sound'
my_path = f'./{files_name}/UrbanSound8K.csv'   #f_string


df = pd.read_csv(my_path)
df['relative_path'] =  './' + files_name+ '/fold'+  df['fold'].astype(str) + '/' + df['slice_file_name'].astype(str)
df = df[['relative_path', 'classID']]
print('count: ', df.shape[0])
# np.save('y.npy', df['classID'])

# OPTION 1
# get_audio_spec_plot(df['relative_path'][101],plot=True, fft_size=1024,overlap_ratio=0.25)


# OPTION 2
data = get_audio_df_data(df['relative_path'][:6248], fft_size=1024,overlap_ratio=0.25)   #4803   #all 8732 #6246 fold8/36429-2-0-13.wav  #6249
print(data.shape)
np.save('data6248.npy', data)
np.save('y_6248.npy', df.iloc[:6248]['classID'])


# load numpyy array form files:
# -----------------------------
# data =np.load('data.npy')
# print(data.shape)
# print(data[0].shape)


# y =np.load('y.npy')

# TEST CASE
# -----------
# fs = 1000
# amp = 10
# x = amp*np.sin(2*np.pi*(fs+np.arange(176400)/176)*np.arange(176400)/44100)
# plt.plot(x)
# plt.show()
# wavfile.write('sin_testcase.wav',44100,x)
# get_audio_file_data('sin_testcase.wav', plot=True, fft_size=1024)



