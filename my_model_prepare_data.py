from model_exec import *
from audio_functions import *
import time

start_time = time.time()
# from numpy import save

# Construct feature by concatenating fold and file name
df = get_paths_lables_df('files_urban_sound')


rows = 50 
# rows = df.shape[0]
y_labels = df['classID'][:rows]
data = get_df_spec_data(df['relative_path'][:rows], fft_size=1024,overlap_ratio=0.75)

# data = np.array(data)
# r = []
# for sxx in data:
#     sxx_i = np.array(sxx)
#     r.append(np.mean(sxx_i, axis =1))
# r = np.array(r)
# Exec

model_exec2(np.mean(data, axis =2), y_labels, test_size=0.25, random_state=1, epochs=5, batch_size=5)
# model_exec(data , y_labels, test_size=0.25, random_state=1)


# np.savez_compressed('data8732-a', X=data, y=y_labels)
# print("done")



print('model: ', "our_model")
print('rows: ', rows)
print(round((time.time() - start_time),0), " sec")
print(round((time.time() - start_time)/60,1), " min")