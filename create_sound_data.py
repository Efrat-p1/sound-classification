from audio_functions import get_df_spec_data, get_paths_lables_df
import time
import numpy as np


start_time = time.time()

# user parameters
rows = 8732  #max rows = 8732
save_spec_data= [True,'data8732_256_2048_0.25']

df = get_paths_lables_df('files_urban_sound')
y = df['classID'][:rows]

# 2. get spectrogeam data for each row in the df  (i.e 8000 x 500 x 120)
X_ = get_df_spec_data(df['relative_path'][:rows],
                        fft_size=2048,
                        overlap_ratio=0.25,
                        samplerate_fixed = 22050,
                        n_mels= 256)

# 3. save results to zip file

if save_spec_data[0]:
    np.savez_compressed(save_spec_data[1], X=X_, y=y)
    print(save_spec_data[1], " saved")


print('rows: ', rows)
print(round((time.time() - start_time),0), " sec")
print(round((time.time() - start_time)/60,1), " min")