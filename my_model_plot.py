from audio_functions import *

row_num = 15

df = get_paths_lables_df('files_urban_sound')


print('path: ', df['relative_path'][row_num])
spec_plotting(df['relative_path'][row_num])


# signal_plotting(df['relative_path'][row_num])