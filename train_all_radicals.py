import os
from label_images import RadicalOneHotEncoder, get_all_radicals 
from train_loop_single_rad import train_model
import shutil

trained_radicals = os.listdir('data_out')
e = RadicalOneHotEncoder(get_all_radicals())

warm = True
num_epochs = 3 
batch_size = 512

message = lambda radical, status: print(f"""
--- --- --- --- --- --- --- --- --- --- --- ---

\t \t {status} training radical {radical}.

--- --- --- --- --- --- --- --- --- --- --- ---""")

for radical in e.rads:

    message(radical, 'Starting')
    
    if radical not in trained_radicals or os.listdir(os.path.join('data_out', radical)) == []:
        train_model(radical, warm, num_epochs, batch_size)
    elif os.listdir(os.path.join('data_out', radical, 'models')) == []:
        shutil.rmtree(os.path.join('data_out', radical, 'models'))
        shutil.rmtree(os.path.join('data_out', radical, 'tblogs'))
        train_model(radical, warm, num_epochs, batch_size) 
    else:
        message(radical, 'Skipping')
        continue

    message(radical, 'Finished')
        


