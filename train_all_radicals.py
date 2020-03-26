import os
from label_images import *
from train_for_rad_detection import * 
import shutil

trained_radicals = os.listdir('alexnet_data_out')
e = RadicalOneHotEncoder(get_all_radicals())

for rad in e.rads:
    message = f""" --- --- --- --- --- ---  --- --- --- --- --- ---

    \t \t Finished training radical {rad}.

     --- --- --- --- --- --- --- --- --- --- --- --- """

    print(message.replace('Finished', 'Starting'))
    if rad not in trained_radicals:
        train_model(rad)
    elif os.listdir(os.path.join('alexnet_data_out', rad)) == []:
        train_model(rad)
    elif os.listdir(os.path.join('alexnet_data_out', rad, 'models')) == []:
        shutil.rmtree(os.path.join('alexnet_data_out', rad, 'models'))
        shutil.rmtree(os.path.join('alexnet_data_out', rad, 'tblogs'))
        train_model(rad)
    else:
        continue

    print(message)
        


