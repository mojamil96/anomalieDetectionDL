from preprocess_data import *
from superposition_wirebreak_noise import *
import numpy as np
import pandas as pd
import vallenae as vae
import os
import sys

# Constants
N = 170000000
SLIDING_WINDOW = 1024

if __name__=='__main__':

    local   = True
    cluster = False
    train   = True

    if local:
        noise_path = '/mnt/c/Users/ma90802/Documents/rArbeit/Project/measurement_data/noise/tradb/2021-06-16_15-00-46_05171_vallen_FileSW.tradb'
        new_noise_path = '/mnt/c/Users/ma90802/Documents/Arbeit/Project/measurement_data/noise/tradb/2021-06-16_16-00-46_05171_vallen_FileSW.tradb'
        wirebreak_tradbPath = '/mnt/c/Users/ma90802/Documents/Arbeit/Project/measurement_data/wirebreak/tradb/20200106_T1_TS1_SDB_01.tradb'
        model_path = '/mnt/c/Users/ma90802/Documents/Arbeit/Project/models/VarationalAE/modelID_1_5_Epochs_1024_seqLength_timeseries.model'
        model_path_f = '/mnt/c/Users/ma90802/Documents/Arbeit/Project/models/VarationalAE/modelID_1_5_Epochs_1024_seqLength_dft.model'
        path = '/mnt/c/Users/ma90802/Documents/Arbeit/Project/models/VarationalAE'
        day_path = '/mnt/c/Users/ma90802/Documents/Arbeit/Project/measurement_data/noise/tradb'
        training_log = 'loggs/training_varational.log'
    if cluster:
        noise_path = '/data/damage_detection/2020_KISWind/01_Messdaten/2021_WEA-Deining_Betriebsmessungen/tradb/2021_Juni/2021-06-16/2021-06-16_15-00-46_05171_vallen_FileSW.tradb'
        new_noise_path = '/data/damage_detection/2020_KISWind/01_Messdaten/2021_WEA-Deining_Betriebsmessungen/tradb/2021_Juni/2021-06-16/2021-06-16_16-00-46_05171_vallen_FileSW.tradb'
        wirebreak_tradbPath = '/data/damage_detection/2020_KISWind/01_Messdaten/2021_WEA-Deining_Betriebsmessungen/tmp/measurement_data/tradb/20200106_T1_TS1_SDB_01.tradb'
        model_path = '/home/aljamilm/Schreibtisch/MasterArbeit/anomalieDetection/models/VarationalAE/modelID_1_25_Epochs_1024_seqLength_timeseries_dataNotSacled.model'
        model_path_f = '/home/aljamilm/Schreibtisch/MasterArbeit/anomalieDetection/models/VarationalAE/modelID_1_25_Epochs_1024_seqLength_dft.model'
        path = '/home/aljamilm/Schreibtisch/MasterArbeit/anomalieDetection/models/VarationalAE'
        day_path = '/data/damage_detection/2020_KISWind/01_Messdaten/2021_WEA-Deining_Betriebsmessungen/tradb/2021_Juni/2021-06-16'
        training_log = 'loggs/training_varational.log'

    model_t = 'model_ID_1'
    data_t = None
    normalisation = False
    data_processor = Data_Processor(wirebreak_tradbPath, N, normalisation, seq_length=SLIDING_WINDOW, model_type=model_t)
    signal_processor = Process_Wirebreak_Noise()

    noise_tradb_path = None
    train_data      = []
    for file in os.listdir(day_path):
        filename = os.fsdecode(file)
        if filename.endswith(".tradb"):
            noise_tradb_path = os.path.join(day_path, filename)
        else: 
            continue 
        # Get Data in timeseries
        valid, train_data_file, target_data, new_noise_file, sup_new_noise_file = data_processor.prepare_data(signal_processor, noise_tradb_path, new_noise_path, data_type=data_t)
        # Get Data in dft
        valid, train_data_f_file, target_data_f, new_noise_test_f, sup_new_noise_f = data_processor.prepare_data(signal_processor, noise_tradb_path, new_noise_path, data_type='dft')
        if not valid:
            print("Skipping file " + filename + " due to invalid result.")
            continue
        train_data.append(train_data_file)
        print("Size of list: " + str(sys.getsizeof(train_data)) + "bytes")
        # train_data_f.append(train_data_f_file)
    if noise_tradb_path is None:
        raise Exception("No tradb files were found!") 
    train_data = np.vstack(train_data)
    # train_data_f = np.vstack(train_data_f)
    train_data_df = pd.DataFrame(train_data)
    train_data_df.to_csv('/home/aljamilm/Schreibtisch/MasterArbeit/anomalieDetection/training_data_sequences.csv')
