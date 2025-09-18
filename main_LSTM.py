from matplotlib import pyplot as plt
from tensorflow import keras
from keras import layers
from keras.callbacks import CSVLogger
from keras.models import Model, Sequential
from keras.layers import Input, Dense
from superposition_wirebreak_noise import *
from training_sequentialModel import *
from feed_data_to_model import *
from preprocess_data import *
from evaluate_model_results import plot_loss_function, evaluate_model_LSTM, evaluate_results
from scipy.fft import fft
import tensorflow as tf
import numpy as np
import pandas as pd
import vallenae as vae
import os

# Constants
N = 170000
SLIDING_WINDOW = 512


if __name__=='__main__':

    local   = False
    cluster = True
    train   = True


    if local:
        noise_path = '/mnt/c/Users/ma90802/Documents/Arbeit/Project/measurement_data/noise/tradb/2021-06-16_15-00-46_05171_vallen_FileSW.tradb'
        new_noise_path = '/mnt/c/Users/ma90802/Documents/Arbeit/Project/measurement_data/noise/tradb/2021-06-16_16-00-46_05171_vallen_FileSW.tradb'
        wirebreak_tradbPath = '/mnt/c/Users/ma90802/Documents/Arbeit/Project/measurement_data/wirebreak/tradb/20200106_T1_TS1_SDB_01.tradb'
        model_path = '/mnt/c/Users/ma90802/Documents/Arbeit/Project/models/VarationalAE/modelID_7_30_Epochs_1024_seqLength_timeseries.model'
        model_path_f = '/mnt/c/Users/ma90802/Documents/Arbeit/Project/models/VarationalAE/modelID_7_30_Epochs_1024_seqLength_timeseries.model'
        path = '/mnt/c/Users/ma90802/Documents/Arbeit/Project/models/VarationalAE'
        training_log = '/mnt/c/Users/ma90802/Documents/Arbeit/Project/loggs/training_model_ID_7.log'
    if cluster:
        noise_path = '/data/damage_detection/2020_KISWind/01_Messdaten/2021_WEA-Deining_Betriebsmessungen/tradb/2021_Juni/2021-06-16/2021-06-16_15-00-46_05171_vallen_FileSW.tradb'
        new_noise_path = '/data/damage_detection/2020_KISWind/01_Messdaten/2021_WEA-Deining_Betriebsmessungen/tradb/2021_Juni/2021-06-16/2021-06-16_16-00-46_05171_vallen_FileSW.tradb'
        wirebreak_tradbPath = '/data/damage_detection/2020_KISWind/01_Messdaten/2021_WEA-Deining_Betriebsmessungen/tmp/measurement_data/tradb/20200106_T1_TS1_SDB_01.tradb'
        model_path = '/home/aljamilm/Schreibtisch/MasterArbeit/anomalieDetection/models/VarationalAE/modelID_7_30_Epochs_1024_seqLength_timeseries.model'
        model_path_f = '/home/aljamilm/Schreibtisch/MasterArbeit/anomalieDetection/models/VarationalAE/modelID_7_30_Epochs_1024_seqLength_timeseries.model'
        days_path = '/data/damage_detection/2020_KISWind/01_Messdaten/2021_WEA-Deining_Betriebsmessungen/tradb/2021_Juni'
        path = '/home/aljamilm/Schreibtisch/MasterArbeit/anomalieDetection/models/VarationalAE'
        training_log = 'loggs/training_varational.log'
    
    # For schleife: os.walk    
    epochs = 25
    model_t = 'model_ID_6'
    data_t = None
    normalisation = False
    data_processor = Data_Processor(wirebreak_tradbPath, noise_path, new_noise_path, N, normalisation, seq_length=SLIDING_WINDOW, model_type=model_t)
    signal_processor = Process_Wirebreak_Noise()
    model_processor = sequentialModel_AEProcessor(epochs, path, SLIDING_WINDOW, model_type=model_t)
    # Get Data in timeseries
    train_data, target_data, new_noise_test, sup_new_noise = data_processor.prepare_data(signal_processor, noise_path, new_noise_pathdata_type=data_t)

    if train:
        model_path, training_log = model_processor.fit_model(train_data, target_data, days_path, data_type=data_t)

    data_to_model = Feed_Data(wirebreak_tradbPath, new_noise_path, N, normalisation, model_type=model_t)
    train_mae_ts, test_mae_ts, sup_mae_ts = data_to_model.mae_to_dataframe_timeseries_LSTM(train_data, new_noise_test, sup_new_noise, model_path)
    # plot_loss_function(training_log, model_path, model_type=model_t)

    results_path = '/home/aljamilm/Schreibtisch/MasterArbeit/anomalieDetection/results'
    # results_path = '/mnt/c/Users/ma90802/Documents/Arbeit/Project/results'
    excel_name = (model_path.split('/')[-1].split('.')[0]) + '.xlsx'
    sheet_name = model_t + '_timeseries'
    percentage_quantile = [.5, .6, .7, .75, .8, .85, .9, .95]
    metrics = []
    headers = ['Percentage Quantile', 'Total Samples','Threshold', 'True positives', 'False positives', 'Accuracy', 'F1 Scores']
    for quantile in percentage_quantile:
        threshold_ts, true_pos, false_pos, accuracy, F1 = evaluate_model_LSTM(train_mae_ts, test_mae_ts, sup_mae_ts, quantile)
        metrics.append([quantile, len(train_mae_ts), threshold_ts, true_pos, false_pos, accuracy, F1])
    # Create a DataFrame
    results = pd.DataFrame(metrics, columns=headers)
    results.to_excel(os.path.join(results_path, excel_name), index=False, sheet_name='Sheet1')

    
    