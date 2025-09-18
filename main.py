from matplotlib import pyplot as plt
from tensorflow import keras
from keras import layers
from keras.callbacks import CSVLogger
from keras.models import Model, Sequential
from keras.layers import Input, Dense

from src.superposition_wirebreak_noise import *
from src.training_sequentialModel import *
from src.feed_data_to_model import *
from src.preprocess_data import *
from src.evaluate_model_results import plot_loss_function, define_threshold, evaluate_results

from scipy.fft import fft
import tensorflow as tf
import numpy as np
import pandas as pd
import vallenae as vae
import os
    
# Constants
N = 170000
SLIDING_WINDOW = 1024


if __name__=='__main__':

    local   = True
    cluster = False
    train   = True


    if local:
        noise_path = ''
        new_noise_path = ''
        wirebreak_tradbPath = ''
        model_path = ''
        model_path_f = ''
        path = '/mnt/c/Users/ma90802/Documents/Arbeit/Project/models/VarationalAE'
        training_log = 'loggs/training_varational.log'
    if cluster:
        noise_path = '/data/damage_detection/2020_KISWind/01_Messdaten/2021_WEA-Deining_Betriebsmessungen/tradb/2021_Juni/2021-06-16/2021-06-16_15-00-46_05171_vallen_FileSW.tradb'
        new_noise_path = '/data/damage_detection/2020_KISWind/01_Messdaten/2021_WEA-Deining_Betriebsmessungen/tradb/2021_Juni/2021-06-16/2021-06-16_16-00-46_05171_vallen_FileSW.tradb'
        wirebreak_tradbPath = '/data/damage_detection/2020_KISWind/01_Messdaten/2021_WEA-Deining_Betriebsmessungen/tmp/measurement_data/tradb/20200106_T1_TS1_SDB_01.tradb'
        model_path = '/home/aljamilm/Schreibtisch/MasterArbeit/anomalieDetection/models/LSTM/modelID_4_LSTM_100_Epochs_100_seqLength.model'
        model_path_f = '/home/aljamilm/Schreibtisch/MasterArbeit/anomalieDetection/models/VarationalAE/modelID_4_LSTM_100_Epochs_100_seqLength.model'
        path = '/home/aljamilm/Schreibtisch/MasterArbeit/anomalieDetection/models/VarationalAE'
        training_log = 'loggs/training_varational.log'

    # For schleife: os.walk    
    epochs = 2
    model_t = 'model_ID_6'
    data_t = None
    normalisation = False
    data_processor = Data_Processor(wirebreak_tradbPath, noise_path, new_noise_path, N, normalisation, seq_length=SLIDING_WINDOW, model_type=model_t)
    signal_processor = Process_Wirebreak_Noise()
    model_processor = sequentialModel_AEProcessor(epochs, path, model_type=model_t)
    # Get Data in timeseries
    train_data, target_data, new_noise_test, sup_new_noise = data_processor.prepare_data(signal_processor, data_type=data_t)
    # Get Data in dft
    train_data_f, target_data_f, new_noise_test_f, sup_new_noise_f = data_processor.prepare_data(signal_processor, data_type='dft')

    if train:
        model_path, training_log = model_processor.fit_model(train_data, target_data, data_type=data_t)
        model_path_f, training_log = model_processor.fit_model(train_data_f, target_data_f, data_type='dft')

    data_to_model = Feed_Data(wirebreak_tradbPath, new_noise_path, N, normalisation, model_type=model_t)
    train_mae_ts, test_mae_ts, sup_mae_ts = data_to_model.mae_to_dataframe_timeseries(train_data, new_noise_test, sup_new_noise, model_path)
    train_mae_ft, test_mae_ft, sup_mae_ft = data_to_model.mae_to_dataframe_dft(train_data_f, new_noise_test_f, sup_new_noise_f, model_path_f)
    plot_loss_function(training_log, model_path, model_type=model_t)

    results_path = '/mnt/c/Users/ma90802/Documents/Arbeit/Project/results'
    excel_name = (model_path.split('/')[-1].split('.')[0]) + '.xlsx'
    sheet_name = model_t + '_timeseries'
    percentage_quantile = [.5, .6, .7, .75, .8, .85, .9, .95]
    metrics = []
    headers = ['Percentage Quantile', 'Threshold timeseries', 'Threshold FFT', 'True positives', 'False positives', 'Accuracy', 'F1 Scores']
    for quantile in percentage_quantile:
        threshold_ts, threshold_ft, true_pos, false_pos, accuracy, F1 = evaluate_results((train_mae_ts, train_mae_ft), (test_mae_ts, test_mae_ft), (sup_mae_ts, sup_mae_ft), quantile)
        metrics.append([quantile, threshold_ts, threshold_ft, true_pos, false_pos, accuracy, F1])
    # Create a DataFrame
    results = pd.DataFrame(metrics, columns=headers)
    results.to_excel(os.path.join(results_path, excel_name), index=False, sheet_name='Sheet1')

    