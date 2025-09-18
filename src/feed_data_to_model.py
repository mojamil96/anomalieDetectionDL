"""
In this script signals of wind turbine records are fed as input to the trained autoencoder model resulting MAE vectors as outputs

Author: Mouhammed Al Jamil
18.04.2023
@Leibniz Uni
"""

from matplotlib import pyplot as plt
from tensorflow import keras
from keras import layers
from keras.callbacks import CSVLogger
from scipy.fft import fft
import tensorflow as tf
import numpy as np
import pandas as pd
import vallenae as vae
import os

class Feed_Data():

    def __init__(self, wirebreak_tradbPath, new_noise_path, noise_length, normalise, model_type):
        self.wirebreak_tradbPath = wirebreak_tradbPath
        self.new_noise_path      = new_noise_path 
        self.noise_length        = noise_length
        self.model_type          = model_type
        self.normalise           = normalise

    def mae_to_dataframe_timeseries(self, train_data, test_data, superimposed_newNoise_wb, model_path):

        model_name = model_path.split('/')[-1].split('.')[0]
        csv_path = os.path.join('csv', ('evaluation_' + model_name + '_2' + '.csv'))
        
        autoencoder = tf.keras.models.load_model(model_path)

        x_pred_train = autoencoder.predict(train_data)
        # x_pred_oldSuperimposed_test = autoencoder.predict(superimposed_train_wb)
        x_pred_noise_test = autoencoder.predict(test_data)
        x_pred_newSuperimposed_test = autoencoder.predict(superimposed_newNoise_wb)

        train_mae = np.mean(np.abs(x_pred_train - train_data), axis=1)
        # superimposed_old_noise_mae = np.mean(np.abs(x_pred_oldSuperimposed_test - superimposed_train_wb), axis=1)
        new_noise_mae = np.mean(np.abs(x_pred_noise_test - test_data), axis=1)
        superimposed_new_noise_mae = np.mean(np.abs(x_pred_newSuperimposed_test - superimposed_newNoise_wb), axis=1)
        if np.array_equal(new_noise_mae, superimposed_new_noise_mae):
            print("Noise and Sup MAE are equal!")
        return train_mae, new_noise_mae, superimposed_new_noise_mae

    def mae_to_dataframe_dft(self, train_data, test_data, superimposed_newNoise_wb, model_path):

        model_name = model_path.split('/')[-1].split('.')[0]
        csv_path = os.path.join('csv', ('evaluation_' + model_name + '_2' + '.csv'))

        autoencoder = tf.keras.models.load_model(model_path)

        x_pred_train = autoencoder.predict(train_data)
        # x_pred_oldSuperimposed_test = autoencoder.predict(superimposed_train_wb)
        x_pred_noise_test = autoencoder.predict(test_data)
        x_pred_newSuperimposed_test = autoencoder.predict(superimposed_newNoise_wb)

        train_mae = np.mean(np.abs(x_pred_train - train_data), axis=1)
        # superimposed_old_noise_mae = np.mean(np.abs(x_pred_oldSuperimposed_test - superimposed_train_wb), axis=1)
        new_noise_mae = np.mean(np.abs(x_pred_noise_test - test_data), axis=1)
        superimposed_new_noise_mae = np.mean(np.abs(x_pred_newSuperimposed_test - superimposed_newNoise_wb), axis=1)
        return train_mae, new_noise_mae, superimposed_new_noise_mae

    def mae_to_dataframe_timeseries_LSTM(self, train_data, test_data, superimposed_newNoise_wb, model_path):

        model_name = model_path.split('/')[-1].split('.')[0]
        csv_path = os.path.join('csv', ('evaluation_' + model_name + '_2' + '.csv'))
        
        autoencoder = tf.keras.models.load_model(model_path)

        if np.array_equal(test_data, superimposed_newNoise_wb):
            print("Error!")
        x_pred_train = autoencoder.predict(train_data)
        # x_pred_oldSuperimposed_test = autoencoder.predict(superimposed_train_wb)
        x_pred_noise_test = autoencoder.predict(test_data)
        x_pred_newSuperimposed_test = autoencoder.predict(superimposed_newNoise_wb)

        train_mae = np.mean(np.abs(x_pred_train - train_data), axis=(1,2))
        # superimposed_old_noise_mae = np.mean(np.abs(x_pred_oldSuperimposed_test - superimposed_train_wb), axis=1)
        new_noise_mae = np.mean(np.abs(x_pred_noise_test - test_data), axis=(1,2))
        superimposed_new_noise_mae = np.mean(np.abs(x_pred_newSuperimposed_test - superimposed_newNoise_wb), axis=(1,2))
        if np.array_equal(new_noise_mae, superimposed_new_noise_mae):
            print("Noise and Sup MAE are equal!")
        return train_mae, new_noise_mae, superimposed_new_noise_mae
        
if __name__ == '__main__':

    local = True
    cluster = False

    if local:
        noise_path = '/mnt/c/Users/ma90802/Documents/MasterArbeit/Project/measurement_data/noise/tradb/2021-06-16_15-00-46_05171_vallen_FileSW.tradb'
        new_noise_path = '/mnt/c/Users/ma90802/Documents/MasterArbeit/Project/measurement_data/noise/tradb/2021-06-16_16-00-46_05171_vallen_FileSW.tradb'
        wirebreak_tradbPath = '/mnt/c/Users/ma90802/Documents/MasterArbeit/Project/measurement_data/wirebreak/tradb/20200106_T1_TS1_SDB_01.tradb'
        model_path = '/mnt/c/Users/ma90802/Documents/MasterArbeit/Project/models/VarationalAE/modelID_3_complex_50_Epochs.model'
    if cluster:
        noise_path = '/home/aljamilm/Schreibtisch/MasterArbeit/anomalieDetection/measeurment_data/noise/2021-06-16_15-00-46_05171_vallen_FileSW.tradb'
        new_noise_path = '/home/aljamilm/Schreibtisch/MasterArbeit/anomalieDetection/measeurment_data/noise/2021-06-16_16-00-46_05171_vallen_FileSW.tradb'
        wirebreak_tradbPath = '/home/aljamilm/Schreibtisch/MasterArbeit/anomalieDetection/measeurment_data/wirebreaks/20200106_T1_TS1_SDB_01.tradb'
        model_path = '/home/aljamilm/Schreibtisch/MasterArbeit/LSTM_Test_WholeTradb_1024_10.model'

    #####################################################################################################
    # Prepare test data using differnt noise data that differ from the original noise data used for 
    # training (new tradb-file)
    # Calculate the resulted MAE of pure noise data as well as superimposed data and save info to dataframe
    #####################################################################################################
    mae_to_dataframe(model_path, data_type='time_series')
    