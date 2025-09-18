from matplotlib import pyplot as plt
from scipy.fft import fft
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import numpy as np
import pandas as pd
import vallenae as vae
import yaml

class Data_Processor:

    def __init__(self, wirebreak_tradbPath, N, normalise, seq_length, model_type):
        self.wirebreak_path = wirebreak_tradbPath
        self.model_type = model_type
        self.number_of_samples = N
        self.seq_length = seq_length
        self.normalise = normalise

    def get_signals(self, signal_processor):
        noise = signal_processor.get_noise_signal(self.noise_path)
        noise = noise[:, 0:self.number_of_samples]
        return noise
    
    def create_sequences(self, values):
        x = []
        y = []
        if self.model_type == 'model_ID_5':
            data = values.reshape(-1)
            for i in range(len(data) - self.seq_length):
                x.append(data[i:(i+self.seq_length)])
                y.append(data[i+self.seq_length])
        else:
            data = values.reshape(-1)
            idx = len(data)-self.seq_length
            for i in range(0, idx+1, self.seq_length):
                x.append(data[i:i+self.seq_length])
        return np.array(x), np.array(y)
    
    def pad_signal(self, signals, desired_shape):
        padding_shape = [(0, 0) if i >= len(desired_shape) else (0, desired_shape[i] - signals.shape[i]) for i in range(len(signals.shape))]
        padded_signals = np.pad(signals, padding_shape, mode='constant')
        return padded_signals
    
    def normalise_data(self, data):
        # for idx in range(sequence.shape[0]):
        #     sequence[idx] = (sequence[idx] - np.min(sequence[idx])) / (np.max(sequence[idx]) - np.min(sequence[idx]))
        #     sequence[idx] = 2 * sequence[idx] - 1
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_data = scaler.fit_transform(data)
        return scaled_data

    def to_fft(self, ts_data):
        fft_list = []
        for idx in range(ts_data.shape[0]):
            seq_fft = np.abs(fft(ts_data[idx, :ts_data.shape[1]//2]))
            fft_list.append(seq_fft)
        fft_data = np.array(fft_list)
        return fft_data

    # def prepare_data(self, signal_processor, data_type):
    #     data = self.get_signals(signal_processor)
    #     train_data, target_data = self.create_sequences(data)
    #     if data_type == 'dft':
    #         train_data = self.to_fft(train_data)
    #     if self.normalise:
    #         train_data = self.normalise_data(train_data)
    #     if self.model_type == 'model_ID_5' or self.model_type == 'TranAD':
    #         # Reshape the sequences for LSTM input
    #         train_data = np.reshape(train_data, (train_data.shape[0], train_data.shape[1], 1))
    #     return train_data, target_data
    
    def prepare_data(self, signal_processor, noise_path, new_noise_path, data_type):
        valid = True
        wirebreak = signal_processor.get_wirebreak(self.wirebreak_path, 400)
        valid, noise = signal_processor.get_noise_signal(noise_path)
        if not valid:
            train_data = 0
            target_data = 0
            new_noise_test = 0
            sup_newNoise_wb = 0
            return valid, train_data,  target_data, new_noise_test, sup_newNoise_wb
        valid, new_noise_pure = signal_processor.get_noise_signal(new_noise_path)
        noise = noise[:, 0:self.number_of_samples]
        new_noise_pure = new_noise_pure[:, 0:self.number_of_samples]

        train_data, target_data = self.create_sequences(noise)
        new_noise_test, target_test = self.create_sequences(new_noise_pure) #New noise data for testing
        if data_type == 'dft':
            wirebreak = self.to_fft(wirebreak)
            train_data = self.to_fft(train_data)
            new_noise_test = self.to_fft(new_noise_test)
        sup_newNoise_wb = signal_processor.superposition_noise_wirebreak(wirebreak, new_noise_test, samples_in_seq=1024)
        if self.model_type == 'model_ID_7':
            # Reshape the sequences for LSTM input
            train_data = np.reshape(train_data, (train_data.shape[0], train_data.shape[1], 1))
            new_noise_test = np.reshape(new_noise_test, (new_noise_test.shape[0], new_noise_test.shape[1], 1))
            sup_newNoise_wb = np.reshape(sup_newNoise_wb, (sup_newNoise_wb.shape[0], sup_newNoise_wb.shape[1], 1))
        if self.normalise:
            train_data = self.normalise_data(train_data)
            new_noise_test = self.normalise_data(new_noise_test)
            sup_newNoise_wb = self.normalise_data(sup_newNoise_wb)
        return valid, train_data,  target_data, new_noise_test, sup_newNoise_wb

