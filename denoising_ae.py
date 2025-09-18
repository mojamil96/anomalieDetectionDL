
from matplotlib import pyplot as plt
from tensorflow import keras
from keras import layers
from keras.callbacks import CSVLogger
from keras.models import Model, Sequential
from keras.layers import Input, Dense
from superposition_wirebreak_noise import *
from scipy.fft import fft
import tensorflow as tf
import numpy as np
import pandas as pd
import vallenae as vae
import math

#Constants 
N = 170000000

class Denoising_AE:

    def __init__(self, model_path, noise_path, wirebreak_path, epochs, csv_log):

        self.model_path = model_path
        self.noise_path = noise_path
        self.wirebreak_path = wirebreak_path
        self.epochs = epochs
        self.csv_log = csv_log
        
        self.N = N
        
    def get_signals(self):
        signal_processor = Process_Wirebreak_Noise()
        
        noise = signal_processor.get_noise_signal(self.noise_path)
        noise = noise[:, 0:N]
        wirebreaks = signal_processor.get_wirebreak_signals(self.wirebreak_path, 1024)
        superimposed_train = signal_processor.superposition_noise_multipleWirebreaks(wirebreaks, noise)
        wbs = wirebreaks.to_numpy()
        desired_shape = (wbs.shape[0], 1024)
        padded_wbs = self.pad_signal(wbs, desired_shape)

        return padded_wbs, superimposed_train

    def create_sequences(self, values):
        output = []
        sliding_window = 1024
        data = values.reshape(-1)
        idx = len(data)-sliding_window
        for i in range(0, idx+1, sliding_window):
            output.append(data[i:i+sliding_window])
        x = np.array(output)
        return x

    def to_fft(self, ts_data):
        fft_list = []
        for idx in range(ts_data.shape[0]):
            seq_fft = np.abs(fft(ts_data[idx, :ts_data.shape[1]//2]))
            fft_list.append(seq_fft)
        fft_data = np.array(fft_list)
        return fft_data

    def calculate_rms(self, signal):
        square = 0
        for i in range(len(signal)):
            square += signal[i]**2
        rms = math.sqrt(np.mean(square))
        return rms

    def calcualte_noise_power(self, nosie):
        power_noise = np.mean(np.square(noise))
        return power_noise

    def calculate_snr(self, signal, noise):
        power_noise = np.mean(np.square(noise))
        power_signal = np.mean(np.square(signal))

        SNR = 10 * math.log10(power_signal / power_noise)
        return SNR

    def pad_signal(self, signals, desired_shape):
        padding_shape = [(0, 0) if i >= len(desired_shape) else (0, desired_shape[i] - signals.shape[i]) for i in range(len(signals.shape))]
        padded_signals = np.pad(signals, padding_shape, mode='constant')
        return padded_signals

    def normalise_data(self, sequence):
        for idx in range(sequence.shape[0]):
            sequence[idx] = (sequence[idx] - np.min(sequence[idx])) / (np.max(sequence[idx]) - np.min(sequence[idx]))
        return sequence

    def get_new_noise(self, noise_path):
        new_noise = signal_processor.get_noise_signal(noise_path)
        new_noise = new_noise[:, 0:N]
        new_noise_seq = self.create_sequences(new_noise)
        return new_noise_seq

    def prepare_data(self):
        padded_wirebreaks, superimposed_data = self.get_signals()

        # Create Sequences that match with train data dimensions
        padded_wbs_seq = []
        idx = 0
        for i in range(0, superimposed_data.shape[0]):
            padded_wbs_seq.append(padded_wirebreaks[idx])
            idx+=1
            if idx == (padded_wirebreaks.shape[0]-1):
                idx = 0
        padded_wbs_seq = np.stack(padded_wbs_seq)

        # Normalise Data
        target_data = self.normalise_data(padded_wbs_seq)
        train_data = self.normalise_data(superimposed_data)

        return target_data, train_data, padded_wbs_seq, superimposed_data


    def fit_model(self, model_type = None):

        target_data, train_data = self.prepare_data()
        if model_type is None:
            model_name = f"modelID_1_{self.epochs}_Epochs_Denoising.model"
            # Define the input shape and noise level
            input_shape = (1024,)
            noise_level = 0.2

            # Encoder
            model_1 = Sequential()
            model_1.add(Dense(512, activation='relu', input_shape=input_shape))
            model_1.add(Dense(256, activation='relu'))

            # Decoder
            model_1.add(Dense(512, activation='relu'))
            model_1.add(Dense(1024, activation='sigmoid'))  # Output layer, using sigmoid activation for values between 0 and 1

            # Compile the model_1
            model_1.compile(optimizer='adam', loss='mse')

            # Print the model_1 summary
            model_1.summary()

            model_history = model_1.fit(
                train_data,
                target_data,
                epochs=self.epochs,
                batch_size=32,
                validation_split=0.1,
                callbacks=[self.csv_log]
            )
            model_1.save(os.path.join(self.model_path, model_name))
            return os.path.join(self.model_path, model_name)

        if model_type == 'complex':
            model_name = f"modelID_3_{self.epochs}_Epochs_Denoising.model"
            # Define input shape
            input_shape = (1024,)

            # Encoder
            inputs = Input(shape=input_shape)
            encoded = Dense(512, activation='relu')(inputs)
            encoded = Dense(256, activation='relu')(encoded)
            encoded = Dense(128, activation='relu')(encoded)
            encoded = Dense(64, activation='relu')(encoded)
            encoded = Dense(32, activation='relu')(encoded)

            # Decoder
            decoded = Dense(64, activation='relu')(encoded)
            decoded = Dense(128, activation='relu')(decoded)
            decoded = Dense(256, activation='relu')(decoded)
            decoded = Dense(512, activation='relu')(decoded)
            decoded = Dense(1024, activation='sigmoid')(decoded)

            # Create the autoencoder model
            model_complex = Model(inputs=inputs, outputs=decoded)

            # Compile the model
            model_complex.compile(optimizer='adam', loss='mean_squared_error')

            # Print the model summary
            model_complex.summary()

            # fit model
            model_history = model_complex.fit(
                train_data,
                target_data,
                epochs=self.epochs,
                batch_size=32,
                validation_split=0.1,
                shuffle=False,
                callbacks=[self.csv_log]
            )
            model_complex.save(os.path.join(self.model_path, model_name))
            return os.path.join(self.model_path, model_name)

    def predict_from_model(self, model_name, random_sample = None):
        metrics = []
        target_data, train_data, padded_wbs_seq, superimposed_data = self.prepare_data()
        
        model = tf.keras.models.load_model(model_name)
        denoised_signal = model.predict(train_data)
        if random_sample is None:
            random_sample = 10
        rms_predicted = self.calculate_rms(denoised_signal[random_sample])
        rms_original = self.calculate_rms(padded_wbs_seq[random_sample])
        rms_noise = self.calculate_rms(superimposed_data[random_sample])

        snr_predicted = self.calculate_snr(denoised_signal[random_sample], noise)
        snr_clean = self.calculate_snr(padded_wbs_seq[random_sample], noise)
        snr_noisy = self.calculate_snr(superimposed_data[random_sample], noise)

        metrics.append([snr_predicted, snr_clean, snr_noisy])
        results_path = '/home/aljamilm/Schreibtisch/MasterArbeit/anomalieDetection/results'
        excel_name = (model_name.split('/')[-1].split('.')[0]) + '.xlsx'
        headers = ['SNR_Predicted', 'SNR_Clean', 'SNR_Noisy']
        results = pd.DataFrame(metrics, columns=headers)
        results.to_excel(os.path.join(results_path, excel_name), index=False, sheet_name='Sheet1')



if __name__ == '__main__':

    local   = True
    cluster = False
    train   = False


    if local:
        noise_path = '/mnt/c/Users/ma90802/Documents/MasterArbeit/Project/measurement_data/noise/tradb/2021-06-16_15-00-46_05171_vallen_FileSW.tradb'
        new_noise_path = '/mnt/c/Users/ma90802/Documents/MasterArbeit/Project/measurement_data/noise/tradb/2021-06-16_16-00-46_05171_vallen_FileSW.tradb'
        wirebreak_tradbPath = '/mnt/c/Users/ma90802/Documents/MasterArbeit/Project/measurement_data/wirebreak/tradb'
        model_path = '/mnt/c/Users/ma90802/Documents/MasterArbeit/Project/models/VarationalAE/modelID_1_5_Epochs_1024_seqLength_dft.model'
        path = '/mnt/c/Users/ma90802/Documents/MasterArbeit/Project/models/VarationalAE'
        training_log = 'loggs/training_varational.log'
    if cluster:
        noise_path = '/data/damage_detection/2020_KISWind/01_Messdaten/2021_WEA-Deining_Betriebsmessungen/tradb/2021_Juni/2021-06-16/2021-06-16_15-00-46_05171_vallen_FileSW.tradb'
        new_noise_path = '/data/damage_detection/2020_KISWind/01_Messdaten/2021_WEA-Deining_Betriebsmessungen/tradb/2021_Juni/2021-06-16/2021-06-16_16-00-46_05171_vallen_FileSW.tradb'
        wirebreak_tradbPath = '/data/damage_detection/2020_KISWind/01_Messdaten/2021_WEA-Deining_Betriebsmessungen/tmp/measurement_data/tradb'
        model_path = '/home/aljamilm/Schreibtisch/MasterArbeit/anomalieDetection/models/VarationalAE/modelID_1_25_Epochs_1024_seqLength_timeseries.model'
        path = '/home/aljamilm/Schreibtisch/MasterArbeit/anomalieDetection/models/DenoisingAE'

    epochs = 50
    csv_log = CSVLogger('training_denoising_model_1.log', separator=',', append=False)
    
    denoising_processor = Denoising_AE(model_path, noise_path, path, epochs, csv_log)
    model_name = denoising_processor.fit_model()
    denoising_processor.predict_from_model(model_name)
