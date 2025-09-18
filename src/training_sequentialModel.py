"""
In this script signals of wind turbine records are preprocessed and given as input to an autoencoder model.

Author: Mouhammed Al Jamil
14.05.2022
@Leibniz Uni
"""

from matplotlib import pyplot as plt
from tensorflow import keras
from keras import layers
from keras.callbacks import CSVLogger
from keras.models import Model, Sequential
from keras.layers import Input, Dense
from scipy.fft import fft
import tensorflow as tf
import numpy as np
import pandas as pd
import vallenae as vae
import os
import yaml


class sequentialModel_AEProcessor:
    
    def __init__(self, epochs_nr: int, path_to_save: str, SLIDING_WINDOW: int,  model_type: str):
        self.epochs_nr = epochs_nr
        self.model_type = model_type
        self.path_to_save = path_to_save
        self.window = SLIDING_WINDOW

    def fit_model(self, train_data, target_data, day, data_type):
        day = day.split('/')[-1]
        if self.model_type == 'model_ID_1':
            training_log = 'loggs/training_varational.log'
            csv_logger = CSVLogger(training_log, separator=',', append=False)
            if data_type == 'dft':
                model_name = f"modelID_1_{self.epochs_nr}_Epochs_{self.window}_seqLength_dft_" + day + ".model"
            else:
                model_name = f"modelID_1_{self.epochs_nr}_Epochs_{self.window}_seqLength_2_layers_" + day + ".model"
            input_dim = train_data.shape[1]

            encoder = tf.keras.Sequential([
                tf.keras.layers.Dense(input_dim, activation='relu', input_shape=(input_dim,)),
                tf.keras.layers.Dense(int(input_dim/2), activation='relu'),
                tf.keras.layers.Dense(int(input_dim/4), activation='relu'),
                tf.keras.layers.Dense(int(input_dim/6), activation='relu')
            ])

            decoder = tf.keras.Sequential([
                tf.keras.layers.Dense(int(input_dim/4), activation='relu', input_shape=(int(input_dim/6),)),
                tf.keras.layers.Dense(int(input_dim/2), activation='relu'),
                tf.keras.layers.Dense(input_dim, activation='sigmoid'),
            ])
            autoencoder = tf.keras.Model(inputs=encoder.input, outputs=decoder(encoder.output))

            autoencoder.compile(loss='mse', optimizer='adam')
            autoencoder.summary()

            model_history = autoencoder.fit(
                train_data,
                train_data,
                epochs=self.epochs_nr,
                batch_size=32,
                validation_split=0.1,
                callbacks=[csv_logger]
            )
            autoencoder.save(os.path.join(self.path_to_save, model_name))

            return os.path.join(self.path_to_save, model_name), training_log

        #Try LTSM model
        elif self.model_type == 'model_ID_5':
            training_log = 'loggs/training_LSTM.log'
            csv_logger = CSVLogger(training_log, separator=',', append=False)
            model_name = f"modelID_5_{self.epochs_nr}_Epochs_{self.window}_seqLength.model"

            model = tf.keras.Sequential()
            model.add(tf.keras.layers.LSTM(units=128, input_shape=(train_data.shape[1], train_data.shape[2])))
            model.add(tf.keras.layers.Dropout(rate=0.2))

            model.add(tf.keras.layers.RepeatVector(train_data.shape[1]))

            model.add(tf.keras.layers.LSTM(units=128, return_sequences=True))
            model.add(tf.keras.layers.Dropout(rate=0.2))
            model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=train_data.shape[2])))
            model.compile(optimizer='adam', loss='mae')
            model.summary()

            # fit model
            model_history = model.fit(
                train_data,
                target_data,
                epochs=self.epochs_nr,
                batch_size=32,
                validation_split=0.1,
                shuffle=False,
                callbacks=[csv_logger]
            )
            model.save(os.path.join(self.path_to_save, model_name))
            return os.path.join(self.path_to_save, model_name), training_log
        
        elif self.model_type == 'model_ID_3':
            training_log = 'loggs/training_model_ID_3.log'
            csv_logger = CSVLogger(training_log, separator=',', append=False)
            if data_type == 'dft':
                model_name = f"modelID_1_{self.epochs_nr}_Epochs_{self.window}_seqLength_dft.model"
            else:
                model_name = f"modelID_1_{self.epochs_nr}_Epochs_{self.window}_seqLength_timeseries.model"

            # Define input shape
            if data_type == 'dft':
                input_shape = (512,)
            else:
                input_shape = (1024,)

            # Encoder
            inputs = Input(shape=input_shape)
            encoded = Dense(768, activation='relu')(inputs)
            encoded = Dense(256, activation='relu')(encoded)

            # Decoder
            decoded = Dense(256, activation='relu')(encoded)
            decoded = Dense(768, activation='relu')(decoded)
            decoded = Dense(input_shape[0], activation='sigmoid')(decoded)

            # Create the autoencoder model
            model_complex = Model(inputs=inputs, outputs=decoded)

            # Compile the model
            model_complex.compile(optimizer='adam', loss='mean_squared_error')

            # Print the model summary
            model_complex.summary()

            # fit model
            model_history = model_complex.fit(
                train_data,
                train_data,
                epochs=self.epochs_nr,
                batch_size=32,
                validation_split=0.1,
                shuffle=False,
                callbacks=[csv_logger]
            )
            model_complex.save(os.path.join(self.path_to_save, model_name))
            return os.path.join(self.path_to_save, model_name), training_log
        
        elif self.model_type == 'model_ID_4':
            training_log = 'loggs/training_model_ID_4.log'
            csv_logger = CSVLogger(training_log, separator=',', append=False)
            model_name = f"modelID_4_{self.epochs_nr}_Epochs_{self.window}_seqLength.model"

            # Define input shape
            if data_type == 'dft':
                input_shape = (512,)
            else:
                input_shape = (1024,)

            # Encoder
            inputs = Input(shape=input_shape)
            encoded = Dense(256, activation='relu')(encoded)

            # Decoder
            decoded = Dense(256, activation='relu')(encoded)
            decoded = Dense(input_shape[0], activation='sigmoid')(decoded)

            # Create the autoencoder model
            model_complex = Model(inputs=inputs, outputs=decoded)

            # Compile the model
            model_complex.compile(optimizer='adam', loss='mean_squared_error')

            # Print the model summary
            model_complex.summary()

            # fit model
            model_history = model_complex.fit(
                train_data,
                train_data,
                epochs=self.epochs_nr,
                batch_size=32,
                validation_split=0.1,
                shuffle=False,
                callbacks=[csv_logger]
            )
            model_complex.save(os.path.join(self.path_to_save, model_name))
            return os.path.join(self.path_to_save, model_name), training_log
        
        elif self.model_type == 'model_ID_7':
            training_log = 'loggs/training_model_ID_7.log'
            csv_logger = CSVLogger(training_log, separator=',', append=False)
            if data_type == 'dft':
                model_name = f"modelID_7_{self.epochs_nr}_Epochs_{self.window}_seqLength_dft.model"
            else:
                model_name = f"modelID_7_{self.epochs_nr}_Epochs_{self.window}_seqLength_timeseries.model"

            input_shape = (1024, 1)
            # Encoder
            encoder_inputs = layers.Input(shape=input_shape)
            encoder = layers.LSTM(64, return_sequences=True)(encoder_inputs)
            encoder = layers.Dropout(0.2)(encoder)  # Add dropout to the first LSTM layer
            encoder = layers.LSTM(32)(encoder)      # Second LSTM layer with 32 units

            # Decoder
            decoder = layers.RepeatVector(input_shape[0])(encoder)
            decoder = layers.LSTM(32, return_sequences=True)(decoder)  # Second LSTM layer in decoder
            decoder = layers.LSTM(64, return_sequences=True)(decoder)  # First LSTM layer in decoder
            decoder_outputs = layers.TimeDistributed(layers.Dense(input_shape[1]))(decoder)

            # Autoencoder model
            autoencoder = tf.keras.Model(encoder_inputs, decoder_outputs, name="autoencoder")
            # Compile the model
            autoencoder.compile(optimizer=tf.keras.optimizers.Adam(), loss='mse')

            # Print the model summary
            autoencoder.summary()

            # fit model
            model_history = autoencoder.fit(
                train_data,
                train_data,
                epochs=self.epochs_nr,
                batch_size=32,
                validation_split=0.1,
                shuffle=False,
                callbacks=[csv_logger]
            )
            autoencoder.save(os.path.join(self.path_to_save, model_name))
            return os.path.join(self.path_to_save, model_name), training_log
    
    def create_transformer_model(self, seq_length, d_model, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout_rate):
        inputs = layers.Input(shape=(seq_length, 1))
        x = inputs
        for _ in range(num_transformer_blocks):
            # Multi-head self-attention
            x = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
            x = layers.Dropout(rate=dropout_rate)(x)
            x = layers.LayerNormalization(epsilon=1e-6)(x)

            # Feed-forward neural network
            x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
            x = layers.Dropout(rate=dropout_rate)(x)
            x = layers.LayerNormalization(epsilon=1e-6)(x)

        # Time-distributed dense layer
        x = layers.TimeDistributed(layers.Dense(units=mlp_units, activation="relu"))(x)
        x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)

        # Output layer
        outputs = layers.Dense(units=1, activation="linear")(x)
        
        return tf.keras.Model(inputs=inputs, outputs=outputs)
    
    def fit_TranAD(self, seq_length, d_model, num_heads, ff_dim, num_transformer_blocks, mlp_unites, dropout_rate):
        model_name = f"modelID_5_{100}_Epochs_{1024}_seqLength.model"
        training_log = 'loggs/training_TranAD.log'
        csv_logger = CSVLogger(training_log, separator=',', append=False)
        train_data, target_data = self.prepare_data(data_type=None)
        model = self.create_transformer_model(seq_length, d_model, num_heads, ff_dim, num_transformer_blocks, mlp_unites, dropout_rate)
        # Compile the model
        model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.MeanSquaredError())
        model.summary()

        callbacks = [keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]
        # Train the model
        model.fit(train_data, target_data, batch_size=64, epochs=100, validation_split=0.2, callbacks=[csv_logger, callbacks])
        model.save(os.path.join('models/TranAD', model_name))
        

if __name__ == '__main__':

    local = True
    cluster = False
    if local:
        sensor_configFile = '/mnt/c/Users/ma90802/Documents/MasterArbeit/Project/sensorcalibration_Deining2021.json'
        noise_tradbPath = '/mnt/c/Users/ma90802/Documents/MasterArbeit/Project/measurement_data/noise/tradb/2021-06-16_15-00-46_05171_vallen_FileSW.tradb'
        wirebreak_tradbPath = '/mnt/c/Users/ma90802/Documents/MasterArbeit/Project/measurement_data/wirebreak/tradb/20200106_T1_TS1_SDB_01.tradb'
        pridb_path = '/mnt/c/Users/ma90802/Documents/MasterArbeit/Project/measurement_data/noise/pridb'
        csv_file = '/mnt/c/Users/ma90802/Documents/MasterArbeit/Project/csv/oneSecondSignal.csv'
    elif cluster:
        noise_tradbPath = 'measeurment_data/noise/2021-06-16_15-00-46_05171_vallen_FileSW.tradb'
        wirebreak_tradbPath = 'measeurment_data/wirebreak/20200106_T1_TS1_SDB_01.tradb'

    #####################################################################################################
    # Process half an hour of the day 16.06.2021 using tradb file 2021-06-16_15-00-46_05171_vallen_FileSW
    # x number of samples in each Sequence will be given separetly to the Autoencoder. This means that in 
    # half an hour 170e6/x sample set will be used as training data.
    # Create the model and train it with sequence data.
    # Parameters: 
    #   sliding window: amount of samples in each sequence
    #   Window hop time: time period of signal to be extracted from tradb file
    #   Epochs number: for how many epochs the model to be trained
    #   validation loss fraction: fraction of data part to be used for validation
    #####################################################################################################
    epochs_number = 100
    model_path = 'models/AE_Test_WholeTradb_1024_' + str(epochs_number) + '.model'
    model_processor = sequentialModel_AEProcessor(wirebreak_tradbPath, noise_tradbPath, epochs_number, model_type="LSTM")

    # Create and train model
    model_processor.fit_model(data_type=None)




        




