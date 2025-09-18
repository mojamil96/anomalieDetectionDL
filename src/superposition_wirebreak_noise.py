"""
Process class serves for extracting and superpositioning wirebreak signals and noise data to be used in testing the functionality of the anamoly detection algorithm (anomalieProjekt script)

Author: Mouhammed Al Jamil
18.03.223
@Leibniz Uni
"""
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import vallenae as vae
import os
import random
import scipy


class Process_Wirebreak_Noise():

    def __init__(self):
        pass

    def read_tra(self, tradb_path):
        tra = vae.io.TraDatabase(tradb_path)
        return tra
    
    def get_wirebreak(self, wirebreak_tradb_path, desired_length, window_hoptime = None, window_time = None):
        wirebreak_tra = self.read_tra(wirebreak_tradb_path)
        print("Rows: ", wirebreak_tra.rows())
        if window_hoptime is None and window_time is None:
            wirebreak, t = wirebreak_tra.read_continuous_wave(channel=8)
        else:
            count = 0
            if window_time > wirebreak_tra.rows()//8:
               raise Exception('Time slot selected bigger than total time of file to be processed')
            while(window_hoptime*count+window_time <= wirebreak_tra.rows()//8):
                wirebreak, t = wirebreak_tra.read_continuous_wave(channel=1, time_start=window_hoptime*count, time_stop=window_hoptime*count+window_time)
                count += 1
        print("Length of wirebreak signal before downsampling: ", len(wirebreak))
        wirebreak_downsampled = self.downsample_signal(wirebreak, downsampling_factors=[5,4])
        if wirebreak_downsampled.shape[1] > desired_length:
            wirebreak_downsampled = wirebreak_downsampled[:, 0:desired_length]
        print('Shape after downsamling: ', wirebreak_downsampled.shape)
        return wirebreak_downsampled
    
    def get_wirebreak_signals(self, wirebreak_tadb_path, desired_length, onlyOne = None):
        tmp_wirebreaks = pd.DataFrame()
        channels = 8
        for filename in os.listdir(wirebreak_tadb_path):
            if filename.endswith(".tradb"):
                path = os.path.join(wirebreak_tadb_path, filename)
                tradata = vae.io.TraDatabase(path)
                if tradata == None:
                    continue
                for channel in range(channels):
                    wirebreak, t = tradata.read_continuous_wave(channel=channel)
                    if wirebreak.shape[0]==0:
                        print("Couldn't extract data from file")
                        continue
                    wirebreak_downsampled = self.downsample_signal(wirebreak.reshape(1,-1), downsampling_factors=[5,4])
                    print("Shape:", wirebreak_downsampled.shape)
                    if wirebreak_downsampled.shape[1] > desired_length:
                        wirebreak_downsampled = wirebreak_downsampled[:, 0:desired_length]
                    wirebreaks = pd.DataFrame(wirebreak_downsampled, columns=np.arange(0,wirebreak_downsampled.shape[1]))
                    tmp_wirebreaks=pd.concat([tmp_wirebreaks,wirebreaks],axis=0)
            if onlyOne:
                break
        tmp_wirebreaks.fillna(0,inplace=True)
        return tmp_wirebreaks

    def get_noise_signal(self, noise_tradb_path, window_hoptime = None, window_time = None):
        valid = True
        noise_tra = self.read_tra(noise_tradb_path)
        if window_hoptime is None and window_time is None:
            try:
                noise_signal, t = noise_tra.read_continuous_wave(channel=1)
            except:
                print(f"File {noise_tradb_path} is corrupted")
                noise_signal = 0
                valid = False
                return valid, noise_signal
            if len(noise_signal) == 0:
                print(f"File {noise_tradb_path} is corrupted!")
                valid = False
                return valid, noise_signal
        else:
            count = 0
            if window_time > noise_tra.rows()//8:
                raise Exception('Time slot selected bigger than total time of file to be processed')
            while(window_hoptime*count+window_time <= noise_tra.rows()//8):
                noise_signal, t = noise_tra.read_continuous_wave(channel=1, time_start=window_hoptime*count, time_stop=window_hoptime*count+window_time)
                count+=1
        return valid, noise_signal.reshape(1,-1)
    
    def downsample_signal(self, rawsignals, downsampling_factors: list):
        overall_downsampling_factor=1
        for factor in downsampling_factors:
            overall_downsampling_factor*=factor
            if factor > 10:
                raise ValueError('Downsampling factor should be an Integer!')
        tmp_downsampled = rawsignals
        for q in downsampling_factors:
            tmp_downsampled = scipy.signal.decimate(tmp_downsampled, q=q)
        return tmp_downsampled.reshape(1,-1)
    
    def create_sequences(self, values, sliding_window):
        output = []
        data = values.reshape(-1)
        idx = len(data)-sliding_window
        for i in range(0, idx+1, sliding_window):
            output.append(data[i:i+sliding_window])
        x = np.stack(output)
        return x

    def create_sequences_LSTM(self, values, sliding_window):
        output = []
        data = np.transpose(values)
        print("Shape after reshaping: ", data.shape)
        idx = len(data)-sliding_window

        for i in range(0, idx+1, sliding_window):
            output.append(data[i:i+sliding_window])
        x = np.array(output)
        print("Shape = ", x.shape)
        print("Type: ", type(x))
        return x
    
    def superposition_one_sequence_1024(self, wirebreak, noise):
        superimposed_data = self.create_sequences(noise, 1024)
        # idx = random.randrange(superimposed_data.shape[0])
        idx = 200
        wirebreak = wirebreak*(np.max(np.abs(superimposed_data[idx]))/(np.max(np.abs(wirebreak)))).reshape(-1,1)
        superimposed_slice = superimposed_data[idx, 512:512+wirebreak.shape[1]]
        superimposed_slice = superimposed_slice + wirebreak
        return superimposed_data
    
    def superposition_one_sequence_2048(self, wirebreak, noise):
        superimposed_data = self.create_sequences(noise, 2048)
        # idx = random.randrange(superimposed_data.shape[0])
        idx = 200
        print(len(superimposed_data[idx]))
        superimposed_slice = superimposed_data[idx, 1024:1024+wirebreak.shape[1]]
        superimposed_data[idx, 1024:1024+wirebreak.shape[1]] = superimposed_slice + wirebreak*(np.max(np.abs(superimposed_data[idx]))/(np.max(np.abs(wirebreak)))).reshape(-1,1)
        return superimposed_data

    def superposition_noise_wirebreak(self, wirebreak, noise, samples_in_seq):
        superimposed_data = noise.copy()
        superimpose_index = int(superimposed_data.shape[1]/2)
        for idx in range(superimposed_data.shape[0]):
            superimposed_data[idx, superimpose_index:superimpose_index+wirebreak.shape[1]] = superimposed_data[idx, superimpose_index:superimpose_index+wirebreak.shape[1]] + wirebreak*(np.max(np.abs(superimposed_data[idx]))/(np.max(np.abs(wirebreak)))).reshape(-1,1)
        return superimposed_data

    def superposition_noise_wirebreak_LTSM(self, wirebreak, noise_seq, samples_in_seq):
        superimposed_data = noise_seq
        for idx in range(superimposed_data.shape[0]):
            wirebreak = wirebreak*(np.max(np.abs(superimposed_data[idx]))/(np.max(np.abs(wirebreak)))).reshape(-1,1)
            x = superimposed_data[idx, :wirebreak.shape[1]].flatten()
            x = x + wirebreak
            superimposed_data[idx, :wirebreak.shape[1]] = x.reshape(x.shape[1],1)
        return superimposed_data
    
    def superposition_noise_multipleWirebreaks(self, wirebreaks: pd.DataFrame, noise):
        superimposed_data = self.create_sequences(noise, sliding_window=1024)
        for idx_noise in range(0, superimposed_data.shape[0], wirebreaks.shape[0]):
            for idx_wb in range(wirebreaks.shape[0]):
                wb = np.array(wirebreaks.iloc[idx_wb,:])
                wb = wb*(np.max(np.abs(superimposed_data[idx_noise]))/(np.max(np.abs(wb)))).reshape(-1,1)
                superimposed_data[idx_noise, :wb.shape[1]] = superimposed_data[idx_noise, :wb.shape[1]] + wb
                idx_noise+=1
                if idx_noise == superimposed_data.shape[0]:
                    break
        return superimposed_data

    def superposition_noise_wirebreak_noSequences(self, wirebreak, noise):
        superimposed_data = noise.copy()
        superimposed_index = int(superimposed_data.shape[1]/2)
        superimposed_data[0, superimposed_index:superimposed_index+wirebreak.shape[1]] = superimposed_data[0, superimposed_index:superimposed_index+wirebreak.shape[1]] + wirebreak*(np.max(np.abs(superimposed_data[0]))/(np.max(np.abs(wirebreak)))).reshape(-1,1)

        return superimposed_data, superimposed_index


if __name__ == '__main__':

    wirebreak_tradb = '/mnt/c/Users/ma90802/Documents/MasterArbeit/Project/measurement_data/wirebreak/tradb/20200106_T1_TS1_SDB_01.tradb'
    noise_tradb = '/mnt/c/Users/ma90802/Documents/MasterArbeit/Project/measurement_data/noise/tradb/2021-06-16_15-00-46_05171_vallen_FileSW.tradb'

    NOISE_HOPTIME = 1700
    NOISE_WINDOWTIME = 1700

    signal_processor = Process_Wirebreak_Noise()

    wirebreak = signal_processor.get_wirebreak(wirebreak_tradb)
    # print('Wirebreak: ')
    # print('Shape of Wirebreak: ', wirebreak.shape)
    # print('Length: ', len(wirebreak))

    noise = signal_processor.get_noise_signal(noise_tradb, NOISE_HOPTIME, NOISE_WINDOWTIME)
    # print('Noise: ')
    # print('Shape: ', noise.shape)
    # print('Length: ', len(noise))
    superimposed = signal_processor.superposition_noise_wirebreak(wirebreak, noise, samples_in_seq=1024)
    # noise_seq = signal_processor.create_sequences(noise, 2048)
    # x = wirebreak*(np.max(np.abs(noise_seq[2,:]))/(np.max(np.abs(wirebreak)))).reshape(-1,1)
    # print('shape of x: ', x.shape)


