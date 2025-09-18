#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A module to load wirebreak signals from Vallen databases (*pridb and *tradb)
"""

#%% Import ext. libraries
import librosa
import vallenae as vae
import datetime
import pandas as pd
import os
import numpy as np
import scipy
import logging
import json
import warnings
import sqlite3

#Constants
AE_REFERENCE = 1E-6 # in [V] = 1 microVolt (for dB-Conversion)

class Vallen_AEProcessor():
    
    def __init__(self, pridb_path: str, tradb_path: str, sensor_config_file = None, excel_summary = None):
        self.pridb_path = pridb_path
        self.tradb_path = tradb_path
        if excel_summary is not None:
            self.wirebreak_infos = self.load_wirebreakinfos_from_excel(excel_summary)
        else:
            self.wirebreak_infos = None
            self.sensor_calibration = self.read_config(sensor_config_file)
            
    def read_config(self, sensor_config_file):
        with open(sensor_config_file, 'r') as f:
            config = json.load(f)
        f.close()
        return config
        
    def load_wirebreakinfos_from_excel(self, excel_summary):
        wb_infos= pd.read_excel(excel_summary)
        wb_infos.dropna(inplace=True)
        wb_infos = wb_infos.astype({"TRAI": int})
        wb_infos = wb_infos.set_index(['Channel','TRAI'])
        return wb_infos
        
    def get_wirebreak_features(self):    
        #if-condition to check, whether to process whole folder or only filenames from excel summary
        if self.wirebreak_infos is not None:
            files_to_traverse = self.wirebreak_infos.drop_duplicates(subset='Filename')['Filename'].to_list()
        else:
            files_to_traverse = [file[:-6] for file in os.listdir(self.pridb_path) if file.endswith('.pridb')]
    
        # the processing of *pridb-files starts here
        for file in files_to_traverse:
            hits = self.load_hit_parameters_from_pridb(os.path.join(file))
            try:
                wirebreak_stats = pd.concat([wirebreak_stats,hits],axis=0)
            except NameError:
                #Copy hits dataframe on first call
                wirebreak_stats = hits.copy()
        return wirebreak_stats
        
    def load_hit_parameters_from_pridb(self, filename):
        try:
            pridata = vae.io.PriDatabase(os.path.join(self.pridb_path, filename+'.pridb'))
            hits = pridata.read_hits()
        except sqlite3.DatabaseError:
            logging.warning('Database Error in File %s. This file is skipped!' % filename)
            return None
        hits = hits.set_index(['channel','trai'])
        hits.index.names = ['Channel','TRAI']
        hits.drop(['param_id'], axis=1, inplace=True)
        hits = hits.rename(columns={"amplitude": "Amplitude [dB]", "duration": "Duration [ms]", "threshold": "Threshold [dB]", "rise_time": "RiseTime [ms]"})
        hits = self.convert_units_pridb(hits, filename)
        return hits
        
    def __convert_units_based_on_excel(self, pridb_hits, filename=None):
        for idx in pridb_hits.index:
            try:
                sys_gain = self.wirebreak_infos.loc[idx,'sys_gain [dB]']
                calc_gain = self.wirebreak_infos.loc[idx,'calc_gain [dB]']
                pridb_hits.loc[idx, 'Amplitude [dB]'] = 20*np.log10(pridb_hits.loc[idx, 'Amplitude [dB]']/AE_REFERENCE)-sys_gain+calc_gain
                pridb_hits.loc[idx, 'Threshold [dB]'] = 20*np.log10(pridb_hits.loc[idx, 'Threshold [dB]']/AE_REFERENCE)    
            except KeyError:
                logging.warning('Skipped one signal in File '+filename)
                pridb_hits.drop(idx,inplace=True)
        # Convert time values
        pridb_hits['Duration [ms]'] *= 1000
        pridb_hits['RiseTime [ms]'] *= 1000
        return pridb_hits
    
    def __convert_units_based_on_sensor_config(self, pridb_hits):
        for ch in ['Ch'+str(x) for x in pd.unique(pridb_hits.index.get_level_values('Channel'))]:
            sys_gain = self.sensor_calibration[ch]['sys_gain [dB]']
            calc_gain = self.sensor_calibration[ch]['calc_gain [dB]']
            pridb_hits.loc[pridb_hits.index.get_level_values('Channel')==int(ch[2:]),'Amplitude [dB]'] = 20*np.log10(pridb_hits.loc[pridb_hits.index.get_level_values('Channel')==int(ch[2:]),'Amplitude [dB]']/AE_REFERENCE)+sys_gain-calc_gain
            pridb_hits['Threshold [dB]'] = 20*np.log10(pridb_hits['Threshold [dB]']/AE_REFERENCE)    

        # Convert time values
        pridb_hits['Duration [ms]'] *= 1000
        pridb_hits['RiseTime [ms]'] *= 1000
        return pridb_hits
    
    def convert_units_pridb(self, pridb_hits, filename=None):
        # Convert Amplitude values
        if self.wirebreak_infos is not None:
            return self.__convert_units_based_on_excel(pridb_hits, filename)
        else:
            return self.__convert_units_based_on_sensor_config(pridb_hits)
    
    def get_wirebreak_signals(self, sample_interval = None):
        #if-condition to check, whether to process whole folder or only filenames from excel summary
        if self.wirebreak_infos is not None:
            files_to_traverse = self.wirebreak_infos.drop_duplicates(subset='Filename')['Filename'].to_list()
        else:
            files_to_traverse = [file[:-6] for file in os.listdir(self.tradb_path) if file.endswith('.tradb')]
        
        #if-condition to check, whether all waveforms or just every $sample_interval$-th event shall be loaded 
        if sample_interval is None: 
            for file in files_to_traverse:
                try:
                    wirebreaks_from_tradb = self.read_signals_from_tradb(file)
                except:
                    continue
                try:
                    wirebreak_signals=pd.concat([wirebreak_signals,wirebreaks_from_tradb],axis=0)
                except:
                    #Copy wirebreaks_from_tradb on first call
                    wirebreak_signals = wirebreaks_from_tradb.copy()
            wirebreak_signals.fillna(0,inplace=True)
        else:
            for file in files_to_traverse:
                #Determine trais to extract and skip if pridb-database is damaged
                hit_parameters = self.load_hit_parameters_from_pridb(file)
                if hit_parameters is None: #Skip if pridb is damaged and None is returned from load-function
                    continue
                #Use channel one to determine number of independent events
                n_events = hit_parameters.loc[1].shape[0]
                time_indices_to_read = hit_parameters.loc[1,'time'].iloc[[idx for idx in range(sample_interval//2,n_events,sample_interval)]].to_list()
                trais_and_channels = []
                for time_idx in time_indices_to_read:
                    trais_and_channels+=hit_parameters[hit_parameters['time'] == time_idx].index.to_list()
                #Load waveforms of specific trais
                wirebreaks_from_tradb = self.read_signals_from_tradb(file, index_with_trais=trais_and_channels)
                try:
                    wirebreak_signals=pd.concat([wirebreak_signals,wirebreaks_from_tradb],axis=0)
                except:
                    #Copy wirebreaks_from_tradb on first call
                    wirebreak_signals = wirebreaks_from_tradb.copy()
            wirebreak_signals.fillna(0,inplace=True)
        return wirebreak_signals
    
    def read_signals_from_tradb(self, filename, index_with_trais = None):
        if self.wirebreak_infos is not None:
            return self.__read_signals_from_tradb_based_on_excel_summary(filename)
        elif index_with_trais is not None:
            return self.__read_specific_trais_from_tradb_using_sensor_config(filename, index_with_trais)
        else:
            return self.__read_all_trais_from_tradb_using_sensor_config(filename)

    def __read_specific_trais_from_tradb_using_sensor_config(self, filename, index_with_trais):
        multi_index = pd.MultiIndex(levels=[[],[],[]],codes=[[],[],[]],names=['File','Channel','TRAI'])
        tmp_breaksignal = pd.DataFrame(index=multi_index)
        tradata = vae.io.TraDatabase(os.path.join(self.tradb_path, filename+'.tradb'))
        for idx in index_with_trais:
            current_index=pd.MultiIndex(levels=[[filename],[idx[0]],[idx[1]]], codes=[[0],[0],[0]], names=['File','Channel','TRAI'])
            wav, t = tradata.read_wave(idx[1])
            amplified_signal = self.amplify_signal(wav, 'Ch'+str(idx[0])).reshape(1,-1)
            amplified_signal = pd.DataFrame(amplified_signal,columns=np.arange(0,amplified_signal.shape[1]),index=current_index)
            tmp_breaksignal=pd.concat([tmp_breaksignal,amplified_signal],axis=0)
        tmp_breaksignal.fillna(0,inplace=True)
        return tmp_breaksignal
                
    def __read_all_trais_from_tradb_using_sensor_config(self, filename):
        multi_index = pd.MultiIndex(levels=[[],[],[]],codes=[[],[],[]],names=['File','Channel','TRAI'])
        tmp_breaksignal = pd.DataFrame(index=multi_index)
        tradata = vae.io.TraDatabase(os.path.join(self.tradb_path, filename+'.tradb'))
        transient_data_matrix = tradata.read()
        for trai in transient_data_matrix.index:
            channel = transient_data_matrix.loc[trai,'channel']
            current_index=pd.MultiIndex(levels=[[filename],[channel],[trai]], codes=[[0],[0],[0]], names=['File','Channel','TRAI'])
            wav, t = tradata.read_wave(trai)
            amplified_signal = self.amplify_signal(wav, 'Ch'+str(channel)).reshape(1,-1)
            amplified_signal = pd.DataFrame(amplified_signal,columns=np.arange(0,amplified_signal.shape[1]),index=current_index)
            tmp_breaksignal=pd.concat([tmp_breaksignal,amplified_signal],axis=0)
        tmp_breaksignal.fillna(0,inplace=True)
        return tmp_breaksignal
    
    def __read_signals_from_tradb_based_on_excel_summary(self, filename):
        multi_index = pd.MultiIndex(levels=[[],[],[]],codes=[[],[],[]],names=['File','Channel','TRAI'])
        tmp_breaksignal = pd.DataFrame(index=multi_index)
        tradata = vae.io.TraDatabase(os.path.join(self.tradb_path, filename+'.tradb'))
        for idx in self.wirebreak_infos[self.wirebreak_infos['Filename']==filename].index:
            current_index=pd.MultiIndex(levels=[[filename],[idx[0]],[idx[1]]], codes=[[0],[0],[0]], names=['File','Channel','TRAI'])
            wav, t = tradata.read_wave(idx[1])
            amplified_signal = self.amplify_signal(wav, idx).reshape(1,-1)
            amplified_signal = pd.DataFrame(amplified_signal,columns=np.arange(0,amplified_signal.shape[1]),index=current_index)
            tmp_breaksignal=pd.concat([tmp_breaksignal,amplified_signal],axis=0)
        tmp_breaksignal.fillna(0,inplace=True)
        return tmp_breaksignal
            
    def amplify_signal(self, signal, sensorinfo):
        if self.wirebreak_infos is not None:
            return self.__amplify_signal_based_on_excel_summary(signal, sensorinfo)
        else:
            return self.__amplify_signal_based_on_sensor_config(signal, sensorinfo)
    
    def __amplify_signal_based_on_sensor_config(self, signal, sensorinfo_idx):
        sys_gain = self.sensor_calibration[sensorinfo_idx]['sys_gain [dB]']
        calc_gain = self.sensor_calibration[sensorinfo_idx]['calc_gain [dB]']
        signal *= 10**((-sys_gain+calc_gain)/20)
        return signal
    
    def __amplify_signal_based_on_excel_summary(self, signal, sensorinfo_idx):
        sys_gain = self.wirebreak_infos.loc[sensorinfo_idx,'sys_gain [dB]']
        calc_gain = self.wirebreak_infos.loc[sensorinfo_idx,'calc_gain [dB]']
        signal *= 10**((-sys_gain+calc_gain)/20)
        return signal
        
    def downsample_signal(self, rawsignals: pd.DataFrame, downsampling_factors: list):
        overall_downsampling_factor=1
        for factor in downsampling_factors:
            overall_downsampling_factor*=factor
            if factor > 10:
                raise ValueError('Downsampling factor should be an Integer!')
        downsampled_signal = pd.DataFrame(np.zeros(shape=(len(rawsignals.index),rawsignals.shape[1]//overall_downsampling_factor)), index = rawsignals.index, columns=np.arange(0,rawsignals.shape[1]//overall_downsampling_factor))
        for idx in rawsignals.index:
            tmp_downsampled = np.array(rawsignals.loc[idx,:])
            tmp_downsampled=tmp_downsampled[tmp_downsampled!=0]
            for q in downsampling_factors:
                tmp_downsampled = scipy.signal.decimate(tmp_downsampled, q=q)
            downsampled_signal.loc[idx,:] = np.hstack((tmp_downsampled, np.zeros(downsampled_signal.shape[1]-tmp_downsampled.shape[0]))).reshape(1,-1)
        return downsampled_signal
    

if __name__=='__main__':

    # =============================================================================
    # Load wirebreaks from witzenhausen bridge girder based on excel file
    # =============================================================================
    pridb_witzenhausen = '/data/damage_detection/2020_KISWind/01_Messdaten/2021_WEA-Deining_Betriebsmessungen/tmp/measurement_data/pridb'
    tradb_witzenhausen = '/data/damage_detection/2020_KISWind/01_Messdaten/2021_WEA-Deining_Betriebsmessungen/tmp/measurement_data/tradb'
    excelsummary_witzenhausen = '/home/aljamilm/Schreibtisch/Superposition_Wirebreak/summary_witzenhausen.xlsx'
    witzenhausen_processor = Vallen_AEProcessor(pridb_witzenhausen, tradb_witzenhausen, excel_summary = excelsummary_witzenhausen)
    
    #load pridb stats and raw wirebreak signals
    witzenhausen_features = witzenhausen_processor.get_wirebreak_features()
    witzenhausen_rawsignals = witzenhausen_processor.get_wirebreak_signals()
    witzenhausen_downsampled = witzenhausen_processor.downsample_signal(witzenhausen_rawsignals, downsampling_factors=[5,4])

    # =============================================================================
    # Load Operational AE features for Deining wind turbine (One day: 16.06.2022)
    # =============================================================================
    pridb_path_deining = '/data/damage_detection/2020_KISWind/01_Messdaten/2021_WEA-Deining_Betriebsmessungen/pridb/2021_Juni/2021-06-16/'
    tradb_path_deining = '/data/damage_detection/2020_KISWind/01_Messdaten/2021_WEA-Deining_Betriebsmessungen/tradb/2021_Juni/2021-06-16'
    sensors_deining = '/home/lange/Dokumente/CodeProjects/KISWind/wirebreaks_in_wind_turbine/sensorcalibration_Deining2021.json'
    deining_processor = Vallen_AEProcessor(pridb_path = pridb_path_deining, tradb_path=tradb_path_deining, sensor_config_file=sensors_deining)
    
    #Load pridb features
    deining_features = deining_processor.get_wirebreak_features()
    
    
    # =============================================================================
    # Load wirebreaks from laboratory experiments at TUD (One day: 16.06.2022)
    # =============================================================================
    pridb_TUD_wirebreaks_1 = '/data/damage_detection/2020_KISWind/01_Messdaten/2022_Spanndrahtbrüche_Versuchsstand_TUD/2022-09-08_Drahtbruch#1-#3/Vallen-Messsystem/pridb/'
    tradb_TUD_wirebreaks_1 = '/data/damage_detection/2020_KISWind/01_Messdaten/2022_Spanndrahtbrüche_Versuchsstand_TUD/2022-09-08_Drahtbruch#1-#3/Vallen-Messsystem/tradb/'
    sensors_TUD = '/home/lange/Dokumente/CodeProjects/KISWind/wirebreaks_in_wind_turbine/sensorcalibration_TUD.json'
    wirebreaks_TUD_processor = Vallen_AEProcessor(pridb_path = pridb_TUD_wirebreaks_1, tradb_path=tradb_TUD_wirebreaks_1, sensor_config_file=sensors_TUD)
    
    #Load wirebreaks
    TUD_wirebreaks = wirebreaks_TUD_processor.get_wirebreak_signals()
