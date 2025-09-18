#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filtermodul
"""

import scipy

class Signalfilter(object):
    
    def __init__(self, filterorder, fsampling, cutoff_highfreq, cutoff_lowfreq=None, filtertype='highpass', filterart = 'butterworth', filteraxis=0):
        self.__filteraxis=filteraxis
        self.__fs=fsampling
        self.__filterart = filterart
        if self.__filterart =='butterworth':
            self.__filter_coeff = self.__get_filtercoefficients_butterworth(filterorder, cutoff_highfreq, cutoff_lowfreq, filtertype)
        elif self.__filterart == 'linearFIR':
            self.__filter_coeff = self.__get_filtercoefficients_FIR(filterorder, cutoff_highfreq, filtertype)
        else:
            raise ValueError('Filterart not known')
        
    def __get_filtercoefficients_butterworth(self, filterorder, cutoff_highfreq, cutoff_lowfreq, filtertype):
        if filtertype=='bandpass':    
            sos = scipy.signal.butter(filterorder, [cutoff_lowfreq, cutoff_highfreq], filtertype, fs=self.__fs, output='sos')
        elif cutoff_lowfreq==None:
            sos = scipy.signal.butter(filterorder, cutoff_highfreq, filtertype, fs=self.__fs, output='sos')
        else:
            raise Exception('cutoff_lowfreq can only be specified, when Bandpassfilter is applied!')
        return sos
    
    def __get_filtercoefficients_FIR(self, filterorder, cutoff_highfreq, filtertype):
        if filtertype=='lowpass':
            fir_coeff = scipy.signal.firwin(filterorder, cutoff=cutoff_highfreq, fs = self.__fs)
        elif filtertype=='highpass':
            fir_coeff = scipy.signal.firwin(filterorder, cutoff=cutoff_highfreq, fs = self.__fs, pass_zero=False)
        else:
            raise ValueError('Wrong Filtertype specified!')
        return fir_coeff
    
    def applyfilter(self, rawsignal_datamatrix):
        if self.__filterart == 'butterworth':
            return self.__applyfilter_butterworth(rawsignal_datamatrix)
        elif self.__filterart == 'linearFIR':
            return self.__applyfilter_FIR(rawsignal_datamatrix)

    def __applyfilter_butterworth(self, rawsignal_datamatrix):
        datatemp_filt = scipy.signal.sosfiltfilt(self.__filter_coeff, rawsignal_datamatrix, axis=self.__filteraxis)
        return datatemp_filt
    
    def __applyfilter_FIR(self, rawsignal_datamatrix):
        datatemp_filt = rawsignal_datamatrix.copy()
        if self.__filteraxis==0:
            for col in range(0,rawsignal_datamatrix.shape[1]):
                datatemp_filt[:,col] = scipy.signal.fftconvolve(rawsignal_datamatrix[:,col], self.__filter_coeff, mode='same')
        elif self.__filteraxis==1:
            for row in range(0,rawsignal_datamatrix.shape[0]):
                datatemp_filt[row,:] = scipy.signal.fftconvolve(rawsignal_datamatrix[:,col], self.__filter_coeff, mode='same')
        return datatemp_filt

    def plot_freq_response(self):
        fig, ax = plt.subplots(2)
        w, h = scipy.signal.freqz(self.__filter_coeff, worN=2048, fs = self.__fs)
        db = 20*np.log10(np.maximum(np.abs(h),1E-5))
        ax[0].plot(w, db)
        ax[0].set_xlabel('Frequency [Hz]')
        ax[0].set_ylabel('Gain [dB]')
        ax[1].plot(w, np.unwrap(np.angle(h)))
        ax[1].set_xlabel('Frequency [Hz]')
        ax[1].set_ylabel('Angle [radian]')
    