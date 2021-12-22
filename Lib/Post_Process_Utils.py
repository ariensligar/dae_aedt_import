# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 09:01:31 2020

@author: asligar
"""


import numpy as np
import numpy.matlib as npm
from numpy.fft import fft,fftshift,ifft,ifftshift,fft2
import glob
import time as walltime
from copy import deepcopy

def get_results_files(path,wildcard=''):
    """
    wildcard is if we want to seperate different results folder
    different solution setups would be named something like
    DV551_S17_V518_Data.transient
    where the wild card could be "s17_V518" to indicate that specific setup
    """
    results_files=[]
    all_paths = glob.glob(path + '\\*' + wildcard + '_Data.transient')
    index_num = []
    for filename in all_paths:
        index_num.append( int(filename.split('\\DV')[1].split('_')[0]))
    
    all_paths_sorted = sorted(zip(index_num,all_paths))
    #all_paths = sorted(all_paths)
    for each in all_paths_sorted:
        results_files.append(each[1]+'\\RxSignal.frtm')
    return results_files

def range_profile(data, window=False,size=1024):
    """
    range profile calculation
    
    input: 1D array [freq_samples]
        
    returns: 1D array in original_lenth*upsample
        
    """

    nfreq = int(np.shape(np.squeeze(data))[0])
    #scale factors used for windowing function
    if window:
        win_range = np.hanning(nfreq)
        win_range_sum= np.sum(win_range)
        sf_rng = nfreq/(win_range_sum)
        win_range = win_range*sf_rng
        pulse_f = np.multiply(data,win_range) #apply windowing
    else:
        pulse_f = data
        

    sf_upsample = size/nfreq
    #should probaby upsample to closest power of 2 for faster processing, but not going to for now
    pulse_t_win_up = (sf_upsample*np.fft.ifft(pulse_f,n=size))
    

    return pulse_t_win_up

def convert_freqpulse_to_rangepulse(data,output_size = 256,pulse=None):
    '''
    input: 3D array [channel][freq_samples][pulses], size is desired output in (ndoppler,nrange)
            output_size is up/down samping in range dimensions
            pulse=None, this is the pulse to use, if set to none it will extract from center pulse
    returns: 3D array in [channel][range]
    '''
    

    rPixels = output_size
    #input shape
    rng_dims = np.shape(data)[1]
    dop_dims = np.shape(data)[2]
    
    if pulse==None:
        pulse = int(dop_dims/2)
    else:
        pulse = int(pulse)
    
    freq_ch = np.swapaxes(data,0,2)
    freq_ch = freq_ch[pulse] #only extract this pulse
    ch_freq = np.swapaxes(freq_ch,0,1)
    
    #window
    h_rng = np.hanning(rng_dims)
    sf_rng = len(h_rng)/np.sum(h_rng)
    sf_upsample_rng = rPixels/rng_dims
    h_rng = h_rng*sf_rng
    
    #apply windowing
    ch_freq_win =sf_upsample_rng* np.multiply(ch_freq,h_rng)
    
    #take fft
    ch_rng_win =np.fft.ifft(ch_freq_win,n=rPixels) #[ch][range][dop]fft across dop dimenions
    ch_rng_win = np.fliplr(ch_rng_win)
    
    return ch_rng_win



def range_doppler_map(data, window=False,size=(256,256)):
    """
    range doppler calculation
    
    input: 2D array [freq_samples][pulses], size is desired output in (ndoppler,nrange)
        
    returns: 2D array in [range][doppler]
        
    """
    
    time_before = walltime.time()
    #I think something is wrong with data being returned as opposte, freq and pulse are swaped
    nfreq = int(np.shape(data)[0])
    ntime= int(np.shape(data)[1])
    
    rPixels = size[0]
    dPixels = size[1]
    
    h_dop = np.hanning(ntime)
    sf_dop = len(h_dop)/np.sum(h_dop)
    sf_upsample_dop = dPixels/ntime
    
    h_rng = np.hanning(nfreq)
    sf_rng = len(h_rng)/np.sum(h_rng)
    sf_upsample_rng = rPixels/nfreq
    
    
    h_dop  = h_dop*sf_rng
    h_rng = h_rng*sf_dop
    
    
    fp_win =sf_upsample_dop* np.multiply(data,h_dop)
    s1 = np.fft.ifft(fp_win,n=dPixels)
    s1 = np.rot90(s1)

    s1_win = sf_upsample_rng*np.multiply(h_rng,s1)
    s2 = np.fft.ifft(s1_win,n=rPixels)
    s2 = np.rot90(s2)
    s2_shift = np.fft.fftshift(s2,axes=1)
    #range_doppler = np.flipud(s2_shift)
    range_doppler = np.flipud(s2_shift)
    #range_doppler=s2_shift
    time_after = walltime.time()
    duration_time = time_after-time_before
    if duration_time==0:
        duration_time=1
    duration_fps = 1/duration_time

    rp=0
    return range_doppler, rp, duration_fps




