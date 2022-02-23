# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 14:18:56 2021

@author: asligar
"""

from Lib.Read_Frtm import read_frtm
import os
import Lib.Post_Process_Utils as utils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
plt.close('all')

def get_script_path():
    return os.path.dirname(os.path.realpath(__file__))


path = 'C:\\Users\\asligar\\OneDrive - ANSYS, Inc\\Desktop\\delete\\Pedestrian_and_Vehicle.aedtresults\\design1.results\\'


results_files = utils.get_results_files(path)
num_results = len(results_files)
t_start = 0
t_stop = 1
t_sweep= np.linspace(t_start,t_stop,num=num_results)

rPixels =400
dPixels = 300


#read just the header of one file to get simulation parameters like num frequency points
#assumes data is the same for all files in the directory
data= read_frtm(results_files[0])

nfreq = data.nfreq
fc = data.freq_center
ntime = data.ntime

r_period = data.range_period()
r_resolution = data.range_resolution()
v_period = data.velocity_period()
v_resolution = data.velocity_resolution()
cpi_time = data.time_duration
prf = 1/data.time_delta
vmin = -v_period/2
vmax = v_period/2
rmax = r_period
fov = [-90,90]

channel_names = data.channel_names
num_channels = data.num_channels
#order is if we want first index to be freq or pulse, post processing here assumes [freq][pulse] order
single_frame = data.load_data(order='FreqPulse') 

#load all the data into a single array
#probably a more efficient way to do this, but good enough for now
four_d_radar_data_cube = np.zeros((num_results,num_channels,nfreq,ntime),dtype='complex')
for n, frtm_file in enumerate(results_files):
    data= read_frtm(frtm_file)
    for m, channel in enumerate(channel_names):
        four_d_radar_data_cube[n][m]= data.load_data(order='FreqPulse')[channel]
        

test_pd = four_d_radar_data_cube[0][0]
test_pd_ang = np.angle(test_pd)
test_fmcw = np.conjugate(four_d_radar_data_cube[1][0])
test_fmcw_ang = np.angle(test_fmcw)
#################
# Create range doppler plot for the first channel

data_all_channels_rd = np.zeros((num_channels,rPixels,dPixels),dtype=complex)
for n, ch in enumerate(channel_names):
    data_all_channels_rd[n], range_profile, processing_fps =utils.range_doppler_map(single_frame[ch], window=False,size=(rPixels,dPixels))

#range doppler for a single channel
range_doppler = data_all_channels_rd[0]

max_rd = np.max(20*np.log10(np.abs(range_doppler)))
dynamic_range=100


plt.figure(1)
rd_to_plot1 = 20*np.log10(np.abs(range_doppler))
vel_vals= np.linspace(vmin,vmax,num=dPixels)
range_vals = np.linspace(0,rmax,num=rPixels)

#normalize plot to maximum value so values stay constant for animation
max_of_plot1 = max_rd
min_of_plot1 = max_rd-dynamic_range
levels = np.linspace(min_of_plot1,max_of_plot1,64)

normi = colors.Normalize(vmin=min_of_plot1, vmax=max_of_plot1)

x, y = np.meshgrid(vel_vals,range_vals)

colbar_PSD = plt.contourf(x,y,rd_to_plot1,64,cmap='rainbow',
       levels=levels, norm=normi, extend='both')
plt.ylabel('Range')
plt.xlabel('Doppler')
plt.title('Doppler Range')
v_min = np.min(x)
plt.show()



