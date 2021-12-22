# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 08:41:28 2020

@author: asligar
"""

import numpy as np


class read_frtm(object):
    """
    reads frtm data output in .transient file
    some of the loops are hardcoded, in the future I will read them
    """
    def __init__(self,filepath):
        """
        reads frtm file from path
        """

        string_to_stop_reading_header = '@ BeginData'
        header = []    
        with open(filepath, "rb") as binary_file:
            line = binary_file.readline()
            line_str = line.decode('ascii')
            while string_to_stop_reading_header not in line_str:
                header.append(line_str)
                line = binary_file.readline()
                line_str = line.decode('ascii')
                if line_str.replace(" ","") =="":
                    pass
                elif 'DlxCdVersion' in line_str:
                    dlxcd_vers_line = line_str
                    vers = dlxcd_vers_line.split("=")
                    self.dlxcd_vers = vers
                elif '@ RowCount' in line_str:
                    c = line_str.split("=")
                    c = c[1].replace("\n","").replace("\"","").replace(" ","")
                    self.row_count =  int(c)
                elif '@ ColumnCount' in line_str:
                    c = line_str.split("=")
                    c = c[1].replace("\n","").replace("\"","").replace(" ","")
                    self.col_count =  int(c)
                elif '@ BinaryRecordLength ' in line_str:
                    bin_record_length_line = line_str  
                    c = bin_record_length_line.split("=")
                    self.bin_record_lenth = c[1]
                elif '@ BinaryStartByte ' in line_str:
                    c = line_str.split("=")
                    c = c[1].replace("\n","").replace("\"","").replace(" ","")
                    self.binary_start_byte = int(c)
                elif '@ BinaryRecordSchema ' in line_str:
                    self.bin_byte_type_line = line_str
                elif '@ RadarWaveform ' in line_str:
                    radarwaveform_line = line_str
                    rw = radarwaveform_line.split("=")
                    self.radar_waveform = rw[1].replace("\n","").replace("\"","").replace(" ","" )
                elif '@ RadarChannels ' in line_str:
                    radarchannels_line = line_str
                    rc = radarchannels_line.split("=")
                    self.radarchannels =  rc[1].replace("\n","").replace("\"","").replace(" ","" )
                elif '@ TimeSteps ' in line_str:
                    time_steps_line = line_str
                    c = time_steps_line.split("=")
                    c = c[1].split(" ")
                    c = [i for i in c if i] 
                    self.time_start= float(c[0].replace("\"",""))
                    self.time_stop= float(c[1].replace("\"",""))  
                    c = time_steps_line.split("=")
                    c = c[1].split(" ")
                    c = [i for i in c if i] 
                    self.ntime= int(c[2].replace("\"",""))+1
                    self.time_sweep = np.linspace(self.time_start,self.time_stop,num=self.ntime)
                    self.time_delta= self.time_sweep[1]-self.time_sweep[0]
                    self.time_duration =  self.time_sweep[-1]-self.time_sweep[0]
                elif '@ FreqDomainType ' in line_str:
                    freq_dom_type_line = line_str
                    self.freq_dom_type = freq_dom_type_line.split("=")[1]
                elif '@ FreqSweep ' in line_str:
                    freq_sweep_line = line_str  
                    c = freq_sweep_line.split("=")
                    c = c[1].split(" ")
                    c = [i for i in c if i] 
                    self.freq_start =  float(c[0].replace("\"",""))
                    self.freq_stop = float(c[1].replace("\"","")) 
                    self.nfreq = int(c[2].replace("\"",""))+1
                    self.freq_sweep = np.linspace(self.freq_start,self.freq_stop,num=self.nfreq)
                    self.freq_delta = self.freq_sweep[1]-self.freq_sweep[0]
                    self.freq_bandwidth = self.freq_sweep[-1]-self.freq_sweep[0]
                    center_index = int(self.nfreq/2)
                    self.freq_center = float(self.freq_sweep[center_index])
                elif '@ AntennaNames ' in line_str:
                    ant_names_line = line_str
                    c = ant_names_line.split("=")
                    c = c[1].replace("\n","")
                    c = c.replace("\"","").replace(" ","" )
                    an = c.split(';')
                    self.antenna_names = an
                elif '@ CouplingCombos '  in line_str:
                    coupling_combos_line = line_str
                    c = coupling_combos_line.replace("\"","")
                    c = c.replace("\n","")
                    c = c.split("=")[1]
                    c = c.split(" ")
                    c = [i for i in c if i]
                    self.num_channels = int(c[0])
                    self.coupling_combos = c[1].split(";")
            self.filepath = filepath
        
        #this is the order in the frtm file
            
        self.channel_names = []
        for each in self.coupling_combos:
            index_values = each.split(',')
            rx_idx =index_values[0]
            tx_idx =index_values[1]
            if ":" in rx_idx:
                rx_idx = int(rx_idx.split(':')[0])
            if ":" in tx_idx:
                tx_idx = int(tx_idx.split(':')[0])
            tx_idx = int(tx_idx)-1
            rx_idx = int(rx_idx)-1
            self.channel_names.append(self.antenna_names[rx_idx] + ":"+ self.antenna_names[tx_idx] )
    
        dt = np.dtype([('ScatSgnlReal', float), ('ScatSgnlImag', float)])
    
        raw_data = np.fromfile(filepath, dtype=dt,offset =self.binary_start_byte )
        

        
        cdat_real = np.reshape(raw_data['ScatSgnlReal'],(self.num_channels,self.ntime,self.nfreq))
        #cdat_real = np.moveaxis(cdat_real,-1,0)
        cdat_imag = np.reshape(raw_data['ScatSgnlImag'],(self.num_channels,self.ntime,self.nfreq))
        #cdat_imag = np.moveaxis(cdat_imag,-1,0)
        self.all_data = {}
        for n, ch in enumerate(self.channel_names):
            self.all_data[ch] = cdat_real[n]+cdat_imag[n]*1j



    def range_period(self):
        rr = self.range_resolution()
        max_range = rr*self.nfreq
        return max_range
    def range_resolution(self):
        bw = self.freq_bandwidth
        rr = 299792458.0/2/bw
        return rr
    def velocity_resolution(self):
        fc = self.freq_center
        tpt= self.time_duration
        vr = 299792458.0/(2*fc*tpt)
        return vr
    def velocity_period(self):
        vr = self.velocity_resolution()
        np = self.ntime
        vp = np*vr
        return vp

    def load_data(self,order='FreqPulse'):
        '''

        Parameters
        ----------
        order : TYPE, optional
            DESCRIPTION. order is either 'FreqPulse' or 'PulseFreq'. this is
            the index order of the array, [numFreq][numPulse] or [numPulse][numFreq] 
            many of the post processing depend on this order so just choose accordingly

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        if order.lower()=='freqpulse':
            for ch in self.all_data.keys():
                self.all_data[ch] = self.all_data[ch].T
            return self.all_data
        else: #this is how the data is read in with [pulse][freq] order
            return self.all_data