# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 15:30:27 2020

@author: alexa
"""

# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as pl
import pandas as pd
#import scipy
import scipy.signal as sgl
from scipy.optimize import curve_fit, minimize
#import binascii
#import StreamViewerFilter_CLASS
from numpy import pi, sin, cos, exp, sqrt

''' NOTE: This code was written by Alex Shilcusky in 2019-20 for Scott Hertel in order to build an optimal filter (OF)
for DM data obtained from the Edelweiss group in Lyon, France. I wrote this code in anticipation for applying an OF 
to our own DM data. If you intend to use this code as a guide for applying an OF to your own data, I highly 
recommend applying and bastardizing these functions one at a time in a new program so you can understand what
they are doing and how they relate to your own needs. '''


#----------------------------------------------------------------------------------------------------------------
font =11.5
pl.rcParams.update({'font.size': font})
pl.rcParams['axes.grid'] = True
pl.rcParams['grid.linewidth'] = 0.5
pl.rc('figure' , figsize=(10,6))

data_offset = 7077
voltages = np.memmap("LyonData_se26g000_000", dtype='int16', mode='r', offset=data_offset, shape=(1000000,1))
fe = 400 # sampling rate (Hz)

ti, tf = 0, 46000
ni, nf = ti*fe, tf*fe

time = np.arange(len(voltages))/fe
time = time[int(ni):int(nf)]
V = voltages[int(ni):int(nf)]
V = V[:,0] # THIS LINE IS ESSENTIAL - DO NOT REMOVE 
#----------------------------------------------------------------------------------------------------------------

def high_pass_filter(timeseries, cutoff_freq, order, inverse=False):
        '''
        Butterworth High Pass Filter
        filter size in Hz
        '''
        b, a = sgl.butter(order, 2*cutoff_freq/fe ,btype="highpass")
        if inverse == False:
            z = sgl.lfilter(b,a,timeseries)
#            z = sgl.filtfilt(a,b,timeseries)
        else:            
            z = sgl.lfilter(a,b,timeseries)
#            z = sgl.filtfilt(a,b,timeseries)
        return z
    
def low_pass_filter(timeseries, filtersize, order, inverse=False):
        '''
        Butterworth Low Pass Filter
        filter size in Hz
        '''
        
        b, a =sgl.butter(order, 2*filtersize/fe ,btype="lowpass")
        if inverse == False:
            z=sgl.lfilter(b,a,timeseries)
        else:
            z=sgl.lfilter(a,b,timeseries) 
        return z
    
#----------------------------------------------------------------------------------------------------------------
order = 2
fc = 2 # cutoff frequency [Hz]
V_hpf = high_pass_filter(V,fc,order=order)
'Plot the frequency fresponse of the High-Pass Filter'
b, a = sgl.butter(order, 2*fc/fe ,btype="highpass")
fw, response = sgl.freqz(b,a,worN=201, fs=fe)
ones = np.ones(len(fw))
rt = sqrt(0.5)*ones
pl.figure()
pl.ylabel('gain')
pl.plot(fw,abs(response))
pl.plot(fw, rt,'--', label='sqrt(0.5)')
#pl.xlim(0,10)
pl.xlabel('frequency [Hz]')
pl.xscale('log')
pl.legend()
pl.show()


t_0 = .5 # time within window pulse starts at
PULSE_WINDOW = 1.0
#----------------------------------------------------------------------------------------------------------------

def step(x):
    return 1 * (x > 0) 

def expo_temp1(t, A, T1, T2, T3):
    '''Julien suggests three term exponential function because of thermal decay time
    as well as pulse decay time (???)
    Capital letters are constants'''
    return A*(-np.exp(-t/T1) + np.exp(-t/T2) + np.exp(-t/T3))
    
def expo_temp2(t,a,b,c):
    return (-a*t*np.exp(-t**2/b - t/c))

def theta_exp_pulse(t, A, T1,T2,T3):
    shifted_t = np.asarray(t) - t_0    
    m = expo_temp1(shifted_t, A,T1,T2,T3)
    return step(shifted_t)*m 

# begin def getNoisePSD
#******************************************************************************************************************
'''
                                                                                                      getNoisePSD
'''
def getNoisePSD(data,fe):
    stream_RAW = data
    stream_HPF = high_pass_filter(data,fc,order=order)
    total_timesteps = len(stream_HPF) # total time * frequency
    stream_end_time = total_timesteps/fe # time
    sample_fraction = 0.5
    sample_size = int(sample_fraction*stream_end_time/PULSE_WINDOW) # number of windows to sample
    
    np.random.seed(1996)
    start_times = (stream_end_time - 4.0*PULSE_WINDOW)*np.random.sample(sample_size) + 2.0*PULSE_WINDOW
    start_times = np.sort(start_times)
    
    start_indices = fe * start_times # indices of sample starttimes
    start_indices = (start_indices.astype(int))
    
    end_indices = start_indices + int(fe*PULSE_WINDOW)
    
    hpf_PSD_list = []
    raw_PSD_list = []
    HPF_windows = []
    RAW_windows = []
    rms_amplitudes = []
    time_windows = []
    
    for i in range(len(start_indices)):
        HPF_window = stream_HPF[start_indices[i]:end_indices[i]]
        RAW_window = stream_RAW[start_indices[i]:end_indices[i]]
        
        time_window = time[start_indices[i]:end_indices[i]]
        time_windows.append(time_window)
        
        f, hpf_PSD = sgl.periodogram(HPF_window, fe) 
        f, raw_PSD = sgl.periodogram(RAW_window,fe)
        
        HPF_rms = np.sqrt(np.mean((HPF_window**2)))
        
        rms_amplitudes.append(HPF_rms)
        HPF_windows.append(HPF_window)
        RAW_windows.append(RAW_window)
        hpf_PSD_list.append(hpf_PSD)
        raw_PSD_list.append(raw_PSD)
        
       
    #    if i%10 == 0:
        if 0:
            pl.figure(figsize=(12,4))
            pl.subplot(121)
            pl.plot(time_window, HPF_window)
            pl.xlabel('Time [s]')
            pl.ylabel('ADU')
            pl.title('HPF time trace')
            pl.subplot(122)
            pl.title('hpf PSD')
            pl.loglog(f,hpf_PSD)
            pl.xlabel('Frequency [Hz]')
            pl.show()
            
            pl.figure(figsize=(12,4))
            pl.subplot(121)
            pl.title('RAW time trace')
            pl.plot(time_window, RAW_window)
            pl.xlabel('Time [s]')
            pl.ylabel('ADU')
            pl.subplot(122)
            pl.title('raw PSD')
            pl.loglog(f,raw_PSD)
            pl.xlabel('Frequency [Hz]')
            pl.show()
            
            print('HPF_rms=',HPF_rms)
    
    noise_dataframe = pd.DataFrame(data={'hpf_PSD_list': hpf_PSD_list,
                                        'time': time_windows,
                                        'HPF Windows': HPF_windows,
                                        'Raw Windows': RAW_windows,
                                        'RMS_amplitude': rms_amplitudes,
                                        'hpf_PSD_list': hpf_PSD_list,
                                        'raw_PSD_list': raw_PSD_list
                                        })
        
    rms_threshold = 50
    noise_dataframe = noise_dataframe[noise_dataframe.RMS_amplitude < rms_threshold] # initial RMS cut
    
    noise_PSD_temp = noise_dataframe['hpf_PSD_list'].mean()
    
    noise_dataframe['chisq_HPF'] = noise_dataframe.apply(
                    lambda row: sum( row.hpf_PSD_list[1:]/noise_PSD_temp[1:] ), axis=1)
    noise_dataframe['chisq_RAW'] = noise_dataframe.apply(
                    lambda row: sum( row.raw_PSD_list[1:]/noise_PSD_temp[1:] ), axis=1)
    
    chi_RAW = noise_dataframe['chisq_RAW']
    chi_HPF = noise_dataframe['chisq_HPF']
    chiraw_mean = chi_RAW.mean()
    chihpf_mean = chi_HPF.mean()
    
    i = 0
    pl.figure()
    while (chihpf_mean != 200.0):
#    while i < 10:
#        noise_dataframe = \
#                    noise_dataframe[noise_dataframe.chisq_RAW < max(noise_dataframe["chisq_RAW"])]
        noise_dataframe = \
                    noise_dataframe[noise_dataframe.chisq_HPF < max(noise_dataframe["chisq_HPF"])]
        
        noise_PSD_temp = noise_dataframe['hpf_PSD_list'].mean()
        pl.loglog(noise_PSD_temp/abs(response),label='%i samples'%len(noise_dataframe['hpf_PSD_list']))
        i += 1
    #    print('Length of hpf_PSD_list:',len(noise_dataframe['hpf_PSD_list']))
        noise_dataframe['chisq_HPF'] = noise_dataframe.apply(
                    lambda row: sum( row.hpf_PSD_list[1:]/noise_PSD_temp[1:] ), axis=1)
        noise_dataframe['chisq_RAW'] = noise_dataframe.apply(
                    lambda row: sum( row.raw_PSD_list[1:]/noise_PSD_temp[1:] ), axis=1)
        
#        chiraw_mean = noise_dataframe['chisq_RAW'].mean()
        chihpf_mean = noise_dataframe['chisq_HPF'].mean()
#        print('Raw chi mean =',chiraw_mean)
        print('LENGTH OF PSD_list',len(noise_dataframe['hpf_PSD_list']))
        print('HPF chi mean =',chihpf_mean)
    #    print('Sum of Chi_mean =', chiraw_mean+chihpf_mean)
    
    pl.legend()
    pl.xlabel('Frequency [Hz]')
    pl.ylabel(r'Power Spectral $([ADU]^{2} / Hz) ??$')
    pl.ylim(1e-4,1e1)
    pl.show()
    
    return f, noise_PSD_temp

def getWindowedFFT(x, scaling="density", window = "boxcar", removeDC = True, oneSided = True):
    '''
    set scaling to "density" for comparison with LPSD
    returns frequencies, FFT
    '''
    win_vals = sgl.get_window(window, len(x))
    windowed_x = win_vals*x
    
    FTfreq=np.arange(0,len(x))*fe/len(x)
    FTfreq=FTfreq[0:int(1+len(FTfreq)/2)]
    FT=np.fft.rfft(windowed_x)
    
    if scaling == "density":
        S1 = np.sum(win_vals)
        S2 = np.sum(win_vals**2)
        ENBW = fe*S2/(float(S1**2))
        FT = np.sqrt(2)*FT/(S1*np.sqrt(ENBW))

    if removeDC:
        FT[0]=0
            
    return FTfreq, FT

def getInverseFFT(x, scaling="density", window = "boxcar", oneSided = True):
    '''
    unscales: scaling to "density" for comparison with LPSD
    returns time domain
    '''
    win_vals = sgl.get_window(window, 2*len(x))
#    windowed_x = win_vals*x
    inverseFT=np.fft.irfft(x)
    
    if scaling == "density":
        S1 = np.sum(win_vals)
        S2 = np.sum(win_vals**2)
        ENBW = fe*S2/(float(S1**2))
        inverseFT = inverseFT*(S1*np.sqrt(ENBW))

    return inverseFT

#******************************************************************************************************************
'''
                                                                                                      initial_pulse_trigger
'''
def initial_pulse_trigger(V_hpf,fe):
    '''This funciton identifies pulses of interest in order to fit them to an analystical model to 
    find a template function for the OF. This method is specialized to the given data, and uses
    'by eye' techniques to determine pulses max/min thresholds. YOU HAVE BEEN WARNED'''
    stream = V_hpf
    pulse_windows = []
#    pulse_timewindows = []
#    time = np.arange(len(stream))/fe
    
    i = 0
    while( i < len(stream)-1):
        if( stream[i] > 500 and i-200 > 0): # 6keV pulses exist at ~800 ADU according to Julien
            window = stream[int(i-0.5*fe):int(i+0.5*fe)]
            pulse_windows.append(window)
#            pulse_timewindows.append(time[int(i-0.25*fe):int(i+0.25*fe)])
#            pl.figure()
#            pl.plot(time[int(i-0.25*fe):int(i+0.25*fe)],window)
#            pl.show()
            i = int(i+0.5*fe)
        else:
            i = i+1
            
    pulses = []
    for i in range(len(pulse_windows)):
        window = pulse_windows[i]
        pulse = 1 
        for i in window:
            if i < -500: # this is kind of jank. By inspection we see 800 ADU
                        # pulses don't pass below ~300 ADU whereas all others do
                pulse = 0
        if pulse == 1:
#            pl.figure()
#            pl.plot(window)
#            pl.show()
            pulses.append(window)
    print('NUMBER OF PULSES FOUND: ', len(pulses))
    return pulses

#******************************************************************************************************************
'''
                                                                                                        OFTransferFn
'''
def OFTransferFn(stream,template, J, freq ): # , parameters, template_function): 
    #stream = V_hpf
    '''
    stream should be unfiltered
    Computes the Optimal Filter transfer function based on the supplied template
    Returns the optimally filtered stream, the OF transfer function, and the resolution
    '''
    
    '''
    The transfer function is defined as 
        H(w) = h s(w)exp(-jw*t_max)
    
    
    Step 1: Compute the matched filter transfer fn H(f_i) of length M using the
    signal template s_i and the noise PSD, J, both of length M.
    '''
    time=np.arange(0,len(stream))/fe
    time_window = np.linspace(0, PULSE_WINDOW, int(fe*PULSE_WINDOW) ) 
    # 2/12/2020

    ss = template
#    s = s/max(s)
    
#    s = pulse_average
#    ss = pulse_average
    s = ss/max(ss)
    
    time_window = np.linspace(0, PULSE_WINDOW, int(fe*PULSE_WINDOW) ) 
    
#    s = template_function(time_window, *parameters) # the template function is
                                                    # the expected pulse shape      
#    s = pulse_av/max(pulse_av)               

    s_HPF_timedomain = high_pass_filter(s, fc, order) # apply HPF to kill non-periodic frequencies
    
    
    # Here we obtain the FT/LPSD of the template function. 's_hpf_freqdomain'
    # is the FT of s(t), but with a scaling factor applied such that its 
    # magnitude is the LPSD of s(t)
#    f, s_HPF_freqdomain = getWindowedFFT(s_HPF_timedomain, scaling="density")
    f, s_freqdomain = getWindowedFFT(s, scaling="density")
    
    if 0:
        pl.figure()
        pl.loglog(f,abs(s_freqdomain), label='template fn')
        pl.legend()
        pl.show()

    s_freqdomain[0]=0
    one_over_h =  sqrt(2)*np.sum( (np.abs(s_freqdomain)**2)/J ) 
    print('sum |Si|^2/Ji =',one_over_h)
#    h = (1.0/6.49)*1.0/one_over_h
    h = (1.0)/one_over_h
    
    
    print( "\nResolution after OF (sqrt(h)): " , np.sqrt(h) )
    print( "Resolution before OF: " , np.sqrt(2*np.sum(J)) )
    

#    t_max=2.05*np.pi*t_delay       #   time position of template max
    t_max=1.00*t_0 # desired time position of template max within pulse window
                            # t_0 = 0.25s 
#    print('t_max:',t_max)
    phases = []
    for i in range(len(freq)):
        phases.append(np.complex(0,-freq[i]*2*np.pi*t_max))  
#        phases.append(np.complex(0,-freq[i]*2*np.pi*0.5))  
    phases = np.array(phases)
    print( "type phases", type(phases), type(phases[0]) )
    
    H = h*np.conjugate(s_freqdomain)*np.exp(phases)/J #one sided transfer function

    OF_filtered_template_freqdomain = H*s_freqdomain

        
    print('length of H(f_i) =',len(OF_filtered_template_freqdomain) )
    print('number of freq bins =',len(freq))

    OF_filtered_template_timedomain = getInverseFFT(OF_filtered_template_freqdomain, scaling="density")
    
    '''
    Step 1 is now complete. Our matched filter transfer fn is <<H>>
    '''
    
    '''
    Step 2: Transform H(f_i) into the time domain, obtaining the response
    funciton h_t of length M
    '''

    h_t = np.fft.irfft(H) #impulse response function of length M
    '''
    Step 3: Insert M zeros in the middle of h_t, obtaining the response fn
    h_padded of length 2M
      
    Step 4: Smooth h_padded in proximity of the zeros insertion, obtaining
    smoothed response function h_padded
    
    '''
    if 0: # plot impulse response function 
        pl.figure()
        pl.title('impulse response function')
        pl.plot(abs(h_t)) # 9/26/19
        pl.xlabel('time (s)')
        pl.show()
    
    '''Here we split h_t in half, then insert an array of zeros in the middle
    to avoid DFT Convolution Wraparound.
    Then we multiply the whole padded impulse response function <<h_padded>>
    by a quarter period cos shape to smooth it.'''
    # Here we split h_t in half in order to insert zeros in the middle
    h_tfirsthalf = h_t[:int(len(h_t)/2)] 
    h_tsecondhalf = h_t[int(len(h_t)/2):] 

    
    zeros = np.zeros(len(h_t))
    h_padded = np.concatenate( (h_tfirsthalf, zeros, h_tsecondhalf),0)
    
    if 0: # plot NON-smoothed padded response fn
        pl.figure()
        pl.title('h_padded with zeros (no smoothing)')
        pl.plot(abs(h_padded))
        pl.show()
    
    # mulitply by cos func. to smooth h_padded
    w_h_tfirsthalf = np.cos(np.linspace(0, np.pi/2.0, len(h_tfirsthalf)))*h_tfirsthalf # first quarter of period
    w_h_tsecondhalf = np.cos(np.linspace(3*np.pi/2.0, 2*np.pi, len(h_tsecondhalf) ) )*h_tsecondhalf # last quarter of period
                                    # h_padded is impulse response fn of length 2M
    h_padded = np.concatenate( (w_h_tfirsthalf, zeros, w_h_tsecondhalf),0)    
    
    if 0: # plot the SMOOTHED padded response function <<h_padded>>
        pl.figure()
        pl.title('h_padded with zeros after smoothing')
        pl.plot(abs(h_padded))
        pl.show()
    print('length of h_padded =',len(h_padded))
    
    # FT of h_padded
    H_pad_onesided = np.fft.rfft(h_padded)
    
    print('length of H_pad_onesided =',len(H_pad_onesided))
    
    ''' When filtering a data sample <<stream>> of length 2M, the first and last
    M/2 samples will be spoiled and the middle M will be filtered correctly. 
    
    Split the stream into a number of substreams, such that the length of each
    substream is twice the length of a PULSE_WINDOW. This way, the middle M 
    points will be a given pulse window which will be filtered properly.
    
    The first and last M/2 points are spoiled, so we append M/2 zeros at the 
    beginning and end of the OF sample.
    '''
     
    print('stream length =',len(stream))
    
    # currently throws out first and last half stream segment of data
    OFdata = []
    segments = np.split(stream, int(len(stream)/(2*PULSE_WINDOW*fe))) # split stream into 2x pulse_window segments
    M = int(len(segments[0])/2)
    
    zeros = np.zeros(int(M/2))
    OFdata.append(zeros)
    
    '''
    Here we iterate through our <<segments>> array. data_a is the i-th segment
    to which we apply the  OF and append the middle M pts to our result. 
    If this_segment is not the last segment, then we make data_b an array of 
    the second M points in this_segment and the first M points in next_segment.
    Then we apply the OF to data_b and append the middle M points to our result,
    which means we appended unspoiled data for this_segment from M/2 to 3M/2, 
    then from this_segment 3M/2 to next_segment M/2.
    '''
    for i in range(len(segments)):
        this_segment = segments[i]
        data_a = this_segment
        
        data_a = np.fft.rfft(data_a)
        data_a = H_pad_onesided*data_a
        data_a = np.fft.irfft(data_a)
        data_a = data_a[int(M/2):int(3*M/2)]
        OFdata.append(data_a)
        
        if i != (len(segments)-1):
            next_segment = segments[i+1]
            data_b = []
            for j in this_segment[M:]:
                data_b.append(j)
            for j in next_segment[:M]:
                data_b.append(j)
    
            data_b = np.fft.rfft(data_b)
            data_b = H_pad_onesided*data_b
            
            data_b = np.fft.irfft(data_b)
            OFdata.append(data_b[int(M/2):int(3*M/2)])
            
    OFdata.append(zeros) # append M/2 zeros at the end.
    OFdatalist = []
    
    for seg in OFdata:
        for i in seg:
            OFdatalist.append(i)
    
    OFdatalist = np.asarray(OFdatalist)
    print('Length of OFdatalist =',len(OFdatalist))
                
    print('***************************************************************************************************')
    
    return OFdatalist, H_pad_onesided, h
#******************************************************************************************************************
'''
                                                                                                        getPulseAVG
'''
def getPulseAVG(pulses,fe):
    print('Length of pulses:', len(pulses))
    pulseAVG = sum(pulses)/len(pulses)
#    pulseAVG = np.average(pulses)
    
    def sigma(data,avg):
        sigma = (data-avg)**2
        sigma = np.average(sigma)
        sigma = np.sqrt(sigma)
        return sigma
        
    sigmas = []
    goodpulses = []
    for pulse in pulses:
        sig = sigma(pulse,pulseAVG)
        if sig < len(pulses):
            goodpulses.append(pulse)
            sigmas.append(sig)
    print(np.average(sigmas))
    if np.average(sigmas) < len(pulses):
        print(len(goodpulses))
        return sum(goodpulses)/len(goodpulses)
    else:
#        print(len(goodpulses))
        return getPulseAVG(goodpulses)
    


t1, t2 = 0, 200
n1, n2 = t1*fe, t2*fe

sample_stream = V[n1:n2]
sample_stream_hpf = V_hpf[n1:n2]

pulses = initial_pulse_trigger(sample_stream_hpf,fe)
pulseAVG = getPulseAVG(pulses,fe) # compute the signal template from avg of 6keV pulses in first 200s

f, f_pulseAVG = getWindowedFFT(pulseAVG)
f_pulseAVG_corr = f_pulseAVG/abs(response)
f_pulseAVG_corr[0] = 0

pl.figure()
pl.loglog(f,abs(f_pulseAVG),label='pulse_AVG')
pl.loglog(f,abs(f_pulseAVG_corr),label='2Hz HPF correction')
pl.legend()
pl.show()
#pulseAVG = getInverseFFT(f_pulseAVG_corr)

time_window = np.linspace(0, PULSE_WINDOW, int(fe*PULSE_WINDOW) ) 
a,T1,T2,T3 = 809.4118131886025,0.0035474244745647573,0.010974546445523942,0.010974546475159132 
                # figures above aquired from julien_data_pulseID_and_fitting
template_fn = theta_exp_pulse(time_window, a,T1,T2,T3)
f, fft_template = getWindowedFFT(template_fn)
fft_template_corr = fft_template/abs(response)
fft_template_corr[0] = 0

pl.figure()
pl.loglog(f,abs(fft_template),label='template fn')
pl.loglog(f,abs(fft_template_corr),label='2Hz HPF correction')
pl.legend()
pl.show()
#template_fn = getInverseFFT(fft_template_corr)

template_fn = high_pass_filter(template_fn, fc,order=order)
template_fn = template_fn/max(template_fn)
 

pulseAVG = pulseAVG/max(pulseAVG)

freq_array, J = getNoisePSD(sample_stream,fe)
OF_stream, H_trans, h = OFTransferFn(V_hpf, template_fn, (J/abs(response)), freq_array) #(params1), theta_exp_pulse)

#******************************************************************************************************************
pl.figure()
pl.grid(b=True,which='minor',axis='x')
pl.loglog(J,label='PSD template')
pl.loglog(J/abs(response),label='2Hz Correction')
pl.xlabel('Frequency [Hz]')
pl.ylim(1e-4,1e2)
pl.legend()
pl.show()



f, f_template_fn1 = getWindowedFFT(template_fn,scaling='density')
#f, f_pulseAVG = getWindowedFFT(pulseAVG)
f, f_template_fn2 = sgl.periodogram(template_fn, fe)
f, f_pulseAVG = sgl.periodogram(pulseAVG, fe)

pl.figure()
pl.plot(pulseAVG,label='pulseAVG')
pl.plot(template_fn, label='template function')
pl.legend()
pl.show()

#J = J/abs(response)

pl.figure()
#pl.title('OF Transfer fn???')
pl.loglog(abs(f_pulseAVG),label='PulseAVG')
#pl.loglog(abs(H_trans),'red', label='transfer fn')
#pl.loglog(abs(f_template_fn1), label='template fn - FFT')
pl.loglog(abs(f_template_fn2), label='template fn - periodogram')
pl.loglog(J,label = 'Noise PSD')
#pl.ylim(1e-2,1e6)
pl.ylim(1e-5,1e1)
pl.xlabel('Frequency [Hz]')
pl.legend()
pl.show()


#def trigger(data):
n_peaks = sgl.find_peaks(OF_stream,height=200,distance=int(fe/2))[0]

    
pl.figure()
#pl.plot(time[n1:n2], V_hpf[n1:n2],'darkgray',label='V_hpf')
#pl.plot(time[n1:n2], OF_stream[n1:n2],'--r',label='OFstream')
pl.plot(time, V_hpf,'darkgrey',label='V_hpf')
pl.plot(time, V,'--k',label='V_raw')
pl.plot(time, OF_stream,'--r',label='OFstream')
#    pl.plot(data    markersize=5)
for i in (n_peaks[n1:n2]):
#    print(i)
    pl.plot(time[i],OF_stream[i],'bo',markersize=5)
#pl.plot(200,1000,'o')
pl.legend()
pl.xlabel('Time [s]')
pl.ylabel('Amplitude [ADU]')
#pl.yscale('log')
pl.xlim(44,46)
#pl.ylim(1,1500)
pl.ylim(-500,1500)
pl.show()


pl.figure()
pl.plot(time, V,'darkgrey',label='V_raw')
#for i in (n_peaks[n1:n2]):
#    pl.plot(time[i],OF_stream[i],'bo',markersize=5)
pl.legend()
pl.xlabel('Time [s]')
pl.ylabel('Amplitude [ADU]')
#pl.yscale('log')
pl.xlim(26.9,27.9)
#pl.ylim(1,1500)
pl.ylim(-500,1500)
pl.show()


#windows = []
amplitudes = []
chisquares = []
event_amps = []
event_chis = []
sigmas = np.array(np.sqrt(J))
#sigmas[0] = 0
for i in n_peaks:
    if i > fe/2:
        left = int(i - 0.5*fe)
        right = int( i + 0.5*fe )
        window = OF_stream[left:right]
#        windows.append(window)
#        print(len(window))
        
        freqs, vf = getWindowedFFT(window)
        freqs, sf = getWindowedFFT(template_fn)
            
#        print('len vf, sf:', len(vf),len(sf))
        def chisqfunc(a):
            model = a*sf
            chisq = 0
            for i in range(len(vf)):
                chisq = chisq + np.abs(vf[i]-model[i])**2/J[i]
#            chisq = np.sum(abs(vf-model)**2)/J
            return chisq
       
        def fitfunc(x,a):
            return a*abs(sf)
        
    
        pot = curve_fit(fitfunc,freqs,abs(vf),sigma=sigmas)
        a = pot[0]
#        print(chisqfunc(a))
        chisq = chisqfunc(a)
#        if i%10 == 0:
        if i==18056:
            pl.figure(figsize=(12,3))
            pl.subplot(121)
            pl.plot(window,'r',label='OF data')
            pl.plot(a*template_fn,'b--',label='fit, a=%f'%a)
            pl.plot(template_fn,label='template fn')
            pl.title(chisq)
            pl.legend()
            pl.subplot(122)
            pl.loglog(abs(vf),'r',label='OF data')
            pl.loglog(abs(sf),label='template fn')
            #    pl.loglog(f, periodogram)
            #    pl.ylim(10**(-10),100)
            #    pl.title('Power Spectra for Sample %i of 90'%i)
            pl.xlabel('Frequency (Hz)')
            pl.legend()
            pl.show()
#        x0 = np.array([5])
#        result =  minimize(chisqfunc, x0)
#    #    print result
#        assert result.success==True
#        a=result.x*max(window)
##        print(chisq)
        amplitudes.append(a[0]*max(template_fn))
        chisquares.append(chisq)
      
event_times = []
for i in range(len(amplitudes)):
    if amplitudes[i] < 5000:
        event_amps.append(amplitudes[i])
        event_chis.append(chisquares[i])

    event_times.append(time[n_peaks[i]])
        

pl.figure()
pl.hist(chisquares,bins=150)
pl.ylabel('Chi-Square')
pl.xlabel('events')
pl.show()


pl.figure()
pl.hist(amplitudes,bins=150)
pl.ylabel('amplitudes')
pl.xlabel('events')
pl.show()


pl.figure()
pl.hist(event_chis,bins=150,rwidth = 0.9)
pl.xlabel('Chi-Square')
pl.xscale('log')
pl.show()

pl.figure()
pl.scatter(amplitudes,chisquares,marker='.')
pl.yscale('log')
pl.xlabel('Amplitude [ADU]')
pl.ylabel('Chi-Square')
pl.show()
        

pl.figure()
pl.scatter(event_amps,event_chis,marker='.')
pl.yscale('log')
pl.xlabel('Amplitude [ADU]')
pl.ylabel('Chi-Square')
pl.show()
#    

pl.figure()
pl.scatter(event_times,amplitudes,marker='.')
pl.ylabel('Amplitude [ADU]')
pl.xlabel('Time [s]')
pl.ylim(500,2000)
#pl.yscale('log')
pl.show()


#pl.figure()
#pl.plot(time[ni:nf], V_hpf[ni:nf],'darkgray',label='V_hpf')
#pl.plot(time[ni:nf], OF_stream[ni:nf],'--r',label='OFstream')
#pl.plot(200,1000,'o')
#pl.legend()
#pl.xlabel('Time [s]')
#pl.ylabel('Amplitude [ADU]')
#pl.show()
#    

'''Plotting RAW data '''
if 1:
    pl.figure()
    #pl.plot(little_time,V[int(index_i):int(index_f)])
    pl.plot(time,V)
    #pl.ylim(-100,1000)
    pl.ylim(-0,2000)
    pl.xlim(316,318)
    pl.title('Raw Data')
    pl.ylabel('Amplitude (ADU?)')
    pl.xlabel('Time [s]')
#    pl.yscale('log')
    pl.show()

'''Plotting RAW data vs. HIGH-PASS-FILTERED data'''
if 1:
    pl.figure()
    pl.title('HPF of order %i and cutoff %i Hz'%(order,fc))
    #pl.plot(time,V,'darkgray',label='V_raw')
    pl.plot(time,V_hpf,label='V_hpf')
    #pl.ylim(-100,1000)
    #pl.ylim(-1000,2000)
    #pl.xlim(ti,tf)
    #pl.title('HPF_2Hz Applied')
    pl.ylabel('Amplitude (ADU?)')
    pl.xlabel('time (s)')
    pl.legend()
    pl.show()

