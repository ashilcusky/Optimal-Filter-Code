# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 19:50:50 2019

@author: rdpuser
"""
import os
import numpy as np
import matplotlib.pyplot as pl
import scipy.signal as sgl
from scipy.optimize import curve_fit
import binascii


#with open("LyonData_se26g000_000", "rb") as input:
#    aByte= input.read(1)
##    print(type(aByte))
#    while aByte and ord(aByte) != 0: aByte= input.read(1)
    
#data = binascii.a2b_base64(aByte).decode('utf-8')
#data = binascii.a2b_base64(aByte).decode('7077')
#print(type(data))
#print(type(aByte),len(aByte))
fe = 400 # sampling rate (Hz)
t = 5 # seconds to measure
timespace = t*400
    
"""
with open("LyonData_se26g000_000_copy.txt", encoding = 'utf8', errors = 'ignore') as f:
    w, h = [x for x in next(f).split()] # read first line
    array = []
    for line in f: # read rest of lines
        array.append([x for x in line.split()])
        
print(array)
"""
voltages = np.memmap("LyonData_se26g000_000", dtype='int16', mode='r+')

#V = voltages[10000:10000+timespace]
V = voltages[67000:67000+timespace]
t = np.arange(len(V))
print(len(V))
pl.figure(figsize=(15,10))
pl.plot(t/fe,V)
pl.show()

"Hey Alex, I found some things that look like pulses?  I am unsure of what the three super noisy bits are though...Any ideas?"