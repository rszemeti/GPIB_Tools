import pyvisa
import time
import math
import re
import sys
import os
import json
from datetime import datetime
import numpy as np
from scipy.signal import find_peaks
from scipy.fft import fft
from modules.Data import ResultSet
from modules.Data import Trace

class HPSA:
    def __init__(self,id):
        try:
            self.rm = pyvisa.ResourceManager()
            self.id=id
            self.inst = self.rm.open_resource('GPIB0::%d::INSTR' % (id))
            self.model=self.inst.query("ID")
            self.serial=self.inst.query("SER")
            self.version=self.inst.query("REV")
        except:
            print("failed to init GPIB")
            
    def lockBand(self,i):
        self.inst.write("HNLOCK %d" % (i))
        time.sleep(1)
        
    def unlockBand(self):
        self.inst.write("HNLOCK OFF")
        
    def correctionOn(self):
        self.inst.write("CAL ON")
        
    def correctionOff(self):
        self.inst.write("CAL OFF")
        
    def markerNormal(self,f):
        self.inst.write("MKN %0.3f GHZ" % (f))
        
    def markerTrackOn(self):
        self.inst.write("MKTRACK ON")
        
    def setSpan(self,f):
        self.inst.write("SP %d MHZ" % (f))
        
    def setBandwidth(self,f):
        self.inst.write("RB %d MHZ" % (f))
        
    def preselectorPeak(self):
        self.inst.write("PP*")

    def preset(self):
        str = self.inst.write("IP")
        time.sleep(2)
            
    def setFreq(self,f):
        str = self.inst.write("CF %f.3 GHZ" % (f))

    def getPeakedMarkerLevel(self):
        str = self.inst.query("PP*; MA")
        return float(str)
    
    def getMarkerLevel(self):
        str = self.inst.query("MA")
        return float(str)
    
    def getAmpcor(self):
        str = self.inst.query("AMPCOR?")
        return str
    
    def setAmpcor(self,f,c):
        self.inst.write("AMPCOR %0.3fGHZ %0.2fDB" % (f,c))
        
    def saveAmpcor(self):
        self.inst.write("SAVET AMPCOR")

    def readCalData(self):
        self.inst.write("CAL FETCH")
        str = self.inst.query("CAL DUMP")
        return str

    def readCat(self):
        #self.inst.write("CAT *,INT")
        str = self.inst.query("CAT *,INT")
        return str

    def getSig(self):
        self.inst.write("RL 10.0DB");
        self.inst.write("SNGLS;TS;")
        time.sleep(1)
        self.inst.write("MKPK;DONE;MKCF;")
        time.sleep(1)
        freq = float(self.inst.query("MKF?"))
        lvl = float(self.inst.query("MKA?"))
        self.pkLvl=lvl
        self.freq=freq
        return (freq,lvl)

    def initSignal(self):
        (freq,lvl) = self.getSig()
        self.inst.write("RL %sDB" % lvl)
        self.inst.write("SP 20KHZ")
        (freq,lvl) = self.getSig()
        return(freq,lvl)

    def levelOffset(self,offset):
        self.inst.write("RL %0.2fDB" % (self.pkLvl-offset))


    def acquire(self,freq,span,levelOffset,count=5):
        self.levelOffset(levelOffset)
        self.inst.write("SP %dKHZ" % span)
        bw=300
        if(span==20):
            self.inst.write("RBW 300HZ")
            bw=602.6
        if(span==100):
            self.inst.write("RBW 1KHZ")
            bw=955
        if(span==1000):
            self.inst.write("RBW 10KHZ")
            bw=10964
        if(span==5000):
            self.inst.write("RBW 30KHZ")
            bw=38018
        self.inst.write("FA %dHZ; FB %dHZ" % (freq, freq+span*1000))
        
        self.inst.write("SNGLS;")
        results = [self.getTrace() for _ in range(count)]
        self.inst.write("CONTS;")   
        return (bw,self.average_arrays(results))

    def reset(self):
        # Put analyser back into sensible state.
        self.inst.write("RL %0.2fDB" % (self.pkLvl))
        self.inst.write("SP 1MHZ")
        self.inst.write("RBW AUTO")
        self.inst.write("CF %dHZ" % self.freq)
    
    def average_arrays(self,arrays):
        num_arrays = len(arrays)
        num_elements = len(arrays[0])
        
        # Initialize result array with zeros
        result = [0] * num_elements
        
        # Calculate sum of corresponding elements
        for array in arrays:
            for i, value in enumerate(array):
                result[i] += value
        
        # Calculate average
        result = [value / num_arrays for value in result]
        return result

    def getTrace(self):
        trace = self.inst.query("TS;TRA?")
        data_list = trace.split(',')
        data_float = [float(item) for item in data_list if item]
        return data_float
    
    def acquirePlot(self,startF=1E3,endF=10E6,nAvg=1):
        result = ResultSet(startF,endF);
        
        (freq,lvl)=self.initSignal()
        result.setFreq(freq)
        result.setLevel(lvl)

        nAvg=1
        fPlotted=startF
        levOffset=10.0
        freqOffset=startF
        for span in [20,100,1000,5000]:
            t = Trace(span)
            if(span==1000):
                levOffset=40.0
            for c in range (0,3):
                (bw,data)=self.acquire(freq+freqOffset,span,levOffset,nAvg)
                for i, value in enumerate(data):
                    f = (i * (span*1000/(len(data)-1)))+ freqOffset
                    if(f>fPlotted):
                        fPlotted=f
                        point = (f,getNoiseLevel(value,bw,lvl))
                        t.addPoint(point)
                result.addTrace(t)
                freqOffset += span*1000
                if(freqOffset>endF):
                    break
                if(levOffset <= 20):
                    levOffset=30.0

        self.reset()
        return result
