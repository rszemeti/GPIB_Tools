import pyvisa
import time
import re
import sys
import os
from datetime import datetime

class MarconiSigGen:
    def __init__(self,rm,id):
        self.id=id
        self.rm=rm
        self.inst = rm.open_resource('GPIB0::%d::INSTR' % (id))
        str = self.inst.query("*IDN?")
        self.serial = str
        
    def read(self):
        self.readFreq()
        self.readLevel()

    def readLevel(self):
        str = self.inst.query("RFLV?")
        if re.match("^:RFLV:", str):
            self.level = float(re.findall("VALUE (-?\d+\.\d+);",str)[0])
            self.units = re.findall("UNITS (\w+);",str)[0]
            return self.freq
        elif re.match("^[SUDK][A-E][DRWO]", str):
            print("Failed to get frequency")
            print(str)
        else:
            print("Eh?")
            print(str)

    def readFreq(self):
        str = self.inst.query("CFRQ?")
        print(str)
        if re.match("^:CFRQ:", str):
            self.freq = float(re.findall("VALUE (\d+\.\d+);",str)[0])
            return self.freq
        elif re.match("^[SUDK][A-E][DRWO]", str):
            print("Failed to get frequency")
            print(str)
        else:
            print("Eh?")
            print(str)

    def setFreq(self,f,units="GHZ"):
        self.inst.write("CFRQ:VALUE %f%s;" % (f,units))

    def setLevel(self,l,units="DBM"):
        self.inst.write("RFLV:VALUE %0.1f%s;" % (l,units))
        self.level = round(l,1)

    def getLevel(self):
        return self.level
        
    def setRfOn(self):
        self.inst.write("RFLV:ON;")
        
    def setRfOff(self):
        self.inst.write("RFLV:OFF;")

class WiltronSigGen:
    def __init__(self,rm,id):
        self.id=id
        self.rm=rm
        self.inst = rm.open_resource('GPIB0::%d::INSTR' % (id))
        
    def read(self):
        self.readFreq()
        self.readLevel()

    def readLevel(self):
        str = self.inst.query("RFLV?")
        if re.match("^:RFLV:", str):
            self.level = float(re.findall("VALUE (-?\d+\.\d+);",str)[0])
            self.units = re.findall("UNITS (\w+);",str)[0]
            return self.freq
        elif re.match("^[SUDK][A-E][DRWO]", str):
            print("Failed to get frequency")
            print(str)
        else:
            print("Eh?")
            print(str)
            
    def setFreq(self,f):
        str = self.inst.write("CF1 %f.4 GH" % (f))

    def setLevel(self,l):
        str = self.inst.write("L1 %f.2 DB" % (l))
        self.level = round(l,2)

    def getLevel(self):
        return self.level

class HPSA:
    def __init__(self,rm,id):
        self.id=id
        self.rm=rm
        self.inst = rm.open_resource('GPIB0::%d::INSTR' % (id))
        self.model=self.inst.query("ID")
        self.serial=self.inst.query("SER")
        self.version=self.inst.query("REV")

            
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

class Band:
    def __init__(self,fStart,fEnd,step,id, bw):
        self.fStart=fStart
        self.fEnd=fEnd
        self.step=step
        self.id=id
        self.bw=bw

class Calibrator:
    def __init__(self,sg,sa,t,file):
        self.sg=sg
        self.sa=sa
        self.file=file
        self.sgl=-10.0
        self.target=t
        self.sa.preset()
        self.sa.correctionOff()

    def get300(self):
        self.sg.setFreq(0.3)  
        self.sg.setLevel(self.sgl)
        self.sa.setFreq(0.3)
        self.sa.lockBand(0)
        self.sa.setSpan(10)
        self.sa.setBandwidth(10)
        self.sa.setFreq(0.3)
        self.sa.markerNormal(0.3)
        self.sa.markerTrackOn()
        time.sleep(2)
        lvl=self.getLevel()
        self.zero=lvl-self.target
        return self.zero

    def calibrateBand(self,band):
        print("Calibrating Band %d from %0.3f to %0.3f GHz in steps of %0.4f GHz" %(band.id, band.startF, band.endF, band.step))
        f=band.fStart
        
        self.sg.setFreq(f)  
        self.sg.setLevel(self.sgl)
        
        self.sa.lockBand(band.id)
        time.sleep(2)
        self.sa.setSpan(10)
        self.sa.setBandwidth(band.bw)
        self.sa.setFreq(f)
        self.sa.markerNormal(f)
        self.sa.markerTrackOn()

        count = 0
        while(f<=(band.fEnd+.00001)):
            self.sg.setFreq(f)
            self.sg.setLevel(self.sgl)
            self.sa.setFreq(f)
            time.sleep(2)
            lvl = self.getLevel()
            corr = lvl-self.target-self.zero
            print("Freq: %0.3f, Correction: %0.2f" %(f,corr))
            self.file.write("%0.3f,%0.2f\n" %(f,corr))
            f+=band.step
        self.sa.unlockBand()

    def getLevel(self):
        self.sa.getPeakedMarkerLevel()
        time.sleep(1)
        lvl = self.sa.getMarkerLevel()
        self.err = lvl-self.target
        while(abs(self.err) > 0.1):
            self.sg.setLevel(self.sg.getLevel() - self.err)
            #print("Marker: %0.2f, error: %0.2f, new level: %0.2f" % (lvl,self.err,self.sg.getLevel()))
            time.sleep(1)
            lvl = sa.getMarkerLevel()
            self.err =  lvl - self.target
            if(abs(self.err) > 15):
                print("Quitting, signal too low, check signal path.")
                sys.exit()
        corrLevel = self.sg.getLevel() - self.err
        #print("Marker: %0.2f, error: %0.2f, o/p level: %0.2f, final level: %0.2f" % (lvl,self.err,self.sg.getLevel(), corrLevel))
        return round(corrLevel,2)

bandParameters = [
    {'fStart': 0.012, 'fEnd': 2.676, 'step': 0.072, 'id': 0, 'bw': 10},
]

bandList = [Band(**params) for params in bandParameters]


bandParameters2 = [
    {'fStart':  2.028, 'fEnd':  2.892, 'step': 0.072,  'bw': 10, 'id': 0},
    {'fStart':  2.750, 'fEnd':  6.500, 'step': 0.2343, 'bw': 10, 'id': 1},
    {'fStart':  6.100, 'fEnd': 12.724, 'step': 0.184,  'bw': 10, 'id': 2},
    {'fStart': 12.450, 'fEnd': 19.350, 'step': 0.230,  'bw': 10, 'id': 3},
    {'fStart': 19.100, 'fEnd': 26.500, 'step': 0.148,  'bw': 10, 'id': 4},
]

bandList2 = [Band(**params) for params in bandParameters2]

rm = pyvisa.ResourceManager()

# Instruments are defined here!
# HP spectrum analyser on ID 18
sa=HPSA(rm,18)

# 0-2.7G Marconi sig gen on ID 2
sg1=MarconiSigGen(rm,2)

# 2 to 20G sig gen on ID 7
sg2=WiltronSigGen(rm,7)

# Get the filename from the user
file_name = input("Enter a filename for cal data storage: ")

# Check if the file exists
if os.path.isfile(file_name):
    # Ask the user if they want to overwrite the file
    user_response = input("File already exists. Do you want to overwrite? (y/n): ")

    if user_response.lower() == 'n':
        print("Exiting without overwriting the file.")
        sys.exit()

# Get the current date and time
now = datetime.now()

# Print the current date and time in local format
print(now.strftime("Calibration started at %Y-%m-%d %H:%M:%S"))

with open(file_name, 'w') as file:
    print("Sig gen 1 is %s" % sg1.serial)
    print("Analyser: %s  Serial number: %s  Firmware version: %s" % (sa.model, sa.serial, sa.version))
    file.write("Analyser: %s  Serial number: %s  Firmware version: %s\n" % (sa.model, sa.serial, sa.version))
    file.write("Signal gernator is %s\n" % sg1.serial)
    file.write(now.strftime("Calibration date %Y-%m-%d %H:%M:%S\n"))
    sys.exit()

    input("Connect Sig Gen 1 to analyser and press enter to continue")
    
    cal = Calibrator(sg1,sa,-9,file)
    print("Finding reference level at 300MHz")
    zero=cal.get300()
    print("Zero is %0.2f, starting calibration run ..." % (zero))
    for band in bandList:
        cal.calibrateBand(band)

        
    input("Now connect Sig Gen 2 and press enter to continue")
    
    cal = Calibrator(sg2,sa,-9,file)
    cal.zero = -3.27
    for band in bandList2:
        cal.calibrateBand(band)

print("Calibration complete.")














