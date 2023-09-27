import pyvisa
import time
import re
import sys
import os
from datetime import datetime

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

# Get the current date and time
now = datetime.now()

# Print the current date and time in local format        

rm = pyvisa.ResourceManager()

sa=HPSA(rm,18)

# Get the filename from the user
file_name = input("Enter a filename for cal data storage: ")

# Check if the file exists
if os.path.isfile(file_name):
    # Ask the user if they want to overwrite the file
    user_response = input("File already exists. Do you want to overwrite? (y/n): ")

    if user_response.lower() == 'n':
        print("Exiting without overwriting the file.")
        quit()
        
c=sa.readCalData()
vals = c.split(',')
i=0

# Open the file for writing
with open(file_name, 'w') as file:
    file.write(now.strftime("Calibration dump at %Y-%m-%d %H:%M:%S\n"))
    file.write("Analyser: %s  Serial number: %s  Firmware version: %s\n" % (sa.model, sa.serial, sa.version))
    for v in vals:
        if(v=="12000000"):
            print(i)
            break
        i += 1
    for b in range(5):
        print("# Band %d"%b)
        file.write("# Band %d\n"%b)
        startF=int(vals[i])
        i += 1
        endF=int(vals[i])
        i+=1
        step=int(vals[i])
        i+=1
        f=startF
        while (f<= endF):
            lvl=float(vals[i])
            print("%d,%0.2f"%(f,lvl))
            file.write("%d,%0.2f\n"%(f,lvl))
            f+=step;
            i+=1













