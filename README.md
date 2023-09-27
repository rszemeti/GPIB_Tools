# GPIB_Tools
Random GPIB utility tools in Python ... 

# Really, did the world really need some more random GPIB utilities?

Probably not, but I wrote them anyway.  You'll need a National Instruments GPIB interface or one of the cheap clones available in eBay (hint: the real ones are £1,800 the eBay ones are £60 ... they will NOT work
with the current NI drivers, however you can find the older (pre V 17.5) drivers on Archive.org 

If you don't already have Python, install Python from here: https://www.python.org/downloads/

Install the Python "VISA" library to communicate with the GPIB interface 

Easiest way is from the command line run:  pip install pyvisa

## Calibrator.py

Simple python program that will control a signal generator or two and produces a set of flatness calibration data from an HP 8592L spectrum analyser. 
No, sadly I haven't found a way to load it back int to the analyser automatically, you'll have to key it in, but hey, its faster than taking the readings by hand.

You'll need to edit the device numbers in the file to match your setup:

```python
# Instruments are defined here!
# HP spectrum analyser on ID 18
sa=HPSA(rm,18)

# 0-2.7G Marconi sig gen on ID 2
sg1=MarconiSigGen(rm,2)

# 2 to 20G sig gen on ID 7
sg2=WiltronSigGen(rm,7)
```

Change the GPIB addresses to match your setup  .. I used two sig gens from 0-2.7 GHz and then 2.7 to 20GHz ... the config should be reasonably obvious, adjust the bits at the bottom of the file to suit your needs.

## CalibrationReader.py

Very similar, reads the current flatness data out of the instrument and stores it in a file, just in case you ever lose it.
