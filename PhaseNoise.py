import pyvisa
import time
import math
import re
import sys
import os
import json
from datetime import datetime
from typing import List
import numpy as np
from scipy.signal import find_peaks
from scipy.fft import fft

import sounddevice as sd


import PySimpleGUI as sg

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

class Trace:
    def __init__(self, bw, points=None):
        self.bw = bw
        self.points = points if points is not None else []

    def addPoint(self, point):
        self.points.append(point)

    def getPoints(self):
        return self.points

class ResultSet:
    def __init__(self,startF,endF):
        self.startF=startF
        self.endF=endF
        self.traces=[]

    def addTrace(self, trace: Trace):
        self.traces.append(trace)

    def getTraces(self) ->List[Trace]:
        return self.traces

    def setFreq(self,freq):
        self.freq=freq

    def setLevel(self,lvl):
        self.level=lvl

    def save(self, filename):
        data = {
            'startF': self.startF,
            'endF': self.endF,
            'freq': self.freq,
            'level': self.level,
            'traces': [trace.__dict__ for trace in self.traces]
        }
        with open(filename, 'w') as file:
            json.dump(data, file)

    def load(self, filename):
        with open(filename, 'r') as file:
            data = json.load(file)
            self.startF = data['startF']
            self.endF = data['endF']
            try:
                self.freq = data['freq']
                self.level = data['level']
            except KeyError as e:
                print(f"Missing key in JSON data: {e}")
                self.freq = 1
                self.level = 0
                            
            self.traces = [Trace(**trace) for trace in data['traces']]
 

def map_to_log_scale(value, lower_bound, upper_bound, output_lower_bound, output_upper_bound):
    # Ensure the value is within the specified range
    value = max(min(value, upper_bound), lower_bound)
    
    # Calculate the logarithmic mapping
    log_lower = math.log10(lower_bound)
    log_upper = math.log10(upper_bound)
    log_value = math.log10(value)
    
    mapped_value = ((log_value - log_lower) / (log_upper - log_lower)) * (output_upper_bound - output_lower_bound) + output_lower_bound
    return mapped_value+15

def getNoiseLevel(value, bw,lvl):
    return value - 10*math.log10(bw*1.3) -lvl +2.0

def moving_average(data, window_size):
    smoothed_data = []
    for i in range(len(data)):
        start_idx = max(0, i - window_size + 1)
        end_idx = i + 1
        window = data[start_idx:end_idx]
        avg_level = sum(point[1] for point in window) / len(window)
        smoothed_data.append((data[i][0], avg_level))
    return smoothed_data

def triangular_window(data, window_size):
    half_size = window_size // 2
    smoothed_data = []

    for i in range(len(data)):
        start_idx = max(0, i - half_size)
        end_idx = min(len(data), i + half_size + 1)
        window = data[start_idx:end_idx]

        # Apply triangular weighting to the window
        weights = [1 - abs(j - half_size) / half_size for j in range(len(window))]

        avg_level = sum(weights[j] * point[1] for j, point in enumerate(window)) / sum(weights)
        smoothed_data.append((data[i][0], avg_level))

    return smoothed_data

def log_interpolation(smoothed_data, num_points):
    freq_values = np.array([point[0] for point in smoothed_data])
    level_values = np.array([point[1] for point in smoothed_data])

    log_freq_values = np.log10(freq_values)
    new_log_freq_values = np.linspace(min(log_freq_values), max(log_freq_values), num_points)
    new_freq_values = 10 ** new_log_freq_values

    new_level_values = np.interp(new_freq_values, freq_values, level_values)

    resampled_data = list(zip(new_freq_values, new_level_values))
    return resampled_data

def extract_peaks(smoothed_data):
    freq_values = [point[0] for point in smoothed_data]
    level_values = [point[1] for point in smoothed_data]

    peaks, _ = find_peaks(level_values, height=5, distance=5)

    peak_positions = [(freq_values[i], level_values[i]) for i in peaks]
    return peak_positions

def find_peaks_derivative(data, threshold):
    derivative = np.diff(data)

    # Find points where the derivative changes sign
    sign_changes = np.where(np.diff(np.sign(derivative)))[0] + 1

    # Filter for points above the threshold
    peaks = [i for i in sign_changes if data[i] > threshold]

    return peaks


def plotPoints(graph,result,color='blue'):
    # Plot the data points
    data_points = []
    minF=10
    for trace in result.getTraces():
        for (f, level) in trace.getPoints():
            if(f>minF):
                minF=f
                data_points.append((f, level))
    resampled_data = log_interpolation(data_points, num_points=2000)
    smoothed = triangular_window(resampled_data,12)
    
    for (f,level) in data_points:
        graph.draw_point((map_to_log_scale(f, 10, 10000000, 0, 400), level), color='light grey', size=1)

    for i in range(len(smoothed) - 1):
        x1 = map_to_log_scale(smoothed[i][0], 10, 10000000, 0, 400)
        y1 = smoothed[i][1]
        x2 = map_to_log_scale(smoothed[i+1][0], 10, 10000000, 0, 400)
        y2 = smoothed[i+1][1]

        graph.draw_line((x1, y1), (x2, y2), color=color, width=1)
        
    window.refresh()       

def acquirePlot(startF=1E3,endF=10E6,nAvg=1,addr=0):
    result = ResultSet(startF,endF);
    
    rm = pyvisa.ResourceManager()

    sa=HPSA(rm,addr)
    (freq,lvl)=sa.initSignal()
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
            (bw,data)=sa.acquire(freq+freqOffset,span,levOffset,nAvg)
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

    sa.reset()
    return result

class Config:
    def __init__(self):
        self.analyserAddress = None
        self.audioSource = None
        self.sample_rate = None

    def save_to_file(self, filename):
        data = {
            'analyserAddress': self.analyserAddress,
            'audioSource': self.audioSource,
            'sample_rate': self.sample_rate
        }
        with open(filename, 'w') as file:
            json.dump(data, file)

    @staticmethod
    def load_from_file(filename):
        try:
            with open(filename, 'r') as file:
                data = json.load(file)
                config = Config()
                config.analyserAddress = data.get('analyserAddress')
                config.audioSource = data.get('audioSource')
                config.sample_rate = data.get('sample_rate')
                return config
        except FileNotFoundError:
            return Config()
        
import sounddevice as sd


class AudioAnalyser:

    def __init__(self,config):
        self.debug=False
        self.duration = 5
        self.sample_rate = config.sample_rate
        self.channels = 1
        self.startF=10
        self.endF=20000
        self.audio_data = np.array([])
        self.devices = sd.query_devices()
        self.device_list = []

    # Define a callback function to capture audio
    def audio_callback(self,indata, frames, time, status):
        self.audio_data = np.append(self.audio_data, indata[:, 0])  # Assuming mono input

    def hamming_window(self,data):
        window = np.hamming(len(data))
        return data * window

    def acquire(self):
        print("Acquiring audio data at %d sample rate for %d seconds" %(self.sample_rate,self.duration))
        self.audio_data = np.array([])
        # Start audio stream
        with sd.InputStream(callback=self.audio_callback, channels=self.channels, samplerate=self.sample_rate):
            sd.sleep(self.duration * 1000)  # Sleep for the specified duration in milliseconds

        # Assuming audio_data contains your acquired samples
        windowed_data = self.hamming_window(self.audio_data)
        fft_result = fft(windowed_data)
        
        num_points = len(fft_result)
        points_per_decade = 500

        result=ResultSet(self.startF,self.endF)
        
        # Generate an array of frequencies from 10 Hz to 20,000 Hz (log scale)
        frequencies = np.logspace(np.log10(self.startF), np.log10(self.endF), num=points_per_decade * 4)
        indices = (frequencies / (self.sample_rate / num_points)).astype(int)
        levels = [abs(fft_result[index]) for index in indices]
        if(self.debug):
            for frequency, level in zip(frequencies, 20 * np.log10(levels)):
                print(f"Frequency: {frequency:.2f} Hz, Level: {level:.2f}")
        points = [(frequency, 20 * np.log10(level)-80) for frequency, level in zip(frequencies, levels)]

        t = Trace(50,points)
        result.addTrace(t)
        return result

    def get_audio_devices(self):


        for device in self.devices:
            if device['max_input_channels'] > 0:
                device_name = device['name']
                self.device_list.append(f"{device_name}")

        return self.device_list

    def is_sample_rate_supported(self,device_index, sample_rate):
        try:
            with sd.InputStream(device=device_index, channels=1, samplerate=sample_rate):
                pass
            return True
        except sd.PortAudioError:
            return False

    def get_supported_sample_rates(self,device_name):
        supported_rates = []

        for device_index, device in enumerate(self.devices):
            if device['name'] == device_name:
                supported_rates = [44100, 48000, 96000, 192000]  # Add other sample rates to this list if needed
                supported_rates = [rate for rate in supported_rates if self.is_sample_rate_supported(device_index, rate)]
                break

        return supported_rates

    def get_default_samplerate(self,device_name):
        for device in self.devices:
            if device['name'] == device_name:
                return device['default_samplerate']
    

def config_audio_analyser(config,audio_analyser):

    layout = [
        [sg.Text('Audio Source:'), sg.InputCombo(audio_analyser.get_audio_devices(), key='-AUDIO-', default_value=config.audioSource, enable_events=True)],
        [sg.Text('Supported Sample Rates:'), sg.InputCombo(audio_analyser.get_supported_sample_rates(config.audioSource), key='-SAMPLERATE-', size=(20, 10), default_value=audio_analyser.get_default_samplerate(config.audioSource), enable_events=True)],
        [sg.Button('OK')]
    ]

    window = sg.Window('Config', layout)

    while True:
        event, values = window.read()

        if event == sg.WIN_CLOSED or event == 'OK':
            config.audioSource = values['-AUDIO-']
            config.sample_rate = values['-SAMPLERATE-']
            audio_analyser.sample_rate=int(config.sample_rate)
            config.save_to_file('.config_phase_noise.json')
            break
        elif event == '-AUDIO-':  # Device selection event
            print("audio")
            selected_device = values['-AUDIO-']
            sample_rates = audio_analyser.get_supported_sample_rates(selected_device)
            print(sample_rates)
            window['-SAMPLERATE-'].update(values=sample_rates)

    window.close()

def config_rf_analyser(config):
    layout = [
        [sg.Text('Analyser Address:'), sg.InputCombo(list(range(1, 33)), default_value=config.analyserAddress, key='-NUMBER-')],
        [sg.Button('OK')]
    ]

    window = sg.Window('RF Analyser Configuration', layout, modal=True)

    while True:
        event, values = window.read()

        if event == sg.WIN_CLOSED or event == 'OK':
            config.analyserAddress = int(values['-NUMBER-'])
            config.save_to_file('.config_phase_noise.json')
            break

    window.close()
    return values['-NUMBER-']
    

config = Config.load_from_file('.config_phase_noise.json')
print(vars(config))

graphTop=-40
graphBottom = -150

# Define the layout of the window
layout = [
    [sg.Menu([['File', ['Open File']],
        ['Config', ['Configure Audio Analyser', 'Configure RF Analyser']]])],  # Add a menu with 'Open File' option
    [sg.Graph(canvas_size=(800, 600), graph_bottom_left=(0, graphBottom-5), graph_top_right=(415, graphTop+5), background_color='white', key='-GRAPH-')],
    [sg.Button('Acquire Data'),sg.Button('Save Data', disabled=True), sg.Button('Audio Data')]   
]

# Create the window
window = sg.Window('Phase Noise Plot', layout)

# Get the graph element
graph = window['-GRAPH-']
window.Finalize()

# Clear the graph
graph.erase()

# Draw horizontal grid lines
for y in range(graphTop, graphBottom-1, -10):
    graph.draw_line((15, y), (415, y), color='lightgray')
    graph.draw_text(y, (10, y), font='Any 8')

# Draw vertical grid lines
m=10
mul=["Hz","0Hz","00Hz","KHz","0KHz","00KHz","M","0MHz"]
for c in range(6):
    for f in range(1,10):
        x=map_to_log_scale(f*m, 10, 10000000, 0, 400)
        graph.draw_line((x, graphTop), (x, graphBottom), color='lightgray')
        if (f==1):
            i = int(np.log10(m))
            graph.draw_text(" %d%s" % (f,mul[i]), (x+6, graphBottom-3), font='Any 8')
        else:
            graph.draw_text(f, (x, graphBottom-3), font='Any 8')
    m *= 10
    
window.refresh()

audio_analyser = AudioAnalyser(config)

# Define a list of colors
colours = ['blue', 'red', 'green', 'orange']

# Initialize a counter
colour_counter = 0

# Event loop
while True:
    event, values = window.read()

    if event == sg.WIN_CLOSED or event == 'Exit':
        break

    if event == 'Open File':  # Check if 'Open File' was selected
        filepath = sg.popup_get_file('Open File', no_window=True)  # Show file dialog
        if filepath:
            print(f'Selected file: {filepath}')  # Do something with the selected file path
            resultf = ResultSet(1E3,10E6)
            resultf.load(filepath)
            plotPoints(graph,resultf,colours[colour_counter])
            colour_counter += 1
            if colour_counter >= len(colours):
                colour_counter = 0

    if event == 'Acquire Data':  
        print("Acquiring data!")
        progress_popup = None  # Initialize progress_popup variable
        try:
            # Display in progress message
            progress_popup = sg.Window('', [[sg.Text('Acquiring RF Data...', text_color='white')]], background_color='black', no_titlebar=True, keep_on_top=True)
            progress_popup.finalize()  # Finalize the window
            progress_popup.refresh()

            result = acquirePlot(1E3, 10E6, 1, config.analyserAddress)
            plotPoints(graph, result, colours[colour_counter])

            colour_counter += 1
            if colour_counter >= len(colours):
                colour_counter = 0

            window['Save Data'].update(disabled=False) 
            window.refresh()  # Refresh the window to update the display

            progress_popup.close()  # Close progress popup after completion
        except Exception as e:
            progress_popup.close()  # Close progress popup in case of an error
            sg.popup_error(f"An error occurred: {e}")
        finally:
            # Close progress popup if it's open
            if progress_popup is not None:
                progress_popup.close()


    if event == 'Audio Data':  
        print("Audio data!")
        progress_popup = None  # Initialize progress_popup variable
        
        try:
            # Display in progress message
            progress_popup = sg.Window('', [[sg.Text('Acquiring Audio Data...', text_color='white')]],background_color='black', no_titlebar=True, keep_on_top=True)
            progress_popup.finalize()  # Finalize the window
            progress_popup.refresh()
            
            result = audio_analyser.acquire()
            
            plotPoints(graph, result, colours[colour_counter])
            
            colour_counter += 1
            if colour_counter >= len(colours):
                colour_counter = 0
        except Exception as e:
            # pop up an error message
            sg.popup_error(f"An error occurred: {e}")
        finally:
            # Close progress popup if it's open
            if progress_popup is not None:
                progress_popup.close()

    if event == 'Configure Audio Analyser':
        # Handle audio input configuration
        config_audio_analyser(config,audio_analyser);


    if event == 'Configure RF Analyser':
        config_rf_analyser(config)
        # Handle RF analyser configuration
            
    if event == 'Save Data':  # Check if 'Save Data' button was clicked
        filepath = sg.popup_get_file('Save Data', save_as=True, no_window=True, default_extension='.pnp', file_types=(("Phase Noise Files", "*.pnp"), ("All Files", "*.*")))
        if filepath:
            print(f'Selected file: {filepath}')
            # Handle saving data here
            result.save(filepath)

# Close the window
window.close()

        














