import time
import math
from datetime import datetime
from typing import List
import numpy as np
from scipy.signal import find_peaks
from scipy.fft import fft
from modules.Data import ResultSet
from modules.Data import Trace

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
    

