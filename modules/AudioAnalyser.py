import time
import math
from datetime import datetime
from typing import List
import numpy as np
from scipy.signal import find_peaks
from scipy.fft import fft
from scipy.ndimage import gaussian_filter1d
from modules.Data import ResultSet
from modules.Data import Trace

import sounddevice as sd

class AudioAnalyser:

    def __init__(self,config):
        self.debug=False
        self.sample_rate = config.sample_rate
        self.channels = 2
        self.audio_data = np.array([])
        self.devices = sd.query_devices()
        self.device_list = []

    # Define a callback function to capture audio
    def audio_callback(self,indata, frames, time, status):
        self.audio_data = np.append(self.audio_data, indata[:, 0])  # Assuming mono input

    def hamming_window(self,data):
        window = np.hamming(len(data))
        return data * window
    
    def gaussian_smooth(self, data, sigma):
        return gaussian_filter1d(data, sigma)

    def acquire(self, start_freq, end_freq, duration=5):
        print(f"Acquiring audio data at {self.sample_rate} sample rate for {duration} seconds")

        self.audio_data = np.array([])

        # Start audio stream
        with sd.InputStream(callback=self.audio_callback, channels=self.channels, samplerate=self.sample_rate):
            sd.sleep(duration * 1000)  # Sleep for the specified duration in milliseconds

        # Assuming audio_data contains your acquired samples
        windowed_data = self.hamming_window(self.audio_data)
        fft_result = fft(windowed_data)
        fft_frequencies = [i * self.sample_rate / len(fft_result) for i in range(len(fft_result))]

        # Generate an array of frequencies exactly a decade apart
        num_points = 500
        frequencies = [10 ** (np.log10(start_freq) + i * (np.log10(end_freq) - np.log10(start_freq)) / (num_points - 1)) for i in range(num_points)]

        # Compute energies
        index = 0
        points=[]
        peak=-500
        for resampled_frequency in frequencies:
            energy=0
            while(fft_frequencies[index] < resampled_frequency):
                energy += np.abs(fft_result[index])**2
                index += 1
            if(energy>0):
                lvl = 20 * np.log10(np.sqrt(energy))-97
                points.append((resampled_frequency, lvl))
                if(lvl > peak):
                    peak = lvl
        result=ResultSet(start_freq,end_freq)
        t = Trace(50, points)
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
    

