import time
import math
import re
import sys
import os
from datetime import datetime
import numpy as np
from scipy.signal import find_peaks
from scipy.fft import fft

from modules.AudioAnalyser import AudioAnalyser
from modules.HP8590 import HPSA
from modules.Data import ResultSet
from modules.Data import Trace
from modules.Config import Config

import PySimpleGUI as sg


def map_to_log_scale(value, lower_bound, upper_bound, output_lower_bound, output_upper_bound):
    # Ensure the value is within the specified range
    value = max(min(value, upper_bound), lower_bound)
    
    # Calculate the logarithmic mapping
    log_lower = math.log10(lower_bound)
    log_upper = math.log10(upper_bound)
    log_value = math.log10(value)
    
    mapped_value = ((log_value - log_lower) / (log_upper - log_lower)) * (output_upper_bound - output_lower_bound) + output_lower_bound
    return mapped_value+15

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
    minF=config.start_freq
    for trace in result.getTraces():
        for (f, level) in trace.getPoints():
            if(f>minF):
                minF=f
                data_points.append((f, level))
    resampled_data = log_interpolation(data_points, num_points=2000)
    smoothed = triangular_window(resampled_data,12)
    
    for (f,level) in data_points:
        graph.draw_point((map_to_log_scale(f, config.start_freq, config.end_freq, 0, 400), level), color='light grey', size=1)

    for i in range(len(smoothed) - 1):
        x1 = map_to_log_scale(smoothed[i][0], config.start_freq, config.end_freq, 0, 400)
        y1 = smoothed[i][1]
        x2 = map_to_log_scale(smoothed[i+1][0], config.start_freq, config.end_freq, 0, 400)
        y2 = smoothed[i+1][1]

        graph.draw_line((x1, y1), (x2, y2), color=color, width=1)
        
    window.refresh()       

def is_valid_digit_input(input_string):
    return input_string.isdigit() and 1 <= int(input_string) <= 100 or input_string == ""

def config_audio_analyser(config, audio_analyser):
    layout = [
        [sg.Text('Audio Source:'), sg.InputCombo(audio_analyser.get_audio_devices(), key='-AUDIO-', default_value=config.audioSource, enable_events=True)],
        [sg.Text('Supported Sample Rates:'), sg.InputCombo(audio_analyser.get_supported_sample_rates(config.audioSource), key='-SAMPLERATE-', size=(20, 10), default_value=str(config.sample_rate), enable_events=True)],  # Modified default_value
        [sg.Text('Capture Time (in seconds):'), sg.InputText(key='-CAPTURETIME-', size=(3, 1), default_text=str(config.capture_time))],
        [sg.Button('OK')]
    ]

    window = sg.Window('Config', layout)

    while True:
        event, values = window.read()

        if event == sg.WIN_CLOSED or event == 'OK':
            config.audioSource = values['-AUDIO-']
            config.sample_rate = int(values['-SAMPLERATE-'])
            
            # Check if the input is within the specified range (1 to 100)
            input_value = int(values['-CAPTURETIME-'])
            if 1 <= input_value <= 100:
                config.capture_time = input_value
                
            audio_analyser.sample_rate = int(config.sample_rate)
            config.save_to_file('.config_phase_noise.json')
            break
        elif event == '-AUDIO-':
            selected_device = values['-AUDIO-']
            sample_rates = audio_analyser.get_supported_sample_rates(selected_device)
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

def config_system(config):
    layout = [
        [sg.Text('System Mode')],
        [sg.Radio('RF with Spectrum Analyser', 'MODE', default=(config.mode == 'RF'), key='-RF-')],
        [sg.Radio('PLL with Spectrum Analyser', 'MODE', default=(config.mode == 'PLL-RF'), key='-PLL_SPECTRUM-')],
        [sg.Radio('PLL with Audio Analyser', 'MODE', default=(config.mode == 'PLL-AUDIO'), key='-PLL_AUDIO-')],
        [sg.Radio('PLL with Audio and RF Spectrum Analyser', 'MODE', default=(config.mode == 'PLL-AUDIO-RF'), key='-PLL_AUDIO_RF-')],
        [sg.Button('OK')]
    ]

    window = sg.Window('System Mode', layout)

    while True:
        event, values = window.read()

        if event == 'OK':
            if values['-RF-']:
                config.mode = 'RF'
                config.start_freq=1E3
                config.end_freq=10E6
            elif values['-PLL_SPECTRUM-']:
                config.mode = 'PLL-RF'
                config.start_freq=10E3
                config.end_freq=10E6
            elif values['-PLL_AUDIO-']:
                config.mode = 'PLL-AUDIO'
                config.start_freq=10
                config.end_freq=20E3
            elif values['-PLL_AUDIO_RF-']:
                config.mode = 'PLL-AUDIO-RF'
                config.start_freq=10
                config.end_freq=10E6
            config.save_to_file()
            window.close()
            return

def draw_grid(graph,config):
    graph.erase()
    # Draw horizontal grid lines
    for y in range(graphTop, graphBottom-1, -10):
        graph.draw_line((15, y), (415, y), color='lightgray')
        graph.draw_text(y, (10, y), font='Any 8')

    # Draw vertical grid lines
    m=config.start_freq
    mul=["Hz","0Hz","00Hz","KHz","0KHz","00KHz","M","0MHz"]
    for c in range(int(math.log10(config.end_freq)) - int(math.log10(config.start_freq))):
        for f in range(1,10):
            x=map_to_log_scale(f*m, config.start_freq, config.end_freq, 0, 400)
            graph.draw_line((x, graphTop), (x, graphBottom), color='lightgray')
            if (f==1):
                i = int(np.log10(m))
                graph.draw_text(" %d%s" % (f,mul[i]), (x+6, graphBottom-3), font='Any 8')
            else:
                graph.draw_text(f, (x, graphBottom-3), font='Any 8')
        m *= 10
        


config = Config.load_from_file('.config_phase_noise.json')
print(vars(config))
graphTop=-40
graphBottom = -150

# Define the layout of the window
layout = [
    [sg.Menu([['File', ['Open File']],
        ['Config', ['Configure System', 'Configure Audio Analyser', 'Configure RF Analyser']]])],  # Add a menu with 'Open File' option
    [sg.Graph(canvas_size=(800, 600), graph_bottom_left=(0, graphBottom-5), graph_top_right=(415, graphTop+5), background_color='white', enable_events=True, key='-GRAPH-')],
    [sg.Button('Acquire Data'),sg.Button('Save Data', disabled=True)]   
]

# Create the window
window = sg.Window('Phase Noise Plot', layout)

# Get the graph element
graph = window['-GRAPH-']
window.Finalize()

# Clear the graph
draw_grid(graph,config)
window.refresh()

audio_analyser = AudioAnalyser(config)
rf_analyser=HPSA(config.analyserAddress)

# Define a list of colors
colours = ['blue', 'red', 'green', 'orange']

# Initialize a counter
colour_counter = 0
prev_cursor =None

# Event loop
while True:
    event, values = window.read()

    if event == '-GRAPH-':
        print("graph")
        mouse_x, mouse_y = values['-GRAPH-']
        print(mouse_x)

        #if prev_cursor is not None:
            #graph.delete_figure(prev_cursor)  # Remove previous cursor

        prev_cursor = graph.draw_line((mouse_x, 0), (mouse_x, 400), color='red')  # Draw new cursor
        window.refresh();


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
        progress_popup = None  # Initialize progress_popup variable
        try:
            # Display in progress message

            if(config.mode=='RF'):
                progress_popup = sg.Window('', [[sg.Text('Acquiring RF Data...', text_color='white')]], background_color='black', no_titlebar=True, keep_on_top=True)
                progress_popup.finalize()  # Finalize the window
                progress_popup.refresh()
                result = rf_analyser.acquire(config.start_freq, config.end_freq, 1)
                progress_popup.close()  # Close progress popup after completion
                plotPoints(graph, result, colours[colour_counter])
            if(config.mode=='PLL-AUDIO'):
                progress_popup = sg.Window('', [[sg.Text('Acquiring Audio Data...', text_color='white')]], background_color='black', no_titlebar=True, keep_on_top=True)
                progress_popup.finalize()  # Finalize the window
                progress_popup.refresh()
                result = audio_analyser.acquire(config.start_freq, config.changeover, config.capture_time)
                progress_popup.close()  # Close progress popup after completion
                plotPoints(graph, result, colours[colour_counter])
            if(config.mode=='PLL-RF'):
                progress_popup = sg.Window('', [[sg.Text('Acquiring RF Baseband Data...', text_color='white')]], background_color='black', no_titlebar=True, keep_on_top=True)
                progress_popup.finalize()  # Finalize the window
                progress_popup.refresh()
                result = rf_analyser.acquire_baseband(config.changeover, config.end_freq, 1)
                progress_popup.close()  # Close progress popup after completion
                plotPoints(graph, result, colours[colour_counter])
            if(config.mode=='PLL-AUDIO-RF'):
                progress_popup = sg.Window('', [[sg.Text('Acquiring Audio Data...', text_color='white')]], background_color='black', no_titlebar=True, keep_on_top=True)
                progress_popup.finalize()  # Finalize the window
                progress_popup.refresh()
                result = audio_analyser.acquire(config.start_freq, config.changeover, config.capture_time)
                progress_popup.close()  # Close progress popup after completion
                plotPoints(graph, result, colours[colour_counter])
                progress_popup = sg.Window('', [[sg.Text('Acquiring Baseband RF Data...', text_color='white')]], background_color='black', no_titlebar=True, keep_on_top=True)
                progress_popup.finalize()  # Finalize the window
                progress_popup.refresh()
                result = rf_analyser.acquire_baseband(config.changeover, config.end_freq, 1)
                progress_popup.close()  # Close progress popup after completion
                plotPoints(graph, result, colours[colour_counter])
            else:
                print("Mode not configured")

            colour_counter += 1
            if colour_counter >= len(colours):
                colour_counter = 0

            window['Save Data'].update(disabled=False) 
            window.refresh()  # Refresh the window to update the display

            
        except Exception as e:
            progress_popup.close()  # Close progress popup in case of an error
            sg.popup_error(f"An error occurred: {e}")
        finally:
            # Close progress popup if it's open
            if progress_popup is not None:
                progress_popup.close()
                
    if event == 'Configure System':
        # Handle audio input configuration
        mode = config.mode
        config_system(config)
        if(mode != config.mode):
            draw_grid(graph,config)
            

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

        














