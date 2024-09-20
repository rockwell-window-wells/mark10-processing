# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 12:57:34 2024

@author: Ryan.Larson
"""

import numpy as np
from scipy.signal import butter, filtfilt, savgol_filter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

def butter_lowpass_filter(data, cutoff, order=5, padlen=None):
    time = df['Time [s]']
    dt = np.mean(np.diff(time))
    fs = 1 / dt
    
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data, padlen=padlen)
    return y

def apply_savgol_filter(data, window_size, poly_order):
    """
    Applies the Savitzky-Golay filter to smooth the input data.

    Parameters:
    data (array-like): The noisy data to be smoothed.
    window_size (int): The size of the filter window (must be an odd integer).
    poly_order (int): The order of the polynomial used to fit the data.

    Returns:
    smoothed_data (numpy array): The smoothed data.
    """
    # Ensure window_size is odd and greater than poly_order
    if window_size % 2 == 0:
        raise ValueError("Window size must be an odd integer.")
    if poly_order >= window_size:
        raise ValueError("Polynomial order must be less than window size.")
    
    # Apply Savitzky-Golay filter
    smoothed_data = savgol_filter(data, window_size, poly_order)
    
    return smoothed_data

def CUSUM_filter(data):
    initial_baseline = data.iloc[:50].mean()
    deviations = data - initial_baseline
    cusum = deviations.cumsum()
    threshold = 5
    rising_index = cusum[cusum > threshold].index[0]
    return rising_index

def start_filter(load_column, target_value=10, zero_value=0):
    # Find the first occurrence of the target value (e.g., 10 N)
    target_index = load_column[load_column >= target_value].index[0]
    
    # Find the closest previous occurrence of the zero_value (e.g., 0 N)
    zero_indices = load_column[:target_index][load_column[:target_index] == zero_value]
    
    if not zero_indices.empty:
        starting_index = zero_indices.index[-1]  # Get the last occurrence of 0 before the target value
    else:
        starting_index = None  # Handle case if there's no zero before target_value
    
    return starting_index

def read_log_file(filepath):
    metadata = {}
    data_lines = []
    reading_data = False
    
    with open(filepath, 'r') as file:
        for line in file:
            # Detect the start of the data table by looking for the "Reading" header
            if "Reading" in line:
                reading_data = True
                headers = line.strip().split("\t")  # Capture the table headers
                continue
            
            # If reading the data part, collect the rows
            if reading_data:
                data_lines.append(line.strip().split("\t"))
            else:
                # Extract metadata key-value pairs before the table starts
                if ":" in line:
                    key, value = line.strip().split(":", 1)
                    metadata[key.strip()] = value.strip()
                elif ".log" in line:
                    metadata["Filename"] = line

                    
    # Create DataFrame from the data table portion
    df = pd.DataFrame(data_lines, columns=headers)
    
    # Convert relevant columns to numeric types (handle any conversion errors gracefully)
    df = df.apply(pd.to_numeric, errors='coerce')
    
    return metadata, df

def recalculate_distance(df, rate):
    df.reset_index(drop=True, inplace=True)
    recalculated_distance = np.zeros(len(df))
    for i in range(len(df)):
        if i == 0:
            recalculated_distance[i] = df.loc[i,'Distance [mm]']
        else:
            recalculated_distance[i] = (14/60)*(df.loc[i, 'Time [s]'] - df.loc[0,'Time [s]']) + df.loc[0,'Distance [mm]']
    df['Recalculated Distance [mm]'] = recalculated_distance
    
def find_zero_distance(df):
    df.reset_index(drop=True, inplace=True)
    df['Smoothed Slope'] = np.gradient(df['Savitzky-Golay Smoothed Load [N]'], df['Recalculated Distance [mm]'])
    b = df.loc[0,'Savitzky-Golay Smoothed Load [N]'] - df.loc[0,'Smoothed Slope'] * df.loc[0,'Recalculated Distance [mm]']
    x = -b / df.loc[0,'Smoothed Slope']
    
    # Get new force-distance curve
    dfnew = df[['Recalculated Distance [mm]',
                'Savitzky-Golay Smoothed Load [N]']].copy()
    
    # Add a row to the beginning of dfnew for the recalculated distance
    new_row = pd.DataFrame({'Recalculated Distance [mm]': [x],
                            'Savitzky-Golay Smoothed Load [N]': [0.0]})
    
    dfnew = pd.concat([new_row, dfnew], ignore_index=True)
    dfnew.reset_index(drop=True, inplace=True)
    
    return dfnew

# Example usage
# fs = 200.0       # Sampling frequency
cutoff = 0.25     # Desired cutoff frequency in Hz
padlen = 0
# data = np.random.randn(1000)  # Replace with your quantized data
directory = r"G:\Shared drives\RockWell Shared\Engineering\Engineering Projects\DLFT\DLFT Testing\Production Testing\Flexural Tests"
files = [f for f in os.listdir(Path(directory)) if (os.path.isfile(os.path.join(directory, f))) & (".log" in f)]
logfiles = [os.path.join(directory, f) for f in files if ".log" in f]
for i,file in enumerate(logfiles):
    metadata, df = read_log_file(file)
    df['Distance [mm]'] = -df['Distance [mm]']
    # df = df.drop_duplicates(subset='Distance [mm]', keep='first')
    
    # rising_index = CUSUM_filter(df['Load [N]'])
    # df_filtered = df.iloc[rising_index:].copy()
    
    rising_index = start_filter(df['Load [N]'])
    df_filtered = df.iloc[rising_index:].copy()
    
    df_filtered['Butterworth Smoothed Load [N]'] = butter_lowpass_filter(df_filtered['Load [N]'], cutoff, padlen=padlen)
    window_size = 201
    poly_order = 2
    df_filtered['Savitzky-Golay Smoothed Load [N]'] = apply_savgol_filter(df_filtered['Load [N]'], window_size, poly_order)
    # df_filtered['Moving Avg Load [N]'] = df_filtered['Load [N]'].rolling(window=window_size).mean()
    # df['Smoothed Load [N]'] = butter_lowpass_filter(df['Load [N]'], cutoff, padlen=padlen)
    # window_size = 40
    # df['Moving Avg Load [N]'] = df['Load [N]'].rolling(window=window_size).mean()
    
    # # Get gradients and curvature labels for smoothed data
    # df_filtered['First Derivative'] = np.gradient(df_filtered['Smoothed Load [N]'], df_filtered['Distance [mm]'])
    # df_filtered['Second Derivative'] = np.gradient(df_filtered['First Derivative'], df_filtered['Distance [mm]'])
    # df_filtered['Curvature'] = np.where(df_filtered['Second Derivative'] > 0, 'Positive', 'Negative')
    
    # Get new force-displacement
    recalculate_distance(df_filtered, 14.0)
    dfnew = find_zero_distance(df_filtered)
    dfnew['Recalculated Distance [mm]'] = dfnew['Recalculated Distance [mm]'] - dfnew.loc[0,'Recalculated Distance [mm]']
    
    
    plt.figure(figsize=(8,6), dpi=300)
    sns.set_palette("bright")
    sns.lineplot(data=df_filtered, x='Distance [mm]', y='Load [N]', label='Raw Data', errorbar=None)
    # sns.lineplot(data=df_filtered, x='Distance [mm]', y='Butterworth Smoothed Load [N]', label='Butterworth', errorbar=None)
    sns.lineplot(data=df_filtered, x='Distance [mm]', y='Savitzky-Golay Smoothed Load [N]', label='Savitsky-Golay', errorbar=None)
    sns.lineplot(data=dfnew, x='Recalculated Distance [mm]', y='Savitzky-Golay Smoothed Load [N]', label='Smoothed Shifted', errorbar=None)
    # sns.lineplot(data=df, x='Distance [mm]', y='Moving Avg Load [N]', label=f'Moving Avg: {window_size}', errorbar=None)
    # sns.lineplot(data=df, x='Distance [mm]', y='Load [N]', label='Raw Data', errorbar=None)
    # sns.lineplot(data=df, x='Distance [mm]', y='Smoothed Load [N]', label='Smoothed Data', errorbar=None)
    # # sns.lineplot(data=df, x='Distance [mm]', y='Moving Avg Load [N]', label=f'Moving Avg: {window_size}', errorbar=None)
    plt.xlabel('Distance [mm]')
    plt.ylabel('Load [N]')
    plt.legend()
    plt.title(f'{files[i]}')
    plt.show()
    plt.close()
