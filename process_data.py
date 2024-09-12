# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 13:40:25 2024

@author: Ryan.Larson
"""

import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import Tk
from tkinter import filedialog as fd
from tkinter import StringVar
from pathlib import Path
import os
import re
from datetime import datetime
from sklearn.linear_model import LinearRegression
from scipy.interpolate import interp1d
from threading import Thread
from pyupdater.client import Client
from update_config import ClientConfig

APP_NAME = "Mark 10 Post-Processor"
APP_VERSION = "1.0.0"

def check_for_updates():
    client = Client(ClientConfig(), refresh=True)  # ClientConfig is the configuration class for PyUpdater
    app_update = client.update_check(APP_NAME, APP_VERSION)  # Define your app name and version here
    
    if app_update is not None:
        print("Update available! Downloading now...")
        app_update.download()
        if app_update.is_downloaded():
            app_update.extract_restart()  # Restarts the app after the update

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

def extract_datetime_string(filename):
    # Regular expression to match the date and time in the filename
    pattern = r"([A-Za-z]{3})-(\d{1,2})-(\d{4})-(\d{2})-(\d{2})-(\d{2})-([A-Za-z]{2})"
    
    # Month abbreviations mapping to numbers for datetime parsing
    month_map = { 'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                  'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12 }
    
    match = re.search(pattern, str(filename))
    if match:
        return match.group(0)
    else:
        return None
    
def find_results_file(datetime_string, filepaths_column):
    matches = []
    for file in filepaths_column:
        if (datetime_string in file) and (".rsl" in file):
            matches.append(file)
    
    if len(matches) == 1:
        return matches[0]
    else:
        return None
    
def read_rsl_file(filepath):
    # metadata = {}
    data_lines = []
    reading_data = False
    # reading_metadata = False
    
    with open(filepath, 'r') as file:
        for line in file:
            # Detect the start of the data table by looking for the "Reading" header
            if "Run No." in line:
                reading_data = True
                # reading_metadata = False
                headers = line.strip().split("\t")  # Capture the table headers
                continue
            elif "Statistics" in line:
                reading_data = False
                # reading_metadata = False
            elif line == "\n":
                reading_data = False
            
            # If reading the data part, collect the rows
            if reading_data:
                data_lines.append(line.strip().split("\t"))

                    
    # Create DataFrame from the data table portion
    df = pd.DataFrame(data_lines, columns=headers)
    
    # Convert columns to the correct data types
    dtype_dict = {
        'Run No.': int,
        'Status': str,
        'Specimen Number': str,
        'Specimen Thickness': float,
        'Specimen Width': float,
        'Date': str,
        'Time': str,
        'Speed (mm/min)': float,
        'Final Load (N)': float,
        'Final Distance (mm)': float,
        'Max Load (N)': float,
        'LD at Max Dist (N)': float,
        'Max Distance (mm)': float,
        'Dist at Max Load (mm)': float,
        'Area Under Curve (N*mm)': float
        }
    
    df = df.astype(dtype_dict)
    
    return df

def find_matching_specimen(datetime_substring, df):
    # Convert datetime_substring to date and time data
    month_map = {'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04', 'May': '05', 'Jun': '06',
             'Jul': '07', 'Aug': '08', 'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'}
    pattern = r"([A-Za-z]{3})-(\d{1,2})-(\d{4})-(\d{2})-(\d{2})-(\d{2})-([A-Za-z]{2})"
    
    match = re.search(pattern, datetime_substring)
    if match:
        # Extract the components from the regex match
        month_str, day, year, hour, minute, second, am_pm = match.groups()
        
        # Convert the month abbreviation to numeric format
        month = month_map[month_str]
        
        # Format the date as MM/DD/YYYY
        date = f"{month_str} {int(day)}, {year}"
        
        # Format the time as HH:MM:SS AM/PM
        time = f"{int(hour):02}:{minute}:{second} {am_pm.upper()}"
    else:
        pattern = r"([A-Za-z]{3})-(\d{1,2})-(\d{4})-(\d{2})-(\d{2})-(\d{2}))"
        match = re.search(pattern, datetime_substring)
        if match:
            # Extract the components from the regex match
            month_str, day, year, hour, minute, second, am_pm = match.groups()
            
            # Convert the month abbreviation to numeric format
            month = month_map[month_str]
            
            # Format the date as MM/DD/YYYY
            date = f"{month_str} {int(day)}, {year}"
            
            # Format the time as HH:MM:SS AM/PM
            time = f"{int(hour):02}:{minute}:{second} {am_pm.upper()}"
        else:
            date = None
            time = None
    
    # Read results files and find a good run that contains the matching date
    # and time information
    specimen = None
    specimen_thickness = None
    specimen_width = None
    for filepath in df[df["Results"]==True]["Filepath"]:
        df_results = read_rsl_file(filepath)
        
        for i in range(len(df_results)):
            if (df_results.loc[i,"Date"] == date) and (df_results.loc[i,"Time"] == time) and (df_results.loc[i,"Status"] == "Complete"):
                specimen = df_results.loc[i,"Specimen Number"]
                specimen_thickness = df_results.loc[i,"Specimen Thickness"]
                specimen_width = df_results.loc[i,"Specimen Width"]
                break
        if specimen and specimen_thickness and specimen_width:
            specimen_thickness = float(specimen_thickness)
            specimen_width = float(specimen_width)
            break
    if (specimen is None) and (specimen_thickness is None) and (specimen_width is None):
        print("Specimen details not detected")
    return specimen, specimen_thickness, specimen_width
            
def process_tensile_data_directory(directory):    
    # files = os.listdir(Path(directory))
    files = [f for f in os.listdir(Path(directory)) if os.path.isfile(os.path.join(directory, f))]
    data = {"File": files}
    df = pd.DataFrame(data)
    
    df["Filepath"] = [(directory + "/" + file) for file in df["File"]]
    # df["Filepath"] = [directory / file for file in df["File"]]
    df["Data"] = [True if (".log" in file) else False for file in df["File"]]
    df["Results"] = [True if (".rsl" in file) else False for file in df["File"]]
    
    # Load the data files only
    for filepath in df[df["Data"]==True]["Filepath"]:
        try:
            # Get the matching datetime from the file name
            dt = extract_datetime_string(filepath)
            
            # Step through the results files and get the specimen parameters
            specimen, specimen_thickness, specimen_width = find_matching_specimen(dt, df)
            
            # Find the matching log file and search it for the 
            # Extract the force-displacement data from the data file
            metadata, dfdata = read_log_file(filepath)
            
            # Process the tensile data into stress and strain
            A = specimen_thickness * specimen_width     # cross-sectional area, mm^2
            # print(f'\nSpecimen {specimen}:')
            # print(f'Thickness: {specimen_thickness}')
            # print(f'Width: {specimen_width}')
            gauge_length = 115.0
            dfdata['Stress (MPa)'] = -dfdata['Load [N]'] / A
            dfdata['Strain'] = dfdata['Distance [mm]'] / gauge_length
            
            # Filter out strain data so only strictly increasing strain is included
            dfdata['diff'] = dfdata['Strain'].diff()
            mask = dfdata['diff'] > 0
            dfdata = dfdata[mask]
            dfdata = dfdata.drop(columns=['diff'])
            
            # Save dfdata with the specimen name
            data_filename = 'Processed Test Data/' + specimen + '.csv'
            data_filepath = directory + "/" + data_filename
            
            # Add more data at the beginning of the output CSV file
            # Ultimate Tensile Strength (MPa)
            uts = np.max(dfdata['Stress (MPa)'])
            
            # Chord method for Young's Modulus
            interp_func = interp1d(dfdata['Strain'], dfdata['Stress (MPa)'], kind='linear')
            sigma0005 = interp_func(0.0005)
            sigma0025 = interp_func(0.0025)
            Et_chord = (sigma0025-sigma0005)/(0.0025 - 0.0005)
            
            # Linear regression method for Young's Modulus
            filtered_df = dfdata[(dfdata['Strain'] >= 0.0005) & (dfdata['Strain'] <= 0.0025)]
            X = filtered_df[['Strain']]
            y = filtered_df['Stress (MPa)']
            model = LinearRegression().fit(X, y)
            Et_regr = model.coef_[0]
            
            specimen_info = {
                'Specimen Code': specimen,
                'Specimen Thickness': specimen_thickness,
                'Specimen Width': specimen_width,
                'Ultimate Tensile Strength (MPa)': uts,
                'Modulus of Elasticity - Chord': Et_chord,
                'Modulus of Elasticity - Regression': Et_regr}
            
            # Write the specimen information to the file
            with open(data_filepath, 'w') as f:
                for key, value in specimen_info.items():
                    f.write(f'{key},{value}\n')     # Write each key-value pair as a new line
                    
                f.write('\n')   # Add an empty line between metadata and DataFrame content
                
            dfdata.to_csv(data_filepath, mode='a', index=False)
            
            
        except Exception as e:
            print(f"Error: {e}")
            
    tensile_message.set("Tensile data processed successfully")
    root.after(10000, clear_tensile_message)
                

# def process_tensile_data(tensile_dir):
#     # Dummy function to simulate processing tensile data
#     tensile_message.set("Tensile data processed successfully")
#     root.after(10000, clear_tensile_message)  # Clear message after 10 seconds

def process_flexural_data(flexural_dir):
    # Dummy function to simulate processing flexural data
    flexural_message.set("Flexural data processed successfully")
    root.after(10000, clear_flexural_message)  # Clear message after 10 seconds

def clear_tensile_message():
    tensile_message.set("")

def clear_flexural_message():
    flexural_message.set("")

def select_tensile_directory():
    root = Tk()
    root.withdraw()  # Hide the main window
    directory = fd.askdirectory(title="Select the Tensile Data Directory")
    root.destroy()
    tensile_directory.set(directory)
    
def select_flexural_directory():
    root = Tk()
    root.withdraw()  # Hide the main window
    directory = fd.askdirectory(title="Select the Flexural Data Directory")
    root.destroy()
    flexural_directory.set(directory)
    
if __name__ == "__main__":
    check_for_updates()
    
    # Main application window
    root = tk.Tk()
    root.attributes("-topmost", True)
    root.title("Mark-10 Data Processing")
    
    # Variables for directory paths and messages
    tensile_directory = StringVar(value="G:\Shared drives\RockWell Shared\Engineering\Engineering Projects\DLFT\DLFT Testing\Production Testing\Tensile Tests")
    flexural_directory = StringVar(value="G:\Shared drives\RockWell Shared\Engineering\Engineering Projects\DLFT\DLFT Testing\Production Testing\Flexural Tests")
    tensile_message = StringVar(value="")
    flexural_message = StringVar(value="")
    
    # Buttons and labels for selecting directories
    btn_select_tensile = tk.Button(root, text="Select Tensile Directory", command=lambda: select_tensile_directory())
    btn_select_flexural = tk.Button(root, text="Select Flexural Directory", command=lambda: select_flexural_directory())
    lbl_tensile_dir = tk.Label(root, textvariable=tensile_directory)
    lbl_flexural_dir = tk.Label(root, textvariable=flexural_directory)
    
    # Buttons for processing data
    btn_process_tensile = tk.Button(root, text="Process Tensile Data",
                                    command=lambda: process_tensile_data_directory(tensile_directory.get()))
    btn_process_flexural = tk.Button(root, text="Process Flexural Data",
                                     command=lambda: process_flexural_data(flexural_directory.get()))
    
    # Labels for processing messages
    lbl_tensile_message = tk.Label(root, textvariable=tensile_message)
    lbl_flexural_message = tk.Label(root, textvariable=flexural_message)
    
    # Layout for the first set (Select buttons and directory labels)
    btn_select_tensile.grid(row=0, column=0, padx=10, pady=5, sticky="w")
    lbl_tensile_dir.grid(row=0, column=1, padx=10, pady=5, sticky="w")
    btn_select_flexural.grid(row=1, column=0, padx=10, pady=5, sticky="w")
    lbl_flexural_dir.grid(row=1, column=1, padx=10, pady=5, sticky="w")
    
    # Adding some visual separation between the two sets of buttons
    separator = tk.Frame(root, height=2, bd=1, relief="sunken")
    separator.grid(row=2, columnspan=2, pady=10, padx=10, sticky="ew")
    
    # Layout for the second set (Process buttons and messages)
    btn_process_tensile.grid(row=3, column=0, padx=10, pady=5, sticky="w")
    lbl_tensile_message.grid(row=3, column=1, padx=10, pady=5, sticky="w")
    btn_process_flexural.grid(row=4, column=0, padx=10, pady=5, sticky="w")
    lbl_flexural_message.grid(row=4, column=1, padx=10, pady=5, sticky="w")
    
    # Run the application
    root.mainloop()