# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 13:40:25 2024

@author: Ryan.Larson
"""

import pandas as pd
import numpy as np
from tkinter import Tk
from tkinter import filedialog as fd
from pathlib import Path
import os
import re
from datetime import datetime
from sklearn.linear_model import LinearRegression
from scipy.interpolate import interp1d

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
            # elif ".rsl" in line:
            #     reading_data = False
            #     # reading_metadata = True
            elif "Statistics" in line:
                reading_data = False
                # reading_metadata = False
            elif line == "\n":
                reading_data = False
            
            # If reading the data part, collect the rows
            if reading_data:
                data_lines.append(line.strip().split("\t"))
            # elif reading_metadata:
            #     if ".rsl" in line:
            #         metadata["Filename"] = line
            #     else:
            #         key, value = line.strip().split(":", 1)
            #         metadata[key.strip()] = value.strip()
            # else:
            #     # Extract metadata key-value pairs before the table starts
            #     if ":" in line:
            #         key, value = line.strip().split(":", 1)
            #         metadata[key.strip()] = value.strip()
            #     elif ".log" in line:
            #         metadata["Filename"] = line

                    
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
            
def process_tensile_data_directory():
    root = Tk()
    root.wm_attributes('-topmost', 1)
    directory = Path(fd.askdirectory(title="Select Data Directory"))
    root.destroy()
    
    files = os.listdir(directory)
    data = {"File": files}
    df = pd.DataFrame(data)
    
    df["Filepath"] = [directory / file for file in df["File"]]
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
            print(f'\nSpecimen {specimen}:')
            print(f'Thickness: {specimen_thickness}')
            print(f'Width: {specimen_width}')
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
            data_filepath = directory / data_filename
            
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
                
    

if __name__ == "__main__":
    process_tensile_data_directory()