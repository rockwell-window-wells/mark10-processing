# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 13:40:25 2024

@author: Ryan.Larson
"""

import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import Tk
from tkinter import ttk
from tkinter import filedialog as fd
from tkinter import StringVar
from pathlib import Path
import os
import re
from sklearn.linear_model import LinearRegression
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from datetime import datetime, timedelta
from threading import Thread
import logging
from logging.handlers import QueueHandler, QueueListener
import queue

# # log_queue = queue.Queue()

# # def setup_logger():
# #     """
# #     Configures the logging system to use a queue handler.
# #     """
# #     logger = logging.getLogger("ThreadSafeLogger")
# #     logger.setLevel(logging.DEBUG)

# #     # Create a handler that processes messages from the log queue
# #     handler = logging.StreamHandler()
# #     formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# #     handler.setFormatter(formatter)
# #     logger.addHandler(handler)

# #     return logger

# def setup_logger(log_queue):
#     """
#     Configures the logging system to use a queue handler and file-based logging.
#     """
#     # Determine the current datetime for the log filename
#     start_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#     log_filename = f"logs/{start_datetime}.log"

#     # Create a logger
#     logger = logging.getLogger("ThreadSafeLogger")
#     logger.setLevel(logging.INFO)
    
#     # Remove existing handlers to avoid duplication
#     if logger.hasHandlers():
#         logger.handlers.clear()

#     # Create a file handler for logging to a file
#     file_handler = logging.FileHandler(log_filename, mode="a", encoding="utf-8")
#     file_handler.setFormatter(logging.Formatter(
#         "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
#     ))

#     # Create a QueueHandler for thread-safe logging
#     queue_handler = QueueHandler(log_queue)

#     # Add handlers to the logger
#     logger.addHandler(file_handler)
#     logger.addHandler(queue_handler)

#     return logger

# # logger = setup_logger(log_queue)

# def log_worker(log_queue):
#     """
#     Worker thread to process log messages from the queue.
#     """
#     while True:
#         try:
#             record = log_queue.get()
#             if record is None:  # Exit signal
#                 break
#             logger = logging.getLogger(record.name)
            
#             # Only process the record if it's a valid logger object
#             if isinstance(logger, logging.Logger):
#                 # Handle the log record
#                 logger.handle(record)
#                 logger.handlers[0].flush()
#             else:
#                 print(f"Invalid logger object: {logger}")
#             # logger.handle(record)
#             # logger.handlers[0].flush()
#         except Exception:
#             import traceback
#             print("Exception in log_worker:", traceback.format_exc())
#             break

# # Start the logging worker thread
# log_thread = Thread(target=log_worker, args=(log_queue,))
# log_thread.daemon = True
# log_thread.start()

# def thread_safe_log(logger_name, level, msg):
#     """
#     Logs a message in a thread-safe manner by adding it to the queue.
#     """
#     logger = logging.getLogger(logger_name)  # Get the correct logger by name
#     record = logger.makeRecord(logger_name, level, None, None, msg, None, None)
#     log_queue.put(record)  # Put the log record into the queue
#     log_queue.put(None)  # Signal to process the log immediately
#     # # logger = logging.getLogger(logger_name)
#     # # record = logger.makeRecord(
#     # #     logger_name, level, fn="", lno=0, msg=msg, args=(), exc_info=None
#     # # )
#     # # log_queue.put(record)
#     # # log_queue.put(None)
#     # log_queue.put(logger.makeRecord(
#     #     logger.name, level, None, None, msg, None, None))
#     # log_queue.put(None)  # Signal to process the log immediately

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

def recalculate_distance(df, rate):
    df.reset_index(drop=True, inplace=True)
    recalculated_distance = np.zeros(len(df))
    for i in range(len(df)):
        if i == 0:
            recalculated_distance[i] = df.loc[i,'Distance [mm]']
        else:
            recalculated_distance[i] = (14./60.)*(df.loc[i, 'Time [s]'] - df.loc[0,'Time [s]']) + df.loc[0,'Distance [mm]']
    df['Recalculated Distance [mm]'] = recalculated_distance
    
def find_zero_distance(df):
    # df.reset_index(drop=True, inplace=True)
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
    
    match = re.search(pattern, str(filename))
    if match:
        return match.group(0)
    else:
        pattern = r"([A-Za-z]{3})-(\d{1,2})-(\d{4})-(\d{2})-(\d{2})-(\d{2})"
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
    def detect_delimiter(filepath):
        """
        Detect the delimiter by analyzing the line starting with 'Run No.' and 
        testing various delimiters for the best match.
        """
        common_delimiters = ['\t', ',', ';', '|', ' ']
    
        with open(filepath, 'r') as file:
            for line in file:
                if line.startswith("Run No."):  # Look for the 'Run No.' header line
                    # Try each delimiter and return the one that produces the most reasonable split
                    for delim in common_delimiters:
                        split_line = line.strip().split(delim)
                        # Heuristic: The header should produce more than 2 columns if split correctly
                        if len(split_line) > 2:  # Adjust this condition if needed
                            # print(f"Detected delimiter: {delim}")  # Debugging output
                            return delim
    
        raise ValueError("Could not detect a valid delimiter or find a line starting with 'Run No.' in the file.")
    
    delim = detect_delimiter(filepath)
    
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
                headers = line.strip().split(delim)  # Capture the table headers
                # headers = line.strip().split("\t")  # Capture the table headers
                continue
            elif "Statistics" in line:
                reading_data = False
                # reading_metadata = False
            elif line == "\n":
                reading_data = False
            
            # If reading the data part, collect the rows
            if reading_data:
                data_lines.append(line.strip().split(delim))
                # data_lines.append(line.strip().split("\t"))

                    
    # Create DataFrame from the data table portion
    df = pd.DataFrame(data_lines, columns=headers)
    
    # Convert columns to the correct data types
    dtype_dict = {
        'Run No.': int,
        'Status': str,
        'Specimen Code': str,
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
    
    dtype_dict_sub = {col:dtype_dict[col] for col in df.columns}
    
    # df = df.astype(dtype_dict_sub)
    
    # Function to typecast columns while handling errors
    def safe_cast_column(df, column_name, target_dtype):
        if target_dtype == float:
            df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
        elif target_dtype == int:
            df[column_name] = pd.to_numeric(df[column_name], errors='coerce').dropna().astype(int)
        else:
            # If target_dtype is str or other types, just ensure the type is consistent
            df[column_name] = df[column_name].astype(target_dtype, errors='ignore')
    
    # Apply typecasting to each column
    for col, dtype in dtype_dict_sub.items():
        safe_cast_column(df, col, dtype)
    
    return df

def combine_rsl_files(df):
    df_results_files = df[df['Results']==True]
    df_rsl_combined = pd.DataFrame()
    for index, row in df_results_files.iterrows():
        filepath = row['Filepath']
        # print(row['Filepath'])
        df_rsl = read_rsl_file(filepath)
        df_rsl_combined = pd.concat([df_rsl_combined, df_rsl], axis=0, ignore_index=True)
        
    return df_rsl_combined
        
    

def find_matching_specimen(datetime_substring, filepath, df_rsl_combined):
    """
    

    Parameters
    ----------
    datetime_substring : TYPE
        DESCRIPTION.
    df : Pandas DataFrame
        DataFrame with columns "Filepath", "Data", and "Results" that indicate
        filepaths (not contents) and booleans to tell whether the filepath
        points to a .log file or a .rsl file.

    Returns
    -------
    specimen : TYPE
        DESCRIPTION.
    specimen_thickness : TYPE
        DESCRIPTION.
    specimen_width : TYPE
        DESCRIPTION.

    """
    # Convert datetime_substring to date and time data
    # month_map = {'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04', 'May': '05', 'Jun': '06',
    #          'Jul': '07', 'Aug': '08', 'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'}
    pattern = r"([A-Za-z]{3})-(\d{1,2})-(\d{4})-(\d{2})-(\d{2})-(\d{2})-([A-Za-z]{2})"
    
    match = re.search(pattern, datetime_substring)
    if match:
        # Extract the components from the regex match
        month_str, day, year, hour, minute, second, am_pm = match.groups()
        
        # # Convert the month abbreviation to numeric format
        # month = month_map[month_str]
        
        # Format the date as MM/DD/YYYY
        date = f"{month_str} {int(day)}, {year}"
        
        # Format the time as HH:MM:SS AM/PM
        time = f"{int(hour):02}:{minute}:{second} {am_pm.upper()}"
    else:
        pattern = r"([A-Za-z]{3})-(\d{1,2})-(\d{4})-(\d{2})-(\d{2})-(\d{2})"
        match = re.search(pattern, datetime_substring)
        if match:
            # Extract the components from the regex match
            month_str, day, year, hour, minute, second = match.groups()
            
            # # Convert the month abbreviation to numeric format
            # month = month_map[month_str]
            
            # Format the date as MM/DD/YYYY
            date = f"{month_str} {int(day)}, {year}"
            
            # Format the time as HH:MM:SS AM/PM
            time = f"{int(hour):02}:{minute}:{second}"
        else:
            date = None
            time = None
    
    # Read results files and find a good run that contains the matching date
    # and time information
    specimen = None
    specimen_thickness = None
    specimen_width = None
    tolerance_seconds = 3
        
    for i in range(len(df_rsl_combined)):
        df_time = df_rsl_combined.loc[i, "Time"]
        if (df_rsl_combined.loc[i,"Date"] == date) and is_time_within_tolerance(df_time, time, tolerance_seconds) and (df_rsl_combined.loc[i,"Status"] == "Complete"):
            if "Specimen Code" in df_rsl_combined.columns:
                specimen = df_rsl_combined.loc[i,"Specimen Code"]
            else:
                specimen = df_rsl_combined.loc[i,"Specimen Number"]
            specimen_thickness = df_rsl_combined.loc[i,"Specimen Thickness"]
            specimen_width = df_rsl_combined.loc[i,"Specimen Width"]
            break
    if specimen and specimen_thickness and specimen_width:
        specimen_thickness = float(specimen_thickness)
        specimen_width = float(specimen_width)
    if (specimen is None) and (specimen_thickness is None) and (specimen_width is None):
        # print(f"\nSpecimen details not detected. File: {filepath}")
        error_message.set(f"Specimen details not detected. File: {filepath}")
    return specimen, specimen_thickness, specimen_width

def is_time_within_tolerance(df_time_str, target_time_str, tolerance_seconds):
    # Convert the time strings into datetime objects
    df_time = datetime.strptime(df_time_str, '%H:%M:%S').time()
    target_time = datetime.strptime(target_time_str, '%H:%M:%S').time()
    
    # Convert times into full datetime objects to allow subtraction
    df_time_dt = datetime.combine(datetime.today(), df_time)
    target_time_dt = datetime.combine(datetime.today(), target_time)
    
    # Calculate the absolute difference
    time_diff = abs((df_time_dt - target_time_dt).total_seconds())
    
    # Check if the difference is within the tolerance
    return time_diff <= tolerance_seconds
            
def process_tensile_data_directory(directory, progress_bar, progress_label):
    logger = logging.getLogger("ThreadSafeLogger")
    logger.info("BEGIN PROCESSING TENSILE DATA\n")
    
    files = [f for f in os.listdir(Path(directory)) if os.path.isfile(os.path.join(directory, f))]
    data = {"File": files}
    df = pd.DataFrame(data)
    
    df["Filepath"] = [(directory + "/" + file) for file in df["File"]]
    df["Data"] = [True if (".log" in file) else False for file in df["File"]]
    df["Results"] = [True if (".rsl" in file) else False for file in df["File"]]
    
    #################################
    # Old approach: search through all .rsl files for every .log file
    # New approach: Compile .rsl data into a single dataframe and search that
    # for each .log file to determine the specimen parameters and name/code.
    # Then perform calculations as normal.
    df_rsl_combined = combine_rsl_files(df)
    
    # df_rsl_combined.to_excel('df_rsl_combined.xlsx')
    
    df_results = pd.DataFrame()
    
    total_files = len(df[df["Data"] == True])
    processed_files = 0
    
    # Load the data files only
    for filepath in df[df["Data"]==True]["Filepath"]:
        try:
            # Get the matching datetime from the file name
            dt = extract_datetime_string(filepath)
            
            # Step through the results files and get the specimen parameters
            specimen, specimen_thickness, specimen_width = find_matching_specimen(dt, filepath, df_rsl_combined)
            
            # Find the matching log file and search it for the 
            # Extract the force-displacement data from the data file
            metadata, dfdata = read_log_file(filepath)
            
            # Process the tensile data into stress and strain
            A = specimen_thickness * specimen_width     # cross-sectional area, mm^2
            gauge_length = 115.0
            dfdata['Stress (MPa)'] = -dfdata['Load [N]'] / A
            dfdata['Strain'] = dfdata['Distance [mm]'] / gauge_length
            
            # Filter out strain data so only strictly increasing strain is included
            dfdata['diff'] = dfdata['Strain'].diff()
            mask = dfdata['diff'] > 0
            dfdata = dfdata[mask]
            dfdata = dfdata.drop(columns=['diff'])
            
            # Save dfdata with the specimen name
            data_filename = specimen + '.csv'
            # data_filename = 'Processed Test Data/' + specimen + '.csv'
            data_filepath = directory + "/Processed Test Data/" + data_filename
            # data_filepath = directory + "/" + data_filename
            
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
            if len(filtered_df) < 2:
                Et_regr = np.nan
            else:
                X = filtered_df[['Strain']]
                y = filtered_df['Stress (MPa)']
                model = LinearRegression().fit(X, y)
                Et_regr = model.coef_[0]
            
            specimen_info = {
                'Specimen Code': [specimen],
                'Specimen Thickness': [specimen_thickness],
                'Specimen Width': [specimen_width],
                'Ultimate Tensile Strength (MPa)': [uts],
                'Modulus of Elasticity - Chord': [Et_chord],
                'Modulus of Elasticity - Regression': [Et_regr]}
            
            new_row = pd.DataFrame(specimen_info)
            df_results = pd.concat([new_row, df_results], ignore_index=True)
            df_results.reset_index(drop=True, inplace=True)
            
            # Write the specimen information to the file
            with open(data_filepath, 'w') as f:
                for key, value in specimen_info.items():
                    f.write(f'{key},{value[0]}\n')     # Write each key-value pair as a new line
                    
                f.write('\n')   # Add an empty line between metadata and DataFrame content
                
            dfdata.to_csv(data_filepath, mode='a', index=False)
            
            # logging.info(f"Processed TENSILE {os.path.basename(filepath)}")
            
            processed_files += 1
            progress_percent = int((processed_files / total_files) * 100)
            progress_bar["value"] = progress_percent
            progress_label.config(text=f"Progress: {progress_percent}%")
            root.update_idletasks()
            
            
        except Exception as e:
            # print(f"\nError: {e}")
            # print(f"Filepath: {filepath}")
            logger.error(f"{e}")
            logger.error(f"Filepath: {filepath}\n")
            # thread_safe_log("ThreadSafeLogger", logging.ERROR, f"{filepath}")
            # thread_safe_log("ThreadSafeLogger", logging.ERROR, f"Error in tensile data processing: {str(e)}\n")
            
            # error_message.set(f"Error: {e}\t({filepath})")
            # logging.exception(f"Error for {os.path.basename(filepath)}: {e}")
    
    results_filepath = directory + "/Processed Test Data/Tensile_results.csv"
    df_results.to_csv(results_filepath)
    
    logger.info(f"COMPLETED PROCESSING OF TENSILE DATA IN {directory}")
    
    # thread_safe_log("ThreadSafeLogger", logging.INFO, f"Completed processing of tensile data in {directory}")
    
    progress_label.config(text="Processing Complete")
    progress_bar["value"] = 100
    tensile_message.set("Tensile data processed successfully")
    root.after(10000, clear_tensile_message)

def start_process_tensile_data_directory(directory, progress_bar, progress_label):
    task_thread = Thread(target=process_tensile_data_directory, args=(directory, progress_bar, progress_label))
    task_thread.daemon = True
    task_thread.start()


def process_flexural_data_directory(directory, progress_bar, progress_label):
    # logging.info("\nBEGIN PROCESSING FLEXURAL DATA")
    # import pdb; pdb.set_trace()
    logger = logging.getLogger("ThreadSafeLogger")
    logger.info("BEGIN PROCESSING FLEXURAL DATA\n")
    
    # files = os.listdir(Path(directory))
    files = [f for f in os.listdir(Path(directory)) if os.path.isfile(os.path.join(directory, f))]
    data = {"File": files}
    df = pd.DataFrame(data)
    
    df["Filepath"] = [(directory + "/" + file) for file in df["File"]]
    # df["Filepath"] = [directory / file for file in df["File"]]
    df["Data"] = [True if (".log" in file) else False for file in df["File"]]
    df["Results"] = [True if (".rsl" in file) else False for file in df["File"]]
    
    df_rsl_combined = combine_rsl_files(df)
    
    df_results = pd.DataFrame()
    
    total_files = len(df[df["Data"] == True])
    processed_files = 0
    
    # Load the data files only
    for filepath in df[df["Data"]==True]["Filepath"]:
        try:
            # Get the matching datetime from the file name
            dt = extract_datetime_string(filepath)
            
            # Step through the results files and get the specimen parameters
            specimen, specimen_thickness, specimen_width = find_matching_specimen(dt, df, df_rsl_combined)
            
            # Find the matching log file and search it for the 
            # Extract the force-displacement data from the data file
            metadata, dfdata = read_log_file(filepath)
            
            # Smooth and shift force-distance data for better determination of
            # properties
            if dfdata.loc[100, 'Distance [mm]'] < dfdata.loc[0, 'Distance [mm]']:
                dfdata['Distance [mm]'] = -dfdata['Distance [mm]']
            rising_index = start_filter(dfdata['Load [N]'])
            dfdata = dfdata.iloc[rising_index:].copy()
            window_size = 201
            poly_order = 2
            dfdata['Savitzky-Golay Smoothed Load [N]'] = apply_savgol_filter(dfdata['Load [N]'], window_size, poly_order)
            recalculate_distance(dfdata, 14.0)
            dfnew = find_zero_distance(dfdata)
            dfnew['Recalculated Distance [mm]'] = dfnew['Recalculated Distance [mm]'] - dfnew.loc[0,'Recalculated Distance [mm]']
            
            # Process the flexural data into stress and strain
            L = 64  # span, mm
            h = specimen_thickness
            b = specimen_width
            if h is None:
                print(f'h is None for {filepath}')
            if b is None:
                print(f'b is None for {filepath}')
            
            dfnew['Stress (MPa)'] = (3 * L * dfnew['Savitzky-Golay Smoothed Load [N]']) / (2 * b * h**2)
            dfnew['Strain'] = (6 * h * dfnew['Recalculated Distance [mm]']) / (L**2)
            # dfdata['Stress (MPa)'] = -(3 * L * dfdata['Load [N]']) / (2 * b * h**2)
            # dfdata['Strain'] = (6 * h * dfdata['Distance [mm]']) / (L**2)
            
            # Filter out strain data so only strictly increasing strain is included
            # dfnew['diff'] = dfnew['Strain'].diff()
            # mask = dfnew['diff'] > 0
            # dfnew = dfnew[mask]
            # dfnew = dfnew.drop(columns=['diff'])
            # # dfdata['diff'] = dfdata['Strain'].diff()
            # # mask = dfdata['diff'] > 0
            # # dfdata = dfdata[mask]
            # # dfdata = dfdata.drop(columns=['diff'])
            
            # Save dfdata with the specimen name
            data_filename = 'Processed Test Data/' + specimen + '.csv'
            data_filepath = directory + "/" + data_filename
            
            # Add more data at the beginning of the output CSV file
            # Ultimate Flexural Strength (MPa)
            ufs = np.max(dfnew['Stress (MPa)'])
            # ufs = np.max(dfdata['Stress (MPa)'])
            
            # Chord method for Young's Modulus
            interp_func = interp1d(dfnew['Strain'], dfnew['Stress (MPa)'], kind='linear')
            # interp_func = interp1d(dfdata['Strain'], dfdata['Stress (MPa)'], kind='linear')
            sigma0005 = interp_func(0.0005)
            sigma0025 = interp_func(0.0025)
            Et_chord = (sigma0025-sigma0005)/(0.0025 - 0.0005)
            
            # Linear regression method for Young's Modulus
            filtered_df = dfnew[(dfnew['Strain'] >= 0.0005) & (dfnew['Strain'] <= 0.0025)]
            if len(filtered_df) < 2:
                Et_regr = np.nan
            else:
                X = filtered_df[['Strain']]
                y = filtered_df['Stress (MPa)']
                model = LinearRegression().fit(X, y)
                Et_regr = model.coef_[0]
            
            specimen_info = {
                'Specimen Code': [specimen],
                'Specimen Thickness': [specimen_thickness],
                'Specimen Width': [specimen_width],
                'Ultimate Tensile Strength (MPa)': [ufs],
                'Modulus of Elasticity - Chord': [Et_chord],
                'Modulus of Elasticity - Regression': [Et_regr]}
            
            new_row = pd.DataFrame(specimen_info)
            df_results = pd.concat([new_row, df_results], ignore_index=True)
            df_results.reset_index(drop=True, inplace=True)
            
            # Write the specimen information to the file
            with open(data_filepath, 'w') as f:
                for key, value in specimen_info.items():
                    f.write(f'{key},{value[0]}\n')     # Write each key-value pair as a new line
                    
                f.write('\n')   # Add an empty line between metadata and DataFrame content
                
            dfnew.to_csv(data_filepath, mode='a', index=False)
            # dfdata.to_csv(data_filepath, mode='a', index=False)
            
            # logging.info(f"Processed FLEXURAL {os.path.basename(filepath)}")
            
            processed_files += 1
            progress_percent = int((processed_files / total_files) * 100)
            progress_bar["value"] = progress_percent
            progress_label.config(text=f"Progress: {progress_percent}%")
            root.update_idletasks()
            
        except Exception as e:
            # print(f"\nError: {e}\t({filepath})")
            logger.error(f"{e}")
            logger.error(f"Filepath: {filepath}\n")
            # error_message.set(f"Error: {e}\t({filepath})")
            # logging.exception(f"Error for {os.path.basename(filepath)}: {e}")
    
    results_filepath = directory + "/Processed Test Data/Flexural_results.csv"
    df_results.to_csv(results_filepath)
    
    logger.info(f"COMPLETED PROCESSING OF FLEXURAL DATA IN {directory}")
    
    progress_label.config(text="Processing Complete")
    progress_bar["value"] = 100
    flexural_message.set("Flexural data processed successfully")
    root.after(10000, clear_flexural_message)
    
def start_process_flexural_data_directory(directory, progress_bar, progress_label):
    task_thread = Thread(target=process_flexural_data_directory, args=(directory, progress_bar, progress_label))
    task_thread.start()

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
    # check_for_updates()
    
    start_datetime = str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    
    # # log_queue = queue.Queue()
    
    # # class QueueHandler(logging.Handler):
    # #     def __init__(self, log_queue):
    # #         super().__init__()
    # #         self.log_queue = log_queue

    # #     def emit(self, record):
    # #         self.log_queue.put(self.format(record))

    # # # Logging setup
    # # logging.basicConfig(
    # #     filename=f"logs/{start_datetime}.log",
    # #     encoding="utf-8",
    # #     filemode="a",
    # #     level=logging.INFO,
    # #     format="%(asctime)s - %(levelname)s - %(message)s",
    # #     handlers=[QueueHandler(log_queue)],
    # #     style="%",
    # #     datefmt="%Y-%m-%d %H:%M:%S"
    # #     )
    
    # log_queue = queue.Queue()
    # logger = setup_logger(log_queue)
    
    # # for handler in logger.handlers:
    # #     print(f"Handler type: {type(handler)}")
    
    # listener = QueueListener(log_queue, log_worker)
    # listener.start()
    
    # # Start the logging worker thread
    # log_thread = Thread(target=log_worker, args=(log_queue,))
    # log_thread.daemon = True
    # log_thread.start()
    
    # Basic logging configuration
    log_filename = f"logs/{start_datetime}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        # handlers=[logging.StreamHandler()]
        handlers=[logging.FileHandler(log_filename, mode='a', encoding='utf-8')]  # Log only to a file
    )
    
    # Main application window
    root = tk.Tk()
    root.attributes("-topmost", True)
    root.title("Mark-10 Data Processing")
    
    upper_frame = tk.Frame(root, padx=10, pady=10)
    upper_frame.grid(row=0, column=0, sticky="w")

    # Adding some visual separation between the two sets of buttons
    separator = tk.Frame(root, height=2, bd=1, relief="sunken")
    separator.grid(row=1, columnspan=2, pady=10, padx=10, sticky="ew")
    
    lower_frame = tk.Frame(root, padx=10, pady=10)
    lower_frame.grid(row=2, column=0, sticky="w")
    
    # Variables for directory paths and messages
    # tensile_directory = StringVar(value=r"H:/Shared drives/RockWell Shared/Engineering/Engineering Projects/DLFT/DLFT Testing/Production Testing/Tensile Tests")
    # flexural_directory = StringVar(value=r"H:/Shared drives/RockWell Shared/Engineering/Engineering Projects/DLFT/DLFT Testing/Production Testing/Flexural Tests")
    tensile_directory = StringVar(value=r"G:/Shared drives/RockWell Shared/Engineering/Engineering Projects/DLFT/DLFT Testing/Production Testing/Tensile Tests")
    flexural_directory = StringVar(value=r"G:/Shared drives/RockWell Shared/Engineering/Engineering Projects/DLFT/DLFT Testing/Production Testing/Flexural Tests")
    tensile_message = StringVar(value="")
    flexural_message = StringVar(value="")
    error_message = StringVar(value="")
    
    # Buttons and labels for selecting directories
    btn_select_tensile = tk.Button(upper_frame, text="Select Tensile Directory", command=lambda: select_tensile_directory())
    btn_select_flexural = tk.Button(upper_frame, text="Select Flexural Directory", command=lambda: select_flexural_directory())
    lbl_tensile_dir = tk.Label(upper_frame, textvariable=tensile_directory)
    lbl_flexural_dir = tk.Label(upper_frame, textvariable=flexural_directory)
    
    # Buttons for processing data
    btn_process_tensile = tk.Button(lower_frame, text="Process Tensile Data",
                                    command=lambda: start_process_tensile_data_directory(tensile_directory.get(), tensile_progress_bar, tensile_progress_label))
    # btn_process_tensile = tk.Button(root, text="Process Tensile Data",
    #                                 command=lambda: process_tensile_data_directory(tensile_directory.get()))
    btn_process_flexural = tk.Button(lower_frame, text="Process Flexural Data",
                                     command=lambda: start_process_flexural_data_directory(flexural_directory.get(), flexural_progress_bar, flexural_progress_label))
    # btn_process_flexural = tk.Button(lower_frame, text="Process Flexural Data",
    #                                  command=lambda: process_flexural_data_directory(flexural_directory.get()))
    
    # Labels for processing messages
    lbl_tensile_message = tk.Label(lower_frame, textvariable=tensile_message)
    lbl_flexural_message = tk.Label(lower_frame, textvariable=flexural_message)
    
    # Label for error messages
    lbl_error_message = tk.Label(root, textvariable=error_message)
    
    # Layout for the first set (Select buttons and directory labels)
    btn_select_tensile.grid(row=0, column=0, padx=10, pady=5, sticky="w")
    lbl_tensile_dir.grid(row=0, column=1, padx=10, pady=5, sticky="w")
    btn_select_flexural.grid(row=1, column=0, padx=10, pady=5, sticky="w")
    lbl_flexural_dir.grid(row=1, column=1, padx=10, pady=5, sticky="w")
    
    
    # Layout for the second set (Process buttons and messages)
    btn_process_tensile.grid(row=0, column=0, padx=10, pady=5, sticky="w")
    # lbl_tensile_message.grid(row=3, column=1, padx=10, pady=5, sticky="w")
    btn_process_flexural.grid(row=1, column=0, padx=10, pady=5, sticky="w")
    # lbl_flexural_message.grid(row=4, column=1, padx=10, pady=5, sticky="w")
    # lbl_error_message.grid(row=5, column=0, padx=10, pady=5, sticky="w")
    
    tensile_progress_label = tk.Label(lower_frame, text="Progress: 0%")
    tensile_progress_label.grid(row=0, column=1, padx=10, pady=5, sticky="w")
    tensile_progress_bar = ttk.Progressbar(lower_frame, orient="horizontal", length=400, mode="determinate")
    tensile_progress_bar.grid(row=0, column=2, padx=10, pady=5, sticky="e")
    flexural_progress_label = tk.Label(lower_frame, text="Progress: 0%")
    flexural_progress_label.grid(row=1, column=1, padx=10, pady=5, sticky="w")
    flexural_progress_bar = ttk.Progressbar(lower_frame, orient="horizontal", length=400, mode="determinate")
    flexural_progress_bar.grid(row=1, column=2, padx=10, pady=5, sticky="e")
    
    # Run the application
    root.mainloop()