# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 13:40:25 2024

@author: Ryan.Larson

Modified to include a processing registry that tracks which .log source files
have already been processed, skipping them on subsequent runs.

Registry file: "Processed Test Data/processing_registry.csv"
  - source_file   : basename of the .log file (date-time stamp name)
  - specimen_code : the specimen identifier resolved from the .rsl file
  - processed_at  : ISO-8601 timestamp of when the file was processed
  - test_type     : "tensile" or "flexural"

The registry is keyed on the .log filename (not the specimen code) because:
  - .log filenames are unique per specimen (one file per test)
  - .rsl files are NOT unique -- a single .rsl can list many specimens, and
    the same specimen code can appear in multiple .rsl files
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
import shutil


# ── Registry helpers ──────────────────────────────────────────────────────────

REGISTRY_FILENAME = "processing_registry.csv"
REGISTRY_COLUMNS  = ["source_file", "specimen_code", "processed_at", "test_type"]


def _registry_path(directory: str) -> str:
    """Return the full path to the registry CSV for a given data directory."""
    return os.path.join(directory, "Processed Test Data", REGISTRY_FILENAME)


def load_registry(directory: str) -> pd.DataFrame:
    """
    Load (or create) the processing registry for *directory*.

    Returns a DataFrame with REGISTRY_COLUMNS.  If the file does not yet
    exist, an empty DataFrame is returned and will be written on the first
    successful specimen processing.
    """
    path = _registry_path(directory)
    if os.path.isfile(path):
        try:
            df = pd.read_csv(path, dtype=str)
            # Guarantee all columns exist even if the file predates a schema change
            for col in REGISTRY_COLUMNS:
                if col not in df.columns:
                    df[col] = ""
            return df[REGISTRY_COLUMNS]
        except Exception as e:
            logging.getLogger("ThreadSafeLogger").warning(
                f"Could not read registry at {path}: {e} – starting fresh.")
    return pd.DataFrame(columns=REGISTRY_COLUMNS)


def save_registry(directory: str, registry: pd.DataFrame) -> None:
    """Persist the registry DataFrame to disk."""
    path = _registry_path(directory)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    registry.to_csv(path, index=False)


def is_already_processed(source_basename: str, registry: pd.DataFrame) -> bool:
    """Return True if *source_basename* (.log filename) is in the registry."""
    return source_basename in registry["source_file"].values


def register_specimen(
    directory: str,
    registry: pd.DataFrame,
    source_basename: str,
    specimen_code: str,
    test_type: str,
) -> pd.DataFrame:
    """
    Add a new entry to *registry*, persist it immediately, and return the
    updated DataFrame.  Writing after every specimen means a crash mid-run
    won't lose the records of files already completed in that session.
    """
    new_row = pd.DataFrame([{
        "source_file":   source_basename,
        "specimen_code": specimen_code,
        "processed_at":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "test_type":     test_type,
    }])
    registry = pd.concat([registry, new_row], ignore_index=True)
    save_registry(directory, registry)
    return registry


# ── Existing helper functions (unchanged) ─────────────────────────────────────

def apply_savgol_filter(data, window_size, poly_order):
    """
    Applies the Savitzky-Golay filter to smooth the input data.
    """
    if window_size % 2 == 0:
        raise ValueError("Window size must be an odd integer.")
    if poly_order >= window_size:
        raise ValueError("Polynomial order must be less than window size.")
    return savgol_filter(data, window_size, poly_order)


def start_filter(load_column, target_value=10, zero_value=0):
    target_index = load_column[load_column >= target_value].index[0]
    zero_indices = load_column[:target_index][load_column[:target_index] == zero_value]
    if not zero_indices.empty:
        starting_index = zero_indices.index[-1]
    else:
        starting_index = None
    return starting_index


def recalculate_distance(df, rate):
    df.reset_index(drop=True, inplace=True)
    recalculated_distance = np.zeros(len(df))
    for i in range(len(df)):
        if i == 0:
            recalculated_distance[i] = df.loc[i, 'Distance [mm]']
        else:
            recalculated_distance[i] = (14./60.) * (df.loc[i, 'Time [s]'] - df.loc[0, 'Time [s]']) + df.loc[0, 'Distance [mm]']
    df['Recalculated Distance [mm]'] = recalculated_distance


def find_zero_distance(df):
    df['Smoothed Slope'] = np.gradient(df['Savitzky-Golay Smoothed Load [N]'], df['Recalculated Distance [mm]'])
    b = df.loc[0, 'Savitzky-Golay Smoothed Load [N]'] - df.loc[0, 'Smoothed Slope'] * df.loc[0, 'Recalculated Distance [mm]']
    x = -b / df.loc[0, 'Smoothed Slope']

    dfnew = df[['Recalculated Distance [mm]', 'Savitzky-Golay Smoothed Load [N]']].copy()
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
            if "Reading" in line:
                reading_data = True
                headers = line.strip().split("\t")
                continue
            if reading_data:
                data_lines.append(line.strip().split("\t"))
            else:
                if ":" in line:
                    key, value = line.strip().split(":", 1)
                    metadata[key.strip()] = value.strip()
                elif ".log" in line:
                    metadata["Filename"] = line

    df = pd.DataFrame(data_lines, columns=headers)
    df = df.apply(pd.to_numeric, errors='coerce')
    return metadata, df


def extract_datetime_string(filename):
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
        common_delimiters = ['\t', ',', ';', '|', ' ']
        with open(filepath, 'r') as file:
            for line in file:
                if line.startswith("Run No."):
                    for delim in common_delimiters:
                        split_line = line.strip().split(delim)
                        if len(split_line) > 2:
                            return delim
        raise ValueError("Could not detect a valid delimiter or find a line starting with 'Run No.' in the file.")

    delim = detect_delimiter(filepath)
    data_lines = []
    reading_data = False

    with open(filepath, 'r') as file:
        for line in file:
            if "Run No." in line:
                reading_data = True
                headers = line.strip().split(delim)
                continue
            elif "Statistics" in line:
                reading_data = False
            elif line == "\n":
                reading_data = False
            if reading_data:
                data_lines.append(line.strip().split(delim))

    df = pd.DataFrame(data_lines, columns=headers)

    dtype_dict = {
        'Run No.': int, 'Status': str, 'Specimen Code': str, 'Specimen Number': str,
        'Specimen Thickness': float, 'Specimen Width': float, 'Date': str, 'Time': str,
        'Speed (mm/min)': float, 'Final Load (N)': float, 'Final Distance (mm)': float,
        'Max Load (N)': float, 'LD at Max Dist (N)': float, 'Max Distance (mm)': float,
        'Dist at Max Load (mm)': float, 'Area Under Curve (N*mm)': float
    }
    dtype_dict_sub = {col: dtype_dict[col] for col in df.columns if col in dtype_dict}

    def safe_cast_column(df, column_name, target_dtype):
        if target_dtype == float:
            df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
        elif target_dtype == int:
            df[column_name] = pd.to_numeric(df[column_name], errors='coerce').dropna().astype(int)
        else:
            df[column_name] = df[column_name].astype(target_dtype, errors='ignore')

    for col, dtype in dtype_dict_sub.items():
        safe_cast_column(df, col, dtype)

    return df


def combine_rsl_files(df):
    df_results_files = df[df['Results'] == True]
    df_rsl_combined = pd.DataFrame()
    for index, row in df_results_files.iterrows():
        filepath = row['Filepath']
        df_rsl = read_rsl_file(filepath)
        df_rsl_combined = pd.concat([df_rsl_combined, df_rsl], axis=0, ignore_index=True)
    return df_rsl_combined


def find_matching_specimen(datetime_substring, filepath, df_rsl_combined):
    pattern = r"([A-Za-z]{3})-(\d{1,2})-(\d{4})-(\d{2})-(\d{2})-(\d{2})-([A-Za-z]{2})"
    match = re.search(pattern, datetime_substring)
    if match:
        month_str, day, year, hour, minute, second, am_pm = match.groups()
        date = f"{month_str} {int(day)}, {year}"
        time = f"{int(hour):02}:{minute}:{second} {am_pm.upper()}"
    else:
        pattern = r"([A-Za-z]{3})-(\d{1,2})-(\d{4})-(\d{2})-(\d{2})-(\d{2})"
        match = re.search(pattern, datetime_substring)
        if match:
            month_str, day, year, hour, minute, second = match.groups()
            date = f"{month_str} {int(day)}, {year}"
            time = f"{int(hour):02}:{minute}:{second}"
        else:
            date = None
            time = None

    specimen = None
    specimen_thickness = None
    specimen_width = None
    tolerance_seconds = 3

    for i in range(len(df_rsl_combined)):
        df_time = df_rsl_combined.loc[i, "Time"]
        if (df_rsl_combined.loc[i, "Date"] == date) and \
           is_time_within_tolerance(df_time, time, tolerance_seconds) and \
           (df_rsl_combined.loc[i, "Status"] == "Complete"):
            if "Specimen Code" in df_rsl_combined.columns:
                specimen = df_rsl_combined.loc[i, "Specimen Code"]
            else:
                specimen = df_rsl_combined.loc[i, "Specimen Number"]
            specimen_thickness = df_rsl_combined.loc[i, "Specimen Thickness"]
            specimen_width = df_rsl_combined.loc[i, "Specimen Width"]
            break

    if specimen and specimen_thickness and specimen_width:
        specimen_thickness = float(specimen_thickness)
        specimen_width = float(specimen_width)
    if (specimen is None) and (specimen_thickness is None) and (specimen_width is None):
        error_message.set(f"Specimen details not detected. File: {filepath}")
    return specimen, specimen_thickness, specimen_width


def is_time_within_tolerance(df_time_str, target_time_str, tolerance_seconds):
    df_time = datetime.strptime(df_time_str, '%H:%M:%S').time()
    target_time = datetime.strptime(target_time_str, '%H:%M:%S').time()
    df_time_dt = datetime.combine(datetime.today(), df_time)
    target_time_dt = datetime.combine(datetime.today(), target_time)
    time_diff = abs((df_time_dt - target_time_dt).total_seconds())
    return time_diff <= tolerance_seconds


# ── Processing functions (registry-aware) ─────────────────────────────────────

def process_tensile_data_directory(directory, progress_bar, progress_label):
    logger = logging.getLogger("ThreadSafeLogger")
    logger.info("BEGIN PROCESSING TENSILE DATA\n")

    # Ensure output folder exists
    os.makedirs(os.path.join(directory, "Processed Test Data"), exist_ok=True)

    files = [f for f in os.listdir(Path(directory)) if os.path.isfile(os.path.join(directory, f))]
    df = pd.DataFrame({"File": files})
    df["Filepath"] = [(directory + "/" + file) for file in df["File"]]
    df["Data"]     = [True if (".log" in file) else False for file in df["File"]]
    df["Results"]  = [True if (".rsl" in file) else False for file in df["File"]]

    df_rsl_combined = combine_rsl_files(df)

    # ── Load the registry and filter to unprocessed .log files ──
    registry = load_registry(directory)
    all_log_filepaths = df[df["Data"] == True]["Filepath"].tolist()
    unprocessed_filepaths = [
        fp for fp in all_log_filepaths
        if not is_already_processed(os.path.basename(fp), registry)
    ]

    skipped = len(all_log_filepaths) - len(unprocessed_filepaths)
    logger.info(f"Tensile: {len(all_log_filepaths)} total .log files, "
                f"{skipped} already in registry, {len(unprocessed_filepaths)} to process.")

    df_results = pd.DataFrame()
    total_files = len(unprocessed_filepaths)
    processed_files = 0

    for filepath in unprocessed_filepaths:
        source_basename = os.path.basename(filepath)
        try:
            dt = extract_datetime_string(filepath)
            specimen, specimen_thickness, specimen_width = find_matching_specimen(dt, filepath, df_rsl_combined)

            metadata, dfdata = read_log_file(filepath)

            A = specimen_thickness * specimen_width
            gauge_length = 115.0
            dfdata['Stress (MPa)'] = -dfdata['Load [N]'] / A
            dfdata['Strain'] = dfdata['Distance [mm]'] / gauge_length

            dfdata['diff'] = dfdata['Strain'].diff()
            mask = dfdata['diff'] > 0
            dfdata = dfdata[mask]
            dfdata = dfdata.drop(columns=['diff'])

            data_filepath = directory + "/Processed Test Data/" + specimen + '.csv'

            uts = np.max(dfdata['Stress (MPa)'])

            interp_func = interp1d(dfdata['Strain'], dfdata['Stress (MPa)'], kind='linear')
            sigma0005 = interp_func(0.0005)
            sigma0025 = interp_func(0.0025)
            Et_chord = (sigma0025 - sigma0005) / (0.0025 - 0.0005)

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

            with open(data_filepath, 'w') as f:
                for key, value in specimen_info.items():
                    f.write(f'{key},{value[0]}\n')
                f.write('\n')
            dfdata.to_csv(data_filepath, mode='a', index=False)

            # ── Mark as processed in the registry ──
            registry = register_specimen(directory, registry, source_basename, specimen, "tensile")

            processed_files += 1
            progress_percent = int((processed_files / total_files) * 100) if total_files else 100
            progress_bar["value"] = progress_percent
            progress_label.config(text=f"Progress: {progress_percent}%")
            root.update_idletasks()

        except Exception as e:
            logger.error(f"{e}")
            logger.error(f"Filepath: {filepath}\n")

    # Append new results to the running Tensile_results.csv rather than overwriting.
    # Deduplication is by Specimen Code so a rerun can never produce duplicate rows.
    results_filepath = directory + "/Processed Test Data/Tensile_results.csv"
    if not df_results.empty:
        if os.path.isfile(results_filepath):
            df_existing = pd.read_csv(results_filepath, index_col=0)
            df_results = pd.concat([df_existing, df_results], ignore_index=True)
            df_results = df_results.drop_duplicates(subset=["Specimen Code"], keep="last")
            df_results.reset_index(drop=True, inplace=True)
        df_results.to_csv(results_filepath)

    logger.info(f"COMPLETED PROCESSING OF TENSILE DATA IN {directory}")

    progress_label.config(text="Processing Complete")
    progress_bar["value"] = 100
    tensile_message.set("Tensile data processed successfully")
    root.after(10000, clear_tensile_message)


def start_process_tensile_data_directory(directory, progress_bar, progress_label):
    task_thread = Thread(target=process_tensile_data_directory, args=(directory, progress_bar, progress_label))
    task_thread.daemon = True
    task_thread.start()


def process_flexural_data_directory(directory, progress_bar, progress_label):
    logger = logging.getLogger("ThreadSafeLogger")
    logger.info("BEGIN PROCESSING FLEXURAL DATA\n")

    # Ensure output folder exists
    os.makedirs(os.path.join(directory, "Processed Test Data"), exist_ok=True)

    files = [f for f in os.listdir(Path(directory)) if os.path.isfile(os.path.join(directory, f))]
    df = pd.DataFrame({"File": files})
    df["Filepath"] = [(directory + "/" + file) for file in df["File"]]
    df["Data"]     = [True if (".log" in file) else False for file in df["File"]]
    df["Results"]  = [True if (".rsl" in file) else False for file in df["File"]]

    df_rsl_combined = combine_rsl_files(df)

    # ── Load the registry and filter to unprocessed .log files ──
    registry = load_registry(directory)
    all_log_filepaths = df[df["Data"] == True]["Filepath"].tolist()
    unprocessed_filepaths = [
        fp for fp in all_log_filepaths
        if not is_already_processed(os.path.basename(fp), registry)
    ]

    skipped = len(all_log_filepaths) - len(unprocessed_filepaths)
    logger.info(f"Flexural: {len(all_log_filepaths)} total .log files, "
                f"{skipped} already in registry, {len(unprocessed_filepaths)} to process.")

    df_results = pd.DataFrame()
    total_files = len(unprocessed_filepaths)
    processed_files = 0

    for filepath in unprocessed_filepaths:
        source_basename = os.path.basename(filepath)
        try:
            dt = extract_datetime_string(filepath)
            specimen, specimen_thickness, specimen_width = find_matching_specimen(dt, filepath, df_rsl_combined)

            metadata, dfdata = read_log_file(filepath)

            if dfdata.loc[100, 'Distance [mm]'] < dfdata.loc[0, 'Distance [mm]']:
                dfdata['Distance [mm]'] = -dfdata['Distance [mm]']
            rising_index = start_filter(dfdata['Load [N]'])
            dfdata = dfdata.iloc[rising_index:].copy()
            window_size = 201
            poly_order = 2
            dfdata['Savitzky-Golay Smoothed Load [N]'] = apply_savgol_filter(dfdata['Load [N]'], window_size, poly_order)
            recalculate_distance(dfdata, 14.0)
            dfnew = find_zero_distance(dfdata)
            dfnew['Recalculated Distance [mm]'] = dfnew['Recalculated Distance [mm]'] - dfnew.loc[0, 'Recalculated Distance [mm]']

            L = 64
            h = specimen_thickness
            b = specimen_width
            if h is None:
                print(f'h is None for {filepath}')
            if b is None:
                print(f'b is None for {filepath}')

            dfnew['Stress (MPa)'] = (3 * L * dfnew['Savitzky-Golay Smoothed Load [N]']) / (2 * b * h**2)
            dfnew['Strain'] = (6 * h * dfnew['Recalculated Distance [mm]']) / (L**2)

            data_filepath = directory + "/Processed Test Data/" + specimen + '.csv'

            ufs = np.max(dfnew['Stress (MPa)'])

            interp_func = interp1d(dfnew['Strain'], dfnew['Stress (MPa)'], kind='linear')
            sigma0005 = interp_func(0.0005)
            sigma0025 = interp_func(0.0025)
            Et_chord = (sigma0025 - sigma0005) / (0.0025 - 0.0005)

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
                'Ultimate Flexural Strength (MPa)': [ufs],
                'Modulus of Elasticity - Chord': [Et_chord],
                'Modulus of Elasticity - Regression': [Et_regr]}

            new_row = pd.DataFrame(specimen_info)
            df_results = pd.concat([new_row, df_results], ignore_index=True)
            df_results.reset_index(drop=True, inplace=True)

            with open(data_filepath, 'w') as f:
                for key, value in specimen_info.items():
                    f.write(f'{key},{value[0]}\n')
                f.write('\n')
            dfnew.to_csv(data_filepath, mode='a', index=False)

            # ── Mark as processed in the registry ──
            registry = register_specimen(directory, registry, source_basename, specimen, "flexural")

            processed_files += 1
            progress_percent = int((processed_files / total_files) * 100) if total_files else 100
            progress_bar["value"] = progress_percent
            progress_label.config(text=f"Progress: {progress_percent}%")
            root.update_idletasks()

        except Exception as e:
            logger.error(f"{e}")
            logger.error(f"Filepath: {filepath}\n")

    # Append new results to the running Flexural_results.csv rather than overwriting.
    # Rename the legacy mis-labelled column if present in an existing file, then
    # deduplicate by Specimen Code so reruns can never produce duplicate rows.
    results_filepath = directory + "/Processed Test Data/Flexural_results.csv"
    if not df_results.empty:
        if os.path.isfile(results_filepath):
            df_existing = pd.read_csv(results_filepath, index_col=0)
            # Heal the old copy-paste column name error if still present
            df_existing = df_existing.rename(
                columns={"Ultimate Tensile Strength (MPa)": "Ultimate Flexural Strength (MPa)"}
            )
            df_results = pd.concat([df_existing, df_results], ignore_index=True)
            df_results = df_results.drop_duplicates(subset=["Specimen Code"], keep="last")
            df_results.reset_index(drop=True, inplace=True)
        df_results.to_csv(results_filepath)

    logger.info(f"COMPLETED PROCESSING OF FLEXURAL DATA IN {directory}")

    progress_label.config(text="Processing Complete")
    progress_bar["value"] = 100
    flexural_message.set("Flexural data processed successfully")
    root.after(10000, clear_flexural_message)


def start_process_flexural_data_directory(directory, progress_bar, progress_label):
    task_thread = Thread(target=process_flexural_data_directory, args=(directory, progress_bar, progress_label))
    task_thread.start()


# ── GUI helpers (unchanged) ───────────────────────────────────────────────────

def clear_tensile_message():
    tensile_message.set("")

def clear_flexural_message():
    flexural_message.set("")

def select_tensile_directory():
    root = Tk()
    root.withdraw()
    directory = fd.askdirectory(title="Select the Tensile Data Directory")
    root.destroy()
    tensile_directory.set(directory)

def select_flexural_directory():
    root = Tk()
    root.withdraw()
    directory = fd.askdirectory(title="Select the Flexural Data Directory")
    root.destroy()
    flexural_directory.set(directory)

def remove_logger_handlers():
    logger = logging.getLogger("ThreadSafeLogger")
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

def on_gui_close(root):
    remove_logger_handlers()
    root.destroy()

def repair(filepath, strength_col_rename=None):
    if not os.path.isfile(filepath):
        print(f"  Not found, skipping: {filepath}")
        return

    df = pd.read_csv(filepath, index_col=0)
    original_rows = len(df)

    # Fix mis-labelled column if present
    if strength_col_rename:
        old_name, new_name = strength_col_rename
        if old_name in df.columns:
            df = df.rename(columns={old_name: new_name})
            print(f"  Renamed column '{old_name}' → '{new_name}'")

    # Collapse duplicate Specimen Code rows, keeping the last occurrence
    df = df.drop_duplicates(subset=["Specimen Code"], keep="last")
    df.reset_index(drop=True, inplace=True)
    removed = original_rows - len(df)

    if removed == 0 and strength_col_rename and strength_col_rename[0] not in pd.read_csv(filepath, index_col=0).columns:
        print(f"  No changes needed: {filepath}")
        return

    # Back up the original before overwriting
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = filepath.replace(".csv", f"_backup_{timestamp}.csv")
    shutil.copy2(filepath, backup_path)
    print(f"  Backup saved to: {backup_path}")

    df.to_csv(filepath)
    print(f"  Removed {removed} duplicate row(s). Final row count: {len(df)}")

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    start_datetime = str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))

    log_dir = "mark10_logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_filename = f"{log_dir}/{start_datetime}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_filename, mode='a', encoding='utf-8')]
    )

    root = tk.Tk()
    root.attributes("-topmost", True)
    root.title("Mark-10 Data Processing")
    root.protocol("WM_DELETE_WINDOW", lambda: on_gui_close(root))

    upper_frame = tk.Frame(root, padx=10, pady=10)
    upper_frame.grid(row=0, column=0, sticky="w")

    separator = tk.Frame(root, height=2, bd=1, relief="sunken")
    separator.grid(row=1, columnspan=2, pady=10, padx=10, sticky="ew")

    lower_frame = tk.Frame(root, padx=10, pady=10)
    lower_frame.grid(row=2, column=0, sticky="w")

    tensile_directory  = StringVar(value=r"G:/Shared drives/RockWell Shared/Engineering/CAD/Engineering Efforts/DLFT/DLFT Testing/Production Testing/Tensile Tests")
    flexural_directory = StringVar(value=r"G:/Shared drives/RockWell Shared/Engineering/CAD/Engineering Efforts/DLFT/DLFT Testing/Production Testing/Flexural Tests")
    tensile_message    = StringVar(value="")
    flexural_message   = StringVar(value="")
    error_message      = StringVar(value="")

    btn_select_tensile  = tk.Button(upper_frame, text="Select Tensile Directory",  command=lambda: select_tensile_directory())
    btn_select_flexural = tk.Button(upper_frame, text="Select Flexural Directory", command=lambda: select_flexural_directory())
    lbl_tensile_dir     = tk.Label(upper_frame, textvariable=tensile_directory)
    lbl_flexural_dir    = tk.Label(upper_frame, textvariable=flexural_directory)

    btn_process_tensile  = tk.Button(lower_frame, text="Process Tensile Data",
                                     command=lambda: start_process_tensile_data_directory(tensile_directory.get(), tensile_progress_bar, tensile_progress_label))
    btn_process_flexural = tk.Button(lower_frame, text="Process Flexural Data",
                                     command=lambda: start_process_flexural_data_directory(flexural_directory.get(), flexural_progress_bar, flexural_progress_label))

    lbl_tensile_message  = tk.Label(lower_frame, textvariable=tensile_message)
    lbl_flexural_message = tk.Label(lower_frame, textvariable=flexural_message)
    lbl_error_message    = tk.Label(root, textvariable=error_message)

    btn_select_tensile.grid( row=0, column=0, padx=10, pady=5, sticky="w")
    lbl_tensile_dir.grid(    row=0, column=1, padx=10, pady=5, sticky="w")
    btn_select_flexural.grid(row=1, column=0, padx=10, pady=5, sticky="w")
    lbl_flexural_dir.grid(   row=1, column=1, padx=10, pady=5, sticky="w")

    btn_process_tensile.grid( row=0, column=0, padx=10, pady=5, sticky="w")
    btn_process_flexural.grid(row=1, column=0, padx=10, pady=5, sticky="w")

    tensile_progress_label = tk.Label(lower_frame, text="Progress: 0%")
    tensile_progress_label.grid(row=0, column=1, padx=10, pady=5, sticky="w")
    tensile_progress_bar = ttk.Progressbar(lower_frame, orient="horizontal", length=400, mode="determinate")
    tensile_progress_bar.grid(row=0, column=2, padx=10, pady=5, sticky="e")

    flexural_progress_label = tk.Label(lower_frame, text="Progress: 0%")
    flexural_progress_label.grid(row=1, column=1, padx=10, pady=5, sticky="w")
    flexural_progress_bar = ttk.Progressbar(lower_frame, orient="horizontal", length=400, mode="determinate")
    flexural_progress_bar.grid(row=1, column=2, padx=10, pady=5, sticky="e")

    root.mainloop()