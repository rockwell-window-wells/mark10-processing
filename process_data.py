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


# ── Specimen code parser ──────────────────────────────────────────────────────

def parse_specimen_code(code: str) -> dict:
    """
    Break a specimen code into its component parts.

    Expected pattern (all parts except Truck, Well, Material Test Type, and
    Orientation are optional):

        T###  W#(#)  P#(#)  [I|P|T]  [F|T]  [V|H]  [1|2]

    Examples that should all parse correctly:
        T001W1P1FV1   → truck=T001, well=W1, pos=P1, phys=None, mat=F, ori=V, sample=1
        T012W3P2ITFH  → truck=T012, well=W3, pos=P2, phys=I, mat=T (but wait —
                         this is ambiguous with the Physical letter; see note below)
        T005W10P3TV2  → truck=T005, well=W10, pos=P3, phys=T, mat=V... no —
                         orientation is V/H, material is F/T

    Ambiguity note: the Physical Test Type letter and the Material Test Type
    letter share 'T' as a possible value.  The parser resolves this by treating
    the Material Test Type as the LAST single-letter token before the optional
    sample number, and the Physical Test Type (if present) as the token
    immediately before it.

    Returns a dict with keys:
        Truck Number, Well Number, Position Number,
        Physical Test Type, Material Test Type, Orientation, Sample Number
    All values are strings or None if absent.
    """
    if not isinstance(code, str) or not code.strip():
        return {k: None for k in (
            "Truck Number", "Well Number", "Position Number",
            "Physical Test Type", "Material Test Type", "Orientation", "Sample Number"
        )}

    s = code.strip()

    # ── Fixed-position anchored components ────────────────────────────────────
    truck    = re.search(r'(T\d{3})',          s, re.IGNORECASE)
    well     = re.search(r'(W\d{1,2})',        s, re.IGNORECASE)
    position = re.search(r'(P\d{1,2})',        s, re.IGNORECASE)

    truck_val    = truck.group(1).upper()    if truck    else None
    well_val     = well.group(1).upper()     if well     else None
    position_val = position.group(1).upper() if position else None

    # ── Trailing variable-length suffix after the positional tokens ───────────
    # Strip everything we've already identified to isolate the suffix.
    suffix = s
    for pat in (r'T\d{3}', r'W\d{1,2}', r'P\d{1,2}'):
        suffix = re.sub(pat, '', suffix, flags=re.IGNORECASE)
    suffix = suffix.strip()

    # Optional trailing sample number (1 or 2)
    sample_match = re.search(r'([12])$', suffix)
    sample_val   = sample_match.group(1) if sample_match else None
    if sample_match:
        suffix = suffix[:sample_match.start()].strip()

    # Orientation — last remaining V or H
    ori_match = re.search(r'([VH])$', suffix, re.IGNORECASE)
    ori_val   = ori_match.group(1).upper() if ori_match else None
    if ori_match:
        suffix = suffix[:ori_match.start()].strip()

    # Material test type — last remaining F or T
    mat_match = re.search(r'([FT])$', suffix, re.IGNORECASE)
    mat_val   = mat_match.group(1).upper() if mat_match else None
    if mat_match:
        suffix = suffix[:mat_match.start()].strip()

    # Physical test type — whatever single letter remains (P, I, or T)
    phys_match = re.search(r'([PIT])$', suffix, re.IGNORECASE)
    phys_val   = phys_match.group(1).upper() if phys_match else None

    return {
        "Truck Number":        truck_val,
        "Well Number":         well_val,
        "Position Number":     position_val,
        "Physical Test Type":  phys_val,
        "Material Test Type":  mat_val,
        "Orientation":         ori_val,
        "Sample Number":       sample_val,
    }


def insert_parsed_code_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse the 'Specimen Code' column and insert the component columns
    immediately after it.  Safe to call on DataFrames that already have
    some or all of those columns (they will be overwritten in place).
    """
    component_cols = [
        "Truck Number", "Well Number", "Position Number",
        "Physical Test Type", "Material Test Type", "Orientation", "Sample Number",
    ]

    parsed = df["Specimen Code"].apply(parse_specimen_code).apply(pd.Series)

    # Drop any already-existing component columns so we can re-insert cleanly
    df = df.drop(columns=[c for c in component_cols if c in df.columns])

    # Find insertion point: one position after 'Specimen Code'
    insert_at = df.columns.get_loc("Specimen Code") + 1

    for i, col in enumerate(component_cols):
        df.insert(insert_at + i, col, parsed[col])

    return df


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


def find_matching_specimen(datetime_substring, filepath, df_rsl_combined, allow_invalid=False):
    pattern = r"([A-Za-z]{3})-(\d{1,2})-(\d{4})-(\d{2})-(\d{2})-(\d{2})-([A-Za-z]{2})"
    match = re.search(pattern, datetime_substring)
    if match:
        month_str, day, year, hour, minute, second, am_pm = match.groups()
        date = f"{month_str} {int(day)}, {year}"
        raw = f"{int(hour):02}:{minute}:{second} {am_pm.upper()}"
        time = datetime.strptime(raw, '%I:%M:%S %p').strftime('%H:%M:%S')
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
        status = df_rsl_combined.loc[i, "Status"]
        status_ok = (status == "Complete") or (allow_invalid and status == "Invalid")
        if (df_rsl_combined.loc[i, "Date"] == date) and \
           is_time_within_tolerance(df_time, time, tolerance_seconds) and \
           status_ok:
            if "Specimen Code" in df_rsl_combined.columns:
                specimen = df_rsl_combined.loc[i, "Specimen Code"].strip().upper()
            else:
                specimen = df_rsl_combined.loc[i, "Specimen Number"].strip().upper()
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
    def parse_time(s):
        for fmt in ('%H:%M:%S %p', '%I:%M:%S %p', '%H:%M:%S'):
            try:
                return datetime.strptime(s.strip(), fmt)
            except ValueError:
                continue
        raise ValueError(f"Cannot parse time string: {s!r}")

    df_time_dt     = parse_time(df_time_str)
    target_time_dt = parse_time(target_time_str)
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
            new_row = insert_parsed_code_columns(new_row)
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
        # Re-parse all codes so any rows loaded from an older file get the columns too
        df_results = insert_parsed_code_columns(df_results)
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
            specimen, specimen_thickness, specimen_width = find_matching_specimen(dt, filepath, df_rsl_combined, allow_invalid=True)

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
            new_row = insert_parsed_code_columns(new_row)
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
        # Re-parse all codes so any rows loaded from an older file get the columns too
        df_results = insert_parsed_code_columns(df_results)
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