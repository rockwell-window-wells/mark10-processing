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

Chord modulus strain window is configurable via the GUI (defaults: 0.05%–0.25%
strain, matching the original behavior).  The selected window is stored in the
per-specimen CSV header so results are always traceable to the window used.

A "Force Reprocess All" checkbox bypasses the registry so the full dataset can
be reprocessed with a new strain window without deleting the registry manually.
"""

import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import Tk
from tkinter import ttk
from tkinter import filedialog as fd
from tkinter import StringVar, BooleanVar, DoubleVar
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


def _parse_time_to_seconds(s: str) -> int:
    """
    Parse a time string to an integer number of seconds since midnight.
    Tries 24-hour format first, then 12-hour with AM/PM.
    Raises ValueError if no format matches.
    """
    for fmt in ('%H:%M:%S', '%H:%M:%S %p', '%I:%M:%S %p'):
        try:
            t = datetime.strptime(s.strip(), fmt)
            return t.hour * 3600 + t.minute * 60 + t.second
        except ValueError:
            continue
    raise ValueError(f"Cannot parse time string: {s!r}")


def combine_rsl_files(df):
    """
    Read all .rsl files, concatenate them, and pre-parse timestamps so that
    find_matching_specimen can do a fast indexed lookup instead of a row-by-row
    strptime scan.

    Adds two columns to the returned DataFrame:
        _time_s  : integer seconds-since-midnight (for fast tolerance check)
        _date    : the Date string, kept as-is for groupby index
    """
    frames = []
    for _, row in df[df['Results'] == True].iterrows():
        try:
            frames.append(read_rsl_file(row['Filepath']))
        except Exception as e:
            logging.getLogger("ThreadSafeLogger").warning(
                f"Could not read RSL file {row['Filepath']}: {e}")

    if not frames:
        return pd.DataFrame()

    # Single concat — avoids the O(n²) repeated copies of the old loop
    df_rsl_combined = pd.concat(frames, axis=0, ignore_index=True)

    # Pre-parse every time string once so lookups are O(1) arithmetic
    def _safe_parse(s):
        try:
            return _parse_time_to_seconds(str(s))
        except ValueError:
            return None

    df_rsl_combined['_time_s'] = df_rsl_combined['Time'].apply(_safe_parse)
    return df_rsl_combined


def find_matching_specimen(datetime_substring, filepath, df_rsl_combined, allow_invalid=False):
    """
    Locate the RSL row that matches a .log file's embedded timestamp.

    Uses the pre-parsed _time_s column from combine_rsl_files() to avoid
    calling strptime on every row for every specimen lookup.
    """
    # ── Parse the target date/time from the filename ───────────────────────
    pattern = r"([A-Za-z]{3})-(\d{1,2})-(\d{4})-(\d{2})-(\d{2})-(\d{2})-([A-Za-z]{2})"
    match = re.search(pattern, datetime_substring)
    if match:
        month_str, day, year, hour, minute, second, am_pm = match.groups()
        date = f"{month_str} {int(day)}, {year}"
        raw = f"{int(hour):02}:{minute}:{second} {am_pm.upper()}"
        target_s = _parse_time_to_seconds(
            datetime.strptime(raw, '%I:%M:%S %p').strftime('%H:%M:%S')
        )
    else:
        pattern = r"([A-Za-z]{3})-(\d{1,2})-(\d{4})-(\d{2})-(\d{2})-(\d{2})"
        match = re.search(pattern, datetime_substring)
        if match:
            month_str, day, year, hour, minute, second = match.groups()
            date = f"{month_str} {int(day)}, {year}"
            target_s = _parse_time_to_seconds(f"{int(hour):02}:{minute}:{second}")
        else:
            error_message.set(f"Specimen details not detected. File: {filepath}")
            return None, None, None

    tolerance_seconds = 3

    # ── Fast date pre-filter, then integer arithmetic for time tolerance ───
    date_mask   = df_rsl_combined['Date'] == date
    status_mask = (df_rsl_combined['Status'] == 'Complete')
    if allow_invalid:
        status_mask = status_mask | (df_rsl_combined['Status'] == 'Invalid')

    candidates = df_rsl_combined[date_mask & status_mask].copy()
    if candidates.empty:
        error_message.set(f"Specimen details not detected. File: {filepath}")
        return None, None, None

    time_mask = candidates['_time_s'].apply(
        lambda s: s is not None and abs(s - target_s) <= tolerance_seconds
    )
    match_rows = candidates[time_mask]

    if match_rows.empty:
        error_message.set(f"Specimen details not detected. File: {filepath}")
        return None, None, None

    row = match_rows.iloc[0]
    if "Specimen Code" in df_rsl_combined.columns:
        specimen = str(row["Specimen Code"]).strip().upper()
    else:
        specimen = str(row["Specimen Number"]).strip().upper()

    specimen_thickness = float(row["Specimen Thickness"])
    specimen_width     = float(row["Specimen Width"])
    return specimen, specimen_thickness, specimen_width


def is_time_within_tolerance(df_time_str, target_time_str, tolerance_seconds):
    """Retained for any external callers; not used in the main processing loop."""
    t1 = _parse_time_to_seconds(df_time_str)
    t2 = _parse_time_to_seconds(target_time_str)
    return abs(t1 - t2) <= tolerance_seconds


# ── Chord modulus calculation ─────────────────────────────────────────────────

def compute_chord_modulus(strain_series, stress_series, strain_start: float, strain_end: float):
    """
    Compute chord modulus and regression modulus over [strain_start, strain_end].

    Parameters
    ----------
    strain_series : array-like
        Strain data (must be monotonically increasing and cover the window).
    stress_series : array-like
        Corresponding stress data (MPa).
    strain_start : float
        Lower bound of the chord window (e.g. 0.0005 for 0.05 % strain).
    strain_end : float
        Upper bound of the chord window (e.g. 0.0025 for 0.25 % strain).

    Returns
    -------
    Et_chord : float  – (σ₂ − σ₁) / (ε₂ − ε₁)
    Et_regr  : float  – slope of OLS fit over the window (NaN if < 2 points)
    """
    interp_func = interp1d(strain_series, stress_series, kind='linear')
    sigma_start = float(interp_func(strain_start))
    sigma_end   = float(interp_func(strain_end))
    Et_chord = (sigma_end - sigma_start) / (strain_end - strain_start)

    filtered = pd.DataFrame({'Strain': strain_series, 'Stress': stress_series})
    filtered = filtered[(filtered['Strain'] >= strain_start) & (filtered['Strain'] <= strain_end)]
    if len(filtered) < 2:
        Et_regr = np.nan
    else:
        model = LinearRegression().fit(filtered[['Strain']], filtered['Stress'])
        Et_regr = model.coef_[0]

    return Et_chord, Et_regr


# ── Processing functions (registry-aware, configurable modulus window) ─────────

def process_tensile_data_directory(
    directory,
    progress_bar,
    progress_label,
    strain_start: float = 0.0005,
    strain_end:   float = 0.0025,
    force_reprocess: bool = False,
):
    """
    Process all tensile .log files in *directory*.

    Parameters
    ----------
    strain_start : float
        Lower strain bound for chord modulus (default 0.0005 = 0.05 %).
    strain_end : float
        Upper strain bound for chord modulus (default 0.0025 = 0.25 %).
    force_reprocess : bool
        When True, all .log files are processed regardless of registry state.
        The registry is still updated after each specimen so a partial run
        can be resumed with force_reprocess=False.
    """
    logger = logging.getLogger("ThreadSafeLogger")
    logger.info("BEGIN PROCESSING TENSILE DATA\n")
    logger.info(f"Chord modulus window: {strain_start*100:.3f}% – {strain_end*100:.3f}% strain")
    if force_reprocess:
        logger.info("Force-reprocess enabled: registry check bypassed for all files.")

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

    if force_reprocess:
        unprocessed_filepaths = all_log_filepaths
    else:
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

            Et_chord, Et_regr = compute_chord_modulus(
                dfdata['Strain'].values,
                dfdata['Stress (MPa)'].values,
                strain_start,
                strain_end,
            )

            specimen_info = {
                'Specimen Code':                    [specimen],
                'Specimen Thickness':               [specimen_thickness],
                'Specimen Width':                   [specimen_width],
                'Ultimate Tensile Strength (MPa)':  [uts],
                'Modulus of Elasticity - Chord':    [Et_chord],
                'Modulus of Elasticity - Regression': [Et_regr],
                'Chord Modulus Strain Start':       [strain_start],
                'Chord Modulus Strain End':         [strain_end],
            }

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


def start_process_tensile_data_directory(directory, progress_bar, progress_label,
                                         strain_start, strain_end, force_reprocess):
    task_thread = Thread(
        target=process_tensile_data_directory,
        args=(directory, progress_bar, progress_label, strain_start, strain_end, force_reprocess)
    )
    task_thread.daemon = True
    task_thread.start()


def process_flexural_data_directory(
    directory,
    progress_bar,
    progress_label,
    strain_start: float = 0.0005,
    strain_end:   float = 0.0025,
    force_reprocess: bool = False,
):
    """
    Process all flexural .log files in *directory*.

    Parameters
    ----------
    strain_start : float
        Lower strain bound for chord modulus (default 0.0005 = 0.05 %).
    strain_end : float
        Upper strain bound for chord modulus (default 0.0025 = 0.25 %).
    force_reprocess : bool
        When True, all .log files are processed regardless of registry state.
    """
    logger = logging.getLogger("ThreadSafeLogger")
    logger.info("BEGIN PROCESSING FLEXURAL DATA\n")
    logger.info(f"Chord modulus window: {strain_start*100:.3f}% – {strain_end*100:.3f}% strain")
    if force_reprocess:
        logger.info("Force-reprocess enabled: registry check bypassed for all files.")

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

    if force_reprocess:
        unprocessed_filepaths = all_log_filepaths
    else:
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

            Et_chord, Et_regr = compute_chord_modulus(
                dfnew['Strain'].values,
                dfnew['Stress (MPa)'].values,
                strain_start,
                strain_end,
            )

            specimen_info = {
                'Specimen Code':                       [specimen],
                'Specimen Thickness':                  [specimen_thickness],
                'Specimen Width':                      [specimen_width],
                'Ultimate Flexural Strength (MPa)':    [ufs],
                'Modulus of Elasticity - Chord':       [Et_chord],
                'Modulus of Elasticity - Regression':  [Et_regr],
                'Chord Modulus Strain Start':          [strain_start],
                'Chord Modulus Strain End':            [strain_end],
            }

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
    results_filepath = directory + "/Processed Test Data/Flexural_results.csv"
    if not df_results.empty:
        if os.path.isfile(results_filepath):
            df_existing = pd.read_csv(results_filepath, index_col=0)
            df_existing = df_existing.rename(
                columns={"Ultimate Tensile Strength (MPa)": "Ultimate Flexural Strength (MPa)"}
            )
            df_results = pd.concat([df_existing, df_results], ignore_index=True)
            df_results = df_results.drop_duplicates(subset=["Specimen Code"], keep="last")
            df_results.reset_index(drop=True, inplace=True)
        df_results = insert_parsed_code_columns(df_results)
        df_results.to_csv(results_filepath)

    logger.info(f"COMPLETED PROCESSING OF FLEXURAL DATA IN {directory}")

    progress_label.config(text="Processing Complete")
    progress_bar["value"] = 100
    flexural_message.set("Flexural data processed successfully")
    root.after(10000, clear_flexural_message)


def start_process_flexural_data_directory(directory, progress_bar, progress_label,
                                          strain_start, strain_end, force_reprocess):
    task_thread = Thread(
        target=process_flexural_data_directory,
        args=(directory, progress_bar, progress_label, strain_start, strain_end, force_reprocess)
    )
    task_thread.start()


# ── GUI helpers ────────────────────────────────────────────────────────────────

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

def validate_tensile_strain_inputs(*args):
    """Enable/disable tensile process button based on validity of tensile strain inputs."""
    try:
        s = float(tensile_strain_start.get())
        e = float(tensile_strain_end.get())
        ok = (0 < s < e < 1)
    except ValueError:
        ok = False
    btn_process_tensile.config(state="normal" if ok else "disabled")


def validate_flexural_strain_inputs(*args):
    """Enable/disable flexural process button based on validity of flexural strain inputs."""
    try:
        s = float(flexural_strain_start.get())
        e = float(flexural_strain_end.get())
        ok = (0 < s < e < 1)
    except ValueError:
        ok = False
    btn_process_flexural.config(state="normal" if ok else "disabled")


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

    tensile_directory  = StringVar(value=r"G:/Shared drives/RockWell Shared/Engineering/CAD/Engineering Efforts/DLFT/DLFT Testing/Production Testing/Tensile Tests")
    flexural_directory = StringVar(value=r"G:/Shared drives/RockWell Shared/Engineering/CAD/Engineering Efforts/DLFT/DLFT Testing/Production Testing/Flexural Tests")
    tensile_message    = StringVar(value="")
    flexural_message   = StringVar(value="")
    error_message      = StringVar(value="")

    # ── Directory selection frame ──────────────────────────────────────────────
    upper_frame = tk.Frame(root, padx=10, pady=10)
    upper_frame.grid(row=0, column=0, sticky="ew")

    btn_select_tensile  = tk.Button(upper_frame, text="Select Tensile Directory",  command=lambda: select_tensile_directory())
    btn_select_flexural = tk.Button(upper_frame, text="Select Flexural Directory", command=lambda: select_flexural_directory())
    lbl_tensile_dir     = tk.Label(upper_frame, textvariable=tensile_directory)
    lbl_flexural_dir    = tk.Label(upper_frame, textvariable=flexural_directory)

    btn_select_tensile.grid( row=0, column=0, padx=10, pady=5, sticky="w")
    lbl_tensile_dir.grid(    row=0, column=1, padx=10, pady=5, sticky="w")
    btn_select_flexural.grid(row=1, column=0, padx=10, pady=5, sticky="w")
    lbl_flexural_dir.grid(   row=1, column=1, padx=10, pady=5, sticky="w")

    tk.Frame(root, height=2, bd=1, relief="sunken").grid(
        row=1, columnspan=2, pady=5, padx=10, sticky="ew")

    # ── Tensile chord modulus options ──────────────────────────────────────────
    tensile_opts = tk.LabelFrame(root, text="Tensile — Chord Modulus Options", padx=10, pady=8)
    tensile_opts.grid(row=2, column=0, padx=10, pady=(5, 2), sticky="ew")

    # Explanatory note
    tensile_note_text = (
        "The strain window below controls where the chord modulus is calculated on the tensile\n"
        "stress-strain curve. The ISO 527 default window (0.05 %–0.25 % strain) can underestimate\n"
        "modulus when grip slip or specimen seating produces a compliant toe region at low strains.\n"
        "If your tensile modulus appears low relative to a reference lab but strengths agree,\n"
        "try shifting the window past the toe (e.g. 1.0 %–1.2 % strain) and use\n"
        "\"Force reprocess all\" to apply the new window to your full dataset.\n"
        "Leave flexural options at their defaults — flexural tests are not affected by grip slip."
    )
    tk.Label(
        tensile_opts, text=tensile_note_text,
        justify="left", fg="#555555", wraplength=620,
    ).grid(row=0, column=0, columnspan=3, sticky="w", padx=5, pady=(2, 8))

    tensile_strain_start = StringVar(value="0.01")
    tensile_strain_end   = StringVar(value="0.012")
    tensile_strain_start.trace_add("write", validate_tensile_strain_inputs)
    tensile_strain_end.trace_add("write",   validate_tensile_strain_inputs)

    tk.Label(tensile_opts, text="Strain window start:").grid(
        row=1, column=0, sticky="w", padx=5, pady=3)
    tk.Entry(tensile_opts, textvariable=tensile_strain_start, width=10).grid(
        row=1, column=1, sticky="w", padx=5)
    tk.Label(tensile_opts, text="ISO 527 default: 0.0005  |  Toe-corrected example: 0.01",
             fg="#777777").grid(row=1, column=2, sticky="w", padx=10)

    tk.Label(tensile_opts, text="Strain window end:  ").grid(
        row=2, column=0, sticky="w", padx=5, pady=3)
    tk.Entry(tensile_opts, textvariable=tensile_strain_end, width=10).grid(
        row=2, column=1, sticky="w", padx=5)
    tk.Label(tensile_opts, text="ISO 527 default: 0.0025  |  Toe-corrected example: 0.012",
             fg="#777777").grid(row=2, column=2, sticky="w", padx=10)

    tensile_force_reprocess_var = BooleanVar(value=False)
    tk.Checkbutton(
        tensile_opts,
        text="Force reprocess all tensile files (ignores registry — use to apply a new strain window to existing data)",
        variable=tensile_force_reprocess_var,
    ).grid(row=3, column=0, columnspan=3, sticky="w", padx=5, pady=(6, 2))

    tk.Frame(root, height=2, bd=1, relief="sunken").grid(
        row=3, columnspan=2, pady=5, padx=10, sticky="ew")

    # ── Flexural chord modulus options ─────────────────────────────────────────
    flexural_opts = tk.LabelFrame(root, text="Flexural — Chord Modulus Options", padx=10, pady=8)
    flexural_opts.grid(row=4, column=0, padx=10, pady=(2, 5), sticky="ew")

    flexural_note_text = (
        "Flexural tests don't experience stark seating artifacts, so the default\n"
        "strain window is appropriate. Only change these values if you have a\n"
        "specific reason to do so."
    )
    tk.Label(
        flexural_opts, text=flexural_note_text,
        justify="left", fg="#555555", wraplength=620,
    ).grid(row=0, column=0, columnspan=3, sticky="w", padx=5, pady=(2, 8))

    flexural_strain_start = StringVar(value="0.0005")
    flexural_strain_end   = StringVar(value="0.0025")
    flexural_strain_start.trace_add("write", validate_flexural_strain_inputs)
    flexural_strain_end.trace_add("write",   validate_flexural_strain_inputs)

    tk.Label(flexural_opts, text="Strain window start:").grid(
        row=1, column=0, sticky="w", padx=5, pady=3)
    tk.Entry(flexural_opts, textvariable=flexural_strain_start, width=10).grid(
        row=1, column=1, sticky="w", padx=5)
    tk.Label(flexural_opts, text="Default: 0.0005", fg="#777777").grid(
        row=1, column=2, sticky="w", padx=10)

    tk.Label(flexural_opts, text="Strain window end:  ").grid(
        row=2, column=0, sticky="w", padx=5, pady=3)
    tk.Entry(flexural_opts, textvariable=flexural_strain_end, width=10).grid(
        row=2, column=1, sticky="w", padx=5)
    tk.Label(flexural_opts, text="Default: 0.0025", fg="#777777").grid(
        row=2, column=2, sticky="w", padx=10)

    flexural_force_reprocess_var = BooleanVar(value=False)
    tk.Checkbutton(
        flexural_opts,
        text="Force reprocess all flexural files (ignores registry — use to apply a new strain window to existing data)",
        variable=flexural_force_reprocess_var,
    ).grid(row=3, column=0, columnspan=3, sticky="w", padx=5, pady=(6, 2))

    tk.Frame(root, height=2, bd=1, relief="sunken").grid(
        row=5, columnspan=2, pady=5, padx=10, sticky="ew")

    # ── Processing buttons / progress frame ───────────────────────────────────
    lower_frame = tk.Frame(root, padx=10, pady=10)
    lower_frame.grid(row=6, column=0, sticky="w")

    btn_process_tensile = tk.Button(
        lower_frame, text="Process Tensile Data",
        command=lambda: start_process_tensile_data_directory(
            tensile_directory.get(),
            tensile_progress_bar,
            tensile_progress_label,
            float(tensile_strain_start.get()),
            float(tensile_strain_end.get()),
            tensile_force_reprocess_var.get(),
        )
    )
    btn_process_flexural = tk.Button(
        lower_frame, text="Process Flexural Data",
        command=lambda: start_process_flexural_data_directory(
            flexural_directory.get(),
            flexural_progress_bar,
            flexural_progress_label,
            float(flexural_strain_start.get()),
            float(flexural_strain_end.get()),
            flexural_force_reprocess_var.get(),
        )
    )

    tk.Label(lower_frame, textvariable=tensile_message).grid( row=0, column=3, padx=10)
    tk.Label(lower_frame, textvariable=flexural_message).grid(row=1, column=3, padx=10)
    tk.Label(root, textvariable=error_message).grid(row=7, column=0, padx=10, sticky="w")

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