# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 10:02:16 2026

@author: Ryan.Larson
"""

# repair_results_csvs.py
#
# Run this ONCE to:
#   1. Remove duplicate rows caused by the double-append bug
#   2. Fix the mis-labelled 'Ultimate Tensile Strength' column in Flexural_results.csv
#   3. Add the parsed specimen code component columns to both files
#      (Truck Number, Well Number, Position Number, Physical Test Type,
#       Material Test Type, Orientation, Sample Number)
#
# Usage:
#   1. Set TENSILE_DIR and FLEXURAL_DIR below to your actual folder paths.
#   2. Run:  python repair_results_csvs.py
#
# A timestamped backup of each file is created before any changes are written.

import os
import re
import shutil
import pandas as pd
from datetime import datetime


TENSILE_DIR  = r"G:/Shared drives/RockWell Shared/Engineering/CAD/Engineering Efforts/DLFT/DLFT Testing/Production Testing/Tensile Tests"
FLEXURAL_DIR = r"G:/Shared drives/RockWell Shared/Engineering/CAD/Engineering Efforts/DLFT/DLFT Testing/Production Testing/Flexural Tests"

FILES_TO_REPAIR = [
    {
        "path": os.path.join(TENSILE_DIR,  "Processed Test Data", "Tensile_results.csv"),
        "strength_col_rename": None,
    },
    {
        "path": os.path.join(FLEXURAL_DIR, "Processed Test Data", "Flexural_results.csv"),
        "strength_col_rename": ("Ultimate Tensile Strength (MPa)", "Ultimate Flexural Strength (MPa)"),
    },
]


# ── Specimen code parser (copy of the version in mark10_processor.py) ─────────

def parse_specimen_code(code: str) -> dict:
    """
    Break a specimen code into its component parts.

    Pattern (optional parts noted):
        T###  W#(#)  P#(#)  [I|P|T]  [F|T]  [V|H]  [1|2]

    The suffix tokens are consumed right-to-left so that the ambiguous 'T'
    (used for both Physical and Material test types) is resolved correctly:
      - Sample number  : trailing 1 or 2
      - Orientation    : trailing V or H
      - Material type  : trailing F or T
      - Physical type  : whatever single letter (P/I/T) remains after the above
    """
    empty = {k: None for k in (
        "Truck Number", "Well Number", "Position Number",
        "Physical Test Type", "Material Test Type", "Orientation", "Sample Number"
    )}

    if not isinstance(code, str) or not code.strip():
        return empty

    s = code.strip()

    truck    = re.search(r'(T\d{3})',   s, re.IGNORECASE)
    well     = re.search(r'(W\d{1,2})', s, re.IGNORECASE)
    position = re.search(r'(P\d{1,2})', s, re.IGNORECASE)

    truck_val    = truck.group(1).upper()    if truck    else None
    well_val     = well.group(1).upper()     if well     else None
    position_val = position.group(1).upper() if position else None

    # Strip the anchored tokens to isolate the free-form suffix
    suffix = s
    for pat in (r'T\d{3}', r'W\d{1,2}', r'P\d{1,2}'):
        suffix = re.sub(pat, '', suffix, flags=re.IGNORECASE)
    suffix = suffix.strip()

    # Consume right-to-left
    sample_match = re.search(r'([12])$', suffix)
    sample_val   = sample_match.group(1) if sample_match else None
    if sample_match:
        suffix = suffix[:sample_match.start()].strip()

    ori_match = re.search(r'([VH])$', suffix, re.IGNORECASE)
    ori_val   = ori_match.group(1).upper() if ori_match else None
    if ori_match:
        suffix = suffix[:ori_match.start()].strip()

    mat_match = re.search(r'([FT])$', suffix, re.IGNORECASE)
    mat_val   = mat_match.group(1).upper() if mat_match else None
    if mat_match:
        suffix = suffix[:mat_match.start()].strip()

    phys_match = re.search(r'([PIT])$', suffix, re.IGNORECASE)
    phys_val   = phys_match.group(1).upper() if phys_match else None

    return {
        "Truck Number":       truck_val,
        "Well Number":        well_val,
        "Position Number":    position_val,
        "Physical Test Type": phys_val,
        "Material Test Type": mat_val,
        "Orientation":        ori_val,
        "Sample Number":      sample_val,
    }


def insert_parsed_code_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Insert parsed specimen code columns immediately after 'Specimen Code'."""
    component_cols = [
        "Truck Number", "Well Number", "Position Number",
        "Physical Test Type", "Material Test Type", "Orientation", "Sample Number",
    ]
    parsed = df["Specimen Code"].apply(parse_specimen_code).apply(pd.Series)
    df = df.drop(columns=[c for c in component_cols if c in df.columns])
    insert_at = df.columns.get_loc("Specimen Code") + 1
    for i, col in enumerate(component_cols):
        df.insert(insert_at + i, col, parsed[col])
    return df


# ── Repair logic ───────────────────────────────────────────────────────────────

def repair(filepath, strength_col_rename):
    if not os.path.isfile(filepath):
        print(f"  Not found, skipping: {filepath}")
        return

    df = pd.read_csv(filepath, index_col=0)
    original_rows = len(df)
    changes = []

    # 1. Fix mis-labelled column
    if strength_col_rename:
        old_name, new_name = strength_col_rename
        if old_name in df.columns:
            df = df.rename(columns={old_name: new_name})
            changes.append(f"renamed column '{old_name}' -> '{new_name}'")

    # 2. Deduplicate
    df = df.drop_duplicates(subset=["Specimen Code"], keep="last")
    df.reset_index(drop=True, inplace=True)
    removed = original_rows - len(df)
    if removed:
        changes.append(f"removed {removed} duplicate row(s)")

    # 3. Add / refresh parsed specimen code columns
    df = insert_parsed_code_columns(df)
    changes.append("added/refreshed parsed specimen code columns")

    if not changes:
        print(f"  No changes needed: {filepath}")
        return

    # Back up before writing
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = filepath.replace(".csv", f"_backup_{timestamp}.csv")
    shutil.copy2(filepath, backup_path)
    print(f"  Backup saved:  {backup_path}")

    df.to_csv(filepath)
    print(f"  Changes applied: {'; '.join(changes)}")
    print(f"  Final row count: {len(df)}")


if __name__ == "__main__":
    for entry in FILES_TO_REPAIR:
        print(f"\nRepairing: {entry['path']}")
        repair(entry["path"], entry["strength_col_rename"])
    print("\nDone.")