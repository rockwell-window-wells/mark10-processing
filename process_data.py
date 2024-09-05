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
    


if __name__ == "__main__":
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
            metadata, dfdata = read_log_file(filepath)
            print("\nMetadata:", metadata)
            print("\nDataframe:\n", df.head())
        except Exception as e:
            print(f"Error: {e}")
                