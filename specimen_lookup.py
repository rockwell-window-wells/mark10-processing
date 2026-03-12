# -*- coding: utf-8 -*-
"""
specimen_lookup.py

Utility for extracting ordered rows from tensile_results.csv or
flexural_results.csv based on a list of specimen codes.

Workflow
--------
1. Select a results CSV file (tensile_results.csv or flexural_results.csv).
2. Select a lookup CSV file — a single column of specimen codes, no header,
   in the order you want them to appear in the output.
3. Choose an output file path.
4. Click "Run Lookup".

The output CSV will contain exactly one row per specimen code in the lookup
list, in that order, ready to paste into your Google Sheet.

Duplicate handling
------------------
If a specimen code appears more than once in the results file (which should
not normally happen), all matching rows are included in the output and a
warning is shown so you can investigate before using the data.

Missing codes
-------------
If a code from the lookup list is not found in the results file, a row of
blank values is written in its place and the code is listed in the warnings
so nothing silently shifts out of alignment in your sheet.
"""

import pandas as pd
import tkinter as tk
from tkinter import filedialog as fd, StringVar, scrolledtext
from pathlib import Path
from threading import Thread


# ── Core lookup logic ─────────────────────────────────────────────────────────

def run_lookup(results_path: str, lookup_path: str, output_path: str) -> list[str]:
    """
    Match lookup codes against the results file and write an ordered output CSV.

    Returns a list of warning strings (empty if everything matched cleanly).
    """
    warnings = []

    # ── Load results file ──────────────────────────────────────────────────
    df_results = pd.read_csv(results_path, index_col=0)

    # Normalise specimen codes: strip whitespace, upper-case
    df_results['Specimen Code'] = df_results['Specimen Code'].astype(str).str.strip().str.upper()

    # ── Check for duplicates in the results file ───────────────────────────
    dupes = df_results[df_results.duplicated(subset='Specimen Code', keep=False)]
    if not dupes.empty:
        dupe_codes = dupes['Specimen Code'].unique().tolist()
        warnings.append(
            f"WARNING — duplicate specimen codes found in results file "
            f"(all matching rows will be included):\n  {', '.join(dupe_codes)}"
        )

    # ── Load lookup codes ──────────────────────────────────────────────────
    df_lookup = pd.read_csv(lookup_path, header=None, dtype=str)
    lookup_codes = df_lookup.iloc[:, 0].str.strip().str.upper().tolist()

    # ── Build output row by row to preserve lookup order ──────────────────
    output_frames = []
    missing_codes = []

    for code in lookup_codes:
        matches = df_results[df_results['Specimen Code'] == code]
        if matches.empty:
            # Insert a blank row so downstream alignment is preserved
            blank = pd.DataFrame([[''] * len(df_results.columns)],
                                  columns=df_results.columns)
            blank['Specimen Code'] = code
            output_frames.append(blank)
            missing_codes.append(code)
        else:
            output_frames.append(matches)

    if missing_codes:
        warnings.append(
            f"WARNING — {len(missing_codes)} code(s) not found in results file "
            f"(blank rows inserted to preserve column alignment):\n"
            f"  {', '.join(missing_codes)}"
        )

    df_output = pd.concat(output_frames, ignore_index=True)
    df_output.to_csv(output_path, index=False)

    return warnings


# ── GUI ───────────────────────────────────────────────────────────────────────

def select_file(string_var: StringVar, title: str, filetypes):
    path = fd.askopenfilename(title=title, filetypes=filetypes)
    if path:
        string_var.set(path)


def select_save_path(string_var: StringVar):
    path = fd.asksaveasfilename(
        title="Save output as",
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv")],
    )
    if path:
        string_var.set(path)


def log(text_widget, message: str):
    text_widget.configure(state="normal")
    text_widget.insert(tk.END, message + "\n")
    text_widget.see(tk.END)
    text_widget.configure(state="disabled")


def run_in_thread(results_var, lookup_var, output_var, log_widget, btn):
    results_path = results_var.get().strip()
    lookup_path  = lookup_var.get().strip()
    output_path  = output_var.get().strip()

    # ── Validate inputs before spinning up a thread ───────────────────────
    errors = []
    if not results_path:
        errors.append("No results file selected.")
    if not lookup_path:
        errors.append("No lookup file selected.")
    if not output_path:
        errors.append("No output path specified.")
    if errors:
        for e in errors:
            log(log_widget, f"ERROR — {e}")
        return

    btn.config(state="disabled")
    log(log_widget, f"Results file : {results_path}")
    log(log_widget, f"Lookup file  : {lookup_path}")
    log(log_widget, f"Output file  : {output_path}")
    log(log_widget, "Running lookup…")

    def task():
        try:
            warnings = run_lookup(results_path, lookup_path, output_path)
            root.after(0, lambda: _finish(warnings, log_widget, btn))
        except Exception as exc:
            root.after(0, lambda: _error(exc, log_widget, btn))

    Thread(target=task, daemon=True).start()


def _finish(warnings, log_widget, btn):
    if warnings:
        for w in warnings:
            log(log_widget, w)
    else:
        log(log_widget, "All codes matched cleanly — no warnings.")
    log(log_widget, "Done. Output file written.\n")
    btn.config(state="normal")


def _error(exc, log_widget, btn):
    log(log_widget, f"ERROR — {exc}\n")
    btn.config(state="normal")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Specimen Lookup")
    root.attributes("-topmost", True)
    root.resizable(True, False)

    pad = {"padx": 10, "pady": 4}

    results_var = StringVar()
    lookup_var  = StringVar()
    output_var  = StringVar()

    # ── Row 0: results file ────────────────────────────────────────────────
    tk.Button(
        root, text="Select results CSV\n(tensile or flexural)",
        command=lambda: select_file(
            results_var, "Select results CSV",
            [("CSV files", "*.csv"), ("All files", "*.*")]
        ), width=24,
    ).grid(row=0, column=0, **pad, sticky="w")
    tk.Label(root, textvariable=results_var, anchor="w").grid(
        row=0, column=1, **pad, sticky="ew")

    # ── Row 1: lookup file ─────────────────────────────────────────────────
    tk.Button(
        root, text="Select lookup CSV\n(specimen codes, no header)",
        command=lambda: select_file(
            lookup_var, "Select lookup CSV",
            [("CSV files", "*.csv"), ("All files", "*.*")]
        ), width=24,
    ).grid(row=1, column=0, **pad, sticky="w")
    tk.Label(root, textvariable=lookup_var, anchor="w").grid(
        row=1, column=1, **pad, sticky="ew")

    # ── Row 2: output path ─────────────────────────────────────────────────
    tk.Button(
        root, text="Set output file path",
        command=lambda: select_save_path(output_var),
        width=24,
    ).grid(row=2, column=0, **pad, sticky="w")
    tk.Label(root, textvariable=output_var, anchor="w").grid(
        row=2, column=1, **pad, sticky="ew")

    tk.Frame(root, height=2, bd=1, relief="sunken").grid(
        row=3, column=0, columnspan=2, padx=10, pady=6, sticky="ew")

    # ── Row 4: run button ──────────────────────────────────────────────────
    btn_run = tk.Button(root, text="Run Lookup", width=24)
    btn_run.config(
        command=lambda: run_in_thread(
            results_var, lookup_var, output_var, log_widget, btn_run
        )
    )
    btn_run.grid(row=4, column=0, **pad, sticky="w")

    # ── Row 5: log output ──────────────────────────────────────────────────
    log_widget = scrolledtext.ScrolledText(root, height=10, width=80, state="disabled")
    log_widget.grid(row=5, column=0, columnspan=2, padx=10, pady=(4, 10), sticky="ew")

    root.columnconfigure(1, weight=1)
    root.mainloop()