"""
Foam Compression Test Plotter
Loads multiple *_log.csv files, optionally correlates with a *.rsl.csv file,
and plots Load vs Distance curves with grouped coloring.
"""

import os
import re
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import colorsys
from datetime import datetime


# ── File Parsing ──────────────────────────────────────────────────────────────

def parse_log_csv(filepath):
    """
    Parse a *_log.csv file produced by IntelliMESUR.
    Returns (datetime_from_filename, DataFrame with columns Load, Distance).
    """
    fname = os.path.basename(filepath)

    # Extract datetime from filename
    # Real instrument format: "Rockwell Foam Compression-Apr-23-2026-11-48-00.log.csv"
    # (dot before log, spaces in name — uploaded copies may have underscores substituted)
    dt_match = re.search(
        r'([A-Za-z]{3}-\d{1,2}-\d{4}-\d{2}-\d{2}-\d{2})\.log\.csv', fname, re.IGNORECASE
    )
    file_dt = None
    if dt_match:
        try:
            file_dt = datetime.strptime(dt_match.group(1), "%b-%d-%Y-%H-%M-%S")
        except ValueError:
            pass

    # Find the header row ("Reading\tLoad ...")
    header_line = None
    with open(filepath, encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if line.strip().lower().startswith("reading"):
                header_line = i
                break

    if header_line is None:
        raise ValueError(f"Could not find data header in {fname}")

    df = pd.read_csv(
        filepath,
        sep="\t",
        skiprows=header_line,
        engine="python",
        encoding="utf-8",
        on_bad_lines="skip",
    )
    df.columns = [c.strip() for c in df.columns]

    # Identify Load and Distance columns (flexible matching)
    load_col = next((c for c in df.columns if "load" in c.lower()), None)
    dist_col = next((c for c in df.columns if "distance" in c.lower()), None)

    if load_col is None or dist_col is None:
        raise ValueError(f"Could not find Load/Distance columns in {fname}. "
                         f"Found: {list(df.columns)}")

    df = df[[load_col, dist_col]].copy()
    df.columns = ["Load", "Distance"]
    df = df.apply(pd.to_numeric, errors="coerce").dropna()

    # Distance comes in negative from the instrument, flip it to positive
    df["Distance"] = -df["Distance"]

    # ── Regularise onto a clean monotonic x-grid ──────────────────────────────
    # The instrument can record duplicate or slightly non-monotonic distance
    # values (multiple readings at the same step, or brief crosshead reversal).
    # These cause polyfit to fail with near-zero x-variance.
    # Fix: keep only the forward-moving portion, then interpolate onto a uniform grid.
    df = df.sort_values("Distance").reset_index(drop=True)

    # Remove any rows where distance does not strictly increase
    # (keeps the first occurrence of each distance value)
    df = df[df["Distance"] > df["Distance"].shift(1).fillna(-1)].reset_index(drop=True)

    if len(df) >= 2:
        import numpy as np
        from scipy.interpolate import interp1d
        n_points = len(df)
        x_uniform = np.linspace(df["Distance"].iloc[0], df["Distance"].iloc[-1], n_points)
        interp_fn = interp1d(df["Distance"], df["Load"], kind="linear", bounds_error=False)
        df_interp = pd.DataFrame({"Distance": x_uniform, "Load": interp_fn(x_uniform)})
        df = df_interp

    # ── Smooth the load signal with Savitzky-Golay ────────────────────────────
    from scipy.signal import savgol_filter
    if len(df) >= 51:
        df["Load_smooth"] = savgol_filter(df["Load"], window_length=51, polyorder=3)
    else:
        df["Load_smooth"] = df["Load"]

    return file_dt, df


def parse_rsl_csv(filepath):
    """
    Parse an IntelliMESUR *.rsl.csv file (tab-separated, same header block
    as log files).

    Format observed:
        Run No. | Status | Description | Date         | Time     | ...
        1       | Complete | W2.1      | Apr 22, 2026 | 15:50:29 | ...
        18      | Invalid (Stop Button Pressed) | Apr 23, 2026 | 11:43:15 | ...

    Invalid rows have no Description value — the date/time shift one column left.
    We detect and skip those rows.

    Returns a list of dicts: {datetime, description}
    where datetime is the test START time parsed from the RSL row.
    """
    # Find the header row (contains "Run No." or "Description")
    header_line = None
    with open(filepath, encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            stripped = line.strip().lower()
            if stripped.startswith("run no") or "description" in stripped:
                header_line = i
                break

    if header_line is None:
        raise ValueError("Could not find data header row in RSL file.")

    df = pd.read_csv(
        filepath,
        sep="\t",
        skiprows=header_line,
        engine="python",
        encoding="utf-8",
        on_bad_lines="skip",
    )
    df.columns = [str(c).strip() for c in df.columns]

    # Stop at any trailing statistics/blank section
    # (rows where "Run No." column is not numeric)
    run_col = next((c for c in df.columns if "run" in c.lower()), None)
    if run_col:
        df = df[pd.to_numeric(df[run_col], errors="coerce").notna()].copy()

    desc_col = next((c for c in df.columns if "description" in c.lower()), None)
    date_col = next((c for c in df.columns if c.strip().lower() == "date"), None)
    time_col = next((c for c in df.columns if c.strip().lower() == "time"), None)

    if desc_col is None:
        raise ValueError(f"No 'Description' column found in RSL file. "
                         f"Columns: {list(df.columns)}")

    records = []
    for _, row in df.iterrows():
        desc = str(row.get(desc_col, "")).strip()

        # Skip rows with no valid description (Invalid/aborted runs).
        # When Status is very long (e.g. "Invalid (Stop Button Pressed)"),
        # the Date value shifts into the Description column — detect by
        # checking if the description looks like a date string.
        if not desc or desc.lower() in ("nan", "") or desc.lower().startswith("invalid"):
            continue
        # Also skip if description looks like a date (e.g. "Apr 23, 2026")
        if re.match(r'^[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4}$', desc):
            continue

        dt = None
        if date_col and time_col:
            date_str = str(row.get(date_col, "")).strip()
            time_str = str(row.get(time_col, "")).strip()
            if date_str and time_str and date_str.lower() != "nan":
                dt_str = f"{date_str} {time_str}"
                # IntelliMESUR format: "Apr 22, 2026 15:50:29"
                for fmt in (
                    "%b %d, %Y %H:%M:%S",   # Apr 22, 2026 15:50:29  ← primary
                    "%B %d, %Y %H:%M:%S",   # April 22, 2026 15:50:29
                    "%m/%d/%Y %H:%M:%S",    # 04/22/2026 15:50:29
                    "%Y-%m-%d %H:%M:%S",    # 2026-04-22 15:50:29
                    "%b %d, %Y %I:%M:%S %p",# Apr 22, 2026 03:50:29 PM
                ):
                    try:
                        dt = datetime.strptime(dt_str, fmt)
                        break
                    except ValueError:
                        continue

        records.append({"datetime": dt, "description": desc})

    return records


# ── Color Generation ─────────────────────────────────────────────────────────

def group_descriptions(descriptions):
    """Group descriptions by their leading letter."""
    groups = {}
    for desc in descriptions:
        key = desc[0].upper() if desc else "?"
        groups.setdefault(key, []).append(desc)
    return groups


def make_palette(groups):
    """
    Assign colors by group letter:
      B → blues, G → greens, W → light grays, others → Set1 fallback.
    Members within a group vary in lightness (dark to light).
    """
    # (hue, sat) in HLS for known groups; W uses zero saturation (gray)
    GROUP_HLS = {
        "B": (0.60, 0.75),   # blue hue, high saturation
        "G": (0.35, 0.70),   # green hue, high saturation
        "W": (0.00, 0.00),   # achromatic — saturation=0 gives pure gray
    }
    # Lightness ranges: darker end → lighter end
    LIGHTNESS_RANGE = {
        "B": (0.25, 0.60),
        "G": (0.25, 0.60),
        "W": (0.55, 0.80),   # light-to-medium gray (never too dark or pure white)
    }
    import seaborn as sns
    fallback = sns.color_palette("Set1", n_colors=9)
    fallback_idx = 0

    palette = {}
    for key in sorted(groups.keys()):
        members = groups[key]
        n = len(members)
        if key in GROUP_HLS:
            hue, sat = GROUP_HLS[key]
            lo, hi = LIGHTNESS_RANGE[key]
            lightnesses = [lo + (hi - lo) * i / (n - 1) for i in range(n)] if n > 1 else [(lo + hi) / 2]
            for m_idx, desc in enumerate(members):
                palette[desc] = colorsys.hls_to_rgb(hue, lightnesses[m_idx], sat)
        else:
            bh, bl, bs = colorsys.rgb_to_hls(*fallback[fallback_idx % len(fallback)])
            fallback_idx += 1
            lightnesses = [0.25 + 0.40 * i / (n - 1) for i in range(n)] if n > 1 else [bl]
            for m_idx, desc in enumerate(members):
                palette[desc] = colorsys.hls_to_rgb(bh, lightnesses[m_idx], bs)

    return palette


# ── Analysis ─────────────────────────────────────────────────────────────────

def find_initial_linear_region(df, max_distance=0.1, deviation_pct=0.05):
    """
    Find the initial linear stiffness region (Distance < max_distance).

    Starting from a small offset to skip seating noise, fit a running OLS
    slope and stop when the residual of a new point exceeds deviation_pct
    of the predicted value. Return the slope (lbF/in) and the x range used.

    deviation_pct: fraction of predicted load at which a point is considered
                   to deviate from linearity (default 5%).
    """
    from numpy.polynomial import polynomial as P
    import numpy as np

    region = df[df["Distance"] < max_distance].copy()

    # Skip the initial seating noise — start once load exceeds 10% of region max
    load_thresh = region["Load_smooth"].max() * 0.10
    region = region[region["Load_smooth"] >= load_thresh].reset_index(drop=True)

    if len(region) < 10:
        return None, None, None

    x = region["Distance"].values
    y = region["Load_smooth"].values

    # Expand the linear window one point at a time; stop when residual > threshold
    MIN_POINTS = 10
    MIN_X_SPAN = 1e-4  # inches — skip fit if x range is too narrow for solver

    last_good = MIN_POINTS
    for end in range(MIN_POINTS + 1, len(x)):
        if (x[end - 1] - x[0]) < MIN_X_SPAN:
            last_good = end
            continue
        try:
            coeffs = np.polyfit(x[:end], y[:end], 1)
        except np.linalg.LinAlgError:
            break
        slope, intercept = coeffs
        if not np.isfinite(slope):
            break
        y_pred = slope * x[:end] + intercept
        residuals_pct = np.abs((y[:end] - y_pred) / np.maximum(np.abs(y_pred), 1e-6))
        if residuals_pct.max() > deviation_pct:
            break
        last_good = end

    x_fit = x[:last_good]
    y_fit = y[:last_good]
    if (x_fit[-1] - x_fit[0]) < MIN_X_SPAN:
        return None, None, None
    try:
        coeffs = np.polyfit(x_fit, y_fit, 1)
    except np.linalg.LinAlgError:
        return None, None, None
    slope = coeffs[0]
    if not np.isfinite(slope):
        return None, None, None
    return slope, x_fit[0], x_fit[-1]


def find_secondary_linear_region(df, start_distance=0.15, tertiary_start=0.7, deviation_pct=0.05):
    """
    Find the secondary linear stiffness region (Distance >= start_distance,
    up to min(peak load, tertiary_start) to avoid being pulled into the
    tertiary stiffening region).
    """
    import numpy as np

    peak_idx = df["Load_smooth"].idxmax()
    region = df[(df["Distance"] >= start_distance) &
                (df["Distance"] <  tertiary_start) &
                (df.index <= peak_idx)].copy().reset_index(drop=True)

    if len(region) < 10:
        return None, None, None

    x = region["Distance"].values
    y = region["Load_smooth"].values

    MIN_POINTS = 10
    MIN_X_SPAN = 1e-4

    last_good = MIN_POINTS
    for end in range(MIN_POINTS + 1, len(x)):
        if (x[end - 1] - x[0]) < MIN_X_SPAN:
            last_good = end
            continue
        try:
            coeffs = np.polyfit(x[:end], y[:end], 1)
        except np.linalg.LinAlgError:
            break
        slope, intercept = coeffs
        if not np.isfinite(slope):
            break
        y_pred = slope * x[:end] + intercept
        residuals_pct = np.abs((y[:end] - y_pred) / np.maximum(np.abs(y_pred), 1e-6))
        if residuals_pct.max() > deviation_pct:
            break
        last_good = end

    x_fit = x[:last_good]
    y_fit = y[:last_good]
    if (x_fit[-1] - x_fit[0]) < MIN_X_SPAN:
        return None, None, None
    try:
        coeffs = np.polyfit(x_fit, y_fit, 1)
    except np.linalg.LinAlgError:
        return None, None, None
    slope = coeffs[0]
    if not np.isfinite(slope):
        return None, None, None
    return slope, x_fit[0], x_fit[-1]


def find_tertiary_linear_region(df, search_start=0.60, deviation_pct=0.05):
    """
    Detect the tertiary stiffness region that appears at high compression.

    Strategy:
      1. Compute a rolling slope (derivative) on the smoothed load curve.
      2. Starting from search_start, find where the slope increases
         significantly above the secondary-region slope — indicating a new,
         stiffer regime.  This is the transition point.
      3. From that transition point, run the same expanding-window linear
         regression as the other region finders to get the tertiary stiffness.

    Returns (slope, x_start, x_end, transition_x) or (None, None, None, None).
    """
    import numpy as np

    peak_idx = df["Load_smooth"].idxmax()
    full = df[df.index <= peak_idx].copy().reset_index(drop=True)
    search = full[full["Distance"] >= search_start].reset_index(drop=True)

    if len(search) < 20:
        return None, None, None, None

    x_all = full["Distance"].values
    y_all = full["Load_smooth"].values
    x_s   = search["Distance"].values
    y_s   = search["Load_smooth"].values

    # Rolling derivative over a window of ~20 points
    WIN = 20
    slopes_rolling = []
    for i in range(len(x_s) - WIN):
        seg_x = x_s[i:i + WIN]
        seg_y = y_s[i:i + WIN]
        if (seg_x[-1] - seg_x[0]) < 1e-4:
            slopes_rolling.append(np.nan)
            continue
        c = np.polyfit(seg_x, seg_y, 1)
        slopes_rolling.append(c[0])
    slopes_rolling = np.array(slopes_rolling)

    # Baseline slope: median of first third of the search window
    # (representative of secondary region slope in this range)
    n_base = max(10, len(slopes_rolling) // 3)
    valid  = slopes_rolling[:n_base][np.isfinite(slopes_rolling[:n_base])]
    if len(valid) == 0:
        return None, None, None, None
    baseline_slope = np.median(valid)

    # Transition: first point where rolling slope exceeds baseline by >25%
    threshold = baseline_slope * 1.25
    transition_idx = None
    for i, s in enumerate(slopes_rolling):
        if np.isfinite(s) and s > threshold:
            transition_idx = i
            break

    if transition_idx is None:
        return None, None, None, None

    transition_x = x_s[transition_idx]

    # Now fit the tertiary region from transition_x to peak
    tert = full[full["Distance"] >= transition_x].reset_index(drop=True)
    if len(tert) < 10:
        return None, None, None, None

    x = tert["Distance"].values
    y = tert["Load_smooth"].values

    MIN_POINTS = 10
    MIN_X_SPAN = 1e-4

    last_good = MIN_POINTS
    for end in range(MIN_POINTS + 1, len(x)):
        if (x[end - 1] - x[0]) < MIN_X_SPAN:
            last_good = end
            continue
        try:
            coeffs = np.polyfit(x[:end], y[:end], 1)
        except np.linalg.LinAlgError:
            break
        slope, intercept = coeffs
        if not np.isfinite(slope):
            break
        y_pred = slope * x[:end] + intercept
        residuals_pct = np.abs((y[:end] - y_pred) / np.maximum(np.abs(y_pred), 1e-6))
        if residuals_pct.max() > deviation_pct:
            break
        last_good = end

    x_fit = x[:last_good]
    y_fit = y[:last_good]
    if (x_fit[-1] - x_fit[0]) < MIN_X_SPAN:
        return None, None, None, None
    try:
        coeffs = np.polyfit(x_fit, y_fit, 1)
    except np.linalg.LinAlgError:
        return None, None, None, None
    slope = coeffs[0]
    if not np.isfinite(slope):
        return None, None, None, None
    return slope, x_fit[0], x_fit[-1], transition_x


def analyze_curve(df, platen_area=15.5, vacuum_ref_psi=13.56):
    """
    Run all analyses on a single load-distance DataFrame.
    Returns a dict with keys:
        initial_stiffness,  initial_x0,     initial_x1,
        secondary_stiffness, secondary_x0,  secondary_x1,
        tertiary_stiffness,  tertiary_x0,   tertiary_x1,  tertiary_transition_x,
        peak_load,
        vacuum_ref_distance  (distance at which load reaches vacuum reference load)
    """
    import numpy as np
    from scipy.interpolate import interp1d

    k1, x1_0, x1_1 = find_initial_linear_region(df)
    k2, x2_0, x2_1 = find_secondary_linear_region(df)
    k3, x3_0, x3_1, x3_trans = find_tertiary_linear_region(df)
    peak_load = df["Load_smooth"].max()

    # Distance at vacuum reference load
    ref_load = vacuum_ref_psi * platen_area
    vac_dist = None
    try:
        # Only interpolate up to peak to avoid the descending tail
        peak_idx = df["Load_smooth"].idxmax()
        sub = df.iloc[:peak_idx + 1]
        if sub["Load_smooth"].max() >= ref_load >= sub["Load_smooth"].min():
            f = interp1d(sub["Load_smooth"], sub["Distance"],
                         kind="linear", bounds_error=False)
            vac_dist = float(f(ref_load))
    except Exception:
        pass

    return {
        "initial_stiffness":      k1,
        "initial_x0":             x1_0,
        "initial_x1":             x1_1,
        "secondary_stiffness":    k2,
        "secondary_x0":           x2_0,
        "secondary_x1":           x2_1,
        "tertiary_stiffness":     k3,
        "tertiary_x0":            x3_0,
        "tertiary_x1":            x3_1,
        "tertiary_transition_x":  x3_trans,
        "peak_load":              peak_load,
        "vacuum_ref_distance":    vac_dist,
    }


# ── GUI ───────────────────────────────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Foam Compression Plotter")
        self.resizable(False, False)
        self.log_files = []   # list of file paths
        self.rsl_file = None  # single file path

        self._build_ui()

    def _build_ui(self):
        pad = dict(padx=10, pady=6)

        # ── Log files section ────────────────────────────────────────────────
        lf1 = ttk.LabelFrame(self, text="Step 1 – Select log files (*_log.csv)")
        lf1.grid(row=0, column=0, columnspan=2, sticky="ew", **pad)

        self.log_listbox = tk.Listbox(lf1, width=70, height=8, selectmode=tk.EXTENDED)
        self.log_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4, pady=4)

        sb = ttk.Scrollbar(lf1, orient=tk.VERTICAL, command=self.log_listbox.yview)
        sb.pack(side=tk.LEFT, fill=tk.Y)
        self.log_listbox.configure(yscrollcommand=sb.set)

        btn_frame1 = ttk.Frame(lf1)
        btn_frame1.pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame1, text="Add files…", command=self._add_log_files).pack(fill=tk.X, pady=2)
        ttk.Button(btn_frame1, text="Remove selected", command=self._remove_log_files).pack(fill=tk.X, pady=2)
        ttk.Button(btn_frame1, text="Clear all", command=self._clear_log_files).pack(fill=tk.X, pady=2)

        # ── RSL file section ─────────────────────────────────────────────────
        lf2 = ttk.LabelFrame(self, text="Step 2 (optional) – Select results file (*.rsl.csv)")
        lf2.grid(row=1, column=0, columnspan=2, sticky="ew", **pad)

        self.rsl_label = ttk.Label(lf2, text="No file selected", foreground="gray")
        self.rsl_label.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=4)

        ttk.Button(lf2, text="Browse…", command=self._pick_rsl).pack(side=tk.LEFT, padx=4, pady=4)
        ttk.Button(lf2, text="Clear", command=self._clear_rsl).pack(side=tk.LEFT, padx=4, pady=4)

        # ── Match status ─────────────────────────────────────────────────────
        lf3 = ttk.LabelFrame(self, text="Match status")
        lf3.grid(row=2, column=0, columnspan=2, sticky="ew", **pad)

        self.status_text = tk.Text(lf3, width=70, height=5, state=tk.DISABLED,
                                   font=("Courier", 9), bg="#f0f0f0", relief=tk.FLAT)
        self.status_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4, pady=4)
        sb2 = ttk.Scrollbar(lf3, orient=tk.VERTICAL, command=self.status_text.yview)
        sb2.pack(side=tk.LEFT, fill=tk.Y)
        self.status_text.configure(yscrollcommand=sb2.set)

        # ── Plot button ──────────────────────────────────────────────────────
        ttk.Button(self, text="Generate Plot", command=self._plot).grid(
            row=3, column=0, columnspan=2, pady=12)

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _add_log_files(self):
        paths = filedialog.askopenfilenames(
            title="Select log CSV files",
            filetypes=[("Log CSV files", "*.csv"), ("All files", "*.*")]
        )
        for p in paths:
            if p not in self.log_files:
                self.log_files.append(p)
                self.log_listbox.insert(tk.END, os.path.basename(p))

    def _remove_log_files(self):
        selected = list(self.log_listbox.curselection())
        for i in reversed(selected):
            self.log_listbox.delete(i)
            self.log_files.pop(i)

    def _clear_log_files(self):
        self.log_listbox.delete(0, tk.END)
        self.log_files.clear()

    def _pick_rsl(self):
        path = filedialog.askopenfilename(
            title="Select RSL CSV file",
            filetypes=[("RSL CSV files", "*.rsl.csv"), ("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if path:
            self.rsl_file = path
            self.rsl_label.config(text=os.path.basename(path), foreground="black")

    def _clear_rsl(self):
        self.rsl_file = None
        self.rsl_label.config(text="No file selected", foreground="gray")

    # ── Plotting ──────────────────────────────────────────────────────────────

    def _plot(self):
        if not self.log_files:
            messagebox.showwarning("No files", "Please add at least one log CSV file.")
            return

        # ── Parse log files ───────────────────────────────────────────────────
        parsed = []
        errors = []
        for fp in self.log_files:
            try:
                dt, df = parse_log_csv(fp)
                parsed.append({"path": fp, "datetime": dt, "df": df,
                                "fname": os.path.basename(fp)})
            except Exception as e:
                errors.append(f"{os.path.basename(fp)}: {e}")

        if errors:
            messagebox.showerror("Parse errors", "\n".join(errors))
            if not parsed:
                return

        # ── Build label map: filepath -> display label ────────────────────────
        label_map = {}

        if self.rsl_file:
            try:
                rsl_records = parse_rsl_csv(self.rsl_file)
            except Exception as e:
                messagebox.showerror("RSL parse error", str(e))
                rsl_records = []

            if rsl_records:
                rsl_with_dt = [r for r in rsl_records if r["datetime"] is not None]
                used_rsl = set()

                for item in parsed:
                    item_dt = item["datetime"]
                    if item_dt is None:
                        continue
                    best_idx = None
                    best_delta = float("inf")
                    for i, r in enumerate(rsl_with_dt):
                        if i in used_rsl:
                            continue
                        delta = abs((item_dt - r["datetime"]).total_seconds())
                        if delta < best_delta:
                            best_delta = delta
                            best_idx = i
                    # Accept match within 60 s — RSL row timestamps match
                    # the timestamp embedded in each log filename exactly
                    if best_idx is not None and best_delta <= 60:
                        label_map[item["path"]] = rsl_with_dt[best_idx]["description"]
                        used_rsl.add(best_idx)

        # Fallback: use a short form of the filename for unmatched files
        unmatched = []
        for item in parsed:
            if item["path"] not in label_map:
                if self.rsl_file:
                    # Has RSL but no match found — exclude from plot and warn
                    unmatched.append(item["fname"])
                else:
                    # No RSL provided — use filename as label
                    short = item["fname"]
                    short = re.sub(r'\.log\.csv$', '', short, flags=re.IGNORECASE)
                    label_map[item["path"]] = short

        if unmatched:
            messagebox.showwarning(
                "Unmatched files",
                "The following files had no matching entry in the RSL file and will not be plotted:\n\n"
                + "\n".join(unmatched)
            )

        # Only plot files that have a label
        to_plot = [item for item in parsed if item["path"] in label_map]
        if not to_plot:
            messagebox.showerror("Nothing to plot", "No files could be matched to RSL descriptions.")
            return

        # ── Update status panel ───────────────────────────────────────────────
        status_lines = []
        for item in to_plot:
            label = label_map[item["path"]]
            dt_str = item["datetime"].strftime("%b %d %Y %H:%M:%S") if item["datetime"] else "no timestamp"
            matched = label not in item["fname"]  # crude check: label is short if matched
            tag = "✓" if (self.rsl_file and item["datetime"] and
                          not re.search(r'\d{4}-\d{2}-\d{2}', label)) else "–"
            status_lines.append(f"{tag}  {item['fname']}  →  {label}")
        self.status_text.configure(state=tk.NORMAL)
        self.status_text.delete("1.0", tk.END)
        self.status_text.insert(tk.END, "\n".join(status_lines))
        self.status_text.configure(state=tk.DISABLED)

        # ── Color palette ─────────────────────────────────────────────────────
        labels = [label_map[item["path"]] for item in to_plot]
        groups = group_descriptions(labels)
        palette = make_palette(groups)

        # ── Analyze each curve ────────────────────────────────────────────────
        import numpy as np
        results = []
        for item in to_plot:
            item["analysis"] = analyze_curve(item["df"])
            results.append({
                "Sample":                        label_map[item["path"]],
                "Initial Stiffness (lbF/in)":    item["analysis"]["initial_stiffness"],
                "Secondary Stiffness (lbF/in)":  item["analysis"]["secondary_stiffness"],
                "Tertiary Stiffness (lbF/in)":   item["analysis"]["tertiary_stiffness"],
                "Tertiary Transition (in)":       item["analysis"]["tertiary_transition_x"],
                "Vacuum Ref Distance (in)":       item["analysis"]["vacuum_ref_distance"],
                "Peak Load (lbF)":               item["analysis"]["peak_load"],
            })

        # ── Draw ──────────────────────────────────────────────────────────────
        PLATEN_AREA_IN2  = 15.5          # in²
        VACUUM_REF_INHG  = 27.6          # inHg gauge
        VACUUM_REF_PSI   = VACUUM_REF_INHG * 0.4912  # 13.56 psi

        fig, ax = plt.subplots(figsize=(11, 6))
        fig.patch.set_facecolor("#f8f8f8")
        ax.set_facecolor("#ffffff")

        for item in to_plot:
            label = label_map[item["path"]]
            color = palette.get(label, (0.3, 0.3, 0.3))
            ax.plot(item["df"]["Distance"], item["df"]["Load"],
                    color=color, label=label, linewidth=1.8, alpha=0.9)

            a = item["analysis"]

            # Initial stiffness regression line
            if a["initial_stiffness"] is not None:
                x0, x1 = a["initial_x0"], a["initial_x1"]
                seg = item["df"][(item["df"]["Distance"] >= x0) &
                                  (item["df"]["Distance"] <= x1)]
                coeffs = np.polyfit(seg["Distance"], seg["Load"], 1)
                xs = np.array([x0, x1])
                ax.plot(xs, np.polyval(coeffs, xs),
                        color=color, linewidth=2.5, linestyle="--", alpha=1.0)

            # Secondary stiffness regression line
            if a["secondary_stiffness"] is not None:
                x0, x1 = a["secondary_x0"], a["secondary_x1"]
                seg = item["df"][(item["df"]["Distance"] >= x0) &
                                  (item["df"]["Distance"] <= x1)]
                coeffs = np.polyfit(seg["Distance"], seg["Load"], 1)
                xs = np.array([x0, x1])
                ax.plot(xs, np.polyval(coeffs, xs),
                        color=color, linewidth=2.5, linestyle=":", alpha=1.0)

            # Tertiary stiffness regression line
            if a["tertiary_stiffness"] is not None:
                x0, x1 = a["tertiary_x0"], a["tertiary_x1"]
                seg = item["df"][(item["df"]["Distance"] >= x0) &
                                  (item["df"]["Distance"] <= x1)]
                coeffs = np.polyfit(seg["Distance"], seg["Load"], 1)
                xs = np.array([x0, x1])
                ax.plot(xs, np.polyval(coeffs, xs),
                        color=color, linewidth=2.5, linestyle=(0, (3, 1, 1, 1)), alpha=1.0)

        # ── Twin y-axis: pressure (psi) ────────────────────────────────────────
        ax2 = ax.twinx()
        # Keep ax2 y-limits in sync with ax via the fixed area conversion
        lbf_min, lbf_max = ax.get_ylim()
        ax2.set_ylim(lbf_min / PLATEN_AREA_IN2, lbf_max / PLATEN_AREA_IN2)
        ax2.set_ylabel("Pressure (psi)", fontsize=12)

        # Sync limits whenever the primary axis is panned/zoomed
        def _sync_pressure(event_ax):
            y0, y1 = ax.get_ylim()
            ax2.set_ylim(y0 / PLATEN_AREA_IN2, y1 / PLATEN_AREA_IN2)
            fig.canvas.draw_idle()
        ax.callbacks.connect("ylim_changed", lambda _: _sync_pressure(ax))

        # Reference line at vacuum operating pressure
        ref_lbf = VACUUM_REF_PSI * PLATEN_AREA_IN2
        ax.axhline(ref_lbf, color="black", linewidth=1.2, linestyle="-.",
                   label=f"Vacuum ref ({VACUUM_REF_INHG} inHg = {VACUUM_REF_PSI:.1f} psi)")

        # Legend for line styles
        from matplotlib.lines import Line2D
        style_legend = [
            Line2D([0], [0], color="gray", linewidth=1.8, alpha=0.9,  label="Raw data"),
            Line2D([0], [0], color="gray", linewidth=2.5, linestyle="--", label="Initial stiffness fit"),
            Line2D([0], [0], color="gray", linewidth=2.5, linestyle=":",  label="Secondary stiffness fit"),
            Line2D([0], [0], color="gray", linewidth=2.5, linestyle=(0, (3, 1, 1, 1)), label="Tertiary stiffness fit"),
            Line2D([0], [0], color="black", linewidth=1.2, linestyle="-.",
                   label=f"Vacuum ref ({VACUUM_REF_INHG} inHg = {VACUUM_REF_PSI:.1f} psi)"),
        ]
        # Sample legend (exclude the reference line which goes in the style legend)
        sample_handles, sample_labels = ax.get_legend_handles_labels()
        # Drop the reference line entry (last item added to ax)
        sample_handles = sample_handles[:-1]
        sample_labels  = sample_labels[:-1]
        first_legend = ax.legend(handles=sample_handles, labels=sample_labels,
                                 loc="upper left", fontsize=9, framealpha=0.9,
                                 title="Sample", title_fontsize=10)
        ax.add_artist(first_legend)
        ax.legend(handles=style_legend, loc="lower right", fontsize=9, framealpha=0.9)

        ax.set_xlabel("Distance (in)", fontsize=12)
        ax.set_ylabel("Load (lbF)", fontsize=12)
        ax.set_title("Foam Compression – Load vs Distance", fontsize=14, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()

        # ── Results table window ──────────────────────────────────────────────
        self._show_results(results)

        # ── Statistical comparison figure ─────────────────────────────────────
        self._show_statistics(results)

    def _show_statistics(self, results):
        """
        Box plots + one-way ANOVA + Tukey HSD for each of the three parameters,
        grouped by the leading letter of the sample name.
        """
        import numpy as np
        import pandas as pd
        from scipy import stats
        from itertools import combinations

        # ── Build DataFrame ───────────────────────────────────────────────────
        df = pd.DataFrame(results)
        df["Group"] = df["Sample"].str[0].str.upper()
        # Only require the core columns to keep a row in the stats frame;
        # optional columns (tertiary, vacuum dist) may be NaN and are handled per-subplot.

        params = [
            ("Initial Stiffness (lbF/in)",   "Initial Stiffness",       "lbF/in"),
            ("Secondary Stiffness (lbF/in)", "Secondary Stiffness",     "lbF/in"),
            ("Tertiary Stiffness (lbF/in)",  "Tertiary Stiffness",      "lbF/in"),
            ("Tertiary Transition (in)",     "Tertiary Transition Dist","in"),
            ("Vacuum Ref Distance (in)",     "Dist at Vacuum Ref Press","in"),
            ("Peak Load (lbF)",              "Peak Load",               "lbF"),
        ]
        groups = sorted(df["Group"].unique())
        n_groups = len(groups)

        # Colors matched to the main plot palette
        group_colors = {}
        for g in groups:
            dummy = make_palette({g: [g]})
            group_colors[g] = dummy[g]

        # ── Tukey HSD (manual, no statsmodels required) ───────────────────────
        def tukey_hsd(group_data):
            """
            Returns a dict of (A, B) -> (mean_diff, p_value) for all pairs,
            using the Tukey-Kramer studentized range distribution via scipy.
            """
            from scipy.stats import studentized_range
            all_vals = np.concatenate(list(group_data.values()))
            grand_n  = len(all_vals)
            k        = len(group_data)
            # Pooled within-group variance
            ss_w = sum((len(v) - 1) * np.var(v, ddof=1)
                       for v in group_data.values() if len(v) > 1)
            df_w = grand_n - k
            if df_w <= 0:
                return {}
            ms_w = ss_w / df_w
            results = {}
            for (a, va), (b, vb) in combinations(group_data.items(), 2):
                na, nb = len(va), len(vb)
                if na < 2 or nb < 2:
                    results[(a, b)] = (np.mean(va) - np.mean(vb), None)
                    continue
                se    = np.sqrt(ms_w * 0.5 * (1/na + 1/nb))
                if se == 0:
                    results[(a, b)] = (np.mean(va) - np.mean(vb), None)
                    continue
                q     = abs(np.mean(va) - np.mean(vb)) / se
                # p-value from studentized range distribution
                p     = 1 - studentized_range.cdf(q * np.sqrt(2), k, df_w)
                results[(a, b)] = (np.mean(va) - np.mean(vb), p)
            return results

        # ── Figure: 3 box plots, one per parameter ────────────────────────────
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes = axes.flatten()
        fig.suptitle("Parameter Comparison by Group", fontsize=14, fontweight="bold")
        fig.patch.set_facecolor("#f8f8f8")

        stat_lines = []  # collect text for the summary panel

        for ax, (col, title, unit) in zip(axes, params):
            group_data = {g: df[df["Group"] == g][col].dropna().values
                          for g in groups}
            group_data = {g: v for g, v in group_data.items() if len(v) > 0}
            active_groups = sorted(group_data.keys())

            # Box plot
            bp = ax.boxplot(
                [group_data[g] for g in active_groups],
                patch_artist=True,
                widths=0.5,
                medianprops=dict(color="black", linewidth=2),
            )
            for patch, g in zip(bp["boxes"], active_groups):
                patch.set_facecolor((*group_colors[g], 0.55))
                patch.set_edgecolor(group_colors[g])
            for element in ("whiskers", "caps", "fliers"):
                for item, g in zip(
                    [bp[element][i*2:(i+1)*2] for i in range(len(active_groups))],
                    active_groups
                ):
                    for line in item:
                        line.set_color(group_colors[g])

            # Overlay individual points (jittered)
            for x_pos, g in enumerate(active_groups, start=1):
                vals = group_data[g]
                jitter = np.random.default_rng(42).uniform(-0.08, 0.08, len(vals))
                ax.scatter(x_pos + jitter, vals,
                           color=group_colors[g], edgecolors="white",
                           linewidths=0.5, zorder=5, s=40, alpha=0.85)

            ax.set_xticks(range(1, len(active_groups) + 1))
            ax.set_xticklabels(active_groups, fontsize=11)
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.set_ylabel(unit, fontsize=10)
            ax.set_facecolor("#ffffff")
            ax.grid(axis="y", linestyle="--", alpha=0.5)

            # ── One-way ANOVA ─────────────────────────────────────────────────
            stat_lines.append(f"\n{'─'*40}")
            stat_lines.append(f"{title}")
            stat_lines.append(f"{'─'*40}")

            group_vals = [group_data[g] for g in active_groups if len(group_data[g]) >= 2]
            if len(group_vals) >= 2:
                f_stat, p_anova = stats.f_oneway(*group_vals)
                sig = "***" if p_anova < 0.001 else "**" if p_anova < 0.01 else "*" if p_anova < 0.05 else "ns"
                stat_lines.append(f"One-way ANOVA:  F = {f_stat:.3f},  p = {p_anova:.4f}  {sig}")

                # Annotate ANOVA result on plot
                ax.set_title(f"{title}\nANOVA p = {p_anova:.3f} {sig}",
                             fontsize=11, fontweight="bold")

                # ── Tukey HSD pairwise ────────────────────────────────────────
                tukey = tukey_hsd(group_data)
                stat_lines.append("Tukey HSD pairwise:")
                y_max   = max(v.max() for v in group_data.values())
                y_range = y_max - min(v.min() for v in group_data.values())
                bracket_step = y_range * 0.12

                for pair_idx, ((a, b), (diff, p)) in enumerate(tukey.items()):
                    if p is None:
                        stat_lines.append(f"  {a} vs {b}:  Δ = {diff:+.2f}  (insufficient n)")
                        continue
                    sig_pair = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                    stat_lines.append(f"  {a} vs {b}:  Δ = {diff:+.2f} {unit},  p = {p:.4f}  {sig_pair}")

                    # Significance bracket on plot
                    x1 = active_groups.index(a) + 1
                    x2 = active_groups.index(b) + 1
                    y  = y_max + bracket_step * (pair_idx + 1)
                    ax.annotate("", xy=(x2, y), xytext=(x1, y),
                                arrowprops=dict(arrowstyle="-", color="black", lw=1.2))
                    ax.plot([x1, x1], [y - bracket_step * 0.15, y], color="black", lw=1.2)
                    ax.plot([x2, x2], [y - bracket_step * 0.15, y], color="black", lw=1.2)
                    ax.text((x1 + x2) / 2, y + bracket_step * 0.05,
                            sig_pair, ha="center", va="bottom", fontsize=10)
            else:
                stat_lines.append("Insufficient groups for ANOVA (need ≥ 2 groups with n ≥ 2)")

            # Per-group descriptive stats
            stat_lines.append("Descriptive stats:")
            for g in active_groups:
                v = group_data[g]
                stat_lines.append(
                    f"  {g}:  n={len(v)},  "
                    f"mean={np.mean(v):.2f},  "
                    f"SD={np.std(v, ddof=1):.2f},  "
                    f"median={np.median(v):.2f}"
                )

        plt.tight_layout()
        plt.show()

        # ── Statistics text window ────────────────────────────────────────────
        win = tk.Toplevel(self)
        win.title("Statistical Summary")

        txt = tk.Text(win, width=60, height=30, font=("Courier", 9),
                      bg="#f8f8f8", relief=tk.FLAT, wrap=tk.NONE)
        sb_y = ttk.Scrollbar(win, orient=tk.VERTICAL,   command=txt.yview)
        sb_x = ttk.Scrollbar(win, orient=tk.HORIZONTAL, command=txt.xview)
        txt.configure(yscrollcommand=sb_y.set, xscrollcommand=sb_x.set)

        txt.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
        sb_y.grid(row=0, column=1, sticky="ns")
        sb_x.grid(row=1, column=0, sticky="ew")
        win.rowconfigure(0, weight=1)
        win.columnconfigure(0, weight=1)

        legend = (
            "Significance codes:  *** p<0.001  ** p<0.01  * p<0.05  ns p≥0.05\n"
            "Tukey HSD controls family-wise error rate across all pairwise comparisons.\n"
        )
        txt.insert(tk.END, legend + "\n".join(stat_lines))
        txt.configure(state=tk.DISABLED)

        def copy_stats():
            win.clipboard_clear()
            win.clipboard_append(legend + "\n".join(stat_lines))
        ttk.Button(win, text="Copy to Clipboard", command=copy_stats).grid(
            row=2, column=0, columnspan=2, pady=(0, 6))

    def _show_results(self, results):
        """Display analysis results in a separate scrollable table window."""
        win = tk.Toplevel(self)
        win.title("Analysis Results")

        cols = ["Sample", "Initial Stiffness (lbF/in)",
                "Secondary Stiffness (lbF/in)", "Tertiary Stiffness (lbF/in)",
                "Tertiary Transition (in)", "Vacuum Ref Distance (in)", "Peak Load (lbF)"]

        tree = ttk.Treeview(win, columns=cols, show="headings", height=min(len(results), 25))
        for col in cols:
            tree.heading(col, text=col)
            tree.column(col, width=175, anchor="center")

        for row in sorted(results, key=lambda r: r["Sample"]):
            def fmt(val, dec=1):
                return f"{val:.{dec}f}" if val is not None and val == val else "N/A"
            tree.insert("", tk.END, values=(
                row["Sample"],
                fmt(row["Initial Stiffness (lbF/in)"]),
                fmt(row["Secondary Stiffness (lbF/in)"]),
                fmt(row["Tertiary Stiffness (lbF/in)"]),
                fmt(row["Tertiary Transition (in)"], 3),
                fmt(row["Vacuum Ref Distance (in)"], 3),
                fmt(row["Peak Load (lbF)"]),
            ))

        sb = ttk.Scrollbar(win, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=sb.set)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)
        sb.pack(side=tk.LEFT, fill=tk.Y, pady=8)

        # Copy to clipboard button
        def copy_to_clipboard():
            header = "\t".join(cols)
            rows_txt = []
            for row in sorted(results, key=lambda r: r["Sample"]):
                def fmt(val, dec=1):
                    return f"{val:.{dec}f}" if val is not None and val == val else "N/A"
                rows_txt.append("\t".join([
                    row["Sample"],
                    fmt(row["Initial Stiffness (lbF/in)"]),
                    fmt(row["Secondary Stiffness (lbF/in)"]),
                    fmt(row["Tertiary Stiffness (lbF/in)"]),
                    fmt(row["Tertiary Transition (in)"], 3),
                    fmt(row["Vacuum Ref Distance (in)"], 3),
                    fmt(row["Peak Load (lbF)"]),
                ]))
            win.clipboard_clear()
            win.clipboard_append(header + "\n" + "\n".join(rows_txt))

        ttk.Button(win, text="Copy to Clipboard", command=copy_to_clipboard).pack(pady=(0, 8))


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = App()
    app.mainloop()