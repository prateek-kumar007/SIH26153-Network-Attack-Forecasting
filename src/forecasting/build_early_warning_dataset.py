from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data" / "processed" / "forecasting_samples" / "all_forecasting_samples.csv"
OUT = ROOT / "data" / "processed" / "early_warning"
OUT.mkdir(parents=True, exist_ok=True)

WINDOW_SEC = 30
HORIZONS = [30, 60, 90, 120, 150]

TRAIN = {
    "Wednesday-14-02-2018", "Thursday-15-02-2018", "Friday-16-02-2018",
    "Thuesday-20-02-2018", "Wednesday-21-02-2018", "Thursday-22-02-2018",
    "Friday-23-02-2018",
}
VALID = {"Wednesday-28-02-2018"}
TEST = {"Thursday-01-03-2018", "Friday-02-03-2018"}

SUFFIX = "_TrafficForML_CICFlowMeter_30s_state.csv"


def parse_timestamp(series):
    s = series.astype(str).str.strip()
    iso = s.str.match(r"^\d{4}-\d{2}-\d{2} ")
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    out.loc[iso] = pd.to_datetime(s.loc[iso], errors="coerce", format="%Y-%m-%d %H:%M:%S")
    out.loc[~iso] = pd.to_datetime(s.loc[~iso], errors="coerce", dayfirst=True)
    return out


def normalize_source(value):
    value = str(value).strip()
    if value.endswith(SUFFIX):
        return value[:-len(SUFFIX)]
    return value


def get_split(source):
    if source in TRAIN:
        return "TRAIN"
    if source in VALID:
        return "VALIDATION"
    if source in TEST:
        return "TEST"
    return "UNKNOWN"


print("=" * 80)
print("SIH26153 - EARLY-WARNING DATASET BUILDER")
print("=" * 80)
print(f"\nInput file:\n{INPUT}\n")

if not INPUT.exists():
    raise FileNotFoundError(f"Input file not found:\n{INPUT}")

df = pd.read_csv(INPUT)
print(f"Input rows:    {len(df):,}")
print(f"Input columns: {len(df.columns):,}")

# Only these columns are required. Input_End_Window is derived below.
required = {"Source_File", "Target_Window", "Target_Attack"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

# Normalize source names BEFORE split assignment.
df["Source_File"] = df["Source_File"].map(normalize_source)

# Parse target windows.
df["Target_Window"] = parse_timestamp(df["Target_Window"])
if df["Target_Window"].isna().any():
    raise ValueError(f"Target_Window contains {int(df['Target_Window'].isna().sum()):,} invalid timestamps.")

# Validate target.
df["Target_Attack"] = pd.to_numeric(df["Target_Attack"], errors="raise").astype(np.int8)
if not set(df["Target_Attack"].unique()).issubset({0, 1}):
    raise ValueError("Target_Attack must contain only 0 and 1.")

# Critical fix: derive Input_End_Window from Target_Window.
df["Input_End_Window"] = df["Target_Window"] - pd.Timedelta(seconds=WINDOW_SEC)

# Assign chronological split after normalization.
df["Split"] = df["Source_File"].map(get_split)
unknown = sorted(df.loc[df["Split"] == "UNKNOWN", "Source_File"].unique())
if unknown:
    raise ValueError("Unknown Source_File values found:\n" + "\n".join(f"  - {x}" for x in unknown))

# Sort and validate unique windows.
df = df.sort_values(["Source_File", "Target_Window"]).reset_index(drop=True)
if df.duplicated(["Source_File", "Target_Window"]).any():
    raise ValueError("Duplicate (Source_File, Target_Window) rows found.")

# Attack status at each 30-second window.
status = df.set_index(["Source_File", "Target_Window"])["Target_Attack"].to_dict()

# Find exact 0 -> 1 attack onsets, never across gaps.
onsets = set()
for source, g in df.groupby("Source_File", sort=False):
    g = g.sort_values("Target_Window").reset_index(drop=True)
    times = g["Target_Window"].tolist()
    y = g["Target_Attack"].to_numpy()
    for i in range(1, len(g)):
        if (times[i] - times[i - 1]).total_seconds() != WINDOW_SEC:
            continue
        if int(y[i - 1]) == 0 and int(y[i]) == 1:
            onsets.add((source, pd.Timestamp(times[i])))

# Build only benign-current samples with a complete +30..+150s future path.
rows = []
max_steps = max(HORIZONS) // WINDOW_SEC

for source, g in df.groupby("Source_File", sort=False):
    g = g.sort_values("Target_Window").reset_index(drop=True)
    times = [pd.Timestamp(x) for x in g["Target_Window"]]
    index = {t: i for i, t in enumerate(times)}
    split = g.loc[0, "Split"]

    for current in times:
        current_key = (source, current)
        if current_key not in status or int(status[current_key]) != 0:
            continue
        i0 = index[current]

        future_times = [current + pd.Timedelta(seconds=WINDOW_SEC * k) for k in range(1, max_steps + 1)]
        if any(i0 + k >= len(times) or times[i0 + k] != t for k, t in enumerate(future_times, start=1)):
            continue

        onset_vec = np.array([int((source, t) in onsets) for t in future_times], dtype=np.int8)

        row = {
            "Source_File": source,
            "Split": split,
            "Input_End_Window": current,
            "Forecast_Target_Window": current + pd.Timedelta(seconds=WINDOW_SEC),
            "Current_Attack": 0,
        }
        for j, h in enumerate(HORIZONS):
            row[f"Onset_At_{h}s"] = int(onset_vec[j])
            row[f"Starts_Within_{h}s"] = int(onset_vec[:j + 1].any())
        rows.append(row)

early = pd.DataFrame(rows)
if early.empty:
    raise RuntimeError("No early-warning samples were created.")
early = early.sort_values(["Split", "Source_File", "Input_End_Window"]).reset_index(drop=True)

# Validation.
if not (early["Current_Attack"] == 0).all():
    raise AssertionError("Non-benign current state found.")

for a, b in zip(HORIZONS[:-1], HORIZONS[1:]):
    if np.any(early[f"Starts_Within_{a}s"].to_numpy() > early[f"Starts_Within_{b}s"].to_numpy()):
        raise AssertionError("Starts_Within targets are not monotonic.")

summary_rows = []
for split in ["TRAIN", "VALIDATION", "TEST"]:
    s = early[early["Split"] == split]
    row = {"Split": split, "Samples": len(s)}
    for h in HORIZONS:
        for prefix in ["Onset_At", "Starts_Within"]:
            col = f"{prefix}_{h}s"
            n = int(s[col].sum())
            row[f"{col}_Positive"] = n
            row[f"{col}_Rate"] = n / len(s) if len(s) else np.nan
    summary_rows.append(row)
summary = pd.DataFrame(summary_rows)

samples_path = OUT / "early_warning_samples.csv"
summary_path = OUT / "early_warning_summary.csv"
definition_path = OUT / "early_warning_definition.txt"

early.to_csv(samples_path, index=False)
summary.to_csv(summary_path, index=False)

definition_path.write_text(
    """SIH26153 - Early Warning Target Definition\n\n"
    "Current state must be benign.\n"
    "Input_End_Window is derived as Target_Window - 30 seconds.\n"
    "Exact attack onset: Attack(t-30s)=0 and Attack(t)=1.\n"
    "Onset_At_Hs = exact onset at +Hs.\n"
    "Starts_Within_Hs = exact onset at any point from +30s through +Hs.\n"
    "Samples crossing capture gaps are excluded.\n"
    "No future features are used as inputs.\n"
    "Chronological split: TRAIN Feb14/15/16/20/21/22/23; VALIDATION Feb28; TEST Mar1/Mar2.\n"
    "This script only builds and validates targets; it does not train a model.\n""".replace('"""', ''),
    encoding="utf-8",
)

print("\n" + "=" * 80)
print("RESULT")
print("=" * 80)
print(f"Input rows:            {len(df):,}")
print(f"Exact attack onsets:   {len(onsets):,}")
print(f"Early-warning samples: {len(early):,}\n")

for _, r in summary.iterrows():
    print(r["Split"])
    print(f"  Samples: {int(r['Samples']):,}")
    for h in HORIZONS:
        ep = int(r[f"Onset_At_{h}s_Positive"])
        er = r[f"Onset_At_{h}s_Rate"]
        wp = int(r[f"Starts_Within_{h}s_Positive"])
        wr = r[f"Starts_Within_{h}s_Rate"]
        print(f"  +{h:>3}s | exact onset: {ep:>4} ({er:.3%}) | starts within: {wp:>4} ({wr:.3%})")

print("\nOutputs:")
print(f"  {samples_path}")
print(f"  {summary_path}")
print(f"  {definition_path}")
print("\nValidation: PASSED")
print("=" * 80)
