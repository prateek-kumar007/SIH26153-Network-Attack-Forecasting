from pathlib import Path
import pandas as pd

# ============================================================
# CONFIG
# ============================================================

INPUT_DIR = Path("data/processed/forecasting_samples")
OUTPUT_DIR = Path("results/forecasting_split")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_FILE = INPUT_DIR / "forecasting_sample_summary.csv"
OUTPUT_FILE = OUTPUT_DIR / "forecasting_split_audit.csv"

# ============================================================
# LOAD SUMMARY
# ============================================================

if not SUMMARY_FILE.exists():
    raise FileNotFoundError(
        f"Missing file:\n{SUMMARY_FILE}\n\n"
        "Run build_forecasting_samples.py first."
    )

summary = pd.read_csv(SUMMARY_FILE)

print("=" * 70)
print("FORECASTING DATASET SPLIT AUDIT")
print("=" * 70)

print("\nDetected columns:")
print(list(summary.columns))

# ============================================================
# VALIDATE REQUIRED COLUMNS
# ============================================================

required_columns = {
    "Source_File",
    "Samples",
    "Attack_Targets",
    "Benign_Targets",
    "Segments_Used",
}

missing = required_columns - set(summary.columns)

if missing:
    raise ValueError(
        f"Missing required columns: {sorted(missing)}"
    )

# ============================================================
# DISPLAY CURRENT DISTRIBUTION
# ============================================================

print("\nCurrent sample distribution:\n")

print(
    summary[
        [
            "Source_File",
            "Samples",
            "Attack_Targets",
            "Benign_Targets",
            "Segments_Used",
        ]
    ].to_string(index=False)
)

# ============================================================
# CANDIDATE CHRONOLOGICAL SPLIT
# ============================================================
#
# IMPORTANT:
# This script ONLY audits the split.
# It does NOT create model datasets yet.
#
# TRAIN:
# Earlier days
#
# VALIDATION:
# Later day used for model selection
#
# TEST:
# Final unseen days
#
# ============================================================

train_days = {
    "Wednesday-14-02-2018_TrafficForML_CICFlowMeter_30s_state.csv",
    "Thursday-15-02-2018_TrafficForML_CICFlowMeter_30s_state.csv",
    "Friday-16-02-2018_TrafficForML_CICFlowMeter_30s_state.csv",
    "Thuesday-20-02-2018_TrafficForML_CICFlowMeter_30s_state.csv",
    "Wednesday-21-02-2018_TrafficForML_CICFlowMeter_30s_state.csv",
    "Thursday-22-02-2018_TrafficForML_CICFlowMeter_30s_state.csv",
    "Friday-23-02-2018_TrafficForML_CICFlowMeter_30s_state.csv",
}

validation_days = {
    "Wednesday-28-02-2018_TrafficForML_CICFlowMeter_30s_state.csv",
}

test_days = {
    "Thursday-01-03-2018_TrafficForML_CICFlowMeter_30s_state.csv",
    "Friday-02-03-2018_TrafficForML_CICFlowMeter_30s_state.csv",
}

# ============================================================
# ASSIGN SPLIT
# ============================================================

def assign_split(filename):
    if filename in train_days:
        return "TRAIN"

    if filename in validation_days:
        return "VALIDATION"

    if filename in test_days:
        return "TEST"

    return "UNASSIGNED"


summary["Split"] = (
    summary["Source_File"]
    .astype(str)
    .apply(assign_split)
)

# ============================================================
# SPLIT COVERAGE
# ============================================================

all_files = set(summary["Source_File"].astype(str))

assigned_files = (
    train_days |
    validation_days |
    test_days
)

unassigned = all_files - assigned_files
unknown = assigned_files - all_files

print("\n" + "=" * 70)
print("SPLIT COVERAGE")
print("=" * 70)

print(f"Dataset files : {len(all_files)}")
print(f"Assigned files: {len(assigned_files)}")

if unassigned:
    print("\nWARNING: Unassigned files:")
    for filename in sorted(unassigned):
        print("  ", filename)

else:
    print("\nPASS: Every dataset file is assigned.")

if unknown:
    print("\nWARNING: Split definition contains unknown files:")
    for filename in sorted(unknown):
        print("  ", filename)

else:
    print("PASS: No unknown files in split definition.")

# ============================================================
# SPLIT SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SPLIT SUMMARY")
print("=" * 70)

split_summary = (
    summary
    .groupby("Split")
    .agg(
        Files=("Source_File", "count"),
        Samples=("Samples", "sum"),
        Attack_Targets=("Attack_Targets", "sum"),
        Benign_Targets=("Benign_Targets", "sum"),
    )
    .reset_index()
)

split_summary["Attack_Rate_%"] = (
    split_summary["Attack_Targets"]
    / split_summary["Samples"]
    * 100
)

print(
    split_summary.to_string(index=False)
)

# ============================================================
# PER-DAY DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("PER-DAY ATTACK DISTRIBUTION")
print("=" * 70)

day_view = summary[
    [
        "Source_File",
        "Split",
        "Samples",
        "Attack_Targets",
        "Benign_Targets",
        "Segments_Used",
    ]
].copy()

day_view["Attack_Rate_%"] = (
    day_view["Attack_Targets"]
    / day_view["Samples"]
    * 100
)

print(
    day_view.to_string(index=False)
)

# ============================================================
# CLASS PRESENCE CHECK
# ============================================================

print("\n" + "=" * 70)
print("CLASS PRESENCE CHECK")
print("=" * 70)

for split in ["TRAIN", "VALIDATION", "TEST"]:

    rows = summary[
        summary["Split"] == split
    ]

    samples = rows["Samples"].sum()
    attacks = rows["Attack_Targets"].sum()
    benign = rows["Benign_Targets"].sum()

    print(f"\n{split}")
    print(f"  Samples : {samples:,}")
    print(f"  Attack  : {attacks:,}")
    print(f"  Benign  : {benign:,}")

    if samples > 0:
        print(
            f"  Attack rate: "
            f"{attacks / samples * 100:.2f}%"
        )

    if attacks == 0:
        print("  WARNING: No attack targets.")

    if benign == 0:
        print("  WARNING: No benign targets.")

# ============================================================
# SPECIAL WARNING
# ============================================================

print("\n" + "=" * 70)
print("METHODOLOGY CHECK")
print("=" * 70)

print(
    """
This is a chronological split audit.

We are NOT randomly splitting the 9,283 forecasting
samples because adjacent forecasting samples overlap
in their historical windows.

The TEST set contains later dates:
    2018-03-01
    2018-03-02

These days contain attack scenarios that may not occur
in the training set.

Therefore, the final experiment must distinguish:

1. Chronological generalization
2. Unseen-attack generalization

We will NOT claim the second until attack-type labels
are mapped to the target windows.
"""
)

# ============================================================
# SAVE
# ============================================================

summary.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)

print(f"\nSaved:")
print(OUTPUT_FILE)