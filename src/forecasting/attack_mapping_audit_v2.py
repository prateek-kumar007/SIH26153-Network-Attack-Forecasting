"""
SIH26153
Attack Type / Family Mapping Audit V3

Fix:
    The daily attack mapping CSVs do NOT contain Mapping_Source_File.
    The source filename is reconstructed from the mapping filename.

Does not modify any existing dataset/results.
"""

from pathlib import Path
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(r"D:\SIH26153")

FORECAST_FILE = (
    ROOT
    / "data"
    / "processed"
    / "forecasting_samples"
    / "all_forecasting_samples.csv"
)

MAPPING_DIR = (
    ROOT
    / "data"
    / "processed"
    / "attack_type_mapping"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "attack_mapping_audit_v3"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SPLIT
# ============================================================

def assign_split(source_file):

    source = str(source_file)

    train_dates = [
        "14-02-2018",
        "15-02-2018",
        "16-02-2018",
        "20-02-2018",
        "21-02-2018",
        "22-02-2018",
        "23-02-2018",
    ]

    validation_dates = [
        "28-02-2018",
    ]

    test_dates = [
        "01-03-2018",
        "02-03-2018",
    ]

    for date in train_dates:
        if date in source:
            return "TRAIN"

    for date in validation_dates:
        if date in source:
            return "VALIDATION"

    for date in test_dates:
        if date in source:
            return "TEST"

    raise ValueError(
        f"Could not assign split: {source}"
    )


# ============================================================
# ATTACK FAMILY
# ============================================================

def attack_family(label):

    if pd.isna(label):
        return "Unknown"

    label = str(label).strip()

    if label == "" or label.lower() == "benign":
        return "Benign"

    x = label.lower()

    if "brute" in x:
        return "Brute Force"

    if "ddos" in x:
        return "DDoS"

    if x.startswith("dos-") or "dos " in x:
        return "DoS"

    if "bot" in x:
        return "Bot"

    if "infilteration" in x or "infiltration" in x:
        return "Infiltration"

    if "sql" in x or "web" in x or "xss" in x:
        return "Web Attack"

    if "ftp" in x or "ssh" in x:
        return "Brute Force"

    return "Other"


# ============================================================
# TIMESTAMP NORMALIZATION
# ============================================================

def normalize_window(series):

    return pd.to_datetime(
        series,
        errors="coerce"
    ).dt.floor("30s")


# ============================================================
# START
# ============================================================

print("=" * 70)
print("SIH26153 - ATTACK MAPPING AUDIT V3")
print("=" * 70)


# ============================================================
# LOAD FORECASTING DATA
# ============================================================

print("\nLoading forecasting samples...")

forecast = pd.read_csv(
    FORECAST_FILE,
    low_memory=False
)

print(
    f"Forecasting samples : {len(forecast):,}"
)

print(
    f"Columns             : {len(forecast.columns)}"
)


required = [
    "Source_File",
    "Target_Window",
    "Target_Attack",
]

missing = [
    c for c in required
    if c not in forecast.columns
]

if missing:
    raise RuntimeError(
        f"Missing forecasting columns: {missing}"
    )


forecast["Source_File"] = (
    forecast["Source_File"]
    .astype(str)
    .str.strip()
)

forecast["Normalized_Window"] = normalize_window(
    forecast["Target_Window"]
)

forecast["Split"] = forecast["Source_File"].apply(
    assign_split
)


# ============================================================
# LOAD DAILY MAPPING FILES
# ============================================================

print("\nFinding daily attack mapping files...")

mapping_files = sorted(
    MAPPING_DIR.glob("*_attack_type_mapping.csv")
)

print(
    f"Daily mapping files found : {len(mapping_files)}"
)


mapping_frames = []


for path in mapping_files:

    df = pd.read_csv(path)

    required_mapping = [
        "Window",
        "Total_Flows",
        "Attack_Flow_Count",
        "Attack_Ratio",
        "Dominant_Attack_Type",
        "Attack_Types",
        "Number_of_Attack_Types",
    ]

    missing_mapping = [
        c for c in required_mapping
        if c not in df.columns
    ]

    if missing_mapping:

        print(
            f"SKIP: {path.name}"
        )

        print(
            f"Missing: {missing_mapping}"
        )

        continue


    # --------------------------------------------------------
    # IMPORTANT:
    # Reconstruct corresponding source state filename.
    # --------------------------------------------------------

    mapping_source_file = path.name.replace(
        "_attack_type_mapping.csv",
        "_30s_state.csv"
    )


    df = df[
        required_mapping
    ].copy()


    df["Source_File"] = mapping_source_file

    df["Normalized_Window"] = normalize_window(
        df["Window"]
    )


    mapping_frames.append(
        df
    )


    print(
        f"LOAD {path.name:<65}"
        f" rows={len(df):,}"
    )


if not mapping_frames:

    raise RuntimeError(
        "No valid attack mapping files found."
    )


mapping = pd.concat(
    mapping_frames,
    ignore_index=True
)


print(
    f"\nTotal mapping rows : {len(mapping):,}"
)


# ============================================================
# KEY CHECK
# ============================================================

print("\nKEY QUALITY CHECK")

print(
    "Forecast invalid windows :",
    forecast["Normalized_Window"].isna().sum()
)

print(
    "Mapping invalid windows  :",
    mapping["Normalized_Window"].isna().sum()
)


forecast_duplicates = forecast.duplicated(
    ["Source_File", "Normalized_Window"]
).sum()

mapping_duplicates = mapping.duplicated(
    ["Source_File", "Normalized_Window"]
).sum()


print(
    "Forecast duplicate keys :",
    forecast_duplicates
)

print(
    "Mapping duplicate keys  :",
    mapping_duplicates
)


# ============================================================
# CHECK SOURCE FILE OVERLAP
# ============================================================

forecast_sources = set(
    forecast["Source_File"].unique()
)

mapping_sources = set(
    mapping["Source_File"].unique()
)


print("\nSOURCE FILE CHECK")

print(
    f"Forecast source files : {len(forecast_sources)}"
)

print(
    f"Mapping source files  : {len(mapping_sources)}"
)


missing_sources = (
    forecast_sources
    - mapping_sources
)

extra_sources = (
    mapping_sources
    - forecast_sources
)


print(
    "Forecast sources missing from mapping:",
    len(missing_sources)
)

if missing_sources:

    for x in sorted(missing_sources):
        print("  MISSING:", x)


print(
    "Extra mapping sources:",
    len(extra_sources)
)


# ============================================================
# REMOVE DUPLICATE MAPPING KEYS
# ============================================================

if mapping_duplicates > 0:

    print(
        "\nWARNING: duplicate mapping keys detected."
    )

    duplicate_rows = mapping[
        mapping.duplicated(
            ["Source_File", "Normalized_Window"],
            keep=False
        )
    ]

    duplicate_rows.to_csv(
        OUTPUT_DIR
        / "duplicate_mapping_keys.csv",
        index=False
    )

    mapping = mapping.drop_duplicates(
        [
            "Source_File",
            "Normalized_Window"
        ],
        keep="first"
    )


# ============================================================
# MERGE
# ============================================================

print("\nJoining forecasting samples with mapping...")

mapping_merge = mapping[
    [
        "Source_File",
        "Normalized_Window",
        "Dominant_Attack_Type",
        "Attack_Types",
        "Attack_Flow_Count",
        "Attack_Ratio",
        "Number_of_Attack_Types",
    ]
].copy()


merged = forecast.merge(
    mapping_merge,
    on=[
        "Source_File",
        "Normalized_Window",
    ],
    how="left",
    indicator=True
)


# ============================================================
# MATCH RATE
# ============================================================

total = len(merged)

matched = (
    merged["_merge"] == "both"
).sum()

unmatched = (
    merged["_merge"] != "both"
).sum()

match_rate = matched / total


print("\n" + "=" * 70)
print("JOIN RESULT")
print("=" * 70)

print(
    f"Total forecasting samples : {total:,}"
)

print(
    f"Matched                   : {matched:,}"
)

print(
    f"Unmatched                 : {unmatched:,}"
)

print(
    f"Match rate                : {match_rate:.6%}"
)


# ============================================================
# SAVE UNMATCHED
# ============================================================

if unmatched > 0:

    unmatched_df = merged[
        merged["_merge"] != "both"
    ]

    unmatched_df[
        [
            "Source_File",
            "Target_Window",
            "Normalized_Window",
            "Split",
            "Target_Attack",
        ]
    ].to_csv(
        OUTPUT_DIR
        / "unmatched_forecasting_rows.csv",
        index=False
    )


# ============================================================
# ATTACK LABEL
# ============================================================

merged["Mapped_Attack_Type"] = (
    merged["Dominant_Attack_Type"]
    .fillna("UNMAPPED")
    .astype(str)
    .str.strip()
)

merged["Attack_Family"] = (
    merged["Mapped_Attack_Type"]
    .apply(attack_family)
)


# ============================================================
# TARGET CONSISTENCY
# ============================================================

merged["Mapping_Attack"] = (
    merged["Attack_Flow_Count"]
    .fillna(0)
    > 0
).astype(int)


merged["Target_Mapping_Match"] = (
    merged["Target_Attack"]
    == merged["Mapping_Attack"]
)


mismatch_count = (
    ~merged["Target_Mapping_Match"]
).sum()


print("\n" + "=" * 70)
print("TARGET / MAPPING CONSISTENCY")
print("=" * 70)

print(
    f"Mismatched target rows : {mismatch_count:,}"
)


# ============================================================
# ATTACK TYPES
# ============================================================

attack_rows = merged[
    merged["Target_Attack"] == 1
].copy()


print("\n" + "=" * 70)
print("ATTACK TYPES")
print("=" * 70)


attack_type_table = (
    attack_rows
    .groupby(
        ["Split", "Mapped_Attack_Type"],
        dropna=False
    )
    .size()
    .reset_index(
        name="Samples"
    )
    .sort_values(
        ["Split", "Samples"],
        ascending=[True, False]
    )
)


print(
    attack_type_table.to_string(
        index=False
    )
)


attack_type_table.to_csv(
    OUTPUT_DIR
    / "attack_type_by_split.csv",
    index=False
)


# ============================================================
# ATTACK FAMILIES
# ============================================================

print("\n" + "=" * 70)
print("ATTACK FAMILIES")
print("=" * 70)


family_table = (
    attack_rows
    .groupby(
        ["Split", "Attack_Family"],
        dropna=False
    )
    .size()
    .reset_index(
        name="Samples"
    )
    .sort_values(
        ["Split", "Samples"],
        ascending=[True, False]
    )
)


print(
    family_table.to_string(
        index=False
    )
)


family_table.to_csv(
    OUTPUT_DIR
    / "attack_family_by_split.csv",
    index=False
)


# ============================================================
# SEEN / UNSEEN TYPES
# ============================================================

train_types = set(
    attack_rows.loc[
        attack_rows["Split"] == "TRAIN",
        "Mapped_Attack_Type"
    ]
)

validation_types = set(
    attack_rows.loc[
        attack_rows["Split"] == "VALIDATION",
        "Mapped_Attack_Type"
    ]
)

test_types = set(
    attack_rows.loc[
        attack_rows["Split"] == "TEST",
        "Mapped_Attack_Type"
    ]
)


test_unseen_train = (
    test_types
    - train_types
)

test_unseen_development = (
    test_types
    - train_types
    - validation_types
)


print("\n" + "=" * 70)
print("UNSEEN ATTACK TYPE ANALYSIS")
print("=" * 70)

print(
    "TRAIN:"
)

for x in sorted(train_types):
    print(
        f"  {x}"
    )


print(
    "\nVALIDATION:"
)

for x in sorted(validation_types):
    print(
        f"  {x}"
    )


print(
    "\nTEST:"
)

for x in sorted(test_types):
    print(
        f"  {x}"
    )


print(
    "\nTEST types unseen in TRAIN:"
)

for x in sorted(test_unseen_train):
    print(
        f"  {x}"
    )


print(
    "\nTEST types unseen in TRAIN + VALIDATION:"
)

for x in sorted(test_unseen_development):
    print(
        f"  {x}"
    )


# ============================================================
# SEEN / UNSEEN FAMILIES
# ============================================================

train_families = set(
    attack_rows.loc[
        attack_rows["Split"] == "TRAIN",
        "Attack_Family"
    ]
)

validation_families = set(
    attack_rows.loc[
        attack_rows["Split"] == "VALIDATION",
        "Attack_Family"
    ]
)

test_families = set(
    attack_rows.loc[
        attack_rows["Split"] == "TEST",
        "Attack_Family"
    ]
)


test_unseen_families_train = (
    test_families
    - train_families
)

test_unseen_families_development = (
    test_families
    - train_families
    - validation_families
)


print("\n" + "=" * 70)
print("UNSEEN ATTACK FAMILY ANALYSIS")
print("=" * 70)

print(
    "TRAIN:",
    sorted(train_families)
)

print(
    "VALIDATION:",
    sorted(validation_families)
)

print(
    "TEST:",
    sorted(test_families)
)

print(
    "\nTEST families unseen in TRAIN:",
    sorted(test_unseen_families_train)
)

print(
    "\nTEST families unseen in TRAIN + VALIDATION:",
    sorted(test_unseen_families_development)
)


# ============================================================
# ATTACK ONSETS
# ============================================================

print("\n" + "=" * 70)
print("ATTACK ONSET TYPE ANALYSIS")
print("=" * 70)


merged = merged.sort_values(
    [
        "Source_File",
        "Normalized_Window"
    ]
).reset_index(drop=True)


merged["Prev_Target"] = (
    merged
    .groupby("Source_File")["Target_Attack"]
    .shift(1)
)


merged["Prev_Window"] = (
    merged
    .groupby("Source_File")["Normalized_Window"]
    .shift(1)
)


merged["Seconds_From_Previous"] = (
    merged["Normalized_Window"]
    - merged["Prev_Window"]
).dt.total_seconds()


merged["Valid_30s_Transition"] = (
    merged["Seconds_From_Previous"] == 30
)


onsets = merged[
    (merged["Valid_30s_Transition"])
    & (merged["Prev_Target"] == 0)
    & (merged["Target_Attack"] == 1)
].copy()


print(
    f"Valid attack onsets : {len(onsets):,}"
)


onset_types = (
    onsets
    .groupby(
        [
            "Split",
            "Mapped_Attack_Type"
        ],
        dropna=False
    )
    .size()
    .reset_index(
        name="Onsets"
    )
    .sort_values(
        ["Split", "Onsets"],
        ascending=[True, False]
    )
)


print(
    onset_types.to_string(
        index=False
    )
)


onset_types.to_csv(
    OUTPUT_DIR
    / "attack_onset_types.csv",
    index=False
)


# ============================================================
# SAVE FULL MERGED DATA
# ============================================================

merged.drop(
    columns=["_merge"],
    errors="ignore"
).to_csv(
    OUTPUT_DIR
    / "forecasting_samples_with_attack_mapping.csv",
    index=False
)


# ============================================================
# SAVE SUMMARY
# ============================================================

with open(
    OUTPUT_DIR / "audit_report_v3.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "SIH26153 ATTACK MAPPING AUDIT V3\n"
    )

    f.write(
        f"Forecast samples: {total:,}\n"
    )

    f.write(
        f"Mapping rows: {len(mapping):,}\n"
    )

    f.write(
        f"Matched: {matched:,}\n"
    )

    f.write(
        f"Unmatched: {unmatched:,}\n"
    )

    f.write(
        f"Match rate: {match_rate:.6%}\n"
    )

    f.write(
        f"Target/mapping mismatches: "
        f"{mismatch_count:,}\n"
    )

    f.write(
        f"Valid attack onsets: "
        f"{len(onsets):,}\n"
    )

    f.write(
        "\nTEST types unseen in TRAIN + VALIDATION:\n"
    )

    for x in sorted(test_unseen_development):
        f.write(
            f"  - {x}\n"
        )

    f.write(
        "\nTEST families unseen in TRAIN + VALIDATION:\n"
    )

    for x in sorted(test_unseen_families_development):
        f.write(
            f"  - {x}\n"
        )


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("AUDIT V3 COMPLETE")
print("=" * 70)

print(
    f"Results saved to:\n{OUTPUT_DIR}"
)

print(
    "\nDo NOT start LSTM yet."
)

print(
    "Send the complete terminal output."
)