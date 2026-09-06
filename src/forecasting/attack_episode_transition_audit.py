"""
SIH26153 - Attack Episode & Transition Audit

Purpose
-------
Audit the temporal structure of the current forecasting target before
building a temporal model.

Questions answered:
1. How many attack episodes exist?
2. How long are attack episodes?
3. How many 0->1 attack onsets occur?
4. Which attack type is associated with each onset?
5. How many 0->0, 0->1, 1->0, 1->1 transitions occur?
6. Which attack types/families occur in TRAIN / VALIDATION / TEST?
7. Which attack types are unseen during model development?
8. How persistent are attacks?

IMPORTANT
---------
This script does NOT modify the dataset.
It only reads existing processed files and writes audit results.

Run from:
D:\\SIH26153

Command:
python src\\forecasting\\attack_episode_transition_audit.py
"""

from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

FORECAST_FILE = (
    ROOT
    / "data"
    / "processed"
    / "forecasting_samples"
    / "all_forecasting_samples.csv"
)

ATTACK_MAPPING_DIR = (
    ROOT
    / "data"
    / "processed"
    / "attack_type_mapping"
)

OUTPUT_DIR = ROOT / "results" / "attack_episode_audit"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SPLIT ASSIGNMENT
# ============================================================

def assign_split(source_file: str) -> str:
    """
    Assign chronological split using the actual CIC-IDS2018
    source filename dates.
    """

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
        f"Could not assign split for source file:\n{source}"
    )


# ============================================================
# ATTACK FAMILY MAPPING
# ============================================================

def map_attack_family(label: str) -> str:
    """
    Map CIC-IDS2018 attack labels to broad attack families.

    This is an audit grouping, NOT a MITRE ATT&CK mapping.
    """

    if pd.isna(label):
        return "Unknown"

    label = str(label).strip()

    if label == "" or label.lower() == "benign":
        return "Benign"

    lower = label.lower()

    if "ftp-brute" in lower:
        return "Brute Force"

    if "ssh-bruteforce" in lower:
        return "Brute Force"

    if "brute force" in lower:
        return "Brute Force"

    if "sql injection" in lower:
        return "Web Attack"

    if "xss" in lower:
        return "Web Attack"

    if "web" in lower:
        return "Web Attack"

    if "goldeneye" in lower:
        return "DoS"

    if "slowloris" in lower:
        return "DoS"

    if "slowhttptest" in lower:
        return "DoS"

    if "hulk" in lower:
        return "DoS"

    if "ddos" in lower:
        return "DDoS"

    if "loic" in lower:
        return "DDoS"

    if "hoic" in lower:
        return "DDoS"

    if "bot" in lower:
        return "Bot"

    if "infilteration" in lower or "infiltration" in lower:
        return "Infiltration"

    return "Other Attack"


# ============================================================
# LOAD FORECASTING DATA
# ============================================================

print("=" * 80)
print("SIH26153 - ATTACK EPISODE & TRANSITION AUDIT")
print("=" * 80)

print("\nLoading forecasting samples...")

if not FORECAST_FILE.exists():
    raise FileNotFoundError(
        f"\nForecasting file not found:\n{FORECAST_FILE}"
    )

df = pd.read_csv(FORECAST_FILE)

print(f"Loaded samples : {len(df):,}")
print(f"Columns        : {len(df.columns):,}")


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required = [
    "Source_File",
    "Target_Window",
    "Target_Attack",
]

missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )


# ============================================================
# CLEAN / SORT
# ============================================================

df["Target_Window"] = pd.to_datetime(
    df["Target_Window"],
    errors="coerce"
)

df["Target_Attack"] = pd.to_numeric(
    df["Target_Attack"],
    errors="coerce"
)

df = df.dropna(
    subset=[
        "Source_File",
        "Target_Window",
        "Target_Attack",
    ]
).copy()

df["Target_Attack"] = df["Target_Attack"].astype(int)

df["Split"] = df["Source_File"].apply(assign_split)

df = df.sort_values(
    ["Source_File", "Target_Window"]
).reset_index(drop=True)


# ============================================================
# BASIC TARGET SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("BASIC TARGET SUMMARY")
print("=" * 80)

print(
    f"TRAIN      : {(df['Split'] == 'TRAIN').sum():,}"
)

print(
    f"VALIDATION : {(df['Split'] == 'VALIDATION').sum():,}"
)

print(
    f"TEST       : {(df['Split'] == 'TEST').sum():,}"
)

print(
    f"Attack     : {(df['Target_Attack'] == 1).sum():,}"
)

print(
    f"Benign     : {(df['Target_Attack'] == 0).sum():,}"
)

print(
    f"Attack rate: {df['Target_Attack'].mean():.4%}"
)


# ============================================================
# TRANSITIONS
# ============================================================

print("\n" + "=" * 80)
print("TEMPORAL TRANSITIONS")
print("=" * 80)

df["Previous_Attack"] = (
    df.groupby("Source_File")["Target_Attack"]
    .shift(1)
)

df["Previous_Window"] = (
    df.groupby("Source_File")["Target_Window"]
    .shift(1)
)

df["Gap_Seconds"] = (
    df["Target_Window"] - df["Previous_Window"]
).dt.total_seconds()

# Only consecutive 30-second transitions are valid.
df["Valid_Transition"] = (
    df["Previous_Attack"].notna()
    & (df["Gap_Seconds"] == 30)
)

valid = df[df["Valid_Transition"]].copy()

valid["Previous_Attack"] = (
    valid["Previous_Attack"].astype(int)
)

valid["Transition"] = (
    valid["Previous_Attack"].astype(str)
    + "->"
    + valid["Target_Attack"].astype(str)
)

transition_counts = (
    valid["Transition"]
    .value_counts()
    .reindex(
        ["0->0", "0->1", "1->0", "1->1"],
        fill_value=0,
    )
)

print("\nTransition counts:")

for transition, count in transition_counts.items():
    print(f"{transition:>5} : {count:,}")


total_transitions = len(valid)

if total_transitions:
    print("\nTransition percentages:")

    for transition, count in transition_counts.items():
        print(
            f"{transition:>5} : "
            f"{count / total_transitions:.4%}"
        )


# ============================================================
# ATTACK PERSISTENCE
# ============================================================

attack_to_attack = transition_counts["1->1"]
attack_end = transition_counts["1->0"]

if attack_to_attack + attack_end > 0:
    persistence_rate = (
        attack_to_attack
        / (attack_to_attack + attack_end)
    )
else:
    persistence_rate = np.nan

print("\nAttack persistence:")
print(
    f"1->1 / (1->1 + 1->0) = "
    f"{persistence_rate:.4%}"
)


# ============================================================
# ATTACK EPISODES
# ============================================================

print("\n" + "=" * 80)
print("ATTACK EPISODES")
print("=" * 80)

episode_records = []

for source_file, group in df.groupby("Source_File"):

    group = group.sort_values("Target_Window").copy()

    group["Previous_Attack"] = (
        group["Target_Attack"].shift(1)
    )

    group["Previous_Window"] = (
        group["Target_Window"].shift(1)
    )

    group["Gap_Seconds"] = (
        group["Target_Window"]
        - group["Previous_Window"]
    ).dt.total_seconds()

    # New episode when:
    # - attack starts (0 -> 1)
    # - OR there is a capture gap
    # - OR first row
    new_episode = (
        (group["Target_Attack"] == 1)
        & (
            group["Previous_Attack"].isna()
            | (group["Previous_Attack"] == 0)
            | (group["Gap_Seconds"] != 30)
        )
    )

    group["Attack_Episode_Start"] = new_episode

    episode_id = 0
    current_attack = False

    for idx, row in group.iterrows():

        if row["Target_Attack"] == 1:

            if (
                not current_attack
                or bool(row["Attack_Episode_Start"])
            ):
                episode_id += 1

                current_attack = True

                episode_records.append(
                    {
                        "Source_File": source_file,
                        "Split": row["Split"],
                        "Episode_ID": episode_id,
                        "Start": row["Target_Window"],
                        "End": row["Target_Window"],
                        "Duration_Windows": 1,
                        "Duration_Seconds": 30,
                    }
                )

            else:
                # Extend current episode
                episode_records[-1]["End"] = (
                    row["Target_Window"]
                )

                episode_records[-1]["Duration_Windows"] += 1

                episode_records[-1]["Duration_Seconds"] += 30

        else:
            current_attack = False


episodes = pd.DataFrame(episode_records)

if len(episodes) == 0:

    print("\nNo attack episodes found.")

else:

    print(
        f"\nTotal attack episodes : "
        f"{len(episodes):,}"
    )

    print(
        f"Mean duration         : "
        f"{episodes['Duration_Seconds'].mean():.2f} sec"
    )

    print(
        f"Median duration       : "
        f"{episodes['Duration_Seconds'].median():.2f} sec"
    )

    print(
        f"Min duration          : "
        f"{episodes['Duration_Seconds'].min():.2f} sec"
    )

    print(
        f"Max duration          : "
        f"{episodes['Duration_Seconds'].max():.2f} sec"
    )


# ============================================================
# LOAD ATTACK TYPE MAPPING
# ============================================================

print("\n" + "=" * 80)
print("LOADING ATTACK TYPE MAPPING")
print("=" * 80)

mapping_files = sorted(
    ATTACK_MAPPING_DIR.glob("*.csv")
)

print(
    f"Mapping files found : {len(mapping_files)}"
)

if not mapping_files:

    raise FileNotFoundError(
        f"No attack mapping CSV files found in:\n"
        f"{ATTACK_MAPPING_DIR}"
    )


mapping_frames = []

for file in mapping_files:

    temp = pd.read_csv(file)

    temp["Mapping_Source_File"] = file.name

    mapping_frames.append(temp)


mapping = pd.concat(
    mapping_frames,
    ignore_index=True
)

print(
    f"Mapping rows        : {len(mapping):,}"
)

print(
    "Mapping columns     :",
    list(mapping.columns)
)


# ============================================================
# NORMALIZE MAPPING COLUMN NAMES
# ============================================================

def find_column(columns, candidates):

    lower_map = {
        str(c).lower(): c
        for c in columns
    }

    for candidate in candidates:

        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    return None


mapping_window_col = find_column(
    mapping.columns,
    [
        "Window",
        "Timestamp",
        "Target_Window",
    ],
)

mapping_label_col = find_column(
    mapping.columns,
    [
        "Attack_Type",
        "AttackType",
        "Label",
        "Attack_Label",
        "Dominant_Attack_Type",
    ],
)

mapping_source_col = find_column(
    mapping.columns,
    [
        "Source_File",
        "source_file",
        "File",
    ],
)


if mapping_window_col is None:
    raise ValueError(
        "Could not identify the window/timestamp column "
        "in attack mapping files."
    )

if mapping_label_col is None:
    raise ValueError(
        "Could not identify the attack label column "
        "in attack mapping files."
    )


# ============================================================
# PREPARE MAPPING
# ============================================================

mapping["Mapping_Window"] = pd.to_datetime(
    mapping[mapping_window_col],
    errors="coerce"
)

mapping["Attack_Label"] = (
    mapping[mapping_label_col]
    .astype(str)
    .str.strip()
)

mapping["Attack_Family"] = (
    mapping["Attack_Label"]
    .apply(map_attack_family)
)

if mapping_source_col is not None:

    mapping["Mapping_Source"] = (
        mapping[mapping_source_col]
        .astype(str)
    )

else:

    # Derive source from filename if not explicitly stored.
    mapping["Mapping_Source"] = (
        mapping["Mapping_Source_File"]
    )


# ============================================================
# MATCH FORECASTING WINDOWS TO ATTACK LABELS
# ============================================================

# Source filenames can differ slightly between mapping and
# forecasting files, so match using the date embedded in name.

def extract_date_key(value):

    value = str(value)

    known_dates = [
        "14-02-2018",
        "15-02-2018",
        "16-02-2018",
        "20-02-2018",
        "21-02-2018",
        "22-02-2018",
        "23-02-2018",
        "28-02-2018",
        "01-03-2018",
        "02-03-2018",
    ]

    for date in known_dates:

        if date in value:
            return date

    return None


df["Date_Key"] = (
    df["Source_File"]
    .apply(extract_date_key)
)

mapping["Date_Key"] = (
    mapping["Mapping_Source"]
    .apply(extract_date_key)
)

mapping_lookup = mapping[
    [
        "Date_Key",
        "Mapping_Window",
        "Attack_Label",
        "Attack_Family",
    ]
].copy()

mapping_lookup = mapping_lookup.dropna(
    subset=["Date_Key", "Mapping_Window"]
)

mapping_lookup = mapping_lookup.drop_duplicates(
    subset=["Date_Key", "Mapping_Window"],
    keep="first"
)

# Only merge attack labels onto attack target windows.
df_attack = df[
    df["Target_Attack"] == 1
].copy()

df_attack = df_attack.merge(
    mapping_lookup,
    left_on=["Date_Key", "Target_Window"],
    right_on=["Date_Key", "Mapping_Window"],
    how="left",
)


# ============================================================
# ATTACK ONSET → ATTACK TYPE
# ============================================================

valid_attack = valid.merge(
    df_attack[
        [
            "Source_File",
            "Target_Window",
            "Attack_Label",
            "Attack_Family",
        ]
    ],
    on=["Source_File", "Target_Window"],
    how="left",
)

onsets = valid_attack[
    (valid_attack["Previous_Attack"] == 0)
    & (valid_attack["Target_Attack"] == 1)
].copy()


print("\n" + "=" * 80)
print("ATTACK ONSETS")
print("=" * 80)

print(
    f"Valid attack onsets : {len(onsets):,}"
)

if len(onsets):

    print("\nAttack types at onset:")

    onset_type_counts = (
        onsets["Attack_Label"]
        .fillna("UNMAPPED")
        .value_counts()
    )

    for label, count in onset_type_counts.items():

        print(
            f"{str(label):30s} : {count:,}"
        )

    print("\nAttack families at onset:")

    onset_family_counts = (
        onsets["Attack_Family"]
        .fillna("Unknown")
        .value_counts()
    )

    for family, count in onset_family_counts.items():

        print(
            f"{str(family):30s} : {count:,}"
        )


# ============================================================
# SPLIT × ATTACK TYPE AUDIT
# ============================================================

print("\n" + "=" * 80)
print("ATTACK TYPE / SPLIT AUDIT")
print("=" * 80)

attack_rows = df_attack[
    [
        "Source_File",
        "Split",
        "Target_Window",
        "Attack_Label",
        "Attack_Family",
    ]
].copy()

attack_rows["Attack_Label"] = (
    attack_rows["Attack_Label"]
    .fillna("UNMAPPED")
)

attack_rows["Attack_Family"] = (
    attack_rows["Attack_Family"]
    .fillna("Unknown")
)

split_type_table = (
    attack_rows
    .groupby(
        ["Attack_Label", "Attack_Family", "Split"]
    )
    .size()
    .reset_index(name="Windows")
)

print(
    split_type_table.to_string(index=False)
)


# ============================================================
# SEEN / UNSEEN ATTACK TYPE ANALYSIS
# ============================================================

train_types = set(
    attack_rows.loc[
        attack_rows["Split"] == "TRAIN",
        "Attack_Label"
    ]
)

validation_types = set(
    attack_rows.loc[
        attack_rows["Split"] == "VALIDATION",
        "Attack_Label"
    ]
)

test_types = set(
    attack_rows.loc[
        attack_rows["Split"] == "TEST",
        "Attack_Label"
    ]
)

all_types = sorted(
    set(attack_rows["Attack_Label"])
)

seen_in_train = []
unseen_in_train = []

for attack_type in all_types:

    if attack_type in train_types:
        seen_in_train.append(attack_type)

    else:
        unseen_in_train.append(attack_type)


print("\n" + "=" * 80)
print("UNSEEN ATTACK TYPE ANALYSIS")
print("=" * 80)

print("\nTRAIN attack types:")

for x in sorted(train_types):
    print(f"  {x}")

print("\nValidation attack types:")

for x in sorted(validation_types):
    print(f"  {x}")

print("\nTest attack types:")

for x in sorted(test_types):
    print(f"  {x}")

print("\nUnseen in TRAIN:")

for x in sorted(unseen_in_train):
    print(f"  {x}")


# ============================================================
# FAMILY-LEVEL SEEN / UNSEEN
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

all_families = sorted(
    set(attack_rows["Attack_Family"])
)

print("\n" + "=" * 80)
print("UNSEEN ATTACK FAMILY ANALYSIS")
print("=" * 80)

print("\nTRAIN families:")

for x in sorted(train_families):
    print(f"  {x}")

print("\nValidation families:")

for x in sorted(validation_families):
    print(f"  {x}")

print("\nTest families:")

for x in sorted(test_families):
    print(f"  {x}")

print("\nFamilies unseen in TRAIN:")

for x in all_families:

    if x not in train_families:
        print(f"  {x}")


# ============================================================
# PER-DAY AUDIT
# ============================================================

print("\n" + "=" * 80)
print("PER-DAY TRANSITION AUDIT")
print("=" * 80)

valid["Date_Key"] = (
    valid["Source_File"]
    .apply(extract_date_key)
)

daily_transition = (
    valid
    .groupby(
        ["Date_Key", "Split", "Transition"]
    )
    .size()
    .unstack(
        fill_value=0
    )
    .reset_index()
)

for col in ["0->0", "0->1", "1->0", "1->1"]:

    if col not in daily_transition.columns:
        daily_transition[col] = 0

daily_transition["Attack_Onsets"] = (
    daily_transition["0->1"]
)

daily_transition["Attack_Ends"] = (
    daily_transition["1->0"]
)

daily_transition["Attack_Persistence"] = np.where(
    (
        daily_transition["1->1"]
        + daily_transition["1->0"]
    ) > 0,
    daily_transition["1->1"]
    / (
        daily_transition["1->1"]
        + daily_transition["1->0"]
    ),
    np.nan,
)

daily_transition = daily_transition[
    [
        "Date_Key",
        "Split",
        "0->0",
        "0->1",
        "1->0",
        "1->1",
        "Attack_Onsets",
        "Attack_Ends",
        "Attack_Persistence",
    ]
].sort_values("Date_Key")

print(
    daily_transition.to_string(index=False)
)


# ============================================================
# EPISODE SPLIT SUMMARY
# ============================================================

if len(episodes):

    episode_split = (
        episodes
        .groupby("Split")
        .agg(
            Episodes=("Episode_ID", "count"),
            Mean_Duration_Seconds=(
                "Duration_Seconds",
                "mean",
            ),
            Median_Duration_Seconds=(
                "Duration_Seconds",
                "median",
            ),
            Max_Duration_Seconds=(
                "Duration_Seconds",
                "max",
            ),
        )
        .reset_index()
    )

else:

    episode_split = pd.DataFrame()


# ============================================================
# SAVE RESULTS
# ============================================================

print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

if len(episodes):

    episodes.to_csv(
        OUTPUT_DIR / "attack_episode_summary.csv",
        index=False,
    )

    episode_split.to_csv(
        OUTPUT_DIR / "attack_episode_by_split.csv",
        index=False,
    )

split_type_table.to_csv(
    OUTPUT_DIR / "transition_by_attack_type.csv",
    index=False,
)

daily_transition.to_csv(
    OUTPUT_DIR / "transition_by_day.csv",
    index=False,
)

# ============================================================
# FAMILY/SPLIT AUDIT TABLE
# ============================================================

family_split = (
    attack_rows
    .groupby(
        [
            "Attack_Family",
            "Split",
        ]
    )
    .agg(
        Attack_Windows=(
            "Target_Window",
            "count",
        ),
        Attack_Types=(
            "Attack_Label",
            "nunique",
        ),
    )
    .reset_index()
)

family_split.to_csv(
    OUTPUT_DIR / "split_attack_family_audit.csv",
    index=False,
)


# ============================================================
# REPORT
# ============================================================

report_path = OUTPUT_DIR / "audit_report.txt"

with open(
    report_path,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "SIH26153 - ATTACK EPISODE & TRANSITION AUDIT\n"
    )

    f.write("=" * 80 + "\n\n")

    f.write(
        f"Forecasting samples: {len(df):,}\n"
    )

    f.write(
        f"Valid 30-second transitions: "
        f"{len(valid):,}\n\n"
    )

    f.write("TRANSITIONS\n")
    f.write("-" * 80 + "\n")

    for transition, count in transition_counts.items():

        percentage = (
            count / len(valid)
            if len(valid)
            else 0
        )

        f.write(
            f"{transition}: "
            f"{count:,} "
            f"({percentage:.4%})\n"
        )

    f.write(
        f"\nAttack persistence: "
        f"{persistence_rate:.4%}\n"
    )

    f.write("\nATTACK EPISODES\n")
    f.write("-" * 80 + "\n")

    if len(episodes):

        f.write(
            f"Total episodes: "
            f"{len(episodes):,}\n"
        )

        f.write(
            f"Mean duration: "
            f"{episodes['Duration_Seconds'].mean():.2f} sec\n"
        )

        f.write(
            f"Median duration: "
            f"{episodes['Duration_Seconds'].median():.2f} sec\n"
        )

        f.write(
            f"Max duration: "
            f"{episodes['Duration_Seconds'].max():.2f} sec\n"
        )

    f.write("\nATTACK TYPES UNSEEN IN TRAIN\n")
    f.write("-" * 80 + "\n")

    for attack_type in sorted(unseen_in_train):

        f.write(
            f"{attack_type}\n"
        )

    f.write("\nATTACK FAMILIES UNSEEN IN TRAIN\n")
    f.write("-" * 80 + "\n")

    for family in sorted(
        set(all_families) - train_families
    ):

        f.write(
            f"{family}\n"
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("AUDIT COMPLETE")
print("=" * 80)

print(
    f"\nOutput directory:\n{OUTPUT_DIR}"
)

print("\nCreated:")

print(
    "  attack_episode_summary.csv"
)

print(
    "  attack_episode_by_split.csv"
)

print(
    "  transition_by_attack_type.csv"
)

print(
    "  transition_by_day.csv"
)

print(
    "  split_attack_family_audit.csv"
)

print(
    "  audit_report.txt"
)

print("\nNext step:")
print(
    "Send me the COMPLETE terminal output."
)

print(
    "\nDo NOT start LSTM yet."
)