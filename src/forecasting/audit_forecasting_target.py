from pathlib import Path
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(r"D:\SIH26153")

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "forecasting_samples"
    / "all_forecasting_samples.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "forecasting_target_audit"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SPLIT DEFINITION
# ============================================================

# We identify the source day from the beginning of Source_File.
# This avoids problems caused by the long generated filename.

TRAIN_PREFIXES = {
    "Wednesday-14-02-2018",
    "Thursday-15-02-2018",
    "Friday-16-02-2018",
    "Thuesday-20-02-2018",
    "Wednesday-21-02-2018",
    "Thursday-22-02-2018",
    "Friday-23-02-2018",
}

VALIDATION_PREFIXES = {
    "Wednesday-28-02-2018",
}

TEST_PREFIXES = {
    "Thursday-01-03-2018",
    "Friday-02-03-2018",
}


def get_split(source_file):
    """
    Determine TRAIN / VALIDATION / TEST from the
    beginning of the source filename.
    """

    source_file = str(source_file)

    for prefix in TRAIN_PREFIXES:
        if source_file.startswith(prefix):
            return "TRAIN"

    for prefix in VALIDATION_PREFIXES:
        if source_file.startswith(prefix):
            return "VALIDATION"

    for prefix in TEST_PREFIXES:
        if source_file.startswith(prefix):
            return "TEST"

    return "UNKNOWN"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("SIH26153 - FORECASTING TARGET AUDIT")
print("=" * 70)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"\nCould not find forecasting dataset:\n{INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print(f"\nLoaded samples : {len(df):,}")
print(f"Columns        : {len(df.columns)}")


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "Source_File",
    "Target_Attack",
    "Target_Attack_Ratio",
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"\nMissing required columns: {missing_columns}"
    )


# ============================================================
# ASSIGN SPLITS
# ============================================================

# This avoids repeated DataFrame insertion.
splits = df["Source_File"].map(get_split)

df = df.assign(Split=splits)


# ============================================================
# CHECK UNKNOWN FILES
# ============================================================

unknown_files = (
    df.loc[df["Split"] == "UNKNOWN", "Source_File"]
    .drop_duplicates()
    .tolist()
)

if unknown_files:

    print("\nERROR: Unknown source files detected:")

    for filename in unknown_files:
        print(f"  {filename}")

    raise ValueError(
        "\nSome source files are not assigned to TRAIN, "
        "VALIDATION, or TEST."
    )


# ============================================================
# 1. TARGET DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("1. TARGET DISTRIBUTION")
print("=" * 70)

target_distribution = (
    df.groupby(
        ["Split", "Target_Attack"]
    )
    .size()
    .reset_index(name="Samples")
)

target_distribution["Percentage"] = (
    target_distribution
    .groupby("Split")["Samples"]
    .transform(
        lambda x: 100 * x / x.sum()
    )
)

print(
    target_distribution.to_string(
        index=False
    )
)


# ============================================================
# 2. DISTRIBUTION BY SOURCE FILE
# ============================================================

print("\n" + "=" * 70)
print("2. TARGET DISTRIBUTION BY DAY")
print("=" * 70)

by_day = (
    df.groupby(
        [
            "Source_File",
            "Split",
            "Target_Attack",
        ]
    )
    .size()
    .reset_index(name="Samples")
)

print(
    by_day.to_string(
        index=False
    )
)


# ============================================================
# 3. ATTACK RATIO
# ============================================================

print("\n" + "=" * 70)
print("3. TARGET ATTACK RATIO")
print("=" * 70)

ratio_summary = (
    df.groupby("Split")["Target_Attack_Ratio"]
    .agg(
        count="count",
        mean="mean",
        median="median",
        min="min",
        max="max",
    )
    .reset_index()
)

print(
    ratio_summary.to_string(
        index=False
    )
)


# ============================================================
# 4. TARGET TRANSITIONS
# ============================================================

print("\n" + "=" * 70)
print("4. ATTACK TRANSITION ANALYSIS")
print("=" * 70)

transition_records = []

for source_file, group in df.groupby(
    "Source_File",
    sort=False
):

    # Keep chronological order.
    # Target_Window is the forecasted window.
    group = group.sort_values(
        "Target_Window"
    )

    targets = (
        group["Target_Attack"]
        .astype(int)
        .tolist()
    )

    for previous, current in zip(
        targets[:-1],
        targets[1:]
    ):

        transition_records.append(
            {
                "Source_File": source_file,
                "Previous_Target": previous,
                "Current_Target": current,
            }
        )


transitions = pd.DataFrame(
    transition_records
)

transition_counts = (
    transitions
    .groupby(
        [
            "Previous_Target",
            "Current_Target",
        ]
    )
    .size()
    .reset_index(name="Count")
)

transition_counts["Transition"] = (
    transition_counts["Previous_Target"]
    .astype(str)
    + " -> "
    + transition_counts["Current_Target"]
    .astype(str)
)

print(
    transition_counts[
        [
            "Transition",
            "Count",
        ]
    ].to_string(index=False)
)


# ============================================================
# 5. IMPORTANT TRANSITIONS
# ============================================================

def count_transition(previous, current):

    return len(
        transitions[
            (transitions["Previous_Target"] == previous)
            &
            (transitions["Current_Target"] == current)
        ]
    )


benign_to_attack = count_transition(0, 1)
attack_to_attack = count_transition(1, 1)
attack_to_benign = count_transition(1, 0)
benign_to_benign = count_transition(0, 0)


print("\nKey transitions:")

print(
    f"Benign -> Attack : {benign_to_attack:,}"
)

print(
    f"Attack -> Attack : {attack_to_attack:,}"
)

print(
    f"Attack -> Benign  : {attack_to_benign:,}"
)

print(
    f"Benign -> Benign  : {benign_to_benign:,}"
)


# ============================================================
# 6. FORECASTABILITY
# ============================================================

print("\n" + "=" * 70)
print("5. FORECASTABILITY")
print("=" * 70)

total_attack_targets = int(
    (df["Target_Attack"] == 1).sum()
)

if total_attack_targets > 0:

    attack_after_benign_pct = (
        100
        * benign_to_attack
        / total_attack_targets
    )

    attack_after_attack_pct = (
        100
        * attack_to_attack
        / total_attack_targets
    )

else:

    attack_after_benign_pct = 0
    attack_after_attack_pct = 0


print(
    f"Total attack targets : "
    f"{total_attack_targets:,}"
)

print(
    f"Attack targets preceded by benign target : "
    f"{attack_after_benign_pct:.2f}%"
)

print(
    f"Attack targets preceded by attack target : "
    f"{attack_after_attack_pct:.2f}%"
)


# ============================================================
# 7. FINAL SPLIT SUMMARY
# ============================================================

split_summary = (
    df.groupby("Split")
    .agg(
        Samples=("Target_Attack", "size"),
        Attack=("Target_Attack", "sum"),
    )
    .reset_index()
)

split_summary["Benign"] = (
    split_summary["Samples"]
    - split_summary["Attack"]
)

split_summary["Attack_Percentage"] = (
    100
    * split_summary["Attack"]
    / split_summary["Samples"]
)


print("\n" + "=" * 70)
print("6. FINAL SPLIT SUMMARY")
print("=" * 70)

print(
    split_summary.to_string(
        index=False
    )
)


# ============================================================
# 8. SAVE RESULTS
# ============================================================

target_distribution.to_csv(
    OUTPUT_DIR / "target_distribution.csv",
    index=False
)

by_day.to_csv(
    OUTPUT_DIR / "target_distribution_by_day.csv",
    index=False
)

ratio_summary.to_csv(
    OUTPUT_DIR / "target_attack_ratio_summary.csv",
    index=False
)

transition_counts.to_csv(
    OUTPUT_DIR / "target_transitions.csv",
    index=False
)

split_summary.to_csv(
    OUTPUT_DIR / "target_split_summary.csv",
    index=False
)


# ============================================================
# 9. REPORT
# ============================================================

report = []

report.append(
    "SIH26153 - FORECASTING TARGET AUDIT"
)

report.append("=" * 60)

report.append(
    f"Total samples: {len(df):,}"
)

report.append(
    "History: 10 x 30-second windows = 5 minutes"
)

report.append(
    "Target: attack presence in next 30-second window"
)

report.append("")

report.append("SPLIT SUMMARY")
report.append("-" * 60)

for _, row in split_summary.iterrows():

    report.append(
        f"{row['Split']}: "
        f"{int(row['Samples']):,} samples | "
        f"{int(row['Attack']):,} attack | "
        f"{int(row['Benign']):,} benign | "
        f"{row['Attack_Percentage']:.2f}% attack"
    )


report.append("")
report.append("TRANSITIONS")
report.append("-" * 60)

report.append(
    f"Benign -> Attack: {benign_to_attack:,}"
)

report.append(
    f"Attack -> Attack: {attack_to_attack:,}"
)

report.append(
    f"Attack -> Benign: {attack_to_benign:,}"
)

report.append(
    f"Benign -> Benign: {benign_to_benign:,}"
)

report.append("")
report.append("INTERPRETATION")
report.append("-" * 60)

report.append(
    "This audit does not train a model."
)

report.append(
    "It verifies whether the forecasting target "
    "has useful temporal structure."
)

report.append(
    "The final choice of history length and forecast "
    "horizon will be based on this audit and later "
    "validation experiments."
)


with open(
    OUTPUT_DIR
    / "forecasting_target_audit_report.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "\n".join(report)
    )


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)

print(
    f"Results saved to:\n{OUTPUT_DIR}"
)