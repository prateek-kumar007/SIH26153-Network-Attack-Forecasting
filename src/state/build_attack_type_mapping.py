from pathlib import Path
from collections import defaultdict
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

INPUT_DIR = Path("data/interim/CIC-IDS2018")
OUTPUT_DIR = Path("data/processed/attack_type_mapping")

CHUNK_SIZE = 100_000

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# TIMESTAMP PARSER
# ============================================================

def parse_timestamp(series):
    """
    Handles both:
        2018-03-01 01:00:00
        02/03/2018 01:00:00

    ISO timestamps are parsed first so dayfirst=True
    does not corrupt ISO-formatted dates.
    """

    series = series.astype(str).str.strip()

    result = pd.Series(
        pd.NaT,
        index=series.index,
        dtype="datetime64[ns]"
    )

    iso_mask = series.str.match(
        r"^\d{4}-\d{2}-\d{2} "
    )

    # ISO format
    if iso_mask.any():
        result.loc[iso_mask] = pd.to_datetime(
            series.loc[iso_mask],
            errors="coerce",
            format="%Y-%m-%d %H:%M:%S"
        )

    # DD/MM/YYYY format
    if (~iso_mask).any():
        result.loc[~iso_mask] = pd.to_datetime(
            series.loc[~iso_mask],
            errors="coerce",
            dayfirst=True
        )

    return result


# ============================================================
# GET INPUT FILES
# ============================================================

files = sorted(
    INPUT_DIR.glob(
        "*_TrafficForML_CICFlowMeter.csv"
    )
)

if not files:
    raise FileNotFoundError(
        f"No cleaned CSV files found in:\n{INPUT_DIR}"
    )


print("=" * 70)
print("ATTACK TYPE → 30-SECOND WINDOW MAPPING")
print("=" * 70)

print(f"\nFiles found: {len(files)}")


# ============================================================
# PROCESS EACH FILE
# ============================================================

all_summary = []

for file_path in files:

    print("\n" + "-" * 70)
    print(f"Processing: {file_path.name}")
    print("-" * 70)

    # --------------------------------------------------------
    # Dictionary:
    #
    # window -> attack_type -> count
    # --------------------------------------------------------

    window_labels = defaultdict(
        lambda: defaultdict(int)
    )

    total_rows = 0
    invalid_timestamps = 0

    # --------------------------------------------------------
    # Read only required columns
    # --------------------------------------------------------

    for chunk in pd.read_csv(
        file_path,
        usecols=["Timestamp", "Label"],
        chunksize=CHUNK_SIZE,
        low_memory=False
    ):

        total_rows += len(chunk)

        # Parse timestamp
        chunk["Timestamp"] = parse_timestamp(
            chunk["Timestamp"]
        )

        invalid_mask = chunk["Timestamp"].isna()

        invalid_timestamps += int(
            invalid_mask.sum()
        )

        # Remove invalid timestamps
        chunk = chunk.loc[
            ~invalid_mask
        ].copy()

        if chunk.empty:
            continue

        # Clean labels
        chunk["Label"] = (
            chunk["Label"]
            .astype(str)
            .str.strip()
        )

        # 30-second window
        chunk["Window"] = (
            chunk["Timestamp"]
            .dt.floor("30s")
        )

        # Count labels per window
        grouped = (
            chunk
            .groupby(
                ["Window", "Label"],
                sort=False
            )
            .size()
            .reset_index(name="Flow_Count")
        )

        # Add to global dictionary
        for row in grouped.itertuples(index=False):

            window = row.Window
            label = row.Label
            count = int(row.Flow_Count)

            window_labels[window][label] += count

    # --------------------------------------------------------
    # Convert dictionary to DataFrame
    # --------------------------------------------------------

    records = []

    for window, labels in window_labels.items():

        total_flows = sum(labels.values())

        # ----------------------------------------------------
        # Separate benign and attacks
        # ----------------------------------------------------

        attack_labels = {
            label: count
            for label, count in labels.items()
            if label.lower() != "benign"
        }

        attack_flow_count = sum(
            attack_labels.values()
        )

        attack_ratio = (
            attack_flow_count / total_flows
            if total_flows > 0
            else 0.0
        )

        # ----------------------------------------------------
        # Dominant attack type
        # ----------------------------------------------------

        if attack_labels:

            dominant_attack = max(
                attack_labels,
                key=attack_labels.get
            )

            # All attack types present
            attack_types = "|".join(
                sorted(attack_labels.keys())
            )

        else:

            dominant_attack = "Benign"
            attack_types = ""

        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

        records.append({
            "Window": window,
            "Total_Flows": total_flows,
            "Attack_Flow_Count": attack_flow_count,
            "Attack_Ratio": attack_ratio,
            "Dominant_Attack_Type": dominant_attack,
            "Attack_Types": attack_types,
            "Number_of_Attack_Types": len(
                attack_labels
            ),
        })

    # --------------------------------------------------------
    # DataFrame
    # --------------------------------------------------------

    mapping = pd.DataFrame(records)

    if mapping.empty:
        print("WARNING: No valid windows found.")
        continue

    mapping = mapping.sort_values(
        "Window"
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    duplicate_windows = int(
        mapping["Window"]
        .duplicated()
        .sum()
    )

    chronological = mapping[
        "Window"
    ].is_monotonic_increasing

    attack_windows = int(
        (mapping["Attack_Flow_Count"] > 0)
        .sum()
    )

    benign_windows = int(
        (mapping["Attack_Flow_Count"] == 0)
        .sum()
    )

    mixed_windows = int(
        (mapping["Number_of_Attack_Types"] > 1)
        .sum()
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_name = (
        file_path.stem
        + "_attack_type_mapping.csv"
    )

    output_path = (
        OUTPUT_DIR / output_name
    )

    mapping.to_csv(
        output_path,
        index=False
    )

    # --------------------------------------------------------
    # Attack type distribution
    # --------------------------------------------------------

    attack_distribution = (
        mapping.loc[
            mapping["Attack_Flow_Count"] > 0,
            "Dominant_Attack_Type"
        ]
        .value_counts()
    )

    print(
        f"Rows processed       : {total_rows:,}"
    )

    print(
        f"Invalid timestamps   : {invalid_timestamps:,}"
    )

    print(
        f"30s windows          : {len(mapping):,}"
    )

    print(
        f"Attack windows       : {attack_windows:,}"
    )

    print(
        f"Benign windows       : {benign_windows:,}"
    )

    print(
        f"Mixed attack windows : {mixed_windows:,}"
    )

    print(
        f"Duplicate windows    : {duplicate_windows}"
    )

    print(
        f"Chronological        : {chronological}"
    )

    print("\nAttack types:")

    for attack_type, count in attack_distribution.items():

        print(
            f"  {attack_type:<35} "
            f"{count:,} windows"
        )

    print(
        f"\nSaved: {output_path}"
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    all_summary.append({
        "Source_File": file_path.name,
        "Rows_Processed": total_rows,
        "Invalid_Timestamps": invalid_timestamps,
        "Windows": len(mapping),
        "Attack_Windows": attack_windows,
        "Benign_Windows": benign_windows,
        "Mixed_Attack_Windows": mixed_windows,
        "Duplicate_Windows": duplicate_windows,
        "Chronological": chronological,
    })


# ============================================================
# SAVE GLOBAL SUMMARY
# ============================================================

summary_df = pd.DataFrame(
    all_summary
)

summary_path = (
    OUTPUT_DIR /
    "attack_type_mapping_summary.csv"
)

summary_df.to_csv(
    summary_path,
    index=False
)


# ============================================================
# FINAL RESULT
# ============================================================

print("\n" + "=" * 70)
print("ATTACK TYPE MAPPING COMPLETE")
print("=" * 70)

print(
    f"\nSummary saved to:\n"
    f"{summary_path}"
)

print("\nValidation:")

if (
    summary_df["Invalid_Timestamps"].sum()
    == 0
):
    print("PASS: No invalid timestamps.")

else:
    print(
        "WARNING: Invalid timestamps found."
    )

if (
    summary_df["Duplicate_Windows"].sum()
    == 0
):
    print("PASS: No duplicate windows.")

else:
    print(
        "WARNING: Duplicate windows found."
    )

if (
    summary_df["Chronological"]
    .all()
):
    print(
        "PASS: All mapping files chronological."
    )

else:
    print(
        "WARNING: Chronological issue found."
    )