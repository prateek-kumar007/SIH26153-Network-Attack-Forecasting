from pathlib import Path
import pandas as pd
import numpy as np
import shutil

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# IMPORTANT:
# Read from the externally sorted files, NOT the old interim files.
SORTED_DIR = PROJECT_ROOT / "data" / "sorted" / "CIC-IDS2018"

# New output directory so the old invalid state files remain untouched.
STATE_DIR = PROJECT_ROOT / "data" / "processed" / "network_state_v2"

WINDOW_SECONDS = 30
CHUNK_SIZE = 100_000

# ============================================================
# REQUIRED COLUMNS
# ============================================================

USECOLS = [
    "Timestamp",
    "Flow Duration",
    "Tot Fwd Pkts",
    "Tot Bwd Pkts",
    "TotLen Fwd Pkts",
    "TotLen Bwd Pkts",
    "Flow Byts/s",
    "Flow Pkts/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd Pkts/s",
    "Bwd Pkts/s",
    "Pkt Len Mean",
    "Pkt Len Std",
    "Pkt Len Max",
    "Pkt Len Min",
    "SYN Flag Cnt",
    "ACK Flag Cnt",
    "RST Flag Cnt",
    "PSH Flag Cnt",
    "Protocol",
    "Dst Port",
    "Label",
]


# ============================================================
# TIMESTAMP PARSER
# ============================================================

def parse_timestamp(series):
    """
    Handles both:
        YYYY-MM-DD HH:MM:SS
    and:
        DD/MM/YYYY HH:MM:SS
    """

    series = series.astype(str).str.strip()

    iso_mask = series.str.match(r"^\d{4}-\d{2}-\d{2} ")

    result = pd.Series(
        pd.NaT,
        index=series.index,
        dtype="datetime64[ns]"
    )

    # ISO format
    result.loc[iso_mask] = pd.to_datetime(
        series.loc[iso_mask],
        errors="coerce",
        format="%Y-%m-%d %H:%M:%S"
    )

    # Non-ISO format
    result.loc[~iso_mask] = pd.to_datetime(
        series.loc[~iso_mask],
        errors="coerce",
        dayfirst=True
    )

    return result


# ============================================================
# SAFE NUMERIC CONVERSION
# ============================================================

def numeric(series):
    return pd.to_numeric(series, errors="coerce")


# ============================================================
# AGGREGATE ONE CHUNK/WINDOW
# ============================================================

def summarize_group(group):
    """
    Convert one 30-second group into a compact dictionary.

    Label-derived fields are kept separately as metadata/target
    information and MUST NOT be used as model input features.
    """

    labels = (
        group["Label"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    is_attack = labels != "benign"

    # Replace infinities before numerical aggregation.
    bytes_rate = numeric(group["Flow Byts/s"]).replace(
        [np.inf, -np.inf], np.nan
    )

    packets_rate = numeric(group["Flow Pkts/s"]).replace(
        [np.inf, -np.inf], np.nan
    )

    return {
        # ----------------------------------------------------
        # Traffic/state features
        # ----------------------------------------------------

        "Flow_Count": len(group),

        "Total_Fwd_Pkts": numeric(
            group["Tot Fwd Pkts"]
        ).sum(),

        "Total_Bwd_Pkts": numeric(
            group["Tot Bwd Pkts"]
        ).sum(),

        "Total_Fwd_Bytes": numeric(
            group["TotLen Fwd Pkts"]
        ).sum(),

        "Total_Bwd_Bytes": numeric(
            group["TotLen Bwd Pkts"]
        ).sum(),

        "Mean_Flow_Duration": numeric(
            group["Flow Duration"]
        ).mean(),

        "Mean_Flow_Bytes_Rate": bytes_rate.mean(),

        "Mean_Flow_Packets_Rate": packets_rate.mean(),

        "Mean_Flow_IAT": numeric(
            group["Flow IAT Mean"]
        ).mean(),

        "Mean_Flow_IAT_Std": numeric(
            group["Flow IAT Std"]
        ).mean(),

        "Mean_Flow_IAT_Max": numeric(
            group["Flow IAT Max"]
        ).mean(),

        "Mean_Flow_IAT_Min": numeric(
            group["Flow IAT Min"]
        ).mean(),

        "Mean_Fwd_Packets_Rate": numeric(
            group["Fwd Pkts/s"]
        ).mean(),

        "Mean_Bwd_Packets_Rate": numeric(
            group["Bwd Pkts/s"]
        ).mean(),

        "Mean_Packet_Length": numeric(
            group["Pkt Len Mean"]
        ).mean(),

        "Mean_Packet_Length_Std": numeric(
            group["Pkt Len Std"]
        ).mean(),

        "Max_Packet_Length": numeric(
            group["Pkt Len Max"]
        ).max(),

        "Min_Packet_Length": numeric(
            group["Pkt Len Min"]
        ).min(),

        "SYN_Count": numeric(
            group["SYN Flag Cnt"]
        ).sum(),

        "ACK_Count": numeric(
            group["ACK Flag Cnt"]
        ).sum(),

        "RST_Count": numeric(
            group["RST Flag Cnt"]
        ).sum(),

        "PSH_Count": numeric(
            group["PSH Flag Cnt"]
        ).sum(),

        # ----------------------------------------------------
        # Communication structure
        # ----------------------------------------------------

        "Unique_Dst_Ports": group["Dst Port"].nunique(),

        "Unique_Protocols": group["Protocol"].nunique(),

        # ----------------------------------------------------
        # Target / analysis metadata
        # IMPORTANT:
        # These are NOT model input features.
        # ----------------------------------------------------

        "Attack_Flow_Count": int(is_attack.sum()),

        "Attack_Ratio": float(is_attack.mean()),
    }


# ============================================================
# MERGE TWO WINDOW SUMMARIES
# ============================================================

def merge_summaries(a, b):
    """
    Merge partial summaries belonging to the same 30-second window.

    This is necessary because a single time window can cross a
    CSV chunk boundary.
    """

    total_flows = a["Flow_Count"] + b["Flow_Count"]

    # Helper for weighted means.
    def weighted_mean(
        value_a,
        count_a,
        value_b,
        count_b
    ):
        if count_a == 0 and count_b == 0:
            return np.nan

        numerator = 0.0
        denominator = 0

        if pd.notna(value_a):
            numerator += value_a * count_a
            denominator += count_a

        if pd.notna(value_b):
            numerator += value_b * count_b
            denominator += count_b

        if denominator == 0:
            return np.nan

        return numerator / denominator

    result = {}

    result["Flow_Count"] = total_flows

    # Additive features
    for col in [
        "Total_Fwd_Pkts",
        "Total_Bwd_Pkts",
        "Total_Fwd_Bytes",
        "Total_Bwd_Bytes",
        "SYN_Count",
        "ACK_Count",
        "RST_Count",
        "PSH_Count",
        "Attack_Flow_Count",
    ]:
        result[col] = a[col] + b[col]

    # Weighted means
    mean_columns = [
        "Mean_Flow_Duration",
        "Mean_Flow_Bytes_Rate",
        "Mean_Flow_Packets_Rate",
        "Mean_Flow_IAT",
        "Mean_Flow_IAT_Std",
        "Mean_Flow_IAT_Max",
        "Mean_Flow_IAT_Min",
        "Mean_Fwd_Packets_Rate",
        "Mean_Bwd_Packets_Rate",
        "Mean_Packet_Length",
        "Mean_Packet_Length_Std",
    ]

    for col in mean_columns:
        result[col] = weighted_mean(
            a[col],
            a["Flow_Count"],
            b[col],
            b["Flow_Count"]
        )

    # Max/min
    result["Max_Packet_Length"] = np.nanmax([
        a["Max_Packet_Length"],
        b["Max_Packet_Length"]
    ])

    result["Min_Packet_Length"] = np.nanmin([
        a["Min_Packet_Length"],
        b["Min_Packet_Length"]
    ])

    # Unique values cannot simply be added.
    # We therefore store temporary sets separately below.
    result["_dst_ports"] = (
        a["_dst_ports"] | b["_dst_ports"]
    )

    result["_protocols"] = (
        a["_protocols"] | b["_protocols"]
    )

    result["Unique_Dst_Ports"] = len(result["_dst_ports"])
    result["Unique_Protocols"] = len(result["_protocols"])

    result["Attack_Ratio"] = (
        result["Attack_Flow_Count"] /
        result["Flow_Count"]
        if result["Flow_Count"] > 0
        else 0.0
    )

    return result


# ============================================================
# CREATE SUMMARY WITH SETS
# ============================================================

def summarize_group_with_sets(group):

    summary = summarize_group(group)

    summary["_dst_ports"] = set(
        group["Dst Port"]
        .dropna()
        .tolist()
    )

    summary["_protocols"] = set(
        group["Protocol"]
        .dropna()
        .tolist()
    )

    summary["Unique_Dst_Ports"] = len(
        summary["_dst_ports"]
    )

    summary["Unique_Protocols"] = len(
        summary["_protocols"]
    )

    return summary


# ============================================================
# PROCESS ONE FILE
# ============================================================

def process_file(input_file, output_file):

    print()
    print("=" * 70)
    print(f"PROCESSING: {input_file.name}")
    print("=" * 70)

    expected_rows = 0

    # Count rows first from cleaned/sorted source.
    print("Counting input rows...")

    for chunk in pd.read_csv(
        input_file,
        usecols=["Timestamp"],
        chunksize=CHUNK_SIZE
    ):
        expected_rows += len(chunk)

    print(f"Expected input rows: {expected_rows:,}")

    # Remove previous output if it exists.
    if output_file.exists():
        output_file.unlink()

    current_window = None
    current_summary = None

    rows_seen = 0
    invalid_timestamps = 0
    output_rows = 0

    total_attack_flows = 0
    total_flow_count = 0

    first_timestamp = None
    last_timestamp = None

    previous_window = None

    # --------------------------------------------------------
    # Output columns
    # --------------------------------------------------------

    output_columns = [
        "Window",

        # State features
        "Flow_Count",
        "Total_Fwd_Pkts",
        "Total_Bwd_Pkts",
        "Total_Fwd_Bytes",
        "Total_Bwd_Bytes",
        "Mean_Flow_Duration",
        "Mean_Flow_Bytes_Rate",
        "Mean_Flow_Packets_Rate",
        "Mean_Flow_IAT",
        "Mean_Flow_IAT_Std",
        "Mean_Flow_IAT_Max",
        "Mean_Flow_IAT_Min",
        "Mean_Fwd_Packets_Rate",
        "Mean_Bwd_Packets_Rate",
        "Mean_Packet_Length",
        "Mean_Packet_Length_Std",
        "Max_Packet_Length",
        "Min_Packet_Length",
        "SYN_Count",
        "ACK_Count",
        "RST_Count",
        "PSH_Count",
        "Unique_Dst_Ports",
        "Unique_Protocols",

        # Metadata / target information
        "Attack_Flow_Count",
        "Attack_Ratio",
    ]

    header_written = False

    # --------------------------------------------------------
    # Read sorted file chunk-by-chunk
    # --------------------------------------------------------

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            input_file,
            usecols=USECOLS,
            chunksize=CHUNK_SIZE
        ),
        start=1
    ):

        rows_seen += len(chunk)

        # Robust timestamp parsing
        chunk["Timestamp"] = parse_timestamp(
            chunk["Timestamp"]
        )

        invalid = chunk["Timestamp"].isna().sum()

        if invalid > 0:
            invalid_timestamps += int(invalid)

            # This should NOT happen because timestamp validation
            # already passed.
            chunk = chunk.dropna(
                subset=["Timestamp"]
            )

        if chunk.empty:
            continue

        # Replace infinities.
        numeric_columns = chunk.select_dtypes(
            include=[np.number]
        ).columns

        chunk[numeric_columns] = chunk[
            numeric_columns
        ].replace(
            [np.inf, -np.inf],
            np.nan
        )

        # Create 30-second window.
        chunk["Window"] = (
            chunk["Timestamp"]
            .dt.floor(f"{WINDOW_SECONDS}s")
        )

        # Since the source file is already globally sorted,
        # the groups inside each chunk are also chronological.
        grouped = chunk.groupby(
            "Window",
            sort=True
        )

        for window, group in grouped:

            partial = summarize_group_with_sets(group)

            if current_window is None:

                current_window = window
                current_summary = partial

            elif window == current_window:

                # Same window crossed chunk boundary.
                current_summary = merge_summaries(
                    current_summary,
                    partial
                )

            else:

                # Finalize previous window.
                result = {
                    "Window": current_window,
                    **{
                        key: value
                        for key, value in current_summary.items()
                        if not key.startswith("_")
                    }
                }

                result["Unique_Dst_Ports"] = len(
                    current_summary["_dst_ports"]
                )

                result["Unique_Protocols"] = len(
                    current_summary["_protocols"]
                )

                result = {
                    key: result[key]
                    for key in output_columns
                }

                # Safety check.
                if previous_window is not None:
                    if current_window <= previous_window:
                        raise RuntimeError(
                            f"Window ordering failure: "
                            f"{previous_window} -> {current_window}"
                        )

                previous_window = current_window

                output_df = pd.DataFrame(
                    [result]
                )

                output_df.to_csv(
                    output_file,
                    mode="a",
                    header=not header_written,
                    index=False
                )

                header_written = True

                output_rows += 1

                total_flow_count += (
                    current_summary["Flow_Count"]
                )

                total_attack_flows += (
                    current_summary["Attack_Flow_Count"]
                )

                # Start new window.
                current_window = window
                current_summary = partial

        if chunk_number % 10 == 0:
            print(
                f"  processed {rows_seen:,} rows..."
            )

    # --------------------------------------------------------
    # Final window
    # --------------------------------------------------------

    if current_window is not None:

        result = {
            "Window": current_window,
            **{
                key: value
                for key, value in current_summary.items()
                if not key.startswith("_")
            }
        }

        result["Unique_Dst_Ports"] = len(
            current_summary["_dst_ports"]
        )

        result["Unique_Protocols"] = len(
            current_summary["_protocols"]
        )

        result = {
            key: result[key]
            for key in output_columns
        }

        if previous_window is not None:
            if current_window <= previous_window:
                raise RuntimeError(
                    "Final window ordering failure."
                )

        pd.DataFrame(
            [result]
        ).to_csv(
            output_file,
            mode="a",
            header=not header_written,
            index=False
        )

        output_rows += 1

        total_flow_count += (
            current_summary["Flow_Count"]
        )

        total_attack_flows += (
            current_summary["Attack_Flow_Count"]
        )

    # ========================================================
    # VALIDATION
    # ========================================================

    print()
    print("VALIDATING OUTPUT...")

    state = pd.read_csv(
        output_file
    )

    state["Window"] = pd.to_datetime(
        state["Window"]
    )

    duplicate_windows = int(
        state["Window"].duplicated().sum()
    )

    chronological = bool(
        state["Window"].is_monotonic_increasing
    )

    state_flow_count = int(
        state["Flow_Count"].sum()
    )

    state_attack_count = int(
        state["Attack_Flow_Count"].sum()
    )

    attack_ratio_min = float(
        state["Attack_Ratio"].min()
    )

    attack_ratio_max = float(
        state["Attack_Ratio"].max()
    )

    missing_values = int(
        state.isna().sum().sum()
    )

    # --------------------------------------------------------
    # Validation assertions
    # --------------------------------------------------------

    checks = {
        "Rows read correctly":
            rows_seen == expected_rows,

        "No invalid timestamps":
            invalid_timestamps == 0,

        "No duplicate windows":
            duplicate_windows == 0,

        "Chronological":
            chronological,

        "Flow counts match":
            state_flow_count == expected_rows,

        "Attack counts valid":
            state_attack_count >= 0,

        "Attack ratio >= 0":
            attack_ratio_min >= 0,

        "Attack ratio <= 1":
            attack_ratio_max <= 1,

        "Output rows match":
            output_rows == len(state),
    }

    print()
    print("-" * 70)
    print("VALIDATION RESULTS")
    print("-" * 70)

    all_passed = True

    for check, passed in checks.items():

        status = "PASS" if passed else "FAIL"

        print(
            f"{status:>5} | {check}"
        )

        if not passed:
            all_passed = False

    print()
    print(f"Input rows:             {expected_rows:,}")
    print(f"Rows processed:         {rows_seen:,}")
    print(f"State windows:          {len(state):,}")
    print(f"State flow count:       {state_flow_count:,}")
    print(f"Attack flow count:      {state_attack_count:,}")
    print(f"Duplicate windows:      {duplicate_windows}")
    print(f"Chronological:          {chronological}")
    print(
        f"Attack ratio range:     "
        f"{attack_ratio_min:.6f} → "
        f"{attack_ratio_max:.6f}"
    )
    print(f"Missing values:         {missing_values}")
    print(f"Invalid timestamps:     {invalid_timestamps}")

    if len(state) > 0:
        print(
            f"Time range:             "
            f"{state['Window'].min()} → "
            f"{state['Window'].max()}"
        )

    # Missing values are allowed only if they come from the
    # known Flow Byts/s / Flow Pkts/s infinity issue.
    # We report them rather than silently deleting data.

    print()

    if all_passed:
        print("STATUS: PASS")
    else:
        print("STATUS: FAIL")
        raise RuntimeError(
            f"Validation failed for {input_file.name}. "
            f"Do NOT use this state file for ML."
        )

    print("-" * 70)

    return {
        "file": input_file.name,
        "input_rows": expected_rows,
        "state_windows": len(state),
        "flow_count": state_flow_count,
        "attack_count": state_attack_count,
        "duplicate_windows": duplicate_windows,
        "chronological": chronological,
        "missing_values": missing_values,
        "invalid_timestamps": invalid_timestamps,
        "passed": all_passed,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CIC-IDS2018 NETWORK STATE BUILDER V2")
    print("=" * 70)

    print(f"Input directory : {SORTED_DIR}")
    print(f"Output directory: {STATE_DIR}")
    print(f"Window size     : {WINDOW_SECONDS} seconds")
    print(f"Chunk size      : {CHUNK_SIZE:,}")

    if not SORTED_DIR.exists():
        raise FileNotFoundError(
            f"Sorted directory not found:\n{SORTED_DIR}"
        )

    input_files = sorted(
        SORTED_DIR.glob("*.csv")
    )

    if not input_files:
        raise FileNotFoundError(
            f"No sorted CSV files found in:\n{SORTED_DIR}"
        )

    # Create output directory.
    STATE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        f"\nFound {len(input_files)} sorted files."
    )

    results = []

    for input_file in input_files:

        output_file = (
            STATE_DIR /
            f"{input_file.stem}_30s_state.csv"
        )

        result = process_file(
            input_file,
            output_file
        )

        results.append(result)

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    summary = pd.DataFrame(
        results
    )

    summary_path = (
        STATE_DIR /
        "network_state_v2_validation.csv"
    )

    summary.to_csv(
        summary_path,
        index=False
    )

    print()
    print("=" * 70)
    print("NETWORK STATE V2 BUILD COMPLETE")
    print("=" * 70)

    print(
        f"Files processed: {len(results)}/{len(input_files)}"
    )

    print(
        f"Output directory:\n{STATE_DIR}"
    )

    print()
    print(
        summary[
            [
                "file",
                "input_rows",
                "state_windows",
                "duplicate_windows",
                "chronological",
                "missing_values",
                "passed",
            ]
        ].to_string(index=False)
    )

    if not summary["passed"].all():
        raise RuntimeError(
            "One or more state files failed validation."
        )

    print()
    print("ALL NETWORK STATE V2 FILES PASSED.")
    print("=" * 70)


if __name__ == "__main__":
    main()