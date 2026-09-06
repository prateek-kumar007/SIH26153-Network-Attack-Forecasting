from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Cleaned CIC-IDS2018 files
# IMPORTANT: cleaning.py stores them here
CLEAN_DIR = PROJECT_ROOT / "data" / "interim" / "CIC-IDS2018"

# Generated network-state files
STATE_DIR = PROJECT_ROOT / "data" / "processed" / "network_state"

STATE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

WINDOW_SECONDS = 30
CHUNK_SIZE = 100_000

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
# HELPER FUNCTIONS
# ============================================================

def clean_numeric(series):
    """
    Convert a Series to numeric and replace +/- infinity with NaN.
    """
    return pd.to_numeric(series, errors="coerce").replace(
        [np.inf, -np.inf],
        np.nan
    )


def aggregate_window(window_df):
    """
    Convert all flows belonging to one 30-second window
    into one network-state row.
    """

    # --------------------------------------------------------
    # Attack identification
    # --------------------------------------------------------

    labels = window_df["Label"].astype(str).str.strip()

    is_attack = labels.str.lower() != "benign"

    flow_count = len(window_df)
    attack_flow_count = int(is_attack.sum())

    if flow_count > 0:
        attack_ratio = attack_flow_count / flow_count
    else:
        attack_ratio = 0.0

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    numeric_columns = [
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
    ]

    data = {}

    for col in numeric_columns:
        data[col] = clean_numeric(window_df[col])

    # --------------------------------------------------------
    # Network traffic state
    # --------------------------------------------------------

    state = {
        "Flow_Count": flow_count,
        "Attack_Flow_Count": attack_flow_count,
        "Attack_Ratio": attack_ratio,

        # Traffic volume
        "Total_Fwd_Pkts": data["Tot Fwd Pkts"].sum(),
        "Total_Bwd_Pkts": data["Tot Bwd Pkts"].sum(),
        "Total_Fwd_Bytes": data["TotLen Fwd Pkts"].sum(),
        "Total_Bwd_Bytes": data["TotLen Bwd Pkts"].sum(),

        # Flow behaviour
        "Mean_Flow_Duration": data["Flow Duration"].mean(),
        "Mean_Flow_Bytes_Rate": data["Flow Byts/s"].mean(),
        "Mean_Flow_Packets_Rate": data["Flow Pkts/s"].mean(),

        # Inter-arrival behaviour
        "Mean_Flow_IAT": data["Flow IAT Mean"].mean(),
        "Mean_Flow_IAT_Std": data["Flow IAT Std"].mean(),
        "Mean_Flow_IAT_Max": data["Flow IAT Max"].mean(),
        "Mean_Flow_IAT_Min": data["Flow IAT Min"].mean(),

        # Packet-rate behaviour
        "Mean_Fwd_Packets_Rate": data["Fwd Pkts/s"].mean(),
        "Mean_Bwd_Packets_Rate": data["Bwd Pkts/s"].mean(),

        # Packet-size behaviour
        "Mean_Packet_Length": data["Pkt Len Mean"].mean(),
        "Mean_Packet_Length_Std": data["Pkt Len Std"].mean(),
        "Max_Packet_Length": data["Pkt Len Max"].max(),
        "Min_Packet_Length": data["Pkt Len Min"].min(),

        # TCP behaviour
        "SYN_Count": data["SYN Flag Cnt"].sum(),
        "ACK_Count": data["ACK Flag Cnt"].sum(),
        "RST_Count": data["RST Flag Cnt"].sum(),
        "PSH_Count": data["PSH Flag Cnt"].sum(),

        # Communication behaviour
        "Unique_Dst_Ports": data["Dst Port"].nunique(),

        # Protocol behaviour
        "Unique_Protocols": data["Protocol"].nunique(),
    }

    return state


# ============================================================
# PROCESS ONE FILE
# ============================================================

def process_file(file_path):

    print()
    print("=" * 70)
    print(f"Processing: {file_path.name}")
    print("=" * 70)

    output_name = file_path.stem + f"_{WINDOW_SECONDS}s_state.csv"
    output_path = STATE_DIR / output_name

    window_rows = []

    current_window = None
    current_rows = []

    total_input_rows = 0
    total_attack_rows = 0

    previous_timestamp = None
    out_of_order_count = 0

    # --------------------------------------------------------
    # Read file chunk-by-chunk
    # --------------------------------------------------------

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            file_path,
            usecols=USECOLS,
            chunksize=CHUNK_SIZE,
            low_memory=False
        ),
        start=1
    ):

        total_input_rows += len(chunk)

        # ----------------------------------------------------
        # Timestamp conversion
        # ----------------------------------------------------

        chunk["Timestamp"] = pd.to_datetime(
            chunk["Timestamp"],
            errors="coerce",
            dayfirst=True
        )

        # Remove invalid timestamps
        chunk = chunk.dropna(subset=["Timestamp"])

        if chunk.empty:
            continue

        # ----------------------------------------------------
        # Check chronological order
        # ----------------------------------------------------

        if previous_timestamp is not None:

            first_timestamp = chunk["Timestamp"].iloc[0]

            if first_timestamp < previous_timestamp:
                out_of_order_count += 1

        previous_timestamp = chunk["Timestamp"].iloc[-1]

        # ----------------------------------------------------
        # Attack count
        # ----------------------------------------------------

        labels = chunk["Label"].astype(str).str.strip()

        total_attack_rows += (
            labels.str.lower() != "benign"
        ).sum()

        # ----------------------------------------------------
        # Sort chunk by timestamp
        # ----------------------------------------------------

        chunk = chunk.sort_values("Timestamp")

        # ----------------------------------------------------
        # Create 30-second windows
        # ----------------------------------------------------

        chunk["Window"] = chunk["Timestamp"].dt.floor(
            f"{WINDOW_SECONDS}s"
        )

        # ----------------------------------------------------
        # Process rows window-by-window
        #
        # IMPORTANT:
        # We keep the current window between chunks so that
        # a 30-second window split across two chunks is not
        # accidentally separated.
        # ----------------------------------------------------

        for window_value, window_group in chunk.groupby(
            "Window",
            sort=True
        ):

            if current_window is None:
                current_window = window_value

            # ------------------------------------------------
            # New window detected
            # ------------------------------------------------

            if window_value != current_window:

                if current_rows:

                    window_df = pd.concat(
                        current_rows,
                        ignore_index=True
                    )

                    state = aggregate_window(window_df)

                    state["Window"] = current_window

                    window_rows.append(state)

                current_rows = []
                current_window = window_value

            current_rows.append(
                window_group.drop(columns=["Window"])
            )

    # ========================================================
    # FLUSH FINAL WINDOW
    # ========================================================

    if current_rows:

        window_df = pd.concat(
            current_rows,
            ignore_index=True
        )

        state = aggregate_window(window_df)

        state["Window"] = current_window

        window_rows.append(state)

    # ========================================================
    # CREATE STATE DATAFRAME
    # ========================================================

    if not window_rows:

        print("ERROR: No valid windows generated.")
        return

    state_df = pd.DataFrame(window_rows)

    state_df["Window"] = pd.to_datetime(
        state_df["Window"]
    )

    state_df = state_df.sort_values(
        "Window"
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Replace infinite values
    # --------------------------------------------------------

    state_df = state_df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    state_df.to_csv(
        output_path,
        index=False
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    state_flow_count = int(
        state_df["Flow_Count"].sum()
    )

    state_attack_count = int(
        state_df["Attack_Flow_Count"].sum()
    )

    duplicate_windows = int(
        state_df["Window"].duplicated().sum()
    )

    chronological = bool(
        state_df["Window"].is_monotonic_increasing
    )

    invalid_attack_ratio = int(
        (
            (state_df["Attack_Ratio"] < 0)
            |
            (state_df["Attack_Ratio"] > 1)
        ).sum()
    )

    missing_values = (
        state_df.isna().sum().sum()
    )

    # ========================================================
    # REPORT
    # ========================================================

    print()
    print("STATE SUMMARY")
    print("-" * 70)

    print(f"Input rows:              {total_input_rows:,}")
    print(f"Attack rows:             {total_attack_rows:,}")

    print(f"State windows:           {len(state_df):,}")

    print(f"State flow count:        {state_flow_count:,}")
    print(f"State attack count:      {state_attack_count:,}")

    print(
        f"Flow count matches:      "
        f"{state_flow_count == total_input_rows}"
    )

    print(
        f"Attack count matches:    "
        f"{state_attack_count == total_attack_rows}"
    )

    print(
        f"Chronological:           "
        f"{chronological}"
    )

    print(
        f"Duplicate windows:      "
        f"{duplicate_windows}"
    )

    print(
        f"Invalid attack ratios:   "
        f"{invalid_attack_ratio}"
    )

    print(
        f"Missing values:          "
        f"{missing_values:,}"
    )

    print(
        f"Chunk order warnings:    "
        f"{out_of_order_count}"
    )

    print(
        f"Time range:              "
        f"{state_df['Window'].min()} → "
        f"{state_df['Window'].max()}"
    )

    print()
    print("Attack ratio:")
    print(state_df["Attack_Ratio"].describe())

    print()
    print(f"Saved:")
    print(output_path)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CIC-IDS2018 NETWORK STATE BUILDER")
    print("=" * 70)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Input directory: {CLEAN_DIR}")
    print(f"Output directory: {STATE_DIR}")
    print(f"Window size: {WINDOW_SECONDS} seconds")

    # --------------------------------------------------------
    # Check input directory
    # --------------------------------------------------------

    if not CLEAN_DIR.exists():

        print()
        print("ERROR: Cleaned CIC-IDS2018 directory not found.")
        print(CLEAN_DIR)
        return

    # --------------------------------------------------------
    # Find cleaned CSV files
    # --------------------------------------------------------

    files = sorted(
        CLEAN_DIR.glob("*.csv")
    )

    if not files:

        print()
        print("ERROR: No cleaned CIC-IDS2018 files found.")
        print(CLEAN_DIR)
        return

    print()
    print(f"Found {len(files)} cleaned files.")

    # --------------------------------------------------------
    # Process every file
    # --------------------------------------------------------

    for file_path in files:

        process_file(file_path)

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    generated_files = sorted(
        STATE_DIR.glob(f"*_{WINDOW_SECONDS}s_state.csv")
    )

    print()
    print("=" * 70)
    print("NETWORK STATE BUILD COMPLETE")
    print("=" * 70)

    print(
        f"Generated state files: {len(generated_files)}"
    )

    for file in generated_files:
        print(f"  {file.name}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()