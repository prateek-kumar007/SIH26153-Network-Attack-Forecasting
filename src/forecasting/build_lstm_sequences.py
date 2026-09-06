"""
build_lstm_sequences.py

Purpose
-------
Build audited temporal sequences for SIH26153 from the already-created
forecasting samples.

IMPORTANT
---------
This script DOES NOT train an LSTM.

It creates:

    X = 10 historical network-state windows
        shape = (10, 24)

    y = next 5 attack labels
        +30s, +60s, +90s, +120s, +150s

It NEVER crosses a temporal gap.

Input
-----
data/processed/forecasting_samples/all_forecasting_samples.csv

Output
------
data/processed/lstm_sequences/

    X_train.npy
    y_train.npy

    X_validation.npy
    y_validation.npy

    X_test.npy
    y_test.npy

    sequence_metadata.csv
    segment_statistics.csv
    sequence_audit.txt
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "forecasting_samples"
    / "all_forecasting_samples.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "lstm_sequences"
)

WINDOW_SECONDS = 30

HISTORY_WINDOWS = 10
FORECAST_HORIZON = 5

EXPECTED_FEATURE_COUNT = 24

RANDOM_SEED = 42


# Chronological split
TRAIN_DATES = {
    "14-02-2018",
    "15-02-2018",
    "16-02-2018",
    "20-02-2018",
    "21-02-2018",
    "22-02-2018",
    "23-02-2018",
}

VALIDATION_DATES = {
    "28-02-2018",
}

TEST_DATES = {
    "01-03-2018",
    "02-03-2018",
}


# ============================================================
# EXPECTED STATE FEATURES
# ============================================================

STATE_FEATURES = [
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
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def print_header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def extract_date_from_source_file(source_file):
    """
    Extract DD-MM-YYYY from source filename.

    Example:
        Friday-02-03-2018.csv
        -> 02-03-2018
    """

    match = re.search(
        r"(\d{2}-\d{2}-\d{4})",
        str(source_file)
    )

    if match is None:
        return None

    return match.group(1)


def assign_split(date_string):
    """
    Assign chronological train/validation/test split.
    """

    if date_string in TRAIN_DATES:
        return "TRAIN"

    if date_string in VALIDATION_DATES:
        return "VALIDATION"

    if date_string in TEST_DATES:
        return "TEST"

    return "UNASSIGNED"


def parse_timestamp(series):
    """
    Robust timestamp parser.

    CIC-IDS2018 contains both:
        DD/MM/YYYY HH:MM:SS
    and
        YYYY-MM-DD HH:MM:SS

    Therefore we cannot blindly use dayfirst=True.
    """

    series = series.astype(str).str.strip()

    iso_mask = series.str.match(
        r"^\d{4}-\d{2}-\d{2} "
    )

    result = pd.Series(
        pd.NaT,
        index=series.index,
        dtype="datetime64[ns]"
    )

    # ISO format
    result.loc[iso_mask] = pd.to_datetime(
        series.loc[iso_mask],
        errors="coerce",
        format="%Y-%m-%d %H:%M:%S",
    )

    # DD/MM/YYYY format
    result.loc[~iso_mask] = pd.to_datetime(
        series.loc[~iso_mask],
        errors="coerce",
        dayfirst=True,
    )

    return result


def get_history_columns(columns):
    """
    Return the 10 x 24 flattened history columns.

    Expected naming:

        t-9_<feature>
        t-8_<feature>
        ...
        t-0_<feature>
    """

    history_columns = []

    for offset in range(
        HISTORY_WINDOWS - 1,
        -1,
        -1,
    ):
        prefix = f"t-{offset}_"

        for feature in STATE_FEATURES:
            column = prefix + feature

            if column not in columns:
                raise ValueError(
                    f"Missing required history column: {column}"
                )

            history_columns.append(column)

    return history_columns


def build_segments(group):
    """
    Split a Source_File into contiguous temporal segments.

    A new segment begins whenever the gap between consecutive
    Target_Window values is not exactly 30 seconds.

    We DO NOT fill missing windows with zeros.

    This is critical because capture gaps must not become fake
    temporal continuity.
    """

    group = group.sort_values(
        "Target_Window"
    ).reset_index(drop=True)

    timestamps = group["Target_Window"].to_numpy()

    if len(timestamps) == 0:
        return []

    if len(timestamps) == 1:
        return [group]

    differences = np.diff(timestamps)

    expected_gap = np.timedelta64(
        WINDOW_SECONDS,
        "s"
    )

    boundary_indices = np.where(
        differences != expected_gap
    )[0]

    segments = []

    start = 0

    for boundary in boundary_indices:

        end = boundary + 1

        segment = group.iloc[start:end].copy()

        if len(segment) > 0:
            segments.append(segment)

        start = end

    final_segment = group.iloc[start:].copy()

    if len(final_segment) > 0:
        segments.append(final_segment)

    return segments


def build_sequences_from_segment(
    segment,
    history_columns,
):
    """
    Build sequences from one contiguous segment.

    For every forecasting row i:

        X[i] =
            current row's 10 historical network states

        y[i] =
            Target_Attack of rows:
                i
                i+1
                i+2
                i+3
                i+4

    Therefore:

        y[0] = [+30s, +60s, +90s, +120s, +150s]

    because Target_Attack of the current forecasting row already
    represents the first future 30-second window.
    """

    segment = (
        segment
        .sort_values("Target_Window")
        .reset_index(drop=True)
    )

    n = len(segment)

    if n < FORECAST_HORIZON:
        return [], [], []

    X_sequences = []
    y_sequences = []
    metadata_rows = []

    timestamps = segment[
        "Target_Window"
    ].to_numpy()

    for i in range(
        0,
        n - FORECAST_HORIZON + 1,
    ):

        # ----------------------------------------------------
        # Validate future continuity
        # ----------------------------------------------------

        future_timestamps = timestamps[
            i:i + FORECAST_HORIZON
        ]

        future_differences = np.diff(
            future_timestamps
        )

        expected_gap = np.timedelta64(
            WINDOW_SECONDS,
            "s"
        )

        if len(future_differences) > 0:

            if not np.all(
                future_differences
                == expected_gap
            ):
                continue

        # ----------------------------------------------------
        # Build X
        # ----------------------------------------------------

        x_flat = segment.iloc[
            i
        ][history_columns].to_numpy(
            dtype=np.float32
        )

        expected_flat_size = (
            HISTORY_WINDOWS
            * EXPECTED_FEATURE_COUNT
        )

        if len(x_flat) != expected_flat_size:
            raise ValueError(
                "Unexpected X feature count: "
                f"{len(x_flat)} "
                f"(expected {expected_flat_size})"
            )

        # ----------------------------------------------------
        # Validate finite values
        # ----------------------------------------------------

        if not np.all(
            np.isfinite(x_flat)
        ):
            continue

        # ----------------------------------------------------
        # Reshape:
        #
        # flattened:
        #   10 * 24 = 240
        #
        # becomes:
        #   (10, 24)
        # ----------------------------------------------------

        x_sequence = x_flat.reshape(
            HISTORY_WINDOWS,
            EXPECTED_FEATURE_COUNT,
        )

        # ----------------------------------------------------
        # Build 5-step future target
        # ----------------------------------------------------

        y_values = (
            segment.iloc[
                i:i + FORECAST_HORIZON
            ]["Target_Attack"]
            .to_numpy(
                dtype=np.float32
            )
        )

        # ----------------------------------------------------
        # Validate binary labels
        # ----------------------------------------------------

        if not np.all(
            np.isin(
                y_values,
                [0, 1],
            )
        ):
            raise ValueError(
                "Target_Attack contains values "
                "other than 0 or 1."
            )

        # ----------------------------------------------------
        # Time semantics
        #
        # Target_Window = first future window
        #
        # Therefore actual input end is 30s before it.
        # ----------------------------------------------------

        first_forecast_window = (
            future_timestamps[0]
        )

        input_end_window = (
            pd.Timestamp(
                first_forecast_window
            )
            - pd.Timedelta(
                seconds=WINDOW_SECONDS
            )
        )

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        source_file = segment.iloc[
            i
        ]["Source_File"]

        date_string = extract_date_from_source_file(
            source_file
        )

        split = assign_split(
            date_string
        )

        metadata_rows.append(
            {
                "Source_File": source_file,

                "Date": date_string,

                "Split": split,

                "Input_End_Window":
                    input_end_window,

                "Forecast_Window_1":
                    pd.Timestamp(
                        future_timestamps[0]
                    ),

                "Forecast_Window_2":
                    pd.Timestamp(
                        future_timestamps[1]
                    ),

                "Forecast_Window_3":
                    pd.Timestamp(
                        future_timestamps[2]
                    ),

                "Forecast_Window_4":
                    pd.Timestamp(
                        future_timestamps[3]
                    ),

                "Forecast_Window_5":
                    pd.Timestamp(
                        future_timestamps[4]
                    ),

                "Y_t+30s":
                    int(y_values[0]),

                "Y_t+60s":
                    int(y_values[1]),

                "Y_t+90s":
                    int(y_values[2]),

                "Y_t+120s":
                    int(y_values[3]),

                "Y_t+150s":
                    int(y_values[4]),
            }
        )

        X_sequences.append(
            x_sequence
        )

        y_sequences.append(
            y_values
        )

    return (
        X_sequences,
        y_sequences,
        metadata_rows,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    np.random.seed(
        RANDOM_SEED
    )

    print_header(
        "SIH26153 LSTM SEQUENCE BUILDER"
    )

    print(
        f"Project root: {PROJECT_ROOT}"
    )

    print(
        f"Input file: {INPUT_FILE}"
    )

    print(
        f"Output directory: {OUTPUT_DIR}"
    )

    print()
    print(
        "IMPORTANT: This script does NOT train an LSTM."
    )

    print(
        "It only builds and audits temporal sequences."
    )

    # --------------------------------------------------------
    # Check input
    # --------------------------------------------------------

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            "\nInput forecasting dataset not found:\n"
            f"{INPUT_FILE}\n\n"
            "Run the forecasting sample builder first."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print_header(
        "1. LOADING FORECASTING DATA"
    )

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Columns: {len(df.columns):,}"
    )

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required_columns = [
        "Source_File",
        "Target_Window",
        "Target_Attack",
    ]

    missing_required = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing_required:

        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(
                missing_required
            )
        )

    # --------------------------------------------------------
    # History columns
    # --------------------------------------------------------

    print_header(
        "2. VALIDATING HISTORY FEATURES"
    )

    history_columns = get_history_columns(
        df.columns
    )

    print(
        f"History windows: {HISTORY_WINDOWS}"
    )

    print(
        f"Features per state: "
        f"{EXPECTED_FEATURE_COUNT}"
    )

    print(
        f"Total flattened input features: "
        f"{len(history_columns)}"
    )

    expected_flat_size = (
        HISTORY_WINDOWS
        * EXPECTED_FEATURE_COUNT
    )

    if len(history_columns) != expected_flat_size:

        raise ValueError(
            "History feature count mismatch."
        )

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    print_header(
        "3. PARSING TARGET TIMESTAMPS"
    )

    df["Target_Window"] = parse_timestamp(
        df["Target_Window"]
    )

    invalid_timestamp_count = int(
        df["Target_Window"]
        .isna()
        .sum()
    )

    print(
        f"Invalid timestamps: "
        f"{invalid_timestamp_count:,}"
    )

    if invalid_timestamp_count > 0:

        raise ValueError(
            "Invalid Target_Window timestamps found."
        )

    # --------------------------------------------------------
    # Target validation
    # --------------------------------------------------------

    print_header(
        "4. VALIDATING TARGET"
    )

    invalid_target = ~df[
        "Target_Attack"
    ].isin([0, 1])

    invalid_target_count = int(
        invalid_target.sum()
    )

    print(
        f"Invalid Target_Attack values: "
        f"{invalid_target_count:,}"
    )

    if invalid_target_count > 0:

        raise ValueError(
            "Target_Attack must contain only 0 and 1."
        )

    print(
        "Target distribution:"
    )

    print(
        df["Target_Attack"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    # --------------------------------------------------------
    # Duplicate check
    # --------------------------------------------------------

    print_header(
        "5. CHECKING DUPLICATE FORECAST WINDOWS"
    )

    duplicate_count = int(
        df.duplicated(
            subset=[
                "Source_File",
                "Target_Window",
            ]
        ).sum()
    )

    print(
        f"Duplicate "
        f"(Source_File, Target_Window): "
        f"{duplicate_count:,}"
    )

    if duplicate_count > 0:

        raise ValueError(
            "Duplicate forecasting windows found."
        )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    print_header(
        "6. SORTING TEMPORALLY"
    )

    df = (
        df
        .sort_values(
            [
                "Source_File",
                "Target_Window",
            ]
        )
        .reset_index(drop=True)
    )

    print(
        "Sorted by Source_File + Target_Window."
    )

    # --------------------------------------------------------
    # Build segments
    # --------------------------------------------------------

    print_header(
        "7. BUILDING CONTIGUOUS TEMPORAL SEGMENTS"
    )

    all_X = []
    all_y = []
    all_metadata = []
    segment_statistics = []

    total_segments = 0

    for source_file, group in df.groupby(
        "Source_File",
        sort=False,
    ):

        segments = build_segments(
            group
        )

        print(
            f"\n{source_file}"
        )

        print(
            f"  Rows: {len(group):,}"
        )

        print(
            f"  Segments: {len(segments):,}"
        )

        total_segments += len(
            segments
        )

        for segment_number, segment in enumerate(
            segments,
            start=1,
        ):

            segment_length = len(
                segment
            )

            start_time = segment[
                "Target_Window"
            ].iloc[0]

            end_time = segment[
                "Target_Window"
            ].iloc[-1]

            possible_sequences = max(
                0,
                segment_length
                - FORECAST_HORIZON
                + 1,
            )

            (
                X_sequences,
                y_sequences,
                metadata_rows,
            ) = build_sequences_from_segment(
                segment,
                history_columns,
            )

            built_count = len(
                X_sequences
            )

            rejected_count = (
                possible_sequences
                - built_count
            )

            segment_statistics.append(
                {
                    "Source_File":
                        source_file,

                    "Segment":
                        segment_number,

                    "Rows":
                        segment_length,

                    "Start":
                        start_time,

                    "End":
                        end_time,

                    "Possible_Sequences":
                        possible_sequences,

                    "Built_Sequences":
                        built_count,

                    "Rejected_Sequences":
                        rejected_count,
                }
            )

            all_X.extend(
                X_sequences
            )

            all_y.extend(
                y_sequences
            )

            all_metadata.extend(
                metadata_rows
            )

    # --------------------------------------------------------
    # Basic sequence count
    # --------------------------------------------------------

    print_header(
        "8. SEQUENCE BUILD SUMMARY"
    )

    print(
        f"Total contiguous segments: "
        f"{total_segments:,}"
    )

    print(
        f"Total sequences built: "
        f"{len(all_X):,}"
    )

    if len(all_X) == 0:

        raise RuntimeError(
            "No LSTM sequences were created."
        )

    # --------------------------------------------------------
    # Convert to numpy
    # --------------------------------------------------------

    print_header(
        "9. CREATING NUMPY TENSORS"
    )

    X = np.asarray(
        all_X,
        dtype=np.float32,
    )

    y = np.asarray(
        all_y,
        dtype=np.float32,
    )

    print(
        f"X shape: {X.shape}"
    )

    print(
        f"y shape: {y.shape}"
    )

    expected_x_shape = (
        len(all_X),
        HISTORY_WINDOWS,
        EXPECTED_FEATURE_COUNT,
    )

    expected_y_shape = (
        len(all_X),
        FORECAST_HORIZON,
    )

    if X.shape != expected_x_shape:

        raise ValueError(
            f"X shape mismatch: "
            f"{X.shape} != "
            f"{expected_x_shape}"
        )

    if y.shape != expected_y_shape:

        raise ValueError(
            f"y shape mismatch: "
            f"{y.shape} != "
            f"{expected_y_shape}"
        )

    # --------------------------------------------------------
    # Numerical validation
    # --------------------------------------------------------

    print_header(
        "10. NUMERICAL VALIDATION"
    )

    X_finite = bool(
        np.all(
            np.isfinite(X)
        )
    )

    y_finite = bool(
        np.all(
            np.isfinite(y)
        )
    )

    y_binary = bool(
        np.all(
            np.isin(
                y,
                [0, 1],
            )
        )
    )

    print(
        f"X finite: {X_finite}"
    )

    print(
        f"y finite: {y_finite}"
    )

    print(
        f"y binary: {y_binary}"
    )

    if not X_finite:

        raise ValueError(
            "X contains NaN or infinity."
        )

    if not y_finite:

        raise ValueError(
            "y contains NaN or infinity."
        )

    if not y_binary:

        raise ValueError(
            "y contains values other than 0/1."
        )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    print_header(
        "11. BUILDING SEQUENCE METADATA"
    )

    metadata = pd.DataFrame(
        all_metadata
    )

    if len(metadata) != len(X):

        raise ValueError(
            "Metadata row count does not match "
            "sequence count."
        )

    duplicate_sequence_keys = int(
        metadata.duplicated(
            subset=[
                "Source_File",
                "Input_End_Window",
            ]
        ).sum()
    )

    print(
        "Duplicate sequence keys: "
        f"{duplicate_sequence_keys:,}"
    )

    if duplicate_sequence_keys > 0:

        raise ValueError(
            "Duplicate sequence keys found."
        )

    # --------------------------------------------------------
    # Split validation
    # --------------------------------------------------------

    print_header(
        "12. CHRONOLOGICAL TRAIN / VALIDATION / TEST SPLIT"
    )

    split_counts = (
        metadata["Split"]
        .value_counts()
        .sort_index()
    )

    print(
        split_counts.to_string()
    )

    unassigned_count = int(
        (
            metadata["Split"]
            == "UNASSIGNED"
        ).sum()
    )

    if unassigned_count > 0:

        unassigned_dates = sorted(
            metadata.loc[
                metadata["Split"]
                == "UNASSIGNED",
                "Date",
            ]
            .dropna()
            .unique()
            .tolist()
        )

        raise ValueError(
            "Sequences with UNASSIGNED split found.\n"
            f"Dates: {unassigned_dates}"
        )

    # --------------------------------------------------------
    # Check source-file split consistency
    # --------------------------------------------------------

    source_split_counts = (
        metadata
        .groupby("Source_File")["Split"]
        .nunique()
    )

    inconsistent_sources = (
        source_split_counts[
            source_split_counts > 1
        ]
    )

    print(
        f"Source files assigned to multiple splits: "
        f"{len(inconsistent_sources)}"
    )

    if len(inconsistent_sources) > 0:

        raise ValueError(
            "A Source_File appears in multiple splits."
        )

    # --------------------------------------------------------
    # Create masks
    # --------------------------------------------------------

    train_mask = (
        metadata["Split"]
        == "TRAIN"
    ).to_numpy()

    validation_mask = (
        metadata["Split"]
        == "VALIDATION"
    ).to_numpy()

    test_mask = (
        metadata["Split"]
        == "TEST"
    ).to_numpy()

    X_train = X[
        train_mask
    ]

    y_train = y[
        train_mask
    ]

    X_validation = X[
        validation_mask
    ]

    y_validation = y[
        validation_mask
    ]

    X_test = X[
        test_mask
    ]

    y_test = y[
        test_mask
    ]

    # --------------------------------------------------------
    # Split shapes
    # --------------------------------------------------------

    print_header(
        "13. FINAL SPLIT SHAPES"
    )

    print(
        f"TRAIN:"
    )

    print(
        f"  X = {X_train.shape}"
    )

    print(
        f"  y = {y_train.shape}"
    )

    print()

    print(
        f"VALIDATION:"
    )

    print(
        f"  X = {X_validation.shape}"
    )

    print(
        f"  y = {y_validation.shape}"
    )

    print()

    print(
        f"TEST:"
    )

    print(
        f"  X = {X_test.shape}"
    )

    print(
        f"  y = {y_test.shape}"
    )

    # --------------------------------------------------------
    # Attack rates per forecast horizon
    # --------------------------------------------------------

    print_header(
        "14. FORECAST HORIZON DISTRIBUTION"
    )

    horizon_names = [
        "Y_t+30s",
        "Y_t+60s",
        "Y_t+90s",
        "Y_t+120s",
        "Y_t+150s",
    ]

    for split_name, y_split in [
        ("TRAIN", y_train),
        ("VALIDATION", y_validation),
        ("TEST", y_test),
    ]:

        print()
        print(
            split_name
        )

        if len(y_split) == 0:

            print(
                "  No samples."
            )

            continue

        for index, name in enumerate(
            horizon_names
        ):

            rate = float(
                y_split[:, index].mean()
            )

            count = int(
                y_split[:, index].sum()
            )

            total = len(
                y_split
            )

            print(
                f"  {name}: "
                f"{count:,}/{total:,} "
                f"({rate:.4%})"
            )

    # --------------------------------------------------------
    # Save files
    # --------------------------------------------------------

    print_header(
        "15. SAVING OUTPUTS"
    )

    np.save(
        OUTPUT_DIR / "X_train.npy",
        X_train,
    )

    np.save(
        OUTPUT_DIR / "y_train.npy",
        y_train,
    )

    np.save(
        OUTPUT_DIR / "X_validation.npy",
        X_validation,
    )

    np.save(
        OUTPUT_DIR / "y_validation.npy",
        y_validation,
    )

    np.save(
        OUTPUT_DIR / "X_test.npy",
        X_test,
    )

    np.save(
        OUTPUT_DIR / "y_test.npy",
        y_test,
    )

    metadata.to_csv(
        OUTPUT_DIR
        / "sequence_metadata.csv",
        index=False,
    )

    segment_statistics_df = pd.DataFrame(
        segment_statistics
    )

    segment_statistics_df.to_csv(
        OUTPUT_DIR
        / "segment_statistics.csv",
        index=False,
    )

    print(
        "Saved:"
    )

    print(
        "  X_train.npy"
    )

    print(
        "  y_train.npy"
    )

    print(
        "  X_validation.npy"
    )

    print(
        "  y_validation.npy"
    )

    print(
        "  X_test.npy"
    )

    print(
        "  y_test.npy"
    )

    print(
        "  sequence_metadata.csv"
    )

    print(
        "  segment_statistics.csv"
    )

    # --------------------------------------------------------
    # Audit report
    # --------------------------------------------------------

    print_header(
        "16. WRITING AUDIT REPORT"
    )

    audit_file = (
        OUTPUT_DIR
        / "sequence_audit.txt"
    )

    with open(
        audit_file,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "SIH26153 LSTM SEQUENCE AUDIT\n"
        )

        f.write(
            "=" * 70
            + "\n\n"
        )

        f.write(
            "This file contains only sequence-building "
            "and validation results.\n"
        )

        f.write(
            "No LSTM model was trained by this script.\n\n"
        )

        f.write(
            f"Input file: {INPUT_FILE}\n"
        )

        f.write(
            f"Total source rows: {len(df):,}\n"
        )

        f.write(
            f"Total sequences: {len(X):,}\n"
        )

        f.write(
            f"History windows: {HISTORY_WINDOWS}\n"
        )

        f.write(
            f"Features per state: "
            f"{EXPECTED_FEATURE_COUNT}\n"
        )

        f.write(
            f"Forecast horizon: "
            f"{FORECAST_HORIZON}\n"
        )

        f.write(
            f"Window size: "
            f"{WINDOW_SECONDS} seconds\n\n"
        )

        f.write(
            "EXPECTED X SHAPE\n"
        )

        f.write(
            f"{X.shape}\n\n"
        )

        f.write(
            "EXPECTED y SHAPE\n"
        )

        f.write(
            f"{y.shape}\n\n"
        )

        f.write(
            "VALIDATION\n"
        )

        f.write(
            f"X finite: {X_finite}\n"
        )

        f.write(
            f"y finite: {y_finite}\n"
        )

        f.write(
            f"y binary: {y_binary}\n"
        )

        f.write(
            "Duplicate source/input keys: "
            f"{duplicate_sequence_keys}\n"
        )

        f.write(
            f"Invalid timestamps: "
            f"{invalid_timestamp_count}\n"
        )

        f.write(
            f"Invalid target values: "
            f"{invalid_target_count}\n"
        )

        f.write(
            f"Source files assigned to multiple splits: "
            f"{len(inconsistent_sources)}\n\n"
        )

        f.write(
            "SPLIT SIZES\n"
        )

        f.write(
            f"TRAIN: {len(X_train):,}\n"
        )

        f.write(
            f"VALIDATION: "
            f"{len(X_validation):,}\n"
        )

        f.write(
            f"TEST: {len(X_test):,}\n\n"
        )

        f.write(
            "FORECAST HORIZON ATTACK RATES\n"
        )

        for split_name, y_split in [
            ("TRAIN", y_train),
            ("VALIDATION", y_validation),
            ("TEST", y_test),
        ]:

            f.write(
                f"\n{split_name}\n"
            )

            if len(y_split) == 0:

                f.write(
                    "No samples.\n"
                )

                continue

            for index, name in enumerate(
                horizon_names
            ):

                rate = float(
                    y_split[:, index].mean()
                )

                count = int(
                    y_split[:, index].sum()
                )

                total = len(
                    y_split
                )

                f.write(
                    f"{name}: "
                    f"{count}/{total} "
                    f"({rate:.4%})\n"
                )

        f.write(
            "\nSEGMENT SUMMARY\n"
        )

        f.write(
            f"Total segments: "
            f"{len(segment_statistics_df):,}\n"
        )

        f.write(
            f"Possible sequences: "
            f"{int(segment_statistics_df['Possible_Sequences'].sum()):,}\n"
        )

        f.write(
            f"Built sequences: "
            f"{int(segment_statistics_df['Built_Sequences'].sum()):,}\n"
        )

        f.write(
            f"Rejected sequences: "
            f"{int(segment_statistics_df['Rejected_Sequences'].sum()):,}\n"
        )

        f.write(
            "\nFINAL STATUS\n"
        )

        if (
            X_finite
            and y_finite
            and y_binary
            and duplicate_sequence_keys == 0
            and invalid_timestamp_count == 0
            and invalid_target_count == 0
            and len(inconsistent_sources) == 0
            and unassigned_count == 0
        ):

            f.write(
                "PASS - sequence dataset passed all automated checks.\n"
            )

        else:

            f.write(
                "FAIL - one or more automated checks failed.\n"
            )

    print(
        f"Audit saved to:\n{audit_file}"
    )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    print_header(
        "17. FINAL STATUS"
    )

    final_pass = (
        X_finite
        and y_finite
        and y_binary
        and duplicate_sequence_keys == 0
        and invalid_timestamp_count == 0
        and invalid_target_count == 0
        and len(inconsistent_sources) == 0
        and unassigned_count == 0
        and len(X_train) > 0
        and len(X_validation) > 0
        and len(X_test) > 0
    )

    if final_pass:

        print(
            "PASS"
        )

        print()
        print(
            "The temporal sequence dataset passed "
            "the automated checks."
        )

        print()
        print(
            "DO NOT TRAIN THE LSTM YET."
        )

        print(
            "Review sequence_audit.txt and the "
            "printed statistics first."
        )

    else:

        print(
            "FAIL"
        )

        print()
        print(
            "The sequence dataset did not pass "
            "all checks."
        )

        raise RuntimeError(
            "Sequence audit failed."
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()