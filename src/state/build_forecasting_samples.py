from pathlib import Path

import pandas as pd


INPUT_DIR = Path("data/processed/network_state_v2")
OUTPUT_DIR = Path("data/processed/forecasting_samples")

WINDOW_SECONDS = 30
HISTORY_WINDOWS = 10

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# These are the network-state features.
# Attack_Flow_Count and Attack_Ratio are NOT inputs.
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


print("=" * 75)
print("SIH26153 FORECASTING SAMPLE BUILDER")
print("=" * 75)

state_files = sorted(INPUT_DIR.glob("*_30s_state.csv"))

print(f"\nState files found: {len(state_files)}")
print(f"History length: {HISTORY_WINDOWS} windows")
print(f"History duration: {HISTORY_WINDOWS * WINDOW_SECONDS} seconds")
print("Forecast horizon: next 30-second window")


all_summary = []
all_samples = []


for file_path in state_files:

    print("\n" + "-" * 75)
    print(f"Processing: {file_path.name}")
    print("-" * 75)

    df = pd.read_csv(file_path)

    # ---------------------------------------------------------
    # Timestamp
    # ---------------------------------------------------------

    df["Window"] = pd.to_datetime(
        df["Window"],
        errors="coerce"
    )

    if df["Window"].isna().any():
        print("❌ Invalid Window values found.")
        continue

    # ---------------------------------------------------------
    # Sort and remove duplicate windows
    # ---------------------------------------------------------

    df = (
        df.sort_values("Window")
        .drop_duplicates("Window")
        .reset_index(drop=True)
    )

    # ---------------------------------------------------------
    # Validate required columns
    # ---------------------------------------------------------

    required_columns = (
        STATE_FEATURES
        + [
            "Window",
            "Attack_Flow_Count",
            "Attack_Ratio",
        ]
    )

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        print("❌ Missing columns:")
        for col in missing:
            print(f"   {col}")
        continue

    # ---------------------------------------------------------
    # Detect contiguous segments
    # ---------------------------------------------------------

    time_diff = df["Window"].diff()

    new_segment = time_diff != pd.Timedelta(
        seconds=WINDOW_SECONDS
    )

    new_segment.iloc[0] = True

    df["Segment_ID"] = (
        new_segment
        .cumsum()
        .astype(int)
    )

    # ---------------------------------------------------------
    # Build forecasting samples
    # ---------------------------------------------------------

    file_samples = []

    for segment_id, segment in df.groupby(
        "Segment_ID",
        sort=True
    ):

        segment = (
            segment
            .sort_values("Window")
            .reset_index(drop=True)
        )

        # Need:
        # HISTORY_WINDOWS historical states
        # + 1 future target window

        if len(segment) <= HISTORY_WINDOWS:
            continue

        for i in range(
            HISTORY_WINDOWS,
            len(segment)
        ):

            history = segment.iloc[
                i - HISTORY_WINDOWS:i
            ]

            future = segment.iloc[i]

            # Safety check:
            # history must really contain consecutive
            # 30-second windows.

            history_diffs = (
                history["Window"]
                .diff()
                .dropna()
            )

            if not (
                history_diffs
                == pd.Timedelta(seconds=WINDOW_SECONDS)
            ).all():
                continue

            # Last history window must be exactly
            # 30 seconds before target.
            if (
                future["Window"]
                - history.iloc[-1]["Window"]
                != pd.Timedelta(seconds=WINDOW_SECONDS)
            ):
                continue

            sample = {
                "Source_File": file_path.name,
                "Segment_ID": segment_id,

                "History_Start": history.iloc[0]["Window"],
                "History_End": history.iloc[-1]["Window"],

                "Target_Window": future["Window"],

                "Target_Attack": int(
                    future["Attack_Flow_Count"] > 0
                ),

                "Target_Attack_Ratio":
                    future["Attack_Ratio"],
            }

            # Flatten the 10 historical states.
            #
            # Example:
            # t-9_Flow_Count
            # t-8_Flow_Count
            # ...
            # t_Flow_Count

            for step, (_, row) in enumerate(
                history.iterrows(),
                start=1
            ):

                for feature in STATE_FEATURES:

                    sample[
                        f"t-{HISTORY_WINDOWS - step}_"
                        f"{feature}"
                    ] = row[feature]

            file_samples.append(sample)

    # ---------------------------------------------------------
    # Save per-file samples
    # ---------------------------------------------------------

    if file_samples:

        file_samples_df = pd.DataFrame(
            file_samples
        )

        output_name = (
            file_path.stem
            + "_forecast_samples.csv"
        )

        output_path = OUTPUT_DIR / output_name

        file_samples_df.to_csv(
            output_path,
            index=False
        )

        attack_count = int(
            file_samples_df["Target_Attack"].sum()
        )

        benign_count = (
            len(file_samples_df)
            - attack_count
        )

        print(
            f"  Samples:              {len(file_samples_df)}"
        )
        print(
            f"  Target attack:        {attack_count}"
        )
        print(
            f"  Target benign:        {benign_count}"
        )
        print(
            f"  Segments:              "
            f"{file_samples_df['Segment_ID'].nunique()}"
        )

        all_samples.append(file_samples_df)

        all_summary.append({
            "Source_File": file_path.name,
            "Samples": len(file_samples_df),
            "Attack_Targets": attack_count,
            "Benign_Targets": benign_count,
            "Segments_Used":
                file_samples_df["Segment_ID"].nunique(),
        })

    else:

        print("  ⚠️ No usable forecasting samples.")


# -------------------------------------------------------------
# Combine all files
# -------------------------------------------------------------

if all_samples:

    combined = pd.concat(
        all_samples,
        ignore_index=True
    )

    combined_path = (
        OUTPUT_DIR
        / "all_forecasting_samples.csv"
    )

    combined.to_csv(
        combined_path,
        index=False
    )

    summary_df = pd.DataFrame(
        all_summary
    )

    summary_path = (
        OUTPUT_DIR
        / "forecasting_sample_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False
    )

    print("\n" + "=" * 75)
    print("FORECASTING SAMPLE BUILD COMPLETE")
    print("=" * 75)

    print(
        f"\nTotal samples: "
        f"{len(combined):,}"
    )

    print(
        f"Attack targets: "
        f"{int(combined['Target_Attack'].sum()):,}"
    )

    print(
        f"Benign targets: "
        f"{int((combined['Target_Attack'] == 0).sum()):,}"
    )

    print(
        f"\nCombined file:"
        f"\n  {combined_path}"
    )

    print(
        f"\nSummary:"
        f"\n  {summary_path}"
    )

else:

    print("\n❌ No forecasting samples were generated.")


print("\nIMPORTANT:")
print("Do NOT train an LSTM yet.")
print("Do NOT normalize yet.")
print("Do NOT randomly split the data yet.")
print("=" * 75)