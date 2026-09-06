from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_DIR = Path("data/processed/network_state_v2")

OUTPUT_DIR = Path(
    "data/processed/contiguous_state"
)

WINDOW_SECONDS = 30


# ============================================================
# SETUP
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

state_files = sorted(
    INPUT_DIR.glob("*_30s_state.csv")
)


print("=" * 75)
print("SIH26153 CONTIGUOUS STATE SEGMENT BUILDER")
print("=" * 75)

print(
    f"\nState files found: {len(state_files)}"
)


# ============================================================
# PROCESS EACH DAY
# ============================================================

all_segment_summary = []


for file_path in state_files:

    print("\n" + "-" * 75)
    print(f"Processing: {file_path.name}")
    print("-" * 75)

    df = pd.read_csv(file_path)

    # --------------------------------------------------------
    # Parse timestamps
    # --------------------------------------------------------

    df["Window"] = pd.to_datetime(
        df["Window"],
        errors="coerce"
    )

    if df["Window"].isna().any():

        print("❌ Invalid Window values found.")
        continue

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    df = (
        df.sort_values("Window")
        .drop_duplicates("Window")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Calculate gap
    # --------------------------------------------------------

    time_diff = df["Window"].diff()

    expected_gap = pd.Timedelta(
        seconds=WINDOW_SECONDS
    )

    # --------------------------------------------------------
    # New segment whenever gap != 30 seconds
    #
    # First row always starts segment 0.
    # --------------------------------------------------------

    new_segment = (
        time_diff != expected_gap
    )

    new_segment.iloc[0] = True

    df["Segment_ID"] = (
        new_segment
        .cumsum()
        .astype(int)
    )

    # --------------------------------------------------------
    # Generate segment statistics
    # --------------------------------------------------------

    segments = (
        df.groupby("Segment_ID")
        .agg(
            Start_Window=("Window", "min"),
            End_Window=("Window", "max"),
            Number_of_Windows=("Window", "count")
        )
        .reset_index()
    )

    # Duration represented by states
    segments["Duration_Seconds"] = (
        segments["Number_of_Windows"] - 1
    ) * WINDOW_SECONDS

    # --------------------------------------------------------
    # Save segment file
    # --------------------------------------------------------

    output_name = (
        file_path.stem
        + "_segments.csv"
    )

    output_path = (
        OUTPUT_DIR / output_name
    )

    segments.to_csv(
        output_path,
        index=False
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    total_segments = len(segments)

    largest_segment = int(
        segments["Number_of_Windows"].max()
    )

    smallest_segment = int(
        segments["Number_of_Windows"].min()
    )

    segments_usable_5 = int(
        (
            segments["Number_of_Windows"] >= 5
        ).sum()
    )

    segments_usable_10 = int(
        (
            segments["Number_of_Windows"] >= 10
        ).sum()
    )

    segments_usable_20 = int(
        (
            segments["Number_of_Windows"] >= 20
        ).sum()
    )

    print(
        f"  Total windows:             {len(df)}"
    )

    print(
        f"  Contiguous segments:       {total_segments}"
    )

    print(
        f"  Smallest segment:          "
        f"{smallest_segment} windows"
    )

    print(
        f"  Largest segment:           "
        f"{largest_segment} windows"
    )

    print(
        f"  Segments >= 5 windows:     "
        f"{segments_usable_5}"
    )

    print(
        f"  Segments >= 10 windows:    "
        f"{segments_usable_10}"
    )

    print(
        f"  Segments >= 20 windows:    "
        f"{segments_usable_20}"
    )

    all_segment_summary.append(
        {
            "File": file_path.name,
            "Total_Windows": len(df),
            "Total_Segments": total_segments,
            "Smallest_Segment": smallest_segment,
            "Largest_Segment": largest_segment,
            "Segments_GE_5": segments_usable_5,
            "Segments_GE_10": segments_usable_10,
            "Segments_GE_20": segments_usable_20,
        }
    )


# ============================================================
# GLOBAL SUMMARY
# ============================================================

summary_df = pd.DataFrame(
    all_segment_summary
)

summary_path = (
    OUTPUT_DIR
    / "contiguous_segment_summary.csv"
)

summary_df.to_csv(
    summary_path,
    index=False
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 75)
print("CONTIGUOUS SEGMENT BUILD COMPLETE")
print("=" * 75)

print(
    f"\nSummary saved to:\n"
    f"  {summary_path}"
)

print(
    f"\nSegment files saved to:\n"
    f"  {OUTPUT_DIR}"
)

print("\nIMPORTANT:")
print(
    "No missing windows were filled."
)

print(
    "Every temporal gap starts a new contiguous segment."
)

print(
    "Do NOT create LSTM sequences yet."
)

print("=" * 75)