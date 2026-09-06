from pathlib import Path
from collections import Counter

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_DIR = Path("data/processed/network_state_v2")
OUTPUT_DIR = Path("results/temporal_continuity")

WINDOW_SECONDS = 30

# Number of largest individual gaps to display per file
TOP_GAPS_TO_SHOW = 15


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


# ============================================================
# HEADER
# ============================================================

print("=" * 75)
print("SIH26153 GAP DISTRIBUTION AUDIT")
print("=" * 75)

if not state_files:
    print("\n❌ No network state files found.")
    print(f"Expected directory: {INPUT_DIR}")
    raise SystemExit(1)

print(f"\nState files found: {len(state_files)}")


# ============================================================
# STORAGE FOR GLOBAL SUMMARY
# ============================================================

global_gap_counts = Counter()

summary_rows = []
gap_rows = []


# ============================================================
# PROCESS EACH FILE
# ============================================================

for file_path in state_files:

    print("\n" + "-" * 75)
    print(f"Processing: {file_path.name}")
    print("-" * 75)

    try:
        df = pd.read_csv(
            file_path,
            usecols=["Window"]
        )
    except Exception as e:
        print(f"❌ Could not read file: {e}")
        continue

    # --------------------------------------------------------
    # Parse Window
    # --------------------------------------------------------

    df["Window"] = pd.to_datetime(
        df["Window"],
        errors="coerce"
    )

    invalid = int(
        df["Window"].isna().sum()
    )

    if invalid > 0:
        print(f"❌ Invalid Window values: {invalid}")
        continue

    # --------------------------------------------------------
    # Sort + remove duplicate timestamps
    # --------------------------------------------------------

    df = (
        df.sort_values("Window")
        .drop_duplicates("Window")
        .reset_index(drop=True)
    )

    if len(df) < 2:
        print("⚠️ Not enough windows for gap analysis.")
        continue

    # --------------------------------------------------------
    # Calculate differences
    # --------------------------------------------------------

    gaps = (
        df["Window"]
        .diff()
        .dropna()
    )

    expected_gap = pd.Timedelta(
        seconds=WINDOW_SECONDS
    )

    # --------------------------------------------------------
    # Only gaps larger than expected 30 seconds
    # --------------------------------------------------------

    larger_gaps = gaps[
        gaps > expected_gap
    ]

    # --------------------------------------------------------
    # Convert each gap to number of missing 30s windows
    #
    # Example:
    # 60 sec gap → 1 missing window
    # 90 sec gap → 2 missing windows
    # 3 min gap → 5 missing windows
    # --------------------------------------------------------

    missing_windows = (
        (
            larger_gaps.dt.total_seconds()
            / WINDOW_SECONDS
        )
        .round()
        .astype(int)
        - 1
    )

    # --------------------------------------------------------
    # Gap length in seconds
    # --------------------------------------------------------

    gap_seconds = (
        larger_gaps
        .dt.total_seconds()
        .astype(int)
    )

    # --------------------------------------------------------
    # Distribution of missing-window counts
    # --------------------------------------------------------

    file_gap_distribution = Counter(
        missing_windows.tolist()
    )

    global_gap_counts.update(
        file_gap_distribution
    )

    # --------------------------------------------------------
    # Basic statistics
    # --------------------------------------------------------

    total_gaps = len(larger_gaps)

    total_missing_windows = int(
        missing_windows.sum()
    )

    if total_gaps > 0:

        max_gap_seconds = int(
            gap_seconds.max()
        )

        max_gap_minutes = (
            max_gap_seconds / 60
        )

        max_gap_hours = (
            max_gap_seconds / 3600
        )

        mean_gap_seconds = (
            gap_seconds.mean()
        )

        median_gap_seconds = (
            gap_seconds.median()
        )

    else:

        max_gap_seconds = 0
        max_gap_minutes = 0
        max_gap_hours = 0
        mean_gap_seconds = 0
        median_gap_seconds = 0

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print(
        f"  Total state windows:       {len(df)}"
    )

    print(
        f"  Number of gaps >30s:       {total_gaps}"
    )

    print(
        f"  Total missing windows:     {total_missing_windows}"
    )

    print(
        f"  Mean gap:                  "
        f"{mean_gap_seconds:.1f} seconds"
    )

    print(
        f"  Median gap:                "
        f"{median_gap_seconds:.1f} seconds"
    )

    print(
        f"  Maximum gap:               "
        f"{max_gap_seconds} seconds "
        f"({max_gap_minutes:.1f} min / "
        f"{max_gap_hours:.2f} hr)"
    )

    # --------------------------------------------------------
    # Gap categories
    # --------------------------------------------------------

    gap_1_window = int(
        (missing_windows == 1).sum()
    )

    gap_2_5_windows = int(
        (
            (missing_windows >= 2)
            & (missing_windows <= 5)
        ).sum()
    )

    gap_6_20_windows = int(
        (
            (missing_windows >= 6)
            & (missing_windows <= 20)
        ).sum()
    )

    gap_21_100_windows = int(
        (
            (missing_windows >= 21)
            & (missing_windows <= 100)
        ).sum()
    )

    gap_over_100_windows = int(
        (
            missing_windows > 100
        ).sum()
    )

    print("\n  Gap categories:")

    print(
        f"    1 missing window:          "
        f"{gap_1_window}"
    )

    print(
        f"    2–5 missing windows:       "
        f"{gap_2_5_windows}"
    )

    print(
        f"    6–20 missing windows:      "
        f"{gap_6_20_windows}"
    )

    print(
        f"    21–100 missing windows:    "
        f"{gap_21_100_windows}"
    )

    print(
        f"    >100 missing windows:      "
        f"{gap_over_100_windows}"
    )

    # --------------------------------------------------------
    # Store summary
    # --------------------------------------------------------

    summary_rows.append(
        {
            "File": file_path.name,
            "State_Windows": len(df),
            "Gaps_GT_30s": total_gaps,
            "Total_Missing_Windows": total_missing_windows,
            "Gap_1_Window": gap_1_window,
            "Gap_2_5_Windows": gap_2_5_windows,
            "Gap_6_20_Windows": gap_6_20_windows,
            "Gap_21_100_Windows": gap_21_100_windows,
            "Gap_Over_100_Windows": gap_over_100_windows,
            "Mean_Gap_Seconds": round(
                mean_gap_seconds,
                2
            ),
            "Median_Gap_Seconds": round(
                median_gap_seconds,
                2
            ),
            "Max_Gap_Seconds": max_gap_seconds,
        }
    )

    # --------------------------------------------------------
    # Store individual gaps
    # --------------------------------------------------------

    for idx, gap in larger_gaps.items():

        previous_window = df.loc[
            idx - 1,
            "Window"
        ]

        current_window = df.loc[
            idx,
            "Window"
        ]

        gap_seconds_value = int(
            gap.total_seconds()
        )

        missing_count = int(
            round(
                gap_seconds_value
                / WINDOW_SECONDS
            )
        ) - 1

        gap_rows.append(
            {
                "File": file_path.name,
                "Previous_Window": previous_window,
                "Next_Window": current_window,
                "Gap_Seconds": gap_seconds_value,
                "Gap_Minutes": round(
                    gap_seconds_value / 60,
                    2
                ),
                "Missing_30s_Windows": missing_count,
            }
        )

    # --------------------------------------------------------
    # Show largest gaps
    # --------------------------------------------------------

    if total_gaps > 0:

        print(
            f"\n  Top {min(TOP_GAPS_TO_SHOW, total_gaps)} "
            "largest gaps:"
        )

        top_gap_indices = (
            larger_gaps
            .sort_values(
                ascending=False
            )
            .head(TOP_GAPS_TO_SHOW)
            .index
        )

        for idx in top_gap_indices:

            previous_window = df.loc[
                idx - 1,
                "Window"
            ]

            current_window = df.loc[
                idx,
                "Window"
            ]

            gap = (
                current_window
                - previous_window
            )

            missing_count = int(
                gap.total_seconds()
                / WINDOW_SECONDS
            ) - 1

            print(
                f"    {previous_window} → "
                f"{current_window} | "
                f"{gap} | "
                f"missing≈{missing_count}"
            )


# ============================================================
# GLOBAL SUMMARY
# ============================================================

print("\n")
print("=" * 75)
print("GLOBAL GAP DISTRIBUTION")
print("=" * 75)

if not global_gap_counts:

    print("\n✅ No gaps greater than 30 seconds found.")

else:

    total_global_gaps = sum(
        global_gap_counts.values()
    )

    total_global_missing = sum(
        missing_count * frequency
        for missing_count, frequency
        in global_gap_counts.items()
    )

    print(
        f"\nTotal gaps across all files: "
        f"{total_global_gaps}"
    )

    print(
        f"Total estimated missing "
        f"30-second windows: "
        f"{total_global_missing}"
    )

    print("\nDistribution:")

    for missing_count in sorted(
        global_gap_counts
    ):

        frequency = global_gap_counts[
            missing_count
        ]

        gap_seconds = (
            (missing_count + 1)
            * WINDOW_SECONDS
        )

        print(
            f"  Missing windows: "
            f"{missing_count:5d} | "
            f"Gap length: "
            f"{gap_seconds:6d}s | "
            f"Occurrences: "
            f"{frequency}"
        )


# ============================================================
# SAVE FILE-LEVEL SUMMARY
# ============================================================

summary_df = pd.DataFrame(
    summary_rows
)

summary_path = (
    OUTPUT_DIR
    / "gap_distribution_summary.csv"
)

summary_df.to_csv(
    summary_path,
    index=False
)


# ============================================================
# SAVE INDIVIDUAL GAP DETAILS
# ============================================================

gap_df = pd.DataFrame(
    gap_rows
)

gap_details_path = (
    OUTPUT_DIR
    / "gap_details.csv"
)

gap_df.to_csv(
    gap_details_path,
    index=False
)


# ============================================================
# PRINT OUTPUT PATHS
# ============================================================

print("\n" + "=" * 75)
print("AUDIT COMPLETE")
print("=" * 75)

print(
    f"\nFile-level summary saved to:\n"
    f"  {summary_path}"
)

print(
    f"\nIndividual gap details saved to:\n"
    f"  {gap_details_path}"
)

print("\n⚠️ IMPORTANT:")
print(
    "Do NOT fill gaps or create LSTM sequences yet."
)
print(
    "Use this report to decide the temporal-gap policy first."
)

print("=" * 75)