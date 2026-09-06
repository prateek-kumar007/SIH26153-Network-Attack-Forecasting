from pathlib import Path
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_DIR = Path("data/processed/network_state_v2")
WINDOW_SECONDS = 30


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("SIH26153 TEMPORAL CONTINUITY AUDIT")
print("=" * 70)


# ============================================================
# FIND ONLY ACTUAL NETWORK STATE FILES
# ============================================================

state_files = sorted(
    INPUT_DIR.glob("*_30s_state.csv")
)

if not state_files:
    print("\n❌ No 30-second network state files found.")
    print(f"Expected directory: {INPUT_DIR}")
    raise SystemExit(1)


print(f"\nState files found: {len(state_files)}")


# ============================================================
# AUDIT
# ============================================================

all_pass = True

for file_path in state_files:

    print("\n" + "-" * 70)
    print(f"Processing: {file_path.name}")
    print("-" * 70)

    # --------------------------------------------------------
    # Read only Window column
    # --------------------------------------------------------

    try:
        df = pd.read_csv(
            file_path,
            usecols=["Window"]
        )
    except Exception as e:
        print(f"  ❌ Could not read file: {e}")
        all_pass = False
        continue

    # --------------------------------------------------------
    # Parse timestamps
    # --------------------------------------------------------

    df["Window"] = pd.to_datetime(
        df["Window"],
        errors="coerce"
    )

    invalid = int(df["Window"].isna().sum())

    if invalid > 0:

        print(f"  ❌ Invalid windows: {invalid}")

        all_pass = False
        continue

    # --------------------------------------------------------
    # Sort chronologically for safety
    # --------------------------------------------------------

    df = (
        df.sort_values("Window")
        .drop_duplicates("Window")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Basic information
    # --------------------------------------------------------

    start = df["Window"].min()
    end = df["Window"].max()

    actual_windows = len(df)

    # --------------------------------------------------------
    # Calculate time differences
    # --------------------------------------------------------

    gaps = df["Window"].diff().dropna()

    expected_gap = pd.Timedelta(
        seconds=WINDOW_SECONDS
    )

    exact_gaps = gaps[gaps == expected_gap]

    larger_gaps = gaps[gaps > expected_gap]

    smaller_gaps = gaps[gaps < expected_gap]

    # --------------------------------------------------------
    # Expected number of windows
    # --------------------------------------------------------

    total_seconds = (
        end - start
    ).total_seconds()

    expected_windows = (
        int(total_seconds / WINDOW_SECONDS) + 1
    )

    estimated_missing = max(
        expected_windows - actual_windows,
        0
    )

    # --------------------------------------------------------
    # Maximum gap
    # --------------------------------------------------------

    if len(gaps) > 0:
        max_gap = gaps.max()
    else:
        max_gap = pd.Timedelta(seconds=0)

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print(f"  Start:              {start}")
    print(f"  End:                {end}")
    print(f"  Actual windows:     {actual_windows}")
    print(f"  Expected windows:   {expected_windows}")
    print(f"  Exact 30s gaps:     {len(exact_gaps)}")
    print(f"  Larger gaps:        {len(larger_gaps)}")
    print(f"  Smaller gaps:       {len(smaller_gaps)}")
    print(f"  Maximum gap:        {max_gap}")
    print(f"  Estimated missing:  {estimated_missing}")

    # --------------------------------------------------------
    # Check continuity
    # --------------------------------------------------------

    if len(larger_gaps) == 0:

        print("  ✅ No temporal gaps detected")

    else:

        print("  ⚠️ Temporal gaps detected")

        all_pass = False

        # ----------------------------------------------------
        # Show largest gaps
        # ----------------------------------------------------

        print("\n  Largest gaps:")

        largest_gaps = (
            larger_gaps
            .sort_values(ascending=False)
            .head(10)
        )

        for idx, gap in largest_gaps.items():

            previous_time = df.loc[
                idx - 1,
                "Window"
            ]

            current_time = df.loc[
                idx,
                "Window"
            ]

            missing_count = max(
                int(
                    gap.total_seconds()
                    / WINDOW_SECONDS
                ) - 1,
                0
            )

            print(
                f"    {previous_time} → {current_time} "
                f"| gap={gap} "
                f"| missing≈{missing_count}"
            )


# ============================================================
# FINAL RESULT
# ============================================================

print("\n" + "=" * 70)

if all_pass:

    print("✅ TEMPORAL CONTINUITY AUDIT PASSED")
    print("No missing 30-second windows detected.")

else:

    print("⚠️ TEMPORAL GAPS EXIST")
    print()
    print("DO NOT create LSTM sequences yet.")
    print("We must decide how to handle these gaps first.")

print("=" * 70)