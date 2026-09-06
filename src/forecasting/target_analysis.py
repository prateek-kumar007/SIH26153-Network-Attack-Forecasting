from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# SIH26153 - FORECAST TARGET ANALYSIS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT = ROOT / "data" / "processed" / "forecasting_samples" / "all_forecasting_samples.csv"
OUTPUT_DIR = ROOT / "results" / "target_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HISTORY_WINDOWS = 10
WINDOW_SECONDS = 30


# ============================================================
# HELPERS
# ============================================================

def assign_split(source_file):
    s = str(source_file)

    train_dates = [
        "14-02-2018",
        "15-02-2018",
        "16-02-2018",
        "20-02-2018",
        "21-02-2018",
        "22-02-2018",
        "23-02-2018",
    ]

    validation_dates = ["28-02-2018"]

    test_dates = [
        "01-03-2018",
        "02-03-2018",
    ]

    for d in train_dates:
        if d in s:
            return "TRAIN"

    for d in validation_dates:
        if d in s:
            return "VALIDATION"

    for d in test_dates:
        if d in s:
            return "TEST"

    raise ValueError(f"Unknown source file: {source_file}")


def safe_numeric(df, columns):
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("SIH26153 - FORECAST TARGET ANALYSIS")
print("=" * 70)

print("\nLoading forecasting samples...")

df = pd.read_csv(INPUT)

print(f"Samples : {len(df):,}")
print(f"Columns : {len(df.columns)}")

df["Split"] = df["Source_File"].apply(assign_split)

df["Target_Window"] = pd.to_datetime(
    df["Target_Window"],
    errors="coerce"
)

df = df.sort_values(
    ["Source_File", "Target_Window"]
).reset_index(drop=True)


# ============================================================
# BASIC TARGETS
# ============================================================

df["Target_Attack"] = pd.to_numeric(
    df["Target_Attack"],
    errors="coerce"
).astype(int)

df["Target_Attack_Ratio"] = pd.to_numeric(
    df["Target_Attack_Ratio"],
    errors="coerce"
)

df["Current_Attack_Ratio"] = pd.to_numeric(
    df["t-0_Attack_Ratio"]
    if "t-0_Attack_Ratio" in df.columns
    else np.nan,
    errors="coerce"
)


# ============================================================
# CURRENT STATE ATTACK RATIO
# ============================================================

# Attack_Ratio is metadata, not currently part of the 24 model
# features. Recover the previous target/current state through
# the forecasting history columns if available.

ratio_history_cols = [
    c for c in df.columns
    if c.startswith("t-0_")
    and "Attack_Ratio" in c
]

if ratio_history_cols:
    current_ratio_col = ratio_history_cols[0]
    df["Current_Attack_Ratio"] = pd.to_numeric(
        df[current_ratio_col],
        errors="coerce"
    )


# ============================================================
# TEMPORAL CONTINUITY
# ============================================================

df["Prev_Target"] = (
    df.groupby("Source_File")["Target_Attack"]
    .shift(1)
)

df["Prev_Window"] = (
    df.groupby("Source_File")["Target_Window"]
    .shift(1)
)

df["Seconds_From_Previous"] = (
    df["Target_Window"] - df["Prev_Window"]
).dt.total_seconds()

df["Valid_30s_Transition"] = (
    df["Seconds_From_Previous"] == WINDOW_SECONDS
)


# ============================================================
# TRANSITION TYPES
# ============================================================

transition_mask = df["Valid_30s_Transition"]

df["Transition"] = "START/GAP"

df.loc[
    transition_mask &
    (df["Prev_Target"] == 0) &
    (df["Target_Attack"] == 0),
    "Transition"
] = "0->0"

df.loc[
    transition_mask &
    (df["Prev_Target"] == 0) &
    (df["Target_Attack"] == 1),
    "Transition"
] = "0->1"

df.loc[
    transition_mask &
    (df["Prev_Target"] == 1) &
    (df["Target_Attack"] == 0),
    "Transition"
] = "1->0"

df.loc[
    transition_mask &
    (df["Prev_Target"] == 1) &
    (df["Target_Attack"] == 1),
    "Transition"
] = "1->1"


# ============================================================
# MULTI-STEP TARGETS
# ============================================================

print("\nCreating multi-step targets...")

for step in range(1, 6):

    # Since each row represents the next 30-second target,
    # shift future rows within the same source file.
    future_attack = (
        df.groupby("Source_File")["Target_Attack"]
        .shift(-(step - 1))
    )

    future_ratio = (
        df.groupby("Source_File")["Target_Attack_Ratio"]
        .shift(-(step - 1))
    )

    df[f"Future_Attack_t+{step}"] = future_attack
    df[f"Future_Ratio_t+{step}"] = future_ratio


# ============================================================
# EPISODE ANALYSIS
# ============================================================

print("\nAnalyzing attack episodes...")

episodes = []

for source, g in df.groupby("Source_File"):

    g = g.sort_values("Target_Window").reset_index(drop=True)

    previous_attack = 0
    episode_start = None
    episode_rows = []

    for i, row in g.iterrows():

        current_attack = int(row["Target_Attack"])

        valid_previous = (
            i > 0 and
            pd.notna(row["Seconds_From_Previous"]) and
            row["Seconds_From_Previous"] == WINDOW_SECONDS
        )

        if not valid_previous:
            if episode_rows:
                episodes.append(episode_rows)

            episode_rows = []

            if current_attack == 1:
                episode_rows = [row]

            previous_attack = current_attack
            continue

        if current_attack == 1:
            episode_rows.append(row)

        elif current_attack == 0 and episode_rows:
            episodes.append(episode_rows)
            episode_rows = []

        previous_attack = current_attack

    if episode_rows:
        episodes.append(episode_rows)


episode_records = []

for episode_id, rows in enumerate(episodes, start=1):

    first = rows[0]
    last = rows[-1]

    duration = (
        last["Target_Window"] -
        first["Target_Window"]
    ).total_seconds() + WINDOW_SECONDS

    ratios = pd.to_numeric(
        [r["Target_Attack_Ratio"] for r in rows],
        errors="coerce"
    )

    episode_records.append({
        "Episode_ID": episode_id,
        "Source_File": first["Source_File"],
        "Split": first["Split"],
        "Start": first["Target_Window"],
        "End": last["Target_Window"],
        "Duration_Seconds": duration,
        "Windows": len(rows),
        "Mean_Attack_Ratio": np.nanmean(ratios),
        "Max_Attack_Ratio": np.nanmax(ratios),
    })

episodes_df = pd.DataFrame(episode_records)

episodes_df.to_csv(
    OUTPUT_DIR / "attack_episode_details.csv",
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("1. BASIC TARGET")
print("=" * 70)

print(
    df.groupby(["Split", "Target_Attack"])
    .size()
    .unstack(fill_value=0)
)


# ============================================================
# TRANSITIONS
# ============================================================

print("\n" + "=" * 70)
print("2. TRANSITIONS")
print("=" * 70)

valid_transitions = df[transition_mask]

transition_summary = (
    valid_transitions
    .groupby(["Split", "Transition"])
    .size()
    .reset_index(name="Samples")
)

print(transition_summary.to_string(index=False))

transition_summary.to_csv(
    OUTPUT_DIR / "transition_summary.csv",
    index=False
)


# ============================================================
# ATTACK PERSISTENCE
# ============================================================

print("\n" + "=" * 70)
print("3. ATTACK PERSISTENCE")
print("=" * 70)

persistence_rows = []

for split, g in valid_transitions.groupby("Split"):

    attack_rows = g[g["Prev_Target"] == 1]

    if len(attack_rows) == 0:
        continue

    persistence = (
        (attack_rows["Target_Attack"] == 1).mean()
    )

    persistence_rows.append({
        "Split": split,
        "Previous_Attack_Samples": len(attack_rows),
        "Attack_Persistence": persistence,
    })

persistence_df = pd.DataFrame(persistence_rows)

print(persistence_df.to_string(index=False))

persistence_df.to_csv(
    OUTPUT_DIR / "attack_persistence.csv",
    index=False
)


# ============================================================
# EPISODE DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("4. ATTACK EPISODES")
print("=" * 70)

print(f"Total episodes : {len(episodes_df):,}")

if len(episodes_df) > 0:

    print(
        f"Mean duration   : "
        f"{episodes_df['Duration_Seconds'].mean():.2f} sec"
    )

    print(
        f"Median duration : "
        f"{episodes_df['Duration_Seconds'].median():.2f} sec"
    )

    print(
        f"Min duration    : "
        f"{episodes_df['Duration_Seconds'].min():.2f} sec"
    )

    print(
        f"Max duration    : "
        f"{episodes_df['Duration_Seconds'].max():.2f} sec"
    )

    print("\nEpisode duration distribution:")

    bins = [
        0,
        30,
        60,
        120,
        300,
        600,
        1800,
        3600,
        np.inf
    ]

    labels = [
        "30s",
        "31-60s",
        "61-120s",
        "121-300s",
        "301-600s",
        "601-1800s",
        "1801-3600s",
        ">3600s"
    ]

    episodes_df["Duration_Bin"] = pd.cut(
        episodes_df["Duration_Seconds"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    duration_summary = (
        episodes_df["Duration_Bin"]
        .value_counts()
        .sort_index()
        .reset_index()
    )

    duration_summary.columns = [
        "Duration_Bin",
        "Episodes"
    ]

    print(duration_summary.to_string(index=False))

    duration_summary.to_csv(
        OUTPUT_DIR / "episode_duration_distribution.csv",
        index=False
    )


# ============================================================
# MULTI-STEP TARGET QUALITY
# ============================================================

print("\n" + "=" * 70)
print("5. MULTI-STEP TARGET QUALITY")
print("=" * 70)

multi_rows = []

for step in range(1, 6):

    attack_col = f"Future_Attack_t+{step}"
    ratio_col = f"Future_Ratio_t+{step}"

    valid = df[
        df[attack_col].notna()
    ]

    attack_rate = valid[attack_col].mean()

    ratio_mean = valid[ratio_col].mean()
    ratio_median = valid[ratio_col].median()
    ratio_std = valid[ratio_col].std()

    multi_rows.append({
        "Step": step,
        "Horizon_Seconds": step * WINDOW_SECONDS,
        "Valid_Samples": len(valid),
        "Attack_Rate": attack_rate,
        "Mean_Future_Attack_Ratio": ratio_mean,
        "Median_Future_Attack_Ratio": ratio_median,
        "Std_Future_Attack_Ratio": ratio_std,
    })

multi_df = pd.DataFrame(multi_rows)

print(multi_df.to_string(index=False))

multi_df.to_csv(
    OUTPUT_DIR / "multi_step_target_quality.csv",
    index=False
)


# ============================================================
# FUTURE ATTACK RATIO DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("6. FUTURE ATTACK RATIO")
print("=" * 70)

ratio = df["Target_Attack_Ratio"].dropna()

print(f"Samples : {len(ratio):,}")
print(f"Mean    : {ratio.mean():.6f}")
print(f"Median  : {ratio.median():.6f}")
print(f"Std     : {ratio.std():.6f}")
print(f"Min     : {ratio.min():.6f}")
print(f"Max     : {ratio.max():.6f}")

print("\nPercentiles:")

percentiles = ratio.quantile(
    [0, .10, .25, .50, .75, .90, .95, .99, 1.0]
)

print(percentiles.to_string())

percentiles.to_csv(
    OUTPUT_DIR / "attack_ratio_percentiles.csv"
)


# ============================================================
# CORRELATION: CURRENT VS FUTURE ATTACK RATIO
# ============================================================

print("\n" + "=" * 70)
print("7. CURRENT → FUTURE ATTACK RATIO")
print("=" * 70)

# Current attack ratio can be reconstructed from t-0 history
# only if it exists in the forecasting sample.
if "Current_Attack_Ratio" in df.columns:

    valid_ratio = df[
        df["Current_Attack_Ratio"].notna() &
        df["Target_Attack_Ratio"].notna()
    ]

    if len(valid_ratio) > 0:

        correlation = (
            valid_ratio["Current_Attack_Ratio"]
            .corr(valid_ratio["Target_Attack_Ratio"])
        )

        print(
            f"Current vs next-window ratio correlation: "
            f"{correlation:.6f}"
        )

        print(
            f"Valid pairs: {len(valid_ratio):,}"
        )


# ============================================================
# SAVE ANALYSIS DATA
# ============================================================

analysis_columns = [
    "Source_File",
    "Target_Window",
    "Split",
    "Target_Attack",
    "Target_Attack_Ratio",
    "Prev_Target",
    "Transition",
    "Valid_30s_Transition",
]

for step in range(1, 6):
    analysis_columns.append(f"Future_Attack_t+{step}")
    analysis_columns.append(f"Future_Ratio_t+{step}")

analysis_df = df[analysis_columns].copy()

analysis_df.to_csv(
    OUTPUT_DIR / "target_analysis_samples.csv",
    index=False
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("TARGET ANALYSIS COMPLETE")
print("=" * 70)

print(f"Results saved to:")
print(OUTPUT_DIR)

print("\nDO NOT START LSTM UNTIL THESE RESULTS ARE REVIEWED.")