from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
)

# ============================================================
# SIH26153 - ONSET + PERSISTENCE BASELINE
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
    / "results"
    / "onset_persistence_baseline"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = 5


# ============================================================
# TIMESTAMP PARSER
# ============================================================

def parse_timestamp(series):
    series = series.astype(str).str.strip()

    iso_mask = series.str.match(r"^\d{4}-\d{2}-\d{2} ")

    result = pd.Series(
        pd.NaT,
        index=series.index,
        dtype="datetime64[ns]"
    )

    result.loc[iso_mask] = pd.to_datetime(
        series.loc[iso_mask],
        errors="coerce",
        format="%Y-%m-%d %H:%M:%S"
    )

    result.loc[~iso_mask] = pd.to_datetime(
        series.loc[~iso_mask],
        errors="coerce",
        dayfirst=True
    )

    return result


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, y_pred):

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    if len(np.unique(y_true)) == 2:
        pr_auc = average_precision_score(
            y_true,
            y_pred
        )
    else:
        pr_auc = np.nan

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "samples": len(y_true),
        "attack": int(np.sum(y_true)),
        "benign": int(len(y_true) - np.sum(y_true)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": pr_auc,
        "fpr": fpr,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("SIH26153 ONSET + PERSISTENCE BASELINE")
print("=" * 70)

print(f"Input: {INPUT_FILE}")
print(f"Output: {OUTPUT_DIR}")

df = pd.read_csv(INPUT_FILE)

print("\nLoaded:")
print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns):,}")


# ============================================================
# VALIDATION
# ============================================================

required = [
    "Source_File",
    "Target_Window",
    "Target_Attack",
]

missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )

df["Target_Window"] = parse_timestamp(
    df["Target_Window"]
)

if df["Target_Window"].isna().any():
    raise ValueError("Invalid Target_Window timestamps found.")

df["Target_Attack"] = (
    pd.to_numeric(df["Target_Attack"], errors="coerce")
    .astype("Int64")
)

if df["Target_Attack"].isna().any():
    raise ValueError("Invalid Target_Attack values found.")

df["Target_Attack"] = df["Target_Attack"].astype(int)

if not set(df["Target_Attack"].unique()).issubset({0, 1}):
    raise ValueError("Target_Attack must be binary.")


# ============================================================
# SORT
# ============================================================

df = df.sort_values(
    ["Source_File", "Target_Window"]
).reset_index(drop=True)


# ============================================================
# BUILD CONTIGUOUS SEGMENTS
# ============================================================

print("\n" + "=" * 70)
print("BUILDING CONTIGUOUS SEGMENTS")
print("=" * 70)

df["Prev_Window"] = (
    df.groupby("Source_File")["Target_Window"]
    .shift(1)
)

df["Gap_Seconds"] = (
    df["Target_Window"] - df["Prev_Window"]
).dt.total_seconds()

df["New_Segment"] = (
    df["Prev_Window"].isna()
    | (df["Gap_Seconds"] != 30)
)

df["Segment_ID"] = (
    df.groupby("Source_File")["New_Segment"]
    .cumsum()
)

print(
    f"Total segments: "
    f"{df[['Source_File', 'Segment_ID']].drop_duplicates().shape[0]:,}"
)


# ============================================================
# BUILD 5 FUTURE TARGETS
# ============================================================

print("\n" + "=" * 70)
print("BUILDING 5-HORIZON TARGETS")
print("=" * 70)

for h in range(1, HORIZONS + 1):

    # Current forecasting row represents the first future
    # window (+30s relative to its input history).
    #
    # Therefore:
    # h=1 -> current Target_Attack
    # h=2 -> next forecasting row
    # h=3 -> two rows ahead
    # etc.

    df[f"Y_t+{h * 30}s"] = (
        df.groupby(
            ["Source_File", "Segment_ID"]
        )["Target_Attack"]
        .shift(-(h - 1))
    )


# Keep only rows having all five future horizons.
target_cols = [
    f"Y_t+{h * 30}s"
    for h in range(1, HORIZONS + 1)
]

valid = df[target_cols].notna().all(axis=1)

work = df.loc[valid].copy()

for c in target_cols:
    work[c] = work[c].astype(int)


print(f"Original rows: {len(df):,}")
print(f"Valid 5-horizon rows: {len(work):,}")
print(f"Removed incomplete horizon rows: {len(df) - len(work):,}")


# ============================================================
# DETERMINE CURRENT ATTACK STATE
# ============================================================

# The model input ends 30 seconds before Target_Window.
#
# We recover the attack state immediately BEFORE the first
# forecast window from the previous forecasting target.
#
# If no previous forecasting row exists inside the same
# contiguous segment, this row cannot be used for onset analysis.

work["Current_Attack"] = (
    work.groupby(
        ["Source_File", "Segment_ID"]
    )["Target_Attack"]
    .shift(1)
)

onset_valid = work["Current_Attack"].notna()

onset_df = work.loc[onset_valid].copy()

onset_df["Current_Attack"] = (
    onset_df["Current_Attack"]
    .astype(int)
)


# ============================================================
# 1. PERSISTENCE BASELINE
# ============================================================

print("\n" + "=" * 70)
print("1. 5-HORIZON PERSISTENCE BASELINE")
print("=" * 70)

persistence_rows = []

split_dates = {
    "TRAIN": {
        pd.Timestamp("2018-02-14").date(),
        pd.Timestamp("2018-02-15").date(),
        pd.Timestamp("2018-02-16").date(),
        pd.Timestamp("2018-02-20").date(),
        pd.Timestamp("2018-02-21").date(),
        pd.Timestamp("2018-02-22").date(),
        pd.Timestamp("2018-02-23").date(),
    },
    "VALIDATION": {
        pd.Timestamp("2018-02-28").date(),
    },
    "TEST": {
        pd.Timestamp("2018-03-01").date(),
        pd.Timestamp("2018-03-02").date(),
    },
}


def assign_split(timestamp):
    d = timestamp.date()

    for split, dates in split_dates.items():
        if d in dates:
            return split

    return "UNKNOWN"


work["Split"] = work["Target_Window"].apply(assign_split)

onset_df["Split"] = onset_df["Target_Window"].apply(assign_split)


for split in ["TRAIN", "VALIDATION", "TEST"]:

    split_df = work[
        work["Split"] == split
    ]

    for h in range(1, HORIZONS + 1):

        y_true = split_df[
            f"Y_t+{h * 30}s"
        ].values

        # Persistence:
        # predict future attack = current attack.
        #
        # For the forecasting sample, the first future target
        # is already Target_Attack. The best persistence rule
        # therefore uses the state immediately preceding it.
        current_state = (
            split_df
            .groupby(
                ["Source_File", "Segment_ID"]
            )["Target_Attack"]
            .shift(1)
        )

        valid_p = current_state.notna()

        y_true = y_true[valid_p.values]
        y_pred = current_state[valid_p].astype(int).values

        metrics = calculate_metrics(
            y_true,
            y_pred
        )

        metrics["Split"] = split
        metrics["Horizon"] = f"+{h * 30}s"

        persistence_rows.append(metrics)

        print(
            f"{split:10s} "
            f"+{h * 30:3d}s | "
            f"F1={metrics['f1']:.6f} | "
            f"Precision={metrics['precision']:.6f} | "
            f"Recall={metrics['recall']:.6f} | "
            f"FPR={metrics['fpr']:.6f}"
        )


persistence_metrics = pd.DataFrame(
    persistence_rows
)

persistence_metrics.to_csv(
    OUTPUT_DIR / "persistence_5horizon_metrics.csv",
    index=False
)


# ============================================================
# 2. ACTUAL ATTACK ONSET AUDIT
# ============================================================

print("\n" + "=" * 70)
print("2. ATTACK ONSET AUDIT")
print("=" * 70)

# Onset = current state is benign (0)
# and future state becomes attack (1).

onset_rows = []

for split in ["TRAIN", "VALIDATION", "TEST"]:

    split_df = onset_df[
        onset_df["Split"] == split
    ]

    for h in range(1, HORIZONS + 1):

        target = split_df[
            f"Y_t+{h * 30}s"
        ]

        current = split_df[
            "Current_Attack"
        ]

        actual_onset = (
            (current == 0)
            & (target == 1)
        )

        n_onsets = int(
            actual_onset.sum()
        )

        n_candidates = int(
            (current == 0).sum()
        )

        rate = (
            n_onsets / n_candidates
            if n_candidates > 0
            else 0
        )

        onset_rows.append({
            "Split": split,
            "Horizon": f"+{h * 30}s",
            "Benign_Current_Candidates": n_candidates,
            "Actual_Attack_Onsets": n_onsets,
            "Onset_Rate": rate,
        })

        print(
            f"{split:10s} "
            f"+{h * 30:3d}s | "
            f"candidate benign states={n_candidates:,} | "
            f"actual onsets={n_onsets:,} | "
            f"rate={rate:.6f}"
        )


onset_metrics = pd.DataFrame(
    onset_rows
)

onset_metrics.to_csv(
    OUTPUT_DIR / "attack_onset_audit.csv",
    index=False
)


# ============================================================
# 3. PERSISTENCE'S PRE-ONSET PERFORMANCE
# ============================================================

print("\n" + "=" * 70)
print("3. PERSISTENCE PRE-ONSET PERFORMANCE")
print("=" * 70)

pre_onset_rows = []

for split in ["TRAIN", "VALIDATION", "TEST"]:

    split_df = onset_df[
        onset_df["Split"] == split
    ]

    for h in range(1, HORIZONS + 1):

        y_true = split_df[
            f"Y_t+{h * 30}s"
        ].astype(int)

        current = split_df[
            "Current_Attack"
        ].astype(int)

        # We only evaluate genuine pre-onset situations:
        # current state = benign.
        #
        # Persistence predicts 0 in this situation.
        # Therefore every actual onset becomes a false negative.

        candidate_mask = (
            current == 0
        )

        y_onset = (
            y_true[candidate_mask] == 1
        )

        actual_onsets = int(
            y_onset.sum()
        )

        predicted_onsets = 0

        tp = 0
        fp = 0
        fn = actual_onsets

        precision = 0.0
        recall = 0.0
        f1 = 0.0

        pre_onset_rows.append({
            "Split": split,
            "Horizon": f"+{h * 30}s",
            "Actual_Onsets": actual_onsets,
            "Predicted_Onsets": predicted_onsets,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
        })

        print(
            f"{split:10s} "
            f"+{h * 30:3d}s | "
            f"actual onsets={actual_onsets:,} | "
            f"predicted={predicted_onsets:,} | "
            f"Recall={recall:.6f}"
        )


pre_onset_metrics = pd.DataFrame(
    pre_onset_rows
)

pre_onset_metrics.to_csv(
    OUTPUT_DIR / "persistence_pre_onset_metrics.csv",
    index=False
)


# ============================================================
# 4. ATTACK EPISODE / TRANSITION SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("4. TRANSITION SUMMARY")
print("=" * 70)

transition_rows = []

for split in ["TRAIN", "VALIDATION", "TEST"]:

    split_df = onset_df[
        onset_df["Split"] == split
    ].copy()

    current = split_df["Current_Attack"]

    next_attack = split_df["Y_t+30s"]

    transitions = pd.crosstab(
        current,
        next_attack,
        rownames=["Current"],
        colnames=["Next"],
        dropna=False
    )

    tn = int(
        transitions.loc[0, 0]
        if 0 in transitions.index and 0 in transitions.columns
        else 0
    )

    fp = int(
        transitions.loc[0, 1]
        if 0 in transitions.index and 1 in transitions.columns
        else 0
    )

    fn = int(
        transitions.loc[1, 0]
        if 1 in transitions.index and 0 in transitions.columns
        else 0
    )

    tp = int(
        transitions.loc[1, 1]
        if 1 in transitions.index and 1 in transitions.columns
        else 0
    )

    transition_rows.append({
        "Split": split,
        "0_to_0": tn,
        "0_to_1_ONSET": fp,
        "1_to_0": fn,
        "1_to_1": tp,
    })

    print(
        f"{split:10s} | "
        f"0→0={tn:,} | "
        f"0→1={fp:,} | "
        f"1→0={fn:,} | "
        f"1→1={tp:,}"
    )


transition_summary = pd.DataFrame(
    transition_rows
)

transition_summary.to_csv(
    OUTPUT_DIR / "transition_summary.csv",
    index=False
)


# ============================================================
# 5. WRITE FINAL AUDIT
# ============================================================

audit_file = (
    OUTPUT_DIR
    / "onset_persistence_audit.txt"
)

with open(audit_file, "w", encoding="utf-8") as f:

    f.write(
        "SIH26153 ONSET + PERSISTENCE BASELINE AUDIT\n"
    )
    f.write("=" * 70 + "\n\n")

    f.write(
        "Purpose:\n"
        "Measure how much of the forecasting task is explained by\n"
        "attack persistence and quantify genuine benign-to-attack\n"
        "onset events.\n\n"
    )

    f.write(
        f"Input rows: {len(df):,}\n"
    )

    f.write(
        f"Valid 5-horizon rows: {len(work):,}\n"
    )

    f.write(
        f"Total contiguous segments: "
        f"{df[['Source_File', 'Segment_ID']].drop_duplicates().shape[0]:,}\n\n"
    )

    f.write(
        "IMPORTANT INTERPRETATION:\n"
        "- Persistence is a continuation baseline, not a genuine\n"
        "  pre-attack forecasting model.\n"
        "- A persistence model predicts 0 while the current state\n"
        "  is benign, so it cannot forecast a future 0→1 onset.\n"
        "- Very high persistence scores therefore do not demonstrate\n"
        "  early-warning capability.\n"
        "- Genuine SIH forecasting value should be demonstrated by\n"
        "  improving detection of future attack onset while controlling\n"
        "  false positives.\n"
    )

print("\n" + "=" * 70)
print("FINAL STATUS")
print("=" * 70)

print("PASS")
print()
print("Saved:")
print(f"  {OUTPUT_DIR / 'persistence_5horizon_metrics.csv'}")
print(f"  {OUTPUT_DIR / 'attack_onset_audit.csv'}")
print(f"  {OUTPUT_DIR / 'persistence_pre_onset_metrics.csv'}")
print(f"  {OUTPUT_DIR / 'transition_summary.csv'}")
print(f"  {audit_file}")

print("\nDO NOT TRAIN THE LSTM UNTIL THESE RESULTS ARE REVIEWED.")