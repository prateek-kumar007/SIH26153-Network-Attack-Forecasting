"""
SIH26153 - Persistence Baseline

Baseline idea:
    Predict the next target using the previous target.

    Previous target = 0 -> predict 0
    Previous target = 1 -> predict 1

This establishes how well a trivial "attack continues" assumption
performs before we train Logistic Regression / XGBoost / LSTM.

IMPORTANT:
- Uses the existing forecasting samples.
- Respects TRAIN / VALIDATION / TEST split.
- Does NOT randomly split data.
- Does NOT use future information.
- Does NOT cross source-file boundaries.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
)


# ============================================================
# PATHS
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
    / "persistence_baseline"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SPLIT DEFINITION
# ============================================================

SPLIT_PREFIXES = {
    "TRAIN": [
        "Wednesday-14-02-2018",
        "Thursday-15-02-2018",
        "Friday-16-02-2018",
        "Thuesday-20-02-2018",
        "Wednesday-21-02-2018",
        "Thursday-22-02-2018",
        "Friday-23-02-2018",
    ],
    "VALIDATION": [
        "Wednesday-28-02-2018",
    ],
    "TEST": [
        "Thursday-01-03-2018",
        "Friday-02-03-2018",
    ],
}


# ============================================================
# HELPERS
# ============================================================

def assign_split(source_file):
    """Assign TRAIN / VALIDATION / TEST from source filename."""

    for split, prefixes in SPLIT_PREFIXES.items():
        for prefix in prefixes:
            if str(source_file).startswith(prefix):
                return split

    return None


def calculate_metrics(y_true, y_pred, split_name):
    """Calculate classification metrics."""

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    # Average Precision requires scores.
    # Persistence gives a binary score:
    # previous target itself.
    average_precision = average_precision_score(
        y_true,
        y_pred,
    )

    # ROC-AUC cannot be calculated if y_true contains only one class.
    if len(np.unique(y_true)) == 2:
        roc_auc = roc_auc_score(
            y_true,
            y_pred,
        )
    else:
        roc_auc = np.nan

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    # False Positive Rate
    if (fp + tn) > 0:
        fpr = fp / (fp + tn)
    else:
        fpr = np.nan

    return {
        "Split": split_name,
        "Samples": len(y_true),
        "Actual_Attack": int(np.sum(y_true)),
        "Actual_Benign": int(len(y_true) - np.sum(y_true)),
        "Predicted_Attack": int(np.sum(y_pred)),
        "Predicted_Benign": int(len(y_pred) - np.sum(y_pred)),
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "PR_AUC": average_precision,
        "ROC_AUC": roc_auc,
        "False_Positive_Rate": fpr,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("SIH26153 - PERSISTENCE BASELINE")
    print("=" * 70)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    print(f"\nInput file:")
    print(INPUT_FILE)

    df = pd.read_csv(INPUT_FILE)

    print(f"\nLoaded samples : {len(df):,}")
    print(f"Columns        : {len(df.columns):,}")

    required_columns = [
        "Source_File",
        "Target_Window",
        "Target_Attack",
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    # --------------------------------------------------------
    # Assign split
    # --------------------------------------------------------

    df["Split"] = df["Source_File"].apply(assign_split)

    unknown = df[df["Split"].isna()]

    if len(unknown) > 0:
        print("\nERROR: Unknown source files detected:")

        print(
            unknown["Source_File"]
            .drop_duplicates()
            .to_string(index=False)
        )

        raise ValueError(
            f"{len(unknown):,} samples could not be assigned to a split."
        )

    # --------------------------------------------------------
    # Validate target
    # --------------------------------------------------------

    df["Target_Attack"] = pd.to_numeric(
        df["Target_Attack"],
        errors="coerce",
    )

    if df["Target_Attack"].isna().any():
        raise ValueError(
            "Target_Attack contains invalid/missing values."
        )

    df["Target_Attack"] = df["Target_Attack"].astype(int)

    invalid_target = ~df["Target_Attack"].isin([0, 1])

    if invalid_target.any():
        raise ValueError(
            "Target_Attack contains values other than 0/1."
        )

    # --------------------------------------------------------
    # Sort correctly
    # --------------------------------------------------------

    df["Target_Window"] = pd.to_datetime(
        df["Target_Window"],
        errors="coerce",
    )

    if df["Target_Window"].isna().any():
        raise ValueError(
            "Target_Window contains invalid timestamps."
        )

    df = df.sort_values(
        ["Source_File", "Target_Window"]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Create persistence prediction
    # --------------------------------------------------------
    #
    # For each source file:
    #
    # previous target -> prediction for current target
    #
    # The first sample of each source file has no previous
    # forecasting target, so it is excluded.
    #
    # --------------------------------------------------------

    df["Previous_Target"] = (
        df.groupby("Source_File")["Target_Attack"]
        .shift(1)
    )

    evaluation_df = df.dropna(
        subset=["Previous_Target"]
    ).copy()

    evaluation_df["Previous_Target"] = (
        evaluation_df["Previous_Target"]
        .astype(int)
    )

    evaluation_df["Prediction"] = (
        evaluation_df["Previous_Target"]
    )

    print("\nPersistence rule:")
    print("Previous target = 0 -> Predict 0")
    print("Previous target = 1 -> Predict 1")

    print(
        f"\nExcluded first sample of each source file: "
        f"{len(df) - len(evaluation_df):,}"
    )

    print(
        f"Evaluation samples: "
        f"{len(evaluation_df):,}"
    )

    # ========================================================
    # METRICS
    # ========================================================

    results = []

    print("\n" + "=" * 70)
    print("1. OVERALL PERSISTENCE PERFORMANCE")
    print("=" * 70)

    for split in ["TRAIN", "VALIDATION", "TEST"]:

        subset = evaluation_df[
            evaluation_df["Split"] == split
        ]

        y_true = subset["Target_Attack"].to_numpy()
        y_pred = subset["Prediction"].to_numpy()

        metrics = calculate_metrics(
            y_true,
            y_pred,
            split,
        )

        results.append(metrics)

    results_df = pd.DataFrame(results)

    display_columns = [
        "Split",
        "Samples",
        "Actual_Attack",
        "Actual_Benign",
        "Predicted_Attack",
        "Predicted_Benign",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "PR_AUC",
        "False_Positive_Rate",
    ]

    print(
        results_df[display_columns]
        .to_string(index=False)
    )

    # ========================================================
    # CONFUSION MATRICES
    # ========================================================

    print("\n" + "=" * 70)
    print("2. CONFUSION MATRICES")
    print("=" * 70)

    for split in ["TRAIN", "VALIDATION", "TEST"]:

        subset = evaluation_df[
            evaluation_df["Split"] == split
        ]

        y_true = subset["Target_Attack"].to_numpy()
        y_pred = subset["Prediction"].to_numpy()

        cm = confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1],
        )

        print(f"\n{split}")
        print("                 Predicted")
        print("                 Benign  Attack")
        print(
            f"Actual Benign    {cm[0,0]:6d}  {cm[0,1]:6d}"
        )
        print(
            f"Actual Attack    {cm[1,0]:6d}  {cm[1,1]:6d}"
        )

    # ========================================================
    # ATTACK TRANSITION PERFORMANCE
    # ========================================================

    print("\n" + "=" * 70)
    print("3. PERSISTENCE BEHAVIOUR")
    print("=" * 70)

    correct = (
        evaluation_df["Target_Attack"]
        == evaluation_df["Prediction"]
    )

    print(
        f"\nCorrect persistence predictions : "
        f"{correct.sum():,}"
    )

    print(
        f"Incorrect persistence predictions : "
        f"{(~correct).sum():,}"
    )

    # --------------------------------------------------------
    # Specifically examine Benign -> Attack transitions
    # --------------------------------------------------------

    attack_start = (
        (evaluation_df["Previous_Target"] == 0)
        & (evaluation_df["Target_Attack"] == 1)
    )

    attack_continuation = (
        (evaluation_df["Previous_Target"] == 1)
        & (evaluation_df["Target_Attack"] == 1)
    )

    benign_continuation = (
        (evaluation_df["Previous_Target"] == 0)
        & (evaluation_df["Target_Attack"] == 0)
    )

    attack_end = (
        (evaluation_df["Previous_Target"] == 1)
        & (evaluation_df["Target_Attack"] == 0)
    )

    print(
        f"\nBenign -> Attack transitions : "
        f"{attack_start.sum():,}"
    )

    print(
        f"Attack -> Attack transitions : "
        f"{attack_continuation.sum():,}"
    )

    print(
        f"Attack -> Benign transitions : "
        f"{attack_end.sum():,}"
    )

    print(
        f"Benign -> Benign transitions : "
        f"{benign_continuation.sum():,}"
    )

    # ========================================================
    # BY-DAY PERFORMANCE
    # ========================================================

    print("\n" + "=" * 70)
    print("4. PERFORMANCE BY DAY")
    print("=" * 70)

    daily_results = []

    for source_file, subset in evaluation_df.groupby(
        "Source_File"
    ):

        y_true = subset["Target_Attack"].to_numpy()
        y_pred = subset["Prediction"].to_numpy()

        split = subset["Split"].iloc[0]

        metrics = calculate_metrics(
            y_true,
            y_pred,
            split,
        )

        metrics["Source_File"] = source_file

        daily_results.append(metrics)

    daily_df = pd.DataFrame(daily_results)

    daily_display = [
        "Source_File",
        "Split",
        "Samples",
        "Actual_Attack",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "PR_AUC",
        "False_Positive_Rate",
    ]

    print(
        daily_df[daily_display]
        .to_string(index=False)
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results_path = (
        OUTPUT_DIR
        / "persistence_metrics.csv"
    )

    daily_path = (
        OUTPUT_DIR
        / "persistence_metrics_by_day.csv"
    )

    predictions_path = (
        OUTPUT_DIR
        / "persistence_predictions.csv"
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    daily_df.to_csv(
        daily_path,
        index=False,
    )

    # Save only useful columns instead of all 247 columns.
    prediction_columns = [
        "Source_File",
        "Split",
        "Target_Window",
        "Previous_Target",
        "Target_Attack",
        "Prediction",
    ]

    evaluation_df[prediction_columns].to_csv(
        predictions_path,
        index=False,
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("5. FINAL TEST BASELINE")
    print("=" * 70)

    test_result = results_df[
        results_df["Split"] == "TEST"
    ].iloc[0]

    print(
        f"\nTEST Accuracy  : "
        f"{test_result['Accuracy']:.4f}"
    )

    print(
        f"TEST Precision : "
        f"{test_result['Precision']:.4f}"
    )

    print(
        f"TEST Recall    : "
        f"{test_result['Recall']:.4f}"
    )

    print(
        f"TEST F1        : "
        f"{test_result['F1']:.4f}"
    )

    print(
        f"TEST PR-AUC    : "
        f"{test_result['PR_AUC']:.4f}"
    )

    print(
        f"TEST FPR       : "
        f"{test_result['False_Positive_Rate']:.4f}"
    )

    print("\n" + "=" * 70)
    print("FILES SAVED")
    print("=" * 70)

    print(results_path)
    print(daily_path)
    print(predictions_path)

    print("\n" + "=" * 70)
    print("PERSISTENCE BASELINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()