from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
)


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
    / "results"
    / "persistence_baseline_v2"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TIME_GAP_SECONDS = 30


# ============================================================
# SPLIT ASSIGNMENT
# ============================================================

def assign_split(source_file):
    """
    Assign TRAIN / VALIDATION / TEST using the same
    chronological split used for the forecasting dataset.
    """

    source_file = str(source_file)

    if source_file.startswith("Friday-02-03-2018"):
        return "TEST"

    if source_file.startswith("Friday-16-02-2018"):
        return "TRAIN"

    if source_file.startswith("Friday-23-02-2018"):
        return "TRAIN"

    if source_file.startswith("Thuesday-20-02-2018"):
        return "TRAIN"

    if source_file.startswith("Thursday-01-03-2018"):
        return "TEST"

    if source_file.startswith("Thursday-15-02-2018"):
        return "TRAIN"

    if source_file.startswith("Thursday-22-02-2018"):
        return "TRAIN"

    if source_file.startswith("Wednesday-14-02-2018"):
        return "TRAIN"

    if source_file.startswith("Wednesday-21-02-2018"):
        return "TRAIN"

    if source_file.startswith("Wednesday-28-02-2018"):
        return "VALIDATION"

    return "UNKNOWN"


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, y_pred):
    """
    Calculate binary classification metrics.
    """

    accuracy = accuracy_score(y_true, y_pred)

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

    # Persistence produces hard 0/1 predictions.
    # Average precision is still useful as a comparable
    # ranking-style metric, although it is not the main
    # metric for this baseline.
    try:
        pr_auc = average_precision_score(
            y_true,
            y_pred
        )
    except ValueError:
        pr_auc = np.nan

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    if (tn + fp) > 0:
        fpr = fp / (tn + fp)
    else:
        fpr = np.nan

    return {
        "Samples": len(y_true),
        "Actual_Attack": int(np.sum(y_true)),
        "Actual_Benign": int(len(y_true) - np.sum(y_true)),
        "Predicted_Attack": int(np.sum(y_pred)),
        "Predicted_Benign": int(len(y_pred) - np.sum(y_pred)),
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "PR_AUC": pr_auc,
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
    print("SIH26153 - PERSISTENCE BASELINE V2")
    print("GAP-AWARE TEMPORAL BASELINE")
    print("=" * 70)

    print("\nInput file:")
    print(INPUT_FILE)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"\nInput file not found:\n{INPUT_FILE}"
        )

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    df = pd.read_csv(INPUT_FILE)

    print(f"\nLoaded samples : {len(df):,}")
    print(f"Columns        : {len(df.columns):,}")

    required_columns = [
        "Source_File",
        "Target_Window",
        "Target_Attack",
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"\nMissing required columns: {missing_columns}"
        )

    # --------------------------------------------------------
    # ASSIGN SPLITS
    # --------------------------------------------------------

    splits = df["Source_File"].map(assign_split)

    unknown_sources = sorted(
        df.loc[splits == "UNKNOWN", "Source_File"]
        .dropna()
        .unique()
        .tolist()
    )

    if unknown_sources:
        print("\nERROR: Unknown source files detected:")

        for source in unknown_sources:
            print("  ", source)

        raise ValueError(
            "\nUpdate assign_split() before continuing."
        )

    # Avoid DataFrame fragmentation warning.
    df = df.assign(Split=splits)

    # --------------------------------------------------------
    # PARSE TARGET WINDOW
    # --------------------------------------------------------

    df["Target_Window"] = pd.to_datetime(
        df["Target_Window"],
        errors="coerce"
    )

    invalid_windows = df["Target_Window"].isna().sum()

    if invalid_windows > 0:
        raise ValueError(
            f"\nInvalid Target_Window values: {invalid_windows}"
        )

    # --------------------------------------------------------
    # SORT TEMPORALLY
    # --------------------------------------------------------

    df = df.sort_values(
        by=["Source_File", "Target_Window"]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # CALCULATE PREVIOUS WINDOW
    # --------------------------------------------------------

    df["Previous_Target_Window"] = (
        df.groupby("Source_File")["Target_Window"]
        .shift(1)
    )

    df["Previous_Target"] = (
        df.groupby("Source_File")["Target_Attack"]
        .shift(1)
    )

    # --------------------------------------------------------
    # GAP CHECK
    # --------------------------------------------------------

    df["Gap_Seconds"] = (
        df["Target_Window"]
        - df["Previous_Target_Window"]
    ).dt.total_seconds()

    # A valid persistence relationship exists ONLY when
    # consecutive forecasting targets are exactly 30 seconds apart.

    df["Valid_Previous"] = (
        df["Gap_Seconds"] == TIME_GAP_SECONDS
    )

    # First sample of every source file has no previous target.
    df.loc[
        df["Previous_Target_Window"].isna(),
        "Valid_Previous"
    ] = False

    # --------------------------------------------------------
    # GAP STATISTICS
    # --------------------------------------------------------

    source_first_rows = (
        df["Previous_Target_Window"].isna()
    )

    gap_rows = (
        (~source_first_rows)
        & (~df["Valid_Previous"])
    )

    print("\n" + "=" * 70)
    print("1. TEMPORAL CONTINUITY AUDIT")
    print("=" * 70)

    print(
        f"\nFirst sample of each source file : "
        f"{source_first_rows.sum():,}"
    )

    print(
        f"Gap/reset samples detected       : "
        f"{gap_rows.sum():,}"
    )

    print(
        f"Valid 30-second transitions      : "
        f"{df['Valid_Previous'].sum():,}"
    )

    # Distribution of gaps.
    invalid_gap_values = (
        df.loc[
            gap_rows,
            "Gap_Seconds"
        ]
        .dropna()
    )

    if len(invalid_gap_values) > 0:

        print("\nGap statistics:")

        print(
            f"Minimum gap : "
            f"{invalid_gap_values.min():,.0f} seconds"
        )

        print(
            f"Maximum gap : "
            f"{invalid_gap_values.max():,.0f} seconds"
        )

        print(
            f"Median gap  : "
            f"{invalid_gap_values.median():,.0f} seconds"
        )

    # --------------------------------------------------------
    # CREATE EVALUATION DATASET
    # --------------------------------------------------------

    eval_df = df[
        df["Valid_Previous"]
    ].copy()

    if len(eval_df) == 0:
        raise ValueError(
            "\nNo valid 30-second temporal transitions found."
        )

    eval_df["Previous_Target"] = (
        eval_df["Previous_Target"]
        .astype(int)
    )

    eval_df["Target_Attack"] = (
        eval_df["Target_Attack"]
        .astype(int)
    )

    # Persistence prediction:
    # previous attack status == current prediction.

    eval_df["Prediction"] = (
        eval_df["Previous_Target"]
    )

    print("\nPersistence rule:")

    print("Previous target = 0 -> Predict 0")
    print("Previous target = 1 -> Predict 1")

    print(
        f"\nEvaluation samples : "
        f"{len(eval_df):,}"
    )

    # --------------------------------------------------------
    # OVERALL METRICS
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("2. OVERALL GAP-AWARE PERSISTENCE PERFORMANCE")
    print("=" * 70)

    metric_rows = []

    for split in ["TRAIN", "VALIDATION", "TEST"]:

        split_df = eval_df[
            eval_df["Split"] == split
        ]

        metrics = calculate_metrics(
            split_df["Target_Attack"].values,
            split_df["Prediction"].values
        )

        metrics["Split"] = split

        metric_rows.append(metrics)

    metrics_df = pd.DataFrame(metric_rows)

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
        metrics_df[display_columns]
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # CONFUSION MATRICES
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("3. CONFUSION MATRICES")
    print("=" * 70)

    for split in ["TRAIN", "VALIDATION", "TEST"]:

        split_df = eval_df[
            eval_df["Split"] == split
        ]

        cm = confusion_matrix(
            split_df["Target_Attack"],
            split_df["Prediction"],
            labels=[0, 1]
        )

        tn, fp, fn, tp = cm.ravel()

        print(f"\n{split}")

        print("                 Predicted")
        print("                 Benign  Attack")
        print(
            f"Actual Benign    {tn:6d}  {fp:6d}"
        )
        print(
            f"Actual Attack    {fn:6d}  {tp:6d}"
        )

    # --------------------------------------------------------
    # TRANSITION ANALYSIS
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("4. VALID TEMPORAL TRANSITIONS")
    print("=" * 70)

    transition_table = (
        eval_df
        .groupby(
            [
                "Previous_Target",
                "Target_Attack"
            ]
        )
        .size()
        .reset_index(name="Count")
    )

    transition_table["Transition"] = (
        transition_table["Previous_Target"]
        .astype(str)
        + " -> "
        + transition_table["Target_Attack"]
        .astype(str)
    )

    transition_table = transition_table[
        [
            "Transition",
            "Count"
        ]
    ]

    print(
        transition_table
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # PERSISTENCE RATE
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("5. PERSISTENCE BEHAVIOUR")
    print("=" * 70)

    correct_predictions = (
        eval_df["Target_Attack"]
        == eval_df["Prediction"]
    ).sum()

    incorrect_predictions = (
        eval_df["Target_Attack"]
        != eval_df["Prediction"]
    ).sum()

    print(
        f"\nCorrect persistence predictions : "
        f"{correct_predictions:,}"
    )

    print(
        f"Incorrect persistence predictions : "
        f"{incorrect_predictions:,}"
    )

    # Attack persistence.
    attack_to_attack = len(
        eval_df[
            (eval_df["Previous_Target"] == 1)
            & (eval_df["Target_Attack"] == 1)
        ]
    )

    attack_transitions = len(
        eval_df[
            eval_df["Previous_Target"] == 1
        ]
    )

    if attack_transitions > 0:

        attack_persistence_rate = (
            attack_to_attack
            / attack_transitions
        )

        print(
            f"\nAttack persistence rate : "
            f"{attack_persistence_rate:.4%}"
        )

    # --------------------------------------------------------
    # PERFORMANCE BY DAY
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("6. PERFORMANCE BY DAY")
    print("=" * 70)

    daily_rows = []

    for source_file, group in eval_df.groupby(
        "Source_File"
    ):

        metrics = calculate_metrics(
            group["Target_Attack"].values,
            group["Prediction"].values
        )

        metrics["Source_File"] = source_file
        metrics["Split"] = group["Split"].iloc[0]

        daily_rows.append(metrics)

    daily_df = pd.DataFrame(daily_rows)

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
        .sort_values(
            ["Split", "Source_File"]
        )
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # GAP ANALYSIS BY SOURCE
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("7. GAP ANALYSIS BY DAY")
    print("=" * 70)

    gap_rows_output = []

    for source_file, group in df.groupby(
        "Source_File"
    ):

        first_count = int(
            group["Previous_Target_Window"]
            .isna()
            .sum()
        )

        gap_count = int(
            (
                (~group["Valid_Previous"])
                & group["Previous_Target_Window"].notna()
            ).sum()
        )

        valid_count = int(
            group["Valid_Previous"].sum()
        )

        gap_rows_output.append({
            "Source_File": source_file,
            "Split": group["Split"].iloc[0],
            "First_Sample": first_count,
            "Gap_Reset_Samples": gap_count,
            "Valid_Transitions": valid_count,
        })

    gap_df = pd.DataFrame(
        gap_rows_output
    )

    print(
        gap_df.to_string(index=False)
    )

    # --------------------------------------------------------
    # FINAL TEST RESULT
    # --------------------------------------------------------

    test_metrics = metrics_df[
        metrics_df["Split"] == "TEST"
    ].iloc[0]

    print("\n" + "=" * 70)
    print("8. FINAL GAP-AWARE TEST BASELINE")
    print("=" * 70)

    print(
        f"\nTEST Samples   : "
        f"{int(test_metrics['Samples']):,}"
    )

    print(
        f"TEST Accuracy  : "
        f"{test_metrics['Accuracy']:.4f}"
    )

    print(
        f"TEST Precision : "
        f"{test_metrics['Precision']:.4f}"
    )

    print(
        f"TEST Recall    : "
        f"{test_metrics['Recall']:.4f}"
    )

    print(
        f"TEST F1        : "
        f"{test_metrics['F1']:.4f}"
    )

    print(
        f"TEST PR-AUC    : "
        f"{test_metrics['PR_AUC']:.4f}"
    )

    print(
        f"TEST FPR       : "
        f"{test_metrics['False_Positive_Rate']:.4f}"
    )

    # --------------------------------------------------------
    # SAVE OUTPUTS
    # --------------------------------------------------------

    metrics_path = (
        OUTPUT_DIR
        / "persistence_metrics_v2.csv"
    )

    daily_path = (
        OUTPUT_DIR
        / "persistence_metrics_by_day_v2.csv"
    )

    predictions_path = (
        OUTPUT_DIR
        / "persistence_predictions_v2.csv"
    )

    transitions_path = (
        OUTPUT_DIR
        / "persistence_transitions_v2.csv"
    )

    gaps_path = (
        OUTPUT_DIR
        / "persistence_gap_audit_v2.csv"
    )

    metrics_df.to_csv(
        metrics_path,
        index=False
    )

    daily_df.to_csv(
        daily_path,
        index=False
    )

    eval_df[
        [
            "Source_File",
            "Split",
            "Target_Window",
            "Previous_Target_Window",
            "Gap_Seconds",
            "Previous_Target",
            "Target_Attack",
            "Prediction",
        ]
    ].to_csv(
        predictions_path,
        index=False
    )

    transition_table.to_csv(
        transitions_path,
        index=False
    )

    gap_df.to_csv(
        gaps_path,
        index=False
    )

    print("\n" + "=" * 70)
    print("FILES SAVED")
    print("=" * 70)

    print(metrics_path)
    print(daily_path)
    print(predictions_path)
    print(transitions_path)
    print(gaps_path)

    print("\n" + "=" * 70)
    print("PERSISTENCE BASELINE V2 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()