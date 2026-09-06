"""
SIH26153 - Logistic Regression Baseline

Purpose
-------
Build a non-temporal Logistic Regression baseline for the
30-second-ahead attack forecasting task.

Design
------
- Uses ONLY the current network state (t-0) features.
- Does NOT use the 10-window history.
- Uses the same chronological TRAIN / VALIDATION / TEST split
  as the persistence baseline.
- Imputer and scaler are fitted ONLY on TRAIN.
- No random split.
- No threshold tuning.
- Also evaluates attack-onset prediction (0 -> 1).

Input
-----
data/processed/forecasting_samples/all_forecasting_samples.csv

Output
------
results/logistic_regression_baseline/
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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
    / "logistic_regression_baseline"
)

RANDOM_STATE = 42
THRESHOLD = 0.50


# ============================================================
# SPLIT DEFINITION
# ============================================================

# These are the actual CIC-IDS2018 dates used in our
# chronological experiment.

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
# ASSIGN CHRONOLOGICAL SPLIT
# ============================================================

def assign_split(source_file):
    """
    Assign TRAIN / VALIDATION / TEST using the actual
    date contained in the CIC-IDS2018 source filename.

    Example:
    Friday-02-03-2018_TrafficForML_CICFlowMeter_30s_state.csv

    -> TEST
    """

    source = str(source_file).replace("\\", "/")

    # -------------------------
    # TRAIN
    # -------------------------

    for date in TRAIN_DATES:
        if date in source:
            return "TRAIN"

    # -------------------------
    # VALIDATION
    # -------------------------

    for date in VALIDATION_DATES:
        if date in source:
            return "VALIDATION"

    # -------------------------
    # TEST
    # -------------------------

    for date in TEST_DATES:
        if date in source:
            return "TEST"

    # -------------------------
    # UNKNOWN FILE
    # -------------------------

    raise ValueError(
        f"\nCould not assign split for source file:\n"
        f"{source_file}\n\n"
        "Expected one of these CIC-IDS2018 dates:\n"
        f"TRAIN       : {sorted(TRAIN_DATES)}\n"
        f"VALIDATION  : {sorted(VALIDATION_DATES)}\n"
        f"TEST        : {sorted(TEST_DATES)}"
    )


# ============================================================
# CLASSIFICATION METRICS
# ============================================================

def calculate_metrics(y_true, y_pred, y_prob):

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

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

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    fpr = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0.0
    )

    try:
        pr_auc = average_precision_score(
            y_true,
            y_prob,
        )
    except ValueError:
        pr_auc = np.nan

    try:
        roc_auc = roc_auc_score(
            y_true,
            y_prob,
        )
    except ValueError:
        roc_auc = np.nan

    return {
        "samples": len(y_true),
        "actual_attack": int(np.sum(y_true)),
        "actual_benign": int(len(y_true) - np.sum(y_true)),
        "predicted_attack": int(np.sum(y_pred)),
        "predicted_benign": int(len(y_pred) - np.sum(y_pred)),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "fpr": fpr,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


# ============================================================
# PRINT METRICS
# ============================================================

def print_metrics(name, metrics):

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    print(f"Samples            : {metrics['samples']}")
    print(f"Actual attack      : {metrics['actual_attack']}")
    print(f"Actual benign      : {metrics['actual_benign']}")
    print(f"Predicted attack   : {metrics['predicted_attack']}")
    print(f"Predicted benign   : {metrics['predicted_benign']}")

    print()
    print(f"Accuracy           : {metrics['accuracy']:.6f}")
    print(f"Precision          : {metrics['precision']:.6f}")
    print(f"Recall             : {metrics['recall']:.6f}")
    print(f"F1                 : {metrics['f1']:.6f}")
    print(f"PR-AUC             : {metrics['pr_auc']:.6f}")
    print(f"ROC-AUC            : {metrics['roc_auc']:.6f}")
    print(f"FPR                : {metrics['fpr']:.6f}")

    print()
    print("Confusion Matrix")
    print(f"TN = {metrics['tn']}")
    print(f"FP = {metrics['fp']}")
    print(f"FN = {metrics['fn']}")
    print(f"TP = {metrics['tp']}")


# ============================================================
# ATTACK ONSET METRICS
# ============================================================

def calculate_onset_metrics(predictions_df):

    df = predictions_df.copy()

    # Sort within each original capture file.
    df = df.sort_values(
        ["Source_File", "Target_Window"]
    ).reset_index(drop=True)

    df["Target_Window"] = pd.to_datetime(
        df["Target_Window"],
        errors="coerce",
    )

    # Previous actual target
    df["Previous_Actual"] = (
        df.groupby("Source_File")["Actual"]
        .shift(1)
    )

    # Previous prediction
    df["Previous_Predicted"] = (
        df.groupby("Source_File")["Predicted"]
        .shift(1)
    )

    # Previous timestamp
    df["Previous_Window"] = (
        df.groupby("Source_File")["Target_Window"]
        .shift(1)
    )

    # Time gap
    df["Time_Difference"] = (
        df["Target_Window"]
        - df["Previous_Window"]
    ).dt.total_seconds()

    # Only consider exact 30-second transitions.
    df["Continuous_30s"] = (
        df["Time_Difference"] == 30
    )

    # --------------------------------------------------------
    # Actual attack onset:
    #
    # previous = benign
    # current  = attack
    # --------------------------------------------------------

    df["Actual_Onset"] = (
        (df["Previous_Actual"] == 0)
        & (df["Actual"] == 1)
        & df["Continuous_30s"]
    )

    # --------------------------------------------------------
    # Predicted attack onset:
    #
    # previous prediction = benign
    # current prediction  = attack
    # --------------------------------------------------------

    df["Predicted_Onset"] = (
        (df["Previous_Predicted"] == 0)
        & (df["Predicted"] == 1)
        & df["Continuous_30s"]
    )

    onset_true = (
        df["Actual_Onset"]
        .astype(int)
        .values
    )

    onset_pred = (
        df["Predicted_Onset"]
        .astype(int)
        .values
    )

    # For onset evaluation, use the model probability
    # of the current window.
    onset_prob = df["Probability"].values

    return calculate_metrics(
        onset_true,
        onset_pred,
        onset_prob,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("SIH26153 - LOGISTIC REGRESSION BASELINE")
    print("=" * 70)

    # ========================================================
    # CHECK INPUT
    # ========================================================

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"\nInput file not found:\n{INPUT_FILE}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # LOAD DATA
    # ========================================================

    print()
    print("Loading:")
    print(INPUT_FILE)

    df = pd.read_csv(
        INPUT_FILE
    )

    print()
    print(f"Loaded samples : {len(df):,}")
    print(f"Total columns  : {len(df.columns)}")

    # ========================================================
    # REQUIRED COLUMNS
    # ========================================================

    required_columns = {
        "Source_File",
        "Target_Window",
        "Target_Attack",
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:

        raise ValueError(
            "\nMissing required columns:\n"
            + "\n".join(
                sorted(missing_columns)
            )
        )

    # ========================================================
    # FIND t-0 FEATURES
    # ========================================================

    # Correct format:
    #
    # t-0_Flow_Count
    # t-0_Total_Fwd_Pkts
    # t-0_SYN_Count
    #
    # Therefore startswith("t-0_") is used.

    current_state_columns = [
        col
        for col in df.columns
        if str(col).startswith("t-0_")
    ]

    print()
    print(
        "Current-state columns found : "
        f"{len(current_state_columns)}"
    )

    print()
    print("Current-state features:")

    for col in current_state_columns:
        print(f"  {col}")

    if not current_state_columns:

        raise ValueError(
            "\nNo t-0 network-state features found."
            "\nExpected columns such as:"
            "\n  t-0_Flow_Count"
            "\n  t-0_Total_Fwd_Pkts"
            "\n  t-0_Total_Bwd_Pkts"
        )

    # Our current Network State V2 contains 24 features.
    if len(current_state_columns) != 24:

        print()
        print(
            "WARNING:"
        )

        print(
            "Expected 24 current-state features, "
            f"but found {len(current_state_columns)}."
        )

        print(
            "The script will continue, but inspect "
            "the feature list before trusting results."
        )

    # ========================================================
    # ASSIGN CHRONOLOGICAL SPLITS
    # ========================================================

    print()
    print(
        "Assigning chronological splits..."
    )

    df["Split"] = (
        df["Source_File"]
        .apply(assign_split)
    )

    print()
    print("Split counts:")

    print(
        df["Split"]
        .value_counts()
        .to_string()
    )

    # ========================================================
    # SORT
    # ========================================================

    df["Target_Window"] = pd.to_datetime(
        df["Target_Window"],
        errors="coerce",
    )

    invalid_windows = (
        df["Target_Window"].isna()
    )

    if invalid_windows.any():

        raise ValueError(
            f"\nFound "
            f"{invalid_windows.sum():,} "
            "invalid Target_Window values."
        )

    df = df.sort_values(
        [
            "Source_File",
            "Target_Window",
        ]
    ).reset_index(
        drop=True
    )

    # ========================================================
    # TARGET
    # ========================================================

    y = pd.to_numeric(
        df["Target_Attack"],
        errors="coerce",
    )

    if y.isna().any():

        raise ValueError(
            "\nTarget_Attack contains invalid values."
        )

    y = y.astype(int)

    unique_targets = sorted(
        y.unique().tolist()
    )

    print()
    print(
        f"Target values found : {unique_targets}"
    )

    if not set(unique_targets).issubset(
        {0, 1}
    ):

        raise ValueError(
            "\nTarget_Attack must contain "
            "only 0 and 1."
        )

    # ========================================================
    # FEATURES
    # ========================================================

    X = df[
        current_state_columns
    ].copy()

    # Convert every feature to numeric.
    for col in current_state_columns:

        X[col] = pd.to_numeric(
            X[col],
            errors="coerce",
        )

    # Replace +/- infinity with NaN.
    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # ========================================================
    # SPLIT MASKS
    # ========================================================

    train_mask = (
        df["Split"] == "TRAIN"
    )

    validation_mask = (
        df["Split"] == "VALIDATION"
    )

    test_mask = (
        df["Split"] == "TEST"
    )

    X_train = X.loc[
        train_mask
    ]

    y_train = y.loc[
        train_mask
    ]

    X_validation = X.loc[
        validation_mask
    ]

    y_validation = y.loc[
        validation_mask
    ]

    X_test = X.loc[
        test_mask
    ]

    y_test = y.loc[
        test_mask
    ]

    print()
    print("=" * 70)
    print("DATA SPLIT")
    print("=" * 70)

    print(
        f"TRAIN       : {len(X_train):,}"
    )

    print(
        f"VALIDATION  : {len(X_validation):,}"
    )

    print(
        f"TEST        : {len(X_test):,}"
    )

    print()

    print(
        f"TRAIN attack rate      : "
        f"{y_train.mean():.6f}"
    )

    print(
        f"VALIDATION attack rate : "
        f"{y_validation.mean():.6f}"
    )

    print(
        f"TEST attack rate       : "
        f"{y_test.mean():.6f}"
    )

    # ========================================================
    # LOGISTIC REGRESSION PIPELINE
    # ========================================================

    model = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),

            (
                "scaler",
                StandardScaler(),
            ),

            (
                "logistic_regression",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    print()
    print("=" * 70)
    print("TRAINING LOGISTIC REGRESSION")
    print("=" * 70)

    model.fit(
        X_train,
        y_train,
    )

    print(
        "Training completed."
    )

    # ========================================================
    # PREDICTION FUNCTION
    # ========================================================

    def predict_split(X_split):

        probabilities = (
            model
            .predict_proba(X_split)[:, 1]
        )

        predictions = (
            probabilities >= THRESHOLD
        ).astype(int)

        return (
            predictions,
            probabilities,
        )

    # ========================================================
    # PREDICTIONS
    # ========================================================

    train_pred, train_prob = (
        predict_split(X_train)
    )

    validation_pred, validation_prob = (
        predict_split(X_validation)
    )

    test_pred, test_prob = (
        predict_split(X_test)
    )

    # ========================================================
    # METRICS
    # ========================================================

    train_metrics = calculate_metrics(
        y_train.values,
        train_pred,
        train_prob,
    )

    validation_metrics = calculate_metrics(
        y_validation.values,
        validation_pred,
        validation_prob,
    )

    test_metrics = calculate_metrics(
        y_test.values,
        test_pred,
        test_prob,
    )

    print_metrics(
        "TRAIN",
        train_metrics,
    )

    print_metrics(
        "VALIDATION",
        validation_metrics,
    )

    print_metrics(
        "TEST",
        test_metrics,
    )

    # ========================================================
    # BUILD PREDICTION DATAFRAME
    # ========================================================

    prediction_parts = []

    split_predictions = [
        (
            train_mask,
            train_pred,
            train_prob,
        ),
        (
            validation_mask,
            validation_pred,
            validation_prob,
        ),
        (
            test_mask,
            test_pred,
            test_prob,
        ),
    ]

    for (
        mask,
        pred,
        prob,
    ) in split_predictions:

        part = df.loc[
            mask,
            [
                "Source_File",
                "Target_Window",
                "Target_Attack",
                "Split",
            ],
        ].copy()

        part["Actual"] = (
            part["Target_Attack"]
            .astype(int)
        )

        part["Predicted"] = pred

        part["Probability"] = prob

        prediction_parts.append(
            part
        )

    predictions_df = pd.concat(
        prediction_parts,
        ignore_index=True,
    )

    predictions_df = (
        predictions_df
        .sort_values(
            [
                "Source_File",
                "Target_Window",
            ]
        )
        .reset_index(drop=True)
    )

    # ========================================================
    # PREVIOUS PREDICTION
    # ========================================================

    predictions_df[
        "Previous_Predicted"
    ] = (
        predictions_df
        .groupby("Source_File")["Predicted"]
        .shift(1)
    )

    # ========================================================
    # SAVE PREDICTIONS
    # ========================================================

    prediction_file = (
        OUTPUT_DIR
        / "logistic_predictions.csv"
    )

    predictions_df.to_csv(
        prediction_file,
        index=False,
    )

    # ========================================================
    # SAVE MAIN METRICS
    # ========================================================

    metrics_rows = []

    for (
        split_name,
        metrics,
    ) in [
        (
            "TRAIN",
            train_metrics,
        ),
        (
            "VALIDATION",
            validation_metrics,
        ),
        (
            "TEST",
            test_metrics,
        ),
    ]:

        metrics_rows.append(
            {
                "Split": split_name,
                **metrics,
            }
        )

    metrics_df = pd.DataFrame(
        metrics_rows
    )

    metrics_file = (
        OUTPUT_DIR
        / "logistic_metrics.csv"
    )

    metrics_df.to_csv(
        metrics_file,
        index=False,
    )

    # ========================================================
    # ATTACK ONSET ANALYSIS
    # ========================================================

    print()
    print("=" * 70)
    print("ATTACK ONSET ANALYSIS")
    print("=" * 70)

    onset_results = []

    for split_name in [
        "TRAIN",
        "VALIDATION",
        "TEST",
    ]:

        split_predictions = (
            predictions_df[
                predictions_df["Split"]
                == split_name
            ]
            .copy()
        )

        onset_metrics = (
            calculate_onset_metrics(
                split_predictions
            )
        )

        onset_results.append(
            {
                "Split": split_name,
                **onset_metrics,
            }
        )

        print_metrics(
            f"{split_name} - ATTACK ONSET (0 -> 1)",
            onset_metrics,
        )

    onset_df = pd.DataFrame(
        onset_results
    )

    onset_file = (
        OUTPUT_DIR
        / "logistic_onset_metrics.csv"
    )

    onset_df.to_csv(
        onset_file,
        index=False,
    )

    # ========================================================
    # FEATURE COEFFICIENTS
    # ========================================================

    logistic_model = (
        model
        .named_steps[
            "logistic_regression"
        ]
    )

    coefficients = (
        logistic_model.coef_[0]
    )

    coefficient_df = pd.DataFrame(
        {
            "Feature": current_state_columns,
            "Coefficient": coefficients,
            "Absolute_Coefficient": np.abs(
                coefficients
            ),
        }
    )

    coefficient_df = (
        coefficient_df
        .sort_values(
            "Absolute_Coefficient",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    coefficient_file = (
        OUTPUT_DIR
        / "logistic_feature_coefficients.csv"
    )

    coefficient_df.to_csv(
        coefficient_file,
        index=False,
    )

    # ========================================================
    # TOP FEATURES
    # ========================================================

    print()
    print("=" * 70)
    print("TOP LOGISTIC REGRESSION FEATURES")
    print("=" * 70)

    print(
        coefficient_df
        .head(15)
        .to_string(index=False)
    )

    # ========================================================
    # SAVE CONFIGURATION
    # ========================================================

    config_file = (
        OUTPUT_DIR
        / "baseline_config.txt"
    )

    with open(
        config_file,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "SIH26153 Logistic Regression Baseline\n"
        )

        f.write(
            "====================================\n\n"
        )

        f.write(
            f"Input file: {INPUT_FILE}\n"
        )

        f.write(
            f"Samples: {len(df)}\n"
        )

        f.write(
            f"Current-state features: "
            f"{len(current_state_columns)}\n"
        )

        f.write(
            "History used: NO\n"
        )

        f.write(
            "Current state only: YES\n"
        )

        f.write(
            "Random split: NO\n"
        )

        f.write(
            "Chronological split: YES\n"
        )

        f.write(
            "Imputer fitted on TRAIN only: YES\n"
        )

        f.write(
            "Scaler fitted on TRAIN only: YES\n"
        )

        f.write(
            "Threshold: 0.50\n"
        )

        f.write(
            "Class weight: balanced\n"
        )

        f.write(
            "Random state: 42\n"
        )

        f.write(
            "\nTRAIN dates:\n"
        )

        for date in sorted(TRAIN_DATES):
            f.write(
                f"  {date}\n"
            )

        f.write(
            "\nVALIDATION dates:\n"
        )

        for date in sorted(
            VALIDATION_DATES
        ):
            f.write(
                f"  {date}\n"
            )

        f.write(
            "\nTEST dates:\n"
        )

        for date in sorted(TEST_DATES):
            f.write(
                f"  {date}\n"
            )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print()
    print("=" * 70)
    print("FILES SAVED")
    print("=" * 70)

    print(
        f"Metrics       : {metrics_file}"
    )

    print(
        f"Predictions   : {prediction_file}"
    )

    print(
        f"Onset metrics : {onset_file}"
    )

    print(
        f"Coefficients  : {coefficient_file}"
    )

    print(
        f"Config        : {config_file}"
    )

    print()
    print("=" * 70)
    print(
        "LOGISTIC REGRESSION BASELINE COMPLETE"
    )
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()