"""
SIH26153 - Early Warning LSTM Baseline

Goal:
    Predict whether a NEW attack episode will START within:
        +30s, +60s, +90s, +120s, +150s

Dataset:
    data/processed/early_warning/early_warning_samples.csv
    data/processed/forecasting_samples/all_forecasting_samples.csv

Input:
    10 historical 30-second windows
    24 network-state features per window

Shape:
    X = (samples, 10, 24)

Important:
    - Chronological train/validation/test split
    - No random split
    - Scaler fitted on TRAIN only
    - Class weights fitted on TRAIN only
    - Validation used for threshold selection
    - TEST used only for final evaluation
"""

from pathlib import Path
import re
import json
import random

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
)

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

FORECAST_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "forecasting_samples"
    / "all_forecasting_samples.csv"
)

EARLY_WARNING_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "early_warning"
    / "early_warning_samples.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "early_warning"
    / "lstm_results"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

TIME_STEPS = list(range(9, -1, -1))
N_TIME_STEPS = 10
N_FEATURES = 24

HORIZONS = [30, 60, 90, 120, 150]

EPOCHS = 50
BATCH_SIZE = 64

LEARNING_RATE = 1e-3


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass


# ============================================================
# HELPERS
# ============================================================

def normalize_source_name(value):
    """
    Normalize Source_File names so that they match between
    forecasting and early-warning datasets.
    """
    value = str(value).strip()

    suffixes = [
        "_TrafficForML_CICFlowMeter_30s_state.csv",
        "_TrafficForML_CICFlowMeter_attack_type_mapping.csv",
        "_TrafficForML_CICFlowMeter.csv",
    ]

    for suffix in suffixes:
        if value.endswith(suffix):
            value = value[: -len(suffix)]

    return value


def detect_historical_features(columns):
    """
    Detect columns such as:
        t-9_Flow_Count
        t-8_Flow_Count
        ...
        t-0_Unique_Protocols

    Returns features grouped by timestep.
    """

    pattern = re.compile(r"^t-(\d+)_(.+)$")

    grouped = {}

    for col in columns:
        match = pattern.match(col)

        if not match:
            continue

        timestep = int(match.group(1))
        feature_name = match.group(2)

        grouped.setdefault(timestep, []).append(
            (feature_name, col)
        )

    expected_timesteps = set(TIME_STEPS)

    if set(grouped.keys()) != expected_timesteps:
        raise ValueError(
            f"Unexpected timesteps.\n"
            f"Expected: {sorted(expected_timesteps)}\n"
            f"Found: {sorted(grouped.keys())}"
        )

    for timestep in TIME_STEPS:
        grouped[timestep].sort(key=lambda x: x[0])

        if len(grouped[timestep]) != N_FEATURES:
            raise ValueError(
                f"t-{timestep} contains "
                f"{len(grouped[timestep])} features; "
                f"expected {N_FEATURES}."
            )

    # Make sure feature names are identical at every timestep.
    reference_features = [
        x[0] for x in grouped[TIME_STEPS[0]]
    ]

    for timestep in TIME_STEPS:
        current_features = [
            x[0] for x in grouped[timestep]
        ]

        if current_features != reference_features:
            raise ValueError(
                f"Feature mismatch at t-{timestep}."
            )

    return grouped, reference_features


def build_feature_matrix(forecast_df, grouped_features):
    """
    Convert flattened forecasting columns into:

        (samples, 10, 24)
    """

    matrices = []

    for timestep in TIME_STEPS:

        columns = [
            column
            for _, column in grouped_features[timestep]
        ]

        matrices.append(
            forecast_df[columns].to_numpy(dtype=np.float32)
        )

    X = np.stack(matrices, axis=1)

    return X


def make_split_mask(df):
    """
    Same chronological split used throughout the project.

    TRAIN:
        Feb14, Feb15, Feb16, Feb20, Feb21, Feb22, Feb23

    VALID:
        Feb28

    TEST:
        Mar1, Mar2
    """

    source = df["Source_Key"]

    train_days = {
        "Wednesday-14-02-2018",
        "Thursday-15-02-2018",
        "Friday-16-02-2018",
        "Thuesday-20-02-2018",
        "Wednesday-21-02-2018",
        "Thursday-22-02-2018",
        "Friday-23-02-2018",
    }

    valid_days = {
        "Wednesday-28-02-2018",
    }

    test_days = {
        "Thursday-01-03-2018",
        "Friday-02-03-2018",
    }

    train_mask = source.isin(train_days)
    valid_mask = source.isin(valid_days)
    test_mask = source.isin(test_days)

    unknown = ~(
        train_mask
        | valid_mask
        | test_mask
    )

    if unknown.any():
        unknown_sources = sorted(
            source.loc[unknown].unique()
        )

        raise ValueError(
            "Unknown Source_Key values found:\n"
            + "\n".join(unknown_sources)
        )

    return train_mask, valid_mask, test_mask


def compute_class_weights(y):
    """
    Balanced binary class weights.

    Returns:
        {0: weight_for_negative,
         1: weight_for_positive}
    """

    y = np.asarray(y).astype(int)

    negative = np.sum(y == 0)
    positive = np.sum(y == 1)

    if positive == 0:
        raise ValueError(
            "Training split contains no positive samples."
        )

    if negative == 0:
        raise ValueError(
            "Training split contains no negative samples."
        )

    total = negative + positive

    weight_0 = total / (2.0 * negative)
    weight_1 = total / (2.0 * positive)

    return {
        0: float(weight_0),
        1: float(weight_1),
    }


def validation_threshold_search(
    y_true,
    probabilities,
):
    """
    Select threshold using VALIDATION F1 only.

    Threshold range:
        0.01 -> 0.99

    TEST is never used here.
    """

    best_threshold = 0.50
    best_f1 = -1.0

    for threshold in np.arange(
        0.01,
        1.00,
        0.01
    ):

        predictions = (
            probabilities >= threshold
        ).astype(int)

        score = f1_score(
            y_true,
            predictions,
            zero_division=0,
        )

        if score > best_f1:
            best_f1 = score
            best_threshold = float(
                round(threshold, 2)
            )

    return best_threshold, best_f1


def evaluate_binary(
    y_true,
    probabilities,
    threshold,
):
    """
    Calculate final classification metrics.
    """

    predictions = (
        probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0,
    )

    pr_auc = average_precision_score(
        y_true,
        probabilities,
    )

    try:
        roc_auc = roc_auc_score(
            y_true,
            probabilities,
        )
    except ValueError:
        roc_auc = np.nan

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    fpr = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0.0
    )

    return {
        "threshold": threshold,
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
        "positive_count": int(np.sum(y_true)),
        "sample_count": int(len(y_true)),
    }


def print_metrics(prefix, metrics):
    print(
        f"{prefix} "
        f"P={metrics['precision']:.4f} "
        f"R={metrics['recall']:.4f} "
        f"F1={metrics['f1']:.4f} "
        f"PR-AUC={metrics['pr_auc']:.4f} "
        f"ROC-AUC={metrics['roc_auc']:.4f} "
        f"FPR={metrics['fpr']:.4f}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("SIH26153 - EARLY WARNING LSTM")
    print("=" * 80)

    print("\nBase directory:")
    print(BASE_DIR)

    print("\nForecasting file:")
    print(FORECAST_FILE)

    print("\nEarly-warning file:")
    print(EARLY_WARNING_FILE)

    # --------------------------------------------------------
    # 1. LOAD DATA
    # --------------------------------------------------------

    print("\n[1/9] Loading datasets...")

    if not FORECAST_FILE.exists():
        raise FileNotFoundError(
            f"Forecasting dataset not found:\n{FORECAST_FILE}"
        )

    if not EARLY_WARNING_FILE.exists():
        raise FileNotFoundError(
            f"Early-warning dataset not found:\n{EARLY_WARNING_FILE}"
        )

    forecast_df = pd.read_csv(
        FORECAST_FILE
    )

    early_df = pd.read_csv(
        EARLY_WARNING_FILE
    )

    print(
        f"Forecasting rows: "
        f"{len(forecast_df):,}"
    )

    print(
        f"Early-warning rows: "
        f"{len(early_df):,}"
    )

    # --------------------------------------------------------
    # 2. VALIDATE EARLY-WARNING DATA
    # --------------------------------------------------------

    print("\n[2/9] Validating early-warning data...")

    required_early_columns = {
        "Source_File",
        "Forecast_Target_Window",
        "Starts_Within_30s",
        "Starts_Within_60s",
        "Starts_Within_90s",
        "Starts_Within_120s",
        "Starts_Within_150s",
    }

    missing = (
        required_early_columns
        - set(early_df.columns)
    )

    if missing:
        raise ValueError(
            "Missing early-warning columns:\n"
            + "\n".join(sorted(missing))
        )

    early_df["Source_Key"] = (
        early_df["Source_File"]
        .map(normalize_source_name)
    )

    early_df["Time_Key"] = pd.to_datetime(
        early_df["Forecast_Target_Window"],
        errors="coerce",
    )

    if early_df["Time_Key"].isna().any():
        raise ValueError(
            "Invalid timestamps found in early-warning dataset."
        )

    # Every early-warning sample must be currently benign.
    if "Current_Attack" in early_df.columns:

        current_attack = pd.to_numeric(
            early_df["Current_Attack"],
            errors="coerce",
        )

        if current_attack.isna().any():
            raise ValueError(
                "Current_Attack contains invalid/missing values."
            )

        if (current_attack != 0).any():
            raise ValueError(
                "Early-warning dataset contains "
                "currently attacking samples."
            )

    # Binary validation.
    for horizon in HORIZONS:

        column = (
            f"Starts_Within_{horizon}s"
        )

        if early_df[column].isna().any():
            raise ValueError(
                f"{column} contains missing (NaN) values. "
                "All early-warning labels must be present."
            )

        values = set(
            early_df[column]
            .astype(int)
            .unique()
        )

        if not values.issubset({0, 1}):
            raise ValueError(
                f"{column} contains non-binary values: "
                f"{values}"
            )

    print("Early-warning validation passed.")

    # --------------------------------------------------------
    # 3. PREPARE FORECASTING FEATURES
    # --------------------------------------------------------

    print("\n[3/9] Detecting historical features...")

    grouped_features, feature_names = (
        detect_historical_features(
            forecast_df.columns
        )
    )

    total_features = sum(
        len(grouped_features[t])
        for t in TIME_STEPS
    )

    print(
        f"Time steps: {TIME_STEPS}"
    )

    print(
        f"Features per timestep: "
        f"{N_FEATURES}"
    )

    print(
        f"Total historical features: "
        f"{total_features}"
    )

    if total_features != 240:
        raise ValueError(
            f"Expected 240 historical features, "
            f"found {total_features}."
        )

    # Normalize source names.
    forecast_df["Source_Key"] = (
        forecast_df["Source_File"]
        .map(normalize_source_name)
    )

    if "Target_Window" not in forecast_df.columns:
        raise ValueError(
            "Forecasting dataset does not contain "
            "'Target_Window'."
        )

    forecast_df["Time_Key"] = pd.to_datetime(
        forecast_df["Target_Window"],
        errors="coerce",
    )

    if forecast_df["Time_Key"].isna().any():
        raise ValueError(
            "Invalid Target_Window timestamps found."
        )

    # --------------------------------------------------------
    # 4. JOIN EARLY-WARNING LABELS
    # --------------------------------------------------------

    print("\n[4/9] Joining early-warning labels...")

    forecast_keys = [
        "Source_Key",
        "Time_Key",
    ]

    early_keys = [
        "Source_Key",
        "Time_Key",
    ]

    if forecast_df.duplicated(
        forecast_keys
    ).any():

        duplicate_count = forecast_df.duplicated(
            forecast_keys
        ).sum()

        raise ValueError(
            f"Forecasting dataset contains "
            f"{duplicate_count} duplicate join keys."
        )

    if early_df.duplicated(
        early_keys
    ).any():

        duplicate_count = early_df.duplicated(
            early_keys
        ).sum()

        raise ValueError(
            f"Early-warning dataset contains "
            f"{duplicate_count} duplicate join keys."
        )

    label_columns = [
        "Source_Key",
        "Time_Key",
    ] + [
        f"Starts_Within_{horizon}s"
        for horizon in HORIZONS
    ]

    labels = early_df[label_columns].copy()

    merged = forecast_df.merge(
        labels,
        on=[
            "Source_Key",
            "Time_Key",
        ],
        how="inner",
        validate="one_to_one",
    )

    print(
        f"Merged rows: "
        f"{len(merged):,}"
    )

    if len(merged) != len(early_df):
        raise ValueError(
            "Not all early-warning rows matched "
            "forecasting samples."
        )

    # --------------------------------------------------------
    # 5. BUILD SEQUENCE
    # --------------------------------------------------------

    print("\n[5/9] Building LSTM sequence...")

    X = build_feature_matrix(
        merged,
        grouped_features,
    )

    print(
        f"X shape: {X.shape}"
    )

    if X.shape[1] != N_TIME_STEPS:
        raise ValueError(
            f"Expected {N_TIME_STEPS} timesteps, "
            f"got {X.shape[1]}"
        )

    if X.shape[2] != N_FEATURES:
        raise ValueError(
            f"Expected {N_FEATURES} features, "
            f"got {X.shape[2]}"
        )

    # --------------------------------------------------------
    # 6. CHRONOLOGICAL SPLIT
    # --------------------------------------------------------

    print("\n[6/9] Creating chronological split...")

    train_mask, valid_mask, test_mask = (
        make_split_mask(merged)
    )

    print(
        f"TRAIN: {train_mask.sum():,}"
    )

    print(
        f"VALID: {valid_mask.sum():,}"
    )

    print(
        f"TEST : {test_mask.sum():,}"
    )

    if (
        train_mask.sum()
        + valid_mask.sum()
        + test_mask.sum()
        != len(merged)
    ):
        raise ValueError(
            "Train/validation/test masks do not "
            "cover the dataset exactly."
        )

    # --------------------------------------------------------
    # 7. TRAIN-ONLY IMPUTATION + SCALING
    # --------------------------------------------------------

    print(
        "\n[7/9] Fitting imputer/scaler on TRAIN only..."
    )

    # Flatten only for preprocessing.
    X_flat = X.reshape(
        X.shape[0],
        -1,
    )

    train_X_flat = X_flat[train_mask]

    imputer = SimpleImputer(
        strategy="median"
    )

    scaler = StandardScaler()

    train_X_flat = imputer.fit_transform(
        train_X_flat
    )

    scaler.fit(
        train_X_flat
    )

    X_flat_imputed = imputer.transform(
        X_flat
    )

    X_flat_scaled = scaler.transform(
        X_flat_imputed
    )

    X_scaled = X_flat_scaled.reshape(
        X.shape
    ).astype(np.float32)

    # Make sure everything is finite.
    if not np.isfinite(X_scaled).all():
        raise ValueError(
            "Non-finite values remain after "
            "imputation/scaling."
        )

    X_train = X_scaled[train_mask]
    X_valid = X_scaled[valid_mask]
    X_test = X_scaled[test_mask]

    print(
        f"X_train: {X_train.shape}"
    )

    print(
        f"X_valid: {X_valid.shape}"
    )

    print(
        f"X_test : {X_test.shape}"
    )

    # --------------------------------------------------------
    # 8. TRAIN ONE MODEL PER HORIZON
    # --------------------------------------------------------

    print("\n[8/9] Training LSTM models...")
    print("=" * 80)

    results = []

    for horizon in HORIZONS:

        print("\n")
        print("-" * 80)
        print(
            f"EARLY-WARNING HORIZON: +{horizon}s"
        )
        print("-" * 80)

        label_column = (
            f"Starts_Within_{horizon}s"
        )

        y_numeric = pd.to_numeric(
            merged[label_column],
            errors="coerce",
        )

        if y_numeric.isna().any():
            raise ValueError(
                f"{label_column} contains invalid/missing values "
                "that could not be converted to numeric labels."
            )

        y = y_numeric.astype(int).to_numpy()

        y_train = y[train_mask]
        y_valid = y[valid_mask]
        y_test = y[test_mask]

        print(
            f"TRAIN positives: "
            f"{y_train.sum():,} / {len(y_train):,} "
            f"({100 * y_train.mean():.3f}%)"
        )

        print(
            f"VALID positives: "
            f"{y_valid.sum():,} / {len(y_valid):,} "
            f"({100 * y_valid.mean():.3f}%)"
        )

        print(
            f"TEST positives:  "
            f"{y_test.sum():,} / {len(y_test):,} "
            f"({100 * y_test.mean():.3f}%)"
        )

        # --------------------------------------------
        # Class weights
        # --------------------------------------------

        class_weights = compute_class_weights(
            y_train
        )

        print(
            f"Class weights: "
            f"{class_weights}"
        )

        # --------------------------------------------
        # Build model
        # --------------------------------------------

        model = models.Sequential(
            [
                layers.Input(
                    shape=(
                        N_TIME_STEPS,
                        N_FEATURES,
                    )
                ),

                layers.LSTM(
                    64,
                    return_sequences=False,
                ),

                layers.Dropout(
                    0.25
                ),

                layers.Dense(
                    32,
                    activation="relu",
                ),

                layers.Dropout(
                    0.15
                ),

                layers.Dense(
                    1,
                    activation="sigmoid",
                ),
            ]
        )

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE,
            clipnorm=1.0,
        )

        model.compile(
            optimizer=optimizer,
            loss="binary_crossentropy",
            metrics=[
                tf.keras.metrics.Precision(
                    name="precision"
                ),
                tf.keras.metrics.Recall(
                    name="recall"
                ),
                tf.keras.metrics.AUC(
                    name="roc_auc"
                ),
                tf.keras.metrics.AUC(
                    name="pr_auc",
                    curve="PR",
                ),
            ],
        )

        model_path = (
            OUTPUT_DIR
            / f"early_warning_lstm_{horizon}s.keras"
        )

        callback_list = [
            callbacks.EarlyStopping(
                monitor="val_loss",
                patience=8,
                restore_best_weights=True,
                verbose=1,
            ),

            callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=4,
                min_lr=1e-6,
                verbose=1,
            ),

            callbacks.ModelCheckpoint(
                filepath=str(model_path),
                monitor="val_loss",
                save_best_only=True,
                verbose=0,
            ),
        ]

        print("\nModel:")
        model.summary()

        # --------------------------------------------
        # Train
        # --------------------------------------------

        history = model.fit(
            X_train,
            y_train,
            validation_data=(
                X_valid,
                y_valid,
            ),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            shuffle=False,
            class_weight=class_weights,
            callbacks=callback_list,
            verbose=1,
        )

        # --------------------------------------------
        # Validation probabilities
        # --------------------------------------------

        valid_probabilities = (
            model.predict(
                X_valid,
                batch_size=BATCH_SIZE,
                verbose=0,
            )
            .reshape(-1)
        )

        # --------------------------------------------
        # TEST probabilities
        # --------------------------------------------

        test_probabilities = (
            model.predict(
                X_test,
                batch_size=BATCH_SIZE,
                verbose=0,
            )
            .reshape(-1)
        )

        # --------------------------------------------
        # Threshold selection
        # --------------------------------------------

        threshold, valid_f1 = (
            validation_threshold_search(
                y_valid,
                valid_probabilities,
            )
        )

        print(
            f"\nSelected threshold: "
            f"{threshold:.2f}"
        )

        print(
            f"Validation F1 at threshold: "
            f"{valid_f1:.4f}"
        )

        # --------------------------------------------
        # Evaluate
        # --------------------------------------------

        train_probabilities = (
            model.predict(
                X_train,
                batch_size=BATCH_SIZE,
                verbose=0,
            )
            .reshape(-1)
        )

        train_metrics = evaluate_binary(
            y_train,
            train_probabilities,
            threshold,
        )

        valid_metrics = evaluate_binary(
            y_valid,
            valid_probabilities,
            threshold,
        )

        test_metrics = evaluate_binary(
            y_test,
            test_probabilities,
            threshold,
        )

        print("\nTRAIN")
        print_metrics(
            "LSTM",
            train_metrics,
        )

        print("\nVALID")
        print_metrics(
            "LSTM",
            valid_metrics,
        )

        print("\nTEST")
        print_metrics(
            "LSTM",
            test_metrics,
        )

        print(
            "\nTEST confusion matrix:"
        )

        print(
            f"TN={test_metrics['tn']} "
            f"FP={test_metrics['fp']} "
            f"FN={test_metrics['fn']} "
            f"TP={test_metrics['tp']}"
        )

        # --------------------------------------------
        # Save history
        # --------------------------------------------

        history_path = (
            OUTPUT_DIR
            / f"training_history_{horizon}s.csv"
        )

        pd.DataFrame(
            history.history
        ).to_csv(
            history_path,
            index=False,
        )

        # --------------------------------------------
        # Store result
        # --------------------------------------------

        results.append(
            {
                "horizon_seconds": horizon,

                "train_samples": train_metrics[
                    "sample_count"
                ],

                "valid_samples": valid_metrics[
                    "sample_count"
                ],

                "test_samples": test_metrics[
                    "sample_count"
                ],

                "train_positive": train_metrics[
                    "positive_count"
                ],

                "valid_positive": valid_metrics[
                    "positive_count"
                ],

                "test_positive": test_metrics[
                    "positive_count"
                ],

                "threshold": threshold,

                "train_precision": train_metrics[
                    "precision"
                ],

                "train_recall": train_metrics[
                    "recall"
                ],

                "train_f1": train_metrics[
                    "f1"
                ],

                "train_pr_auc": train_metrics[
                    "pr_auc"
                ],

                "train_roc_auc": train_metrics[
                    "roc_auc"
                ],

                "train_fpr": train_metrics[
                    "fpr"
                ],

                "valid_precision": valid_metrics[
                    "precision"
                ],

                "valid_recall": valid_metrics[
                    "recall"
                ],

                "valid_f1": valid_metrics[
                    "f1"
                ],

                "valid_pr_auc": valid_metrics[
                    "pr_auc"
                ],

                "valid_roc_auc": valid_metrics[
                    "roc_auc"
                ],

                "valid_fpr": valid_metrics[
                    "fpr"
                ],

                "test_precision": test_metrics[
                    "precision"
                ],

                "test_recall": test_metrics[
                    "recall"
                ],

                "test_f1": test_metrics[
                    "f1"
                ],

                "test_pr_auc": test_metrics[
                    "pr_auc"
                ],

                "test_roc_auc": test_metrics[
                    "roc_auc"
                ],

                "test_fpr": test_metrics[
                    "fpr"
                ],

                "test_tn": test_metrics[
                    "tn"
                ],

                "test_fp": test_metrics[
                    "fp"
                ],

                "test_fn": test_metrics[
                    "fn"
                ],

                "test_tp": test_metrics[
                    "tp"
                ],
            }
        )

        # Clear TensorFlow graph between horizons.
        tf.keras.backend.clear_session()

    # --------------------------------------------------------
    # 9. SAVE RESULTS
    # --------------------------------------------------------

    print("\n[9/9] Saving results...")

    results_df = pd.DataFrame(
        results
    )

    results_file = (
        OUTPUT_DIR
        / "early_warning_lstm_results.csv"
    )

    results_df.to_csv(
        results_file,
        index=False,
    )

    # Save experiment configuration.
    config = {
        "random_state": RANDOM_STATE,
        "time_steps": TIME_STEPS,
        "n_time_steps": N_TIME_STEPS,
        "n_features": N_FEATURES,
        "horizons": HORIZONS,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "model": {
            "type": "LSTM",
            "lstm_units": 64,
            "dropout": 0.25,
            "dense_units": 32,
            "dense_dropout": 0.15,
        },
        "split": {
            "train": [
                "Wednesday-14-02-2018",
                "Thursday-15-02-2018",
                "Friday-16-02-2018",
                "Thuesday-20-02-2018",
                "Wednesday-21-02-2018",
                "Thursday-22-02-2018",
                "Friday-23-02-2018",
            ],
            "valid": [
                "Wednesday-28-02-2018",
            ],
            "test": [
                "Thursday-01-03-2018",
                "Friday-02-03-2018",
            ],
        },
        "preprocessing": {
            "imputation": "median",
            "scaler": "StandardScaler",
            "fit_on_train_only": True,
        },
        "threshold_selection": (
            "validation F1"
        ),
    }

    config_file = (
        OUTPUT_DIR
        / "experiment_config.json"
    )

    with open(
        config_file,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            config,
            file,
            indent=2,
        )

    print(
        "\nResults saved to:"
    )

    print(results_file)

    print(
        "\nModels saved in:"
    )

    print(OUTPUT_DIR)

    print("\n")
    print("=" * 80)
    print("FINAL EARLY-WARNING LSTM RESULTS")
    print("=" * 80)

    display_columns = [
        "horizon_seconds",
        "threshold",
        "valid_f1",
        "test_precision",
        "test_recall",
        "test_f1",
        "test_pr_auc",
        "test_roc_auc",
        "test_fpr",
        "test_tp",
        "test_fp",
        "test_fn",
    ]

    print(
        results_df[
            display_columns
        ].to_string(index=False)
    )

    print("\nDone.")


if __name__ == "__main__":
    main()