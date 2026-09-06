from pathlib import Path
import re
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

FORECAST_PATH = (
    ROOT
    / "data"
    / "processed"
    / "forecasting_samples"
    / "all_forecasting_samples.csv"
)

EARLY_PATH = (
    ROOT
    / "data"
    / "processed"
    / "early_warning"
    / "early_warning_samples.csv"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "early_warning"
    / "baseline_results"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


HORIZONS = [30, 60, 90, 120, 150]


# ============================================================
# HELPERS
# ============================================================

def normalize_source(series):
    """
    Normalize source filenames so forecasting and early-warning
    datasets can be joined reliably.
    """
    s = series.astype(str).str.strip()

    suffixes = [
        "_TrafficForML_CICFlowMeter_30s_state.csv",
        "_TrafficForML_CICFlowMeter.csv",
        ".csv",
    ]

    for suffix in suffixes:
        s = s.str.replace(
            suffix,
            "",
            regex=False,
        )

    return s


def parse_datetime(series):
    return pd.to_datetime(
        series.astype(str).str.strip(),
        errors="coerce",
    )


def evaluate_binary(y_true, y_prob, threshold):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)

    y_pred = (y_prob >= threshold).astype(int)

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

    pr_auc = average_precision_score(
        y_true,
        y_prob,
    )

    if len(np.unique(y_true)) == 2:
        roc_auc = roc_auc_score(
            y_true,
            y_prob,
        )
    else:
        roc_auc = np.nan

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
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
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }


def find_best_threshold(y_true, y_prob):
    """
    Select threshold using VALIDATION F1 only.
    """
    best = None

    for threshold in np.arange(
        0.05,
        0.96,
        0.01,
    ):
        result = evaluate_binary(
            y_true,
            y_prob,
            threshold,
        )

        if best is None or result["f1"] > best["f1"]:
            best = result

    return best


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("EARLY WARNING — LOGISTIC REGRESSION BASELINE")
print("=" * 70)

print("\nLoading forecasting dataset:")
print(FORECAST_PATH)

forecast = pd.read_csv(
    FORECAST_PATH,
    low_memory=False,
)

print(f"Forecasting rows : {len(forecast):,}")
print(f"Forecasting cols : {len(forecast.columns):,}")


print("\nLoading early-warning labels:")
print(EARLY_PATH)

early = pd.read_csv(
    EARLY_PATH,
    low_memory=False,
)

print(f"Early-warning rows: {len(early):,}")
print(f"Early-warning cols: {len(early.columns):,}")


# ============================================================
# IDENTIFY THE 240 HISTORICAL FEATURES
# ============================================================

feature_cols = [
    c
    for c in forecast.columns
    if re.match(r"^t-\d+_", str(c))
]

print("\nHistorical feature count:", len(feature_cols))

if len(feature_cols) != 240:
    print("\nWARNING:")
    print(
        f"Expected 240 historical features, "
        f"but found {len(feature_cols)}."
    )

    print("\nDetected feature columns:")
    print(feature_cols[:50])

    raise ValueError(
        "Feature count is not 240. "
        "Stop before training to avoid an incorrect experiment."
    )

print("✓ Exactly 240 historical features detected.")

time_steps = sorted(
    {
        int(re.match(r"^t-(\d+)_", c).group(1))
        for c in feature_cols
    },
    reverse=True,
)

print("Time steps:", time_steps)

if time_steps != list(range(9, -1, -1)):
    raise ValueError(
        f"Unexpected time steps: {time_steps}"
    )

print("✓ Time steps t-9 ... t-0 confirmed.")


# ============================================================
# BUILD JOIN KEYS
# ============================================================

forecast["Source_Key"] = normalize_source(
    forecast["Source_File"]
)

early["Source_Key"] = normalize_source(
    early["Source_File"]
)


forecast["Time_Key"] = parse_datetime(
    forecast["Target_Window"]
)

early["Time_Key"] = parse_datetime(
    early["Forecast_Target_Window"]
)


if forecast["Time_Key"].isna().any():
    raise ValueError(
        "Forecasting dataset contains invalid Target_Window timestamps."
    )

if early["Time_Key"].isna().any():
    raise ValueError(
        "Early-warning dataset contains invalid "
        "Forecast_Target_Window timestamps."
    )


print("\nJoin columns:")
print("Forecasting time column : Target_Window")
print("Early-warning time column: Forecast_Target_Window")


# ============================================================
# PREPARE FEATURE TABLE
# ============================================================

forecast_features = forecast[
    [
        "Source_Key",
        "Time_Key",
    ]
    + feature_cols
].copy()


duplicate_count = forecast_features.duplicated(
    ["Source_Key", "Time_Key"]
).sum()

print("\nForecasting duplicate join keys:", duplicate_count)

if duplicate_count > 0:
    raise ValueError(
        "Duplicate Source_Key + Time_Key rows found."
    )


# ============================================================
# MERGE EARLY-WARNING LABELS WITH FEATURES
# ============================================================

merged = early.merge(
    forecast_features,
    on=["Source_Key", "Time_Key"],
    how="left",
    validate="one_to_one",
)

print("\nMerged rows:", f"{len(merged):,}")


# ============================================================
# VERIFY ALL EARLY-WARNING ROWS MATCH
# ============================================================

missing_feature_rows = merged[
    feature_cols
].isna().all(axis=1).sum()

print(
    "Rows with completely missing feature history:",
    missing_feature_rows,
)

if missing_feature_rows > 0:
    print(
        "\nERROR: Some early-warning samples could not "
        "be matched to forecasting features."
    )

    print(
        merged.loc[
            merged[feature_cols].isna().all(axis=1),
            [
                "Source_File",
                "Forecast_Target_Window",
                "Source_Key",
                "Time_Key",
            ],
        ].head(20)
    )

    raise ValueError(
        "Early-warning → forecasting feature join failed."
    )


if len(merged) != len(early):
    raise ValueError(
        "Merged row count does not match early-warning dataset."
    )

print("✓ All early-warning rows matched to forecasting features.")


# ============================================================
# VERIFY CURRENT STATE IS BENIGN
# ============================================================

if "Current_Attack" in merged.columns:

    current_attack = pd.to_numeric(
        merged["Current_Attack"],
        errors="coerce",
    )

    if current_attack.isna().any():
        raise ValueError(
            "Current_Attack contains invalid values."
        )

    if not (current_attack == 0).all():
        raise ValueError(
            "Early-warning dataset contains current attack rows."
        )

    print("✓ All early-warning samples are currently benign.")

else:
    print(
        "WARNING: 'Current_Attack' column not found — "
        "skipped verifying that all early-warning samples "
        "are currently benign."
    )


# ============================================================
# VERIFY SPLITS
# ============================================================

required_splits = {"TRAIN", "VALIDATION", "TEST"}

actual_splits = set(
    merged["Split"].astype(str).str.upper().unique()
)

print("\nSplits:", actual_splits)

if not required_splits.issubset(actual_splits):
    raise ValueError(
        f"Missing required splits. Found: {actual_splits}"
    )


# ============================================================
# PREPARE X
# ============================================================

X = merged[feature_cols].apply(
    pd.to_numeric,
    errors="coerce",
).astype(float)

print("\nFeature matrix:", X.shape)

if X.shape[1] != 240:
    raise ValueError(
        f"Expected 240 features, got {X.shape[1]}"
    )


# ============================================================
# TRAIN / VALID / TEST MASKS
# ============================================================

split = (
    merged["Split"]
    .astype(str)
    .str.upper()
)

train_mask = split == "TRAIN"
valid_mask = split == "VALIDATION"
test_mask = split == "TEST"

print("\nSplit sizes:")
print("TRAIN:", train_mask.sum())
print("VALID:", valid_mask.sum())
print("TEST :", test_mask.sum())


# ============================================================
# TRAIN-ONLY PREPROCESSING
# ============================================================

imputer = SimpleImputer(
    strategy="median"
)

scaler = StandardScaler()


X_train = imputer.fit_transform(
    X.loc[train_mask]
)

X_valid = imputer.transform(
    X.loc[valid_mask]
)

X_test = imputer.transform(
    X.loc[test_mask]
)


X_train = scaler.fit_transform(
    X_train
)

X_valid = scaler.transform(
    X_valid
)

X_test = scaler.transform(
    X_test
)

print("\n✓ Imputer fitted on TRAIN only.")
print("✓ Scaler fitted on TRAIN only.")


# ============================================================
# TRAIN ONE MODEL PER EARLY-WARNING HORIZON
# ============================================================

results = []

for horizon in HORIZONS:

    target_col = f"Starts_Within_{horizon}s"

    if target_col not in merged.columns:
        raise ValueError(
            f"Missing target column: {target_col}"
        )

    y = pd.to_numeric(
        merged[target_col],
        errors="coerce",
    )

    if y.isna().any():
        raise ValueError(
            f"{target_col} contains invalid/missing values "
            "that could not be converted to numeric labels."
        )

    y = y.astype(int)

    y_train = y.loc[train_mask].to_numpy()
    y_valid = y.loc[valid_mask].to_numpy()
    y_test = y.loc[test_mask].to_numpy()

    print("\n" + "=" * 70)
    print(f"EARLY WARNING HORIZON: +{horizon}s")
    print("=" * 70)

    print(
        f"TRAIN positives: {y_train.sum():,} "
        f"/ {len(y_train):,} "
        f"({100 * y_train.mean():.3f}%)"
    )

    print(
        f"VALID positives: {y_valid.sum():,} "
        f"/ {len(y_valid):,} "
        f"({100 * y_valid.mean():.3f}%)"
    )

    print(
        f"TEST positives : {y_test.sum():,} "
        f"/ {len(y_test):,} "
        f"({100 * y_test.mean():.3f}%)"
    )

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=42,
    )

    model.fit(
        X_train,
        y_train,
    )

    valid_prob = model.predict_proba(
        X_valid
    )[:, 1]

    best_valid = find_best_threshold(
        y_valid,
        valid_prob,
    )

    threshold = best_valid["threshold"]

    print("\nBest validation threshold:")
    print(f"{threshold:.2f}")

    print(
        f"VALID F1: "
        f"{best_valid['f1']:.4f}"
    )

    test_prob = model.predict_proba(
        X_test
    )[:, 1]

    test_result = evaluate_binary(
        y_test,
        test_prob,
        threshold,
    )

    print("\nTEST RESULTS")

    print(
        f"Precision : {test_result['precision']:.4f}"
    )

    print(
        f"Recall    : {test_result['recall']:.4f}"
    )

    print(
        f"F1        : {test_result['f1']:.4f}"
    )

    print(
        f"PR-AUC    : {test_result['pr_auc']:.4f}"
    )

    print(
        f"ROC-AUC   : {test_result['roc_auc']:.4f}"
    )

    print(
        f"FPR       : {test_result['fpr']:.4f}"
    )

    print("\nConfusion matrix:")
    print(
        f"TN={test_result['TN']} "
        f"FP={test_result['FP']} "
        f"FN={test_result['FN']} "
        f"TP={test_result['TP']}"
    )

    row = {
        "horizon_seconds": horizon,
        "threshold": threshold,
        "train_rows": len(y_train),
        "train_positives": int(y_train.sum()),
        "valid_rows": len(y_valid),
        "valid_positives": int(y_valid.sum()),
        "test_rows": len(y_test),
        "test_positives": int(y_test.sum()),
        "precision": test_result["precision"],
        "recall": test_result["recall"],
        "f1": test_result["f1"],
        "pr_auc": test_result["pr_auc"],
        "roc_auc": test_result["roc_auc"],
        "fpr": test_result["fpr"],
        "TN": test_result["TN"],
        "FP": test_result["FP"],
        "FN": test_result["FN"],
        "TP": test_result["TP"],
    }

    results.append(row)


# ============================================================
# SAVE ALL RESULTS
# ============================================================

results_df = pd.DataFrame(results)

output_path = (
    OUTPUT_DIR
    / "early_warning_logistic_results.csv"
)

results_df.to_csv(
    output_path,
    index=False,
)

print("\n" + "=" * 70)
print("FINAL EARLY-WARNING LOGISTIC RESULTS")
print("=" * 70)

print(
    results_df[
        [
            "horizon_seconds",
            "threshold",
            "precision",
            "recall",
            "f1",
            "pr_auc",
            "roc_auc",
            "fpr",
        ]
    ].to_string(index=False)
)

print("\nSaved:")
print(output_path)

print("\nDONE.")
