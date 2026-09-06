"""
SIH26153 - LSTM Forecasting Inference

Purpose:
    Load the trained LSTM model and preprocessing artifacts,
    accept the latest 10 network-state windows, and return
    attack-risk probabilities for the next 30/60/90/120/150 seconds.

Current model:
    Flow-based 24-feature network state.

Input:
    NumPy array:
        shape = (10, 24)

    OR pandas DataFrame:
        10 rows × 24 required feature columns

Output:
    Dictionary containing probability, threshold,
    prediction and risk for each forecast horizon.

IMPORTANT:
    This inference script reflects the current flow-only LSTM baseline.
    It is NOT evidence of successful early-warning detection.

CLI Usage:
    python src/forecasting/predict.py \
        --input data/sample/sample_sequence.csv \
        --model_dir results/lstm_baseline \
        --output data/sample/predictions.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = ROOT / "results" / "lstm_baseline"

MODEL_PATH = MODEL_DIR / "model.keras"
SCALER_PATH = MODEL_DIR / "scaler.joblib"
THRESHOLD_PATH = MODEL_DIR / "thresholds.csv"


# ============================================================
# MODEL CONFIGURATION
# ============================================================

HISTORY_WINDOWS = 10

HORIZONS = [
    "+30s",
    "+60s",
    "+90s",
    "+120s",
    "+150s",
]

FEATURES = [
    "Flow_Count",
    "Total_Fwd_Pkts",
    "Total_Bwd_Pkts",
    "Total_Fwd_Bytes",
    "Total_Bwd_Bytes",
    "Mean_Flow_Duration",
    "Mean_Flow_Bytes_Rate",
    "Mean_Flow_Packets_Rate",
    "Mean_Flow_IAT",
    "Mean_Flow_IAT_Std",
    "Mean_Flow_IAT_Max",
    "Mean_Flow_IAT_Min",
    "Mean_Fwd_Packets_Rate",
    "Mean_Bwd_Packets_Rate",
    "Mean_Packet_Length",
    "Mean_Packet_Length_Std",
    "Max_Packet_Length",
    "Min_Packet_Length",
    "SYN_Count",
    "ACK_Count",
    "RST_Count",
    "PSH_Count",
    "Unique_Dst_Ports",
    "Unique_Protocols",
]


# ============================================================
# LOAD ARTIFACTS
# ============================================================

def load_artifacts(model_dir: str | None = None):
    """Load trained model, scaler and validation thresholds."""

    if model_dir is None:
        model_dir = MODEL_DIR
    else:
        model_dir = Path(model_dir)

    model_path = model_dir / "model.keras"
    scaler_path = model_dir / "scaler.joblib"
    threshold_path = model_dir / "thresholds.csv"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found:\n{model_path}"
        )

    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Scaler not found:\n{scaler_path}"
        )

    if not threshold_path.exists():
        raise FileNotFoundError(
            f"Threshold file not found:\n{threshold_path}"
        )

    model = tf.keras.models.load_model(
        model_path,
        compile=False
    )

    scaler = joblib.load(scaler_path)

    threshold_df = pd.read_csv(threshold_path)

    required_columns = {"horizon", "threshold"}

    if not required_columns.issubset(threshold_df.columns):
        raise ValueError(
            "thresholds.csv must contain columns: "
            "'horizon' and 'threshold'"
        )

    thresholds = {}

    for horizon in HORIZONS:

        row = threshold_df[
            threshold_df["horizon"].astype(str) == horizon
        ]

        if row.empty:
            raise ValueError(
                f"Missing threshold for horizon: {horizon}"
            )

        thresholds[horizon] = float(
            row.iloc[0]["threshold"]
        )

    return model, scaler, thresholds


# ============================================================
# INPUT VALIDATION
# ============================================================

def prepare_history(history):
    """
    Convert input history into shape (1, 10, 24).

    Accepted input:
        - pandas DataFrame with 24 named columns
        - NumPy array with shape (10, 24)
    """

    # --------------------------------------------------------
    # DataFrame input
    # --------------------------------------------------------

    if isinstance(history, pd.DataFrame):

        missing = [
            feature
            for feature in FEATURES
            if feature not in history.columns
        ]

        if missing:
            raise ValueError(
                "Input DataFrame is missing features:\n"
                + "\n".join(missing)
            )

        history = history[FEATURES].copy()

        if history.shape[0] != HISTORY_WINDOWS:
            raise ValueError(
                f"Expected {HISTORY_WINDOWS} history windows, "
                f"received {history.shape[0]}"
            )

        values = history.to_numpy(dtype=np.float64)

    # --------------------------------------------------------
    # NumPy / array-like input
    # --------------------------------------------------------

    else:

        values = np.asarray(history, dtype=np.float64)

        if values.shape != (
            HISTORY_WINDOWS,
            len(FEATURES)
        ):
            raise ValueError(
                "Invalid history shape.\n"
                f"Expected: ({HISTORY_WINDOWS}, {len(FEATURES)})\n"
                f"Received: {values.shape}"
            )

    # --------------------------------------------------------
    # Numerical validation
    # --------------------------------------------------------

    if not np.isfinite(values).all():
        raise ValueError(
            "Input history contains NaN or infinite values."
        )

    return values


# ============================================================
# RISK LABEL
# ============================================================

def risk_level(probability):
    """
    UI-oriented risk label.

    IMPORTANT:
        These are NOT official cybersecurity risk levels.
    """

    if probability < 0.30:
        return "LOW"

    if probability < 0.70:
        return "MEDIUM"

    return "HIGH"


# ============================================================
# MAIN PREDICTION FUNCTION
# ============================================================

def predict(history, model_dir: str | None = None):
    """
    Generate multi-horizon attack forecasts.

    Parameters
    ----------
    history : pandas.DataFrame or array-like
        Latest 10 network-state windows containing
        the 24 required features.

    model_dir : str or None
        Directory containing model.keras, scaler.joblib, thresholds.csv.
        If None, uses the default MODEL_DIR.

    Returns
    -------
    dict
        Forecast results for +30s, +60s, +90s,
        +120s and +150s.
    """

    # Load model artifacts
    model, scaler, thresholds = load_artifacts(model_dir)

    # Validate and prepare input
    values = prepare_history(history)

    # --------------------------------------------------------
    # Apply EXACT same scaler used during training
    # --------------------------------------------------------

    scaled = scaler.transform(values)

    # Restore sequence shape:
    # (10, 24) -> (1, 10, 24)

    X = scaled.reshape(
        1,
        HISTORY_WINDOWS,
        len(FEATURES)
    )

    # --------------------------------------------------------
    # Model prediction
    # --------------------------------------------------------

    probabilities = model.predict(
        X,
        verbose=0
    )[0]

    # --------------------------------------------------------
    # Build result
    # --------------------------------------------------------

    forecasts = {}

    for index, horizon in enumerate(HORIZONS):

        probability = float(probabilities[index])
        threshold = thresholds[horizon]

        predicted_attack = probability >= threshold

        forecasts[horizon] = {
            "probability": round(probability, 6),

            "probability_percent": round(
                probability * 100,
                2
            ),

            "threshold": round(
                threshold,
                6
            ),

            "predicted_attack": bool(
                predicted_attack
            ),

            "risk": risk_level(probability),
        }

    return {
        "forecasts": forecasts
    }


# ============================================================
# SIMPLE TERMINAL DISPLAY
# ============================================================

def print_prediction(result):
    """Pretty-print prediction results."""

    print("\n" + "=" * 60)
    print("SIH26153 — NETWORK ATTACK FORECAST")
    print("=" * 60)

    for horizon, result_data in result["forecasts"].items():

        print(
            f"{horizon:>6} | "
            f"Probability: "
            f"{result_data['probability_percent']:>6.2f}% | "
            f"Threshold: "
            f"{result_data['threshold']:.3f} | "
            f"Prediction: "
            f"{'ATTACK' if result_data['predicted_attack'] else 'BENIGN':>6} | "
            f"Risk: "
            f"{result_data['risk']}"
        )

    print("=" * 60)


# ============================================================
# CLI FOR BATCH PREDICTION ON CSV
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "SIH26153 LSTM inference. "
            "Accepts a CSV with 10×24 feature rows and writes predictions."
        )
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input CSV with 10 rows and 24 feature columns.",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Directory containing model.keras, scaler.joblib, thresholds.csv",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output CSV with predictions.",
    )
    return parser.parse_args()


def predict_from_csv(input_path: str, model_dir: str, output_path: str):
    """
    Batch prediction for a CSV where each row is one 10-window history.

    Expected CSV:
        - 10 rows per sample is NOT expected here.
        - Instead: each ROW is one time step, and we assume the file
          contains exactly 10 rows representing one history sequence.

    For a more general batch format (many sequences in one file),
    you can extend this later. For now, we treat the entire file as
    one 10-row history.
    """

    df = pd.read_csv(input_path)

    if df.shape[0] != HISTORY_WINDOWS:
        raise ValueError(
            f"Expected exactly {HISTORY_WINDOWS} rows in input CSV, "
            f"got {df.shape[0]}."
        )

    result = predict(df, model_dir=model_dir)

    # Build a compact output table
    rows = []
    for horizon, data in result["forecasts"].items():
        rows.append(
            {
                "horizon": horizon,
                "probability": data["probability"],
                "probability_percent": data["probability_percent"],
                "threshold": data["threshold"],
                "predicted_attack": int(data["predicted_attack"]),
                "risk": data["risk"],
            }
        )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_path, index=False)

    print(f"Predictions written to {output_path}")
    print_prediction(result)


def main():
    args = parse_args()
    predict_from_csv(args.input, args.model_dir, args.output)


if __name__ == "__main__":
    main()