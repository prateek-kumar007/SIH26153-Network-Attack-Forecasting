from pathlib import Path
import re

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

INPUT = (
    ROOT
    / "data"
    / "processed"
    / "forecasting_samples"
    / "all_forecasting_samples.csv"
)

OUTPUT_DIR = ROOT / "data" / "sample"
OUTPUT = OUTPUT_DIR / "sample_sequence.csv"


FEATURE_PATTERN = re.compile(r"^t-(\d+)_(.+)$")


def main():

    print("=" * 60)
    print("Creating sample sequence for SIH26153 inference")
    print("=" * 60)

    if not INPUT.exists():
        raise FileNotFoundError(
            f"Forecasting dataset not found:\n{INPUT}"
        )

    df = pd.read_csv(INPUT, nrows=1)

    # Find historical feature columns
    historical_columns = []

    for column in df.columns:

        match = FEATURE_PATTERN.match(column)

        if match:
            historical_columns.append(column)

    # Group by timestep
    timestep_features = {}

    for column in historical_columns:

        match = FEATURE_PATTERN.match(column)

        timestep = int(match.group(1))
        feature = match.group(2)

        timestep_features.setdefault(
            timestep,
            []
        ).append(feature)

    expected_timesteps = list(range(9, -1, -1))

    if sorted(timestep_features.keys(), reverse=True) != expected_timesteps:
        raise ValueError(
            "Unexpected historical timesteps:\n"
            f"{sorted(timestep_features.keys(), reverse=True)}"
        )

    # Check feature consistency
    feature_sets = [
        set(features)
        for features in timestep_features.values()
    ]

    first_features = feature_sets[0]

    for features in feature_sets:

        if features != first_features:
            raise ValueError(
                "Historical features are not consistent "
                "across all timesteps."
            )

    features = sorted(first_features)

    if len(features) != 24:
        raise ValueError(
            f"Expected 24 features, found {len(features)}"
        )

    # Read first actual sample
    full_df = pd.read_csv(INPUT, nrows=1)

    rows = []

    for timestep in expected_timesteps:

        row = {}

        for feature in features:

            column = f"t-{timestep}_{feature}"

            row[feature] = full_df.iloc[0][column]

        rows.append(row)

    sample = pd.DataFrame(rows)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    sample.to_csv(
        OUTPUT,
        index=False
    )

    print()
    print(f"Input dataset : {INPUT}")
    print(f"Output file   : {OUTPUT}")
    print(f"Shape         : {sample.shape}")
    print()

    print("Timesteps:")
    print("row 0 -> t-9")
    print("row 1 -> t-8")
    print("...")
    print("row 9 -> t-0")

    print()
    print("Features:")
    for feature in features:
        print(f"  {feature}")

    print()
    print("Sample created successfully.")


if __name__ == "__main__":
    main()