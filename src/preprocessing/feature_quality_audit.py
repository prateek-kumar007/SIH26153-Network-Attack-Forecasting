from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path("data/interim/CIC-IDS2018")
OUTPUT_DIR = Path("results/feature_quality")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 100_000

# Features we currently consider candidates for S_t
EXCLUDE = {"Timestamp", "Label"}

# Statistics collected across the complete dataset
stats = {}

# Correlation sample: we don't need all 16M rows.
SAMPLE_PER_FILE = 10_000


def update_stats(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        if col not in stats:
            stats[col] = {
                "count": 0,
                "missing": 0,
                "zero": 0,
                "sum": 0.0,
                "sum_sq": 0.0,
                "min": np.inf,
                "max": -np.inf,
            }

        s = pd.to_numeric(df[col], errors="coerce")

        valid = s.notna()

        stats[col]["count"] += len(s)
        stats[col]["missing"] += s.isna().sum()
        stats[col]["zero"] += (s == 0).sum()

        if valid.any():
            values = s[valid].astype(float)

            stats[col]["sum"] += values.sum()
            stats[col]["sum_sq"] += (values ** 2).sum()
            stats[col]["min"] = min(stats[col]["min"], values.min())
            stats[col]["max"] = max(stats[col]["max"], values.max())


print("=" * 70)
print("CIC-IDS2018 FEATURE QUALITY AUDIT")
print("=" * 70)

sample_parts = []

files = sorted(DATA_DIR.glob("*.csv"))

for file_path in files:

    print(f"\nProcessing: {file_path.name}")

    first_sample = True

    for chunk in pd.read_csv(
        file_path,
        chunksize=CHUNK_SIZE,
        low_memory=False
    ):

        # Remove metadata/target from numerical quality analysis
        feature_cols = [
            c for c in chunk.columns
            if c not in EXCLUDE
        ]

        numeric = chunk[feature_cols].apply(
            pd.to_numeric,
            errors="coerce"
        )

        update_stats(numeric)

        # Small sample for correlation analysis
        if first_sample:
            sample_parts.append(
                numeric.sample(
                    min(SAMPLE_PER_FILE, len(numeric)),
                    random_state=42
                )
            )
            first_sample = False


# ---------------------------------------------------------
# BUILD QUALITY REPORT
# ---------------------------------------------------------

rows = []

for feature, s in stats.items():

    count = s["count"]

    mean = s["sum"] / count if count else np.nan

    variance = (
        (s["sum_sq"] / count) - (mean ** 2)
        if count
        else np.nan
    )

    variance = max(variance, 0)

    std = np.sqrt(variance)

    missing_pct = (
        s["missing"] / count * 100
        if count
        else np.nan
    )

    zero_pct = (
        s["zero"] / count * 100
        if count
        else np.nan
    )

    if s["min"] == s["max"]:
        status = "CONSTANT"
    elif missing_pct > 5:
        status = "HIGH_MISSING"
    elif zero_pct > 99:
        status = "MOSTLY_ZERO"
    else:
        status = "USABLE"

    rows.append({
        "Feature": feature,
        "Count": count,
        "Missing": s["missing"],
        "Missing_%": missing_pct,
        "Zero": s["zero"],
        "Zero_%": zero_pct,
        "Min": s["min"],
        "Max": s["max"],
        "Mean": mean,
        "Std": std,
        "Status": status
    })


quality_df = pd.DataFrame(rows)

quality_df = quality_df.sort_values("Feature")

quality_path = OUTPUT_DIR / "CIC2018_feature_quality.csv"

quality_df.to_csv(
    quality_path,
    index=False
)


# ---------------------------------------------------------
# CORRELATION ANALYSIS
# ---------------------------------------------------------

sample_df = pd.concat(
    sample_parts,
    ignore_index=True
)

sample_df = sample_df.replace(
    [np.inf, -np.inf],
    np.nan
)

correlation = sample_df.corr(
    numeric_only=True
)

correlation_path = OUTPUT_DIR / "CIC2018_feature_correlation.csv"

correlation.to_csv(
    correlation_path
)


# ---------------------------------------------------------
# HIGH CORRELATION PAIRS
# ---------------------------------------------------------

pairs = []

columns = correlation.columns

for i in range(len(columns)):
    for j in range(i + 1, len(columns)):

        value = correlation.iloc[i, j]

        if pd.notna(value) and abs(value) >= 0.95:

            pairs.append({
                "Feature_A": columns[i],
                "Feature_B": columns[j],
                "Correlation": value
            })


pairs_df = pd.DataFrame(pairs)

pairs_path = OUTPUT_DIR / "high_correlation_pairs.csv"

pairs_df.to_csv(
    pairs_path,
    index=False
)


# ---------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)

print(f"\nFeatures analysed: {len(quality_df)}")

print("\nQuality status:")
print(
    quality_df["Status"]
    .value_counts()
)

print("\nHigh-correlation pairs (|r| >= 0.95):")
print(len(pairs_df))

print(f"\nSaved:")
print(f"  {quality_path}")
print(f"  {correlation_path}")
print(f"  {pairs_path}")

print("\nTop features with highest missing percentage:")
print(
    quality_df
    .sort_values("Missing_%", ascending=False)
    [["Feature", "Missing_%"]]
    .head(10)
    .to_string(index=False)
)

print("\nTop features with highest zero percentage:")
print(
    quality_df
    .sort_values("Zero_%", ascending=False)
    [["Feature", "Zero_%"]]
    .head(10)
    .to_string(index=False)
)