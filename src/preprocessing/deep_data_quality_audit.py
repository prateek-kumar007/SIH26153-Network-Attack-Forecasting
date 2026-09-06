import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

DATA_DIR = Path(r"D:\SIH26153\data\interim\CIC-IDS2018")
RESULT_DIR = Path(r"D:\SIH26153\results\deep_quality")

RESULT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 100_000

# ============================================================
# FIND FILES
# ============================================================

files = sorted(DATA_DIR.glob("*.csv"))

if not files:
    raise FileNotFoundError(f"No CSV files found in {DATA_DIR}")

print("=" * 70)
print("CIC-IDS2018 DEEP DATA QUALITY AUDIT")
print("=" * 70)

# ============================================================
# GLOBAL STORAGE
# ============================================================

stats = {}

sample_frames = []

# ============================================================
# PROCESS EACH FILE
# ============================================================

for file_path in files:

    print(f"\nProcessing: {file_path.name}")

    # --------------------------------------------------------
    # Read header
    # --------------------------------------------------------

    header = pd.read_csv(file_path, nrows=0)

    columns = list(header.columns)

    # --------------------------------------------------------
    # Initialize statistics for columns
    # --------------------------------------------------------

    for col in columns:

        if col not in stats:

            stats[col] = {
                "count": 0,
                "nan": 0,
                "pos_inf": 0,
                "neg_inf": 0,
                "zero": 0,
                "finite": 0,
                "min": np.inf,
                "max": -np.inf,
                "first_finite": None,
                "all_same": True,
                "files_present": 0,
            }

        stats[col]["files_present"] += 1

    # --------------------------------------------------------
    # Chunk processing
    # --------------------------------------------------------

    sampled_chunks = []

    for chunk in pd.read_csv(file_path, chunksize=CHUNK_SIZE):

        for col in columns:

            series = chunk[col]

            stats[col]["count"] += len(series)

            # ------------------------------------------------
            # Numeric conversion
            # ------------------------------------------------

            numeric = pd.to_numeric(series, errors="coerce")

            nan_mask = numeric.isna()
            pos_inf_mask = np.isposinf(numeric)
            neg_inf_mask = np.isneginf(numeric)

            finite_mask = np.isfinite(numeric)

            finite_values = numeric[finite_mask]

            stats[col]["nan"] += nan_mask.sum()
            stats[col]["pos_inf"] += pos_inf_mask.sum()
            stats[col]["neg_inf"] += neg_inf_mask.sum()

            stats[col]["finite"] += len(finite_values)

            if len(finite_values) > 0:

                chunk_min = finite_values.min()
                chunk_max = finite_values.max()

                stats[col]["min"] = min(
                    stats[col]["min"],
                    chunk_min
                )

                stats[col]["max"] = max(
                    stats[col]["max"],
                    chunk_max
                )

                stats[col]["zero"] += (
                    finite_values == 0
                ).sum()

                # --------------------------------------------
                # Constant detection
                # --------------------------------------------

                unique_values = finite_values.unique()

                if stats[col]["first_finite"] is None:

                    stats[col]["first_finite"] = unique_values[0]

                    if len(unique_values) > 1:
                        stats[col]["all_same"] = False

                else:

                    if (
                        len(unique_values) > 1
                        or unique_values[0] != stats[col]["first_finite"]
                    ):
                        stats[col]["all_same"] = False

        # ----------------------------------------------------
        # Sample data for correlation analysis
        # ----------------------------------------------------

        numeric_chunk = chunk.select_dtypes(
            include=[np.number]
        )

        if len(numeric_chunk) > 0:

            sample_size = min(1000, len(numeric_chunk))

            sampled_chunks.append(
                numeric_chunk.sample(
                    n=sample_size,
                    random_state=42
                )
            )

    # --------------------------------------------------------
    # Save per-file sample
    # --------------------------------------------------------

    if sampled_chunks:

        sample_frames.append(
            pd.concat(sampled_chunks, ignore_index=True)
        )

# ============================================================
# BUILD QUALITY TABLE
# ============================================================

rows = []

for feature, s in stats.items():

    count = s["count"]

    nan_pct = (
        s["nan"] / count * 100
        if count else 0
    )

    inf_pct = (
        (s["pos_inf"] + s["neg_inf"])
        / count * 100
        if count else 0
    )

    zero_pct = (
        s["zero"] / s["finite"] * 100
        if s["finite"] else 0
    )

    if s["finite"] == 0:

        status = "NON_NUMERIC"

    elif s["all_same"]:

        status = "CONSTANT"

    elif nan_pct > 5:

        status = "HIGH_MISSING"

    elif inf_pct > 0:

        status = "HAS_INFINITY"

    elif zero_pct >= 99:

        status = "MOSTLY_ZERO"

    else:

        status = "USABLE"

    rows.append({
        "Feature": feature,
        "Rows": count,
        "Finite": s["finite"],
        "NaN": s["nan"],
        "NaN_%": nan_pct,
        "Positive_Inf": s["pos_inf"],
        "Negative_Inf": s["neg_inf"],
        "Inf_%": inf_pct,
        "Zero": s["zero"],
        "Zero_%": zero_pct,
        "Min": (
            s["min"]
            if s["finite"] > 0
            else np.nan
        ),
        "Max": (
            s["max"]
            if s["finite"] > 0
            else np.nan
        ),
        "Status": status,
        "Files_Present": s["files_present"],
    })

quality_df = pd.DataFrame(rows)

quality_df = quality_df.sort_values(
    ["Status", "Feature"]
)

quality_path = (
    RESULT_DIR /
    "CIC2018_deep_data_quality.csv"
)

quality_df.to_csv(
    quality_path,
    index=False
)

# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("QUALITY SUMMARY")
print("=" * 70)

print(
    quality_df["Status"]
    .value_counts()
)

# ============================================================
# IMPORTANT FEATURES
# ============================================================

print("\n" + "=" * 70)
print("CONSTANT FEATURES")
print("=" * 70)

print(
    quality_df[
        quality_df["Status"] == "CONSTANT"
    ][
        ["Feature", "Zero_%", "Min", "Max"]
    ].to_string(index=False)
)

print("\n" + "=" * 70)
print("MOSTLY ZERO FEATURES")
print("=" * 70)

print(
    quality_df[
        quality_df["Status"] == "MOSTLY_ZERO"
    ][
        ["Feature", "Zero_%", "Min", "Max"]
    ].to_string(index=False)
)

print("\n" + "=" * 70)
print("FEATURES WITH NaN")
print("=" * 70)

print(
    quality_df[
        quality_df["NaN"] > 0
    ][
        ["Feature", "NaN", "NaN_%"]
    ]
    .sort_values("NaN_%", ascending=False)
    .to_string(index=False)
)

print("\n" + "=" * 70)
print("FEATURES WITH INFINITY")
print("=" * 70)

print(
    quality_df[
        quality_df["Inf_%"] > 0
    ][
        [
            "Feature",
            "Positive_Inf",
            "Negative_Inf",
            "Inf_%"
        ]
    ]
    .sort_values("Inf_%", ascending=False)
    .to_string(index=False)
)

# ============================================================
# CORRELATION ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("CORRELATION ANALYSIS")
print("=" * 70)

if sample_frames:

    correlation_sample = pd.concat(
        sample_frames,
        ignore_index=True
    )

    # Remove infinities
    correlation_sample = correlation_sample.replace(
        [np.inf, -np.inf],
        np.nan
    )

    correlation = correlation_sample.corr(
        numeric_only=True
    )

    correlation_pairs = []

    columns = correlation.columns

    for i in range(len(columns)):

        for j in range(i + 1, len(columns)):

            a = columns[i]
            b = columns[j]

            value = correlation.loc[a, b]

            if pd.notna(value):

                correlation_pairs.append({
                    "Feature_A": a,
                    "Feature_B": b,
                    "Correlation": value,
                    "Abs_Correlation": abs(value)
                })

    corr_df = pd.DataFrame(
        correlation_pairs
    )

    corr_df = corr_df.sort_values(
        "Abs_Correlation",
        ascending=False
    )

    corr_path = (
        RESULT_DIR /
        "CIC2018_correlations.csv"
    )

    corr_df.to_csv(
        corr_path,
        index=False
    )

    high_corr = corr_df[
        corr_df["Abs_Correlation"] >= 0.95
    ]

    high_corr_path = (
        RESULT_DIR /
        "CIC2018_high_correlations.csv"
    )

    high_corr.to_csv(
        high_corr_path,
        index=False
    )

    print(
        f"High correlation pairs (|r| >= 0.95): "
        f"{len(high_corr)}"
    )

    print("\nTop 30:")

    print(
        high_corr.head(30).to_string(
            index=False
        )
    )

# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)

print(f"Saved:")
print(f"  {quality_path}")

if sample_frames:
    print(f"  {corr_path}")
    print(f"  {high_corr_path}")