import pandas as pd
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

QUALITY_FILE = Path(
    r"D:\SIH26153\results\deep_quality\CIC2018_deep_data_quality.csv"
)

OUTPUT_DIR = Path(
    r"D:\SIH26153\results\feature_selection"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD QUALITY AUDIT
# ============================================================

df = pd.read_csv(QUALITY_FILE)


# ============================================================
# INITIAL DECISIONS
# ============================================================

REMOVE = {
    # Constant
    "Bwd Blk Rate Avg",
    "Bwd Byts/b Avg",
    "Bwd PSH Flags",
    "Bwd Pkts/b Avg",
    "Bwd URG Flags",
    "Fwd Blk Rate Avg",
    "Fwd Byts/b Avg",
    "Fwd Pkts/b Avg",

    # Extremely sparse
    "CWE Flag Count",
    "FIN Flag Cnt",
    "Fwd URG Flags",

    # Identifiers
    "Flow ID",
    "Src IP",
    "Dst IP",
}


KEEP_SPECIAL = {
    "Timestamp",
    "Label",
}


# ============================================================
# DECISION FUNCTION
# ============================================================

def decide(feature):

    if feature in REMOVE:
        return "REMOVE"

    if feature in KEEP_SPECIAL:
        return "SPECIAL"

    return "CANDIDATE"


df["Decision"] = df["Feature"].apply(decide)


# ============================================================
# ADD REASON
# ============================================================

def reason(row):

    feature = row["Feature"]

    if feature in KEEP_SPECIAL:

        if feature == "Timestamp":
            return "Used for chronological ordering and window construction"

        if feature == "Label":
            return "Target variable only; never an input feature"

    if feature in REMOVE:

        if feature in {
            "Flow ID",
            "Src IP",
            "Dst IP"
        }:
            return "Identifier/address field; not used as behavioural state feature"

        if row["Status"] == "CONSTANT":
            return "Constant across dataset; contains no variance"

        if row["Zero_%"] >= 99:
            return "Extremely sparse; remove from initial model"

    return "Candidate traffic-behaviour feature"


df["Reason"] = df.apply(reason, axis=1)


# ============================================================
# SAVE
# ============================================================

output_file = (
    OUTPUT_DIR /
    "CIC2018_feature_retention.csv"
)

df[
    [
        "Feature",
        "Status",
        "NaN_%",
        "Inf_%",
        "Zero_%",
        "Min",
        "Max",
        "Decision",
        "Reason",
    ]
].to_csv(
    output_file,
    index=False
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print("=" * 70)
print("FEATURE RETENTION REPORT")
print("=" * 70)

print(
    df["Decision"].value_counts()
)

print("\nREMOVE:")
print(
    df[df["Decision"] == "REMOVE"][
        ["Feature", "Reason"]
    ].to_string(index=False)
)

print("\nSPECIAL:")
print(
    df[df["Decision"] == "SPECIAL"][
        ["Feature", "Reason"]
    ].to_string(index=False)
)

print("\nCANDIDATE FEATURES:")
print(
    df[df["Decision"] == "CANDIDATE"][
        ["Feature", "Status", "Zero_%"]
    ].to_string(index=False)
)

print("\nSaved:")
print(output_file)