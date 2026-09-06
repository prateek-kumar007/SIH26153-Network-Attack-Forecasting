from pathlib import Path
import pandas as pd

INPUT_FILE = next(Path("data/interim/CIC-IDS2018").glob("*.csv"))
OUTPUT_DIR = Path("results/feature_audit")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Read only the header — no large data loaded
columns = pd.read_csv(INPUT_FILE, nrows=0).columns.tolist()

audit = []

def add(features, category, decision, reason, sih_requirement="Flow-level"):
    for feature in features:
        audit.append({
            "Feature": feature,
            "Category": category,
            "Decision": decision,
            "Reason": reason,
            "SIH_Level": sih_requirement
        })


# -----------------------------
# 1. IDENTIFIERS / METADATA
# -----------------------------

add(
    ["Timestamp"],
    "Temporal metadata",
    "KEEP_FOR_ORDERING",
    "Required to preserve temporal sequence and create time windows.",
    "Temporal"
)

add(
    ["Label"],
    "Target",
    "TARGET_ONLY",
    "Represents the observed traffic/attack label. Must never be included in S_t.",
    "Target"
)


# -----------------------------
# 2. TRAFFIC VOLUME
# -----------------------------

add(
    [
        "Tot Fwd Pkts",
        "Tot Bwd Pkts",
        "TotLen Fwd Pkts",
        "TotLen Bwd Pkts",
        "Flow Byts/s",
        "Flow Pkts/s"
    ],
    "Traffic volume",
    "KEEP",
    "Captures traffic intensity and communication volume."
)


# -----------------------------
# 3. FLOW TIMING
# -----------------------------

add(
    [
        "Flow Duration",
        "Flow IAT Mean",
        "Flow IAT Std",
        "Flow IAT Max",
        "Flow IAT Min",
        "Fwd IAT Tot",
        "Fwd IAT Mean",
        "Fwd IAT Std",
        "Fwd IAT Max",
        "Fwd IAT Min",
        "Bwd IAT Tot",
        "Bwd IAT Mean",
        "Bwd IAT Std",
        "Bwd IAT Max",
        "Bwd IAT Min"
    ],
    "Timing",
    "KEEP",
    "Temporal behaviour is important for forecasting evolving network states."
)


# -----------------------------
# 4. PACKET LENGTH / SIZE
# -----------------------------

add(
    [
        "Fwd Pkt Len Max",
        "Fwd Pkt Len Min",
        "Fwd Pkt Len Mean",
        "Fwd Pkt Len Std",
        "Bwd Pkt Len Max",
        "Bwd Pkt Len Min",
        "Bwd Pkt Len Mean",
        "Bwd Pkt Len Std",
        "Pkt Len Min",
        "Pkt Len Max",
        "Pkt Len Mean",
        "Pkt Len Std",
        "Pkt Len Var",
        "Pkt Size Avg",
        "Fwd Seg Size Avg",
        "Bwd Seg Size Avg",
        "Fwd Seg Size Min"
    ],
    "Packet behaviour",
    "KEEP",
    "Captures changes in packet-size behaviour."
)


# -----------------------------
# 5. TCP FLAGS
# -----------------------------

add(
    [
        "Fwd PSH Flags",
        "Bwd PSH Flags",
        "Fwd URG Flags",
        "Bwd URG Flags",
        "FIN Flag Cnt",
        "SYN Flag Cnt",
        "RST Flag Cnt",
        "PSH Flag Cnt",
        "ACK Flag Cnt",
        "URG Flag Cnt",
        "CWE Flag Count",
        "ECE Flag Cnt"
    ],
    "TCP flags",
    "KEEP",
    "TCP flag behaviour can reveal changes in connection and attack patterns."
)


# -----------------------------
# 6. COMMUNICATION FEATURES
# -----------------------------

add(
    [
        "Dst Port",
        "Protocol",
        "Down/Up Ratio"
    ],
    "Communication",
    "KEEP",
    "Useful for characterising protocol and communication behaviour."
)


# -----------------------------
# 7. WINDOW / CONNECTION STATE
# -----------------------------

add(
    [
        "Init Fwd Win Byts",
        "Init Bwd Win Byts"
    ],
    "TCP window",
    "KEEP_WITH_CAVEAT",
    "Useful flow-level indicators, but do not claim they fully satisfy packet-level TCP window-size requirements."
)


# -----------------------------
# 8. ACTIVE / IDLE BEHAVIOUR
# -----------------------------

add(
    [
        "Active Mean",
        "Active Std",
        "Active Max",
        "Active Min",
        "Idle Mean",
        "Idle Std",
        "Idle Max",
        "Idle Min"
    ],
    "Activity pattern",
    "KEEP",
    "Can capture bursty and periodic traffic behaviour."
)


# -----------------------------
# 9. FLOW BLOCK / SUBFLOW FEATURES
# -----------------------------

add(
    [
        "Fwd Byts/b Avg",
        "Fwd Pkts/b Avg",
        "Fwd Blk Rate Avg",
        "Bwd Byts/b Avg",
        "Bwd Pkts/b Avg",
        "Bwd Blk Rate Avg",
        "Subflow Fwd Pkts",
        "Subflow Fwd Byts",
        "Subflow Bwd Pkts",
        "Subflow Bwd Byts",
        "Fwd Header Len",
        "Bwd Header Len",
        "Fwd Pkts/s",
        "Bwd Pkts/s",
        "Fwd Act Data Pkts"
    ],
    "Flow-derived",
    "KEEP_FOR_INITIAL_PROTOTYPE",
    "Potentially useful, but should be tested for redundancy and predictive value."
)


# -----------------------------
# 10. PACKET-LEVEL REQUIREMENTS NOT PRESENT
# -----------------------------

packet_missing = [
    ("TTL", "Packet-level TTL and session TTL variance are not present."),
    ("TCP Window Size", "Full packet-level TCP window information is not present; initial window bytes are only partial."),
    ("IP Fragment Flags", "Explicit packet-level fragmentation information is not present."),
    ("Payload Size Distribution", "Raw packet payload-size distribution is not available directly."),
    ("Retransmission Count", "Explicit packet-level retransmission information is not available."),
    ("Port Scan Signatures", "Not directly available as a packet-level feature; may need derived behavioural features.")
]

for feature, reason in packet_missing:
    add(
        [feature],
        "Packet-level SIH requirement",
        "REQUIRES_PCAP",
        reason,
        "Packet-level"
    )


# -----------------------------
# SAVE
# -----------------------------

audit_df = pd.DataFrame(audit)

# Check whether every actual CSV column was accounted for
audited_features = set(audit_df["Feature"])

actual_columns = set(columns)

missing_from_audit = actual_columns - audited_features

if missing_from_audit:
    print("\nWARNING — columns not classified:")
    for col in sorted(missing_from_audit):
        print("  ", col)

audit_path = OUTPUT_DIR / "CIC2018_feature_audit.csv"
audit_df.to_csv(audit_path, index=False)

print("\nFeature audit complete.")
print(f"Input file: {INPUT_FILE.name}")
print(f"Actual CSV columns: {len(columns)}")
print(f"Audit entries: {len(audit_df)}")
print(f"Saved to: {audit_path}")

print("\nDecision summary:")
print(audit_df["Decision"].value_counts())