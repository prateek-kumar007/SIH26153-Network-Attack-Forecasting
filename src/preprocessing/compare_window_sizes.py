from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/interim/CIC-IDS2018")
OUTPUT_DIR = Path("results/window_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOWS = ["10s", "30s", "60s"]
CHUNK_SIZE = 100_000


def analyze_file(file_path):

    print(f"\nProcessing: {file_path.name}")

    window_data = {
        window: {}
        for window in WINDOWS
    }

    for chunk in pd.read_csv(
        file_path,
        usecols=["Timestamp", "Label"],
        chunksize=CHUNK_SIZE
    ):

        chunk["Timestamp"] = pd.to_datetime(
            chunk["Timestamp"],
            errors="coerce"
        )

        chunk = chunk.dropna(subset=["Timestamp"])

        chunk["IsAttack"] = (
            chunk["Label"]
            .astype(str)
            .str.strip()
            .str.lower()
            != "benign"
        )

        for window in WINDOWS:

            temp = chunk[["Timestamp", "IsAttack"]].copy()

            temp["Window"] = temp["Timestamp"].dt.floor(window)

            grouped = temp.groupby("Window").agg(
                total_flows=("IsAttack", "size"),
                attack_flows=("IsAttack", "sum")
            )

            grouped["attack_ratio"] = (
                grouped["attack_flows"]
                / grouped["total_flows"]
            )

            for timestamp, row in grouped.iterrows():

                if timestamp not in window_data[window]:
                    window_data[window][timestamp] = {
                        "total_flows": 0,
                        "attack_flows": 0
                    }

                window_data[window][timestamp]["total_flows"] += int(
                    row["total_flows"]
                )

                window_data[window][timestamp]["attack_flows"] += int(
                    row["attack_flows"]
                )

    results = []

    for window in WINDOWS:

        records = []

        for timestamp, values in window_data[window].items():

            total = values["total_flows"]
            attack = values["attack_flows"]

            records.append({
                "Window": timestamp,
                "total_flows": total,
                "attack_flows": attack,
                "attack_ratio": attack / total
            })

        df = pd.DataFrame(records)

        if df.empty:
            continue

        results.append({
            "File": file_path.name,
            "Window_Size": window,

            "Total_Windows": len(df),

            "Attack_Containing_Windows": (
                (df["attack_flows"] > 0).sum()
            ),

            "Completely_Benign_Windows": (
                (df["attack_flows"] == 0).sum()
            ),

            "Mean_Attack_Ratio": df["attack_ratio"].mean(),

            "Median_Attack_Ratio": df["attack_ratio"].median(),

            "P95_Attack_Ratio": df["attack_ratio"].quantile(0.95),

            "Maximum_Attack_Ratio": df["attack_ratio"].max(),

            "Low_Attack_Windows_<5%": (
                (
                    (df["attack_ratio"] > 0)
                    & (df["attack_ratio"] < 0.05)
                ).sum()
            ),

            "Medium_Attack_Windows_5-50%": (
                (
                    (df["attack_ratio"] >= 0.05)
                    & (df["attack_ratio"] < 0.50)
                ).sum()
            ),

            "High_Attack_Windows_>=50%": (
                (df["attack_ratio"] >= 0.50).sum()
            ),

            "First_Attack_Window": (
                df.loc[
                    df["attack_flows"] > 0,
                    "Window"
                ].min()
            ),

            "First_Attack_Ratio": (
                df.loc[
                    df["attack_flows"] > 0,
                    "attack_ratio"
                ].iloc[0]
                if (df["attack_flows"] > 0).any()
                else None
            )
        })

    return results


all_results = []

for file_path in sorted(DATA_DIR.glob("*.csv")):

    file_results = analyze_file(file_path)

    all_results.extend(file_results)


result_df = pd.DataFrame(all_results)

output_file = (
    OUTPUT_DIR / "window_size_comparison.csv"
)

result_df.to_csv(
    output_file,
    index=False
)

print("\n" + "=" * 70)
print("WINDOW SIZE COMPARISON COMPLETE")
print("=" * 70)

print(result_df.to_string(index=False))

print(f"\nSaved to:")
print(output_file)