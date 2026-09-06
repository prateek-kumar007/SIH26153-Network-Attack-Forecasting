from pathlib import Path
import pandas as pd


DATA_DIR = Path("data/interim/CIC-IDS2018")

for file_path in sorted(DATA_DIR.glob("*.csv")):

    print("\n" + "=" * 70)
    print(file_path.name)
    print("=" * 70)

    window_stats = {}

    for chunk in pd.read_csv(
        file_path,
        usecols=["Timestamp", "Label"],
        chunksize=100_000
    ):

        chunk["Timestamp"] = pd.to_datetime(chunk["Timestamp"])

        chunk["Window"] = chunk["Timestamp"].dt.floor("60s")

        # Anything other than Benign is treated as attack traffic
        chunk["IsAttack"] = (
            chunk["Label"].str.strip().str.lower() != "benign"
        )

        grouped = (
            chunk.groupby("Window")
            .agg(
                total_flows=("Label", "size"),
                attack_flows=("IsAttack", "sum")
            )
        )

        for window, row in grouped.iterrows():

            if window not in window_stats:

                window_stats[window] = {
                    "total_flows": 0,
                    "attack_flows": 0
                }

            window_stats[window]["total_flows"] += row["total_flows"]
            window_stats[window]["attack_flows"] += row["attack_flows"]

    result = pd.DataFrame.from_dict(
        window_stats,
        orient="index"
    )

    result.index.name = "Window"

    result["benign_flows"] = (
        result["total_flows"] -
        result["attack_flows"]
    )

    result["attack_ratio"] = (
        result["attack_flows"] /
        result["total_flows"]
    )

    result["attack_ratio_percent"] = (
        result["attack_ratio"] * 100
    )

    result = result.sort_index()

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print("\nNumber of windows:", len(result))

    print("\nWindows containing attack traffic:",
          (result["attack_flows"] > 0).sum())

    print("Completely benign windows:",
          (result["attack_flows"] == 0).sum())

    print("\nHighest attack-ratio windows:")

    print(
        result.sort_values(
            "attack_ratio",
            ascending=False
        ).head(10).to_string()
    )

    print("\nFirst attack-containing windows:")

    attack_windows = result[
        result["attack_flows"] > 0
    ]

    print(
        attack_windows.head(10).to_string()
    )