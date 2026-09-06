from pathlib import Path
import pandas as pd


DATA_DIR = Path("data/interim/CIC-IDS2018")

files = sorted(DATA_DIR.glob("*.csv"))

if not files:
    print("No cleaned CSV files found.")
    exit()


WINDOW = "60s"


for file_path in files:

    print("\n" + "=" * 70)
    print(file_path.name)
    print("=" * 70)

    window_counts = {}

    for chunk in pd.read_csv(
        file_path,
        usecols=["Timestamp", "Label"],
        chunksize=100_000
    ):

        chunk["Timestamp"] = pd.to_datetime(chunk["Timestamp"])

        # Create 60-second windows
        chunk["Window"] = chunk["Timestamp"].dt.floor(WINDOW)

        grouped = (
            chunk.groupby(["Window", "Label"])
            .size()
            .reset_index(name="Count")
        )

        for _, row in grouped.iterrows():

            key = (row["Window"], row["Label"])

            window_counts[key] = (
                window_counts.get(key, 0) + row["Count"]
            )

    window_df = pd.DataFrame(
        [
            {
                "Window": key[0],
                "Label": key[1],
                "Count": count
            }
            for key, count in window_counts.items()
        ]
    )

    window_df = window_df.sort_values("Window")

    print("\nNumber of windows:", window_df["Window"].nunique())

    print("\nFirst 10 windows:")
    print(window_df.head(10).to_string(index=False))

    print("\nLast 10 windows:")
    print(window_df.tail(10).to_string(index=False))

    print("\nLabels:")
    print(window_df["Label"].value_counts())