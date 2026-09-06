from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CLEAN_DIR = PROJECT_ROOT / "data" / "interim" / "CIC-IDS2018"

CHUNK_SIZE = 100_000


def check_file(file_path):

    print()
    print("=" * 70)
    print(file_path.name)
    print("=" * 70)

    previous_timestamp = None
    out_of_order = 0
    rows_checked = 0

    first_timestamp = None
    last_timestamp = None

    for chunk in pd.read_csv(
        file_path,
        usecols=["Timestamp"],
        chunksize=CHUNK_SIZE,
        low_memory=False
    ):

        chunk["Timestamp"] = pd.to_datetime(
            chunk["Timestamp"],
            errors="coerce",
            dayfirst=True
        )

        chunk = chunk.dropna(subset=["Timestamp"])

        if chunk.empty:
            continue

        timestamps = chunk["Timestamp"]

        if first_timestamp is None:
            first_timestamp = timestamps.iloc[0]

        # Check ordering INSIDE the chunk
        internal_bad = (
            timestamps.diff().dt.total_seconds() < 0
        ).sum()

        out_of_order += int(internal_bad)

        # Check boundary between chunks
        if previous_timestamp is not None:

            boundary_bad = int(
                timestamps.iloc[0] < previous_timestamp
            )

            out_of_order += boundary_bad

        previous_timestamp = timestamps.iloc[-1]
        last_timestamp = timestamps.iloc[-1]

        rows_checked += len(chunk)

    print(f"Rows checked:        {rows_checked:,}")
    print(f"First timestamp:     {first_timestamp}")
    print(f"Last timestamp:      {last_timestamp}")
    print(f"Out-of-order events: {out_of_order:,}")

    if out_of_order == 0:
        print("STATUS: CHRONOLOGICALLY ORDERED")
    else:
        print("STATUS: NOT CHRONOLOGICALLY ORDERED")


def main():

    files = sorted(
        CLEAN_DIR.glob("*.csv")
    )

    print("=" * 70)
    print("CIC-IDS2018 TIMESTAMP ORDER AUDIT")
    print("=" * 70)

    print(f"Files found: {len(files)}")

    for file_path in files:
        check_file(file_path)


if __name__ == "__main__":
    main()