from pathlib import Path
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

RAW_DIR = Path("data/raw/CIC-IDS2018")
INTERIM_DIR = Path("data/interim/CIC-IDS2018")
REPORT_DIR = Path("results/cleaning")

CHUNK_SIZE = 100_000

INTERIM_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CLEAN ONE FILE
# ============================================================

def clean_file(input_path: Path):

    output_path = INTERIM_DIR / input_path.name

    total_rows = 0
    repeated_headers = 0
    invalid_timestamps = 0
    valid_rows = 0

    first_timestamp = None
    last_timestamp = None

    print("\n" + "=" * 70)
    print(f"Processing: {input_path.name}")
    print("=" * 70)

    # Remove an existing output so we don't accidentally append
    # to an old result.
    if output_path.exists():
        output_path.unlink()

    first_chunk = True

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            input_path,
            chunksize=CHUNK_SIZE,
            low_memory=False
        ),
        start=1
    ):

        original_chunk_rows = len(chunk)
        total_rows += original_chunk_rows

        # --------------------------------------------------------
        # 1. REMOVE REPEATED HEADER ROWS
        # --------------------------------------------------------

        # Repeated headers have values such as:
        # Timestamp -> Timestamp
        # Label     -> Label

        header_mask = (
            chunk["Timestamp"].astype(str).str.strip().eq("Timestamp")
            |
            chunk["Label"].astype(str).str.strip().eq("Label")
        )

        repeated_headers += header_mask.sum()

        chunk = chunk.loc[~header_mask].copy()

        # --------------------------------------------------------
        # 2. PARSE TIMESTAMP
        # --------------------------------------------------------

        chunk["Timestamp"] = pd.to_datetime(
            chunk["Timestamp"],
            dayfirst=True,
            errors="coerce"
        )

        # --------------------------------------------------------
        # 3. REMOVE INVALID TIMESTAMPS
        # --------------------------------------------------------

        # The dataset contains 14 genuine 1970 timestamps.
        # They cannot be reliably reconstructed, so we exclude
        # them from model-ready data.

        invalid_mask = (
            chunk["Timestamp"].isna()
            |
            (chunk["Timestamp"].dt.year < 2018)
        )

        invalid_timestamps += invalid_mask.sum()

        chunk = chunk.loc[~invalid_mask].copy()

        # --------------------------------------------------------
        # 4. TRACK TIME RANGE
        # --------------------------------------------------------

        if not chunk.empty:

            chunk_min = chunk["Timestamp"].min()
            chunk_max = chunk["Timestamp"].max()

            if first_timestamp is None:
                first_timestamp = chunk_min
            else:
                first_timestamp = min(first_timestamp, chunk_min)

            if last_timestamp is None:
                last_timestamp = chunk_max
            else:
                last_timestamp = max(last_timestamp, chunk_max)

        # --------------------------------------------------------
        # 5. WRITE CLEANED CHUNK
        # --------------------------------------------------------

        if not chunk.empty:

            chunk.to_csv(
                output_path,
                mode="w" if first_chunk else "a",
                header=first_chunk,
                index=False
            )

            first_chunk = False

            valid_rows += len(chunk)

        # Progress
        if chunk_number % 10 == 0:
            print(
                f"Chunk {chunk_number:>4} | "
                f"Rows read: {total_rows:,} | "
                f"Valid: {valid_rows:,}"
            )

    # ============================================================
    # FILE REPORT
    # ============================================================

    removed_rows = repeated_headers + invalid_timestamps

    report = {
        "file": input_path.name,
        "total_rows": total_rows,
        "repeated_headers_removed": repeated_headers,
        "invalid_timestamps_removed": invalid_timestamps,
        "total_rows_removed": removed_rows,
        "valid_rows": valid_rows,
        "first_timestamp": first_timestamp,
        "last_timestamp": last_timestamp
    }

    print("\nResult:")
    print(f"Total rows              : {total_rows:,}")
    print(f"Repeated headers        : {repeated_headers:,}")
    print(f"Invalid timestamps      : {invalid_timestamps:,}")
    print(f"Total removed           : {removed_rows:,}")
    print(f"Valid rows              : {valid_rows:,}")
    print(f"Time range              : {first_timestamp} → {last_timestamp}")
    print(f"Saved to                : {output_path}")

    return report


# ============================================================
# PROCESS ALL CSV FILES
# ============================================================

def main():

    csv_files = sorted(RAW_DIR.glob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in: {RAW_DIR}")
        return

    print(f"Found {len(csv_files)} CSV files.")

    reports = []

    for file_path in csv_files:

        try:
            report = clean_file(file_path)
            reports.append(report)

        except Exception as e:

            print(f"\nERROR processing {file_path.name}")
            print(e)

    # ============================================================
    # SAVE CLEANING REPORT
    # ============================================================

    report_df = pd.DataFrame(reports)

    report_path = REPORT_DIR / "cic_ids2018_cleaning_report.csv"

    report_df.to_csv(
        report_path,
        index=False
    )

    print("\n" + "=" * 70)
    print("CLEANING COMPLETE")
    print("=" * 70)

    print(report_df.to_string(index=False))

    print(f"\nReport saved to:")
    print(report_path)


if __name__ == "__main__":
    main()