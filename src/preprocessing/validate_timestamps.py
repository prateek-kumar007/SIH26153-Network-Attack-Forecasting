from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CLEAN_DIR = PROJECT_ROOT / "data" / "interim" / "CIC-IDS2018"

CHUNK_SIZE = 100_000


def parse_timestamp(series):

    series = series.astype(str).str.strip()

    iso_mask = series.str.match(
        r"^\d{4}-\d{2}-\d{2} "
    )

    result = pd.Series(
        pd.NaT,
        index=series.index,
        dtype="datetime64[ns]"
    )

    result.loc[iso_mask] = pd.to_datetime(
        series.loc[iso_mask],
        errors="coerce",
        format="%Y-%m-%d %H:%M:%S"
    )

    result.loc[~iso_mask] = pd.to_datetime(
        series.loc[~iso_mask],
        errors="coerce",
        dayfirst=True
    )

    return result


def audit_file(file_path):

    print()
    print("=" * 70)
    print(file_path.name)
    print("=" * 70)

    rows = 0
    invalid = 0

    global_min = None
    global_max = None

    for chunk in pd.read_csv(
        file_path,
        usecols=["Timestamp"],
        chunksize=CHUNK_SIZE,
        low_memory=False
    ):

        parsed = parse_timestamp(chunk["Timestamp"])

        invalid += parsed.isna().sum()

        valid = parsed.dropna()

        if len(valid) == 0:
            continue

        chunk_min = valid.min()
        chunk_max = valid.max()

        if global_min is None or chunk_min < global_min:
            global_min = chunk_min

        if global_max is None or chunk_max > global_max:
            global_max = chunk_max

        rows += len(chunk)

    print(f"Rows:              {rows:,}")
    print(f"Invalid timestamps: {invalid:,}")
    print(f"Minimum:            {global_min}")
    print(f"Maximum:            {global_max}")


def main():

    files = sorted(CLEAN_DIR.glob("*.csv"))

    print("=" * 70)
    print("CIC-IDS2018 TIMESTAMP PARSING VALIDATION")
    print("=" * 70)

    for file_path in files:
        audit_file(file_path)


if __name__ == "__main__":
    main()