from pathlib import Path
import pandas as pd
import csv
import heapq
import shutil
import time


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "CIC-IDS2018"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "sorted"
    / "CIC-IDS2018"
)

TEMP_ROOT = (
    PROJECT_ROOT
    / "data"
    / "temp_sort"
)

CHUNK_SIZE = 100_000


# ============================================================
# TIMESTAMP PARSER
# ============================================================

def parse_timestamp(series):
    """
    Safely parse the mixed timestamp formats in CIC-IDS2018.

    ISO:
        YYYY-MM-DD HH:MM:SS

    Non-ISO:
        parsed with dayfirst=True
    """

    series = series.astype(str).str.strip()

    iso_mask = series.str.match(
        r"^\d{4}-\d{2}-\d{2} "
    )

    result = pd.Series(
        pd.NaT,
        index=series.index,
        dtype="datetime64[ns]"
    )

    # ISO timestamps
    result.loc[iso_mask] = pd.to_datetime(
        series.loc[iso_mask],
        errors="coerce",
        format="%Y-%m-%d %H:%M:%S"
    )

    # Non-ISO timestamps
    result.loc[~iso_mask] = pd.to_datetime(
        series.loc[~iso_mask],
        errors="coerce",
        dayfirst=True
    )

    return result


# ============================================================
# PHASE 1
# CREATE SORTED TEMPORARY CHUNKS
# ============================================================

def create_sorted_chunks(input_file, temp_dir):

    print()
    print("PHASE 1: Creating sorted chunks")
    print("-" * 70)

    chunk_files = []

    total_rows = 0
    invalid_rows = 0

    start_time = time.time()

    reader = pd.read_csv(
        input_file,
        chunksize=CHUNK_SIZE,
        low_memory=False
    )

    for chunk_number, chunk in enumerate(reader):

        total_rows += len(chunk)

        print(
            f"  Chunk {chunk_number + 1:03d} | "
            f"rows={len(chunk):,}"
        )

        # ----------------------------------------------------
        # Parse timestamps
        # ----------------------------------------------------

        chunk["Timestamp"] = parse_timestamp(
            chunk["Timestamp"]
        )

        invalid = int(
            chunk["Timestamp"].isna().sum()
        )

        if invalid > 0:
            invalid_rows += invalid

            print(
                f"      WARNING: {invalid:,} invalid timestamps"
            )

        # We already validated timestamps earlier.
        # Therefore unexpected invalid timestamps are fatal.
        if invalid > 0:
            raise RuntimeError(
                f"Invalid timestamps found in "
                f"{input_file.name}"
            )

        # ----------------------------------------------------
        # Sort chunk
        # ----------------------------------------------------

        chunk = chunk.sort_values(
            by="Timestamp",
            kind="mergesort"
        )

        # ----------------------------------------------------
        # Convert timestamp to ISO string
        #
        # This makes the final merge easy because
        # ISO timestamps sort lexicographically.
        # ----------------------------------------------------

        chunk["Timestamp"] = chunk["Timestamp"].dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        # ----------------------------------------------------
        # Save temporary sorted chunk
        # ----------------------------------------------------

        temp_file = (
            temp_dir
            / f"chunk_{chunk_number:05d}.csv"
        )

        chunk.to_csv(
            temp_file,
            index=False
        )

        chunk_files.append(temp_file)

    elapsed = time.time() - start_time

    print()
    print(f"  Total rows:       {total_rows:,}")
    print(f"  Invalid rows:     {invalid_rows:,}")
    print(f"  Sorted chunks:    {len(chunk_files)}")
    print(f"  Time:             {elapsed:.1f} seconds")

    return chunk_files, total_rows


# ============================================================
# PHASE 2
# STREAMING K-WAY MERGE
# ============================================================

def merge_sorted_chunks(
    chunk_files,
    output_file,
    expected_rows
):

    print()
    print("PHASE 2: Streaming merge")
    print("-" * 70)

    start_time = time.time()

    # --------------------------------------------------------
    # Open all temporary files.
    #
    # We only keep ONE row from each file in memory.
    # --------------------------------------------------------

    files = []
    readers = []

    try:

        for temp_file in chunk_files:

            f = open(
                temp_file,
                "r",
                newline="",
                encoding="utf-8"
            )

            reader = csv.reader(f)

            # Read header
            header = next(reader)

            files.append(f)
            readers.append(
                {
                    "reader": reader,
                    "header": header
                }
            )

        # ----------------------------------------------------
        # Verify all headers match
        # ----------------------------------------------------

        main_header = readers[0]["header"]

        for item in readers[1:]:

            if item["header"] != main_header:
                raise RuntimeError(
                    "Temporary chunk headers do not match."
                )

        # ----------------------------------------------------
        # Create output
        # ----------------------------------------------------

        output_file.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(
            output_file,
            "w",
            newline="",
            encoding="utf-8"
        ) as out:

            writer = csv.writer(out)

            writer.writerow(main_header)

            # ------------------------------------------------
            # Heap:
            #
            # timestamp
            # chunk index
            # row
            # ------------------------------------------------

            heap = []

            for index, item in enumerate(readers):

                try:
                    row = next(item["reader"])

                    timestamp = row[
                        main_header.index("Timestamp")
                    ]

                    heapq.heappush(
                        heap,
                        (
                            timestamp,
                            index,
                            row
                        )
                    )

                except StopIteration:
                    pass

            rows_written = 0
            previous_timestamp = None

            # ------------------------------------------------
            # Main merge loop
            # ------------------------------------------------

            while heap:

                timestamp, index, row = heapq.heappop(
                    heap
                )

                # --------------------------------------------
                # Chronological validation while writing
                # --------------------------------------------

                if (
                    previous_timestamp is not None
                    and timestamp < previous_timestamp
                ):
                    raise RuntimeError(
                        "Chronological merge failed."
                    )

                previous_timestamp = timestamp

                writer.writerow(row)

                rows_written += 1

                # --------------------------------------------
                # Progress
                # --------------------------------------------

                if rows_written % 500_000 == 0:

                    elapsed = time.time() - start_time

                    print(
                        f"  Merged: {rows_written:,} / "
                        f"{expected_rows:,} "
                        f"({rows_written / expected_rows * 100:.1f}%) "
                        f"| {elapsed:.1f}s"
                    )

                # --------------------------------------------
                # Get next row from same chunk
                # --------------------------------------------

                reader = readers[index]["reader"]

                try:

                    next_row = next(reader)

                    next_timestamp = next_row[
                        main_header.index("Timestamp")
                    ]

                    heapq.heappush(
                        heap,
                        (
                            next_timestamp,
                            index,
                            next_row
                        )
                    )

                except StopIteration:
                    pass

        # ----------------------------------------------------
        # Final row count validation
        # ----------------------------------------------------

        print()
        print(f"Rows written: {rows_written:,}")

        if rows_written != expected_rows:

            raise RuntimeError(
                f"ROW COUNT MISMATCH\n"
                f"Expected: {expected_rows:,}\n"
                f"Written:  {rows_written:,}"
            )

        elapsed = time.time() - start_time

        print(
            f"Merge time: {elapsed:.1f} seconds"
        )

    finally:

        for f in files:

            try:
                f.close()
            except Exception:
                pass


# ============================================================
# VALIDATE FINAL FILE
# ============================================================

def validate_sorted_file(
    output_file,
    expected_rows
):

    print()
    print("PHASE 3: Final validation")
    print("-" * 70)

    total_rows = 0

    previous_timestamp = None

    min_timestamp = None
    max_timestamp = None

    chronological = True

    with open(
        output_file,
        "r",
        newline="",
        encoding="utf-8"
    ) as f:

        reader = csv.DictReader(f)

        if "Timestamp" not in reader.fieldnames:
            raise RuntimeError(
                "Timestamp column missing."
            )

        for row in reader:

            timestamp = row["Timestamp"]

            if min_timestamp is None:
                min_timestamp = timestamp

            max_timestamp = timestamp

            if (
                previous_timestamp is not None
                and timestamp < previous_timestamp
            ):
                chronological = False

                raise RuntimeError(
                    "FINAL VALIDATION FAILED: "
                    "file is not chronological."
                )

            previous_timestamp = timestamp

            total_rows += 1

    print(f"Rows:             {total_rows:,}")
    print(f"Expected rows:    {expected_rows:,}")
    print(f"Minimum:          {min_timestamp}")
    print(f"Maximum:          {max_timestamp}")
    print(f"Chronological:    {chronological}")

    if total_rows != expected_rows:
        raise RuntimeError(
            "Final row count mismatch."
        )

    if not chronological:
        raise RuntimeError(
            "Final chronological validation failed."
        )

    print()
    print("FINAL VALIDATION PASSED")


# ============================================================
# PROCESS ONE FILE
# ============================================================

def process_file(input_file):

    print()
    print("=" * 70)
    print(f"PROCESSING: {input_file.name}")
    print("=" * 70)

    temp_dir = (
        TEMP_ROOT
        / input_file.stem
    )

    temp_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        OUTPUT_DIR
        / input_file.name
    )

    try:

        # ----------------------------------------------------
        # Phase 1
        # ----------------------------------------------------

        chunk_files, expected_rows = create_sorted_chunks(
            input_file,
            temp_dir
        )

        # ----------------------------------------------------
        # Phase 2
        # ----------------------------------------------------

        merge_sorted_chunks(
            chunk_files,
            output_file,
            expected_rows
        )

        # ----------------------------------------------------
        # Phase 3
        # ----------------------------------------------------

        validate_sorted_file(
            output_file,
            expected_rows
        )

        print()
        print(f"SUCCESS: {input_file.name}")

    finally:

        # Delete temporary chunks
        if temp_dir.exists():

            shutil.rmtree(
                temp_dir,
                ignore_errors=True
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CIC-IDS2018 EXTERNAL CHRONOLOGICAL SORT")
    print("=" * 70)

    if not INPUT_DIR.exists():

        raise FileNotFoundError(
            f"Input directory not found:\n{INPUT_DIR}"
        )

    files = sorted(
        INPUT_DIR.glob("*.csv")
    )

    if not files:

        raise FileNotFoundError(
            f"No CSV files found in:\n{INPUT_DIR}"
        )

    print()
    print(f"Input directory:  {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Files found:      {len(files)}")
    print(f"Chunk size:       {CHUNK_SIZE:,}")

    print()
    print("IMPORTANT:")
    print("  Raw files will NOT be modified.")
    print("  Interim files will NOT be modified.")
    print("  Temporary chunks will be deleted.")
    print("  Sorted files will be created separately.")

    # --------------------------------------------------------
    # Prepare directories
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    if TEMP_ROOT.exists():

        shutil.rmtree(
            TEMP_ROOT,
            ignore_errors=True
        )

    TEMP_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Process files one at a time
    # --------------------------------------------------------

    successful = 0

    for file in files:

        process_file(file)

        successful += 1

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    shutil.rmtree(
        TEMP_ROOT,
        ignore_errors=True
    )

    print()
    print("=" * 70)
    print("ALL FILES SORTED SUCCESSFULLY")
    print("=" * 70)
    print(f"Successful files: {successful}/{len(files)}")
    print(f"Output directory: {OUTPUT_DIR}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()