# src/forecasting/forecasting_target_audit_v2.py

from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

MAPPING_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "attack_type_mapping"
)

FORECAST_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "forecasting_samples"
)

RESULT_DIR = (
    BASE_DIR
    / "results"
    / "forecasting_target_audit_v2"
)

RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# EXACT CIC-IDS2018 SOURCE FILENAMES
# ============================================================

TRAIN_DAYS = {
    "Wednesday-14-02-2018",
    "Thursday-15-02-2018",
    "Friday-16-02-2018",
    "Thuesday-20-02-2018",
    "Wednesday-21-02-2018",
    "Thursday-22-02-2018",
    "Friday-23-02-2018",
}

VALIDATION_DAYS = {
    "Wednesday-28-02-2018",
}

TEST_DAYS = {
    "Thursday-01-03-2018",
    "Friday-02-03-2018",
}


# ============================================================
# FORECASTING CONFIGURATION
# ============================================================

WINDOW_SECONDS = 30

HORIZONS = [
    30,
    60,
    90,
    120,
    150,
]


# ============================================================
# TIMESTAMP PARSER
# ============================================================

def parse_timestamp(series):
    """
    Robust CIC-IDS2018 timestamp parser.

    Supports:

        02/03/2018 08:47:38

    and:

        2018-03-01 00:00:00
    """

    series = (
        series
        .astype(str)
        .str.strip()
    )

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

    # DD/MM/YYYY timestamps
    result.loc[~iso_mask] = pd.to_datetime(
        series.loc[~iso_mask],
        errors="coerce",
        dayfirst=True
    )

    return result


# ============================================================
# NORMALIZE SOURCE FILE NAME
# ============================================================

def normalize_source_file(source_series):
    """
    Converts forecasting-state filenames into the base
    CIC-IDS2018 source filename used by attack mappings.

    Example:

        Friday-02-03-2018_TrafficForML_CICFlowMeter_30s_state.csv

    becomes:

        Friday-02-03-2018
    """

    source_series = (
        source_series
        .astype(str)
        .str.strip()
    )

    # Remove forecasting state suffix
    source_series = source_series.str.replace(
        "_TrafficForML_CICFlowMeter_30s_state.csv",
        "",
        regex=False
    )

    # Remove mapping suffix if ever encountered
    source_series = source_series.str.replace(
        "_TrafficForML_CICFlowMeter_attack_type_mapping.csv",
        "",
        regex=False
    )

    return source_series


# ============================================================
# LOAD ATTACK MAPPING FILES
# ============================================================

def load_attack_mapping():

    files = sorted(
        MAPPING_DIR.glob(
            "*_attack_type_mapping.csv"
        )
    )

    if not files:
        raise FileNotFoundError(
            f"No attack mapping files found in:\n"
            f"{MAPPING_DIR}"
        )

    all_frames = []

    required_columns = {
        "Window",
        "Total_Flows",
        "Attack_Flow_Count",
        "Attack_Ratio",
        "Dominant_Attack_Type",
        "Attack_Types",
        "Number_of_Attack_Types",
    }

    print("\n" + "=" * 70)
    print("LOADING ATTACK TYPE MAPPING")
    print("=" * 70)

    for file in files:

        print(
            f"Loading: {file.name}"
        )

        df = pd.read_csv(file)

        missing = (
            required_columns
            - set(df.columns)
        )

        if missing:
            raise ValueError(
                f"{file.name} missing columns: "
                f"{missing}\n"
                f"Actual columns: "
                f"{list(df.columns)}"
            )

        # ----------------------------------------------------
        # Derive source filename
        # ----------------------------------------------------

        source_file = file.name.replace(
            "_TrafficForML_CICFlowMeter_attack_type_mapping.csv",
            ""
        )

        df["Source_File"] = (
            source_file
        )

        # ----------------------------------------------------
        # Parse Window
        # ----------------------------------------------------

        df["Window"] = pd.to_datetime(
            df["Window"],
            errors="coerce"
        )

        # ----------------------------------------------------
        # Attack flag
        # ----------------------------------------------------

        df["Attack_Flow_Count"] = pd.to_numeric(
            df["Attack_Flow_Count"],
            errors="coerce"
        ).fillna(0)

        df["Attack_Ratio"] = pd.to_numeric(
            df["Attack_Ratio"],
            errors="coerce"
        )

        df["Is_Attack"] = (
            df["Attack_Flow_Count"]
            > 0
        )

        # ----------------------------------------------------
        # Keep required columns
        # ----------------------------------------------------

        df = df[
            [
                "Source_File",
                "Window",
                "Total_Flows",
                "Attack_Flow_Count",
                "Attack_Ratio",
                "Dominant_Attack_Type",
                "Attack_Types",
                "Number_of_Attack_Types",
                "Is_Attack",
            ]
        ]

        all_frames.append(df)

    mapping = pd.concat(
        all_frames,
        ignore_index=True
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    invalid_timestamp = (
        mapping["Window"]
        .isna()
        .sum()
    )

    if invalid_timestamp > 0:
        print(
            f"WARNING: "
            f"{invalid_timestamp} invalid mapping timestamps"
        )

    duplicate_keys = mapping.duplicated(
        subset=[
            "Source_File",
            "Window",
        ]
    ).sum()

    if duplicate_keys > 0:
        raise ValueError(
            f"Duplicate mapping keys detected: "
            f"{duplicate_keys}"
        )

    mapping = mapping.sort_values(
        [
            "Source_File",
            "Window",
        ]
    ).reset_index(drop=True)

    print(
        f"\nMapping files loaded: "
        f"{len(files)}"
    )

    print(
        f"Total mapping rows: "
        f"{len(mapping):,}"
    )

    print(
        f"Attack windows: "
        f"{mapping['Is_Attack'].sum():,}"
    )

    print(
        f"Benign windows: "
        f"{(~mapping['Is_Attack']).sum():,}"
    )

    return mapping


# ============================================================
# LOAD FORECASTING SAMPLES
# ============================================================

def load_forecasting_samples():

    file = (
        FORECAST_DIR
        / "all_forecasting_samples.csv"
    )

    if not file.exists():
        raise FileNotFoundError(
            f"Forecasting sample file not found:\n"
            f"{file}"
        )

    print("\n" + "=" * 70)
    print("LOADING FORECASTING SAMPLES")
    print("=" * 70)

    df = pd.read_csv(file)

    required = {
        "Source_File",
        "Target_Window",
        "Target_Attack",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"Forecasting samples missing columns: "
            f"{missing}"
        )

    # --------------------------------------------------------
    # IMPORTANT:
    # Normalize forecasting source filenames
    # --------------------------------------------------------

    original_source_names = (
        df["Source_File"]
        .astype(str)
        .copy()
    )

    df["Source_File"] = (
        normalize_source_file(
            df["Source_File"]
        )
    )

    # --------------------------------------------------------
    # Show normalization example
    # --------------------------------------------------------

    print(
        "\nSource filename normalization:"
    )

    example_df = pd.DataFrame(
        {
            "Original":
                original_source_names.head(3),

            "Normalized":
                df["Source_File"].head(3),
        }
    )

    print(
        example_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Parse Target_Window
    # --------------------------------------------------------

    df["Target_Window"] = (
        parse_timestamp(
            df["Target_Window"]
        )
    )

    invalid_timestamp = (
        df["Target_Window"]
        .isna()
        .sum()
    )

    if invalid_timestamp > 0:
        raise ValueError(
            f"Invalid Target_Window timestamps: "
            f"{invalid_timestamp}"
        )

    # --------------------------------------------------------
    # Target_Attack
    # --------------------------------------------------------

    df["Target_Attack"] = (
        pd.to_numeric(
            df["Target_Attack"],
            errors="coerce"
        )
        .fillna(0)
        .astype(int)
    )

    # --------------------------------------------------------
    # Validate binary target
    # --------------------------------------------------------

    invalid_target = (
        ~df["Target_Attack"]
        .isin([0, 1])
    ).sum()

    if invalid_target > 0:
        raise ValueError(
            f"Invalid Target_Attack values: "
            f"{invalid_target}"
        )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    df = df.sort_values(
        [
            "Source_File",
            "Target_Window",
        ]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Duplicate check
    # --------------------------------------------------------

    duplicate_keys = df.duplicated(
        subset=[
            "Source_File",
            "Target_Window",
        ]
    ).sum()

    if duplicate_keys > 0:
        raise ValueError(
            f"Duplicate forecasting keys: "
            f"{duplicate_keys}"
        )

    print(
        f"\nForecasting samples: "
        f"{len(df):,}"
    )

    print(
        f"Unique source files: "
        f"{df['Source_File'].nunique()}"
    )

    print(
        "\nSource files:"
    )

    for source in sorted(
        df["Source_File"].unique()
    ):
        print(
            f"  - {source}"
        )

    return df


# ============================================================
# SPLIT ASSIGNMENT
# ============================================================

def assign_split(source_file):

    if source_file in TRAIN_DAYS:
        return "TRAIN"

    if source_file in VALIDATION_DAYS:
        return "VALIDATION"

    if source_file in TEST_DAYS:
        return "TEST"

    return "UNKNOWN"


# ============================================================
# JOIN FORECASTING + ATTACK MAPPING
# ============================================================

def join_forecasting_and_mapping(
    forecast_df,
    mapping_df
):

    print("\n" + "=" * 70)
    print("JOINING FORECASTING SAMPLES WITH ATTACK MAPPING")
    print("=" * 70)

    forecast = forecast_df.copy()
    mapping = mapping_df.copy()

    # --------------------------------------------------------
    # Normalize source names again defensively
    # --------------------------------------------------------

    forecast["Source_File"] = (
        normalize_source_file(
            forecast["Source_File"]
        )
    )

    mapping["Source_File"] = (
        normalize_source_file(
            mapping["Source_File"]
        )
    )

    # --------------------------------------------------------
    # Normalize timestamps to 30-second boundaries
    # --------------------------------------------------------

    forecast["Join_Window"] = (
        forecast["Target_Window"]
        .dt.floor(
            f"{WINDOW_SECONDS}s"
        )
    )

    mapping["Join_Window"] = (
        mapping["Window"]
        .dt.floor(
            f"{WINDOW_SECONDS}s"
        )
    )

    # --------------------------------------------------------
    # Duplicate mapping keys
    # --------------------------------------------------------

    duplicate_mapping = (
        mapping.duplicated(
            [
                "Source_File",
                "Join_Window",
            ]
        ).sum()
    )

    if duplicate_mapping:
        raise ValueError(
            f"Duplicate mapping join keys: "
            f"{duplicate_mapping}"
        )

    # --------------------------------------------------------
    # JOIN
    # --------------------------------------------------------

    merged = forecast.merge(
        mapping[
            [
                "Source_File",
                "Join_Window",
                "Attack_Flow_Count",
                "Attack_Ratio",
                "Dominant_Attack_Type",
                "Attack_Types",
                "Number_of_Attack_Types",
                "Is_Attack",
            ]
        ],
        on=[
            "Source_File",
            "Join_Window",
        ],
        how="left",
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # Match audit
    # --------------------------------------------------------

    matched = (
        merged[
            "Attack_Flow_Count"
        ]
        .notna()
        .sum()
    )

    unmatched = (
        len(merged)
        - matched
    )

    print(
        f"Forecasting rows: "
        f"{len(merged):,}"
    )

    print(
        f"Matched mapping rows: "
        f"{matched:,}"
    )

    print(
        f"Unmatched mapping rows: "
        f"{unmatched:,}"
    )

    if unmatched > 0:

        print(
            "\nUnmatched examples:"
        )

        print(
            merged.loc[
                merged[
                    "Attack_Flow_Count"
                ].isna(),
                [
                    "Source_File",
                    "Target_Window",
                ],
            ]
            .head(20)
            .to_string(index=False)
        )

        raise ValueError(
            "Some forecasting samples could not "
            "be matched with attack mapping."
        )

    # --------------------------------------------------------
    # Mapping target
    # --------------------------------------------------------

    merged[
        "Mapping_Target_Attack"
    ] = (
        merged["Is_Attack"]
        .astype(int)
    )

    # --------------------------------------------------------
    # Compare existing Target_Attack
    # --------------------------------------------------------

    mismatches = (
        merged["Target_Attack"]
        !=
        merged[
            "Mapping_Target_Attack"
        ]
    ).sum()

    print(
        f"Target/mapping mismatches: "
        f"{mismatches:,}"
    )

    if mismatches > 0:

        print(
            "\nWARNING: "
            "Target_Attack differs from "
            "attack mapping."
        )

    # --------------------------------------------------------
    # Split assignment
    # --------------------------------------------------------

    merged["Split"] = (
        merged["Source_File"]
        .apply(assign_split)
    )

    unknown = (
        merged["Split"]
        == "UNKNOWN"
    ).sum()

    if unknown > 0:

        print(
            f"WARNING: "
            f"{unknown} rows have UNKNOWN split."
        )

        print(
            merged.loc[
                merged["Split"] == "UNKNOWN",
                "Source_File",
            ]
            .drop_duplicates()
            .to_string(index=False)
        )

    return merged


# ============================================================
# CONTIGUOUS SEGMENTS
# ============================================================

def add_contiguous_segments(df):

    print("\n" + "=" * 70)
    print("BUILDING CONTIGUOUS TIME SEGMENTS")
    print("=" * 70)

    df = df.sort_values(
        [
            "Source_File",
            "Target_Window",
        ]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Previous window
    # --------------------------------------------------------

    df["Previous_Window"] = (
        df.groupby(
            "Source_File"
        )["Target_Window"]
        .shift(1)
    )

    # --------------------------------------------------------
    # Gap
    # --------------------------------------------------------

    df["Gap_Seconds"] = (
        df["Target_Window"]
        - df["Previous_Window"]
    ).dt.total_seconds()

    # --------------------------------------------------------
    # Segment starts:
    #
    # first row
    # OR gap != 30 sec
    # --------------------------------------------------------

    df["New_Segment"] = (
        df["Previous_Window"].isna()
        |
        (
            df["Gap_Seconds"]
            != WINDOW_SECONDS
        )
    )

    # --------------------------------------------------------
    # Segment number
    # --------------------------------------------------------

    df["Segment_Number"] = (
        df.groupby(
            "Source_File"
        )["New_Segment"]
        .cumsum()
    )

    df["Segment_ID"] = (
        df["Source_File"]
        + "__SEG_"
        + df["Segment_Number"]
        .astype(str)
    )

    # --------------------------------------------------------
    # Segment summary
    # --------------------------------------------------------

    segment_summary = (
        df.groupby(
            [
                "Source_File",
                "Segment_ID",
            ]
        )
        .agg(
            Start_Window=(
                "Target_Window",
                "min",
            ),
            End_Window=(
                "Target_Window",
                "max",
            ),
            Windows=(
                "Target_Window",
                "size",
            ),
        )
        .reset_index()
    )

    segment_summary[
        "Duration_Seconds"
    ] = (
        segment_summary["Windows"]
        - 1
    ) * WINDOW_SECONDS

    print(
        f"Total segments: "
        f"{len(segment_summary):,}"
    )

    print(
        "\nSegments per source:"
    )

    print(
        segment_summary
        .groupby(
            "Source_File"
        )["Segment_ID"]
        .count()
        .to_string()
    )

    segment_summary.to_csv(
        RESULT_DIR
        / "segment_summary.csv",
        index=False
    )

    return (
        df,
        segment_summary,
    )


# ============================================================
# BUILD ATTACK EPISODES
# ============================================================

def create_attack_episodes(df):

    print("\n" + "=" * 70)
    print("BUILDING ATTACK EPISODES")
    print("=" * 70)

    episodes = []

    for source_file, group in df.groupby(
        "Source_File"
    ):

        group = (
            group
            .sort_values(
                "Target_Window"
            )
            .reset_index(drop=True)
        )

        group["Attack_Flag"] = (
            group[
                "Mapping_Target_Attack"
            ]
            .astype(int)
        )

        # ----------------------------------------------------
        # Previous attack
        # ----------------------------------------------------

        group["Previous_Attack"] = (
            group[
                "Attack_Flag"
            ]
            .shift(1)
            .fillna(0)
            .astype(int)
        )

        group["Previous_Segment"] = (
            group[
                "Segment_ID"
            ]
            .shift(1)
        )

        # ----------------------------------------------------
        # Episode starts:
        #
        # current = attack
        # previous = benign
        #
        # OR new segment
        # ----------------------------------------------------

        group["New_Episode"] = (
            (
                group["Attack_Flag"]
                == 1
            )
            &
            (
                (
                    group[
                        "Previous_Attack"
                    ]
                    == 0
                )
                |
                (
                    group["Segment_ID"]
                    !=
                    group[
                        "Previous_Segment"
                    ]
                )
            )
        )

        group["Episode_Number"] = (
            group["New_Episode"]
            .cumsum()
        )

        attack_rows = group[
            group["Attack_Flag"] == 1
        ]

        for (
            episode_id,
            episode
        ) in attack_rows.groupby(
            "Episode_Number"
        ):

            start = (
                episode[
                    "Target_Window"
                ].min()
            )

            end = (
                episode[
                    "Target_Window"
                ].max()
            )

            duration = (
                end - start
            ).total_seconds() + WINDOW_SECONDS

            split_values = (
                episode["Split"]
                .dropna()
                .unique()
                .tolist()
            )

            # ------------------------------------------------
            # Attack types
            # ------------------------------------------------

            attack_types = set()

            for value in (
                episode[
                    "Attack_Types"
                ]
                .dropna()
            ):

                value = str(
                    value
                ).strip()

                if not value:
                    continue

                for attack_type in (
                    value.split(",")
                ):

                    attack_type = (
                        attack_type.strip()
                    )

                    if attack_type:
                        attack_types.add(
                            attack_type
                        )

            dominant_types = (
                episode[
                    "Dominant_Attack_Type"
                ]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            episodes.append(
                {
                    "Source_File":
                        source_file,

                    "Segment_ID":
                        episode[
                            "Segment_ID"
                        ].iloc[0],

                    "Episode_ID":
                        f"{source_file}"
                        f"__EP_{episode_id}",

                    "Split":
                        (
                            split_values[0]
                            if split_values
                            else "UNKNOWN"
                        ),

                    "Start_Window":
                        start,

                    "End_Window":
                        end,

                    "Duration_Seconds":
                        duration,

                    "Duration_Minutes":
                        duration / 60.0,

                    "Attack_Windows":
                        len(episode),

                    "Dominant_Attack_Types":
                        "|".join(
                            dominant_types
                        ),

                    "Attack_Types":
                        "|".join(
                            sorted(
                                attack_types
                            )
                        ),
                }
            )

    episodes_df = pd.DataFrame(
        episodes
    )

    if episodes_df.empty:

        print(
            "No attack episodes found."
        )

        return episodes_df

    episodes_df = (
        episodes_df
        .sort_values(
            [
                "Start_Window",
                "Source_File",
            ]
        )
        .reset_index(drop=True)
    )

    print(
        f"Total attack episodes: "
        f"{len(episodes_df):,}"
    )

    print(
        "\nEpisode duration statistics:"
    )

    print(
        episodes_df[
            "Duration_Seconds"
        ].describe()
    )

    episodes_df.to_csv(
        RESULT_DIR
        / "attack_episodes.csv",
        index=False
    )

    return episodes_df


# ============================================================
# ATTACK ONSET AUDIT
# ============================================================

def audit_attack_onsets(df):

    print("\n" + "=" * 70)
    print("ATTACK ONSET AUDIT")
    print("=" * 70)

    rows = []

    for source_file, group in df.groupby(
        "Source_File"
    ):

        group = (
            group
            .sort_values(
                "Target_Window"
            )
            .reset_index(drop=True)
        )

        group["Previous_Attack"] = (
            group[
                "Mapping_Target_Attack"
            ]
            .shift(1)
            .fillna(0)
            .astype(int)
        )

        group["Previous_Window"] = (
            group[
                "Target_Window"
            ]
            .shift(1)
        )

        group["Gap_Seconds"] = (
            group[
                "Target_Window"
            ]
            -
            group[
                "Previous_Window"
            ]
        ).dt.total_seconds()

        # ----------------------------------------------------
        # Exact onset:
        #
        # previous = benign
        # current = attack
        # contiguous 30-second windows
        # ----------------------------------------------------

        group["Is_Onset"] = (
            (
                group[
                    "Previous_Attack"
                ] == 0
            )
            &
            (
                group[
                    "Mapping_Target_Attack"
                ] == 1
            )
            &
            (
                group[
                    "Gap_Seconds"
                ] == WINDOW_SECONDS
            )
        )

        onsets = group[
            group["Is_Onset"]
        ]

        for _, row in onsets.iterrows():

            rows.append(
                {
                    "Source_File":
                        source_file,

                    "Split":
                        row["Split"],

                    "Onset_Window":
                        row[
                            "Target_Window"
                        ],

                    "Attack_Type":
                        row[
                            "Dominant_Attack_Type"
                        ],

                    "Attack_Types":
                        row[
                            "Attack_Types"
                        ],

                    "Attack_Ratio":
                        row[
                            "Attack_Ratio"
                        ],
                }
            )

    onset_df = pd.DataFrame(
        rows
    )

    if onset_df.empty:

        print(
            "No valid attack onsets found."
        )

        return onset_df

    print(
        f"Total valid attack onsets: "
        f"{len(onset_df):,}"
    )

    print(
        "\nOnsets by split:"
    )

    print(
        onset_df[
            "Split"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nOnsets by attack type:"
    )

    print(
        onset_df[
            "Attack_Type"
        ]
        .value_counts()
        .to_string()
    )

    onset_df.to_csv(
        RESULT_DIR
        / "attack_onsets.csv",
        index=False
    )

    # --------------------------------------------------------
    # Onset availability by split
    # --------------------------------------------------------

    horizon_rows = []

    for split in [
        "TRAIN",
        "VALIDATION",
        "TEST",
    ]:

        split_onsets = (
            onset_df[
                onset_df["Split"]
                == split
            ]
        )

        for horizon in HORIZONS:

            horizon_rows.append(
                {
                    "Split":
                        split,

                    "Horizon_Seconds":
                        horizon,

                    "Onset_Count":
                        len(split_onsets),
                }
            )

    horizon_df = pd.DataFrame(
        horizon_rows
    )

    horizon_df.to_csv(
        RESULT_DIR
        / "onset_horizon_availability.csv",
        index=False
    )

    return onset_df


# ============================================================
# ATTACK TYPE AUDIT
# ============================================================

def audit_attack_types(df):

    print("\n" + "=" * 70)
    print("ATTACK TYPE AUDIT")
    print("=" * 70)

    attack_df = df[
        df[
            "Mapping_Target_Attack"
        ] == 1
    ].copy()

    attack_df[
        "Attack_Type"
    ] = (
        attack_df[
            "Dominant_Attack_Type"
        ]
        .fillna("UNKNOWN")
        .astype(str)
    )

    # --------------------------------------------------------
    # Attack types by split
    # --------------------------------------------------------

    for split in [
        "TRAIN",
        "VALIDATION",
        "TEST",
    ]:

        subset = attack_df[
            attack_df["Split"]
            == split
        ]

        print(
            f"\n{split} attack types:"
        )

        if subset.empty:
            print("None")
            continue

        print(
            subset[
                "Attack_Type"
            ]
            .value_counts()
            .to_string()
        )

    # --------------------------------------------------------
    # Sets
    # --------------------------------------------------------

    train_types = set(
        attack_df[
            attack_df["Split"]
            == "TRAIN"
        ]["Attack_Type"]
    )

    validation_types = set(
        attack_df[
            attack_df["Split"]
            == "VALIDATION"
        ]["Attack_Type"]
    )

    test_types = set(
        attack_df[
            attack_df["Split"]
            == "TEST"
        ]["Attack_Type"]
    )

    # --------------------------------------------------------
    # Unseen test types
    # --------------------------------------------------------

    unseen_test_vs_train = (
        test_types
        - train_types
    )

    unseen_test_vs_train_valid = (
        test_types
        - train_types
        - validation_types
    )

    print(
        "\nTEST attack types unseen in TRAIN:"
    )

    if unseen_test_vs_train:
        for value in sorted(
            unseen_test_vs_train
        ):
            print(
                f"  - {value}"
            )
    else:
        print(
            "  None"
        )

    print(
        "\nTEST attack types unseen in "
        "TRAIN + VALIDATION:"
    )

    if unseen_test_vs_train_valid:
        for value in sorted(
            unseen_test_vs_train_valid
        ):
            print(
                f"  - {value}"
            )
    else:
        print(
            "  None"
        )

    # --------------------------------------------------------
    # Save audit
    # --------------------------------------------------------

    unseen_rows = []

    for attack_type in sorted(
        test_types
    ):

        unseen_rows.append(
            {
                "Attack_Type":
                    attack_type,

                "In_Train":
                    attack_type
                    in train_types,

                "In_Validation":
                    attack_type
                    in validation_types,

                "Unseen_in_Train":
                    attack_type
                    not in train_types,

                "Unseen_in_Train_Validation":
                    attack_type
                    not in (
                        train_types
                        | validation_types
                    ),
            }
        )

    unseen_df = pd.DataFrame(
        unseen_rows
    )

    unseen_df.to_csv(
        RESULT_DIR
        / "unseen_test_attack_types.csv",
        index=False
    )

    return (
        train_types,
        validation_types,
        test_types,
    )


# ============================================================
# HEURISTIC ATTACK FAMILY MAPPING
# ============================================================

def map_attack_family(attack_type):

    if pd.isna(attack_type):
        return "Unknown"

    value = str(
        attack_type
    ).lower()

    # --------------------------------------------------------
    # Brute Force
    # --------------------------------------------------------

    if (
        "brute" in value
        or "ftp" in value
        or "ssh" in value
    ):
        return "Brute Force"

    # --------------------------------------------------------
    # DDoS
    # --------------------------------------------------------

    if (
        "ddos" in value
        or "loic" in value
        or "hoic" in value
    ):
        return "DDoS"

    # --------------------------------------------------------
    # DoS
    # --------------------------------------------------------

    if (
        value.startswith("dos")
        or "slowhttp" in value
        or "goldeneye" in value
        or "hulk" in value
        or "slowloris" in value
    ):
        return "DoS"

    # --------------------------------------------------------
    # Web attacks
    # --------------------------------------------------------

    if (
        "sql injection" in value
        or "xss" in value
        or "web" in value
    ):
        return "Web Attack"

    # --------------------------------------------------------
    # Infiltration
    # --------------------------------------------------------

    if (
        "infilteration" in value
    ):
        return "Infiltration"

    # --------------------------------------------------------
    # Bot
    # --------------------------------------------------------

    if "bot" in value:
        return "Bot"

    return "Other"


# ============================================================
# ATTACK FAMILY AUDIT
# ============================================================

def audit_attack_families(df):

    print("\n" + "=" * 70)
    print("ATTACK FAMILY AUDIT")
    print("=" * 70)

    attack_df = df[
        df[
            "Mapping_Target_Attack"
        ] == 1
    ].copy()

    attack_df[
        "Attack_Family"
    ] = (
        attack_df[
            "Dominant_Attack_Type"
        ]
        .apply(
            map_attack_family
        )
    )

    for split in [
        "TRAIN",
        "VALIDATION",
        "TEST",
    ]:

        subset = attack_df[
            attack_df["Split"]
            == split
        ]

        print(
            f"\n{split} attack families:"
        )

        if subset.empty:
            print("None")
            continue

        print(
            subset[
                "Attack_Family"
            ]
            .value_counts()
            .to_string()
        )

    train_families = set(
        attack_df[
            attack_df["Split"]
            == "TRAIN"
        ]["Attack_Family"]
    )

    validation_families = set(
        attack_df[
            attack_df["Split"]
            == "VALIDATION"
        ]["Attack_Family"]
    )

    test_families = set(
        attack_df[
            attack_df["Split"]
            == "TEST"
        ]["Attack_Family"]
    )

    unseen_test_train = (
        test_families
        - train_families
    )

    unseen_test_train_valid = (
        test_families
        - train_families
        - validation_families
    )

    print(
        "\nTEST families unseen in TRAIN:"
    )

    for family in sorted(
        unseen_test_train
    ):
        print(
            f"  - {family}"
        )

    print(
        "\nTEST families unseen in "
        "TRAIN + VALIDATION:"
    )

    for family in sorted(
        unseen_test_train_valid
    ):
        print(
            f"  - {family}"
        )

    attack_df.to_csv(
        RESULT_DIR
        / "attack_family_mapping.csv",
        index=False
    )

    return attack_df


# ============================================================
# TARGET PERSISTENCE AUDIT
# ============================================================

def audit_persistence(df):

    print("\n" + "=" * 70)
    print("TARGET PERSISTENCE AUDIT")
    print("=" * 70)

    rows = []

    for split in [
        "TRAIN",
        "VALIDATION",
        "TEST",
    ]:

        subset = df[
            df["Split"]
            == split
        ].copy()

        subset = (
            subset
            .sort_values(
                [
                    "Source_File",
                    "Target_Window",
                ]
            )
        )

        # ----------------------------------------------------
        # Previous attack
        # ----------------------------------------------------

        subset[
            "Previous_Attack"
        ] = (
            subset
            .groupby(
                "Source_File"
            )[
                "Mapping_Target_Attack"
            ]
            .shift(1)
        )

        # ----------------------------------------------------
        # Previous window
        # ----------------------------------------------------

        subset[
            "Previous_Window"
        ] = (
            subset
            .groupby(
                "Source_File"
            )[
                "Target_Window"
            ]
            .shift(1)
        )

        # ----------------------------------------------------
        # Gap
        # ----------------------------------------------------

        subset["Gap"] = (
            subset[
                "Target_Window"
            ]
            -
            subset[
                "Previous_Window"
            ]
        ).dt.total_seconds()

        # Only contiguous transitions
        transitions = subset[
            subset["Gap"]
            == WINDOW_SECONDS
        ].copy()

        if transitions.empty:
            continue

        # ----------------------------------------------------
        # Attack persistence
        # ----------------------------------------------------

        previous_attack = (
            transitions[
                "Previous_Attack"
            ] == 1
        )

        attack_previous = (
            transitions[
                previous_attack
            ]
        )

        persistence = (
            attack_previous[
                "Mapping_Target_Attack"
            ].mean()
            if len(attack_previous)
            else np.nan
        )

        # ----------------------------------------------------
        # Transition counts
        # ----------------------------------------------------

        zero_to_zero = (
            (
                transitions[
                    "Previous_Attack"
                ] == 0
            )
            &
            (
                transitions[
                    "Mapping_Target_Attack"
                ] == 0
            )
        ).sum()

        zero_to_one = (
            (
                transitions[
                    "Previous_Attack"
                ] == 0
            )
            &
            (
                transitions[
                    "Mapping_Target_Attack"
                ] == 1
            )
        ).sum()

        one_to_zero = (
            (
                transitions[
                    "Previous_Attack"
                ] == 1
            )
            &
            (
                transitions[
                    "Mapping_Target_Attack"
                ] == 0
            )
        ).sum()

        one_to_one = (
            (
                transitions[
                    "Previous_Attack"
                ] == 1
            )
            &
            (
                transitions[
                    "Mapping_Target_Attack"
                ] == 1
            )
        ).sum()

        rows.append(
            {
                "Split":
                    split,

                "Valid_Transitions":
                    len(transitions),

                "0_to_0":
                    zero_to_zero,

                "0_to_1":
                    zero_to_one,

                "1_to_0":
                    one_to_zero,

                "1_to_1":
                    one_to_one,

                "Attack_Persistence":
                    persistence,
            }
        )

    persistence_df = pd.DataFrame(
        rows
    )

    print(
        persistence_df.to_string(
            index=False
        )
    )

    persistence_df.to_csv(
        RESULT_DIR
        / "persistence_audit.csv",
        index=False
    )

    return persistence_df


# ============================================================
# SUMMARY REPORT
# ============================================================

def create_summary_report(
    merged,
    episodes,
    onsets,
    persistence,
):

    report_path = (
        RESULT_DIR
        / "audit_summary.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "SIH26153 FORECASTING TARGET AUDIT V2\n"
        )

        f.write(
            "=" * 70
            + "\n\n"
        )

        # ----------------------------------------------------
        # Basic dataset
        # ----------------------------------------------------

        total_samples = len(
            merged
        )

        total_attack = int(
            merged[
                "Mapping_Target_Attack"
            ].sum()
        )

        total_benign = (
            total_samples
            - total_attack
        )

        f.write(
            f"Forecasting samples: "
            f"{total_samples:,}\n"
        )

        f.write(
            f"Attack samples: "
            f"{total_attack:,}\n"
        )

        f.write(
            f"Benign samples: "
            f"{total_benign:,}\n\n"
        )

        # ----------------------------------------------------
        # Split counts
        # ----------------------------------------------------

        f.write(
            "SPLIT COUNTS\n"
        )

        f.write(
            "-" * 70
            + "\n"
        )

        split_summary = (
            merged
            .groupby(
                "Split"
            )[
                "Mapping_Target_Attack"
            ]
            .agg(
                Samples="size",
                Attack="sum",
            )
        )

        split_summary[
            "Benign"
        ] = (
            split_summary[
                "Samples"
            ]
            -
            split_summary[
                "Attack"
            ]
        )

        split_summary[
            "Attack_Rate"
        ] = (
            split_summary[
                "Attack"
            ]
            /
            split_summary[
                "Samples"
            ]
        )

        f.write(
            split_summary.to_string()
        )

        f.write(
            "\n\n"
        )

        # ----------------------------------------------------
        # Attack episodes
        # ----------------------------------------------------

        f.write(
            "ATTACK EPISODES\n"
        )

        f.write(
            "-" * 70
            + "\n"
        )

        if not episodes.empty:

            f.write(
                f"Total episodes: "
                f"{len(episodes):,}\n"
            )

            f.write(
                f"Mean duration: "
                f"{episodes['Duration_Seconds'].mean():.2f} sec\n"
            )

            f.write(
                f"Median duration: "
                f"{episodes['Duration_Seconds'].median():.2f} sec\n"
            )

            f.write(
                f"Minimum duration: "
                f"{episodes['Duration_Seconds'].min():.2f} sec\n"
            )

            f.write(
                f"Maximum duration: "
                f"{episodes['Duration_Seconds'].max():.2f} sec\n"
            )

        # ----------------------------------------------------
        # Onsets
        # ----------------------------------------------------

        f.write(
            "\n\nATTACK ONSETS\n"
        )

        f.write(
            "-" * 70
            + "\n"
        )

        f.write(
            f"Total valid onsets: "
            f"{len(onsets):,}\n"
        )

        if not onsets.empty:

            f.write(
                "\nOnsets by split:\n"
            )

            f.write(
                onsets[
                    "Split"
                ]
                .value_counts()
                .to_string()
            )

            f.write(
                "\n\nOnsets by attack type:\n"
            )

            f.write(
                onsets[
                    "Attack_Type"
                ]
                .value_counts()
                .to_string()
            )

        # ----------------------------------------------------
        # Persistence
        # ----------------------------------------------------

        f.write(
            "\n\nPERSISTENCE\n"
        )

        f.write(
            "-" * 70
            + "\n"
        )

        if not persistence.empty:

            f.write(
                persistence.to_string(
                    index=False
                )
            )

        # ----------------------------------------------------
        # Scientific interpretation
        # ----------------------------------------------------

        f.write(
            "\n\n"
        )

        f.write(
            "SCIENTIFIC INTERPRETATION\n"
        )

        f.write(
            "-" * 70
            + "\n"
        )

        f.write(
            "The current binary target represents attack presence "
            "in a future 30-second window. Because attack episodes "
            "can persist across consecutive windows, a persistence "
            "baseline can achieve high F1 without demonstrating "
            "true pre-attack forecasting ability.\n\n"
        )

        f.write(
            "Therefore, high persistence-baseline F1 must not be "
            "reported as evidence of successful AI forecasting.\n\n"
        )

        f.write(
            "Exact attack onset events should be evaluated separately "
            "from general future attack presence. A future attack "
            "presence label at +60, +90, +120 or +150 seconds is not "
            "automatically an independent attack onset.\n\n"
        )

        f.write(
            "Attack-type and attack-family novelty must also be "
            "interpreted carefully. The family mapping used by this "
            "audit is heuristic and is not an official MITRE taxonomy.\n"
        )

    print(
        "\nSummary report saved to:"
    )

    print(
        report_path
    )


# ============================================================
# FINAL VALIDATION
# ============================================================

def final_validation(merged):

    print("\n" + "=" * 70)
    print("FINAL VALIDATION")
    print("=" * 70)

    checks = []

    # --------------------------------------------------------
    # 1. Every forecast row matched
    # --------------------------------------------------------

    checks.append(
        (
            "All mapping rows matched",
            merged[
                "Attack_Flow_Count"
            ].notna().all(),
        )
    )

    # --------------------------------------------------------
    # 2. No duplicate keys
    # --------------------------------------------------------

    checks.append(
        (
            "No duplicate forecast keys",
            not merged.duplicated(
                [
                    "Source_File",
                    "Target_Window",
                ]
            ).any(),
        )
    )

    # --------------------------------------------------------
    # 3. No unknown splits
    # --------------------------------------------------------

    checks.append(
        (
            "No UNKNOWN split",
            (
                merged["Split"]
                != "UNKNOWN"
            ).all(),
        )
    )

    # --------------------------------------------------------
    # 4. Target matches mapping
    # --------------------------------------------------------

    checks.append(
        (
            "Target matches mapping",
            (
                merged[
                    "Target_Attack"
                ]
                ==
                merged[
                    "Mapping_Target_Attack"
                ]
            ).all(),
        )
    )

    # --------------------------------------------------------
    # 5. Chronological order
    # --------------------------------------------------------

    chronological = True

    for _, group in merged.groupby(
        "Source_File"
    ):

        if not group[
            "Target_Window"
        ].is_monotonic_increasing:

            chronological = False
            break

    checks.append(
        (
            "Chronological order",
            chronological,
        )
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    passed = True

    for name, result in checks:

        status = (
            "PASS"
            if result
            else "FAIL"
        )

        print(
            f"{status}: {name}"
        )

        if not result:
            passed = False

    print(
        "\n" + "=" * 70
    )

    if passed:

        print(
            "ALL CORE AUDIT CHECKS PASSED"
        )

    else:

        print(
            "ONE OR MORE AUDIT CHECKS FAILED"
        )

    print(
        "=" * 70
    )

    return passed


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("SIH26153 FORECASTING TARGET AUDIT V2")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Load mapping
    # --------------------------------------------------------

    mapping = (
        load_attack_mapping()
    )

    # --------------------------------------------------------
    # 2. Load forecasting samples
    # --------------------------------------------------------

    forecast = (
        load_forecasting_samples()
    )

    # --------------------------------------------------------
    # 3. Join
    # --------------------------------------------------------

    merged = (
        join_forecasting_and_mapping(
            forecast,
            mapping,
        )
    )

    # --------------------------------------------------------
    # 4. Build contiguous segments
    # --------------------------------------------------------

    (
        merged,
        segment_summary,
    ) = add_contiguous_segments(
        merged
    )

    # --------------------------------------------------------
    # 5. Build attack episodes
    # --------------------------------------------------------

    episodes = (
        create_attack_episodes(
            merged
        )
    )

    # --------------------------------------------------------
    # 6. Audit exact attack onsets
    # --------------------------------------------------------

    onsets = (
        audit_attack_onsets(
            merged
        )
    )

    # --------------------------------------------------------
    # 7. Audit attack types
    # --------------------------------------------------------

    (
        train_types,
        validation_types,
        test_types,
    ) = audit_attack_types(
        merged
    )

    # --------------------------------------------------------
    # 8. Audit attack families
    # --------------------------------------------------------

    family_df = (
        audit_attack_families(
            merged
        )
    )

    # --------------------------------------------------------
    # 9. Audit persistence
    # --------------------------------------------------------

    persistence = (
        audit_persistence(
            merged
        )
    )

    # --------------------------------------------------------
    # 10. Save joined dataset
    # --------------------------------------------------------

    merged.to_csv(
        RESULT_DIR
        / "forecasting_target_audit_joined.csv",
        index=False
    )

    # --------------------------------------------------------
    # 11. Create summary
    # --------------------------------------------------------

    create_summary_report(
        merged,
        episodes,
        onsets,
        persistence,
    )

    # --------------------------------------------------------
    # 12. Final validation
    # --------------------------------------------------------

    passed = final_validation(
        merged
    )

    # --------------------------------------------------------
    # Final message
    # --------------------------------------------------------

    print(
        "\nOutput directory:"
    )

    print(
        RESULT_DIR
    )

    if passed:

        print(
            "\nNext step:"
        )

        print(
            "Review the attack-episode and onset results "
            "before training the LSTM."
        )

    else:

        print(
            "\nDo NOT proceed to LSTM training yet."
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()