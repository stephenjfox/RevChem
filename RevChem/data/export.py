# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/02_data_export_clean.ipynb.

# %% auto 0
__all__ = ['T_source', 'T_out', 'TOBII_POLARS_SCHEMA', 'REALEYE_POLARS_SCHEMA', 'Resettable', 'iter_parse_raw_to_GazeInfo',
           'resettable_iter_raw', 'RealEyeRawRow', 'cumulative_sum', 'raw_gazes_row_to_df',
           'realeye_timestamp_to_datetime', 'correct_realeye_df_group', 'pipeline_raw_realeye_to_timed_dataframe',
           'prepare_chunk_for_json', 'write_chunks_to_json', 'read_chunks_from_json', 'write_chunks_to_parquet',
           'read_chunks_from_parquet', 'match_tobii_to_realeye_groups']

# %% ../../nbs/02_data_export_clean.ipynb 2
import polars as pl
from datetime import datetime, timedelta, timezone as UTC
from pathlib import Path
from typing import Callable, Iterable, Iterator, TypeVar, Union


# %% ../../nbs/02_data_export_clean.ipynb 7
from dataclasses import dataclass
from ..realeye import GazeInfo, iter_parse_raw_data

T_source = TypeVar("T_source")
T_out = TypeVar("T_out")


class Resettable(Iterable[T_out]):
    def __init__(
        self, source_data: T_source, iter_gen: Callable[[T_source], Iterable[T_out]]
    ):
        self._source = source_data
        self.iter_gen = iter_gen

    def __iter__(self) -> Iterable[tuple[int, int, int, float, float, float]]:
        return self.iter_gen(self._source)


def iter_parse_raw_to_GazeInfo(raw_data: str) -> Iterator[GazeInfo]:
    for sextuple in iter_parse_raw_data(raw_data):
        if (length_ := len(sextuple)) > 6:
            # indicates that the RealEye system captured a mouse click. Nothing more.
            print(f"Got a {length_}-tuple: {sextuple = }")
        yield GazeInfo(*sextuple[:6])


def resettable_iter_raw(test_raw_data: str):
    return Resettable(test_raw_data, iter_parse_raw_to_GazeInfo)


@dataclass
class RealEyeRawRow:
    participant_id: str  # the participant being tested
    item_id: str  # the stimulus being shown
    test_created_at: datetime  # when the web browser started running (NOT THE SAME AS THE STIMULUS START TIME)
    raw_data: Iterable[GazeInfo] = None

    def __post_init__(self):
        self.raw_data = resettable_iter_raw(self.raw_data)

    @classmethod
    def from_row_tuples(cls, tuple) -> "RealEyeRawRow":
        return cls(*tuple)

# %% ../../nbs/02_data_export_clean.ipynb 9
def cumulative_sum(items: list[int|float]) -> list[int|float]:
    """Calculate the cumulative sum up to and including a given index"""
    csum = 0
    res = [None] * len(items)
    for i, item in enumerate(items):
        csum += item
        res[i] = csum
    return res

# %% ../../nbs/02_data_export_clean.ipynb 11
def raw_gazes_row_to_df(
    row: RealEyeRawRow, # typed row from the CSV. should have few, if any changes, from the raw CSV file. Used for semantic tidyness
    *,
    time_since_name: str = "time_since_start", # new name given to the column that records the time (ms) since this stimulus was shown
    x_name: str = "X", # new name given to the column that captures the X-coordinate of the GazeInfo gaze
    y_name: str = "Y", # new name given to the column that captures the Y-coordinate of the GazeInfo gaze
) -> pl.DataFrame:
    df = pl.DataFrame(
        [
            (
                row.test_created_at,
                gaze_info.time_ms_since_start,
                gaze_info.gaze_point_X,
                gaze_info.gaze_point_Y,
            )
            for gaze_info in row.raw_data
        ],
        schema={
            "test_created_at": pl.Datetime,
            time_since_name: pl.Int32,
            x_name: pl.Int32,
            y_name: pl.Int32,
        },
        orient="row",
    )
    return df.with_columns( # force everything to be UTC because that's what it should be (per docs)
        pl.col("test_created_at").dt.replace_time_zone(time_zone="UTC"),
    )

# %% ../../nbs/02_data_export_clean.ipynb 12
from ..common import dt_str_now, group_by

# %% ../../nbs/02_data_export_clean.ipynb 16
def realeye_timestamp_to_datetime(
    datetime_col: str = "test_created_at", # column with the recording start datetime
    timestamp_col: str = "time_ms_since_start", # the integer column representing the milliseconds since stimulus exposure
    *,
    overwrite: bool = True, # Whether, the `timestamp_col` will have a datetime type in the result, or (if False) a new column is created
    additional_offset_ms: int = 0, # Additional offset to 
) -> pl.DataFrame:
    """Update `timestamp_col` to be an increasing datetime rather than the default (i64 or int).

    Corrects the `timestamp_col` of `df` to be a `pl.Datetime`, to ease legibility and computation.
    Sums the `timestamp_col` with that of the reference `datetime_col`, incrementing the time forward.

    Returns:
        A dataframe with the described change to the `timestamp_col`
    """
    new_name = timestamp_col if overwrite else f"{timestamp_col}__new_dt_time"
    new_column = pl.col(datetime_col) + pl.duration(milliseconds=timestamp_col) + pl.duration(milliseconds=additional_offset_ms)
    new_column = new_column.alias(new_name)

    return new_column

# %% ../../nbs/02_data_export_clean.ipynb 17
def correct_realeye_df_group(
    group_dfs: list[pl.DataFrame], *, time_col: str = "time_since_start"
):
    """In-place mutation to correct the dfs' timing, assuming dfs are aggregated by `test_created_at`"""
    # 1. sum the last|largest millisecond offset from each of the dfs
    group_millisecond_offset_maxes = [df[time_col].max() for df in group_dfs]
    total_milliseconds_since_start = sum(group_millisecond_offset_maxes)
    # 1a. assume we have fully contiguous time series
    # 2. compute the start time: "test_created_at" - total relative milliseconds
    # Using min (not max), because the file outputs take time, even though the recording is done,
    # and the trial is started earlier than the one-ish sec it takes to create the output.
    re_recording_end = min(map(lambda df: df["test_created_at"][0], group_dfs))
    re_recording_start = re_recording_end - timedelta(
        microseconds=total_milliseconds_since_start * 1000
    )
    # 3. roll the relative milliseconds forward for each subsequent DataFrame
    # this is like a "scan" or "cummulative sum"
    # We shift everything "left" one, because the first doesn't need anything additional
    # The second df need only add the first, third df only add the two prior, etc.
    addend_group_millisecond_offsets = [0] + cumulative_sum(
        group_millisecond_offset_maxes[:-1]
    )
    # update the dfs
    for group_member_index in range(len(group_dfs)):
        df = group_dfs[group_member_index].with_columns(
            __temp_start_time=re_recording_start
        )
        group_dfs[group_member_index] = df.with_columns(
            realeye_timestamp_to_datetime(
                datetime_col="__temp_start_time",
                timestamp_col=time_col,
                additional_offset_ms=addend_group_millisecond_offsets[
                    group_member_index
                ],
            )
        ).drop("__temp_start_time")

# %% ../../nbs/02_data_export_clean.ipynb 18
# TODO: rename realeye data pipeline function
from ..common import list_concat
from ..tobii import REALEYE_ITEM_IDS, GroupedFrames


def pipeline_raw_realeye_to_timed_dataframe(
    re_raw_df: pl.DataFrame,  # result of pl.read_csv("raw-gazes.csv").
    *,
    do_group_stats_export: bool = False,  # whether compute early stats, write them to EXPORT_ROOT and exit early. Fails if EXPORT_ROOT is undefined.
    debug: bool = False,  # whether we output the first row of each dataframe, to debug what we're looking at.
    dt_timestamp_col: str = "time_since_start",  # name for the datetime timestamp column in the output dataframes
):
    real_eye_rows: list[RealEyeRawRow] = sorted(
        map(RealEyeRawRow.from_row_tuples, re_raw_df.rows()),
        # sorted by item_id, leveraging that list.index(...) -> ordinal position
        key=lambda re_row: REALEYE_ITEM_IDS.index(re_row.item_id),
    )
    # rows to dataframe
    dfs = [
        raw_gazes_row_to_df(row, time_since_name=dt_timestamp_col)
        for row in real_eye_rows
    ]

    if do_group_stats_export:
        run_realeye_df_group_statistics(dfs)
        return

    if debug:
        display(pl.concat([df.head(1) for df in dfs]))

    # group by "creation" time, down to the second, to get first ordering
    # then flatten so we have an overall sequence with subsequences which are in order.
    # Sorting is performed to make sure the test_created_at and test_created_at+1sec are in the correct order.
    dfs = list_concat(
        sorted(
            group_by(lambda df: df["test_created_at"][0], dfs).values(),
            key=lambda dfs: dfs[0]["test_created_at"][0],
        )
    )
    grouped_dfs = GroupedFrames.from_tuples(
        (df["test_created_at"][0].replace(second=0, microsecond=0), df) for df in dfs
    )

    # in particular, several entries are exactly 1 second apart.
    # We assume the later of these entries is "next" chronologically.
    grouped_dfs = GroupedFrames.from_tuples(
        (ts.replace(second=0, microsecond=0), df)
        for ts, group in grouped_dfs.items()
        for df in group
    )

    # group by the minute, order is retained within the group
    # giving us groups that put all entries of a given trial in the right order
    # even if split by a single second, they are collected in the correct stimulus order
    # and are results are output within a minute of each other.
    grouped = GroupedFrames.from_tuples(
        (df["test_created_at"][0].replace(second=0, microsecond=0), df) for df in dfs
    )

    # for group_start_minute, group_dfs in grouped.items():
    #     correct_realeye_df_group(group_dfs, time_col=dt_timestamp_col)

    # now we can apply the timestamp correction algorithm to the groups, functionally.
    # apply the correction, and return the group_dfs so `apply` works and we have a working dict
    grouped = grouped.apply(
        lambda _, group_dfs: (
            correct_realeye_df_group(group_dfs, time_col=dt_timestamp_col),
            group_dfs,
        )[1]
    )

    # lastly, concatenate all of the groups, now that their time columns are fixed
    mapped = GroupedFrames(grouped) #.concat_groups()

    return mapped


# %% ../../nbs/02_data_export_clean.ipynb 27
import json
import gzip
from typing import Any, List, Tuple


def prepare_chunk_for_json(
    df_120hz: pl.DataFrame, df_30hz_list: List[pl.DataFrame], chunk_id: datetime | str
) -> dict:
    """
    Prepare a 120 Hz DataFrame and its associated 30 Hz fragments for JSON export.

    Args:
        df_120hz: Polars DataFrame with 120 Hz data
        df_30hz_list: List of Polars DataFrames with 30 Hz fragments
        chunk_id: Unique time for which these chunks are relevant.

    Returns:
        Dictionary with structured data for JSON serialization
    """
    # Convert 120 Hz DataFrame to dictionary
    chunk_120hz = df_120hz.write_json()

    # Convert each 30 Hz fragment DataFrame to dictionary
    fragments_30hz = [df.write_json() for df in df_30hz_list]

    # Structure the data
    return {
        "chunk_id": chunk_id if isinstance(chunk_id, str) else chunk_id.isoformat(),
        "hz_120": chunk_120hz,
        "hz_30_fragments": fragments_30hz,
    }


def write_chunks_to_json(
    chunk_associations: List[Tuple[pl.DataFrame, List[pl.DataFrame]]],
    output_path: str,
    *,
    key_func: Callable[[pl.DataFrame, list[pl.DataFrame]], Any],
) -> None:
    """
    Process multiple 120 Hz chunks with their 30 Hz fragments and write to compressed JSON.

    Args:
        chunk_associations: List of tuples, each containing a 120 Hz DataFrame and a list of 30 Hz DataFrames
        output_path: Path to write the compressed JSON file (should end in .json.gz)
        key_func: Function to create a dictionary key from the chunk association.
    """
    # Prepare all chunks
    all_chunks = [
        prepare_chunk_for_json(df_120hz, df_30hz_list, key_func(df_120hz, df_30hz_list))
        for df_120hz, df_30hz_list in chunk_associations
    ]

    # Convert to JSON string
    json_data = json.dumps(all_chunks, ensure_ascii=False)

    # Write to compressed JSON file
    with gzip.open(output_path, "wt", encoding="utf-8") as f:
        f.write(json_data)


"""
Tobii dataframe
│ timestamp                      ┆ X    ┆ Y    ┆ source_tsv   │
│ ---                            ┆ ---  ┆ ---  ┆ ---          │
│ datetime[μs, UTC]              ┆ i32  ┆ i32  ┆ str          │


RealEye dataframe
│ test_created_at         ┆ timestamp                   ┆ X    ┆ Y   │
│ ---                     ┆ ---                         ┆ ---  ┆ --- │
│ datetime[μs, UTC]       ┆ datetime[μs, UTC]           ┆ i32  ┆ i32 │
"""

TOBII_POLARS_SCHEMA = {
    "timestamp": pl.Datetime,
    "X": pl.Int32,
    "Y": pl.Int32,
    "source_tsv": pl.String,
}
REALEYE_POLARS_SCHEMA = {
    "test_created_at": pl.Datetime,
    "timestamp": pl.Datetime,
    "X": pl.Int32,
    "Y": pl.Int32,
}


def read_chunks_from_json(
    input_path: str,
    tobii_schema=TOBII_POLARS_SCHEMA,
    realeye_schema=REALEYE_POLARS_SCHEMA,
) -> list[tuple[pl.DataFrame, list[pl.DataFrame]]]:
    """
    Read a compressed JSON file and reconstruct the 120 Hz and 30 Hz DataFrames.

    Args:
        input_path: Path to the compressed JSON file (e.g., output.json.gz)

    Returns:
        list of tuples, each containing a 120 Hz DataFrame and a list of 30 Hz DataFrames
    """
    # Read and decompress the JSON file
    with gzip.open(input_path, "rt", encoding="utf-8") as f:
        data = json.load(f)

    # Reconstruct DataFrames
    chunk_associations = []
    for chunk in data:
        # Convert 120 Hz JSON string to DataFrame
        df_120hz = pl.read_json(
            bytes(chunk["hz_120"], encoding="UTF-8"), schema_overrides=tobii_schema
        )

        # Convert 30 Hz fragment JSON strings to list of DataFrames
        df_30hz_list = [
            pl.read_json(
                bytes(fragment, encoding="UTF-8"), schema_overrides=realeye_schema
            )
            for fragment in chunk["hz_30_fragments"]
        ]

        # Add to result
        chunk_associations.append((df_120hz, df_30hz_list))

    return chunk_associations


def write_chunks_to_parquet(
    chunk_associations: list[tuple[pl.DataFrame, list[pl.DataFrame]]], output_dir: str
) -> None:
    """
    Write 120 Hz chunks and their 30 Hz fragments to Parquet files.

    Args:
        chunk_associations: list of tuples, each containing a 120 Hz DataFrame and a list of 30 Hz DataFrames
        output_dir: Directory to write Parquet files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    for idx, (df_120hz, df_30hz_list) in enumerate(chunk_associations):
        # Write 120 Hz DataFrame
        df_120hz.write_parquet(output_path / f"chunk_{idx}_120hz.parquet")

        # Write 30 Hz fragments
        for frag_idx, df_30hz in enumerate(df_30hz_list):
            df_30hz.write_parquet(output_path / f"chunk_{idx}_30hz_{frag_idx}.parquet")


def read_chunks_from_parquet(
    input_dir: str, num_chunks: int
) -> list[tuple[pl.DataFrame, list[pl.DataFrame]]]:
    """
    Read 120 Hz chunks and their 30 Hz fragments from Parquet files.

    Args:
        input_dir: Directory containing Parquet files
        num_chunks: Number of chunks to read (to know how many files to look for)

    Returns:
        list of tuples, each containing a 120 Hz DataFrame and a list of 30 Hz DataFrames
    """
    input_path = Path(input_dir)
    chunk_associations = []
    for idx in range(num_chunks):
        # Read 120 Hz DataFrame
        df_120hz = pl.read_parquet(input_path / f"chunk_{idx}_120hz.parquet")

        # Read 30 Hz fragments (assume up to 9 fragments based on 7±2)
        df_30hz_list = []
        for frag_idx in range(9):  # Adjust range if needed
            fragment_path = input_path / f"chunk_{idx}_30hz_{frag_idx}.parquet"
            if fragment_path.exists():
                df_30hz_list.append(pl.read_parquet(fragment_path))
            else:
                break  # Stop if no more fragments exist

        chunk_associations.append((df_120hz, df_30hz_list))

    return chunk_associations

# %% ../../nbs/02_data_export_clean.ipynb 29
def match_tobii_to_realeye_groups(
    tobii_dfs: List[pl.DataFrame],
    realeye_groups: GroupedFrames,
    *,
    tobii_df_time_key: str = "timestamp",  # Key in the Tobii DataFrame for matching,
    # realeye_df_time_key: str = "test_created_at"  # Key in the RealEye DataFrame for matching
):
    times_tobii = [df[tobii_df_time_key][0] for df in tobii_dfs]
    times_realeye = list(realeye_groups.keys())
    assert type(times_realeye[0]) is datetime, "RealEye group keys should be datetimes"
    _logging = False  # set to True to see the matching process
    if _logging:
        print(f"Matching {len(times_tobii)} Tobii timestamps to {len(times_realeye)} RealEye timestamps")

    # tobii is the reference, realeye is under scrutiny
    # 50 sec is the shortest time between Tobii "Start record", skipping validation, and starting RealEye
    _MIN_TIME_DELTA = timedelta(seconds=50)
    found_indices: set[int] = set() # just because I don't want to order it
    pair_indices = []
    for tobii_time in times_tobii:
        current_min_time_diff = timedelta(days=1_000) # more time than is sensible
        found_index = -1
        for i, re_time in enumerate(times_realeye):
            latest_diff = re_time - tobii_time
            if (i not in found_indices) and (_MIN_TIME_DELTA <= latest_diff < current_min_time_diff):
                found_index = i # we've found it. Don't need to log it
                current_min_time_diff = latest_diff
                _logging and print(f"Changed: {current_min_time_diff = }")
        found_indices.add(found_index)
        pair_indices.append(found_index)

    result = [
        (tobii_df.sort(tobii_df_time_key), realeye_groups[times_realeye[re_index]]) 
        for re_index, tobii_df in zip(pair_indices, tobii_dfs)
    ]
    return result

