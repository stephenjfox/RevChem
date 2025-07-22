#!/usr/bin/env python3
"""
trajectory_stats.py
Compute time-series statistics on a sequence of 2-D coordinates.

The CSV must contain at least the columns
    x, y, t
where
    x, y : float   (spatial coordinates)
    t    : int/float/datetime (time stamp, monotonically increasing)

The script adds
    delta_dist  – Euclidean distance to previous point
    angle       – absolute direction of the segment (radians, 0 = East, CCW)

and then outputs
    global_stats – overall summary
    rolling_stats– rolling window stats (mean / std / min / max of speed & angle)
"""

import argparse
from pathlib import Path

import polars as pl
from polars import col


def add_step_columns(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add delta_dist and angle to the DataFrame.
    Assumes rows are already sorted by `t`.
    """
    return df.with_columns(
        # previous x, y
        pl.col("x").shift(1).alias("x_prev"),
        pl.col("y").shift(1).alias("y_prev"),
    ).with_columns(
        delta_x=col("x") - col("x_prev"),
        delta_y=col("y") - col("y_prev"),
    ).with_columns(
        delta_dist=(col("delta_x") ** 2 + col("delta_y") ** 2).sqrt(),
        angle=pl.arctan2(col("delta_y"), col("delta_x")),
    ).drop(["x_prev", "y_prev", "delta_x", "delta_y"])


def global_summary(df: pl.DataFrame) -> pl.DataFrame:
    """
    Return a single-row DataFrame with global trajectory statistics.
    """
    return df.select(
        total_distance=col("delta_dist").sum(),
        avg_speed=col("delta_dist").mean(),
        std_speed=col("delta_dist").std(),
        min_speed=col("delta_dist").min(),
        max_speed=col("delta_dist").max(),
        net_displacement=(
            (col("x").last() - col("x").first()) ** 2
            + (col("y").last() - col("y").first()) ** 2
        ).sqrt(),
        total_time=col("t").last() - col("t").first(),
        sinuosity=(
            col("delta_dist").sum()
            / (
                (col("x").last() - col("x").first()) ** 2
                + (col("y").last() - col("y").first()) ** 2
            ).sqrt()
        ),
    )


def rolling_summary(df: pl.DataFrame, window: int) -> pl.DataFrame:
    """
    Rolling statistics on speed and angle.
    """
    return (
        df.with_columns(
            roll_speed_mean=col("delta_dist")
            .rolling_mean(window_size=window, min_periods=1)
            .alias("speed_mean"),
            roll_speed_std=col("delta_dist")
            .rolling_std(window_size=window, min_periods=1)
            .alias("speed_std"),
            roll_angle_mean=col("angle")
            .rolling_mean(window_size=window, min_periods=1)
            .alias("angle_mean"),
            roll_angle_std=col("angle")
            .rolling_std(window_size=window, min_periods=1)
            .alias("angle_std"),
        )
        # Optionally drop the helper columns here if you only want the stats
    )

def fixation_segments(
    df: pl.DataFrame,
    radius: float,
    *,
    x: str = "x",
    y: str = "y",
    t: str = "t",
    min_fixation_time: float | None = None,
    min_n_points: int = 1,
) -> pl.DataFrame:
    """
    Detect contiguous *fixation* segments in an eye-tracking trace.

    A fixation starts when the gaze stays **within `radius` units** from the
    *anchor* point of that fixation for at least `min_fixation_time`
    seconds (or `min_n_points` samples).  While the anchor is fixed,
    every new point is tested against this anchor.  As soon as a point
    exceeds the radius, the current fixation ends and a new anchor is set
    at that point.

    Parameters
    ----------
    df : pl.DataFrame
        Must contain the columns `x`, `y`, and `t` (units are your own).
    radius : float
        Spatial tolerance (in the same units as x, y).
    x, y, t : str
        Column names for the spatial and temporal coordinates.
    min_fixation_time : float, optional
        Minimum duration (in the same units as `t`) for a segment to be
        retained.  If None, `min_n_points` is the only filter.
    min_n_points : int, default 1
        Minimum number of consecutive points inside the radius.

    Returns
    -------
    pl.DataFrame
        One row per fixation segment with columns:
        fixation_id   – unique id per fixation (monotonic)
        t_start    – first timestamp of the fixation
        t_end      – last timestamp of the fixation
        x_anchor   – x-coordinate of the anchor point
        y_anchor   – y-coordinate of the anchor point
        duration   – t_end - t_start
        n_points   – number of rows in the fixation
    """
    if df.is_empty():
        return pl.DataFrame(
            {
                "fixation_id": pl.Series(dtype=pl.Int64),
                "t_start": pl.Series(dtype=pl.Int64),
                "t_end": pl.Series(dtype=pl.Int64),
                "x_anchor": pl.Series(dtype=pl.Float64),
                "y_anchor": pl.Series(dtype=pl.Float64),
                "n_points": pl.Series(dtype=pl.Int64),
                "duration": pl.Series(dtype=pl.Int64),
            }
        )

    # Ensure ascending order
    df = df.sort(t)
    df = df.with_row_index(name="i")

    # Convert to Python dicts for iteration
    rows = df.to_dicts()
    fixations = []
    if not rows:
        return pl.DataFrame(fixations)

    # Initial anchor point
    current_fix_start_idx = 0
    anchor_x = rows[0][x]
    anchor_y = rows[0][y]

    for i in range(1, len(rows)):
        dist = ((rows[i][x] - anchor_x) ** 2 + (rows[i][y] - anchor_y) ** 2) ** 0.5
        if dist > radius:
            # End of fixation, record it
            fixations.append(
                {
                    "i_start": rows[current_fix_start_idx]["i"],
                    "i_end": rows[i - 1]["i"],
                    "x_anchor": anchor_x,
                    "y_anchor": anchor_y,
                }
            )
            # New fixation starts at current point
            current_fix_start_idx = i
            anchor_x = rows[i][x]
            anchor_y = rows[i][y]

    # Add the last fixation
    fixations.append(
        {
            "i_start": rows[current_fix_start_idx]["i"],
            "i_end": rows[-1]["i"],
            "x_anchor": anchor_x,
            "y_anchor": anchor_y,
        }
    )

    # Convert back to polars DataFrame
    fix_df = pl.from_dicts(fixations)

    # Join with original data to get times and counts
    fix_df = fix_df.join(
        df.select(["i", t]).rename({"i": "i_start", t: "t_start"}),
        on="i_start",
    ).join(
        df.select(["i", t]).rename({"i": "i_end", t: "t_end"}),
        on="i_end",
    )

    # Calculate n_points and duration
    fix_df = fix_df.with_columns(
        n_points=pl.col("i_end") - pl.col("i_start") + 1,
        duration=pl.col("t_end") - pl.col("t_start"),
    ).drop(["i_start", "i_end"])

    # Apply filters
    if min_fixation_time is not None:
        fix_df = fix_df.filter(pl.col("duration") >= min_fixation_time)
    fix_df = fix_df.filter(pl.col("n_points") >= min_n_points)

    # Finalize
    return fix_df.with_row_index(name="fixation_id").select(
        [
            "fixation_id",
            "t_start",
            "t_end",
            "x_anchor",
            "y_anchor",
            "n_points",
            "duration",
        ]
    )


def aoi_from_fixations(
    fixations: pl.DataFrame, eps: float, min_samples: int
) -> pl.DataFrame:
    """
    Identify Areas of Interest (AOIs) from fixation data using DBSCAN.

    Parameters
    ----------
    fixations : pl.DataFrame
        A DataFrame of fixation data, as returned by `fixation_segments`.
        Must contain `x_anchor` and `y_anchor` columns.
    eps : float
        The maximum distance between two samples for one to be considered
        as in the neighborhood of the other.
    min_samples : int
        The number of samples in a neighborhood for a point to be
        considered as a core point.

    Returns
    -------
    pl.DataFrame
        A DataFrame with AOI statistics, including the number of fixations
        in each AOI, the total duration, and the bounding box of the AOI.
    """
    if fixations.is_empty():
        return pl.DataFrame(
            {
                "aoi_id": pl.Series(dtype=pl.Int64),
                "n_fixations": pl.Series(dtype=pl.Int64),
                "total_duration": pl.Series(dtype=pl.Int64),
                "x_min": pl.Series(dtype=pl.Float64),
                "y_min": pl.Series(dtype=pl.Float64),
                "x_max": pl.Series(dtype=pl.Float64),
                "y_max": pl.Series(dtype=pl.Float64),
            }
        )

    from sklearn.cluster import DBSCAN

    # Extract anchor points for clustering
    coords = fixations.select(["x_anchor", "y_anchor"]).to_numpy()

    # Perform DBSCAN clustering
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    labels = db.labels_

    # Add cluster labels to fixations DataFrame
    fixations = fixations.with_columns(aoi_id=pl.Series(labels))

    # Filter out noise points (label -1)
    aois = fixations.filter(pl.col("aoi_id") != -1)

    # Aggregate statistics per AOI
    aoi_stats = aois.group_by("aoi_id").agg(
        n_fixations=pl.len(),
        total_duration=pl.col("duration").sum(),
        x_min=pl.col("x_anchor").min(),
        y_min=pl.col("y_anchor").min(),
        x_max=pl.col("x_anchor").max(),
        y_max=pl.col("y_anchor").max(),
    )

    return aoi_stats.sort("aoi_id")


def main():
    parser = argparse.ArgumentParser(description="Compute trajectory statistics.")
    parser.add_argument("--csv", required=True, type=Path, help="Path to CSV file")
    parser.add_argument("--x_col", default="x", help="name of x-coordinate column")
    parser.add_argument("--y_col", default="y", help="name of y-coordinate column")
    parser.add_argument("--time_col", default="t", help="name of time column")
    parser.add_argument(
        "--window", type=int, default=100, help="rolling window size (#rows)"
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="write rolling stats to this CSV"
    )
    args = parser.parse_args()

    # Read and ensure ascending by time
    df = (
        pl.read_csv(args.csv)
        .select(
            pl.col(args.x_col).cast(pl.Float64).alias("x"),
            pl.col(args.y_col).cast(pl.Float64).alias("y"),
            pl.col(args.time_col).alias("t"),
        )
        .sort("t")
    )

    df = add_step_columns(df)

    # Global summary
    g = global_summary(df)
    print("=== Global trajectory statistics ===")
    print(g)

    # Rolling stats
    r = rolling_summary(df, args.window)
    if args.out:
        r.write_csv(args.out)
        print(f"\nRolling stats written to {args.out}")
    else:
        print("\n=== Rolling statistics (first 20 rows) ===")
        print(r.head(20))


if __name__ == "__main__":
    main()
