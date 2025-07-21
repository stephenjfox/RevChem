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
