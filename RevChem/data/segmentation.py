# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/05_image_gen_by_segments.ipynb.

# %% auto 0
__all__ = ['AssociatedTrialSegements', 'join_chunks_as_segments']

# %% ../../nbs/05_image_gen_by_segments.ipynb 2
import polars as pl
from pathlib import Path

# %% ../../nbs/05_image_gen_by_segments.ipynb 4
from typing import NamedTuple


class AssociatedTrialSegements(NamedTuple):
    trial_name_or_id: str
    segments: list[pl.DataFrame]


def join_chunks_as_segments(
    associated_chunks: list[tuple[pl.DataFrame, list[pl.DataFrame]]],
    *,
    join_strategy="backward",
    drop_null=False,
) -> list[AssociatedTrialSegements]:
    """Transform a "Chunk and associated list" to "list of associated chunks"

    Algorithm:
        given a list of tuple[pl.DataFrame, list[pl.DataFrame]] representing tobii and RealEye, resp.
        for each RE dataframe
            join with Tobii "master" frame on the "timestamp" column
            - "how" should be something like "nearest", or "between" the start and end of the RE df in question
            - rename the RE columns "X_re" and "Y_re"
            - drop the "test_created_at" column
            - filter all the nulls, and those outside of the time bounds of the RE df

    Arguments:
        associated_chunks: list of matched tobii dataframe with all the RE dataframes per stimulus

    Returns:
        subsegments of the Tobii df joined on the time column of the RE df, per the algorithm
    """
    output = []
    for tobii_df, re_dfs in associated_chunks:
        trial_name = tobii_df["source_tsv"][0]
        re_rename = dict(X="X_re", Y="Y_re")
        tobii_df = tobii_df.drop("source_tsv")
        segmented_associations = []
        for re_df in re_dfs:
            # NOTE: may need to use the `tolerance` kwarg to better tune the match-up
            associated = tobii_df.join_asof(
                re_df.drop("test_created_at").rename(re_rename),
                on="timestamp",
                strategy=join_strategy,
            )
            associated = associated.filter(
                (pl.col("timestamp") >= re_df["timestamp"].min())
                & (pl.col("timestamp") <= re_df["timestamp"].max())
            )
            # associated.drop_nulls(["X", "Y"])
            if drop_null:
                # NOTE: important to make sure Tobii's nothing don't crowd in
                # NOTE: this can be 10% of the data and shouldn't be done lightly.
                associated = associated.drop_nulls()
            segmented_associations.append(associated)

        output.append(AssociatedTrialSegements(trial_name, segmented_associations))
    return output


