import polars as pl
import pytest
from polars.testing import assert_frame_equal

from basic_ts_stats import fixation_segments


@pytest.fixture
def sample_trace_one_fixation():
    """A simple trace with one fixation."""
    return pl.DataFrame(
        {
            "t": [1, 2, 3, 4, 5],
            "x": [0, 0.1, 0.2, 0.1, 3.0],
            "y": [0, 0.1, 0.2, 0.1, 3.0],
        },
        schema={"t": pl.Int64, "x": pl.Float64, "y": pl.Float64},
    )


@pytest.fixture
def sample_trace_multiple_fixations():
    """A trace with multiple fixations."""
    return pl.DataFrame(
        {
            "t": [1, 2, 3, 10, 11, 12, 20, 21, 22],
            "x": [0, 0.1, 0.2, 5.0, 5.1, 5.2, 0, 0.1, 0.2],
            "y": [0, 0.1, 0.2, 5.0, 5.1, 5.2, 0, 0.1, 0.2],
        },
        schema={"t": pl.Int64, "x": pl.Float64, "y": pl.Float64},
    )


def test_one_fixation(sample_trace_one_fixation):
    """Test a simple case with a single fixation."""
    result = fixation_segments(sample_trace_one_fixation, radius=1.0)
    expected = pl.DataFrame(
        {
            "fixation_id": [0, 1],
            "t_start": [1, 5],
            "t_end": [4, 5],
            "x_anchor": [0.0, 3.0],
            "y_anchor": [0.0, 3.0],
            "n_points": [4, 1],
            "duration": [3, 0],
        }
    )
    assert_frame_equal(result, expected, check_dtypes=False)


def test_multiple_fixations(sample_trace_multiple_fixations):
    """Test a case with multiple fixations."""
    result = fixation_segments(sample_trace_multiple_fixations, radius=1.0)
    expected = pl.DataFrame(
        {
            "fixation_id": [0, 1, 2],
            "t_start": [1, 10, 20],
            "t_end": [3, 12, 22],
            "x_anchor": [0.0, 5.0, 0.0],
            "y_anchor": [0.0, 5.0, 0.0],
            "n_points": [3, 3, 3],
            "duration": [2, 2, 2],
        }
    )
    assert_frame_equal(result, expected, check_dtypes=False)


def test_no_fixations():
    """Test a case with no fixations."""
    trace = pl.DataFrame(
        {
            "t": [1, 2, 3],
            "x": [0, 10, 20],
            "y": [0, 10, 20],
        },
        schema={"t": pl.Int64, "x": pl.Float64, "y": pl.Float64},
    )
    result = fixation_segments(trace, radius=1.0)
    expected = pl.DataFrame(
        {
            "fixation_id": [0, 1, 2],
            "t_start": [1, 2, 3],
            "t_end": [1, 2, 3],
            "x_anchor": [0.0, 10.0, 20.0],
            "y_anchor": [0.0, 10.0, 20.0],
            "n_points": [1, 1, 1],
            "duration": [0, 0, 0],
        }
    )
    assert_frame_equal(result, expected, check_dtypes=False)


def test_min_n_points(sample_trace_multiple_fixations):
    """Test the min_n_points filter."""
    result = fixation_segments(
        sample_trace_multiple_fixations, radius=1.0, min_n_points=3
    )
    expected = pl.DataFrame(
        {
            "fixation_id": [0, 1, 2],
            "t_start": [1, 10, 20],
            "t_end": [3, 12, 22],
            "x_anchor": [0.0, 5.0, 0.0],
            "y_anchor": [0.0, 5.0, 0.0],
            "n_points": [3, 3, 3],
            "duration": [2, 2, 2],
        }
    )
    assert_frame_equal(result, expected, check_dtypes=False)

    # Increase min_n_points to filter out all fixations
    result_filtered = fixation_segments(
        sample_trace_multiple_fixations, radius=1.0, min_n_points=4
    )
    assert result_filtered.is_empty()
