import polars as pl
from RevChem.stats.timeseries import trajectory_iou
import pytest

def test_trajectory_iou_overlapping_squares():
    # Trajectory 1: A square from (0,0) to (2,2)
    traj1_data = {
        "X": [0, 2, 2, 0, 0],
        "Y": [0, 0, 2, 2, 0],
    }
    traj1 = pl.DataFrame(traj1_data)

    # Trajectory 2: A square from (1,1) to (3,3)
    traj2_data = {
        "X": [1, 3, 3, 1, 1],
        "Y": [1, 1, 3, 3, 1],
    }
    traj2 = pl.DataFrame(traj2_data)

    # The intersection is a 1x1 square (area 1)
    # The union is the area of both 2x2 squares minus the intersection area
    # Union area = 4 + 4 - 1 = 7
    # Expected IoU = 1 / 7
    expected_iou = 1.0 / 7.0

    iou = trajectory_iou(traj1, traj2)

    assert iou == pytest.approx(expected_iou)

def test_trajectory_iou_no_overlap():
    # Trajectory 1: A square from (0,0) to (1,1)
    traj1_data = {
        "X": [0, 1, 1, 0, 0],
        "Y": [0, 0, 1, 1, 0],
    }
    traj1 = pl.DataFrame(traj1_data)

    # Trajectory 2: A square from (2,2) to (3,3)
    traj2_data = {
        "X": [2, 3, 3, 2, 2],
        "Y": [2, 2, 3, 3, 2],
    }
    traj2 = pl.DataFrame(traj2_data)

    # No overlap, so IoU should be 0
    expected_iou = 0.0

    iou = trajectory_iou(traj1, traj2)

    assert iou == pytest.approx(expected_iou)

def test_trajectory_iou_identical_squares():
    # Trajectory 1: A square from (0,0) to (2,2)
    traj1_data = {
        "X": [0, 2, 2, 0, 0],
        "Y": [0, 0, 2, 2, 0],
    }
    traj1 = pl.DataFrame(traj1_data)

    # Trajectory 2: Identical to trajectory 1
    traj2 = traj1.clone()

    # Intersection and union are the same, so IoU should be 1
    expected_iou = 1.0

    iou = trajectory_iou(traj1, traj2)

    assert iou == pytest.approx(expected_iou)

def test_trajectory_iou_less_than_3_points():
    # Trajectory 1: A square from (0,0) to (2,2)
    traj1_data = {
        "X": [0, 2, 2, 0, 0],
        "Y": [0, 0, 2, 2, 0],
    }
    traj1 = pl.DataFrame(traj1_data)

    # Trajectory 2: Only 2 points
    traj2_data = {
        "X": [1, 3],
        "Y": [1, 1],
    }
    traj2 = pl.DataFrame(traj2_data)

    # Not enough points to form a polygon, should return 0
    expected_iou = 0.0

    iou = trajectory_iou(traj1, traj2)

    assert iou == pytest.approx(expected_iou)
