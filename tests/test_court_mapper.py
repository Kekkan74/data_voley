"""
Unit Tests for CourtMapper (Perspective Homography & FIVB Zone 1-6 Spatial Resolver).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data_voley.court_mapper.homography import CourtMapper
from data_voley.domain.enums import CourtZone, TeamSide
from data_voley.domain.models import CourtPoint2D, PixelPoint


@pytest.fixture
def calibrated_mapper() -> CourtMapper:
    """
    Create a CourtMapper calibrated with a trapezoidal broadcast camera perspective
    mapped to the official 9m x 18m FIVB metric plane.
    """
    camera_corners = np.array(
        [
            [200.0, 950.0],   # Bottom-Left (HOME Left Endline: 0m, 0m)
            [1720.0, 950.0],  # Bottom-Right (HOME Right Endline: 9m, 0m)
            [1400.0, 200.0],  # Top-Right (AWAY Left Endline: 9m, 18m)
            [520.0, 200.0],   # Top-Left (AWAY Right Endline: 0m, 18m)
        ],
        dtype=np.float32,
    )
    return CourtMapper.from_camera_corners(camera_corners)


def test_homography_projects_corners_accurately(calibrated_mapper: CourtMapper) -> None:
    """Verify that the 4 calibrated camera corners project back to exact FIVB metric corners."""
    p_bl = calibrated_mapper.pixel_to_court(PixelPoint(200.0, 950.0))
    assert p_bl.x == pytest.approx(0.0, abs=1e-2)
    assert p_bl.y == pytest.approx(0.0, abs=1e-2)

    p_tr = calibrated_mapper.pixel_to_court(PixelPoint(1400.0, 200.0))
    assert p_tr.x == pytest.approx(9.0, abs=1e-2)
    assert p_tr.y == pytest.approx(18.0, abs=1e-2)


@pytest.mark.parametrize(
    "x, y, expected_zone, expected_front",
    [
        (7.5, 2.0, CourtZone.Z1, False),  # Right Back (Server)
        (7.5, 7.5, CourtZone.Z2, True),   # Right Front
        (4.5, 7.5, CourtZone.Z3, True),   # Middle Front
        (1.5, 7.5, CourtZone.Z4, True),   # Left Front
        (1.5, 2.0, CourtZone.Z5, False),  # Left Back
        (4.5, 2.0, CourtZone.Z6, False),  # Middle Back
    ],
)
def test_home_court_zones_resolution(
    calibrated_mapper: CourtMapper,
    x: float,
    y: float,
    expected_zone: CourtZone,
    expected_front: bool,
) -> None:
    """Verify all 6 FIVB zones on the HOME court (0 <= y <= 9m)."""
    mapping = calibrated_mapper.resolve_zone(CourtPoint2D(x=x, y=y))

    assert mapping.is_in_bounds is True
    assert mapping.team_side == TeamSide.HOME
    assert mapping.zone == expected_zone
    assert mapping.is_front_zone is expected_front


@pytest.mark.parametrize(
    "x, y, expected_zone, expected_front",
    [
        (1.5, 16.0, CourtZone.Z1, False),  # AWAY Right Back (rotated 180 deg)
        (1.5, 10.5, CourtZone.Z2, True),   # AWAY Right Front
        (4.5, 10.5, CourtZone.Z3, True),   # AWAY Middle Front
        (7.5, 10.5, CourtZone.Z4, True),   # AWAY Left Front
        (7.5, 16.0, CourtZone.Z5, False),  # AWAY Left Back
        (4.5, 16.0, CourtZone.Z6, False),  # AWAY Middle Back
    ],
)
def test_away_court_zones_resolution_with_180_symmetry(
    calibrated_mapper: CourtMapper,
    x: float,
    y: float,
    expected_zone: CourtZone,
    expected_front: bool,
) -> None:
    """Verify all 6 FIVB zones on the AWAY court (9m < y <= 18m) respecting 180-degree orientation."""
    mapping = calibrated_mapper.resolve_zone(CourtPoint2D(x=x, y=y))

    assert mapping.is_in_bounds is True
    assert mapping.team_side == TeamSide.AWAY
    assert mapping.zone == expected_zone
    assert mapping.is_front_zone is expected_front


def test_out_of_bounds_detection(calibrated_mapper: CourtMapper) -> None:
    """Points outside [0, 9] x [0, 18] must be flagged as out of bounds."""
    out_long = calibrated_mapper.resolve_zone(CourtPoint2D(x=4.5, y=19.2))
    assert out_long.is_in_bounds is False
    assert out_long.zone is None

    out_wide = calibrated_mapper.resolve_zone(CourtPoint2D(x=-0.5, y=5.0))
    assert out_wide.is_in_bounds is False
    assert out_wide.zone is None
