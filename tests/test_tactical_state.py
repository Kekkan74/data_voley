"""
Unit Tests for TacticalState (FIVB Match, Set, and Rotation R1-R6 State Machine).
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data_voley.domain.enums import CourtZone, PlayerRole, Rotation, TeamSide
from data_voley.tactical_state.state_machine import TacticalState


def test_initial_tactical_state() -> None:
    """Verify initial match configuration, rotations, and setter zones."""
    state = TacticalState(
        home_initial_rotation=Rotation.R1,
        away_initial_rotation=Rotation.R4,
        initial_server=TeamSide.HOME,
    )

    assert state.current_set == 1
    assert state.serving_team == TeamSide.HOME
    assert state.get_rotation(TeamSide.HOME) == Rotation.R1
    assert state.get_rotation(TeamSide.AWAY) == Rotation.R4
    assert state.get_score() == (0, 0)


def test_break_point_retains_serve_without_rotation() -> None:
    """When the serving team wins the rally, no rotation occurs (Break Point)."""
    state = TacticalState(
        home_initial_rotation=Rotation.R1,
        away_initial_rotation=Rotation.R2,
        initial_server=TeamSide.HOME,
    )

    event = state.register_rally_point(winner=TeamSide.HOME)

    assert event.side_out_occurred is False
    assert event.home_score == 1
    assert event.away_score == 0
    assert event.home_rotation == Rotation.R1
    assert event.away_rotation == Rotation.R2
    assert state.serving_team == TeamSide.HOME


def test_side_out_triggers_clockwise_rotation_for_receiving_team_only() -> None:
    """
    When the receiving team wins the rally (Side-Out), they recover serve
    and advance 1 clockwise rotation (R_t -> R_{t+1}), while the former server stays put.
    """
    state = TacticalState(
        home_initial_rotation=Rotation.R1,
        away_initial_rotation=Rotation.R3,
        initial_server=TeamSide.HOME,
    )

    # AWAY wins the rally -> Side-Out for AWAY
    event1 = state.register_rally_point(winner=TeamSide.AWAY)
    assert event1.side_out_occurred is True
    assert state.serving_team == TeamSide.AWAY
    assert state.get_rotation(TeamSide.AWAY) == Rotation.R4  # R3 -> R4
    assert state.get_rotation(TeamSide.HOME) == Rotation.R1  # Unchanged

    # HOME wins the next rally -> Side-Out for HOME
    event2 = state.register_rally_point(winner=TeamSide.HOME)
    assert event2.side_out_occurred is True
    assert state.serving_team == TeamSide.HOME
    assert state.get_rotation(TeamSide.HOME) == Rotation.R2  # R1 -> R2
    assert state.get_rotation(TeamSide.AWAY) == Rotation.R4  # Unchanged


def test_full_six_rotation_cyclic_arithmetic() -> None:
    """Verify full Z_6 cycle: R1 -> R2 -> R3 -> R4 -> R5 -> R6 -> R1."""
    state = TacticalState(
        home_initial_rotation=Rotation.R1,
        away_initial_rotation=Rotation.R1,
        initial_server=TeamSide.AWAY,
    )

    expected_sequence = [
        Rotation.R2,
        Rotation.R3,
        Rotation.R4,
        Rotation.R5,
        Rotation.R6,
        Rotation.R1,
    ]

    for expected_rot in expected_sequence:
        # HOME wins side-out -> rotates
        state.register_rally_point(winner=TeamSide.HOME)
        assert state.get_rotation(TeamSide.HOME) == expected_rot
        # Give serve back to AWAY so HOME can side-out again
        state.register_rally_point(winner=TeamSide.AWAY)


def test_inferred_roles_across_rotations_without_ocr() -> None:
    """
    Verify that TacticalState accurately infers where the Setter, Opposite,
    Outside Hitters, Middle Blockers, and Libero are located in R1 through R6.
    """
    state = TacticalState(
        home_initial_rotation=Rotation.R1,
        away_initial_rotation=Rotation.R1,
        initial_server=TeamSide.HOME,
    )

    # In R1: Setter is in Zone 1, Opposite is diagonal in Zone 4
    snap_r1 = state.infer_formation(TeamSide.HOME, use_libero_in_back_row=True)
    assert snap_r1.setter_zone == CourtZone.Z1
    assert snap_r1.setter_is_front_row is False
    assert snap_r1.zone_roles[CourtZone.Z1] == PlayerRole.SETTER
    assert snap_r1.zone_roles[CourtZone.Z4] == PlayerRole.OPPOSITE
    assert snap_r1.zone_roles[CourtZone.Z3] == PlayerRole.MIDDLE_1
    assert snap_r1.zone_roles[CourtZone.Z6] == PlayerRole.LIBERO  # Replaces back-row MB2

    # Advance HOME to R4 (3 side-outs later: Setter is in Zone 4, Opposite in Zone 1)
    state.set_rotation(TeamSide.HOME, Rotation.R4)
    snap_r4 = state.infer_formation(TeamSide.HOME, use_libero_in_back_row=True)
    assert snap_r4.setter_zone == CourtZone.Z4
    assert snap_r4.setter_is_front_row is True
    assert snap_r4.zone_roles[CourtZone.Z4] == PlayerRole.SETTER
    assert snap_r4.zone_roles[CourtZone.Z1] == PlayerRole.OPPOSITE


def test_set_win_requires_two_point_advantage() -> None:
    """Verify FIVB 25-point set rule with mandatory 2-point advantage (deuce handling)."""
    state = TacticalState()

    # Bring score to 24-24
    for _ in range(24):
        state.register_rally_point(TeamSide.HOME)
        state.register_rally_point(TeamSide.AWAY)

    assert state.get_score() == (24, 24)
    assert state.current_set == 1

    # 25-24 does NOT end the set
    ev1 = state.register_rally_point(TeamSide.HOME)
    assert ev1.set_finished is False
    assert state.current_set == 1

    # 26-24 wins Set 1 and advances to Set 2
    ev2 = state.register_rally_point(TeamSide.HOME)
    assert ev2.set_finished is True
    assert state.sets_won[TeamSide.HOME] == 1
    assert state.current_set == 2
    assert state.get_score() == (0, 0)


def test_municipal_best_of_three_format_third_set_to_15_points() -> None:
    """
    Verify Municipal format (max_sets=3):
    - Sets 1 and 2 are played to 25 points.
    - Set 3 (deciding tie-break) is played to 15 points (with 2-pt advantage).
    - Winning 2 sets finishes the match.
    """
    state = TacticalState(max_sets=3)

    # Set 1: HOME wins 25-0
    for _ in range(25):
        ev = state.register_rally_point(TeamSide.HOME)
    assert ev.set_finished is True
    assert ev.match_finished is False
    assert state.sets_won[TeamSide.HOME] == 1
    assert state.current_set == 2

    # Set 2: AWAY wins 25-0 (Match tied 1-1 -> advances to Set 3 tie-break)
    for _ in range(25):
        ev = state.register_rally_point(TeamSide.AWAY)
    assert ev.set_finished is True
    assert ev.match_finished is False
    assert state.sets_won[TeamSide.AWAY] == 1
    assert state.current_set == 3

    # Set 3 (Municipal tie-break to 15 points): HOME wins 15-0
    for _ in range(14):
        ev = state.register_rally_point(TeamSide.HOME)
        assert ev.set_finished is False
    ev_final = state.register_rally_point(TeamSide.HOME)
    assert ev_final.set_finished is True
    assert ev_final.match_finished is True
    assert state.sets_won[TeamSide.HOME] == 2
    assert state.sets_won[TeamSide.AWAY] == 1
