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
    and rotate clockwise (Z1 -> Z6 -> Z5 -> Z4 -> Z3 -> Z2 -> Z1), so R3 -> R2 and R1 -> R6.
    """
    state = TacticalState(
        home_initial_rotation=Rotation.R1,
        away_initial_rotation=Rotation.R3,
        initial_server=TeamSide.HOME,
    )

    # AWAY wins the rally -> Side-Out for AWAY (Setter in Z3 rotates clockwise to Z2 -> R2)
    event1 = state.register_rally_point(winner=TeamSide.AWAY)
    assert event1.side_out_occurred is True
    assert state.serving_team == TeamSide.AWAY
    assert state.get_rotation(TeamSide.AWAY) == Rotation.R2  # R3 -> R2
    assert state.get_rotation(TeamSide.HOME) == Rotation.R1  # Unchanged

    # HOME wins the next rally -> Side-Out for HOME (Setter in Z1 rotates clockwise to Z6 -> R6)
    event2 = state.register_rally_point(winner=TeamSide.HOME)
    assert event2.side_out_occurred is True
    assert state.serving_team == TeamSide.HOME
    assert state.get_rotation(TeamSide.HOME) == Rotation.R6  # R1 -> R6
    assert state.get_rotation(TeamSide.AWAY) == Rotation.R2  # Unchanged


def test_full_six_rotation_cyclic_arithmetic() -> None:
    """Verify full clockwise Z_6 cycle: R1 -> R6 -> R5 -> R4 -> R3 -> R2 -> R1."""
    state = TacticalState(
        home_initial_rotation=Rotation.R1,
        away_initial_rotation=Rotation.R1,
        initial_server=TeamSide.AWAY,
    )

    expected_sequence = [
        Rotation.R6,
        Rotation.R5,
        Rotation.R4,
        Rotation.R3,
        Rotation.R2,
        Rotation.R1,
    ]

    for expected_rot in expected_sequence:
        # HOME wins side-out -> rotates clockwise
        state.register_rally_point(winner=TeamSide.HOME)
        assert state.get_rotation(TeamSide.HOME) == expected_rot
        # Give serve back to AWAY so HOME can side-out again
        state.register_rally_point(winner=TeamSide.AWAY)


def test_inferred_roles_across_rotations_without_ocr() -> None:
    """
    Verify that TacticalState maps R1..R6 directly to Setter in Z1..Z6:
    - R1 -> Setter in Z1
    - R2 -> Setter in Z2
    - R3 -> Setter in Z3
    - R4 -> Setter in Z4
    - R5 -> Setter in Z5
    - R6 -> Setter in Z6
    """
    state = TacticalState(
        home_initial_rotation=Rotation.R1,
        away_initial_rotation=Rotation.R1,
        initial_server=TeamSide.HOME,
    )

    for rot in Rotation:
        state.set_rotation(TeamSide.HOME, rot)
        snap = state.infer_formation(TeamSide.HOME, use_libero_in_back_row=True)
        expected_zone = CourtZone(rot.value)
        assert snap.setter_zone == expected_zone
        assert snap.zone_roles[expected_zone] == PlayerRole.SETTER


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
