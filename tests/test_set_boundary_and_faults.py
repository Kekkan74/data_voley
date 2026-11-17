"""
TDD Tests for:
1. Municipal 3-set format allowing 3-0 (with 3rd set to 15 points).
2. Multimodal SetBoundaryDetector (team separation, court swap, formation reset) before 25 pts.
3. Realistic score progression preserving unilateral runs (e.g. 7-0 start).
4. RotationFault detection & resynchronization in Set 2.
5. FastAPI endpoints: /api/sets/detect-boundary and /api/tactical/rotation-fault.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data_voley.domain.enums import Rotation, TeamSide


def test_municipal_format_plays_all_three_sets_allowing_3_0() -> None:
    """Verify MUNICIPAL_3 allows playing all 3 sets (3-0) with Set 3 to 15 points."""
    from data_voley.tactical_state.state_machine import TacticalState

    state = TacticalState(max_sets=3, play_all_sets=True)

    # Set 1: HOME wins 25-18
    for _ in range(25):
        state.register_rally_point(TeamSide.HOME)
    assert state.sets_won[TeamSide.HOME] == 1
    assert state.current_set == 2
    assert state.match_finished is False

    # Set 2: HOME wins 25-20 (2-0 so far, match continues to Set 3 in Municipal)
    for _ in range(25):
        ev2 = state.register_rally_point(TeamSide.HOME)
    assert ev2.set_finished is True
    assert state.sets_won[TeamSide.HOME] == 2
    assert state.current_set == 3
    assert state.match_finished is False

    # Set 3 (to 15 points): HOME wins 15-10 -> 3-0 final
    for _ in range(14):
        ev3 = state.register_rally_point(TeamSide.HOME)
        assert ev3.set_finished is False
    ev_final = state.register_rally_point(TeamSide.HOME)
    assert ev_final.set_finished is True
    assert ev_final.match_finished is True
    assert state.sets_won[TeamSide.HOME] == 3
    assert state.sets_won[TeamSide.AWAY] == 0


def test_set_boundary_detector_without_reaching_25_points() -> None:
    """Verify SetBoundaryDetector detects set change via separation, swap, or formation reset."""
    from data_voley.tactical_state.set_boundary import (
        SetBoundaryDetector,
        SetBoundarySignal,
    )

    detector = SetBoundaryDetector()

    # Normal rally pause (15s, no separation, no swap) -> NOT a new set
    normal = detector.evaluate(
        SetBoundarySignal(
            current_score=(18, 16),
            pause_duration_sec=15.0,
            teams_separated_to_benches=False,
            court_sides_swapped=False,
            expected_home_rotation=Rotation.R3,
            observed_home_rotation=Rotation.R3,
        )
    )
    assert normal.is_new_set is False

    # Set change detected via team separation + court side swap + new rotation (even at 21-19)
    boundary = detector.evaluate(
        SetBoundarySignal(
            current_score=(21, 19),
            pause_duration_sec=110.0,
            teams_separated_to_benches=True,
            court_sides_swapped=True,
            expected_home_rotation=Rotation.R3,
            observed_home_rotation=Rotation.R1,
        )
    )
    assert boundary.is_new_set is True
    assert boundary.confidence >= 0.85
    assert len(boundary.reasons) >= 2


def test_score_progression_preserves_7_0_initial_run() -> None:
    """Verify _build_realistic_point_sequence preserves unilateral 7-0 streak before opponent scores."""
    import random

    from data_voley.api.server import StudioSession

    seq = StudioSession._build_realistic_point_sequence(
        h_target=15,
        a_target=9,
        target_limit=15,
        rng=random.Random(42),
        initial_run_home=7,
    )
    assert len(seq) == 24
    # First 7 points must all be HOME (1-0 .. 7-0)
    assert seq[:7] == [TeamSide.HOME] * 7
    assert seq.count(TeamSide.HOME) == 15
    assert seq.count(TeamSide.AWAY) == 9
    assert seq[-1] == TeamSide.HOME


def test_rotation_fault_in_set_2_awards_point_and_resyncs_rotation() -> None:
    """Verify register_rotation_fault awards point to opponent and resynchronizes rotation."""
    from data_voley.tactical_state.state_machine import TacticalState

    state = TacticalState(
        home_initial_rotation=Rotation.R1,
        away_initial_rotation=Rotation.R4,
        initial_server=TeamSide.HOME,
        max_sets=3,
        play_all_sets=True,
    )
    # Advance to Set 2
    for _ in range(25):
        state.register_rally_point(TeamSide.HOME)
    assert state.current_set == 2

    # In Set 2, HOME commits a rotation fault; actual rotation on court is R5
    fault_ev = state.register_rotation_fault(
        faulting_team=TeamSide.HOME,
        actual_rotation=Rotation.R5,
    )
    assert fault_ev.set_number == 2
    assert fault_ev.faulting_team == TeamSide.HOME
    assert fault_ev.awarded_to == TeamSide.AWAY
    assert state.get_score() == (0, 1)
    assert state.serving_team == TeamSide.AWAY
    assert state.get_rotation(TeamSide.HOME) == Rotation.R5
    assert len(state.rotation_faults) == 1


def test_fastapi_set_boundary_and_rotation_fault_endpoints() -> None:
    """Verify POST /api/sets/detect-boundary and POST /api/tactical/rotation-fault."""
    from data_voley.api.server import (
        RotationFaultRequest,
        SetBoundaryRequest,
        detect_set_boundary_endpoint,
        register_rotation_fault_endpoint,
        reset_studio,
    )

    reset_studio()

    # Trigger rotation fault on HOME -> AWAY gets point and HOME resyncs to R4
    res_fault = register_rotation_fault_endpoint(
        RotationFaultRequest(faulting_team="HOME", actual_rotation=4)
    )
    assert res_fault["tactical"]["HOME"]["rotation"] == "R4"
    assert len(res_fault["rotation_faults"]) == 1

    # Trigger set boundary detection with team separation + side swap
    res_bound = detect_set_boundary_endpoint(
        SetBoundaryRequest(
            pause_duration_sec=120.0,
            teams_separated_to_benches=True,
            court_sides_swapped=True,
            observed_home_rotation=1,
            auto_advance=True,
        )
    )
    assert res_bound["last_set_boundary"]["is_new_set"] is True
