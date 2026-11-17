"""
Unit Tests for MatchStatsEngine and Hybrid Tagging Analytics.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data_voley.action_detector.detector import HybridTaggingController
from data_voley.domain.enums import (
    AttackOutcome,
    CourtZone,
    Rotation,
    ServeOutcome,
    SetOutcome,
    TagValidationStatus,
    TeamSide,
)
from data_voley.stats_engine.analytics import MatchStatsEngine
from data_voley.tactical_state.state_machine import TacticalState


def test_serve_errors_crossed_with_rotations() -> None:
    """Verify serve error tracking per team, set, and active rotation (R1-R6)."""
    tactical = TacticalState(
        home_initial_rotation=Rotation.R1,
        away_initial_rotation=Rotation.R4,
        initial_server=TeamSide.HOME,
    )
    engine = MatchStatsEngine(tactical_port=tactical)

    # HOME serves in R1 and hits the net (Error)
    engine.record_serve(team=TeamSide.HOME, outcome=ServeOutcome.ERROR_NET)
    # Force HOME into R3 and record two out-of-bounds serve errors
    tactical.set_rotation(TeamSide.HOME, Rotation.R3)
    engine.record_serve(team=TeamSide.HOME, outcome=ServeOutcome.ERROR_OUT)
    engine.record_serve(team=TeamSide.HOME, outcome=ServeOutcome.ERROR_NET)

    breakdown = engine.get_serve_errors_by_rotation(TeamSide.HOME)
    assert breakdown[Rotation.R1]["ERROR_NET"] == 1
    assert breakdown[Rotation.R3]["ERROR_OUT"] == 1
    assert breakdown[Rotation.R3]["ERROR_NET"] == 1
    assert breakdown[Rotation.R3]["TOTAL_ERRORS"] == 2

    worst_rot, err_count = engine.get_most_vulnerable_serve_rotation(TeamSide.HOME)
    assert worst_rot == Rotation.R3
    assert err_count == 2


def test_attack_efficiency_by_zone_and_rotation() -> None:
    """Verify attack registration, inferred attacker role, and zone efficiency."""
    tactical = TacticalState(
        home_initial_rotation=Rotation.R1,
        away_initial_rotation=Rotation.R1,
        initial_server=TeamSide.HOME,
    )
    engine = MatchStatsEngine(tactical_port=tactical)

    # Zone 4 attack is executed by Receptor Delantero (OUTSIDE_1), Zone 6 by Receptor Zaguero (OUTSIDE_2)
    ev1 = engine.record_attack(
        team=TeamSide.HOME,
        origin_zone=CourtZone.Z4,
        outcome=AttackOutcome.KILL,
    )
    assert ev1.rotation == Rotation.R1
    assert ev1.inferred_attacker_role.value == "OUTSIDE_1"

    ev_pipe = engine.record_attack(
        team=TeamSide.HOME,
        origin_zone=CourtZone.Z6,
        outcome=AttackOutcome.KILL,
    )
    assert ev_pipe.inferred_attacker_role.value == "OUTSIDE_2"  # Never LIBERO!

    engine.record_attack(
        team=TeamSide.HOME,
        origin_zone=CourtZone.Z4,
        outcome=AttackOutcome.KILL,
    )
    engine.record_attack(
        team=TeamSide.HOME,
        origin_zone=CourtZone.Z4,
        outcome=AttackOutcome.ERROR_OUT,
    )
    engine.record_attack(
        team=TeamSide.HOME,
        origin_zone=CourtZone.Z4,
        outcome=AttackOutcome.IN_PLAY,
    )

    zone_stats = engine.get_attack_stats_by_zone(TeamSide.HOME)
    z4 = zone_stats[CourtZone.Z4]
    assert z4["attempts"] == 4
    assert z4["kills"] == 2
    assert z4["errors"] == 1
    # Efficiency = (2 kills - 1 error) / 4 attempts = 0.25
    assert z4["efficiency"] == pytest.approx(0.25, abs=1e-3)


def test_hybrid_tagging_validation_and_correction() -> None:
    """Verify AI proposal validation and manual hotkey correction workflow."""
    tactical = TacticalState()
    engine = MatchStatsEngine(tactical_port=tactical)
    controller = HybridTaggingController(stats_engine=engine)

    # 1. AI proposes an attack from Zone 2
    proposal_id = controller.propose_attack(
        team=TeamSide.HOME,
        origin_zone=CourtZone.Z2,
        outcome=AttackOutcome.IN_PLAY,
        timestamp_ms=12500.0,
    )

    # 2. Analyst corrects the outcome to KILL and zone to Z4 via keyboard shortcut
    committed_event = controller.apply_hotkey_action(
        proposal_id=proposal_id,
        hotkey="c",  # Correct
        corrected_zone=CourtZone.Z4,
        corrected_outcome=AttackOutcome.KILL,
    )

    assert committed_event is not None
    assert committed_event.origin_zone == CourtZone.Z4
    assert committed_event.outcome == AttackOutcome.KILL
    assert committed_event.validation_status == TagValidationStatus.HUMAN_CORRECTED


def test_setting_stats_successful_and_failed_by_target_zone() -> None:
    """Verify setting statistics: successful vs failed passes and best/worst target zones."""
    tactical = TacticalState()
    engine = MatchStatsEngine(tactical_port=tactical)

    # 3 successful sets toward Zone 4 (100% success)
    engine.record_setting_action(TeamSide.HOME, CourtZone.Z4, SetOutcome.ASSIST_KILL)
    engine.record_setting_action(TeamSide.HOME, CourtZone.Z4, SetOutcome.ASSIST_KILL)
    engine.record_setting_action(TeamSide.HOME, CourtZone.Z4, SetOutcome.GOOD_SET)

    # 2 failed sets and 1 good set toward Zone 2 (33.3% success)
    engine.record_setting_action(TeamSide.HOME, CourtZone.Z2, SetOutcome.ERROR_DOUBLES)
    engine.record_setting_action(TeamSide.HOME, CourtZone.Z2, SetOutcome.ERROR_BAD_PASS)
    engine.record_setting_action(TeamSide.HOME, CourtZone.Z2, SetOutcome.GOOD_SET)

    set_stats = engine.get_setting_stats(TeamSide.HOME)
    assert set_stats["total_sets"] == 6
    assert set_stats["successful_sets"] == 4
    assert set_stats["failed_sets"] == 2
    assert set_stats["most_successful_target"]["zone"] == "Z4"
    assert set_stats["most_successful_target"]["success_rate"] == pytest.approx(1.0)
    assert set_stats["least_successful_target"]["zone"] == "Z2"
    assert set_stats["least_successful_target"]["failed"] == 2
