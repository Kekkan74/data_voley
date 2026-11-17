"""
Unit & Integration Tests for the DataVoley Visual Studio API Endpoints.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data_voley.api.server import (
    CourtClickRequest,
    HybridDecisionRequest,
    RallyRequest,
    RotationOverrideRequest,
    get_state,
    health_check,
    override_rotation,
    process_hybrid_tag,
    register_court_click_action,
    register_rally,
    reset_studio,
)


def test_studio_health_and_initial_seeded_charts() -> None:
    """Verify the visual studio initializes with populated chart data across R1-R6."""
    assert health_check()["status"] == "ok"
    state = reset_studio()

    assert "match" in state
    assert "tactical" in state
    assert "analytics" in state
    assert "attack_by_zone" in state["analytics"]
    assert "serves_by_rotation" in state["analytics"]
    assert len(state["trajectories"]) > 0
    assert state["analytics"]["most_vulnerable_rotation"]["rotation"] == "R3"


def test_interactive_court_click_maps_homography_and_updates_charts() -> None:
    """Verify clicking on the 2D court maps to FIVB Zone 4 and records the spike."""
    reset_studio()
    # Click at (1.5m, 7.5m) -> HOME Zone 4
    res = register_court_click_action(
        CourtClickRequest(
            x=1.5,
            y=7.5,
            action_type="ATTACK",
            outcome="KILL",
            target_zone=5,
        )
    )
    mapped = res["last_mapped_point"]
    assert mapped["is_in_bounds"] is True
    assert mapped["team_side"] == "HOME"
    assert mapped["zone"] == "Z4"


def test_rotation_override_and_hybrid_hotkey_tagging() -> None:
    """Verify rotation selector pill and hybrid keyboard shortcut ('v') handler."""
    reset_studio()
    state = override_rotation(RotationOverrideRequest(team="HOME", rotation=5))
    assert state["tactical"]["HOME"]["rotation"] == "R5"
    assert state["tactical"]["HOME"]["setter_zone"] == 3  # R5 -> Setter in Zone 3

    tagged = process_hybrid_tag(HybridDecisionRequest(hotkey="v"))
    assert len(tagged["recent_attacks"]) > 0
