"""
Unit & Integration Tests for the DataVoley Visual Studio API Endpoints.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data_voley.api.server import (
    ConnectStreamRequest,
    CourtClickRequest,
    HybridDecisionRequest,
    RallyRequest,
    RotationOverrideRequest,
    ScopeFilterRequest,
    connect_stream_endpoint,
    get_state,
    health_check,
    override_rotation,
    process_hybrid_tag,
    register_court_click_action,
    register_rally,
    reset_studio,
    set_analytics_scope,
)


def test_connect_stream_first_step_workflow() -> None:
    """Verify connecting a stream URL configures team names, start minute, and data."""
    state = connect_stream_endpoint(
        ConnectStreamRequest(
            stream_url="https://stream.fivb.org/live/match123.m3u8",
            start_minute="12:30",
            home_name="CV Madrid",
            away_name="Unicaja Almería",
            initial_rotation=2,
            initial_server="HOME",
            auto_analyze=True,
        )
    )
    assert state["match"]["stream_url"] == "https://stream.fivb.org/live/match123.m3u8"
    assert state["match"]["start_offset_seconds"] == 750.0
    assert state["match"]["start_time_formatted"] == "12:30"
    assert state["match"]["home_name"] == "CV Madrid"
    assert state["match"]["away_name"] == "Unicaja Almería"
    assert len(state["available_sets"]) == 3


def test_studio_health_and_initial_seeded_charts() -> None:
    """Verify the visual studio initializes with multi-set and full-match data."""
    assert health_check()["status"] == "ok"
    state = reset_studio()

    assert state["active_scope"] == "ALL"
    assert state["available_sets"] == [1, 2, 3]
    assert "set_comparison" in state["analytics"]
    assert "ALL" in state["analytics"]["set_comparison"]
    assert "1" in state["analytics"]["set_comparison"]
    assert "2" in state["analytics"]["set_comparison"]
    assert "3" in state["analytics"]["set_comparison"]


def test_per_set_vs_full_match_scope_filtering() -> None:
    """Verify filtering charts by Set 1, Set 2, Set 3, and Full Match ('ALL')."""
    reset_studio()

    # Filter by Set 1 (where R3 had the highest serve errors)
    state_s1 = set_analytics_scope(ScopeFilterRequest(scope="1"))
    assert state_s1["active_scope"] == "1"
    assert state_s1["analytics"]["most_vulnerable_rotation"]["rotation"] == "R3"
    assert all(t["set_number"] == 1 for t in state_s1["trajectories"])

    # Filter by Set 2 (where R5 had the highest serve errors)
    state_s2 = set_analytics_scope(ScopeFilterRequest(scope="2"))
    assert state_s2["active_scope"] == "2"
    assert state_s2["analytics"]["most_vulnerable_rotation"]["rotation"] == "R5"
    assert all(t["set_number"] == 2 for t in state_s2["trajectories"])

    # Filter back to ALL (Entire Match)
    state_all = set_analytics_scope(ScopeFilterRequest(scope="ALL"))
    assert state_all["active_scope"] == "ALL"
    s1_atks = state_all["analytics"]["set_comparison"]["1"]["total_attacks"]
    s2_atks = state_all["analytics"]["set_comparison"]["2"]["total_attacks"]
    s3_atks = state_all["analytics"]["set_comparison"]["3"]["total_attacks"]
    tot_atks = state_all["analytics"]["set_comparison"]["ALL"]["total_attacks"]
    assert tot_atks == s1_atks + s2_atks + s3_atks


def test_interactive_court_click_maps_homography_and_updates_charts() -> None:
    """Verify clicking on the 2D court maps to FIVB Zone 4 and records the spike."""
    reset_studio()
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
    assert state["tactical"]["HOME"]["setter_zone"] == 3

    tagged = process_hybrid_tag(HybridDecisionRequest(hotkey="v"))
    assert len(tagged["recent_attacks"]) > 0
