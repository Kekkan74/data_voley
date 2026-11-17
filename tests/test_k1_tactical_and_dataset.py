"""
TDD Tests for K1/K2 Chained Sequences (Reception -> Setting -> Attack),
Advanced K1 Analytics, VisionDetectorPort, JSONL Dataset Exporter, and FastAPI K1 Endpoints.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def test_reception_quality_geometric_evaluation() -> None:
    """Verify Euclidean distance to Zone 2.5 [6.0, 8.5] classifies #, +, -, and /."""
    from data_voley.tactical_state.k1_models import (
        ReceptionQuality,
        evaluate_reception_quality,
    )

    # <= 1.5m from [6.0, 8.5] -> '#' (Perfect)
    assert evaluate_reception_quality((6.0, 8.5)) == ReceptionQuality.PERFECT
    assert evaluate_reception_quality((5.0, 8.0)) == ReceptionQuality.PERFECT

    # 1.5m < d <= 3.0m -> '+' (Positive)
    assert evaluate_reception_quality((4.0, 8.0)) == ReceptionQuality.POSITIVE
    assert evaluate_reception_quality((6.0, 6.0)) == ReceptionQuality.POSITIVE

    # d > 3.0m (same side y <= 9.0) -> '-' (Negative)
    assert evaluate_reception_quality((1.5, 4.0)) == ReceptionQuality.NEGATIVE

    # Crossed net (y > 9.0) -> '/' (Overpass)
    assert evaluate_reception_quality((5.0, 10.5)) == ReceptionQuality.OVERPASS


def test_attack_trajectory_geometric_classification() -> None:
    """Verify geometric classification of LINE, CROSS_LONG, CROSS_SHORT, and TIP."""
    from data_voley.tactical_state.k1_models import (
        AttackTrajectory,
        evaluate_attack_trajectory,
    )

    # Very close to net (y <= 10.8) -> TIP
    assert evaluate_attack_trajectory(4, (3.0, 10.2)) == AttackTrajectory.TIP

    # From Zone 4 (left front):
    assert evaluate_attack_trajectory(4, (1.2, 16.0)) == AttackTrajectory.LINE
    assert evaluate_attack_trajectory(4, (7.5, 12.2)) == AttackTrajectory.CROSS_SHORT
    assert evaluate_attack_trajectory(4, (7.8, 16.5)) == AttackTrajectory.CROSS_LONG

    # From Zone 2 (right front):
    assert evaluate_attack_trajectory(2, (7.8, 15.5)) == AttackTrajectory.LINE
    assert evaluate_attack_trajectory(2, (1.5, 12.0)) == AttackTrajectory.CROSS_SHORT
    assert evaluate_attack_trajectory(2, (1.2, 16.5)) == AttackTrajectory.CROSS_LONG


def test_rally_k1_sequence_derives_in_system_and_defaults() -> None:
    """Verify RallyK1Sequence auto-derives quality, trajectory, and is_in_system."""
    from data_voley.domain.enums import Rotation
    from data_voley.tactical_state.k1_models import (
        AttackTrajectory,
        K1AttackEvent,
        K1AttackOutcome,
        RallyK1Sequence,
        ReceptionEvent,
        ReceptionQuality,
        SetEvent,
        SetQuality,
        SetTempo,
    )

    seq = RallyK1Sequence(
        set_number=1,
        rotation=Rotation.R1,
        reception=ReceptionEvent(
            receptor_zone=5,
            origin_xy=(1.5, 2.5),
            target_xy=(6.0, 8.2),  # <= 1.5m from [6.0, 8.5] -> PERFECT '#'
        ),
        setting=SetEvent(
            setter_origin_xy=(6.0, 8.2),
            target_zone=4,
            tempo=SetTempo.SECOND,
            quality=SetQuality.SUCCESSFUL,
        ),
        attack=K1AttackEvent(
            origin_zone=4,
            target_xy=(7.8, 16.5),
            blockers_faced=1,
            outcome=K1AttackOutcome.KILL,
        ),
        timestamp_sec=125.4,
    )

    assert seq.reception.quality == ReceptionQuality.PERFECT
    assert seq.setting is not None
    assert seq.setting.is_in_system is True
    assert seq.attack is not None
    assert seq.attack.trajectory_type == AttackTrajectory.CROSS_LONG

    # Out-of-system reception
    seq_oos = RallyK1Sequence(
        set_number=1,
        rotation=Rotation.R4,
        reception=ReceptionEvent(
            receptor_zone=6,
            origin_xy=(4.5, 2.0),
            target_xy=(1.5, 3.5),  # > 3.0m -> NEGATIVE '-'
        ),
        setting=SetEvent(
            setter_origin_xy=(2.0, 4.0),
            target_zone=4,
            tempo=SetTempo.THIRD,
            quality=SetQuality.IMPRECISE,
        ),
    )
    assert seq_oos.reception.quality == ReceptionQuality.NEGATIVE
    assert seq_oos.setting is not None
    assert seq_oos.setting.is_in_system is False


def test_k1_advanced_analytics_calculations() -> None:
    """Verify setter distribution, reception/side-out by rotation, and attack efficiency."""
    from data_voley.domain.enums import Rotation
    from data_voley.stats_engine.analytics import compute_k1_advanced_analytics
    from data_voley.tactical_state.k1_models import (
        K1AttackEvent,
        K1AttackOutcome,
        RallyK1Sequence,
        ReceptionEvent,
        ReceptionQuality,
        SetEvent,
        SetQuality,
        SetTempo,
    )

    sequences = [
        # In-system (#) -> Z3 1st tempo, 1 blocker -> Kill (#)
        RallyK1Sequence(
            set_number=1,
            rotation=Rotation.R1,
            reception=ReceptionEvent(
                receptor_zone=5,
                origin_xy=(1.5, 2.5),
                target_xy=(6.0, 8.5),
                quality=ReceptionQuality.PERFECT,
            ),
            setting=SetEvent(
                setter_origin_xy=(6.0, 8.5),
                target_zone=3,
                is_in_system=True,
                tempo=SetTempo.FIRST,
                quality=SetQuality.SUCCESSFUL,
            ),
            attack=K1AttackEvent(
                origin_zone=3,
                target_xy=(4.5, 15.0),
                blockers_faced=1,
                outcome=K1AttackOutcome.KILL,
            ),
        ),
        # In-system (+) -> Z4 2nd tempo, 2 blockers -> Error (=)
        RallyK1Sequence(
            set_number=1,
            rotation=Rotation.R1,
            reception=ReceptionEvent(
                receptor_zone=6,
                origin_xy=(4.5, 2.5),
                target_xy=(4.2, 8.0),
                quality=ReceptionQuality.POSITIVE,
            ),
            setting=SetEvent(
                setter_origin_xy=(4.2, 8.0),
                target_zone=4,
                is_in_system=True,
                tempo=SetTempo.SECOND,
                quality=SetQuality.SUCCESSFUL,
            ),
            attack=K1AttackEvent(
                origin_zone=4,
                target_xy=(1.2, 16.0),
                blockers_faced=2,
                outcome=K1AttackOutcome.ERROR,
            ),
        ),
        # Out-of-system (-) -> Z4 3rd tempo, 2 blockers -> Kill (#)
        RallyK1Sequence(
            set_number=2,
            rotation=Rotation.R2,
            reception=ReceptionEvent(
                receptor_zone=5,
                origin_xy=(1.5, 2.5),
                target_xy=(1.5, 3.5),
                quality=ReceptionQuality.NEGATIVE,
            ),
            setting=SetEvent(
                setter_origin_xy=(2.0, 4.0),
                target_zone=4,
                is_in_system=False,
                tempo=SetTempo.THIRD,
                quality=SetQuality.IMPRECISE,
            ),
            attack=K1AttackEvent(
                origin_zone=4,
                target_xy=(7.5, 16.5),
                blockers_faced=2,
                outcome=K1AttackOutcome.KILL,
            ),
        ),
    ]

    # Full match analytics
    res = compute_k1_advanced_analytics(sequences, set_number=None)
    setter = res["setter_distribution"]
    assert setter["in_system"]["total"] == 2
    assert setter["in_system"]["zones"]["Z3"]["pct"] == 50.0
    assert setter["in_system"]["zones"]["Z4"]["pct"] == 50.0
    assert setter["out_of_system"]["total"] == 1
    assert setter["out_of_system"]["zones"]["Z4"]["pct"] == 100.0
    # 1 out of 3 attacks faced <= 1 blocker -> 33.3%
    assert setter["setter_advantage_rate_pct"] == pytest.approx(33.3, abs=0.2)

    rec = res["reception_performance"]
    assert rec["by_rotation"]["R1"]["total"] == 2
    assert rec["by_rotation"]["R1"]["positive_pct"] == 100.0
    assert rec["by_rotation"]["R1"]["side_out_pct"] == 50.0
    assert rec["by_rotation"]["R2"]["side_out_pct"] == 100.0

    atk = res["attack_efficiency"]
    # Overall: 2 kills, 1 error, 0 blocked / 3 = (2 - 1) / 3 = 0.333
    assert atk["overall_efficiency"] == pytest.approx(0.333, abs=0.01)
    # In-system attack efficiency: (1 - 1) / 2 = 0.0
    assert atk["correlation_reception_to_attack"]["in_system_efficiency"] == 0.0
    # Out-of-system attack efficiency: 1 / 1 = 1.0
    assert atk["correlation_reception_to_attack"]["out_of_system_efficiency"] == 1.0

    # Set 1 filter only
    res_s1 = compute_k1_advanced_analytics(sequences, set_number=1)
    assert res_s1["setter_distribution"]["out_of_system"]["total"] == 0
    assert res_s1["attack_efficiency"]["overall_efficiency"] == 0.0


def test_k1_dataset_exporter_writes_normalized_jsonl(tmp_path: Path) -> None:
    """Verify K1DatasetExporter appends normalized coordinates and labels to JSONL."""
    from data_voley.domain.enums import Rotation
    from data_voley.tactical_state.k1_models import (
        K1AttackEvent,
        K1AttackOutcome,
        RallyK1Sequence,
        ReceptionEvent,
        SetEvent,
        SetTempo,
    )
    from data_voley.vision_detector.dataset_exporter import K1DatasetExporter

    out_file = tmp_path / "k1_annotations.jsonl"
    exporter = K1DatasetExporter(output_path=out_file)

    seq = RallyK1Sequence(
        set_number=1,
        rotation=Rotation.R3,
        reception=ReceptionEvent(
            receptor_zone=5,
            origin_xy=(4.5, 4.5),
            target_xy=(6.0, 8.5),
        ),
        setting=SetEvent(
            setter_origin_xy=(6.0, 8.5),
            target_zone=4,
            tempo=SetTempo.SECOND,
        ),
        attack=K1AttackEvent(
            origin_zone=4,
            target_xy=(7.5, 13.5),
            blockers_faced=1,
            outcome=K1AttackOutcome.KILL,
        ),
        timestamp_sec=42.5,
    )

    record = exporter.export_sample(seq)
    assert out_file.exists()
    lines = out_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1

    parsed = json.loads(lines[0])
    assert parsed["timestamp_sec"] == 42.5
    assert parsed["rotation"] == "R3"
    assert parsed["reception_quality"] == "#"
    assert parsed["set_tempo"] == "2nd"
    assert parsed["attack_outcome"] == "#"
    # Normalized coordinates (x/9.0, y/18.0)
    assert parsed["reception_origin_norm"] == pytest.approx([0.5, 0.25], abs=0.01)
    assert record["attack_outcome"] == "#"


def test_fastapi_k1_endpoints_register_and_query(tmp_path: Path) -> None:
    """Verify POST /api/rallies/k1 and GET /api/stats/k1-advanced endpoints."""
    from data_voley.api.server import (
        K1AttackInput,
        K1RallyRequest,
        K1ReceptionInput,
        K1SetInput,
        SESSION,
        get_k1_advanced_stats,
        register_k1_rally,
        reset_studio,
    )
    from data_voley.vision_detector.dataset_exporter import K1DatasetExporter

    reset_studio()
    SESSION.k1_exporter = K1DatasetExporter(output_path=tmp_path / "api_k1.jsonl")

    res = register_k1_rally(
        K1RallyRequest(
            set_number=1,
            rotation=1,
            timestamp_sec=88.0,
            reception=K1ReceptionInput(
                receptor_zone=5,
                origin_xy=(1.5, 2.5),
                target_xy=(6.0, 8.4),
            ),
            setting=K1SetInput(
                setter_origin_xy=(6.0, 8.4),
                target_zone=4,
                tempo="2nd",
            ),
            attack=K1AttackInput(
                origin_zone=4,
                target_xy=(7.8, 16.2),
                blockers_faced=1,
                outcome="#",
            ),
        )
    )
    assert "k1_stats" in res
    assert (tmp_path / "api_k1.jsonl").exists()

    adv = get_k1_advanced_stats(set_number=1)
    assert adv["setter_distribution"]["in_system"]["total"] >= 1
    assert adv["attack_efficiency"]["total_attacks"] >= 1
