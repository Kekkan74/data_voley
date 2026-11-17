"""
Human-in-the-Loop JSONL Dataset Exporter for K1/K2 Tactical Sequences.

Persists validated RallyK1Sequence samples with normalized homography coordinates
(x/9.0, y/18.0), stream timestamp, and tactical labels into:
    data/training_dataset/k1_annotations.jsonl
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from data_voley.tactical_state.k1_models import RallyK1Sequence

COURT_WIDTH_METERS: float = 9.0
COURT_LENGTH_METERS: float = 18.0


def _normalize_xy(xy: Tuple[float, float]) -> List[float]:
    """Normalize 2D court meters into [0.0, 1.0] range relative to 9m x 18m court."""
    nx = round(max(0.0, min(1.0, xy[0] / COURT_WIDTH_METERS)), 5)
    ny = round(max(0.0, min(1.0, xy[1] / COURT_LENGTH_METERS)), 5)
    return [nx, ny]


class K1DatasetExporter:
    """Appends validated K1/K2 sequences as JSONL training records."""

    def __init__(
        self,
        output_path: Path | str = Path("data/training_dataset/k1_annotations.jsonl"),
    ) -> None:
        self.output_path = Path(output_path)

    def build_record(
        self, sequence: RallyK1Sequence, timestamp_sec: Optional[float] = None
    ) -> Dict[str, Any]:
        """Convert a RallyK1Sequence into a normalized training sample dictionary."""
        ts = timestamp_sec if timestamp_sec is not None else sequence.timestamp_sec
        return {
            "timestamp_sec": round(float(ts), 3),
            "set_number": sequence.set_number,
            "rotation": sequence.rotation.name,
            "side_out_won": bool(sequence.side_out_won),
            "reception_zone": sequence.reception.receptor_zone,
            "reception_origin_norm": _normalize_xy(sequence.reception.origin_xy),
            "reception_target_norm": _normalize_xy(sequence.reception.target_xy),
            "reception_quality": (
                sequence.reception.quality.value if sequence.reception.quality else None
            ),
            "set_origin_norm": (
                _normalize_xy(sequence.setting.setter_origin_xy)
                if sequence.setting
                else None
            ),
            "set_target_zone": sequence.setting.target_zone if sequence.setting else None,
            "set_is_in_system": sequence.setting.is_in_system if sequence.setting else None,
            "set_tempo": sequence.setting.tempo.value if sequence.setting else None,
            "set_quality": sequence.setting.quality.value if sequence.setting else None,
            "attack_origin_zone": sequence.attack.origin_zone if sequence.attack else None,
            "attack_target_norm": (
                _normalize_xy(sequence.attack.target_xy) if sequence.attack else None
            ),
            "attack_trajectory": (
                sequence.attack.trajectory_type.value
                if sequence.attack and sequence.attack.trajectory_type
                else None
            ),
            "attack_outcome": sequence.attack.outcome.value if sequence.attack else None,
            "blockers_faced": sequence.attack.blockers_faced if sequence.attack else None,
        }

    def export_sample(
        self, sequence: RallyK1Sequence, timestamp_sec: Optional[float] = None
    ) -> Dict[str, Any]:
        """Serialize `sequence` and append a single JSON line to `self.output_path`."""
        record = self.build_record(sequence, timestamp_sec=timestamp_sec)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record
