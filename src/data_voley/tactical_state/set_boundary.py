"""
Multimodal Set Boundary Detector (Pure Python + Pydantic v2).

Detects set transitions independently of reaching 25/15 points (which can be
missed due to video cuts) by combining:
  1. Team separation to benches + prolonged pause interval.
  2. Court side swap (HOME <-> AWAY spatial inversion).
  3. Formation reset / new starting rotation inconsistent with expected side-out.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from pydantic import BaseModel, Field

from data_voley.domain.enums import Rotation


class SetBoundarySignal(BaseModel):
    """Observed spatial, temporal, and tactical cues between rallies."""

    current_score: Tuple[int, int] = (0, 0)
    pause_duration_sec: float = Field(ge=0.0, default=0.0)
    teams_separated_to_benches: bool = False
    court_sides_swapped: bool = False
    expected_home_rotation: Optional[Rotation] = None
    observed_home_rotation: Optional[Rotation] = None
    formation_reset_detected: bool = False


class SetBoundaryDetectionResult(BaseModel):
    """Evaluation output indicating whether a set boundary occurred."""

    is_new_set: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reasons: List[str] = Field(default_factory=list)


class SetBoundaryDetector:
    """Evaluates multimodal cues to detect set transitions without relying solely on 25/15 pts."""

    def evaluate(self, signal: SetBoundarySignal) -> SetBoundaryDetectionResult:
        reasons: List[str] = []
        score_weight = 0.0

        rotation_mismatch = (
            signal.expected_home_rotation is not None
            and signal.observed_home_rotation is not None
            and signal.expected_home_rotation != signal.observed_home_rotation
        )

        if signal.court_sides_swapped:
            score_weight += 0.55
            reasons.append("Cambio de campo detectado entre equipos")

        if signal.teams_separated_to_benches and signal.pause_duration_sec >= 90.0:
            score_weight += 0.45
            reasons.append(
                f"Separación hacia banquillos y pausa de {int(signal.pause_duration_sec)}s"
            )
        elif signal.teams_separated_to_benches:
            score_weight += 0.25
            reasons.append("Separación temporal de equipos fuera de pista")

        if signal.formation_reset_detected or rotation_mismatch:
            score_weight += 0.30
            reasons.append("Reinicio de formación / nueva rotación inicial en pista")

        is_new_set = (
            signal.court_sides_swapped
            or (signal.teams_separated_to_benches and signal.pause_duration_sec >= 90.0)
            or (
                (signal.formation_reset_detected or rotation_mismatch)
                and (signal.pause_duration_sec >= 45.0 or signal.teams_separated_to_benches)
            )
        )

        return SetBoundaryDetectionResult(
            is_new_set=is_new_set,
            confidence=round(min(1.0, score_weight), 2),
            reasons=reasons,
        )
