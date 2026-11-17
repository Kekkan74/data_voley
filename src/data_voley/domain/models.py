"""
Immutable Domain Models and Value Objects for Volleyball Analytics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from data_voley.domain.enums import (
    AttackOutcome,
    CourtZone,
    PlayerRole,
    Rotation,
    ServeOutcome,
    TagValidationStatus,
    TeamSide,
)


@dataclass(frozen=True)
class PixelPoint:
    """2D coordinate in the camera image space (pixels)."""

    u: float
    v: float


@dataclass(frozen=True)
class CourtPoint2D:
    """
    2D metric coordinate on the real-world FIVB volleyball court plane (in meters).

    Standard Coordinate System:
    - x in [0.0, 9.0]: Width from left sideline (0.0m) to right sideline (9.0m).
    - y in [0.0, 18.0]: Length from HOME endline (0.0m) through Net (9.0m) to AWAY endline (18.0m).
    """

    x: float
    y: float

    @property
    def is_in_bounds(self) -> bool:
        """Return True if the point falls within or on the 9m x 18m court lines."""
        return 0.0 <= self.x <= 9.0 and 0.0 <= self.y <= 18.0


@dataclass(frozen=True)
class SpatialZoneMapping:
    """Result of mapping a camera pixel or 2D point to a specific court side and FIVB zone."""

    court_point: CourtPoint2D
    team_side: Optional[TeamSide]
    zone: Optional[CourtZone]
    is_in_bounds: bool
    is_front_zone: bool


@dataclass(frozen=True)
class ServeEvent:
    """Immutable domain event representing a service action."""

    set_number: int
    rally_number: int
    team: TeamSide
    rotation: Rotation
    outcome: ServeOutcome
    origin_coord: Optional[CourtPoint2D] = None
    target_zone: Optional[CourtZone] = None
    validation_status: TagValidationStatus = TagValidationStatus.AI_PROPOSED
    timestamp_ms: float = 0.0


@dataclass(frozen=True)
class AttackEvent:
    """Immutable domain event representing an attack/spike action."""

    set_number: int
    rally_number: int
    team: TeamSide
    rotation: Rotation
    origin_zone: CourtZone
    outcome: AttackOutcome
    inferred_attacker_role: PlayerRole
    target_zone: Optional[CourtZone] = None
    origin_coord: Optional[CourtPoint2D] = None
    validation_status: TagValidationStatus = TagValidationStatus.AI_PROPOSED
    timestamp_ms: float = 0.0


@dataclass(frozen=True)
class RallyTransitionEvent:
    """Immutable record emitted after a rally concludes and state transitions."""

    set_number: int
    rally_number: int
    serving_team_before: TeamSide
    point_winner: TeamSide
    side_out_occurred: bool
    home_score: int
    away_score: int
    home_rotation: Rotation
    away_rotation: Rotation
    set_finished: bool
    match_finished: bool


@dataclass(frozen=True)
class RotationFormationSnapshot:
    """Inferred spatial distribution of player roles across Zones 1-6 for a rotation."""

    team: TeamSide
    rotation: Rotation
    setter_zone: CourtZone
    setter_is_front_row: bool
    zone_roles: Dict[CourtZone, PlayerRole]
