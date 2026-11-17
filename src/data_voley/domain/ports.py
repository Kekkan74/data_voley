"""
Hexagonal Architecture Ports (Protocols) for Dependency Inversion.

Defines strict structural subtyping contracts between VideoIngestion,
CourtMapper, TacticalState, ActionDetector, and MatchStatsEngine.
"""

from __future__ import annotations

from typing import Dict, Iterator, Optional, Protocol, Tuple

import numpy as np

from data_voley.domain.enums import CourtZone, PlayerRole, Rotation, TeamSide
from data_voley.domain.models import (
    CourtPoint2D,
    PixelPoint,
    RallyTransitionEvent,
    RotationFormationSnapshot,
    SpatialZoneMapping,
)


class ICourtMapperPort(Protocol):
    """Port contract for 3D camera-to-2D court homography and FIVB zone resolution."""

    def pixel_to_court(self, pixel: PixelPoint) -> CourtPoint2D:
        """Project a 2D pixel coordinate (u, v) into real-world court meters (x, y)."""
        ...

    def resolve_zone(self, point: CourtPoint2D) -> SpatialZoneMapping:
        """Map a 2D court metric point to TeamSide and FIVB CourtZone (1 to 6)."""
        ...

    def pixel_to_zone(self, pixel: PixelPoint) -> SpatialZoneMapping:
        """Project a camera pixel directly into a SpatialZoneMapping."""
        ...


class ITacticalStatePort(Protocol):
    """Port contract for the FIVB Match, Set, and Rotation (R1-R6) State Machine."""

    @property
    def current_set(self) -> int:
        """Return active 1-based set number."""
        ...

    @property
    def serving_team(self) -> TeamSide:
        """Return the team currently in possession of service."""
        ...

    def get_rotation(self, team: TeamSide) -> Rotation:
        """Return the current tactical rotation (R1-R6) for the given team."""
        ...

    def get_score(self) -> Tuple[int, int]:
        """Return (home_points, away_points) in the current set."""
        ...

    def register_rally_point(self, winner: TeamSide) -> RallyTransitionEvent:
        """Process a point won by `winner`, updating score, sets, and side-out rotations."""
        ...

    def infer_formation(
        self, team: TeamSide, use_libero_in_back_row: bool = True
    ) -> RotationFormationSnapshot:
        """Infer the tactical role present in each CourtZone (1-6) for the team's active rotation."""
        ...

    def infer_role_at_zone(self, team: TeamSide, zone: CourtZone) -> PlayerRole:
        """Infer the player role attacking or defending from a specific zone."""
        ...


class IVideoIngestionPort(Protocol):
    """Port contract for HLS (.m3u8) or file video stream ingestion."""

    def frames(self) -> Iterator[Tuple[int, float, np.ndarray]]:
        """Yield (frame_index, timestamp_ms, bgr_frame) tuples."""
        ...

    def release(self) -> None:
        """Release underlying video capture resources."""
        ...
