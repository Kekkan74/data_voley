"""
ActionDetector & HybridTaggingController Module.

Combines automated spatial action detection (YOLO/TrackNet trajectory mapping)
with a Human-in-the-Loop keyboard shortcut validation controller to ensure
100% ground-truth data reliability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from data_voley.domain.enums import (
    AttackOutcome,
    CourtZone,
    ServeOutcome,
    TagValidationStatus,
    TeamSide,
)
from data_voley.domain.models import AttackEvent, PixelPoint, ServeEvent, SpatialZoneMapping
from data_voley.domain.ports import ICourtMapperPort
from data_voley.stats_engine.analytics import MatchStatsEngine


@dataclass
class PendingAttackProposal:
    """Candidate attack event proposed by the CV pipeline awaiting analyst validation."""

    proposal_id: int
    team: TeamSide
    origin_zone: CourtZone
    outcome: AttackOutcome
    target_zone: Optional[CourtZone]
    timestamp_ms: float


class ActionDetector:
    """
    Spatial vision-to-tactics translator.

    Receives bounding box foot contact points and ball impact coordinates from
    YOLO/TrackNet and resolves their FIVB zones via `ICourtMapperPort`.
    """

    def __init__(self, court_mapper: ICourtMapperPort) -> None:
        self._mapper = court_mapper

    def detect_attack_from_pixels(
        self,
        attacker_foot_px: PixelPoint,
        ball_landing_px: PixelPoint,
        ball_touched_by_defense: bool = False,
    ) -> Optional[ Tuple[TeamSide, CourtZone, AttackOutcome, Optional[CourtZone]] ]:
        """
        Infer attack origin team, origin FIVB zone (1-6), outcome, and landing zone
        from camera pixel detections.
        """
        origin_map: SpatialZoneMapping = self._mapper.pixel_to_zone(attacker_foot_px)
        if not origin_map.is_in_bounds or origin_map.team_side is None or origin_map.zone is None:
            return None

        landing_map: SpatialZoneMapping = self._mapper.pixel_to_zone(ball_landing_px)
        attacker_team = origin_map.team_side
        origin_zone = origin_map.zone

        if not landing_map.is_in_bounds:
            outcome = (
                AttackOutcome.KILL if ball_touched_by_defense else AttackOutcome.ERROR_OUT
            )
            return attacker_team, origin_zone, outcome, None

        if landing_map.team_side == attacker_team:
            # Ball landed back on the attacker's own side (blocked or net rebound)
            outcome = (
                AttackOutcome.BLOCKED if landing_map.is_front_zone else AttackOutcome.ERROR_NET
            )
            return attacker_team, origin_zone, outcome, landing_map.zone

        # Ball crossed to opponent court
        outcome = (
            AttackOutcome.IN_PLAY if ball_touched_by_defense else AttackOutcome.KILL
        )
        return attacker_team, origin_zone, outcome, landing_map.zone


class HybridTaggingController:
    """
    Interactive Human-in-the-Loop controller for validating, correcting, or
    discarding AI-proposed volleyball events via keyboard shortcuts:
      - 'v' : Validate AI proposal as-is (HUMAN_VALIDATED)
      - 'c' : Correct zone and/or outcome before committing (HUMAN_CORRECTED)
      - 'd' : Discard false-positive AI proposal
    """

    def __init__(self, stats_engine: MatchStatsEngine) -> None:
        self._stats = stats_engine
        self._pending_attacks: Dict[int, PendingAttackProposal] = {}
        self._next_id: int = 1

    def propose_attack(
        self,
        team: TeamSide,
        origin_zone: CourtZone,
        outcome: AttackOutcome,
        target_zone: Optional[CourtZone] = None,
        timestamp_ms: float = 0.0,
    ) -> int:
        """Queue an AI-proposed attack event for analyst review."""
        pid = self._next_id
        self._next_id += 1
        self._pending_attacks[pid] = PendingAttackProposal(
            proposal_id=pid,
            team=team,
            origin_zone=origin_zone,
            outcome=outcome,
            target_zone=target_zone,
            timestamp_ms=timestamp_ms,
        )
        return pid

    def apply_hotkey_action(
        self,
        proposal_id: int,
        hotkey: str,
        corrected_zone: Optional[CourtZone] = None,
        corrected_outcome: Optional[AttackOutcome] = None,
    ) -> Optional[AttackEvent]:
        """
        Process analyst keyboard shortcut on a pending AI proposal.
        """
        if proposal_id not in self._pending_attacks:
            raise KeyError(f"Proposal ID {proposal_id} not found in pending queue.")

        proposal = self._pending_attacks.pop(proposal_id)
        key = hotkey.strip().lower()

        if key == "d":
            return None

        if key == "v":
            return self._stats.record_attack(
                team=proposal.team,
                origin_zone=proposal.origin_zone,
                outcome=proposal.outcome,
                target_zone=proposal.target_zone,
                validation_status=TagValidationStatus.HUMAN_VALIDATED,
                timestamp_ms=proposal.timestamp_ms,
            )

        if key == "c":
            final_zone = corrected_zone if corrected_zone is not None else proposal.origin_zone
            final_outcome = (
                corrected_outcome if corrected_outcome is not None else proposal.outcome
            )
            return self._stats.record_attack(
                team=proposal.team,
                origin_zone=final_zone,
                outcome=final_outcome,
                target_zone=proposal.target_zone,
                validation_status=TagValidationStatus.HUMAN_CORRECTED,
                timestamp_ms=proposal.timestamp_ms,
            )

        raise ValueError(f"Unsupported hotkey '{hotkey}'. Use 'v' (validate), 'c' (correct), or 'd' (discard).")
