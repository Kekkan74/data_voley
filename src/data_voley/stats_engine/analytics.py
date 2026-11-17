"""
MatchStatsEngine Module — Spatial & Rotational Cross-Tabulation Analytics.

Aggregates serve and attack events, crossing spatial origin/target zones (Z1-Z6)
with active tactical rotations (R1-R6) and inferred player roles.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from data_voley.domain.enums import (
    AttackOutcome,
    CourtZone,
    Rotation,
    ServeOutcome,
    TagValidationStatus,
    TeamSide,
)
from data_voley.domain.models import AttackEvent, CourtPoint2D, ServeEvent
from data_voley.domain.ports import ITacticalStatePort


class MatchStatsEngine:
    """
    Core analytical engine correlating spatial actions with tactical rotations (R1-R6).
    """

    def __init__(self, tactical_port: ITacticalStatePort) -> None:
        self._tactical = tactical_port
        self._serve_events: List[ServeEvent] = []
        self._attack_events: List[AttackEvent] = []
        self._action_seq: int = 0

    @property
    def serve_events(self) -> List[ServeEvent]:
        """Return all recorded service events."""
        return list(self._serve_events)

    @property
    def attack_events(self) -> List[AttackEvent]:
        """Return all recorded attack/spike events."""
        return list(self._attack_events)

    def record_serve(
        self,
        team: TeamSide,
        outcome: ServeOutcome,
        target_zone: Optional[CourtZone] = None,
        origin_coord: Optional[CourtPoint2D] = None,
        validation_status: TagValidationStatus = TagValidationStatus.HUMAN_VALIDATED,
        timestamp_ms: float = 0.0,
    ) -> ServeEvent:
        """
        Record a service event bound to the team's current tactical rotation (R1-R6).
        """
        self._action_seq += 1
        active_rotation = self._tactical.get_rotation(team)
        event = ServeEvent(
            set_number=self._tactical.current_set,
            rally_number=self._action_seq,
            team=team,
            rotation=active_rotation,
            outcome=outcome,
            origin_coord=origin_coord,
            target_zone=target_zone,
            validation_status=validation_status,
            timestamp_ms=timestamp_ms,
        )
        self._serve_events.append(event)
        return event

    def record_attack(
        self,
        team: TeamSide,
        origin_zone: CourtZone,
        outcome: AttackOutcome,
        target_zone: Optional[CourtZone] = None,
        origin_coord: Optional[CourtPoint2D] = None,
        validation_status: TagValidationStatus = TagValidationStatus.HUMAN_VALIDATED,
        timestamp_ms: float = 0.0,
    ) -> AttackEvent:
        """
        Record an attack event from `origin_zone`, automatically inferring the attacker's
        tactical role from the team's active rotation (R1-R6).
        """
        self._action_seq += 1
        active_rotation = self._tactical.get_rotation(team)
        inferred_role = self._tactical.infer_role_at_zone(team, origin_zone)

        event = AttackEvent(
            set_number=self._tactical.current_set,
            rally_number=self._action_seq,
            team=team,
            rotation=active_rotation,
            origin_zone=origin_zone,
            outcome=outcome,
            inferred_attacker_role=inferred_role,
            target_zone=target_zone,
            origin_coord=origin_coord,
            validation_status=validation_status,
            timestamp_ms=timestamp_ms,
        )
        self._attack_events.append(event)
        return event

    def get_serve_errors_by_rotation(
        self, team: TeamSide, set_number: Optional[int] = None
    ) -> Dict[Rotation, Dict[str, int]]:
        """
        Cross-tabulate service outcomes and errors across Rotations R1 to R6.
        """
        matrix: Dict[Rotation, Dict[str, int]] = {
            rot: {
                "TOTAL_SERVES": 0,
                "ACE": 0,
                "IN_PLAY": 0,
                "ERROR_NET": 0,
                "ERROR_OUT": 0,
                "ERROR_FOOT": 0,
                "TOTAL_ERRORS": 0,
            }
            for rot in Rotation
        }

        for ev in self._serve_events:
            if ev.team != team:
                continue
            if set_number is not None and ev.set_number != set_number:
                continue

            bucket = matrix[ev.rotation]
            bucket["TOTAL_SERVES"] += 1
            bucket[ev.outcome.value] += 1
            if ev.outcome.is_error:
                bucket["TOTAL_ERRORS"] += 1

        return matrix

    def get_most_vulnerable_serve_rotation(
        self, team: TeamSide, set_number: Optional[int] = None
    ) -> Tuple[Rotation, int]:
        """
        Identify the rotation (R1-R6) with the highest count of service errors.
        """
        breakdown = self.get_serve_errors_by_rotation(team, set_number=set_number)
        worst_rot = max(
            Rotation,
            key=lambda r: (breakdown[r]["TOTAL_ERRORS"], -r.value),
        )
        return worst_rot, breakdown[worst_rot]["TOTAL_ERRORS"]

    def get_attack_stats_by_zone(
        self, team: TeamSide, set_number: Optional[int] = None
    ) -> Dict[CourtZone, Dict[str, float | int]]:
        """
        Compute attack attempts, kills, errors, and hitting efficiency per CourtZone (1-6).

        Formula:
            Efficiency = (Kills - Errors) / Attempts
        """
        stats: Dict[CourtZone, Dict[str, float | int]] = {
            zone: {
                "attempts": 0,
                "kills": 0,
                "errors": 0,
                "in_play": 0,
                "efficiency": 0.0,
            }
            for zone in CourtZone
        }

        for ev in self._attack_events:
            if ev.team != team:
                continue
            if set_number is not None and ev.set_number != set_number:
                continue

            z_bucket = stats[ev.origin_zone]
            z_bucket["attempts"] = int(z_bucket["attempts"]) + 1
            if ev.outcome == AttackOutcome.KILL:
                z_bucket["kills"] = int(z_bucket["kills"]) + 1
            elif ev.outcome.is_error:
                z_bucket["errors"] = int(z_bucket["errors"]) + 1
            else:
                z_bucket["in_play"] = int(z_bucket["in_play"]) + 1

        for zone, z_bucket in stats.items():
            att = int(z_bucket["attempts"])
            if att > 0:
                kills = int(z_bucket["kills"])
                errs = int(z_bucket["errors"])
                z_bucket["efficiency"] = round((kills - errs) / att, 4)

        return stats

    def get_attack_stats_by_rotation(
        self, team: TeamSide
    ) -> Dict[Rotation, Dict[str, float | int]]:
        """
        Cross-tabulate offensive spike performance across Rotations R1 to R6.
        """
        rot_stats: Dict[Rotation, Dict[str, float | int]] = {
            rot: {
                "attempts": 0,
                "kills": 0,
                "errors": 0,
                "efficiency": 0.0,
            }
            for rot in Rotation
        }

        for ev in self._attack_events:
            if ev.team != team:
                continue
            r_bucket = rot_stats[ev.rotation]
            r_bucket["attempts"] = int(r_bucket["attempts"]) + 1
            if ev.outcome == AttackOutcome.KILL:
                r_bucket["kills"] = int(r_bucket["kills"]) + 1
            elif ev.outcome.is_error:
                r_bucket["errors"] = int(r_bucket["errors"]) + 1

        for rot, r_bucket in rot_stats.items():
            att = int(r_bucket["attempts"])
            if att > 0:
                kills = int(r_bucket["kills"])
                errs = int(r_bucket["errors"])
                r_bucket["efficiency"] = round((kills - errs) / att, 4)

        return rot_stats
