"""
MatchStatsEngine Module — Spatial & Rotational Cross-Tabulation Analytics.

Aggregates serve, setting (Colocaciones), and attack events, crossing spatial
origin/target zones (Z1-Z6) with active tactical rotations (R1-R6) and inferred
player roles, with support for both per-set filtering and full-match aggregation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from data_voley.domain.enums import (
    AttackOutcome,
    CourtZone,
    Rotation,
    ServeOutcome,
    SetOutcome,
    TagValidationStatus,
    TeamSide,
)
from data_voley.domain.models import AttackEvent, CourtPoint2D, ServeEvent, SetActionEvent
from data_voley.domain.ports import ITacticalStatePort
from data_voley.tactical_state.k1_models import (
    AttackTrajectory,
    K1AttackOutcome,
    RallyK1Sequence,
    ReceptionQuality,
    SetTempo,
)

SETTING_TARGET_LABELS_ES: Dict[CourtZone, str] = {
    CourtZone.Z4: "Hacia Zona 4 (Punta Izquierda / Receptor)",
    CourtZone.Z2: "Hacia Zona 2 (Punta Derecha / Opuesto)",
    CourtZone.Z3: "Hacia Zona 3 (Centro / Primer Tiempo)",
    CourtZone.Z6: "Hacia Zona 6 (Zaguero Centro / Pipe)",
    CourtZone.Z1: "Hacia Zona 1 (Zaguero Derecha / Opuesto)",
    CourtZone.Z5: "Hacia Zona 5 (Bola Alta Emergencia)",
}


class MatchStatsEngine:
    """
    Core analytical engine correlating spatial actions (Serves, Setting/Colocaciones,
    and Attacks) with tactical rotations (R1-R6) across individual sets and full match.
    """

    def __init__(self, tactical_port: ITacticalStatePort) -> None:
        self._tactical = tactical_port
        self._serve_events: List[ServeEvent] = []
        self._attack_events: List[AttackEvent] = []
        self._setting_events: List[SetActionEvent] = []
        self._action_seq: int = 0

    @property
    def serve_events(self) -> List[ServeEvent]:
        """Return all recorded service events."""
        return list(self._serve_events)

    @property
    def attack_events(self) -> List[AttackEvent]:
        """Return all recorded attack/spike events."""
        return list(self._attack_events)

    @property
    def setting_events(self) -> List[SetActionEvent]:
        """Return all recorded setter pass (Colocación) events."""
        return list(self._setting_events)

    def get_recorded_sets(self) -> List[int]:
        """Return sorted list of all set numbers present in recorded events or active set."""
        sets = {self._tactical.current_set}
        for ev in self._serve_events:
            sets.add(ev.set_number)
        for ev in self._attack_events:
            sets.add(ev.set_number)
        for ev in self._setting_events:
            sets.add(ev.set_number)
        return sorted(sets)

    def record_serve(
        self,
        team: TeamSide,
        outcome: ServeOutcome,
        target_zone: Optional[CourtZone] = None,
        origin_coord: Optional[CourtPoint2D] = None,
        validation_status: TagValidationStatus = TagValidationStatus.HUMAN_VALIDATED,
        timestamp_ms: float = 0.0,
        set_override: Optional[int] = None,
    ) -> ServeEvent:
        """
        Record a service event bound to the team's current tactical rotation (R1-R6).
        """
        self._action_seq += 1
        active_rotation = self._tactical.get_rotation(team)
        set_num = set_override if set_override is not None else self._tactical.current_set
        event = ServeEvent(
            set_number=set_num,
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

    def record_setting_action(
        self,
        team: TeamSide,
        target_zone: CourtZone,
        outcome: SetOutcome,
        timestamp_ms: float = 0.0,
        set_override: Optional[int] = None,
    ) -> SetActionEvent:
        """
        Record a setter pass (Colocación) directed toward `target_zone`.
        """
        self._action_seq += 1
        active_rotation = self._tactical.get_rotation(team)
        set_num = set_override if set_override is not None else self._tactical.current_set
        event = SetActionEvent(
            set_number=set_num,
            rally_number=self._action_seq,
            team=team,
            rotation=active_rotation,
            target_zone=target_zone,
            outcome=outcome,
            timestamp_ms=timestamp_ms,
        )
        self._setting_events.append(event)
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
        set_override: Optional[int] = None,
    ) -> AttackEvent:
        """
        Record an attack event from `origin_zone`, automatically inferring the attacker's
        tactical role from the team's active rotation (R1-R6).
        """
        self._action_seq += 1
        active_rotation = self._tactical.get_rotation(team)
        inferred_role = self._tactical.infer_role_at_zone(team, origin_zone)
        set_num = set_override if set_override is not None else self._tactical.current_set

        event = AttackEvent(
            set_number=set_num,
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

    def get_setting_stats(
        self, team: TeamSide, set_number: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Compute comprehensive Setter statistics (Colocaciones Exitosas vs Falladas)
        broken down by destination zone (Z4, Z2, Z3, Z6, Z1), identifying the most
        successful target zone and the least successful target zone (viceversa).
        """
        target_zones_order = [
            CourtZone.Z4,
            CourtZone.Z2,
            CourtZone.Z3,
            CourtZone.Z6,
            CourtZone.Z1,
        ]
        by_zone: Dict[str, Dict[str, Any]] = {
            f"Z{z.value}": {
                "zone_num": z.value,
                "label": SETTING_TARGET_LABELS_ES[z],
                "total": 0,
                "assist_kills": 0,
                "good_sets": 0,
                "successful": 0,
                "error_doubles": 0,
                "error_bad_pass": 0,
                "failed": 0,
                "success_rate": 0.0,
            }
            for z in target_zones_order
        }

        for ev in self._setting_events:
            if ev.team != team:
                continue
            if set_number is not None and ev.set_number != set_number:
                continue

            z_key = f"Z{ev.target_zone.value}"
            if z_key not in by_zone:
                continue
            bucket = by_zone[z_key]
            bucket["total"] += 1
            if ev.outcome == SetOutcome.ASSIST_KILL:
                bucket["assist_kills"] += 1
                bucket["successful"] += 1
            elif ev.outcome == SetOutcome.GOOD_SET:
                bucket["good_sets"] += 1
                bucket["successful"] += 1
            elif ev.outcome == SetOutcome.ERROR_DOUBLES:
                bucket["error_doubles"] += 1
                bucket["failed"] += 1
            elif ev.outcome == SetOutcome.ERROR_BAD_PASS:
                bucket["error_bad_pass"] += 1
                bucket["failed"] += 1

        total_sets = 0
        total_successful = 0
        total_failed = 0

        active_zones = []
        for z_key, bucket in by_zone.items():
            tot = int(bucket["total"])
            succ = int(bucket["successful"])
            fail = int(bucket["failed"])
            total_sets += tot
            total_successful += succ
            total_failed += fail
            if tot > 0:
                bucket["success_rate"] = round(succ / tot, 4)
                active_zones.append((z_key, bucket))

        overall_rate = round(total_successful / total_sets, 4) if total_sets > 0 else 0.0

        if active_zones:
            best_key, best_b = max(
                active_zones,
                key=lambda item: (item[1]["success_rate"], item[1]["successful"]),
            )
            worst_key, worst_b = min(
                active_zones,
                key=lambda item: (item[1]["success_rate"], -item[1]["failed"]),
            )
            most_successful = {
                "zone": best_key,
                "label": best_b["label"],
                "success_rate": best_b["success_rate"],
                "successful": best_b["successful"],
                "total": best_b["total"],
            }
            least_successful = {
                "zone": worst_key,
                "label": worst_b["label"],
                "success_rate": worst_b["success_rate"],
                "failed": worst_b["failed"],
                "total": worst_b["total"],
            }
        else:
            most_successful = {
                "zone": "Z4",
                "label": SETTING_TARGET_LABELS_ES[CourtZone.Z4],
                "success_rate": 0.0,
                "successful": 0,
                "total": 0,
            }
            least_successful = {
                "zone": "Z2",
                "label": SETTING_TARGET_LABELS_ES[CourtZone.Z2],
                "success_rate": 0.0,
                "failed": 0,
                "total": 0,
            }

        return {
            "total_sets": total_sets,
            "successful_sets": total_successful,
            "failed_sets": total_failed,
            "overall_success_rate": overall_rate,
            "by_target_zone": by_zone,
            "most_successful_target": most_successful,
            "least_successful_target": least_successful,
        }

    def get_serve_errors_by_rotation(
        self, team: TeamSide, set_number: Optional[int] = None
    ) -> Dict[Rotation, Dict[str, int]]:
        """
        Cross-tabulate service outcomes and errors across Rotations R1 to R6.
        If `set_number` is None, aggregates across the entire match.
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
        Identify the rotation (R1-R6) with the highest count of service errors
        for a specific set or for the entire match (`set_number=None`).
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
        Compute attack attempts, kills, errors, and hitting efficiency per CourtZone (1-6)
        for a given set or the entire match (`set_number=None`).
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
        self, team: TeamSide, set_number: Optional[int] = None
    ) -> Dict[Rotation, Dict[str, float | int]]:
        """
        Cross-tabulate offensive spike performance across Rotations R1 to R6
        for a specific set or for the entire match (`set_number=None`).
        """
        rot_stats: Dict[Rotation, Dict[str, float | int]] = {
            rot: {
                "attempts": 0,
                "kills": 0,
                "errors": 0,
                "in_play": 0,
                "efficiency": 0.0,
            }
            for rot in Rotation
        }

        for ev in self._attack_events:
            if ev.team != team:
                continue
            if set_number is not None and ev.set_number != set_number:
                continue

            r_bucket = rot_stats[ev.rotation]
            r_bucket["attempts"] = int(r_bucket["attempts"]) + 1
            if ev.outcome == AttackOutcome.KILL:
                r_bucket["kills"] = int(r_bucket["kills"]) + 1
            elif ev.outcome.is_error:
                r_bucket["errors"] = int(r_bucket["errors"]) + 1
            else:
                r_bucket["in_play"] = int(r_bucket["in_play"]) + 1

        for rot, r_bucket in rot_stats.items():
            att = int(r_bucket["attempts"])
            if att > 0:
                kills = int(r_bucket["kills"])
                errs = int(r_bucket["errors"])
                r_bucket["efficiency"] = round((kills - errs) / att, 4)

        return rot_stats

    def get_set_comparison_summary(self, team: TeamSide) -> Dict[str, Dict[str, Any]]:
        """
        Build a comparative summary for each individual Set ('1', '2', ...) and
        the Full Match ('ALL'), enabling side-by-side set vs match charts.
        """
        scopes: List[Optional[int]] = [None] + self.get_recorded_sets()
        summary: Dict[str, Dict[str, Any]] = {}

        for scope in scopes:
            key = "ALL" if scope is None else str(scope)
            zone_map = self.get_attack_stats_by_zone(team, set_number=scope)
            srv_map = self.get_serve_errors_by_rotation(team, set_number=scope)
            set_stats = self.get_setting_stats(team, set_number=scope)
            worst_rot, worst_errs = self.get_most_vulnerable_serve_rotation(team, set_number=scope)

            tot_atk = sum(int(z["attempts"]) for z in zone_map.values())
            tot_kills = sum(int(z["kills"]) for z in zone_map.values())
            tot_atk_err = sum(int(z["errors"]) for z in zone_map.values())
            atk_eff = round((tot_kills - tot_atk_err) / tot_atk, 4) if tot_atk > 0 else 0.0

            tot_srv = sum(int(r["TOTAL_SERVES"]) for r in srv_map.values())
            tot_aces = sum(int(r["ACE"]) for r in srv_map.values())
            tot_srv_err = sum(int(r["TOTAL_ERRORS"]) for r in srv_map.values())

            best_zone = max(
                CourtZone,
                key=lambda z: (int(zone_map[z]["kills"]), float(zone_map[z]["efficiency"])),
            )

            summary[key] = {
                "scope": key,
                "label": "Partido Completo" if scope is None else f"Set {scope}",
                "total_attacks": tot_atk,
                "kills": tot_kills,
                "attack_errors": tot_atk_err,
                "attack_efficiency": atk_eff,
                "total_serves": tot_srv,
                "aces": tot_aces,
                "serve_errors": tot_srv_err,
                "vulnerable_rotation": worst_rot.name,
                "vulnerable_rotation_errors": worst_errs,
                "best_zone": f"Z{best_zone.value}",
                "setting_success_rate": set_stats["overall_success_rate"],
                "best_setting_zone": set_stats["most_successful_target"]["zone"],
                "worst_setting_zone": set_stats["least_successful_target"]["zone"],
            }

        return summary


# ---------------------------------------------------------------------------
# Pure K1/K2 Chained Analytics (Reception -> Setting -> Attack)
# ---------------------------------------------------------------------------

_K1_TARGET_ZONES: Tuple[int, ...] = (4, 3, 2, 6, 1, 5)
_RECEPTION_ZONE_LABELS_ES: Dict[int, str] = {
    5: "Z5 (Líbero)",
    6: "Z6 (Pipe / Receptor Zaguero)",
    1: "Z1 (Receptor / Opuesto)",
    4: "Z4 (Punta Corta)",
    3: "Z3 (Centro Corta)",
    2: "Z2 (Derecha Corta)",
}


def _filter_k1_sequences(
    sequences: Sequence[RallyK1Sequence], set_number: Optional[int]
) -> List[RallyK1Sequence]:
    if set_number is None:
        return list(sequences)
    return [s for s in sequences if s.set_number == set_number]


def _calc_fivb_efficiency(attacks: Sequence[RallyK1Sequence]) -> Dict[str, Any]:
    total = len(attacks)
    if total == 0:
        return {
            "attempts": 0,
            "kills": 0,
            "errors": 0,
            "blocked": 0,
            "efficiency": 0.0,
            "efficiency_pct": 0.0,
        }
    kills = sum(1 for s in attacks if s.attack and s.attack.outcome == K1AttackOutcome.KILL)
    errors = sum(1 for s in attacks if s.attack and s.attack.outcome == K1AttackOutcome.ERROR)
    blocked = sum(1 for s in attacks if s.attack and s.attack.outcome == K1AttackOutcome.BLOCKED)
    eff = round((kills - errors - blocked) / total, 4)
    return {
        "attempts": total,
        "kills": kills,
        "errors": errors,
        "blocked": blocked,
        "efficiency": eff,
        "efficiency_pct": round(eff * 100.0, 1),
    }


def compute_setter_distribution(
    sequences: Sequence[RallyK1Sequence], set_number: Optional[int] = None
) -> Dict[str, Any]:
    """
    Compute conditioned setter distribution (In-System '#'/'+' vs Out-of-System '-'/'!')
    across target zones and Setter Advantage Rate (blockers_faced <= 1).
    """
    filtered = _filter_k1_sequences(sequences, set_number)

    def _bucket(items: List[RallyK1Sequence]) -> Dict[str, Any]:
        total = len(items)
        zones: Dict[str, Dict[str, Any]] = {}
        for z in _K1_TARGET_ZONES:
            cnt = sum(1 for s in items if s.setting and s.setting.target_zone == z)
            zones[f"Z{z}"] = {
                "count": cnt,
                "pct": round((cnt / total) * 100.0, 1) if total > 0 else 0.0,
            }
        return {"total": total, "zones": zones}

    in_sys = [
        s
        for s in filtered
        if s.setting is not None
        and s.reception.quality in (ReceptionQuality.PERFECT, ReceptionQuality.POSITIVE)
    ]
    out_sys = [
        s
        for s in filtered
        if s.setting is not None
        and s.reception.quality in (ReceptionQuality.NEGATIVE, ReceptionQuality.MEDIUM)
    ]

    with_attack = [s for s in filtered if s.attack is not None]
    advantage_cnt = sum(
        1 for s in with_attack if s.attack is not None and s.attack.blockers_faced <= 1
    )
    advantage_rate = (
        round((advantage_cnt / len(with_attack)) * 100.0, 1) if with_attack else 0.0
    )

    return {
        "in_system": _bucket(in_sys),
        "out_of_system": _bucket(out_sys),
        "setter_advantage_rate_pct": advantage_rate,
        "advantage_attacks": advantage_cnt,
        "total_attacks_with_block_info": len(with_attack),
    }


def compute_reception_and_sideout_by_rotation(
    sequences: Sequence[RallyK1Sequence], set_number: Optional[int] = None
) -> Dict[str, Any]:
    """
    Compute Reception Positive % ('#' + '+'), Reception Efficiency ((# + + - =) / Total),
    and 1st-attempt Side-Out % broken down by reception zone and rotation (R1-R6).
    """
    filtered = _filter_k1_sequences(sequences, set_number)

    def _rec_metrics(items: List[RallyK1Sequence]) -> Dict[str, Any]:
        total = len(items)
        if total == 0:
            return {
                "total": 0,
                "perfect": 0,
                "positive": 0,
                "errors": 0,
                "positive_pct": 0.0,
                "efficiency_pct": 0.0,
                "side_out_won": 0,
                "side_out_pct": 0.0,
            }
        perf = sum(1 for s in items if s.reception.quality == ReceptionQuality.PERFECT)
        pos = sum(1 for s in items if s.reception.quality == ReceptionQuality.POSITIVE)
        err = sum(1 for s in items if s.reception.quality == ReceptionQuality.ERROR)
        so_won = sum(1 for s in items if s.side_out_won)
        return {
            "total": total,
            "perfect": perf,
            "positive": pos,
            "errors": err,
            "positive_pct": round(((perf + pos) / total) * 100.0, 1),
            "efficiency_pct": round(((perf + pos - err) / total) * 100.0, 1),
            "side_out_won": so_won,
            "side_out_pct": round((so_won / total) * 100.0, 1),
        }

    by_zone: Dict[str, Dict[str, Any]] = {}
    for z in (5, 6, 1, 4, 3, 2):
        z_items = [s for s in filtered if s.reception.receptor_zone == z]
        by_zone[f"Z{z}"] = {
            "label": _RECEPTION_ZONE_LABELS_ES[z],
            **_rec_metrics(z_items),
        }

    by_rotation: Dict[str, Dict[str, Any]] = {}
    for rot in Rotation:
        r_items = [s for s in filtered if s.rotation == rot]
        by_rotation[rot.name] = _rec_metrics(r_items)

    return {
        "overall": _rec_metrics(filtered),
        "by_zone": by_zone,
        "by_rotation": by_rotation,
    }


def compute_attack_efficiency_and_directions(
    sequences: Sequence[RallyK1Sequence], set_number: Optional[int] = None
) -> Dict[str, Any]:
    """
    Compute FIVB attack efficiency ((Kills - Errors - Blocked) / Total) by zone,
    tempo, trajectory direction, and correlation with reception quality (#/+ vs -).
    """
    filtered = [s for s in _filter_k1_sequences(sequences, set_number) if s.attack is not None]
    overall = _calc_fivb_efficiency(filtered)

    by_zone: Dict[str, Dict[str, Any]] = {}
    for z in (4, 3, 2, 6, 1):
        z_items = [s for s in filtered if s.attack and s.attack.origin_zone == z]
        by_zone[f"Z{z}"] = _calc_fivb_efficiency(z_items)

    by_tempo: Dict[str, Dict[str, Any]] = {}
    for tempo in SetTempo:
        t_items = [s for s in filtered if s.setting and s.setting.tempo == tempo]
        by_tempo[tempo.value] = _calc_fivb_efficiency(t_items)

    by_trajectory: Dict[str, Dict[str, Any]] = {}
    for traj in AttackTrajectory:
        tr_items = [s for s in filtered if s.attack and s.attack.trajectory_type == traj]
        by_trajectory[traj.value] = _calc_fivb_efficiency(tr_items)

    in_sys_atks = [
        s
        for s in filtered
        if s.reception.quality in (ReceptionQuality.PERFECT, ReceptionQuality.POSITIVE)
    ]
    neg_rec_atks = [
        s for s in filtered if s.reception.quality == ReceptionQuality.NEGATIVE
    ]

    in_eff = _calc_fivb_efficiency(in_sys_atks)
    neg_eff = _calc_fivb_efficiency(neg_rec_atks)

    return {
        "total_attacks": overall["attempts"],
        "overall_efficiency": overall["efficiency"],
        "overall_efficiency_pct": overall["efficiency_pct"],
        "overall": overall,
        "by_zone": by_zone,
        "by_tempo": by_tempo,
        "by_trajectory": by_trajectory,
        "correlation_reception_to_attack": {
            "in_system_efficiency": in_eff["efficiency"],
            "in_system_efficiency_pct": in_eff["efficiency_pct"],
            "in_system_attempts": in_eff["attempts"],
            "out_of_system_efficiency": neg_eff["efficiency"],
            "out_of_system_efficiency_pct": neg_eff["efficiency_pct"],
            "out_of_system_attempts": neg_eff["attempts"],
        },
    }


def compute_k1_advanced_analytics(
    sequences: Sequence[RallyK1Sequence], set_number: Optional[int] = None
) -> Dict[str, Any]:
    """Return all K1/K2 advanced metrics (Setter, Reception/Side-Out, Attack)."""
    return {
        "set_filter": set_number,
        "total_k1_rallies": len(_filter_k1_sequences(sequences, set_number)),
        "setter_distribution": compute_setter_distribution(sequences, set_number=set_number),
        "reception_performance": compute_reception_and_sideout_by_rotation(
            sequences, set_number=set_number
        ),
        "attack_efficiency": compute_attack_efficiency_and_directions(
            sequences, set_number=set_number
        ),
    }
