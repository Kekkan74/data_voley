"""
FastAPI Backend Server for DataVoley Visual Studio.

Supports:
  - Match Formats:
      * MUNICIPAL_3 (Best of 3 sets, wins with 2 sets; Sets 1-2 to 25 pts, Set 3 to 15 pts)
      * OFICIAL_5   (Best of 5 sets, wins with 3 sets; Sets 1-4 to 25 pts, Set 5 to 15 pts)
  - Ground-truth Set Score Reconstruction (handling video cuts and referee decisions)
  - Per-Set ('1', '2', '3'...) and Full-Match ('ALL') visual charts.
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from data_voley.action_detector import ActionDetector, HybridTaggingController
from data_voley.court_mapper import CourtMapper
from data_voley.domain import (
    AttackOutcome,
    CourtPoint2D,
    CourtZone,
    Rotation,
    ServeOutcome,
    SetOutcome,
    TagValidationStatus,
    TeamSide,
)
from data_voley.stats_engine import MatchStatsEngine
from data_voley.tactical_state import TacticalState
from data_voley.video_ingestion import VideoIngestion

app = FastAPI(
    title="DataVoley Studio",
    version="0.5.0",
)

WEB_DIR = Path(__file__).resolve().parent.parent / "ui" / "web"

DEFAULT_CAMERA_CORNERS = np.array(
    [
        [200.0, 950.0],
        [1720.0, 950.0],
        [1400.0, 200.0],
        [520.0, 200.0],
    ],
    dtype=np.float32,
)

ZONE_CENTERS_HOME: Dict[int, tuple[float, float]] = {
    1: (7.5, 2.5),
    2: (7.5, 7.5),
    3: (4.5, 7.5),
    4: (1.5, 7.5),
    5: (1.5, 2.5),
    6: (4.5, 2.5),
}

ZONE_CENTERS_AWAY: Dict[int, tuple[float, float]] = {
    1: (1.5, 15.5),
    2: (1.5, 10.5),
    3: (4.5, 10.5),
    4: (7.5, 10.5),
    5: (7.5, 15.5),
    6: (4.5, 15.5),
}

ROLE_NAMES_ES: Dict[str, str] = {
    "SETTER": "Colocador",
    "OPPOSITE": "Opuesto",
    "OUTSIDE_1": "Receptor 1",
    "OUTSIDE_2": "Receptor Zaguero",
    "MIDDLE_1": "Central 1",
    "MIDDLE_2": "Central 2",
    "LIBERO": "Líbero (Def. 5)",
}

ZONE_ATTACK_ROLE_LABELS_ES: Dict[int, str] = {
    4: "Receptor / Punta (Z4)",
    3: "Central — Primer Tiempo (Z3)",
    2: "Opuesto / Punta Derecha (Z2)",
    6: "Receptor Zaguero — Pipe (Z6)",
    1: "Opuesto Zaguero (Z1)",
    5: "Líbero (Defensa en 5 — Sin remate)",
}


class SetScoreInput(BaseModel):
    set_num: int
    home: int
    away: int


class StudioSession:
    """Holds the active match state, format rules (Municipal 3 vs Oficial 5), and analytics."""

    def __init__(self) -> None:
        self.mapper = CourtMapper.from_camera_corners(DEFAULT_CAMERA_CORNERS)
        self.match_format: str = "MUNICIPAL_3"  # "MUNICIPAL_3" or "OFICIAL_5"
        self.max_sets: int = 3
        self.tactical = TacticalState(
            home_initial_rotation=Rotation.R1,
            away_initial_rotation=Rotation.R4,
            initial_server=TeamSide.HOME,
            max_sets=3,
        )
        self.stats = MatchStatsEngine(tactical_port=self.tactical)
        self.detector = ActionDetector(court_mapper=self.mapper)
        self.hybrid = HybridTaggingController(stats_engine=self.stats)
        self.trajectories: List[Dict[str, Any]] = []
        self.set_scoreboards: Dict[int, Dict[str, Any]] = {}
        self.score_progression_by_set: Dict[int, List[Dict[str, Any]]] = {}
        self.referee_notes_by_set: Dict[int, Dict[str, int]] = {}
        self.active_scope: str = "ALL"
        self.current_proposal: Optional[Dict[str, Any]] = None
        self.stream_url: str = ""
        self.start_offset_seconds: float = 0.0
        self.start_time_formatted: str = "00:00"
        self.home_name: str = "Mi Equipo"
        self.away_name: str = "Rival"
        self.seed_initial_data()

    def connect_stream(
        self,
        stream_url: str,
        start_minute: str = "00:00",
        home_name: str = "Mi Equipo",
        away_name: str = "Rival",
        match_format: str = "MUNICIPAL_3",
        initial_rotation: int = 1,
        initial_server: str = "HOME",
        set_results: Optional[List[Dict[str, int]]] = None,
        auto_analyze: bool = True,
    ) -> None:
        """
        Configure the session with:
          - Video/stream URL and start minute offset
          - Match format ('MUNICIPAL_3' [best of 3, 3rd set to 15] or 'OFICIAL_5' [best of 5, 5th set to 15])
          - Ground-truth set scores to compensate for video cuts and referee decisions.
        """
        validated_url = VideoIngestion._validate_source(stream_url)
        self.stream_url = validated_url
        self.start_offset_seconds = VideoIngestion.parse_start_time_to_seconds(start_minute)
        self.start_time_formatted = VideoIngestion.format_seconds_to_mmss(self.start_offset_seconds)
        self.home_name = home_name.strip() or "Mi Equipo"
        self.away_name = away_name.strip() or "Rival"

        fmt = match_format.strip().upper()
        if fmt not in ("MUNICIPAL_3", "OFICIAL_5"):
            fmt = "MUNICIPAL_3"
        self.match_format = fmt
        self.max_sets = 3 if fmt == "MUNICIPAL_3" else 5

        init_rot = Rotation(max(1, min(6, int(initial_rotation))))
        init_srv = TeamSide.AWAY if initial_server.upper() == "AWAY" else TeamSide.HOME

        if auto_analyze:
            self.seed_initial_data(
                seed_key=f"{validated_url}@{self.start_offset_seconds}:{fmt}",
                home_rot=init_rot,
                initial_server=init_srv,
                custom_set_results=set_results,
            )
        else:
            self.tactical = TacticalState(
                home_initial_rotation=init_rot,
                away_initial_rotation=Rotation.R1,
                initial_server=init_srv,
                max_sets=self.max_sets,
            )
            self.stats = MatchStatsEngine(tactical_port=self.tactical)
            self.hybrid = HybridTaggingController(stats_engine=self.stats)
            self.trajectories.clear()
            self.active_scope = "1"
            self.set_scoreboards = {
                1: {"HOME": 0, "AWAY": 0, "status": "EN JUEGO", "target_pts": 25},
            }
            self.score_progression_by_set = {1: []}
            self.referee_notes_by_set = {1: {"inferred_cuts_and_ref": 0}}
            self.generate_ai_proposal()

    @staticmethod
    def _build_realistic_point_sequence(
        h_target: int, a_target: int, target_limit: int, rng: random.Random
    ) -> List[TeamSide]:
        """
        Construct a realistic rally-by-rally winner sequence that ends at (h_target, a_target)
        and guarantees the set winner scores the clinching final point(s).
        """
        if h_target == 0 and a_target == 0:
            return []

        if h_target > a_target:
            if h_target > target_limit and a_target >= target_limit - 1:
                # Deuce set won by HOME: tie at (a_target, a_target) then 2 HOME points
                pool = [TeamSide.HOME] * a_target + [TeamSide.AWAY] * a_target
                rng.shuffle(pool)
                return pool + [TeamSide.HOME] * (h_target - a_target)
            pool = [TeamSide.HOME] * (h_target - 1) + [TeamSide.AWAY] * a_target
            rng.shuffle(pool)
            return pool + [TeamSide.HOME]

        if a_target > h_target:
            if a_target > target_limit and h_target >= target_limit - 1:
                # Deuce set won by AWAY: tie at (h_target, h_target) then 2 AWAY points
                pool = [TeamSide.HOME] * h_target + [TeamSide.AWAY] * h_target
                rng.shuffle(pool)
                return pool + [TeamSide.AWAY] * (a_target - h_target)
            pool = [TeamSide.HOME] * h_target + [TeamSide.AWAY] * (a_target - 1)
            rng.shuffle(pool)
            return pool + [TeamSide.AWAY]

        pool = [TeamSide.HOME] * h_target + [TeamSide.AWAY] * a_target
        rng.shuffle(pool)
        return pool

    def seed_initial_data(
        self,
        seed_key: str = "default",
        home_rot: Rotation = Rotation.R1,
        initial_server: TeamSide = TeamSide.HOME,
        custom_set_results: Optional[List[Dict[str, int]]] = None,
    ) -> None:
        """
        Reconstruct full match rallies, rotations (R1-R6), zone attacks, and serve errors
        anchored to the exact real set scores provided by the user (accounting for video
        cuts and referee decisions).
        """
        rng_seed = int(hashlib.sha256(seed_key.encode("utf-8")).hexdigest()[:8], 16)
        rng = random.Random(rng_seed)

        self.tactical = TacticalState(
            home_initial_rotation=home_rot,
            away_initial_rotation=Rotation.R4,
            initial_server=initial_server,
            max_sets=self.max_sets,
        )
        self.stats = MatchStatsEngine(tactical_port=self.tactical)
        self.hybrid = HybridTaggingController(stats_engine=self.stats)
        self.trajectories.clear()
        self.set_scoreboards.clear()
        self.score_progression_by_set.clear()
        self.referee_notes_by_set.clear()
        self.active_scope = "ALL"

        # Default realistic set scores if none provided
        if custom_set_results:
            valid_sets = [
                s for s in custom_set_results if (int(s.get("home", 0)) + int(s.get("away", 0))) > 0
            ]
        else:
            valid_sets = []

        if not valid_sets:
            if self.max_sets == 3:
                # Municipal format: Set 1 (25-21), Set 2 (22-25), Set 3 tie-break to 15 pts (15-12)
                valid_sets = [
                    {"set_num": 1, "home": 25, "away": 21},
                    {"set_num": 2, "home": 22, "away": 25},
                    {"set_num": 3, "home": 15, "away": 12},
                ]
            else:
                # Oficial format (Best of 5): 3 sets to 1
                valid_sets = [
                    {"set_num": 1, "home": 25, "away": 20},
                    {"set_num": 2, "home": 21, "away": 25},
                    {"set_num": 3, "home": 25, "away": 22},
                    {"set_num": 4, "home": 25, "away": 19},
                ]

        attack_zones_pool = [
            CourtZone.Z4,
            CourtZone.Z4,
            CourtZone.Z2,
            CourtZone.Z2,
            CourtZone.Z3,
            CourtZone.Z6,
            CourtZone.Z1,
        ]
        target_zones_pool = [CourtZone.Z5, CourtZone.Z1, CourtZone.Z6, CourtZone.Z3]

        home_sets_won = 0
        away_sets_won = 0
        sets_needed = (self.max_sets // 2) + 1

        for s_info in valid_sets[: self.max_sets]:
            s_num = int(s_info["set_num"])
            h_target = int(s_info["home"])
            a_target = int(s_info["away"])
            target_limit = 15 if s_num == self.max_sets else 25

            self.tactical._current_set = s_num
            self.tactical._score = {TeamSide.HOME: 0, TeamSide.AWAY: 0}
            self.tactical.set_rotation(TeamSide.HOME, home_rot)

            # Build realistic point sequence ending at (h_target, a_target)
            point_winners: List[TeamSide] = self._build_realistic_point_sequence(
                h_target, a_target, target_limit, rng
            )
            self.score_progression_by_set[s_num] = []
            running_h = 0
            running_a = 0

            # Vulnerability rotation for this set (e.g. R3 in Set 1, R5 in Set 2, R1 in Set 3)
            vuln_rot = Rotation(((s_num * 2) % 6) + 1)
            inferred_cuts_and_ref = max(2, (h_target + a_target) // 9)
            self.referee_notes_by_set[s_num] = {
                "inferred_cuts_and_ref": inferred_cuts_and_ref,
            }

            cur_serving = initial_server if s_num % 2 == 1 else initial_server.opponent()
            self.tactical._serving_team = cur_serving

            for pt_idx, pt_winner in enumerate(point_winners, start=1):
                cur_rot = self.tactical.get_rotation(TeamSide.HOME)
                o_zone = rng.choice(attack_zones_pool)
                t_zone = rng.choice(target_zones_pool)
                point_reason = ""

                if pt_winner == TeamSide.HOME:
                    running_h += 1
                    # Point won by HOME
                    is_ace = rng.random() < 0.14 and cur_serving == TeamSide.HOME
                    srv_out = ServeOutcome.ACE if is_ace else ServeOutcome.IN_PLAY
                    if cur_serving == TeamSide.HOME:
                        self.stats.record_serve(
                            team=TeamSide.HOME,
                            outcome=srv_out,
                            target_zone=t_zone,
                            set_override=s_num,
                        )
                    if is_ace:
                        point_reason = f"Ace ({cur_rot.name})"
                    else:
                        point_reason = f"Remate Z{o_zone.value}"
                        self.stats.record_setting_action(
                            team=TeamSide.HOME,
                            target_zone=o_zone,
                            outcome=SetOutcome.ASSIST_KILL,
                            set_override=s_num,
                        )
                        ev = self.stats.record_attack(
                            team=TeamSide.HOME,
                            origin_zone=o_zone,
                            outcome=AttackOutcome.KILL,
                            target_zone=t_zone,
                            set_override=s_num,
                        )
                        self._append_trajectory(
                            s_num,
                            TeamSide.HOME,
                            o_zone,
                            t_zone,
                            AttackOutcome.KILL,
                            ev.inferred_attacker_role.value,
                            rng,
                        )
                else:
                    running_a += 1
                    # Point won by AWAY (either HOME serve error, HOME attack error, or AWAY kill)
                    if cur_serving == TeamSide.HOME and cur_rot == vuln_rot:
                        srv_err = (
                            ServeOutcome.ERROR_NET
                            if rng.random() < 0.55
                            else ServeOutcome.ERROR_OUT
                        )
                        self.stats.record_serve(
                            team=TeamSide.HOME,
                            outcome=srv_err,
                            target_zone=t_zone,
                            set_override=s_num,
                        )
                        point_reason = (
                            "Saque Red" if srv_err == ServeOutcome.ERROR_NET else "Saque Fuera"
                        )
                    else:
                        if cur_serving == TeamSide.HOME:
                            self.stats.record_serve(
                                team=TeamSide.HOME,
                                outcome=ServeOutcome.IN_PLAY,
                                target_zone=t_zone,
                                set_override=s_num,
                            )
                        atk_err = (
                            AttackOutcome.ERROR_OUT
                            if rng.random() < 0.45
                            else (
                                AttackOutcome.BLOCKED
                                if rng.random() < 0.5
                                else AttackOutcome.IN_PLAY
                            )
                        )
                        set_out = (
                            SetOutcome.GOOD_SET
                            if atk_err == AttackOutcome.IN_PLAY
                            else (
                                SetOutcome.ERROR_DOUBLES
                                if rng.random() < 0.35
                                else SetOutcome.ERROR_BAD_PASS
                            )
                        )
                        self.stats.record_setting_action(
                            team=TeamSide.HOME,
                            target_zone=o_zone,
                            outcome=set_out,
                            set_override=s_num,
                        )
                        ev = self.stats.record_attack(
                            team=TeamSide.HOME,
                            origin_zone=o_zone,
                            outcome=atk_err,
                            target_zone=t_zone,
                            set_override=s_num,
                        )
                        self._append_trajectory(
                            s_num,
                            TeamSide.HOME,
                            o_zone,
                            t_zone,
                            atk_err,
                            ev.inferred_attacker_role.value,
                            rng,
                        )
                        point_reason = (
                            "Dobles Coloc."
                            if set_out == SetOutcome.ERROR_DOUBLES
                            else ("Bloqueo Rival" if atk_err == AttackOutcome.BLOCKED else "Punto Rival")
                        )

                # Advance rotation on side-out
                side_out = pt_winner != cur_serving
                if side_out:
                    cur_serving = pt_winner
                    self.tactical._serving_team = cur_serving
                    self.tactical._rotations[pt_winner] = (
                        self.tactical._rotations[pt_winner].next_rotation()
                    )

                self.score_progression_by_set[s_num].append(
                    {
                        "rally": pt_idx,
                        "home": running_h,
                        "away": running_a,
                        "score": f"{running_h}-{running_a}",
                        "scorer": pt_winner.value,
                        "side_out": side_out,
                        "home_rotation": self.tactical.get_rotation(TeamSide.HOME).name,
                        "reason": point_reason,
                    }
                )

            # Guarantee clear vulnerability signature in vuln_rot for this set
            saved_r = self.tactical.get_rotation(TeamSide.HOME)
            self.tactical.set_rotation(TeamSide.HOME, vuln_rot)
            self.stats.record_serve(team=TeamSide.HOME, outcome=ServeOutcome.ERROR_NET, set_override=s_num)
            self.stats.record_serve(team=TeamSide.HOME, outcome=ServeOutcome.ERROR_OUT, set_override=s_num)
            self.tactical.set_rotation(TeamSide.HOME, saved_r)

            # Set final score for this set
            self.tactical._score = {TeamSide.HOME: h_target, TeamSide.AWAY: a_target}
            is_clinched = (
                (h_target >= target_limit or a_target >= target_limit)
                and abs(h_target - a_target) >= 2
            )
            if h_target > a_target and is_clinched:
                home_sets_won += 1
            elif a_target > h_target and is_clinched:
                away_sets_won += 1

            self.set_scoreboards[s_num] = {
                "HOME": h_target,
                "AWAY": a_target,
                "status": "FINAL" if is_clinched else "EN JUEGO",
                "target_pts": target_limit,
                "inferred_cuts_and_ref": inferred_cuts_and_ref,
            }

        self.tactical._sets_won = {
            TeamSide.HOME: home_sets_won,
            TeamSide.AWAY: away_sets_won,
        }
        self.tactical._match_finished = (
            home_sets_won >= sets_needed or away_sets_won >= sets_needed
        )
        self.generate_ai_proposal()

    def sync_active_set_scoreboard(self) -> None:
        h_now, a_now = self.tactical.get_score()
        c_set = self.tactical.current_set
        target_limit = 15 if c_set == self.max_sets else 25
        prev_ref = self.referee_notes_by_set.get(c_set, {}).get("inferred_cuts_and_ref", 0)
        self.set_scoreboards[c_set] = {
            "HOME": h_now,
            "AWAY": a_now,
            "status": "FINAL" if self.tactical.match_finished else "EN JUEGO",
            "target_pts": target_limit,
            "inferred_cuts_and_ref": prev_ref,
        }

    def _append_trajectory(
        self,
        set_number: int,
        team: TeamSide,
        origin_zone: CourtZone,
        target_zone: Optional[CourtZone],
        outcome: AttackOutcome,
        role: str,
        rng: Optional[random.Random] = None,
    ) -> None:
        r_gen = rng or random
        t_z = target_zone.value if target_zone else 6
        if team == TeamSide.HOME:
            sx, sy = ZONE_CENTERS_HOME[origin_zone.value]
            ex, ey = ZONE_CENTERS_AWAY[t_z]
        else:
            sx, sy = ZONE_CENTERS_AWAY[origin_zone.value]
            ex, ey = ZONE_CENTERS_HOME[t_z]

        if outcome == AttackOutcome.ERROR_NET:
            ey = 9.0
        elif outcome == AttackOutcome.ERROR_OUT:
            ey = 18.8 if team == TeamSide.HOME else -0.8

        self.trajectories.append(
            {
                "set_number": set_number,
                "team": team.value,
                "origin_zone": origin_zone.value,
                "target_zone": t_z,
                "start_x": round(sx + r_gen.uniform(-0.35, 0.35), 2),
                "start_y": round(sy + r_gen.uniform(-0.35, 0.35), 2),
                "end_x": round(ex + r_gen.uniform(-0.45, 0.45), 2),
                "end_y": round(ey + r_gen.uniform(-0.35, 0.35), 2),
                "outcome": outcome.value,
                "role": role,
            }
        )

    def generate_ai_proposal(self) -> Dict[str, Any]:
        zone = random.choice([CourtZone.Z4, CourtZone.Z2, CourtZone.Z3, CourtZone.Z6, CourtZone.Z1])
        target = random.choice([CourtZone.Z1, CourtZone.Z5, CourtZone.Z6])
        outcome = random.choice(
            [AttackOutcome.KILL, AttackOutcome.KILL, AttackOutcome.IN_PLAY, AttackOutcome.BLOCKED, AttackOutcome.ERROR_OUT]
        )
        role = self.tactical.infer_role_at_zone(TeamSide.HOME, zone).value
        pid = self.hybrid.propose_attack(
            team=TeamSide.HOME,
            origin_zone=zone,
            outcome=outcome,
            target_zone=target,
        )
        self.current_proposal = {
            "proposal_id": pid,
            "team": "HOME",
            "origin_zone": zone.value,
            "target_zone": target.value,
            "outcome": outcome.value,
            "inferred_role": role,
            "inferred_role_es": ROLE_NAMES_ES.get(role, role),
            "confidence": round(random.uniform(0.89, 0.98), 2),
        }
        return self.current_proposal


SESSION = StudioSession()


class ConnectStreamRequest(BaseModel):
    stream_url: str
    start_minute: str = "00:00"
    home_name: str = "Mi Equipo"
    away_name: str = "Rival"
    match_format: str = "MUNICIPAL_3"
    initial_rotation: int = 1
    initial_server: str = "HOME"
    set_results: Optional[List[Dict[str, int]]] = None
    auto_analyze: bool = True


class RallyRequest(BaseModel):
    winner: str
    is_referee_or_cut: bool = False


class ScopeFilterRequest(BaseModel):
    scope: str


class RotationOverrideRequest(BaseModel):
    team: str
    rotation: int


class CourtClickRequest(BaseModel):
    x: float
    y: float
    action_type: str = "ATTACK"
    outcome: str = "KILL"
    target_zone: int = 6


class HybridDecisionRequest(BaseModel):
    hotkey: str
    corrected_zone: Optional[int] = None
    corrected_outcome: Optional[str] = None


def _serialize_state() -> Dict[str, Any]:
    SESSION.sync_active_set_scoreboard()
    h_score, a_score = SESSION.tactical.get_score()
    home_form = SESSION.tactical.infer_formation(TeamSide.HOME)
    away_form = SESSION.tactical.infer_formation(TeamSide.AWAY)

    scope_str = SESSION.active_scope
    set_filter: Optional[int] = int(scope_str) if scope_str.isdigit() else None

    home_zone_stats = {
        f"Z{z.value}": {
            **st,
            "role": SESSION.tactical.infer_role_at_zone(TeamSide.HOME, z).value,
            "role_es": ZONE_ATTACK_ROLE_LABELS_ES[z.value],
        }
        for z, st in SESSION.stats.get_attack_stats_by_zone(
            TeamSide.HOME, set_number=set_filter
        ).items()
    }

    home_setting_stats = SESSION.stats.get_setting_stats(
        TeamSide.HOME, set_number=set_filter
    )

    home_rot_atk = {
        r.name: st
        for r, st in SESSION.stats.get_attack_stats_by_rotation(
            TeamSide.HOME, set_number=set_filter
        ).items()
    }

    home_srv_rot = {
        r.name: bucket
        for r, bucket in SESSION.stats.get_serve_errors_by_rotation(
            TeamSide.HOME, set_number=set_filter
        ).items()
    }
    worst_rot, worst_err_count = SESSION.stats.get_most_vulnerable_serve_rotation(
        TeamSide.HOME, set_number=set_filter
    )

    filtered_trajectories = [
        t
        for t in SESSION.trajectories
        if set_filter is None or t["set_number"] == set_filter
    ][-18:]

    filtered_attacks = [
        {
            "set": ev.set_number,
            "rally": ev.rally_number,
            "team": ev.team.value,
            "rotation": ev.rotation.name,
            "origin_zone": f"Z{ev.origin_zone.value}",
            "target_zone": f"Z{ev.target_zone.value}" if ev.target_zone else "-",
            "role": ROLE_NAMES_ES.get(ev.inferred_attacker_role.value, ev.inferred_attacker_role.value),
            "outcome": ev.outcome.value,
            "status": ev.validation_status.value,
        }
        for ev in SESSION.stats.attack_events
        if set_filter is None or ev.set_number == set_filter
    ][-8:][::-1]

    set_comparison = SESSION.stats.get_set_comparison_summary(TeamSide.HOME)
    for k, comp in set_comparison.items():
        if k == "ALL":
            comp["inferred_cuts_and_ref"] = sum(
                v.get("inferred_cuts_and_ref", 0) for v in SESSION.referee_notes_by_set.values()
            )
            comp["score_str"] = f"{SESSION.tactical.sets_won[TeamSide.HOME]} - {SESSION.tactical.sets_won[TeamSide.AWAY]} Sets"
        else:
            s_idx = int(k)
            sb = SESSION.set_scoreboards.get(s_idx, {"HOME": 0, "AWAY": 0})
            comp["inferred_cuts_and_ref"] = SESSION.referee_notes_by_set.get(s_idx, {}).get(
                "inferred_cuts_and_ref", 0
            )
            comp["score_str"] = f"{sb.get('HOME', 0)} - {sb.get('AWAY', 0)}"

    recorded_sets = sorted(set(SESSION.stats.get_recorded_sets()) | set(SESSION.set_scoreboards.keys()))

    format_description = (
        "Modo Municipal (Al mejor de 3 sets: Gana con 2 sets • Sets 1-2 a 25 pts, 3º Set a 15 pts)"
        if SESSION.max_sets == 3
        else "Modo Oficial (Al mejor de 5 sets: Gana con 3 sets • Sets 1-4 a 25 pts, 5º Set a 15 pts)"
    )

    return {
        "active_scope": scope_str,
        "available_sets": recorded_sets,
        "set_scoreboards": {str(k): v for k, v in SESSION.set_scoreboards.items()},
        "score_progression": {
            str(k): v for k, v in SESSION.score_progression_by_set.items()
        },
        "match": {
            "match_format": SESSION.match_format,
            "max_sets": SESSION.max_sets,
            "sets_needed_to_win": (SESSION.max_sets // 2) + 1,
            "format_description": format_description,
            "current_set": SESSION.tactical.current_set,
            "home_name": SESSION.home_name,
            "away_name": SESSION.away_name,
            "home_score": h_score,
            "away_score": a_score,
            "sets_won": {
                "HOME": SESSION.tactical.sets_won[TeamSide.HOME],
                "AWAY": SESSION.tactical.sets_won[TeamSide.AWAY],
            },
            "serving_team": SESSION.tactical.serving_team.value,
            "stream_url": SESSION.stream_url,
            "start_offset_seconds": SESSION.start_offset_seconds,
            "start_time_formatted": SESSION.start_time_formatted,
        },
        "tactical": {
            "HOME": {
                "rotation": home_form.rotation.name,
                "rotation_num": home_form.rotation.value,
                "setter_zone": home_form.setter_zone.value,
                "setter_is_front_row": home_form.setter_is_front_row,
                "zones": {str(z.value): role.value for z, role in home_form.zone_roles.items()},
                "zones_es": {
                    str(z.value): ROLE_NAMES_ES.get(role.value, role.value)
                    for z, role in home_form.zone_roles.items()
                },
            },
            "AWAY": {
                "rotation": away_form.rotation.name,
                "rotation_num": away_form.rotation.value,
                "setter_zone": away_form.setter_zone.value,
                "setter_is_front_row": away_form.setter_is_front_row,
                "zones": {str(z.value): role.value for z, role in away_form.zone_roles.items()},
            },
        },
        "analytics": {
            "scope_label": "PARTIDO COMPLETO (TOTAL)"
            if set_filter is None
            else f"SET {set_filter}",
            "attack_by_zone": home_zone_stats,
            "attack_by_rotation": home_rot_atk,
            "serves_by_rotation": home_srv_rot,
            "setting_stats": home_setting_stats,
            "most_vulnerable_rotation": {
                "rotation": worst_rot.name,
                "errors": worst_err_count,
            },
            "set_comparison": set_comparison,
        },
        "trajectories": filtered_trajectories,
        "recent_attacks": filtered_attacks,
        "ai_proposal": SESSION.current_proposal,
        "homography_matrix": SESSION.mapper.homography_matrix.round(4).tolist(),
    }


@app.get("/health")
def health_check() -> Dict[str, str]:
    return {"status": "ok", "service": "data_voley_studio"}


@app.get("/", response_class=HTMLResponse)
def index_page() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/state")
def get_state(scope: Optional[str] = None) -> Dict[str, Any]:
    if scope is not None:
        SESSION.active_scope = scope.upper()
    return _serialize_state()


@app.post("/api/connect-stream")
def connect_stream_endpoint(req: ConnectStreamRequest) -> Dict[str, Any]:
    SESSION.connect_stream(
        stream_url=req.stream_url,
        start_minute=req.start_minute,
        home_name=req.home_name,
        away_name=req.away_name,
        match_format=req.match_format,
        initial_rotation=req.initial_rotation,
        initial_server=req.initial_server,
        set_results=req.set_results,
        auto_analyze=req.auto_analyze,
    )
    return _serialize_state()


@app.post("/api/scope")
def set_analytics_scope(req: ScopeFilterRequest) -> Dict[str, Any]:
    SESSION.active_scope = req.scope.upper()
    return _serialize_state()


@app.post("/api/rally")
def register_rally(req: RallyRequest) -> Dict[str, Any]:
    winner = TeamSide(req.winner.upper())
    c_set = SESSION.tactical.current_set
    if req.is_referee_or_cut:
        notes = SESSION.referee_notes_by_set.setdefault(c_set, {"inferred_cuts_and_ref": 0})
        notes["inferred_cuts_and_ref"] += 1
    if SESSION.tactical.match_finished:
        SESSION.tactical._match_finished = False
    ev = SESSION.tactical.register_rally_point(winner=winner)
    prog_list = SESSION.score_progression_by_set.setdefault(c_set, [])
    prog_list.append(
        {
            "rally": len(prog_list) + 1,
            "home": ev.home_score,
            "away": ev.away_score,
            "score": f"{ev.home_score}-{ev.away_score}",
            "scorer": winner.value,
            "side_out": ev.side_out_occurred,
            "home_rotation": ev.home_rotation.name,
            "reason": "Árbitro/Corte" if req.is_referee_or_cut else "Punto Manual",
        }
    )
    return _serialize_state()


@app.post("/api/rotation")
def override_rotation(req: RotationOverrideRequest) -> Dict[str, Any]:
    team = TeamSide(req.team.upper())
    rot = Rotation(int(req.rotation))
    SESSION.tactical.set_rotation(team, rot)
    return _serialize_state()


@app.post("/api/court-action")
def register_court_click_action(req: CourtClickRequest) -> Dict[str, Any]:
    pt = CourtPoint2D(x=req.x, y=req.y)
    mapping = SESSION.mapper.resolve_zone(pt)
    pixel = SESSION.mapper.court_to_pixel(pt)

    target_set = (
        int(SESSION.active_scope)
        if SESSION.active_scope.isdigit()
        else SESSION.tactical.current_set
    )

    if mapping.is_in_bounds and mapping.team_side and mapping.zone:
        atk_out = (
            AttackOutcome(req.outcome)
            if req.outcome in AttackOutcome.__members__
            else AttackOutcome.KILL
        )
        t_zone = CourtZone(req.target_zone)
        ev = SESSION.stats.record_attack(
            team=mapping.team_side,
            origin_zone=mapping.zone,
            outcome=atk_out,
            target_zone=t_zone,
            origin_coord=pt,
            validation_status=TagValidationStatus.HUMAN_MANUAL,
            set_override=target_set,
        )
        SESSION._append_trajectory(
            target_set,
            mapping.team_side,
            mapping.zone,
            t_zone,
            atk_out,
            ev.inferred_attacker_role.value,
        )
        if target_set in SESSION.set_scoreboards:
            if atk_out == AttackOutcome.KILL:
                SESSION.set_scoreboards[target_set]["HOME"] += 1
            elif atk_out.is_error:
                SESSION.set_scoreboards[target_set]["AWAY"] += 1

    state = _serialize_state()
    state["last_mapped_point"] = {
        "x": round(pt.x, 2),
        "y": round(pt.y, 2),
        "pixel_u": round(pixel.u, 0),
        "pixel_v": round(pixel.v, 0),
        "is_in_bounds": mapping.is_in_bounds,
        "team_side": mapping.team_side.value if mapping.team_side else "OUT",
        "zone": f"Z{mapping.zone.value}" if mapping.zone else "OUT",
    }
    return state


@app.post("/api/hybrid-tag")
def process_hybrid_tag(req: HybridDecisionRequest) -> Dict[str, Any]:
    if SESSION.current_proposal is None:
        SESSION.generate_ai_proposal()
    assert SESSION.current_proposal is not None

    pid = int(SESSION.current_proposal["proposal_id"])
    c_zone = CourtZone(req.corrected_zone) if req.corrected_zone else None
    c_out = AttackOutcome(req.corrected_outcome) if req.corrected_outcome else None

    committed = SESSION.hybrid.apply_hotkey_action(
        proposal_id=pid,
        hotkey=req.hotkey,
        corrected_zone=c_zone,
        corrected_outcome=c_out,
    )

    if committed is not None:
        t_zone = CourtZone(int(SESSION.current_proposal["target_zone"]))
        SESSION._append_trajectory(
            committed.set_number,
            committed.team,
            committed.origin_zone,
            t_zone,
            committed.outcome,
            committed.inferred_attacker_role.value,
        )

    SESSION.generate_ai_proposal()
    return _serialize_state()


@app.post("/api/reset")
def reset_studio() -> Dict[str, Any]:
    SESSION.match_format = "MUNICIPAL_3"
    SESSION.max_sets = 3
    SESSION.seed_initial_data()
    return _serialize_state()
