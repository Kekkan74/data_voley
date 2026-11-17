"""
FastAPI Backend Server for DataVoley Visual Studio.

Exposes the decoupled TacticalState, CourtMapper, HybridTaggingController,
and MatchStatsEngine with per-set ('1', '2', '3'...) and full-match ('ALL')
visual analytics.
"""

from __future__ import annotations

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
    TagValidationStatus,
    TeamSide,
)
from data_voley.stats_engine import MatchStatsEngine
from data_voley.tactical_state import TacticalState

app = FastAPI(
    title="DataVoley Tactical & Spatial Analytics Studio",
    version="0.3.0",
)

WEB_DIR = Path(__file__).resolve().parent.parent / "ui" / "web"

DEFAULT_CAMERA_CORNERS = np.array(
    [
        [200.0, 950.0],   # HOME Bottom-Left (0m, 0m)
        [1720.0, 950.0],  # HOME Bottom-Right (9m, 0m)
        [1400.0, 200.0],  # AWAY Top-Right (9m, 18m)
        [520.0, 200.0],   # AWAY Top-Left (0m, 18m)
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


class StudioSession:
    """Holds the active match state, multi-set history, and active visual filter scope."""

    def __init__(self) -> None:
        self.mapper = CourtMapper.from_camera_corners(DEFAULT_CAMERA_CORNERS)
        self.tactical = TacticalState(
            home_initial_rotation=Rotation.R1,
            away_initial_rotation=Rotation.R4,
            initial_server=TeamSide.HOME,
        )
        self.stats = MatchStatsEngine(tactical_port=self.tactical)
        self.detector = ActionDetector(court_mapper=self.mapper)
        self.hybrid = HybridTaggingController(stats_engine=self.stats)
        self.trajectories: List[Dict[str, Any]] = []
        self.set_scoreboards: Dict[int, Dict[str, int]] = {}
        self.active_scope: str = "ALL"  # "ALL" (Full Match) or "1", "2", "3"...
        self.current_proposal: Optional[Dict[str, Any]] = None
        self.stream_url: str = "https://stream.fivb.org/live/vnl_court_cam1.m3u8"
        self.seed_initial_data()

    def seed_initial_data(self) -> None:
        """
        Seed realistic multi-set match data across Set 1, Set 2, and Set 3 (in progress)
        so both per-set charts and full-match charts are rich on startup.
        """
        self.tactical = TacticalState(
            home_initial_rotation=Rotation.R1,
            away_initial_rotation=Rotation.R4,
            initial_server=TeamSide.HOME,
        )
        self.stats = MatchStatsEngine(tactical_port=self.tactical)
        self.hybrid = HybridTaggingController(stats_engine=self.stats)
        self.trajectories.clear()
        self.active_scope = "ALL"

        # ---------------------------------------------------------------------
        # SET 1 DATA (Won by HOME 25 - 21: Z4 dominance, R3 serve vulnerability)
        # ---------------------------------------------------------------------
        set1_Script = [
            (Rotation.R1, ServeOutcome.ACE, CourtZone.Z4, CourtZone.Z5, AttackOutcome.KILL),
            (Rotation.R1, ServeOutcome.IN_PLAY, CourtZone.Z4, CourtZone.Z1, AttackOutcome.KILL),
            (Rotation.R2, ServeOutcome.IN_PLAY, CourtZone.Z2, CourtZone.Z6, AttackOutcome.KILL),
            (Rotation.R2, ServeOutcome.ERROR_NET, CourtZone.Z3, CourtZone.Z5, AttackOutcome.IN_PLAY),
            (Rotation.R3, ServeOutcome.ERROR_NET, CourtZone.Z4, CourtZone.Z5, AttackOutcome.KILL),
            (Rotation.R3, ServeOutcome.ERROR_OUT, CourtZone.Z4, CourtZone.Z1, AttackOutcome.ERROR_OUT),
            (Rotation.R3, ServeOutcome.ERROR_NET, CourtZone.Z6, CourtZone.Z6, AttackOutcome.KILL),
            (Rotation.R4, ServeOutcome.IN_PLAY, CourtZone.Z2, CourtZone.Z1, AttackOutcome.KILL),
            (Rotation.R5, ServeOutcome.IN_PLAY, CourtZone.Z3, CourtZone.Z4, AttackOutcome.KILL),
            (Rotation.R6, ServeOutcome.ACE, CourtZone.Z4, CourtZone.Z5, AttackOutcome.KILL),
        ]
        for rot, srv_out, o_z, t_z, atk_out in set1_Script:
            self.tactical.set_rotation(TeamSide.HOME, rot)
            self.stats.record_serve(team=TeamSide.HOME, outcome=srv_out, target_zone=t_z, set_override=1)
            ev = self.stats.record_attack(
                team=TeamSide.HOME,
                origin_zone=o_z,
                outcome=atk_out,
                target_zone=t_z,
                set_override=1,
            )
            self._append_trajectory(1, TeamSide.HOME, o_z, t_z, atk_out, ev.inferred_attacker_role.value)

        # ---------------------------------------------------------------------
        # SET 2 DATA (Won by AWAY 22 - 25: Higher Z2 & Z6 Pipe attacks, R5 serve errors)
        # ---------------------------------------------------------------------
        set2_Script = [
            (Rotation.R1, ServeOutcome.IN_PLAY, CourtZone.Z2, CourtZone.Z1, AttackOutcome.KILL),
            (Rotation.R2, ServeOutcome.IN_PLAY, CourtZone.Z6, CourtZone.Z5, AttackOutcome.KILL),
            (Rotation.R2, ServeOutcome.IN_PLAY, CourtZone.Z2, CourtZone.Z6, AttackOutcome.BLOCKED),
            (Rotation.R3, ServeOutcome.IN_PLAY, CourtZone.Z3, CourtZone.Z1, AttackOutcome.KILL),
            (Rotation.R4, ServeOutcome.ERROR_OUT, CourtZone.Z4, CourtZone.Z5, AttackOutcome.ERROR_NET),
            (Rotation.R5, ServeOutcome.ERROR_NET, CourtZone.Z2, CourtZone.Z1, AttackOutcome.KILL),
            (Rotation.R5, ServeOutcome.ERROR_OUT, CourtZone.Z6, CourtZone.Z6, AttackOutcome.KILL),
            (Rotation.R5, ServeOutcome.ERROR_OUT, CourtZone.Z4, CourtZone.Z2, AttackOutcome.IN_PLAY),
            (Rotation.R6, ServeOutcome.IN_PLAY, CourtZone.Z2, CourtZone.Z5, AttackOutcome.KILL),
        ]
        for rot, srv_out, o_z, t_z, atk_out in set2_Script:
            self.tactical.set_rotation(TeamSide.HOME, rot)
            self.stats.record_serve(team=TeamSide.HOME, outcome=srv_out, target_zone=t_z, set_override=2)
            ev = self.stats.record_attack(
                team=TeamSide.HOME,
                origin_zone=o_z,
                outcome=atk_out,
                target_zone=t_z,
                set_override=2,
            )
            self._append_trajectory(2, TeamSide.HOME, o_z, t_z, atk_out, ev.inferred_attacker_role.value)

        # ---------------------------------------------------------------------
        # ADVANCE TACTICAL FSM TO SET 3 (Live Set 3: 14 - 11, Sets Won: 1 - 1)
        # ---------------------------------------------------------------------
        self.tactical._current_set = 3
        self.tactical._sets_won = {TeamSide.HOME: 1, TeamSide.AWAY: 1}
        self.tactical._score = {TeamSide.HOME: 8, TeamSide.AWAY: 6}
        self.tactical.set_rotation(TeamSide.HOME, Rotation.R1)
        self.tactical.set_rotation(TeamSide.AWAY, Rotation.R4)

        set3_live_rallies = [
            (ServeOutcome.IN_PLAY, CourtZone.Z3, CourtZone.Z6, AttackOutcome.KILL, TeamSide.HOME),
            (ServeOutcome.ERROR_NET, CourtZone.Z4, CourtZone.Z5, AttackOutcome.IN_PLAY, TeamSide.AWAY),
            (ServeOutcome.IN_PLAY, CourtZone.Z4, CourtZone.Z1, AttackOutcome.KILL, TeamSide.HOME),
            (ServeOutcome.ACE, CourtZone.Z2, CourtZone.Z5, AttackOutcome.KILL, TeamSide.HOME),
            (ServeOutcome.ERROR_OUT, CourtZone.Z3, CourtZone.Z6, AttackOutcome.ERROR_OUT, TeamSide.AWAY),
            (ServeOutcome.IN_PLAY, CourtZone.Z2, CourtZone.Z1, AttackOutcome.KILL, TeamSide.HOME),
            (ServeOutcome.IN_PLAY, CourtZone.Z1, CourtZone.Z5, AttackOutcome.KILL, TeamSide.HOME),
        ]
        for srv_out, o_z, t_z, atk_out, winner in set3_live_rallies:
            self.stats.record_serve(team=TeamSide.HOME, outcome=srv_out, target_zone=t_z)
            ev = self.stats.record_attack(
                team=TeamSide.HOME,
                origin_zone=o_z,
                outcome=atk_out,
                target_zone=t_z,
            )
            self._append_trajectory(3, TeamSide.HOME, o_z, t_z, atk_out, ev.inferred_attacker_role.value)
            self.tactical.register_rally_point(winner=winner)

        h_now, a_now = self.tactical.get_score()
        self.set_scoreboards = {
            1: {"HOME": 25, "AWAY": 21, "status": "FINAL"},
            2: {"HOME": 22, "AWAY": 25, "status": "FINAL"},
            3: {"HOME": h_now, "AWAY": a_now, "status": "EN JUEGO"},
        }

        self.generate_ai_proposal()

    def sync_active_set_scoreboard(self) -> None:
        h_now, a_now = self.tactical.get_score()
        c_set = self.tactical.current_set
        self.set_scoreboards[c_set] = {
            "HOME": h_now,
            "AWAY": a_now,
            "status": "FINAL" if self.tactical.match_finished else "EN JUEGO",
        }

    def _append_trajectory(
        self,
        set_number: int,
        team: TeamSide,
        origin_zone: CourtZone,
        target_zone: Optional[CourtZone],
        outcome: AttackOutcome,
        role: str,
    ) -> None:
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
                "start_x": round(sx + random.uniform(-0.35, 0.35), 2),
                "start_y": round(sy + random.uniform(-0.35, 0.35), 2),
                "end_x": round(ex + random.uniform(-0.45, 0.45), 2),
                "end_y": round(ey + random.uniform(-0.35, 0.35), 2),
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
            "confidence": round(random.uniform(0.89, 0.98), 2),
        }
        return self.current_proposal


SESSION = StudioSession()


class RallyRequest(BaseModel):
    winner: str


class ScopeFilterRequest(BaseModel):
    scope: str  # "ALL" or "1", "2", "3", ...


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

    # Determine set filter (None = Full Match, int = Specific Set)
    scope_str = SESSION.active_scope
    set_filter: Optional[int] = int(scope_str) if scope_str.isdigit() else None

    home_zone_stats = {
        f"Z{z.value}": {
            **st,
            "role": home_form.zone_roles[z].value,
        }
        for z, st in SESSION.stats.get_attack_stats_by_zone(
            TeamSide.HOME, set_number=set_filter
        ).items()
    }

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

    # Filter trajectories & recent attacks by selected scope
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
            "role": ev.inferred_attacker_role.value,
            "outcome": ev.outcome.value,
            "status": ev.validation_status.value,
        }
        for ev in SESSION.stats.attack_events
        if set_filter is None or ev.set_number == set_filter
    ][-10:][::-1]

    set_comparison = SESSION.stats.get_set_comparison_summary(TeamSide.HOME)
    recorded_sets = SESSION.stats.get_recorded_sets()

    return {
        "active_scope": scope_str,
        "available_sets": recorded_sets,
        "set_scoreboards": {str(k): v for k, v in SESSION.set_scoreboards.items()},
        "match": {
            "current_set": SESSION.tactical.current_set,
            "home_score": h_score,
            "away_score": a_score,
            "sets_won": {
                "HOME": SESSION.tactical.sets_won[TeamSide.HOME],
                "AWAY": SESSION.tactical.sets_won[TeamSide.AWAY],
            },
            "serving_team": SESSION.tactical.serving_team.value,
            "stream_url": SESSION.stream_url,
        },
        "tactical": {
            "HOME": {
                "rotation": home_form.rotation.name,
                "rotation_num": home_form.rotation.value,
                "setter_zone": home_form.setter_zone.value,
                "setter_is_front_row": home_form.setter_is_front_row,
                "zones": {str(z.value): role.value for z, role in home_form.zone_roles.items()},
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
            "scope_label": "PARTIDO COMPLETO (TODOS LOS SETS)"
            if set_filter is None
            else f"SET {set_filter} (ESTADÍSTICAS DEL SET)",
            "attack_by_zone": home_zone_stats,
            "attack_by_rotation": home_rot_atk,
            "serves_by_rotation": home_srv_rot,
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


@app.post("/api/scope")
def set_analytics_scope(req: ScopeFilterRequest) -> Dict[str, Any]:
    SESSION.active_scope = req.scope.upper()
    return _serialize_state()


@app.post("/api/rally")
def register_rally(req: RallyRequest) -> Dict[str, Any]:
    winner = TeamSide(req.winner.upper())
    SESSION.tactical.register_rally_point(winner=winner)
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
        if req.action_type.upper() == "SERVE":
            srv_out = (
                ServeOutcome(req.outcome)
                if req.outcome in ServeOutcome.__members__
                else ServeOutcome.ACE
            )
            SESSION.stats.record_serve(
                team=mapping.team_side,
                outcome=srv_out,
                origin_coord=pt,
                target_zone=CourtZone(req.target_zone),
                set_override=target_set,
            )
            if target_set == SESSION.tactical.current_set:
                if srv_out == ServeOutcome.ACE:
                    SESSION.tactical.register_rally_point(mapping.team_side)
                elif srv_out.is_error:
                    SESSION.tactical.register_rally_point(mapping.team_side.opponent())
        else:
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
            if target_set == SESSION.tactical.current_set:
                if atk_out == AttackOutcome.KILL:
                    SESSION.tactical.register_rally_point(mapping.team_side)
                elif atk_out.is_error:
                    SESSION.tactical.register_rally_point(mapping.team_side.opponent())

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
        srv_team = SESSION.tactical.serving_team
        SESSION.stats.record_serve(team=srv_team, outcome=ServeOutcome.IN_PLAY)
        if committed.outcome == AttackOutcome.KILL:
            SESSION.tactical.register_rally_point(committed.team)
        elif committed.outcome.is_error:
            SESSION.tactical.register_rally_point(committed.team.opponent())

    SESSION.generate_ai_proposal()
    return _serialize_state()


@app.post("/api/reset")
def reset_studio() -> Dict[str, Any]:
    SESSION.seed_initial_data()
    return _serialize_state()
