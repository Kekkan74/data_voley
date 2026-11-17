"""
FastAPI Backend Server for DataVoley Visual Studio.

Exposes the decoupled TacticalState, CourtMapper, HybridTaggingController,
and MatchStatsEngine to the interactive visual graphical interface.
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
    PixelPoint,
    Rotation,
    ServeOutcome,
    TagValidationStatus,
    TeamSide,
)
from data_voley.stats_engine import MatchStatsEngine
from data_voley.tactical_state import TacticalState

app = FastAPI(
    title="DataVoley Tactical & Spatial Analytics Studio",
    version="0.2.0",
)

WEB_DIR = Path(__file__).resolve().parent.parent / "ui" / "web"

# Default broadcast camera calibration (4 outer corners of 9m x 18m court)
DEFAULT_CAMERA_CORNERS = np.array(
    [
        [200.0, 950.0],   # HOME Bottom-Left (0m, 0m)
        [1720.0, 950.0],  # HOME Bottom-Right (9m, 0m)
        [1400.0, 200.0],  # AWAY Top-Right (9m, 18m)
        [520.0, 200.0],   # AWAY Top-Left (0m, 18m)
    ],
    dtype=np.float32,
)

# Representative 2D court center coordinates (x, y) in meters for each zone
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
    """Holds the active match state, homography mapper, and visual trajectory history."""

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
        self.current_proposal: Optional[Dict[str, Any]] = None
        self.stream_url: str = "https://stream.fivb.org/live/vnl_court_cam1.m3u8"
        self.seed_initial_data()

    def seed_initial_data(self) -> None:
        """Populate realistic match rallies across R1-R6 so charts are rich on startup."""
        self.tactical = TacticalState(
            home_initial_rotation=Rotation.R1,
            away_initial_rotation=Rotation.R4,
            initial_server=TeamSide.HOME,
        )
        self.stats = MatchStatsEngine(tactical_port=self.tactical)
        self.hybrid = HybridTaggingController(stats_engine=self.stats)
        self.trajectories.clear()

        seeded_sequence = [
            # (serve_outcome, atk_team, atk_zone, target_zone, atk_outcome, winner)
            (ServeOutcome.IN_PLAY, TeamSide.HOME, CourtZone.Z4, CourtZone.Z5, AttackOutcome.KILL, TeamSide.HOME),
            (ServeOutcome.ACE, TeamSide.HOME, CourtZone.Z2, CourtZone.Z1, AttackOutcome.KILL, TeamSide.HOME),
            (ServeOutcome.ERROR_NET, TeamSide.AWAY, CourtZone.Z4, CourtZone.Z6, AttackOutcome.IN_PLAY, TeamSide.AWAY),
            (ServeOutcome.IN_PLAY, TeamSide.HOME, CourtZone.Z2, CourtZone.Z1, AttackOutcome.KILL, TeamSide.HOME),  # HOME -> R2
            (ServeOutcome.ERROR_OUT, TeamSide.AWAY, CourtZone.Z3, CourtZone.Z6, AttackOutcome.KILL, TeamSide.AWAY),
            (ServeOutcome.IN_PLAY, TeamSide.HOME, CourtZone.Z3, CourtZone.Z5, AttackOutcome.KILL, TeamSide.HOME),  # HOME -> R3
            (ServeOutcome.ERROR_NET, TeamSide.HOME, CourtZone.Z4, CourtZone.Z2, AttackOutcome.ERROR_NET, TeamSide.AWAY),
            (ServeOutcome.IN_PLAY, TeamSide.HOME, CourtZone.Z4, CourtZone.Z6, AttackOutcome.KILL, TeamSide.HOME),  # HOME -> R4
            (ServeOutcome.ERROR_OUT, TeamSide.HOME, CourtZone.Z2, CourtZone.Z1, AttackOutcome.ERROR_OUT, TeamSide.AWAY),
            (ServeOutcome.IN_PLAY, TeamSide.HOME, CourtZone.Z6, CourtZone.Z5, AttackOutcome.KILL, TeamSide.HOME),  # HOME -> R5 (Pipe)
            (ServeOutcome.IN_PLAY, TeamSide.HOME, CourtZone.Z4, CourtZone.Z1, AttackOutcome.KILL, TeamSide.HOME),
            (ServeOutcome.ERROR_NET, TeamSide.AWAY, CourtZone.Z2, CourtZone.Z4, AttackOutcome.BLOCKED, TeamSide.AWAY),
            (ServeOutcome.IN_PLAY, TeamSide.HOME, CourtZone.Z2, CourtZone.Z5, AttackOutcome.KILL, TeamSide.HOME),  # HOME -> R6
            (ServeOutcome.IN_PLAY, TeamSide.HOME, CourtZone.Z3, CourtZone.Z6, AttackOutcome.KILL, TeamSide.HOME),
            # Extra R3 serve error to highlight R3 vulnerability
            (ServeOutcome.ERROR_OUT, TeamSide.AWAY, CourtZone.Z4, CourtZone.Z5, AttackOutcome.KILL, TeamSide.AWAY),
            (ServeOutcome.IN_PLAY, TeamSide.HOME, CourtZone.Z4, CourtZone.Z1, AttackOutcome.KILL, TeamSide.HOME),  # HOME -> R1
        ]

        for idx, (srv_out, atk_team, o_zone, t_zone, atk_out, winner) in enumerate(seeded_sequence):
            srv_team = self.tactical.serving_team
            self.stats.record_serve(
                team=srv_team,
                outcome=srv_out,
                target_zone=t_zone,
                timestamp_ms=float(idx * 14000),
            )
            ev = self.stats.record_attack(
                team=atk_team,
                origin_zone=o_zone,
                outcome=atk_out,
                target_zone=t_zone,
                validation_status=TagValidationStatus.HUMAN_VALIDATED,
                timestamp_ms=float(idx * 14000 + 4200),
            )
            self._append_trajectory(atk_team, o_zone, t_zone, atk_out, ev.inferred_attacker_role.value)
            self.tactical.register_rally_point(winner=winner)

        # Add an extra R3 error so R3 stands out clearly in the rotational chart
        saved_rot = self.tactical.get_rotation(TeamSide.HOME)
        self.tactical.set_rotation(TeamSide.HOME, Rotation.R3)
        self.stats.record_serve(team=TeamSide.HOME, outcome=ServeOutcome.ERROR_NET)
        self.stats.record_serve(team=TeamSide.HOME, outcome=ServeOutcome.ERROR_OUT)
        self.tactical.set_rotation(TeamSide.HOME, saved_rot)

        self.generate_ai_proposal()

    def _append_trajectory(
        self,
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
        if len(self.trajectories) > 18:
            self.trajectories = self.trajectories[-18:]

    def generate_ai_proposal(self) -> Dict[str, Any]:
        """Create a realistic AI candidate detection from YOLO/TrackNet for hybrid validation."""
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
    h_score, a_score = SESSION.tactical.get_score()
    home_form = SESSION.tactical.infer_formation(TeamSide.HOME)
    away_form = SESSION.tactical.infer_formation(TeamSide.AWAY)

    home_zone_stats = {
        f"Z{z.value}": {
            **st,
            "role": home_form.zone_roles[z].value,
        }
        for z, st in SESSION.stats.get_attack_stats_by_zone(TeamSide.HOME).items()
    }

    home_rot_atk = {
        r.name: st for r, st in SESSION.stats.get_attack_stats_by_rotation(TeamSide.HOME).items()
    }

    home_srv_rot = {
        r.name: bucket
        for r, bucket in SESSION.stats.get_serve_errors_by_rotation(TeamSide.HOME).items()
    }
    worst_rot, worst_err_count = SESSION.stats.get_most_vulnerable_serve_rotation(TeamSide.HOME)

    recent_attacks = [
        {
            "rally": ev.rally_number,
            "team": ev.team.value,
            "rotation": ev.rotation.name,
            "origin_zone": f"Z{ev.origin_zone.value}",
            "target_zone": f"Z{ev.target_zone.value}" if ev.target_zone else "-",
            "role": ev.inferred_attacker_role.value,
            "outcome": ev.outcome.value,
            "status": ev.validation_status.value,
        }
        for ev in SESSION.stats.attack_events[-10:]
    ][::-1]

    return {
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
            "attack_by_zone": home_zone_stats,
            "attack_by_rotation": home_rot_atk,
            "serves_by_rotation": home_srv_rot,
            "most_vulnerable_rotation": {
                "rotation": worst_rot.name,
                "errors": worst_err_count,
            },
        },
        "trajectories": SESSION.trajectories,
        "recent_attacks": recent_attacks,
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
def get_state() -> Dict[str, Any]:
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

    if mapping.is_in_bounds and mapping.team_side and mapping.zone:
        if req.action_type.upper() == "SERVE":
            srv_out = ServeOutcome(req.outcome) if req.outcome in ServeOutcome.__members__ else ServeOutcome.ACE
            SESSION.stats.record_serve(
                team=mapping.team_side,
                outcome=srv_out,
                origin_coord=pt,
                target_zone=CourtZone(req.target_zone),
            )
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
            )
            SESSION._append_trajectory(
                mapping.team_side,
                mapping.zone,
                t_zone,
                atk_out,
                ev.inferred_attacker_role.value,
            )
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
            committed.team,
            committed.origin_zone,
            t_zone,
            committed.outcome,
            committed.inferred_attacker_role.value,
        )
        # Also record a serve for the rally and update score
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
