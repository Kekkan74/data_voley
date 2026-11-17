#!/usr/bin/env python3
"""
data_voley — CLI & Interactive Tactical/Spatial Simulation Entrypoint.

Demonstrates the complete decoupled architecture:
  1. CourtMapper (3D Camera Pixels -> 2D Metric Homography -> FIVB Zones 1-6)
  2. TacticalState (R1-R6 Finite State Machine & Zero-OCR Role Inference)
  3. ActionDetector & HybridTaggingController (AI Proposal + Hotkey Validation)
  4. MatchStatsEngine (Rotational Vulnerability & Zone Spike Efficiency)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from data_voley.action_detector import ActionDetector, HybridTaggingController
from data_voley.court_mapper import CourtMapper
from data_voley.domain import (
    AttackOutcome,
    CourtZone,
    PixelPoint,
    Rotation,
    ServeOutcome,
    TeamSide,
)
from data_voley.stats_engine import MatchStatsEngine
from data_voley.tactical_state import TacticalState


def run_demo() -> int:
    """Execute a complete end-to-end tactical and spatial analytics demonstration."""
    console = Console()
    console.print(
        Panel.fit(
            "[bold #00f5ff]🏐 DATA_VOLEY // SPATIAL & TACTICAL ANALYTICS ENGINE[/]\n"
            "[#00ff66]Homography 3D->2D • Rotations R1-R6 FSM • Hybrid AI/Hotkey Tagging[/]",
            border_style="#00ff66",
            box=box.DOUBLE,
        )
    )

    # 1. Calibrate CourtMapper from 4 broadcast camera corners
    camera_corners = np.array(
        [
            [200.0, 950.0],   # HOME Bottom-Left (0m, 0m)
            [1720.0, 950.0],  # HOME Bottom-Right (9m, 0m)
            [1400.0, 200.0],  # AWAY Top-Right (9m, 18m)
            [520.0, 200.0],   # AWAY Top-Left (0m, 18m)
        ],
        dtype=np.float32,
    )
    mapper = CourtMapper.from_camera_corners(camera_corners)

    # 2. Initialize TacticalState Machine
    tactical = TacticalState(
        home_initial_rotation=Rotation.R1,
        away_initial_rotation=Rotation.R4,
        initial_server=TeamSide.HOME,
    )
    stats = MatchStatsEngine(tactical_port=tactical)
    detector = ActionDetector(court_mapper=mapper)
    hybrid = HybridTaggingController(stats_engine=stats)

    # 3. Render Homography Spatial Mapping Table
    sample_pixels = [
        ("Saque Zaguero Derecho (HOME)", PixelPoint(1480.0, 880.0)),
        ("Remate Punta Z4 (HOME)", PixelPoint(520.0, 550.0)),
        ("Remate Opuesto Z2 (HOME)", PixelPoint(1380.0, 550.0)),
        ("Defensa Z6 Centro (AWAY)", PixelPoint(960.0, 310.0)),
        ("Balón Fuera Fondo (OUT)", PixelPoint(960.0, 140.0)),
    ]

    homography_table = Table(
        title="1. MAPEO ESPACIAL POR HOMOGRAFÍA (PÍXELES 3D -> PLANO 2D FIVB 18x9m)",
        border_style="#00f5ff",
        box=box.SIMPLE_HEAVY,
    )
    homography_table.add_column("Acción / Punto Cámara", style="bold white")
    homography_table.add_column("Pixel (u, v)", style="#00f5ff")
    homography_table.add_column("Pista 2D (x, y) [m]", style="#00ff66")
    homography_table.add_column("Campo", style="yellow")
    homography_table.add_column("Zona FIVB", style="bold magenta")

    for label, px in sample_pixels:
        res = mapper.pixel_to_zone(px)
        zone_str = f"Zona {res.zone.value}" if res.zone else "FUERA (OUT)"
        side_str = res.team_side.value if res.team_side else "OUT_OF_BOUNDS"
        homography_table.add_row(
            label,
            f"({px.u:.0f}, {px.v:.0f})",
            f"({res.court_point.x:.2f}m, {res.court_point.y:.2f}m)",
            side_str,
            zone_str,
        )
    console.print(homography_table)

    # 4. Simulate Rally Sequence with Side-Outs, Serves, and Hybrid-Tagged Spikes
    rally_script = [
        # (serve_outcome, attack_zone, attack_outcome, point_winner)
        (ServeOutcome.IN_PLAY, CourtZone.Z4, AttackOutcome.KILL, TeamSide.HOME),
        (ServeOutcome.ERROR_NET, None, None, TeamSide.AWAY),  # Side-out AWAY
        (ServeOutcome.IN_PLAY, CourtZone.Z2, AttackOutcome.KILL, TeamSide.HOME),  # Side-out HOME -> R2
        (ServeOutcome.ACE, None, None, TeamSide.HOME),
        (ServeOutcome.ERROR_OUT, None, None, TeamSide.AWAY),  # Side-out AWAY
        (ServeOutcome.IN_PLAY, CourtZone.Z3, AttackOutcome.KILL, TeamSide.HOME),  # Side-out HOME -> R3
        (ServeOutcome.ERROR_NET, None, None, TeamSide.AWAY),  # Error in R3
        (ServeOutcome.IN_PLAY, CourtZone.Z4, AttackOutcome.KILL, TeamSide.HOME),  # Side-out HOME -> R4
        (ServeOutcome.IN_PLAY, CourtZone.Z4, AttackOutcome.ERROR_OUT, TeamSide.AWAY),
        (ServeOutcome.IN_PLAY, CourtZone.Z6, AttackOutcome.KILL, TeamSide.HOME),  # Pipe attack Z6 -> R5
        (ServeOutcome.ERROR_OUT, None, None, TeamSide.AWAY),
        (ServeOutcome.IN_PLAY, CourtZone.Z2, AttackOutcome.KILL, TeamSide.HOME),  # Side-out HOME -> R6
    ]

    for srv_outcome, atk_zone, atk_outcome, winner in rally_script:
        serving_now = tactical.serving_team
        stats.record_serve(team=serving_now, outcome=srv_outcome, target_zone=CourtZone.Z6)
        if atk_zone is not None and atk_outcome is not None:
            pid = hybrid.propose_attack(
                team=TeamSide.HOME,
                origin_zone=atk_zone,
                outcome=atk_outcome,
            )
            hybrid.apply_hotkey_action(proposal_id=pid, hotkey="v")
        tactical.register_rally_point(winner=winner)

    # 5. Render Inferred Formation Snapshot for Current Rotation
    formation = tactical.infer_formation(TeamSide.HOME)
    rot_table = Table(
        title=f"2. ESTADO TÁCTICO ACTUAL (HOME: {formation.rotation.name} | Colocador en Zona {formation.setter_zone.value})",
        border_style="#00ff66",
        box=box.SIMPLE_HEAVY,
    )
    rot_table.add_column("Zona FIVB", style="bold #00f5ff")
    rot_table.add_column("Fila", style="white")
    rot_table.add_column("Rol Táctico Inferido (Sin OCR)", style="bold #00ff66")

    for z in CourtZone:
        row_type = "Delantero (Ataque)" if z.is_front_row else "Zaguero (Defensa)"
        rot_table.add_row(f"Zona {z.value}", row_type, formation.zone_roles[z].value)
    console.print(rot_table)

    # 6. Render Attack Efficiency by Zone
    atk_table = Table(
        title="3. EFICIENCIA DE REMATES POR ZONA DE ORIGEN (HOME)",
        border_style="magenta",
        box=box.SIMPLE_HEAVY,
    )
    atk_table.add_column("Zona Origen", style="bold magenta")
    atk_table.add_column("Intentos", justify="right")
    atk_table.add_column("Puntos (Kills)", justify="right", style="#00ff66")
    atk_table.add_column("Errores", justify="right", style="red")
    atk_table.add_column("Eficiencia", justify="right", style="bold yellow")

    zone_stats = stats.get_attack_stats_by_zone(TeamSide.HOME)
    for z, st in zone_stats.items():
        if int(st["attempts"]) > 0:
            atk_table.add_row(
                f"Zona {z.value}",
                str(st["attempts"]),
                str(st["kills"]),
                str(st["errors"]),
                f"{float(st['efficiency']) * 100:.1f}%",
            )
    console.print(atk_table)

    # 7. Render Serve Error Breakdown by Rotation (R1-R6)
    srv_table = Table(
        title="4. AUDITORÍA DE SAQUES FALLADOS POR ROTACIÓN (R1 - R6)",
        border_style="yellow",
        box=box.SIMPLE_HEAVY,
    )
    srv_table.add_column("Rotación", style="bold #00f5ff")
    srv_table.add_column("Saques Totales", justify="right")
    srv_table.add_column("Aces", justify="right", style="#00ff66")
    srv_table.add_column("Error Red", justify="right", style="red")
    srv_table.add_column("Error Fuera", justify="right", style="red")
    srv_table.add_column("Total Errores", justify="right", style="bold red")

    srv_matrix = stats.get_serve_errors_by_rotation(TeamSide.HOME)
    for rot, bucket in srv_matrix.items():
        srv_table.add_row(
            rot.name,
            str(bucket["TOTAL_SERVES"]),
            str(bucket["ACE"]),
            str(bucket["ERROR_NET"]),
            str(bucket["ERROR_OUT"]),
            str(bucket["TOTAL_ERRORS"]),
        )
    console.print(srv_table)

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="data_voley — Motor de Análisis Espacial y Táctico de Voleibol."
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Ejecuta la simulación táctica y espacial completa en terminal.",
    )
    parser.add_argument(
        "--stream",
        type=str,
        default=None,
        help="Enlace HLS (.m3u8) o ruta de vídeo local para ingesta.",
    )
    args = parser.parse_args()

    if args.demo or not args.stream:
        return run_demo()

    print(f"[data_voley] Conectando al stream: {args.stream}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
