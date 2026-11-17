"""
K1/K2 Chained Tactical Domain Models (Reception -> Setting -> Attack).

Pure Python + Pydantic v2 models with zero dependencies on OpenCV, FastAPI, or UI.
Provides geometric evaluation of reception quality (distance to Zone 2.5 [6.0, 8.5])
and attack trajectory classification (LINE, CROSS_LONG, CROSS_SHORT, TIP).
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Optional, Tuple

from pydantic import BaseModel, Field, model_validator

from data_voley.domain.enums import Rotation

# Ideal setter target coordinate on the 9m x 18m court: Zone 2.5
IDEAL_SETTER_TARGET_XY: Tuple[float, float] = (6.0, 8.5)
NET_Y_METERS: float = 9.0


class ReceptionQuality(str, Enum):
    """FIVB / DataVolley standard reception quality grades."""

    PERFECT = "#"   # <= 1.5m from Zone 2.5 [6.0, 8.5] (all 1st tempo / pipe options)
    POSITIVE = "+"  # 1.5m - 3.0m from Zone 2.5 (fast wing sets available)
    MEDIUM = "!"    # High/slow ball on target, no 1st tempo
    NEGATIVE = "-"  # > 3.0m from net/target, out-of-system high ball
    OVERPASS = "/"  # Ball crosses directly over the net to opponent court
    ERROR = "="     # Direct reception error / opponent ace


class SetTempo(str, Enum):
    """Attack tempo classification of the setter's pass."""

    FIRST = "1st"   # Quick middle attack
    SECOND = "2nd"  # Fast wing / Super / Pipe
    THIRD = "3rd"   # High out-of-system safety ball
    DUMP = "DUMP"   # Setter second-touch attack


class SetQuality(str, Enum):
    """Execution quality of the setter's pass."""

    SUCCESSFUL = "SUCCESSFUL"
    IMPRECISE = "IMPRECISE"
    FAULT = "FAULT"


class AttackTrajectory(str, Enum):
    """Geometric attack direction on the opponent court."""

    LINE = "LINE"
    CROSS_LONG = "CROSS_LONG"
    CROSS_SHORT = "CROSS_SHORT"
    TIP = "TIP"


class K1AttackOutcome(str, Enum):
    """FIVB / DataVolley standard attack outcome symbols."""

    KILL = "#"           # Direct point
    POSITIVE_CONT = "+"  # Positive continuation / opponent forced defense
    RECYCLE = "!"        # Recycled off block / intentional block-out
    BLOCKED = "/"        # Stuffed for opponent point
    ERROR = "="          # Out or net error


def evaluate_reception_quality(target_xy: Tuple[float, float]) -> ReceptionQuality:
    """
    Evaluate reception quality from 2D target court coordinates (in meters).

    Rules:
    - y > 9.0m (crossed net into opponent court) -> OVERPASS ('/')
    - Euclidean distance to Zone 2.5 [6.0, 8.5] <= 1.5m -> PERFECT ('#')
    - 1.5m < distance <= 3.0m -> POSITIVE ('+')
    - distance > 3.0m -> NEGATIVE ('-')
    """
    x, y = target_xy
    if y > NET_Y_METERS:
        return ReceptionQuality.OVERPASS
    dist = math.hypot(x - IDEAL_SETTER_TARGET_XY[0], y - IDEAL_SETTER_TARGET_XY[1])
    if dist <= 1.5:
        return ReceptionQuality.PERFECT
    if dist <= 3.0:
        return ReceptionQuality.POSITIVE
    return ReceptionQuality.NEGATIVE


def evaluate_attack_trajectory(
    origin_zone: int, target_xy: Tuple[float, float]
) -> AttackTrajectory:
    """
    Classify attack trajectory geometrically from origin_zone (1-6) and
    opponent-court landing coordinates target_xy (x in [0,9], y in (9,18]).
    """
    x, y = target_xy
    y_past_net = abs(y - NET_Y_METERS)

    if y_past_net <= 1.8:
        return AttackTrajectory.TIP

    is_short_depth = y_past_net <= 4.5

    if origin_zone in (4, 5):  # Left side
        if x < 3.0:
            return AttackTrajectory.LINE
        return AttackTrajectory.CROSS_SHORT if is_short_depth else AttackTrajectory.CROSS_LONG

    if origin_zone in (2, 1):  # Right side
        if x > 6.0:
            return AttackTrajectory.LINE
        return AttackTrajectory.CROSS_SHORT if is_short_depth else AttackTrajectory.CROSS_LONG

    # Center / Pipe (Zones 3, 6)
    if 3.0 <= x <= 6.0:
        return AttackTrajectory.LINE
    return AttackTrajectory.CROSS_SHORT if is_short_depth else AttackTrajectory.CROSS_LONG


class ReceptionEvent(BaseModel):
    """First touch in a K1/K2 sequence."""

    receptor_zone: int = Field(ge=1, le=6)
    origin_xy: Tuple[float, float]
    target_xy: Tuple[float, float]
    quality: Optional[ReceptionQuality] = None

    @model_validator(mode="after")
    def _populate_quality(self) -> ReceptionEvent:
        if self.quality is None:
            self.quality = evaluate_reception_quality(self.target_xy)
        return self


class SetEvent(BaseModel):
    """Second touch (setter distribution) in a K1/K2 sequence."""

    setter_origin_xy: Tuple[float, float]
    target_zone: int = Field(ge=1, le=6)
    is_in_system: Optional[bool] = None
    tempo: SetTempo = SetTempo.SECOND
    quality: SetQuality = SetQuality.SUCCESSFUL


class K1AttackEvent(BaseModel):
    """Third touch (attack/spike) in a K1/K2 sequence."""

    origin_zone: int = Field(ge=1, le=6)
    target_xy: Tuple[float, float]
    blockers_faced: int = Field(ge=0, le=3, default=2)
    trajectory_type: Optional[AttackTrajectory] = None
    outcome: K1AttackOutcome

    @model_validator(mode="after")
    def _populate_trajectory(self) -> K1AttackEvent:
        if self.trajectory_type is None:
            self.trajectory_type = evaluate_attack_trajectory(
                self.origin_zone, self.target_xy
            )
        return self


# Alias within K1 domain module
AttackEvent = K1AttackEvent


class RallyK1Sequence(BaseModel):
    """Chained Reception -> Setting -> Attack sequence for a single K1/K2 phase."""

    set_number: int = Field(ge=1)
    rotation: Rotation
    reception: ReceptionEvent
    setting: Optional[SetEvent] = None
    attack: Optional[K1AttackEvent] = None
    side_out_won: Optional[bool] = None
    timestamp_sec: float = 0.0

    @model_validator(mode="after")
    def _sync_sequence_invariants(self) -> RallyK1Sequence:
        if self.setting is not None and self.setting.is_in_system is None:
            self.setting.is_in_system = self.reception.quality in (
                ReceptionQuality.PERFECT,
                ReceptionQuality.POSITIVE,
            )
        if self.side_out_won is None:
            if self.attack is not None:
                self.side_out_won = self.attack.outcome == K1AttackOutcome.KILL
            elif self.reception.quality in (
                ReceptionQuality.ERROR,
                ReceptionQuality.OVERPASS,
            ):
                self.side_out_won = False
            elif self.setting is not None and self.setting.tempo == SetTempo.DUMP:
                self.side_out_won = self.setting.quality == SetQuality.SUCCESSFUL
            else:
                self.side_out_won = False
        return self
