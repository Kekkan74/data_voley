"""
Domain Enumerations for FIVB Volleyball Tactical & Spatial Modeling.

All enumerations are pure Python types with zero external UI or CV dependencies.
"""

from __future__ import annotations

from enum import Enum


class TeamSide(str, Enum):
    """Identifies the two competing teams in a match."""

    HOME = "HOME"  # Near court (0.0m <= y <= 9.0m by default)
    AWAY = "AWAY"  # Far court (9.0m < y <= 18.0m by default)

    def opponent(self) -> TeamSide:
        """Return the opposing team side."""
        return TeamSide.AWAY if self == TeamSide.HOME else TeamSide.HOME


class CourtZone(int, Enum):
    """
    Official FIVB Court Zones (1 to 6).

    Numbered counter-clockwise starting from the right-back service zone (Zone 1):

           [      NET (y = 9.0m)      ]
           +--------+--------+--------+
           | Zone 4 | Zone 3 | Zone 2 |  <- Front Row (Attack Zone: 3m to net)
           +--------+--------+--------+
           | Zone 5 | Zone 6 | Zone 1 |  <- Back Row (Defense Zone: endline to 3m)
           +--------+--------+--------+
           [    ENDLINE (y = 0.0m)    ]
    """

    Z1 = 1  # Right Back (Service position)
    Z2 = 2  # Right Front
    Z3 = 3  # Middle Front
    Z4 = 4  # Left Front
    Z5 = 5  # Left Back
    Z6 = 6  # Middle Back

    @property
    def is_front_row(self) -> bool:
        """Return True if the zone is in the front attack zone (Zones 2, 3, 4)."""
        return self in (CourtZone.Z2, CourtZone.Z3, CourtZone.Z4)

    @property
    def is_back_row(self) -> bool:
        """Return True if the zone is in the back defense zone (Zones 1, 5, 6)."""
        return self in (CourtZone.Z1, CourtZone.Z5, CourtZone.Z6)

    def next_clockwise_zone(self) -> CourtZone:
        """
        Compute the next physical court zone when a player rotates clockwise.

        Because FIVB zones are numbered counter-clockwise (1 -> 2 -> 3 -> 4 -> 5 -> 6),
        a clockwise physical movement on the court corresponds to decrementing the
        zone index modulo 6:
            Z2 -> Z1 -> Z6 -> Z5 -> Z4 -> Z3 -> Z2
            z_next = ((z - 2) mod 6) + 1
        """
        next_val = ((self.value - 2) % 6) + 1
        return CourtZone(next_val)


class Rotation(int, Enum):
    """
    Tactical Rotation States (R1 to R6) in a standard 5-1 system.

    In tactical volleyball analytics:
    - R1: Setter starts in Zone 1 (Right Back)
    - R2: After 1 clockwise rotation, Setter is in Zone 6 (Middle Back)
    - R3: After 2 clockwise rotations, Setter is in Zone 5 (Left Back)
    - R4: After 3 clockwise rotations, Setter is in Zone 4 (Left Front)
    - R5: After 4 clockwise rotations, Setter is in Zone 3 (Middle Front)
    - R6: After 5 clockwise rotations, Setter is in Zone 2 (Right Front)
    """

    R1 = 1
    R2 = 2
    R3 = 3
    R4 = 4
    R5 = 5
    R6 = 6

    def next_rotation(self) -> Rotation:
        """
        Advance to the next tactical rotation upon a side-out (recovery of serve).

        Mathematical transition on the cyclic group Z_6:
            R_{t+1} = ((R_t - 1 + 1) mod 6) + 1 = (R_t mod 6) + 1
        """
        return Rotation((self.value % 6) + 1)

    @property
    def setter_zone(self) -> CourtZone:
        """
        Return the FIVB CourtZone (1-6) occupied by the Setter in this rotation.

        Mapping formula from R_k (k in 1..6) to CourtZone:
            z(R_1) = 1
            z(R_k) = ((1 - (k - 1) - 1) mod 6) + 1 = ((1 - k) mod 6) + 1
        """
        zone_num = ((1 - self.value) % 6) + 1
        return CourtZone(zone_num)


class PlayerRole(str, Enum):
    """Tactical roles in an indoor 5-1 volleyball lineup."""

    SETTER = "SETTER"          # Colocador / Armador (S)
    OPPOSITE = "OPPOSITE"      # Opuesto (OPP) - Diagonal to Setter
    OUTSIDE_1 = "OUTSIDE_1"    # Receptor Principal (OH1)
    OUTSIDE_2 = "OUTSIDE_2"    # Receptor Secundario (OH2) - Diagonal to OH1
    MIDDLE_1 = "MIDDLE_1"      # Central 1 (MB1)
    MIDDLE_2 = "MIDDLE_2"      # Central 2 (MB2) - Diagonal to MB1
    LIBERO = "LIBERO"          # Líbero (L) - Replaces back-row Middle Blocker


class ActionType(str, Enum):
    """Primary match actions tracked by the spatial analytics engine."""

    SERVE = "SERVE"
    RECEPTION = "RECEPTION"
    SET = "SET"
    ATTACK = "ATTACK"
    BLOCK = "BLOCK"
    DIG = "DIG"


class ServeOutcome(str, Enum):
    """Result classification of a service action."""

    ACE = "ACE"                        # Direct point
    IN_PLAY = "IN_PLAY"                # Received by opponent, rally continues
    ERROR_NET = "ERROR_NET"            # Ball hits the net and fails to cross
    ERROR_OUT = "ERROR_OUT"            # Ball lands out of bounds
    ERROR_FOOT_FAULT = "ERROR_FOOT"    # Server steps on/over endline

    @property
    def is_error(self) -> bool:
        """Return True if the serve resulted in an immediate point for the opponent."""
        return self in (
            ServeOutcome.ERROR_NET,
            ServeOutcome.ERROR_OUT,
            ServeOutcome.ERROR_FOOT_FAULT,
        )


class AttackOutcome(str, Enum):
    """Result classification of an attack/spike action."""

    KILL = "KILL"              # Direct point
    IN_PLAY = "IN_PLAY"        # Defended, rally continues
    BLOCKED = "BLOCKED"        # Stuffed by opponent block for a point
    ERROR_NET = "ERROR_NET"    # Attack into the net
    ERROR_OUT = "ERROR_OUT"    # Attack lands out of bounds without touch

    @property
    def is_error(self) -> bool:
        """Return True if the attack resulted in a direct loss of point."""
        return self in (
            AttackOutcome.BLOCKED,
            AttackOutcome.ERROR_NET,
            AttackOutcome.ERROR_OUT,
        )


class TagValidationStatus(str, Enum):
    """Provenance and human-in-the-loop verification status of an event tag."""

    AI_PROPOSED = "AI_PROPOSED"          # Raw prediction from YOLO/TrackNet
    HUMAN_VALIDATED = "HUMAN_VALIDATED"  # Confirmed via keyboard shortcut
    HUMAN_CORRECTED = "HUMAN_CORRECTED"  # Adjusted zone/outcome by analyst
    HUMAN_MANUAL = "HUMAN_MANUAL"        # Manually inserted by analyst
