"""
TacticalState Module — FIVB Match, Set, and Rotation (R1-R6) Finite State Machine.

Models the official FIVB scoring, set termination, side-out transitions, and
spatial role inference across Zones 1-6 using pure Python and modular arithmetic.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from data_voley.domain.enums import CourtZone, PlayerRole, Rotation, TeamSide
from data_voley.domain.models import RallyTransitionEvent, RotationFormationSnapshot


class TacticalState:
    """
    Finite State Machine governing match score, set transitions, and R1-R6 rotations.

    Mathematical Model of Side-Out Rotation Transitions:
    ----------------------------------------------------
    Let R_T(t) in {1..6} be the rotation state of team T at rally t, and S(t) be the
    serving team. When rally t is won by team W:
    1. Break Point (W == S(t)):
       Service is retained (S(t+1) = S(t)) and rotations remain invariant:
           R_HOME(t+1) = R_HOME(t),  R_AWAY(t+1) = R_AWAY(t)
    2. Side-Out (W != S(t)):
       Service is recovered by W (S(t+1) = W) and ONLY team W advances one step
       in the cyclic group Z_6:
           R_W(t+1) = (R_W(t) mod 6) + 1

    Spatial Role Inference Formula (5-1 System):
    --------------------------------------------
    Since FIVB CourtZones (1..6) are numbered counter-clockwise while players rotate
    clockwise (Z2 -> Z1 -> Z6 -> Z5 -> Z4 -> Z3 -> Z2), a role starting at zone z_0
    in R1 occupies zone z(R_k) in rotation R_k (k in 1..6) given by:
        z(R_k) = ((z_0 - k) mod 6) + 1
    """

    # Standard 5-1 base lineup in R1 (Setter in Zone 1, diagonal pairs separated by 3 zones)
    BASE_R1_ROLE_BY_ZONE: Dict[CourtZone, PlayerRole] = {
        CourtZone.Z1: PlayerRole.SETTER,
        CourtZone.Z2: PlayerRole.OUTSIDE_1,
        CourtZone.Z3: PlayerRole.MIDDLE_1,
        CourtZone.Z4: PlayerRole.OPPOSITE,
        CourtZone.Z5: PlayerRole.OUTSIDE_2,
        CourtZone.Z6: PlayerRole.MIDDLE_2,
    }

    def __init__(
        self,
        home_initial_rotation: Rotation = Rotation.R1,
        away_initial_rotation: Rotation = Rotation.R1,
        initial_server: TeamSide = TeamSide.HOME,
        max_sets: int = 5,
    ) -> None:
        self._max_sets: int = max_sets
        self._current_set: int = 1
        self._rally_counter: int = 0
        self._serving_team: TeamSide = initial_server
        self._match_finished: bool = False

        self._rotations: Dict[TeamSide, Rotation] = {
            TeamSide.HOME: home_initial_rotation,
            TeamSide.AWAY: away_initial_rotation,
        }
        self._score: Dict[TeamSide, int] = {
            TeamSide.HOME: 0,
            TeamSide.AWAY: 0,
        }
        self._sets_won: Dict[TeamSide, int] = {
            TeamSide.HOME: 0,
            TeamSide.AWAY: 0,
        }
        self._history: List[RallyTransitionEvent] = []

    @property
    def current_set(self) -> int:
        """Return the active 1-based set number."""
        return self._current_set

    @property
    def serving_team(self) -> TeamSide:
        """Return the team currently holding service possession."""
        return self._serving_team

    @property
    def sets_won(self) -> Dict[TeamSide, int]:
        """Return a copy of sets won by each team."""
        return dict(self._sets_won)

    @property
    def match_finished(self) -> bool:
        """Return True if a team has clinched the required number of sets."""
        return self._match_finished

    @property
    def history(self) -> List[RallyTransitionEvent]:
        """Return chronological list of rally transition events."""
        return list(self._history)

    def get_rotation(self, team: TeamSide) -> Rotation:
        """Return the active tactical rotation (R1-R6) for `team`."""
        return self._rotations[team]

    def set_rotation(self, team: TeamSide, rotation: Rotation) -> None:
        """Explicitly set or override the rotation of `team` (e.g., at set start or manual correction)."""
        self._rotations[team] = rotation

    def get_score(self) -> Tuple[int, int]:
        """Return (home_score, away_score) for the active set."""
        return self._score[TeamSide.HOME], self._score[TeamSide.AWAY]

    def _target_points_for_current_set(self) -> int:
        """Return 15 points for the deciding 5th set, or 25 points for sets 1 to 4."""
        return 15 if self._current_set == self._max_sets else 25

    def _is_set_clinched(self) -> Optional[TeamSide]:
        """Check if either team has reached the target score with a >= 2 point advantage."""
        target = self._target_points_for_current_set()
        h_pts = self._score[TeamSide.HOME]
        a_pts = self._score[TeamSide.AWAY]

        if h_pts >= target and (h_pts - a_pts) >= 2:
            return TeamSide.HOME
        if a_pts >= target and (a_pts - h_pts) >= 2:
            return TeamSide.AWAY
        return None

    def register_rally_point(self, winner: TeamSide) -> RallyTransitionEvent:
        """
        Process the completion of a rally won by `winner`.

        Updates score, evaluates side-out rotation shift, and checks set/match completion.
        """
        if self._match_finished:
            raise RuntimeError("Cannot register rally points on a finished match.")

        self._rally_counter += 1
        serving_before = self._serving_team
        side_out = winner != serving_before

        # 1. Increment rally winner's score
        self._score[winner] += 1

        # 2. Evaluate Side-Out Rotation Transition
        if side_out:
            self._serving_team = winner
            self._rotations[winner] = self._rotations[winner].next_rotation()

        # 3. Check Set / Match victory conditions
        set_winner = self._is_set_clinched()
        set_finished = set_winner is not None
        sets_needed_to_win = (self._max_sets // 2) + 1

        active_set_num = self._current_set
        h_score_snapshot = self._score[TeamSide.HOME]
        a_score_snapshot = self._score[TeamSide.AWAY]

        if set_winner is not None:
            self._sets_won[set_winner] += 1
            if self._sets_won[set_winner] >= sets_needed_to_win:
                self._match_finished = True
            else:
                self._current_set += 1
                self._score[TeamSide.HOME] = 0
                self._score[TeamSide.AWAY] = 0

        event = RallyTransitionEvent(
            set_number=active_set_num,
            rally_number=self._rally_counter,
            serving_team_before=serving_before,
            point_winner=winner,
            side_out_occurred=side_out,
            home_score=h_score_snapshot,
            away_score=a_score_snapshot,
            home_rotation=self._rotations[TeamSide.HOME],
            away_rotation=self._rotations[TeamSide.AWAY],
            set_finished=set_finished,
            match_finished=self._match_finished,
        )
        self._history.append(event)
        return event

    def infer_formation(
        self, team: TeamSide, use_libero_in_back_row: bool = True
    ) -> RotationFormationSnapshot:
        """
        Compute the exact spatial map of PlayerRoles across Zones 1 to 6 for `team`
        in its current rotation without requiring jersey OCR or facial recognition.
        """
        rot = self._rotations[team]
        k = rot.value  # 1..6

        zone_roles: Dict[CourtZone, PlayerRole] = {}
        for base_zone, role in self.BASE_R1_ROLE_BY_ZONE.items():
            # Direct mapping where R_k places Setter (base_zone=1) at CourtZone k:
            # z(R_k) = ((z_0 + k - 2) mod 6) + 1
            current_zone_num = ((base_zone.value + k - 2) % 6) + 1
            current_zone = CourtZone(current_zone_num)

            # Apply FIVB Rule 19: Libero replaces Middle Blocker in back-row defense zones (Z5, Z6)
            if (
                use_libero_in_back_row
                and role in (PlayerRole.MIDDLE_1, PlayerRole.MIDDLE_2)
                and current_zone in (CourtZone.Z5, CourtZone.Z6)
            ):
                zone_roles[current_zone] = PlayerRole.LIBERO
            else:
                zone_roles[current_zone] = role

        setter_zone = rot.setter_zone
        return RotationFormationSnapshot(
            team=team,
            rotation=rot,
            setter_zone=setter_zone,
            setter_is_front_row=setter_zone.is_front_row,
            zone_roles=zone_roles,
        )

    def infer_role_at_zone(self, team: TeamSide, zone: CourtZone) -> PlayerRole:
        """Return the inferred tactical role operating in `zone` for `team`."""
        snapshot = self.infer_formation(team)
        return snapshot.zone_roles[zone]
