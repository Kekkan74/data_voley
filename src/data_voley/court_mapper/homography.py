"""
CourtMapper Module — Perspective Homography (3D to 2D) and FIVB Zone 1-6 Resolver.

Maps camera pixel coordinates (u, v) to the standard FIVB 9.0m x 18.0m metric plane
via a 3x3 projective homography matrix H, and classifies 2D coordinates into official
FIVB Court Zones (1 to 6) for both HOME (near) and AWAY (far) half-courts.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

try:
    import cv2

    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

from data_voley.domain.enums import CourtZone, TeamSide
from data_voley.domain.models import CourtPoint2D, PixelPoint, SpatialZoneMapping


class CourtMapper:
    """
    Spatial Homography & FIVB Zone Classification Engine.

    FIVB Official Court Dimensions (Meters):
    - Width (X-axis):  0.0m (Left Sideline) to 9.0m (Right Sideline)
    - Length (Y-axis): 0.0m (HOME Endline) to 18.0m (AWAY Endline)
    - Center Net Line: y = 9.0m
    - HOME Attack Line (3m): y = 6.0m
    - AWAY Attack Line (3m): y = 12.0m
    """

    COURT_WIDTH_M: float = 9.0
    COURT_LENGTH_M: float = 18.0
    NET_Y_M: float = 9.0
    HOME_ATTACK_LINE_Y_M: float = 6.0
    AWAY_ATTACK_LINE_Y_M: float = 12.0

    # Canonical FIVB 4 outer corners in meters:
    # [Bottom-Left (0,0), Bottom-Right (9,0), Top-Right (9,18), Top-Left (0,18)]
    CANONICAL_COURT_CORNERS: np.ndarray = np.array(
        [
            [0.0, 0.0],
            [9.0, 0.0],
            [9.0, 18.0],
            [0.0, 18.0],
        ],
        dtype=np.float64,
    )

    def __init__(self, homography_matrix: np.ndarray) -> None:
        if homography_matrix.shape != (3, 3):
            raise ValueError(
                f"Homography matrix H must have shape (3, 3), got {homography_matrix.shape}."
            )
        self._H: np.ndarray = homography_matrix.astype(np.float64)
        self._H_inv: np.ndarray = np.linalg.inv(self._H)

    @property
    def homography_matrix(self) -> np.ndarray:
        """Return a copy of the 3x3 homography matrix H."""
        return self._H.copy()

    @staticmethod
    def _solve_dlt_homography(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
        """
        Compute exact 3x3 projective homography matrix H via Direct Linear Transform (SVD).

        Solves s * [x, y, 1]^T = H * [u, v, 1]^T for N >= 4 correspondences.
        """
        n_pts = src_pts.shape[0]
        if n_pts < 4:
            raise ValueError("At least 4 non-collinear point correspondences are required.")

        A = []
        for i in range(n_pts):
            u, v = float(src_pts[i, 0]), float(src_pts[i, 1])
            x, y = float(dst_pts[i, 0]), float(dst_pts[i, 1])
            A.append([-u, -v, -1.0, 0.0, 0.0, 0.0, u * x, v * x, x])
            A.append([0.0, 0.0, 0.0, -u, -v, -1.0, u * y, v * y, y])

        A_mat = np.asarray(A, dtype=np.float64)
        _, _, vh = np.linalg.svd(A_mat)
        h = vh[-1, :]
        H = h.reshape((3, 3))
        if abs(H[2, 2]) > 1e-12:
            H = H / H[2, 2]
        return H

    @classmethod
    def from_camera_corners(
        cls,
        camera_corners_px: np.ndarray,
        court_corners_m: Optional[np.ndarray] = None,
    ) -> CourtMapper:
        """
        Instantiate a calibrated CourtMapper from 4+ camera pixel points.

        Parameters
        ----------
        camera_corners_px : np.ndarray
            Array of shape (N, 2) with pixel coordinates (u, v).
        court_corners_m : Optional[np.ndarray]
            Array of shape (N, 2) with court coordinates (x, y) in meters.
            Defaults to the 4 outer corners of the 9m x 18m FIVB court.
        """
        src = np.asarray(camera_corners_px, dtype=np.float64)
        dst = (
            np.asarray(court_corners_m, dtype=np.float64)
            if court_corners_m is not None
            else cls.CANONICAL_COURT_CORNERS
        )

        if src.shape != dst.shape or src.shape[0] < 4 or src.shape[1] != 2:
            raise ValueError("camera_corners_px and court_corners_m must have shape (N, 2) with N >= 4.")

        if HAS_OPENCV and src.shape[0] > 4:
            H, _ = cv2.findHomography(src.astype(np.float32), dst.astype(np.float32), cv2.RANSAC, 5.0)
            if H is not None:
                return cls(H)

        H = cls._solve_dlt_homography(src, dst)
        return cls(H)

    def pixel_to_court(self, pixel: PixelPoint) -> CourtPoint2D:
        """
        Project a 2D camera pixel (u, v) to 2D real-world court meters (x, y).

        Transformation:
            [x', y', w']^T = H * [u, v, 1]^T
            (x, y) = (x' / w', y' / w')
        """
        vec = np.array([pixel.u, pixel.v, 1.0], dtype=np.float64)
        projected = self._H @ vec
        w = projected[2]
        if abs(w) < 1e-12:
            raise ZeroDivisionError("Point projects to infinity under current homography H.")
        return CourtPoint2D(x=float(projected[0] / w), y=float(projected[1] / w))

    def court_to_pixel(self, point: CourtPoint2D) -> PixelPoint:
        """Back-project a 2D court coordinate (x, y) into camera pixel space (u, v)."""
        vec = np.array([point.x, point.y, 1.0], dtype=np.float64)
        projected = self._H_inv @ vec
        w = projected[2]
        if abs(w) < 1e-12:
            raise ZeroDivisionError("Court point back-projects to infinity.")
        return PixelPoint(u=float(projected[0] / w), v=float(projected[1] / w))

    def resolve_zone(self, point: CourtPoint2D) -> SpatialZoneMapping:
        """
        Classify a 2D metric court point into TeamSide (HOME/AWAY) and FIVB CourtZone (1..6).

        Geometry of Zones:
        - HOME Half-Court (0.0 <= y <= 9.0m, facing +Y toward Net at y=9.0m):
          * Front Row (6.0 <= y <= 9.0m):
              x in [0, 3) -> Z4 (Left Front)
              x in [3, 6) -> Z3 (Middle Front)
              x in [6, 9] -> Z2 (Right Front)
          * Back Row (0.0 <= y < 6.0m):
              x in [0, 3) -> Z5 (Left Back)
              x in [3, 6) -> Z6 (Middle Back)
              x in [6, 9] -> Z1 (Right Back)

        - AWAY Half-Court (9.0 < y <= 18.0m, facing -Y toward Net at y=9.0m, 180° rotated):
          * Front Row (9.0 < y <= 12.0m):
              x in [0, 3) -> Z2 (Right Front from AWAY perspective)
              x in [3, 6) -> Z3 (Middle Front)
              x in [6, 9] -> Z4 (Left Front from AWAY perspective)
          * Back Row (12.0 < y <= 18.0m):
              x in [0, 3) -> Z1 (Right Back from AWAY perspective)
              x in [3, 6) -> Z6 (Middle Back)
              x in [6, 9] -> Z5 (Left Back from AWAY perspective)
        """
        if not point.is_in_bounds:
            return SpatialZoneMapping(
                court_point=point,
                team_side=None,
                zone=None,
                is_in_bounds=False,
                is_front_zone=False,
            )

        x, y = point.x, point.y

        # Determine Lateral Column (0: [0,3), 1: [3,6), 2: [6,9])
        if x < 3.0:
            col = 0
        elif x < 6.0:
            col = 1
        else:
            col = 2

        if y <= self.NET_Y_M:
            team = TeamSide.HOME
            is_front = y >= self.HOME_ATTACK_LINE_Y_M
            if is_front:
                zone_map = {0: CourtZone.Z4, 1: CourtZone.Z3, 2: CourtZone.Z2}
            else:
                zone_map = {0: CourtZone.Z5, 1: CourtZone.Z6, 2: CourtZone.Z1}
            zone = zone_map[col]
        else:
            team = TeamSide.AWAY
            is_front = y <= self.AWAY_ATTACK_LINE_Y_M
            # Rotated 180 degrees relative to HOME
            if is_front:
                zone_map = {0: CourtZone.Z2, 1: CourtZone.Z3, 2: CourtZone.Z4}
            else:
                zone_map = {0: CourtZone.Z1, 1: CourtZone.Z6, 2: CourtZone.Z5}
            zone = zone_map[col]

        return SpatialZoneMapping(
            court_point=point,
            team_side=team,
            zone=zone,
            is_in_bounds=True,
            is_front_zone=is_front,
        )

    def pixel_to_zone(self, pixel: PixelPoint) -> SpatialZoneMapping:
        """Project a camera pixel (u, v) and resolve its FIVB CourtZone in a single step."""
        court_pt = self.pixel_to_court(pixel)
        return self.resolve_zone(court_pt)
