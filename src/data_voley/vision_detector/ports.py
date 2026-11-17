"""
Ports (Protocols) and DTOs for future Computer Vision (YOLO / TrackNet) Adapters.
"""

from __future__ import annotations

from typing import Any, List, Optional, Protocol, Tuple

from pydantic import BaseModel, Field


class FrameProposal(BaseModel):
    """Structured coordinate & action proposal emitted by a CV detector for a video frame."""

    frame_index: int = Field(ge=0)
    timestamp_sec: float = Field(ge=0.0)
    ball_xy: Optional[Tuple[float, float]] = None
    player_positions_xy: List[Tuple[float, float]] = Field(default_factory=list)
    proposed_action: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)


class VisionDetectorPort(Protocol):
    """Port contract for YOLO/TrackNet adapters proposing coordinates without coupling the core."""

    def detect_frame(
        self, frame_index: int, timestamp_sec: float, frame_data: Any
    ) -> FrameProposal:
        """Analyze a video frame and return a spatial coordinate proposal."""
        ...

    def propose_k1_coordinates(self, timestamp_sec: float) -> Optional[FrameProposal]:
        """Return a coordinate proposal near `timestamp_sec`, if available."""
        ...
