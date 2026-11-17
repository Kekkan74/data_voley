"""
VideoIngestion Module — Asynchronous / Multithreaded HLS (.m3u8) & Video Stream Reader.

Isolates video stream decoding (FFmpeg / OpenCV) from tactical logic and homography.
Supports seeking directly to a match start offset (skipping warm-ups or prior matches).
"""

from __future__ import annotations

import queue
import threading
from typing import Iterator, Optional, Tuple
from urllib.parse import urlparse

import numpy as np

try:
    import cv2

    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False


class VideoIngestion:
    """
    Decoupled producer-consumer stream reader for HLS (.m3u8), RTSP, or local video files.

    Supports `start_offset_seconds` to skip earlier matches or warm-up segments in
    long tournament broadcasts.
    """

    ALLOWED_SCHEMES = ("http", "https", "rtsp", "file", "")

    def __init__(
        self,
        source: str,
        start_offset_seconds: float = 0.0,
        buffer_size: int = 128,
        target_fps: Optional[float] = None,
    ) -> None:
        self.source = self._validate_source(source)
        self.start_offset_seconds = max(0.0, float(start_offset_seconds))
        self.buffer_size = buffer_size
        self.target_fps = target_fps
        self._queue: queue.Queue[Optional[Tuple[int, float, np.ndarray]]] = queue.Queue(
            maxsize=buffer_size
        )
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None

    @staticmethod
    def parse_start_time_to_seconds(value: str | int | float) -> float:
        """
        Convert a user-supplied start time into total seconds.

        Accepted formats:
        - Integer/float minutes: "15" -> 900.0 seconds (15 min)
        - "MM:SS": "12:30" -> 750.0 seconds (12 min 30 sec)
        - "HH:MM:SS": "01:05:20" -> 3920.0 seconds
        """
        if isinstance(value, (int, float)):
            return max(0.0, float(value) * 60.0)

        cleaned = str(value).strip()
        if not cleaned:
            return 0.0

        parts = cleaned.split(":")
        try:
            if len(parts) == 1:
                # Treated as minutes
                return max(0.0, float(parts[0]) * 60.0)
            if len(parts) == 2:
                minutes = int(parts[0])
                seconds = float(parts[1])
                return max(0.0, minutes * 60.0 + seconds)
            if len(parts) == 3:
                hours = int(parts[0])
                minutes = int(parts[1])
                seconds = float(parts[2])
                return max(0.0, hours * 3600.0 + minutes * 60.0 + seconds)
        except ValueError as exc:
            raise ValueError(f"Invalid start time format: '{value}'. Use 'MM:SS' or minutes.") from exc

        raise ValueError(f"Unsupported time format: '{value}'.")

    @staticmethod
    def format_seconds_to_mmss(total_seconds: float) -> str:
        """Format seconds as MM:SS (or HH:MM:SS if >= 1 hour)."""
        secs = max(0, int(round(total_seconds)))
        hours = secs // 3600
        minutes = (secs % 3600) // 60
        rem_secs = secs % 60
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{rem_secs:02d}"
        return f"{minutes:02d}:{rem_secs:02d}"

    @classmethod
    def _validate_source(cls, source: str) -> str:
        """Validate URL or file path syntax."""
        cleaned = source.strip()
        if not cleaned:
            raise ValueError("Video stream source cannot be empty.")
        parsed = urlparse(cleaned)
        if parsed.scheme and parsed.scheme.lower() not in cls.ALLOWED_SCHEMES:
            if len(parsed.scheme) > 1:
                raise ValueError(f"Unsupported stream protocol: {parsed.scheme}")
        return cleaned

    def start(self) -> None:
        """Start the background decoding thread."""
        if not HAS_OPENCV:
            raise RuntimeError("OpenCV (cv2) is required to decode live video streams.")
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return

        self._stop_event.clear()
        self._worker_thread = threading.Thread(target=self._decode_worker, daemon=True)
        self._worker_thread.start()

    def _decode_worker(self) -> None:
        """Producer loop pulling frames starting at `start_offset_seconds`."""
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            self._queue.put(None)
            return

        if self.start_offset_seconds > 0.0:
            cap.set(cv2.CAP_PROP_POS_MSEC, self.start_offset_seconds * 1000.0)

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_idx = 0

        try:
            while not self._stop_event.is_set():
                if not self._queue.full():
                    ret, frame = cap.read()
                    if not ret:
                        self._queue.put(None)
                        break
                    timestamp_ms = (self.start_offset_seconds * 1000.0) + (frame_idx / fps) * 1000.0
                    self._queue.put((frame_idx, timestamp_ms, frame))
                    frame_idx += 1
                else:
                    self._stop_event.wait(0.002)
        finally:
            cap.release()

    def frames(self) -> Iterator[Tuple[int, float, np.ndarray]]:
        """Yield (frame_index, timestamp_ms, bgr_frame) tuples until stream ends."""
        if self._worker_thread is None:
            self.start()

        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=0.5)
                if item is None:
                    break
                yield item
            except queue.Empty:
                if self._worker_thread and not self._worker_thread.is_alive():
                    break

    def release(self) -> None:
        """Stop the background worker thread and drain the buffer."""
        self._stop_event.set()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)
