"""
VideoIngestion Module — Asynchronous / Multithreaded HLS (.m3u8) & Video Stream Reader.

Isolates video stream decoding (FFmpeg / OpenCV) from tactical logic and homography.
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

    Validates stream source syntax and buffers decoded frames in a bounded queue
    to prevent I/O stalls during YOLO/TrackNet GPU inference.
    """

    ALLOWED_SCHEMES = ("http", "https", "rtsp", "file", "")

    def __init__(
        self,
        source: str,
        buffer_size: int = 128,
        target_fps: Optional[float] = None,
    ) -> None:
        self.source = self._validate_source(source)
        self.buffer_size = buffer_size
        self.target_fps = target_fps
        self._queue: queue.Queue[Optional[Tuple[int, float, np.ndarray]]] = queue.Queue(
            maxsize=buffer_size
        )
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None

    @classmethod
    def _validate_source(cls, source: str) -> str:
        """Validate URL or file path syntax."""
        cleaned = source.strip()
        if not cleaned:
            raise ValueError("Video stream source cannot be empty.")
        parsed = urlparse(cleaned)
        if parsed.scheme and parsed.scheme.lower() not in cls.ALLOWED_SCHEMES:
            # Allow Windows drive letters like C:\path
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
        """Producer loop pulling frames from HLS/file capture into the bounded queue."""
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            self._queue.put(None)
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_idx = 0

        try:
            while not self._stop_event.is_set():
                if not self._queue.full():
                    ret, frame = cap.read()
                    if not ret:
                        self._queue.put(None)
                        break
                    timestamp_ms = (frame_idx / fps) * 1000.0
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
