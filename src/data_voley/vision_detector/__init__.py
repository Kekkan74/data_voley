"""
Vision Detector Port & Human-in-the-Loop Training Dataset Builder.

Decouples future YOLO/TrackNet computer-vision adapters from the core domain
while capturing user-validated K1/K2 annotations into a structured JSONL dataset.
"""

from data_voley.vision_detector.dataset_exporter import K1DatasetExporter
from data_voley.vision_detector.ports import FrameProposal, VisionDetectorPort

__all__ = [
    "FrameProposal",
    "K1DatasetExporter",
    "VisionDetectorPort",
]
