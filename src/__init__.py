"""
Maritime Domain Awareness System

A comprehensive deep learning system for maritime surveillance using YOLO object detection,
Kalman filtering, Hungarian assignment, and LSTM-based anomaly detection.
"""

__version__ = "1.0.0"
__author__ = "Maritime Surveillance Team"
__email__ = "contact@maritime-surveillance.com"

from .core.maritime_tracking_system import (
    YOLOShipDetector,
    KalmanFilter, 
    ShipTracker,
    TrajectoryAnomalyDetector,
    MaritimeTrackingSystem
)

__all__ = [
    "YOLOShipDetector",
    "KalmanFilter",
    "ShipTracker", 
    "TrajectoryAnomalyDetector",
    "MaritimeTrackingSystem"
]