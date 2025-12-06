"""
Configuration File for AerialMechX Ball Catcher System

This module contains all configuration constants and parameters used across the system.
Centralized configuration makes it easier to manage and modify system parameters.

Sections:
- Camera Configuration
- UDP Communication
- Ball Detection Parameters  
- Trajectory Prediction Parameters
- ArUco Marker Configuration
- Robot Geometry
- Physics Constants
"""

import pyzed.sl as sl
import cv2 as cv


# ==================== CAMERA CONFIGURATION ====================

# ZED Camera Settings
RESOLUTION = sl.RESOLUTION.HD720
FPS = 30
DEPTH_MODE = sl.DEPTH_MODE.ULTRA
UNIT = sl.UNIT.METER

# Webcam/Default Camera Settings
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_FPS = 30


# ==================== UDP COMMUNICATION ====================

# Robot UDP Configuration
DEFAULT_ROBOT_IP = "192.168.0.155"
DEFAULT_ROBOT_PORT = 5005
UDP_SEND_RATE = 30  # Hz - how often to send coordinates


# ==================== BALL DETECTION PARAMETERS ====================

# HSV Color Ranges for Ball Detection
HSV_TENNIS_BALL = {
    'lower': (29, 86, 6),
    'upper': (64, 255, 255)
}

# Orange ball alternative (commented in original)
# HSV_TENNIS_BALL = {
#     'lower': (10, 100, 100),
#     'upper': (25, 255, 255)
# }

HSV_PAPER_BALL = {
    'lower': (50, 90, 90),
    'upper': (60, 100, 100)
}

ACTIVE_HSV = HSV_TENNIS_BALL

# Detection Parameters
MIN_BALL_AREA = 100
MAX_BALL_AREA = 8000
MIN_CIRCULARITY = 0.4

# Coordinate Offset Corrections
CAMERA_HEIGHT_ABOVE_GROUND = 0.0  # meters (was 0.58)
DEPTH_OFFSET = 0.0  # meters (was 1.9)


# ==================== TRAJECTORY PREDICTION PARAMETERS ====================

# Physics Constants
GRAVITY = 9.81  # m/s^2

# Trajectory Collection
MIN_POINTS_FOR_PREDICTION = 8  # Minimum points needed for reliable prediction
MAX_TRAJECTORY_POINTS = 30  # Maximum points to keep in history
TRAJECTORY_TIMEOUT = 2.0  # seconds - clear trajectory if no detection
LANDING_HEIGHT = 0.0  # meters - ground level (y=0)

# RANSAC Parameters for Outlier Rejection
RANSAC_MIN_SAMPLES = 3
RANSAC_RESIDUAL_THRESHOLD = 0.1  # meters
RANSAC_MAX_TRIALS = 50

# Throw Detection Thresholds
THROW_VELOCITY_THRESHOLD = 1.5  # m/s - minimum velocity to consider a throw START
THROW_CONFIRM_FRAMES = 3  # Need this many consecutive high-velocity frames to confirm throw
MIN_TRAJECTORY_DURATION = 0.15  # seconds - minimum time of ball in flight before prediction

# Prediction Confidence
MIN_PREDICTION_CONFIDENCE = 0.3

# Performance Optimization
PREDICTION_INTERVAL = 3  # Run prediction every N frames

# Logging
PREDICTION_LOG_FILE = "predictions.txt"


# ==================== ARUCO MARKER CONFIGURATION ====================

# ArUco Dictionary Mapping
ARUCO_DICT_MAP = {
    '4x4': cv.aruco.DICT_4X4_250,
    '5x5': cv.aruco.DICT_5X5_250,
    '6x6': cv.aruco.DICT_6X6_250,
    '7x7': cv.aruco.DICT_7X7_250,
    # Also support with underscore
    '4_4': cv.aruco.DICT_4X4_250,
    '5_5': cv.aruco.DICT_5X5_250,
    '6_6': cv.aruco.DICT_6X6_250,
    '7_7': cv.aruco.DICT_7X7_250,
}

# Alternative naming (used in aruco_robot_tracker.py)
ARUCO_DICTS = {
    '4x4': cv.aruco.DICT_4X4_250,
    '5x5': cv.aruco.DICT_5X5_250,
    '6x6': cv.aruco.DICT_6X6_250,
    '7x7': cv.aruco.DICT_7X7_250,
}


# ==================== ROBOT GEOMETRY ====================

# Distances from marker center to robot center (in meters)
FRONT_BACK_OFFSET = 0.0  # 14 cm (currently disabled with 0.0)
LEFT_RIGHT_OFFSET = 0.0  # 16 cm (currently disabled with 0.0)

# Marker tilt angle from vertical (degrees)
MARKER_TILT_ANGLE = 45.0
