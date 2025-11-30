"""
Multi-Marker ArUco Robot Pose Detection

Detects robot pose using 4 ArUco markers placed on different sides:
    - Left side:  4x4 dictionary
    - Front side: 5x5 dictionary
    - Right side: 6x6 dictionary
    - Back side:  7x7 dictionary

Computes robot CENTER position by transforming from detected marker poses.

COORDINATE SYSTEM (from camera's perspective):
    +X: Right
    +Y: Up
    +Z: Backward (into camera, away from scene)

ROBOT GEOMETRY:
    - Front/Back markers: 14 cm from robot center
    - Left/Right markers: 16 cm from robot center
    - Markers tilted 45° from vertical, facing outward

Requirements:
    pip install opencv-contrib-python numpy

Usage:
    python aruco_robot_tracker.py --size 100
    python aruco_robot_tracker.py --size 100 --robot-ip 192.168.0.51
    python aruco_robot_tracker.py --size 100 --left-id 0 --front-id 0 --right-id 0 --back-id 0

Controls:
    q/ESC - Quit
    c - Calibrate/set current position as origin
    r - Reset origin calibration  
    s - Send current pose via UDP (manual)
    u - Toggle UDP sending on/off
    d - Toggle debug visualization
"""

import cv2 as cv
import numpy as np
import argparse
import time
import socket
import struct
import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
from enum import Enum

# Try to import ZED SDK
try:
    import pyzed.sl as sl
    ZED_AVAILABLE = True
except ImportError:
    ZED_AVAILABLE = False
    print("Note: ZED SDK not available, will use webcam")


# ==================== CONFIGURATION ====================

DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_FPS = 30

# UDP Configuration
DEFAULT_ROBOT_IP = "192.168.0.51"
DEFAULT_ROBOT_PORT = 5006
UDP_SEND_RATE = 30  # Hz

# ==================== COORDINATE SYSTEM ====================
# From camera's perspective:
#   +X: Right
#   +Y: Up  
#   +Z: Backward (into camera, away from scene)
#
# OpenCV/ArUco uses: +X right, +Y down, +Z forward
# We transform: X_out = X, Y_out = -Y, Z_out = -Z

# ==================== ROBOT GEOMETRY ====================
# Distances from marker center to robot center (in meters)
FRONT_BACK_OFFSET = 0.0  # 14 cm
LEFT_RIGHT_OFFSET = 0.0  # 16 cm

# Marker tilt angle from vertical (degrees)
MARKER_TILT_ANGLE = 45.0


# ==================== MARKER CONFIGURATION ====================

class MarkerSide(Enum):
    LEFT = "left"
    FRONT = "front"
    RIGHT = "right"
    BACK = "back"


@dataclass
class MarkerConfig:
    """Configuration for a single marker"""
    side: MarkerSide
    dict_type: int  # OpenCV ArUco dictionary constant
    marker_id: int
    offset_to_center: np.ndarray  # Offset from marker to robot center in marker's local frame
    dict_name: str = ""


# ArUco dictionary mapping
ARUCO_DICTS = {
    '4x4': cv.aruco.DICT_4X4_250,
    '5x5': cv.aruco.DICT_5X5_250,
    '6x6': cv.aruco.DICT_6X6_250,
    '7x7': cv.aruco.DICT_7X7_250,
}


# ==================== DATA CLASSES ====================

@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters"""
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    dist_coeffs: np.ndarray = None
    
    def __post_init__(self):
        if self.dist_coeffs is None:
            self.dist_coeffs = np.zeros(5)
    
    @property
    def camera_matrix(self) -> np.ndarray:
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float32)


@dataclass
class Pose:
    """6-DOF Pose: position (x, y, z) and orientation (roll, pitch, yaw)"""
    x: float  # meters
    y: float  # meters
    z: float  # meters (negative = in front of camera)
    roll: float   # radians
    pitch: float  # radians
    yaw: float    # radians
    
    def to_degrees(self) -> 'Pose':
        """Return pose with angles in degrees"""
        return Pose(
            x=self.x, y=self.y, z=self.z,
            roll=math.degrees(self.roll),
            pitch=math.degrees(self.pitch),
            yaw=math.degrees(self.yaw)
        )
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array [x, y, z, roll, pitch, yaw]"""
        return np.array([self.x, self.y, self.z, self.roll, self.pitch, self.yaw])
    
    @staticmethod
    def from_array(arr: np.ndarray) -> 'Pose':
        return Pose(x=arr[0], y=arr[1], z=arr[2], 
                   roll=arr[3], pitch=arr[4], yaw=arr[5])


@dataclass
class MarkerDetection:
    """Result of detecting a single marker"""
    side: MarkerSide
    marker_id: int
    corners: np.ndarray
    rvec: np.ndarray
    tvec: np.ndarray
    robot_center_pos: np.ndarray  # Robot center position in camera frame
    robot_yaw: float  # Robot yaw angle (rotation around Y axis)


# ==================== UDP SENDER ====================

class UDPSender:
    """UDP sender for transmitting robot pose"""
    
    def __init__(self, robot_ip: str = DEFAULT_ROBOT_IP, port: int = DEFAULT_ROBOT_PORT):
        self.robot_ip = robot_ip
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.enabled = True
        self.packets_sent = 0
        self.last_send_time = 0
        self.min_send_interval = 1.0 / UDP_SEND_RATE
        
    def send_pose(self, pose: Pose) -> bool:
        """Send 6-DOF pose (24 bytes: 6 floats)"""
        if not self.enabled:
            return False
        
        current_time = time.time()
        if (current_time - self.last_send_time) < self.min_send_interval:
            return False
        
        try:
            packet = struct.pack('6f', pose.x, pose.y, pose.z, 
                                pose.roll, pose.pitch, pose.yaw)
            self.sock.sendto(packet, (self.robot_ip, self.port))
            self.packets_sent += 1
            self.last_send_time = current_time
            return True
        except Exception as e:
            print(f"[UDP ERROR] {e}")
            return False
    
    def send_position(self, x: float, y: float, z: float) -> bool:
        """Send just position (12 bytes: 3 floats)"""
        if not self.enabled:
            return False
        
        try:
            packet = struct.pack('3f', x, y, z)
            self.sock.sendto(packet, (self.robot_ip, self.port))
            self.packets_sent += 1
            return True
        except Exception as e:
            print(f"[UDP ERROR] {e}")
            return False
    
    def toggle(self) -> bool:
        self.enabled = not self.enabled
        print(f"[UDP] {'ENABLED' if self.enabled else 'DISABLED'}")
        return self.enabled
    
    def close(self):
        self.sock.close()


# ==================== MULTI-MARKER DETECTOR ====================

class MultiMarkerRobotTracker:
    """
    Tracks robot using multiple ArUco markers on different sides.
    Computes robot center position from detected marker poses.
    """
    
    def __init__(self, intrinsics: CameraIntrinsics, marker_size_mm: float,
                 left_id: int = 0, front_id: int = 0, 
                 right_id: int = 0, back_id: int = 0):
        self.intrinsics = intrinsics
        self.marker_size = marker_size_mm / 1000.0  # Convert to meters
        
        self.camera_matrix = intrinsics.camera_matrix
        self.dist_coeffs = intrinsics.dist_coeffs
        
        # Create detector for each dictionary
        self.detectors: Dict[MarkerSide, Tuple] = {}
        self.marker_configs: Dict[MarkerSide, MarkerConfig] = {}
        
        # Configure markers
        # Offset is from marker center to robot center in marker's local frame
        # Marker Z points outward (away from robot), so robot center is at -Z
        
        self._setup_marker(MarkerSide.LEFT, '4x4', left_id, 
                          np.array([0, 0, -LEFT_RIGHT_OFFSET]))
        self._setup_marker(MarkerSide.FRONT, '5x5', front_id,
                          np.array([0, 0, -FRONT_BACK_OFFSET]))
        self._setup_marker(MarkerSide.RIGHT, '6x6', right_id,
                          np.array([0, 0, -LEFT_RIGHT_OFFSET]))
        self._setup_marker(MarkerSide.BACK, '7x7', back_id,
                          np.array([0, 0, -FRONT_BACK_OFFSET]))
        
        # Yaw offset for each marker side (how much to add to get robot's yaw)
        # When marker faces camera, its yaw is 0. Robot's forward direction differs.
        self.yaw_offsets = {
            MarkerSide.LEFT: -np.pi / 2,   # Left marker: robot faces 90° CCW from marker normal
            MarkerSide.FRONT: 0,            # Front marker: robot faces same as marker normal
            MarkerSide.RIGHT: np.pi / 2,    # Right marker: robot faces 90° CW from marker normal
            MarkerSide.BACK: np.pi,         # Back marker: robot faces opposite to marker normal
        }
        
        # Origin calibration
        self.origin_position: Optional[np.ndarray] = None
        self.origin_yaw: float = 0.0
        
        # Statistics
        self.detections_count = 0
        self.last_pose: Optional[Pose] = None
        self.last_detections: List[MarkerDetection] = []
        
        print(f"✓ Multi-marker tracker initialized")
        print(f"  Marker size: {marker_size_mm:.1f} mm")
        print(f"  Left (4x4):  ID {left_id}")
        print(f"  Front (5x5): ID {front_id}")
        print(f"  Right (6x6): ID {right_id}")
        print(f"  Back (7x7):  ID {back_id}")
    
    def _setup_marker(self, side: MarkerSide, dict_name: str, marker_id: int,
                     offset: np.ndarray):
        """Setup detector and config for one marker"""
        dict_type = ARUCO_DICTS[dict_name]
        aruco_dict = cv.aruco.getPredefinedDictionary(dict_type)
        aruco_params = cv.aruco.DetectorParameters()
        
        try:
            detector = cv.aruco.ArucoDetector(aruco_dict, aruco_params)
        except AttributeError:
            detector = None  # Will use old API
        
        self.detectors[side] = (aruco_dict, aruco_params, detector)
        self.marker_configs[side] = MarkerConfig(
            side=side,
            dict_type=dict_type,
            marker_id=marker_id,
            offset_to_center=offset,
            dict_name=dict_name
        )
    
    def detect_all_markers(self, frame: np.ndarray) -> List[MarkerDetection]:
        """Detect markers from all dictionaries and compute robot center for each"""
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        
        detections = []
        
        for side, (aruco_dict, aruco_params, detector) in self.detectors.items():
            config = self.marker_configs[side]
            
            # Detect markers
            if detector is not None:
                corners, ids, _ = detector.detectMarkers(gray)
            else:
                corners, ids, _ = cv.aruco.detectMarkers(gray, aruco_dict, 
                                                         parameters=aruco_params)
            
            if ids is None or len(ids) == 0:
                continue
            
            # Find target marker
            for i, marker_id in enumerate(ids.flatten()):
                if marker_id == config.marker_id:
                    # Estimate pose
                    rvec, tvec = self._estimate_pose(corners[i])
                    if rvec is not None:
                        # Compute robot center from marker pose
                        robot_center, robot_yaw = self._compute_robot_center(
                            rvec, tvec, config.offset_to_center, side
                        )
                        
                        detection = MarkerDetection(
                            side=side,
                            marker_id=marker_id,
                            corners=corners[i],
                            rvec=rvec,
                            tvec=tvec,
                            robot_center_pos=robot_center,
                            robot_yaw=robot_yaw
                        )
                        detections.append(detection)
                    break  # Only use first matching marker per dictionary
        
        self.last_detections = detections
        return detections
    
    def _estimate_pose(self, corners: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Estimate marker pose using solvePnP"""
        half_size = self.marker_size / 2
        obj_points = np.array([
            [-half_size,  half_size, 0],
            [ half_size,  half_size, 0],
            [ half_size, -half_size, 0],
            [-half_size, -half_size, 0]
        ], dtype=np.float32)
        
        corners_2d = corners.reshape(-1, 2).astype(np.float32)
        
        success, rvec, tvec = cv.solvePnP(
            obj_points, corners_2d,
            self.camera_matrix, self.dist_coeffs,
            flags=cv.SOLVEPNP_IPPE_SQUARE
        )
        
        if success:
            return rvec, tvec
        return None, None
    
    def _compute_robot_center(self, rvec: np.ndarray, tvec: np.ndarray,
                             offset: np.ndarray, side: MarkerSide) -> Tuple[np.ndarray, float]:
        """
        Compute robot center position from marker pose.
        
        Args:
            rvec: Marker rotation vector (OpenCV frame)
            tvec: Marker translation vector (OpenCV frame)
            offset: Offset from marker to robot center in marker's local frame
            side: Which side of robot this marker is on
        
        Returns:
            robot_center: Position of robot center in camera frame (user coordinates)
            robot_yaw: Robot's yaw angle (rotation around Y axis)
        """
        # Get marker rotation matrix
        R_marker_cv, _ = cv.Rodrigues(rvec)
        t_marker_cv = tvec.flatten()
        
        # Transform offset from marker frame to camera frame (OpenCV)
        offset_camera_cv = R_marker_cv @ offset
        
        # Robot center in OpenCV camera frame
        robot_center_cv = t_marker_cv + offset_camera_cv
        
        # Transform to user coordinate system: X'=X, Y'=-Y, Z'=-Z
        robot_center_user = np.array([
            robot_center_cv[0],
            -robot_center_cv[1],
            -robot_center_cv[2]
        ])
        
        # Compute robot yaw (rotation around vertical Y axis)
        # Extract yaw from marker's rotation and add offset based on which side
        T = np.diag([1.0, -1.0, -1.0])
        R_marker_user = T @ R_marker_cv @ T
        
        # Extract yaw (rotation around Y axis)
        marker_yaw = math.atan2(R_marker_user[0, 2], R_marker_user[2, 2])
        
        # Add offset based on which side the marker is on
        robot_yaw = marker_yaw + self.yaw_offsets[side]
        
        # Normalize to [-pi, pi]
        robot_yaw = math.atan2(math.sin(robot_yaw), math.cos(robot_yaw))
        
        return robot_center_user, robot_yaw
    
    def fuse_detections(self, detections: List[MarkerDetection]) -> Optional[Pose]:
        """
        Fuse multiple marker detections into a single robot pose.
        
        Strategy:
        - 1 marker: Use its estimate directly
        - 2+ markers: Average positions, use circular mean for yaw
        """
        if not detections:
            return None
        
        # Collect positions and yaws
        positions = np.array([d.robot_center_pos for d in detections])
        yaws = np.array([d.robot_yaw for d in detections])
        
        # Average position
        avg_position = np.mean(positions, axis=0)
        
        # Circular mean for yaw angle
        avg_yaw = math.atan2(
            np.mean(np.sin(yaws)),
            np.mean(np.cos(yaws))
        )
        
        # Apply origin calibration if set
        if self.origin_position is not None:
            avg_position = avg_position - self.origin_position
            avg_yaw = avg_yaw - self.origin_yaw
            avg_yaw = math.atan2(math.sin(avg_yaw), math.cos(avg_yaw))
        
        pose = Pose(
            x=avg_position[0],
            y=avg_position[1],
            z=avg_position[2],
            roll=0.0,  # Assuming robot is on flat ground
            pitch=0.0,
            yaw=avg_yaw
        )
        
        self.last_pose = pose
        self.detections_count += 1
        
        return pose
    
    def update(self, frame: np.ndarray) -> Tuple[Optional[Pose], List[MarkerDetection]]:
        """
        Main update function: detect markers and compute fused pose.
        
        Returns:
            pose: Fused robot pose (or None if no markers detected)
            detections: List of individual marker detections
        """
        detections = self.detect_all_markers(frame)
        pose = self.fuse_detections(detections)
        return pose, detections
    
    def set_origin(self) -> bool:
        """Set current position as origin"""
        if not self.last_detections:
            print("[CALIBRATION] No markers detected")
            return False
        
        positions = np.array([d.robot_center_pos for d in self.last_detections])
        yaws = np.array([d.robot_yaw for d in self.last_detections])
        
        self.origin_position = np.mean(positions, axis=0)
        self.origin_yaw = math.atan2(np.mean(np.sin(yaws)), np.mean(np.cos(yaws)))
        
        print(f"[CALIBRATION] Origin set!")
        print(f"  Position: ({self.origin_position[0]:.3f}, {self.origin_position[1]:.3f}, {self.origin_position[2]:.3f})")
        print(f"  Yaw: {math.degrees(self.origin_yaw):.1f}°")
        return True
    
    def reset_origin(self):
        """Reset origin calibration"""
        self.origin_position = None
        self.origin_yaw = 0.0
        print("[CALIBRATION] Origin reset")


# ==================== VISUALIZATION ====================

# Colors for each marker side (BGR)
MARKER_COLORS = {
    MarkerSide.LEFT: (255, 0, 0),    # Blue
    MarkerSide.FRONT: (0, 255, 0),   # Green
    MarkerSide.RIGHT: (0, 0, 255),   # Red
    MarkerSide.BACK: (0, 255, 255),  # Yellow
}

SIDE_NAMES = {
    MarkerSide.LEFT: "LEFT (4x4)",
    MarkerSide.FRONT: "FRONT (5x5)",
    MarkerSide.RIGHT: "RIGHT (6x6)",
    MarkerSide.BACK: "BACK (7x7)",
}


def draw_detections(frame: np.ndarray, detections: List[MarkerDetection],
                   tracker: MultiMarkerRobotTracker, show_axes: bool = True):
    """Draw all detected markers with their info"""
    
    for det in detections:
        color = MARKER_COLORS[det.side]
        
        # Draw marker outline
        corners = det.corners.reshape(-1, 2).astype(np.int32)
        cv.polylines(frame, [corners], True, color, 3)
        
        # Draw center point
        center = np.mean(corners, axis=0).astype(int)
        cv.circle(frame, tuple(center), 5, color, -1)
        
        # Draw label
        label = f"{det.side.value.upper()} ID:{det.marker_id}"
        cv.putText(frame, label, (center[0] - 40, center[1] - 15),
                  cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw pose axes
        if show_axes:
            cv.drawFrameAxes(frame, tracker.camera_matrix, tracker.dist_coeffs,
                           det.rvec, det.tvec, tracker.marker_size * 0.5)


def draw_robot_indicator(frame: np.ndarray, pose: Pose, 
                        intrinsics: CameraIntrinsics):
    """Draw a simple indicator showing estimated robot center"""
    if pose is None:
        return
    
    # Project robot center to image
    # Convert from user coords back to OpenCV for projection
    robot_center_cv = np.array([[pose.x, -pose.y, -pose.z]], dtype=np.float32)
    
    # Project to image
    img_points, _ = cv.projectPoints(
        robot_center_cv,
        np.zeros(3), np.zeros(3),  # No rotation/translation (already in camera frame)
        intrinsics.camera_matrix,
        intrinsics.dist_coeffs
    )
    
    if img_points is not None and len(img_points) > 0:
        pt = img_points[0].flatten().astype(int)
        if 0 <= pt[0] < intrinsics.width and 0 <= pt[1] < intrinsics.height:
            # Draw robot center marker
            cv.circle(frame, tuple(pt), 15, (255, 255, 255), 3)
            cv.circle(frame, tuple(pt), 12, (0, 200, 255), -1)
            cv.putText(frame, "CENTER", (pt[0] + 20, pt[1] + 5),
                      cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            
            # Draw yaw direction arrow
            arrow_len = 40
            end_x = int(pt[0] + arrow_len * math.sin(pose.yaw))
            end_y = int(pt[1] - arrow_len * math.cos(pose.yaw))  # Negative because Y is down in image
            cv.arrowedLine(frame, tuple(pt), (end_x, end_y), (0, 200, 255), 3, tipLength=0.3)


def draw_overlay(frame: np.ndarray, pose: Optional[Pose],
                detections: List[MarkerDetection],
                tracker: MultiMarkerRobotTracker,
                udp_sender: Optional[UDPSender] = None,
                show_debug: bool = False):
    """Draw status overlay"""
    h, w = frame.shape[:2]
    y = 30
    
    # Header
    cv.putText(frame, "Multi-Marker Robot Tracker", (10, y),
              cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y += 28
    
    # Marker size
    cv.putText(frame, f"Marker size: {tracker.marker_size * 1000:.0f}mm", 
              (10, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    y += 22
    
    # UDP status
    if udp_sender:
        udp_color = (0, 255, 0) if udp_sender.enabled else (0, 0, 255)
        status = "ON" if udp_sender.enabled else "OFF"
        cv.putText(frame, f"UDP: {udp_sender.robot_ip}:{udp_sender.port} [{status}]", 
                  (10, y), cv.FONT_HERSHEY_SIMPLEX, 0.45, udp_color, 1)
        y += 20
    
    # Detection status
    n_detected = len(detections)
    if n_detected == 0:
        status_text = "NO MARKERS"
        status_color = (0, 0, 255)
    elif n_detected == 1:
        status_text = f"1 MARKER ({detections[0].side.value})"
        status_color = (0, 255, 255)
    else:
        sides = ", ".join([d.side.value for d in detections])
        status_text = f"{n_detected} MARKERS ({sides})"
        status_color = (0, 255, 0)
    
    cv.putText(frame, f"Detection: {status_text}", (10, y),
              cv.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
    y += 25
    
    # Calibration status
    if tracker.origin_position is not None:
        cv.putText(frame, "Origin: CALIBRATED", (10, y),
                  cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    else:
        cv.putText(frame, "Origin: Camera frame", (10, y),
                  cv.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    y += 28
    
    # Robot pose
    if pose is not None:
        cv.putText(frame, "ROBOT CENTER (meters):", (10, y),
                  cv.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y += 22
        
        cv.putText(frame, f"  X (right):    {pose.x:+.4f}", (10, y),
                  cv.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)
        y += 20
        cv.putText(frame, f"  Y (up):       {pose.y:+.4f}", (10, y),
                  cv.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
        y += 20
        cv.putText(frame, f"  Z (backward): {pose.z:+.4f}", (10, y),
                  cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 1)
        y += 25
        
        yaw_deg = math.degrees(pose.yaw)
        cv.putText(frame, f"  Yaw: {yaw_deg:+.1f} deg", (10, y),
                  cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 100), 1)
        y += 25
        
        # Distance from camera
        distance = math.sqrt(pose.x**2 + pose.y**2 + pose.z**2)
        cv.putText(frame, f"Distance: {distance:.3f} m", (10, y),
                  cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        y += 25
    
    # Debug: show individual marker estimates
    if show_debug and len(detections) > 1:
        cv.putText(frame, "Individual estimates:", (10, y),
                  cv.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
        y += 18
        for det in detections:
            p = det.robot_center_pos
            cv.putText(frame, f"  {det.side.value}: ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})",
                      (10, y), cv.FONT_HERSHEY_SIMPLEX, 0.4, MARKER_COLORS[det.side], 1)
            y += 16
    
    # Detection count
    cv.putText(frame, f"Updates: {tracker.detections_count}", 
              (w - 150, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    
    # Instructions
    cv.putText(frame, "[C]Calibrate [R]Reset [S]Send [U]UDP [D]Debug [Q]Quit",
              (10, h - 10), cv.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)


# ==================== CAMERA INITIALIZATION ====================

def init_zed_camera() -> Tuple[Optional['sl.Camera'], Optional[CameraIntrinsics]]:
    """Initialize ZED stereo camera"""
    if not ZED_AVAILABLE:
        return None, None
    
    zed = sl.Camera()
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD720
    init.camera_fps = DEFAULT_FPS
    init.depth_mode = sl.DEPTH_MODE.ULTRA
    init.coordinate_units = sl.UNIT.METER
    
    err = zed.open(init)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"✗ Failed to open ZED: {err}")
        return None, None
    
    cam_info = zed.get_camera_information()
    
    try:
        calib = cam_info.camera_configuration.calibration_parameters.left_cam
    except AttributeError:
        try:
            calib = cam_info.calibration_parameters.left_cam
        except:
            zed.close()
            return None, None
    
    try:
        res_width = cam_info.camera_configuration.camera_resolution.width
        res_height = cam_info.camera_configuration.camera_resolution.height
    except:
        res_width, res_height = 1280, 720
    
    try:
        dist = np.array(calib.disto, dtype=np.float32)
    except:
        dist = np.zeros(5, dtype=np.float32)
    
    intrinsics = CameraIntrinsics(
        fx=calib.fx, fy=calib.fy, cx=calib.cx, cy=calib.cy,
        width=res_width, height=res_height, dist_coeffs=dist
    )
    
    print(f"✓ ZED camera: {intrinsics.width}x{intrinsics.height}")
    return zed, intrinsics


def init_webcam(camera_index: int = 0, fx: float = None) -> Tuple[Optional[cv.VideoCapture], Optional[CameraIntrinsics]]:
    """Initialize webcam"""
    cap = cv.VideoCapture(camera_index)
    if not cap.isOpened():
        return None, None
    
    cap.set(cv.CAP_PROP_FRAME_WIDTH, DEFAULT_WIDTH)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, DEFAULT_HEIGHT)
    cap.set(cv.CAP_PROP_FPS, DEFAULT_FPS)
    
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    
    if fx is None:
        fx = width * 0.8
    
    intrinsics = CameraIntrinsics(
        fx=fx, fy=fx, cx=width/2, cy=height/2,
        width=width, height=height,
        dist_coeffs=np.zeros(5, dtype=np.float32)
    )
    
    print(f"✓ Webcam: {width}x{height}")
    return cap, intrinsics


# ==================== MAIN ====================

def main():
    parser = argparse.ArgumentParser(
        description='Multi-Marker ArUco Robot Tracker',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Marker Configuration:
  Left side:  4x4 dictionary
  Front side: 5x5 dictionary  
  Right side: 6x6 dictionary
  Back side:  7x7 dictionary

Examples:
  python aruco_robot_tracker.py --size 100
  python aruco_robot_tracker.py --size 100 --left-id 0 --front-id 1 --right-id 2 --back-id 3
  python aruco_robot_tracker.py --size 100 --robot-ip 192.168.0.51
        """
    )
    
    # Marker size (required)
    parser.add_argument('--size', type=float, required=True,
                       help='Marker size in millimeters')
    
    # Marker IDs
    parser.add_argument('--left-id', type=int, default=0,
                       help='Left marker ID (4x4 dict, default: 0)')
    parser.add_argument('--front-id', type=int, default=0,
                       help='Front marker ID (5x5 dict, default: 0)')
    parser.add_argument('--right-id', type=int, default=0,
                       help='Right marker ID (6x6 dict, default: 0)')
    parser.add_argument('--back-id', type=int, default=0,
                       help='Back marker ID (7x7 dict, default: 0)')
    
    # Camera options
    parser.add_argument('--webcam', action='store_true',
                       help='Use webcam instead of ZED')
    parser.add_argument('--camera', type=int, default=0,
                       help='Webcam index')
    parser.add_argument('--fx', type=float, default=None,
                       help='Camera focal length (for webcam)')
    
    # UDP options
    parser.add_argument('--robot-ip', type=str, default=DEFAULT_ROBOT_IP,
                       help=f'Robot IP (default: {DEFAULT_ROBOT_IP})')
    parser.add_argument('--port', type=int, default=DEFAULT_ROBOT_PORT,
                       help=f'UDP port (default: {DEFAULT_ROBOT_PORT})')
    parser.add_argument('--no-udp', action='store_true',
                       help='Disable UDP')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("  MULTI-MARKER ARUCO ROBOT TRACKER")
    print("=" * 60)
    
    # Initialize UDP
    udp_sender = None
    if not args.no_udp:
        udp_sender = UDPSender(args.robot_ip, args.port)
        print(f"\n✓ UDP: {args.robot_ip}:{args.port}")
    
    # Initialize camera
    print("\nInitializing camera...")
    use_zed = not args.webcam and ZED_AVAILABLE
    zed, cap = None, None
    
    if use_zed:
        zed, intrinsics = init_zed_camera()
        if zed is None:
            use_zed = False
    
    if not use_zed:
        cap, intrinsics = init_webcam(args.camera, args.fx)
        if cap is None:
            print("✗ No camera available")
            if udp_sender:
                udp_sender.close()
            return
    
    # Initialize tracker
    print("\nInitializing tracker...")
    tracker = MultiMarkerRobotTracker(
        intrinsics, args.size,
        left_id=args.left_id, front_id=args.front_id,
        right_id=args.right_id, back_id=args.back_id
    )
    
    print("=" * 60)
    print("\nLooking for markers...")
    print("Press [C] to set current position as origin\n")
    
    if use_zed:
        runtime = sl.RuntimeParameters()
        image_left = sl.Mat()
    
    show_debug = False
    
    try:
        while True:
            # Grab frame
            if use_zed:
                if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                    continue
                zed.retrieve_image(image_left, sl.VIEW.LEFT)
                frame = image_left.get_data()[:, :, :3].copy()
            else:
                ret, frame = cap.read()
                if not ret:
                    continue
            
            # Update tracker
            pose, detections = tracker.update(frame)
            
            # Send pose via UDP
            if pose and udp_sender:
                udp_sender.send_pose(pose)
            
            # Draw visualizations
            draw_detections(frame, detections, tracker)
            draw_robot_indicator(frame, pose, intrinsics)
            draw_overlay(frame, pose, detections, tracker, udp_sender, show_debug)
            
            # Display
            cv.imshow("Robot Tracker", frame)
            
            # Handle keyboard
            key = cv.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('c'):
                tracker.set_origin()
            elif key == ord('r'):
                tracker.reset_origin()
            elif key == ord('s'):
                if pose and udp_sender:
                    udp_sender.enabled = True
                    udp_sender.send_pose(pose)
                    print(f"[UDP] Sent: ({pose.x:.3f}, {pose.y:.3f}, {pose.z:.3f})")
            elif key == ord('u'):
                if udp_sender:
                    udp_sender.toggle()
            elif key == ord('d'):
                show_debug = not show_debug
                print(f"[DEBUG] {'ON' if show_debug else 'OFF'}")
    
    finally:
        cv.destroyAllWindows()
        if use_zed and zed:
            zed.close()
        if cap:
            cap.release()
        if udp_sender:
            print(f"\n✓ UDP packets: {udp_sender.packets_sent}")
            udp_sender.close()
        print(f"✓ Total updates: {tracker.detections_count}")
        print("=" * 60)


if __name__ == "__main__":
    main()
