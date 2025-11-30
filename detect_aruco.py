"""
ArUco Marker Robot Pose Detection

Detects ArUco markers and estimates robot pose (position + orientation).
Works with ZED stereo camera or standard webcam.

COORDINATE SYSTEM (from camera's perspective):
    +X: Right
    +Y: Up
    +Z: Backward (into camera, away from scene)

This matches the trajectory predictor coordinate system for easy integration.

Requirements:
    pip install opencv-contrib-python numpy

Usage:
    python aruco_pose.py --dict 4x4 --id 0 --size 100
    python aruco_pose.py --dict 5x5 --id 7 --size 150 --webcam
    python aruco_pose.py --dict 6x6 --id 23 --size 200 --robot-ip 192.168.0.51

Arguments:
    --dict      : Dictionary size (4x4, 5x5, 6x6, or 7x7)
    --id        : Marker ID to track
    --size      : Marker size in millimeters

Controls:
    q/ESC - Quit
    c - Calibrate/set current position as origin
    r - Reset origin calibration
    s - Send current pose via UDP (manual)
    u - Toggle UDP sending on/off
"""

import cv2 as cv
import numpy as np
import argparse
import time
import socket
import struct
import math
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

# Try to import ZED SDK
try:
    import pyzed.sl as sl
    ZED_AVAILABLE = True
except ImportError:
    ZED_AVAILABLE = False
    print("Note: ZED SDK not available, will use webcam")


# ==================== CONFIGURATION ====================

# Camera defaults
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

# ArUco dictionary mapping
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
    z: float  # meters
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
        """
        Send 6-DOF pose as packed floats (24 bytes: 6 floats).
        Format: x, y, z (meters), roll, pitch, yaw (radians)
        """
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
            print(f"[UDP ERROR] Failed to send: {e}")
            return False
    
    def send_position_only(self, x: float, y: float, z: float) -> bool:
        """Send just position (12 bytes: 3 floats)"""
        if not self.enabled:
            return False
        
        try:
            packet = struct.pack('3f', x, y, z)
            self.sock.sendto(packet, (self.robot_ip, self.port))
            self.packets_sent += 1
            return True
        except Exception as e:
            print(f"[UDP ERROR] Failed to send: {e}")
            return False
    
    def toggle(self) -> bool:
        """Toggle UDP sending on/off"""
        self.enabled = not self.enabled
        print(f"[UDP] {'ENABLED' if self.enabled else 'DISABLED'}")
        return self.enabled
    
    def close(self):
        self.sock.close()


# ==================== ARUCO DETECTOR ====================

class ArucoPoseEstimator:
    """Detects ArUco markers and estimates 6-DOF pose"""
    
    def __init__(self, intrinsics: CameraIntrinsics, 
                 dict_type: str = '4x4',
                 marker_id: int = 0,
                 marker_size_mm: float = 100.0):
        self.intrinsics = intrinsics
        self.marker_id = marker_id
        self.marker_size = marker_size_mm / 1000.0  # Convert to meters
        
        # Get ArUco dictionary
        dict_key = dict_type.lower().replace('x', 'x').replace('_', 'x')
        if dict_key not in ARUCO_DICT_MAP:
            raise ValueError(f"Unknown dictionary type: {dict_type}. Use 4x4, 5x5, 6x6, or 7x7")
        
        aruco_dict_id = ARUCO_DICT_MAP[dict_key]
        self.aruco_dict = cv.aruco.getPredefinedDictionary(aruco_dict_id)
        self.aruco_params = cv.aruco.DetectorParameters()
        
        # Create detector (OpenCV 4.7+ API)
        try:
            self.detector = cv.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
            self.use_new_api = True
        except AttributeError:
            # Fallback for older OpenCV versions
            self.use_new_api = False
        
        # Camera matrix and distortion for pose estimation
        self.camera_matrix = intrinsics.camera_matrix
        self.dist_coeffs = intrinsics.dist_coeffs
        
        # Origin calibration
        self.origin_pose: Optional[Pose] = None
        self.origin_rvec: Optional[np.ndarray] = None
        self.origin_tvec: Optional[np.ndarray] = None
        
        # Detection statistics
        self.detections_count = 0
        self.last_pose: Optional[Pose] = None
        
        print(f"✓ ArUco detector initialized")
        print(f"  Dictionary: {dict_type.upper()} (DICT_{dict_type.upper().replace('X', 'x')}_250)")
        print(f"  Target ID: {marker_id}")
        print(f"  Marker size: {marker_size_mm:.1f} mm ({self.marker_size:.4f} m)")
    
    def detect(self, frame: np.ndarray) -> Tuple[List, List, List]:
        """
        Detect all ArUco markers in frame.
        
        Returns:
            corners: List of marker corners
            ids: List of marker IDs
            rejected: List of rejected candidates
        """
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        
        if self.use_new_api:
            corners, ids, rejected = self.detector.detectMarkers(gray)
        else:
            corners, ids, rejected = cv.aruco.detectMarkers(
                gray, self.aruco_dict, parameters=self.aruco_params
            )
        
        return corners, ids, rejected
    
    def estimate_pose(self, corners: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate pose from marker corners.
        
        Returns:
            rvec: Rotation vector
            tvec: Translation vector
        """
        # Define marker corners in marker coordinate system
        # Marker is centered at origin, lying on XY plane
        half_size = self.marker_size / 2
        obj_points = np.array([
            [-half_size,  half_size, 0],
            [ half_size,  half_size, 0],
            [ half_size, -half_size, 0],
            [-half_size, -half_size, 0]
        ], dtype=np.float32)
        
        # Solve PnP
        corners_2d = corners.reshape(-1, 2).astype(np.float32)
        
        success, rvec, tvec = cv.solvePnP(
            obj_points, corners_2d,
            self.camera_matrix, self.dist_coeffs,
            flags=cv.SOLVEPNP_IPPE_SQUARE
        )
        
        if success:
            return rvec, tvec
        return None, None
    
    def get_pose_from_detection(self, corners: np.ndarray) -> Optional[Pose]:
        """
        Extract 6-DOF pose from detected marker corners.
        
        Transforms from OpenCV convention (+X right, +Y down, +Z forward)
        to user convention (+X right, +Y up, +Z backward/into camera).
        
        Returns:
            Pose object with position and orientation in user coordinate system
        """
        rvec, tvec = self.estimate_pose(corners)
        
        if rvec is None or tvec is None:
            return None
        
        # Convert rotation vector to rotation matrix (OpenCV frame)
        R_cv, _ = cv.Rodrigues(rvec)
        
        # Translation in OpenCV frame
        t_cv = tvec.flatten()
        
        # ============================================================
        # COORDINATE TRANSFORMATION
        # OpenCV: +X right, +Y down, +Z forward (into scene)
        # User:   +X right, +Y up,   +Z backward (into camera)
        # 
        # Transform: X' = X, Y' = -Y, Z' = -Z
        # Transformation matrix T = diag(1, -1, -1)
        # ============================================================
        
        # Transform position
        t_user = np.array([t_cv[0], -t_cv[1], -t_cv[2]])
        
        # Transform rotation: R' = T @ R @ T  (T is its own inverse)
        T = np.diag([1.0, -1.0, -1.0])
        R_user = T @ R_cv @ T
        
        # Extract Euler angles from transformed rotation matrix
        roll, pitch, yaw = self._rotation_matrix_to_euler(R_user)
        
        pose = Pose(
            x=t_user[0],
            y=t_user[1],
            z=t_user[2],
            roll=roll,
            pitch=pitch,
            yaw=yaw
        )
        
        # Apply origin calibration if set
        if self.origin_pose is not None:
            pose = self._apply_origin_calibration(pose, rvec, tvec)
        
        self.last_pose = pose
        self.detections_count += 1
        
        return pose
    
    def _rotation_matrix_to_euler(self, R: np.ndarray) -> Tuple[float, float, float]:
        """
        Convert rotation matrix to Euler angles (roll, pitch, yaw).
        Uses ZYX convention (yaw-pitch-roll).
        """
        sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        
        singular = sy < 1e-6
        
        if not singular:
            roll = math.atan2(R[2, 1], R[2, 2])
            pitch = math.atan2(-R[2, 0], sy)
            yaw = math.atan2(R[1, 0], R[0, 0])
        else:
            roll = math.atan2(-R[1, 2], R[1, 1])
            pitch = math.atan2(-R[2, 0], sy)
            yaw = 0
        
        return roll, pitch, yaw
    
    def set_origin(self, corners: np.ndarray) -> bool:
        """Set current detection as origin (0, 0, 0)"""
        rvec, tvec = self.estimate_pose(corners)
        
        if rvec is None or tvec is None:
            print("[CALIBRATION] Failed - could not estimate pose")
            return False
        
        self.origin_rvec = rvec.copy()
        self.origin_tvec = tvec.copy()
        
        # Get transformed pose for display
        R_cv, _ = cv.Rodrigues(rvec)
        t_cv = tvec.flatten()
        
        # Transform to user coordinates
        T = np.diag([1.0, -1.0, -1.0])
        t_user = np.array([t_cv[0], -t_cv[1], -t_cv[2]])
        R_user = T @ R_cv @ T
        
        roll, pitch, yaw = self._rotation_matrix_to_euler(R_user)
        
        self.origin_pose = Pose(
            x=t_user[0], y=t_user[1], z=t_user[2],
            roll=roll, pitch=pitch, yaw=yaw
        )
        
        print(f"[CALIBRATION] Origin set!")
        print(f"  Position: ({t_user[0]:.3f}, {t_user[1]:.3f}, {t_user[2]:.3f}) m")
        print(f"  Rotation: ({math.degrees(roll):.1f}°, {math.degrees(pitch):.1f}°, {math.degrees(yaw):.1f}°)")
        
        return True
    
    def reset_origin(self):
        """Reset origin calibration"""
        self.origin_pose = None
        self.origin_rvec = None
        self.origin_tvec = None
        print("[CALIBRATION] Origin reset - using camera frame")
    
    def _apply_origin_calibration(self, pose: Pose, rvec: np.ndarray, tvec: np.ndarray) -> Pose:
        """Transform pose relative to calibrated origin (in user coordinate system)"""
        # Compute relative translation in OpenCV frame
        rel_tvec_cv = tvec.flatten() - self.origin_tvec.flatten()
        
        # Compute relative rotation in OpenCV frame
        R_current_cv, _ = cv.Rodrigues(rvec)
        R_origin_cv, _ = cv.Rodrigues(self.origin_rvec)
        R_relative_cv = R_current_cv @ R_origin_cv.T
        
        # Transform translation to origin frame (still OpenCV)
        rel_pos_cv = R_origin_cv.T @ rel_tvec_cv
        
        # Now transform to user coordinate system
        # X' = X, Y' = -Y, Z' = -Z
        T = np.diag([1.0, -1.0, -1.0])
        rel_pos_user = np.array([rel_pos_cv[0], -rel_pos_cv[1], -rel_pos_cv[2]])
        R_relative_user = T @ R_relative_cv @ T
        
        # Get relative Euler angles in user frame
        rel_roll, rel_pitch, rel_yaw = self._rotation_matrix_to_euler(R_relative_user)
        
        return Pose(
            x=rel_pos_user[0],
            y=rel_pos_user[1],
            z=rel_pos_user[2],
            roll=rel_roll,
            pitch=rel_pitch,
            yaw=rel_yaw
        )
    
    def find_target_marker(self, corners: List, ids: np.ndarray) -> Optional[np.ndarray]:
        """Find the target marker by ID in detection results"""
        if ids is None or len(ids) == 0:
            return None
        
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id == self.marker_id:
                return corners[i]
        
        return None


# ==================== VISUALIZATION ====================

def draw_marker_detection(frame: np.ndarray, corners: List, ids: np.ndarray,
                          target_id: int, pose: Optional[Pose],
                          intrinsics: CameraIntrinsics,
                          estimator: ArucoPoseEstimator):
    """Draw all detected markers with pose axes for target"""
    
    if ids is None or len(ids) == 0:
        return
    
    # Draw all detected markers
    cv.aruco.drawDetectedMarkers(frame, corners, ids)
    
    # Find and highlight target marker
    for i, marker_id in enumerate(ids.flatten()):
        marker_corners = corners[i][0]
        center = np.mean(marker_corners, axis=0).astype(int)
        
        if marker_id == target_id:
            # Draw thicker outline for target
            pts = marker_corners.astype(np.int32).reshape((-1, 1, 2))
            cv.polylines(frame, [pts], True, (0, 255, 0), 3)
            
            # Draw pose axes if available
            if pose is not None:
                rvec, tvec = estimator.estimate_pose(corners[i])
                if rvec is not None:
                    # Draw coordinate axes
                    cv.drawFrameAxes(frame, intrinsics.camera_matrix, 
                                    intrinsics.dist_coeffs, rvec, tvec, 
                                    estimator.marker_size * 0.5)
            
            # Label
            cv.putText(frame, f"TARGET ID:{marker_id}", 
                      (center[0] - 40, center[1] - 20),
                      cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            # Other markers
            cv.putText(frame, f"ID:{marker_id}", 
                      (center[0] - 20, center[1] - 10),
                      cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)


def draw_overlay(frame: np.ndarray, pose: Optional[Pose], 
                 marker_detected: bool, estimator: ArucoPoseEstimator,
                 dict_type: str, marker_size_mm: float,
                 udp_sender: Optional[UDPSender] = None):
    """Draw status overlay on frame"""
    h, w = frame.shape[:2]
    y = 30
    
    # Header
    cv.putText(frame, "ArUco Robot Pose Estimator", (10, y),
              cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y += 28
    
    # Marker info
    cv.putText(frame, f"Dict: {dict_type.upper()} | ID: {estimator.marker_id} | Size: {marker_size_mm:.0f}mm", 
              (10, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    y += 25
    
    # UDP status
    if udp_sender:
        udp_color = (0, 255, 0) if udp_sender.enabled else (0, 0, 255)
        status = "ON" if udp_sender.enabled else "OFF"
        cv.putText(frame, f"UDP: {udp_sender.robot_ip}:{udp_sender.port} [{status}]", 
                  (10, y), cv.FONT_HERSHEY_SIMPLEX, 0.45, udp_color, 1)
        y += 20
        cv.putText(frame, f"Packets: {udp_sender.packets_sent}", (10, y),
                  cv.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        y += 22
    
    # Detection status
    status_text = "DETECTED" if marker_detected else "SEARCHING"
    status_color = (0, 255, 0) if marker_detected else (0, 165, 255)
    cv.putText(frame, f"Marker: {status_text}", (10, y),
              cv.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
    y += 25
    
    # Calibration status
    if estimator.origin_pose is not None:
        cv.putText(frame, "Origin: CALIBRATED (relative)", (10, y),
                  cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    else:
        cv.putText(frame, "Origin: Camera frame (absolute)", (10, y),
                  cv.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    y += 28
    
    # Pose information
    if pose is not None:
        pose_deg = pose.to_degrees()
        
        cv.putText(frame, "POSITION (meters):", (10, y),
                  cv.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y += 22
        
        # X - right (from camera view)
        cv.putText(frame, f"  X (right):    {pose.x:+.4f}", (10, y),
                  cv.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)
        y += 20
        
        # Y - up (from camera view) 
        cv.putText(frame, f"  Y (up):       {pose.y:+.4f}", (10, y),
                  cv.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
        y += 20
        
        # Z - backward (into camera)
        cv.putText(frame, f"  Z (backward): {pose.z:+.4f}", (10, y),
                  cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 1)
        y += 25
        
        cv.putText(frame, "ORIENTATION (degrees):", (10, y),
                  cv.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y += 22
        cv.putText(frame, f"  Roll:  {pose_deg.roll:+.2f}", (10, y),
                  cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 150, 150), 1)
        y += 20
        cv.putText(frame, f"  Pitch: {pose_deg.pitch:+.2f}", (10, y),
                  cv.FONT_HERSHEY_SIMPLEX, 0.5, (150, 255, 150), 1)
        y += 20
        cv.putText(frame, f"  Yaw:   {pose_deg.yaw:+.2f}", (10, y),
                  cv.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 255), 1)
        y += 25
        
        # Distance from camera
        distance = math.sqrt(pose.x**2 + pose.y**2 + pose.z**2)
        cv.putText(frame, f"Distance: {distance:.4f} m ({distance*1000:.1f} mm)", 
                  (10, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    # Detection count
    cv.putText(frame, f"Detections: {estimator.detections_count}", 
              (w - 150, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    
    # Instructions at bottom
    cv.putText(frame, "[C] Calibrate  [R] Reset  [S] Send  [U] Toggle UDP  [Q] Quit",
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
        print(f"✗ Failed to open ZED camera: {err}")
        return None, None
    
    # Get intrinsics
    cam_info = zed.get_camera_information()
    
    try:
        calib = cam_info.camera_configuration.calibration_parameters.left_cam
    except AttributeError:
        try:
            calib = cam_info.calibration_parameters.left_cam
        except AttributeError:
            print("✗ Could not get ZED calibration")
            zed.close()
            return None, None
    
    try:
        res_width = cam_info.camera_configuration.camera_resolution.width
        res_height = cam_info.camera_configuration.camera_resolution.height
    except AttributeError:
        res_width, res_height = 1280, 720
    
    # Get distortion coefficients if available
    try:
        dist = np.array(calib.disto, dtype=np.float32)
    except:
        dist = np.zeros(5, dtype=np.float32)
    
    intrinsics = CameraIntrinsics(
        fx=calib.fx, fy=calib.fy,
        cx=calib.cx, cy=calib.cy,
        width=res_width, height=res_height,
        dist_coeffs=dist
    )
    
    print(f"✓ ZED camera opened")
    print(f"  Resolution: {intrinsics.width}x{intrinsics.height}")
    print(f"  fx={intrinsics.fx:.2f}, fy={intrinsics.fy:.2f}")
    
    return zed, intrinsics


def init_webcam(camera_index: int = 0, 
                fx: float = None, fy: float = None,
                cx: float = None, cy: float = None) -> Tuple[Optional[cv.VideoCapture], Optional[CameraIntrinsics]]:
    """Initialize standard webcam"""
    cap = cv.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"✗ Failed to open webcam {camera_index}")
        return None, None
    
    # Set resolution
    cap.set(cv.CAP_PROP_FRAME_WIDTH, DEFAULT_WIDTH)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, DEFAULT_HEIGHT)
    cap.set(cv.CAP_PROP_FPS, DEFAULT_FPS)
    
    # Get actual resolution
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    
    # Use provided intrinsics or estimate
    if fx is None:
        fx = width * 0.8  # Approximate focal length
    if fy is None:
        fy = fx
    if cx is None:
        cx = width / 2
    if cy is None:
        cy = height / 2
    
    intrinsics = CameraIntrinsics(
        fx=fx, fy=fy, cx=cx, cy=cy,
        width=width, height=height,
        dist_coeffs=np.zeros(5, dtype=np.float32)
    )
    
    print(f"✓ Webcam {camera_index} opened")
    print(f"  Resolution: {width}x{height}")
    print(f"  fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
    print("  NOTE: For accurate pose, calibrate camera and provide --fx, --fy, --cx, --cy")
    
    return cap, intrinsics


# ==================== MARKER GENERATOR ====================

def generate_marker(dict_type: str, marker_id: int, size_pixels: int = 200,
                    output_file: str = None) -> np.ndarray:
    """
    Generate an ArUco marker image.
    
    Args:
        dict_type: Dictionary type (4x4, 5x5, 6x6, 7x7)
        marker_id: Marker ID
        size_pixels: Output size in pixels
        output_file: Optional file path to save the marker
    
    Returns:
        Marker image as numpy array
    """
    dict_key = dict_type.lower().replace('_', 'x')
    if dict_key not in ARUCO_DICT_MAP:
        raise ValueError(f"Unknown dictionary: {dict_type}")
    
    aruco_dict = cv.aruco.getPredefinedDictionary(ARUCO_DICT_MAP[dict_key])
    
    # Generate marker
    marker_img = cv.aruco.generateImageMarker(aruco_dict, marker_id, size_pixels)
    
    # Add white border
    border = size_pixels // 10
    marker_with_border = cv.copyMakeBorder(
        marker_img, border, border, border, border,
        cv.BORDER_CONSTANT, value=255
    )
    
    if output_file:
        cv.imwrite(output_file, marker_with_border)
        print(f"✓ Marker saved to: {output_file}")
    
    return marker_with_border


# ==================== MAIN ====================

def main():
    parser = argparse.ArgumentParser(
        description='ArUco Marker Robot Pose Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python aruco_pose.py --dict 4x4 --id 0 --size 100
  python aruco_pose.py --dict 5x5 --id 7 --size 150 --webcam
  python aruco_pose.py --dict 6x6 --id 23 --size 200 --robot-ip 192.168.0.51
  
  # Generate a marker image:
  python aruco_pose.py --dict 4x4 --id 0 --size 100 --generate marker.png
        """
    )
    
    # Required arguments
    parser.add_argument('--dict', type=str, required=True,
                       choices=['4x4', '5x5', '6x6', '7x7', '4_4', '5_5', '6_6', '7_7'],
                       help='ArUco dictionary size (e.g., 4x4, 5x5, 6x6, 7x7)')
    parser.add_argument('--id', type=int, required=True,
                       help='Marker ID to track (0-249)')
    parser.add_argument('--size', type=float, required=True,
                       help='Marker size in millimeters')
    
    # Camera options
    parser.add_argument('--webcam', action='store_true',
                       help='Use webcam instead of ZED camera')
    parser.add_argument('--camera', type=int, default=0,
                       help='Webcam index (default: 0)')
    
    # Camera calibration (for webcam)
    parser.add_argument('--fx', type=float, default=None,
                       help='Camera focal length fx (pixels)')
    parser.add_argument('--fy', type=float, default=None,
                       help='Camera focal length fy (pixels)')
    parser.add_argument('--cx', type=float, default=None,
                       help='Camera principal point cx (pixels)')
    parser.add_argument('--cy', type=float, default=None,
                       help='Camera principal point cy (pixels)')
    
    # UDP options
    parser.add_argument('--robot-ip', type=str, default=DEFAULT_ROBOT_IP,
                       help=f'Robot IP for UDP (default: {DEFAULT_ROBOT_IP})')
    parser.add_argument('--port', type=int, default=DEFAULT_ROBOT_PORT,
                       help=f'UDP port (default: {DEFAULT_ROBOT_PORT})')
    parser.add_argument('--no-udp', action='store_true',
                       help='Disable UDP sending')
    
    # Marker generation
    parser.add_argument('--generate', type=str, default=None,
                       help='Generate marker image and save to file (e.g., marker.png)')
    
    args = parser.parse_args()
    
    # Normalize dictionary type
    dict_type = args.dict.replace('_', 'x').lower()
    
    # Generate marker if requested
    if args.generate:
        print(f"\nGenerating ArUco marker...")
        print(f"  Dictionary: {dict_type.upper()}")
        print(f"  ID: {args.id}")
        generate_marker(dict_type, args.id, size_pixels=400, output_file=args.generate)
        print(f"\nPrint the marker at {args.size:.0f}mm x {args.size:.0f}mm size.")
        return
    
    print("\n" + "=" * 60)
    print("  ARUCO MARKER ROBOT POSE ESTIMATOR")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Dictionary: {dict_type.upper()}")
    print(f"  Marker ID: {args.id}")
    print(f"  Marker size: {args.size:.1f} mm")
    
    # Initialize UDP sender
    udp_sender = None
    if not args.no_udp:
        udp_sender = UDPSender(args.robot_ip, args.port)
        print(f"\n✓ UDP sender: {args.robot_ip}:{args.port}")
    else:
        print("\n✗ UDP disabled")
    
    # Initialize camera
    print("\nInitializing camera...")
    
    use_zed = not args.webcam and ZED_AVAILABLE
    zed = None
    cap = None
    
    if use_zed:
        zed, intrinsics = init_zed_camera()
        if zed is None:
            print("Falling back to webcam...")
            use_zed = False
    
    if not use_zed:
        cap, intrinsics = init_webcam(
            args.camera, 
            fx=args.fx, fy=args.fy,
            cx=args.cx, cy=args.cy
        )
        if cap is None:
            print("✗ No camera available")
            if udp_sender:
                udp_sender.close()
            return
    
    # Initialize ArUco detector
    print("\nInitializing ArUco detector...")
    try:
        estimator = ArucoPoseEstimator(
            intrinsics,
            dict_type=dict_type,
            marker_id=args.id,
            marker_size_mm=args.size
        )
    except Exception as e:
        print(f"✗ Failed to initialize detector: {e}")
        if zed:
            zed.close()
        if cap:
            cap.release()
        if udp_sender:
            udp_sender.close()
        return
    
    print("=" * 60)
    print(f"\nLooking for ArUco marker ID {args.id}...")
    print("Press [C] to set current position as origin\n")
    
    # ZED-specific setup
    if use_zed:
        runtime = sl.RuntimeParameters()
        image_left = sl.Mat()
    
    last_target_corners = None
    
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
            
            # Detect markers
            corners, ids, rejected = estimator.detect(frame)
            
            # Find target marker
            target_corners = estimator.find_target_marker(corners, ids)
            pose = None
            
            if target_corners is not None:
                pose = estimator.get_pose_from_detection(target_corners)
                last_target_corners = target_corners
                
                # Send pose via UDP
                if udp_sender and pose:
                    udp_sender.send_pose(pose)
            
            # Draw detections
            if ids is not None and len(ids) > 0:
                draw_marker_detection(frame, corners, ids, args.id, pose, 
                                     intrinsics, estimator)
            
            # Draw overlay
            draw_overlay(frame, pose, target_corners is not None, 
                        estimator, dict_type, args.size, udp_sender)
            
            # Display
            cv.imshow("ArUco Pose Estimator", frame)
            
            # Handle keyboard
            key = cv.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('c'):
                # Calibrate origin
                if last_target_corners is not None:
                    estimator.set_origin(last_target_corners)
                else:
                    print("[CALIBRATION] No marker detected - cannot calibrate")
            elif key == ord('r'):
                # Reset origin
                estimator.reset_origin()
            elif key == ord('s'):
                # Manual send
                if pose and udp_sender:
                    udp_sender.enabled = True  # Ensure enabled
                    if udp_sender.send_pose(pose):
                        print(f"[UDP] Sent: ({pose.x:.3f}, {pose.y:.3f}, {pose.z:.3f})")
            elif key == ord('u'):
                # Toggle UDP
                if udp_sender:
                    udp_sender.toggle()
    
    finally:
        cv.destroyAllWindows()
        if use_zed and zed:
            zed.close()
        if cap:
            cap.release()
        if udp_sender:
            print(f"\n✓ UDP packets sent: {udp_sender.packets_sent}")
            udp_sender.close()
        print(f"✓ Total detections: {estimator.detections_count}")
        print("✓ System stopped")
        print("=" * 60)


if __name__ == "__main__":
    main()
