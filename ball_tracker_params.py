"""
Ball Tracking with Color Detection, Stereo Triangulation, and UDP Output

This script detects and tracks a colored ball using stereo vision from the ZED camera,
and sends real-time 3D coordinates to a robot via UDP.

Usage:
    python ball_tracker_params.py                           # Use SDK parameters
    python ball_tracker_params.py --use-files               # Use file-based parameters
    python ball_tracker_params.py --robot-ip 192.168.0.51   # Set robot IP
    python ball_tracker_params.py --no-udp                  # Disable UDP sending
    
Controls:
    q/ESC - Quit
    r - Reset tracking
    t - Switch to tennis ball HSV
    p - Switch to paper ball HSV
    u - Toggle UDP sending on/off
"""

import pyzed.sl as sl
import cv2 as cv
import numpy as np
import argparse
import os
import time
import socket
import struct
from dataclasses import dataclass
from typing import Optional, Tuple, Dict


# ==================== CONFIGURATION ====================

RESOLUTION = sl.RESOLUTION.HD720
FPS = 30
DEPTH_MODE = sl.DEPTH_MODE.ULTRA
UNIT = sl.UNIT.METER

# HSV color ranges
HSV_TENNIS_BALL = {
    'lower': (29, 86, 6),
    'upper': (64, 255, 255)
}
# orange
# HSV_TENNIS_BALL = {
#     'lower': (10, 100, 100),
#     'upper': (25, 255, 255)
# }
HSV_PAPER_BALL = {
    'lower': (50, 90, 90),
    'upper': (60, 100, 100)
}
ACTIVE_HSV = HSV_TENNIS_BALL

# Detection parameters
MIN_BALL_AREA = 100
MAX_BALL_AREA = 8000
MIN_CIRCULARITY = 0.4

# Coordinate offset corrections
# CAMERA_HEIGHT_ABOVE_GROUND = 0.58
# DEPTH_OFFSET = 1.9
CAMERA_HEIGHT_ABOVE_GROUND = 0.0
DEPTH_OFFSET = 0.0

# Import UDP sender
from sender import UDPSender, DEFAULT_ROBOT_IP, DEFAULT_ROBOT_PORT

UDP_SEND_RATE = 30  # Hz - how often to send coordinates


# ==================== DATA CLASSES ====================

# UDPSender now imported from sender.py module

@dataclass
class CameraParameters:
    """Camera intrinsic and extrinsic parameters"""
    intrinsic_matrix: np.ndarray
    distortion: np.ndarray
    rotation: np.ndarray
    translation: np.ndarray
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass
class CameraIntrinsics:
    """Camera intrinsics for depth projection"""
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


# ==================== PARAMETER LOADING ====================

def load_camera_parameters_from_files(camera_params_dir: str = 'camera_parameters'):
    """Load camera parameters from .dat files"""
    
    def read_intrinsics_file(filepath):
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        intrinsic_matrix = []
        dist_coeffs = []
        reading_intrinsic = False
        reading_distortion = False
        
        for line in lines:
            line = line.strip()
            if line == 'intrinsic:':
                reading_intrinsic = True
                reading_distortion = False
                continue
            elif line == 'distortion:':
                reading_intrinsic = False
                reading_distortion = True
                continue
            elif line == '':
                continue
            
            if reading_intrinsic:
                values = [float(x) for x in line.split()]
                intrinsic_matrix.append(values)
            elif reading_distortion:
                dist_coeffs = [float(x) for x in line.split()]
        
        intrinsic_matrix = np.array(intrinsic_matrix, dtype=np.float32)
        dist_coeffs = np.array(dist_coeffs, dtype=np.float32)
        
        return intrinsic_matrix, dist_coeffs
    
    def read_rot_trans_file(filepath):
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        rotation = []
        translation = []
        reading_rotation = False
        reading_translation = False
        
        for line in lines:
            line = line.strip()
            if line == 'R:':
                reading_rotation = True
                reading_translation = False
                continue
            elif line == 'T:':
                reading_rotation = False
                reading_translation = True
                continue
            elif line == '':
                continue
            
            if reading_rotation:
                values = [float(x) for x in line.split()]
                rotation.append(values)
            elif reading_translation:
                values = [float(x) for x in line.split()]
                translation.extend(values)
        
        rotation = np.array(rotation, dtype=np.float32)
        translation = np.array(translation, dtype=np.float32).reshape(3, 1)
        
        return rotation, translation
    
    # Load camera 0 (left)
    cam0_intrinsic_path = os.path.join(camera_params_dir, 'camera0_intrinsics.dat')
    cam0_rot_trans_path = os.path.join(camera_params_dir, 'camera0_rot_trans.dat')
    
    cam0_intrinsic, cam0_dist = read_intrinsics_file(cam0_intrinsic_path)
    cam0_rot, cam0_trans = read_rot_trans_file(cam0_rot_trans_path)
    
    # Load camera 1 (right)
    cam1_intrinsic_path = os.path.join(camera_params_dir, 'camera1_intrinsics.dat')
    cam1_rot_trans_path = os.path.join(camera_params_dir, 'camera1_rot_trans.dat')
    
    cam1_intrinsic, cam1_dist = read_intrinsics_file(cam1_intrinsic_path)
    cam1_rot, cam1_trans = read_rot_trans_file(cam1_rot_trans_path)
    
    width, height = 1280, 720
    
    cam0_params = CameraParameters(
        intrinsic_matrix=cam0_intrinsic,
        distortion=cam0_dist,
        rotation=cam0_rot,
        translation=cam0_trans,
        width=width,
        height=height,
        fx=cam0_intrinsic[0, 0],
        fy=cam0_intrinsic[1, 1],
        cx=cam0_intrinsic[0, 2],
        cy=cam0_intrinsic[1, 2]
    )
    
    cam1_params = CameraParameters(
        intrinsic_matrix=cam1_intrinsic,
        distortion=cam1_dist,
        rotation=cam1_rot,
        translation=cam1_trans,
        width=width,
        height=height,
        fx=cam1_intrinsic[0, 0],
        fy=cam1_intrinsic[1, 1],
        cx=cam1_intrinsic[0, 2],
        cy=cam1_intrinsic[1, 2]
    )
    
    print(f"✓ Loaded camera parameters from {camera_params_dir}/")
    
    return cam0_params, cam1_params


# ==================== BALL DETECTOR ====================

class BallDetector:
    """Detects colored ball using HSV thresholding"""
    
    def __init__(self, hsv_config: dict):
        self.lower = np.array(hsv_config['lower'])
        self.upper = np.array(hsv_config['upper'])
        self.kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    
    def set_hsv_range(self, lower: tuple, upper: tuple):
        self.lower = np.array(lower)
        self.upper = np.array(upper)
    
    def detect(self, hsv_frame: np.ndarray) -> Optional[Tuple[int, int, int]]:
        mask = cv.inRange(hsv_frame, self.lower, self.upper)
        mask = cv.erode(mask, self.kernel, iterations=1)
        mask = cv.dilate(mask, self.kernel, iterations=2)
        
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        best_detection = None
        best_score = 0
        
        for contour in contours:
            area = cv.contourArea(contour)
            if area < MIN_BALL_AREA or area > MAX_BALL_AREA:
                continue
            
            perimeter = cv.arcLength(contour, True)
            if perimeter == 0:
                continue
            
            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity < MIN_CIRCULARITY:
                continue
            
            (x, y), radius = cv.minEnclosingCircle(contour)
            
            score = circularity * area
            if score > best_score:
                best_score = score
                best_detection = (int(x), int(y), int(radius))
        
        return best_detection


# ==================== DEPTH/3D FUNCTIONS ====================

def get_robust_depth(depth_map: sl.Mat, point_cloud: sl.Mat,
                     x: int, y: int, intrinsics: CameraIntrinsics,
                     sample_radius: int = 5) -> Tuple[Optional[np.ndarray], float]:
    """Get robust 3D position using multi-point sampling"""
    h, w = depth_map.get_height(), depth_map.get_width()
    valid_points = []
    
    for dy in range(-sample_radius, sample_radius + 1, 2):
        for dx in range(-sample_radius, sample_radius + 1, 2):
            px, py = x + dx, y + dy
            if 0 <= px < w and 0 <= py < h:
                err, point = point_cloud.get_value(px, py)
                if err == sl.ERROR_CODE.SUCCESS and np.isfinite(point[2]):
                    valid_points.append(point[:3])
    
    if not valid_points:
        return None, 0.0
    
    valid_points = np.array(valid_points)
    median_point = np.median(valid_points, axis=0)
    confidence = len(valid_points) / ((2 * sample_radius + 1) ** 2)
    
    return median_point, confidence


# ==================== VISUALIZATION ====================

def draw_detection(frame: np.ndarray, detection: Tuple[int, int, int], 
                   label: str, color: Tuple[int, int, int]):
    x, y, radius = detection
    cv.circle(frame, (x, y), radius, color, 2)
    cv.circle(frame, (x, y), 3, color, -1)
    if label:
        cv.putText(frame, label, (x + radius + 5, y),
                  cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def draw_tracking_overlay(frame: np.ndarray, detection: Optional[Tuple], 
                          pos_3d: Optional[np.ndarray], param_source: str,
                          triangulation_success: bool, ball_detected: bool,
                          udp_sender: Optional[UDPSender] = None):
    h, w = frame.shape[:2]
    y = 30
    
    # Header
    cv.putText(frame, "Ball Tracker with UDP Output", (10, y),
              cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    y += 25
    
    # UDP Status
    if udp_sender:
        udp_status = f"UDP: {udp_sender.robot_ip}:{udp_sender.port}"
        udp_color = (0, 255, 0) if udp_sender.enabled else (0, 0, 255)
        status_text = "ON" if udp_sender.enabled else "OFF"
        cv.putText(frame, f"{udp_status} [{status_text}]", (10, y),
                  cv.FONT_HERSHEY_SIMPLEX, 0.45, udp_color, 1)
        y += 20
        cv.putText(frame, f"Packets sent: {udp_sender.packets_sent}", (10, y),
                  cv.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        y += 25
    
    # Detection status
    status = "TRACKING" if ball_detected else "SEARCHING"
    color = (0, 255, 0) if ball_detected else (0, 165, 255)
    cv.putText(frame, f"Ball: {status}", (10, y),
              cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    y += 25
    
    # Triangulation status
    tri_status = "OK" if triangulation_success else "FAILED"
    tri_color = (0, 255, 0) if triangulation_success else (0, 0, 255)
    cv.putText(frame, f"3D: {tri_status} ({param_source})", (10, y),
              cv.FONT_HERSHEY_SIMPLEX, 0.5, tri_color, 1)
    y += 30
    
    # 3D coordinates
    if pos_3d is not None:
        cv.putText(frame, "3D Position (m):", (10, y),
                  cv.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y += 22
        cv.putText(frame, f"  X: {pos_3d[0]:+.3f}", (10, y),
                  cv.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1)
        y += 22
        cv.putText(frame, f"  Y: {pos_3d[1]:+.3f}", (10, y),
                  cv.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
        y += 22
        cv.putText(frame, f"  Z: {pos_3d[2]:+.3f}", (10, y),
                  cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 1)
    
    # Instructions at bottom
    cv.putText(frame, "[R] Reset  [T] Tennis  [P] Paper  [U] Toggle UDP  [Q] Quit",
              (10, h - 10), cv.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)


# ==================== MAIN ====================

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Ball tracking with UDP output')
    parser.add_argument('--use-files', action='store_true',
                       help='Load camera parameters from files instead of ZED SDK')
    parser.add_argument('--robot-ip', type=str, default=DEFAULT_ROBOT_IP,
                       help=f'Robot IP address (default: {DEFAULT_ROBOT_IP})')
    parser.add_argument('--port', type=int, default=DEFAULT_ROBOT_PORT,
                       help=f'UDP port (default: {DEFAULT_ROBOT_PORT})')
    parser.add_argument('--no-udp', action='store_true',
                       help='Disable UDP sending')
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("  BALL TRACKING WITH UDP OUTPUT")
    print("=" * 60)
    
    # Initialize UDP sender
    udp_sender = None
    if not args.no_udp:
        udp_sender = UDPSender(args.robot_ip, args.port, rate_limit=UDP_SEND_RATE)
        print(f"✓ UDP sender: {args.robot_ip}:{args.port}")
        print(f"  Send rate: {UDP_SEND_RATE} Hz")
    else:
        print("✗ UDP sending disabled")
    
    # Initialize ZED camera
    print("\nInitializing ZED camera...")
    zed = sl.Camera()
    init = sl.InitParameters()
    init.camera_resolution = RESOLUTION
    init.camera_fps = FPS
    init.depth_mode = DEPTH_MODE
    init.coordinate_units = UNIT
    init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    
    err = zed.open(init)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"✗ Failed to open ZED camera: {err}")
        if udp_sender:
            udp_sender.close()
        return
    
    print("✓ ZED camera opened")
    
    # Load camera parameters based on user choice
    if args.use_files:
        print("\n✓ Loading camera parameters from files...")
        cam0_params, cam1_params = load_camera_parameters_from_files()
        
        intrinsics = CameraIntrinsics(
            fx=cam0_params.fx,
            fy=cam0_params.fy,
            cx=cam0_params.cx,
            cy=cam0_params.cy,
            width=cam0_params.width,
            height=cam0_params.height
        )
        param_source = "Files"
    else:
        print("\n✓ Loading camera parameters from ZED SDK...")
        cam_info = zed.get_camera_information()
        
        try:
            calib = cam_info.camera_configuration.calibration_parameters.left_cam
        except AttributeError:
            try:
                calib = cam_info.calibration_parameters.left_cam
            except AttributeError:
                calib = cam_info.camera_configuration.calibration_parameters.left_cam
        
        try:
            res_width = cam_info.camera_configuration.camera_resolution.width
            res_height = cam_info.camera_configuration.camera_resolution.height
        except AttributeError:
            try:
                res_width = cam_info.camera_configuration.resolution.width
                res_height = cam_info.camera_configuration.resolution.height
            except AttributeError:
                try:
                    res_width = cam_info.camera_resolution.width
                    res_height = cam_info.camera_resolution.height
                except AttributeError:
                    if RESOLUTION == sl.RESOLUTION.HD720:
                        res_width, res_height = 1280, 720
                    elif RESOLUTION == sl.RESOLUTION.HD1080:
                        res_width, res_height = 1920, 1080
                    else:
                        res_width, res_height = 1280, 720
        
        intrinsics = CameraIntrinsics(
            fx=calib.fx, fy=calib.fy,
            cx=calib.cx, cy=calib.cy,
            width=res_width,
            height=res_height
        )
        param_source = "ZED SDK"
    
    print(f"\n✓ Camera: {intrinsics.width}x{intrinsics.height}")
    print(f"✓ Intrinsics: fx={intrinsics.fx:.2f}, fy={intrinsics.fy:.2f}")
    print("=" * 60)
    
    # Initialize detector
    detector = BallDetector(ACTIVE_HSV)
    
    # Runtime parameters
    runtime = sl.RuntimeParameters()
    runtime.confidence_threshold = 50
    
    # Allocate ZED mats
    image_left = sl.Mat()
    depth_map = sl.Mat()
    point_cloud = sl.Mat()
    
    print("\nStarting tracking loop...")
    print("Real-time coordinates will be sent to robot via UDP.\n")
    
    # Temporal smoothing variables
    last_valid_pos_3d = None
    last_valid_time = 0
    POSITION_TIMEOUT = 0.5
    
    try:
        while True:
            # Grab frame
            if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                continue
            
            # Retrieve left image, depth map, and point cloud
            zed.retrieve_image(image_left, sl.VIEW.LEFT)
            zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH)
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
            
            frame0 = image_left.get_data()[:, :, :3].copy()
            
            # Convert to HSV
            hsv0 = cv.cvtColor(frame0, cv.COLOR_BGR2HSV)
            
            # Detect ball in left view
            detection0 = detector.detect(hsv0)
            
            # Get 3D position if detected
            pos_3d = None
            pos_3d_corrected = None
            depth_confidence = 0.0
            
            if detection0:
                x, y, radius = detection0
                
                # Get 3D position from point cloud/depth
                world_pos, depth_conf = get_robust_depth(
                    depth_map, point_cloud, x, y, intrinsics
                )
                
                current_time = time.time()
                
                if world_pos is not None:
                    # Apply coordinate offset corrections
                    pos_3d_corrected = world_pos.copy()
                    pos_3d_corrected[1] += CAMERA_HEIGHT_ABOVE_GROUND
                    pos_3d_corrected[2] += DEPTH_OFFSET
                    
                    # Update last valid position
                    last_valid_pos_3d = pos_3d_corrected.copy()
                    last_valid_time = current_time
                    depth_confidence = depth_conf
                    
                    # Send coordinates via UDP
                    if udp_sender and udp_sender.enabled:
                        udp_sender.send_coordinates(
                            pos_3d_corrected[0],
                            pos_3d_corrected[1],
                            pos_3d_corrected[2]
                        )
                    
                    # Draw detection with success color
                    draw_detection(frame0, detection0, "", (0, 255, 0))
                else:
                    # Depth failed, try to use last valid position if recent
                    if last_valid_pos_3d is not None and (current_time - last_valid_time) < POSITION_TIMEOUT:
                        pos_3d_corrected = last_valid_pos_3d
                        draw_detection(frame0, detection0, "", (0, 200, 200))
                    else:
                        draw_detection(frame0, detection0, "", (0, 165, 255))
            else:
                # No detection - clear persistence after timeout
                if last_valid_pos_3d is not None and (time.time() - last_valid_time) >= POSITION_TIMEOUT:
                    last_valid_pos_3d = None
            
            # Draw overlay
            triangulation_success = world_pos is not None if detection0 else False
            display_pos = pos_3d_corrected if pos_3d_corrected is not None else None
            draw_tracking_overlay(frame0, detection0, display_pos, 
                                param_source, triangulation_success or (pos_3d_corrected is not None and detection0), 
                                detection0 is not None, udp_sender)
            
            # Display
            cv.imshow("Ball Tracking - UDP Output", frame0)
            
            # Handle keyboard input
            key = cv.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('r'):
                last_valid_pos_3d = None
                print("[RESET]")
            elif key == ord('t'):
                detector.set_hsv_range(HSV_TENNIS_BALL['lower'], HSV_TENNIS_BALL['upper'])
                print("[HSV] Tennis ball")
            elif key == ord('p'):
                detector.set_hsv_range(HSV_PAPER_BALL['lower'], HSV_PAPER_BALL['upper'])
                print("[HSV] Paper ball")
            elif key == ord('u'):
                if udp_sender:
                    udp_sender.toggle()
    
    finally:
        cv.destroyAllWindows()
        zed.close()
        if udp_sender:
            print(f"\n✓ Total UDP packets sent: {udp_sender.packets_sent}")
            udp_sender.close()
        print("✓ Tracking stopped")
        print("=" * 60)


if __name__ == "__main__":
    main()
