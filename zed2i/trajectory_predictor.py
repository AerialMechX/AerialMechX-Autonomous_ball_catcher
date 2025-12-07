"""
Trajectory Predictor for Ball Landing Estimation (YOLO-based)

This script predicts the landing coordinates of a thrown ball using:
- YOLO object detection with HSV fallback
- Physics-based Kalman filtering
- Velocity-based trajectory prediction

Camera parameters are loaded from ../camera_parameters/ folder ONLY.

Usage:
    python trajectory_predictor.py                    # Run with ZED camera
    python trajectory_predictor.py --robot-ip 192.168.0.51  # Set robot IP
    python trajectory_predictor.py --no-udp           # Disable UDP
    python trajectory_predictor.py --no-gpu           # Disable GPU

Controls:
    q/ESC - Quit
    r - Reset tracking
    t - Switch to tennis ball HSV
    p - Switch to paper ball HSV
    u - Toggle UDP sending on/off
"""

import numpy as np
import cv2 as cv
import pyzed.sl as sl
import time
import os
import argparse
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

from sender import UDPSender


# ==================== CONFIGURATION ====================

# YOLO Configuration
YOLO_MODEL = "yolov8n.pt"
YOLO_CONFIDENCE = 0.35
USE_GPU = True

# Import from config
from config import (
    RESOLUTION, FPS, DEPTH_MODE, UNIT,
    HSV_TENNIS_BALL, HSV_PAPER_BALL, ACTIVE_HSV,
    CAMERA_HEIGHT_ABOVE_GROUND, DEPTH_OFFSET,
    MIN_BALL_AREA, MAX_BALL_AREA, MIN_CIRCULARITY,
    DEFAULT_ROBOT_IP, DEFAULT_ROBOT_PORT, UDP_SEND_RATE,
    GRAVITY
)

# Trajectory parameters
MIN_POINTS = 5  # Minimum points before prediction
MIN_CONF_LOCK = 0.35  # Minimum confidence to lock prediction
MIN_TIME_TO_LAND = 0.05  # Minimum flight time to consider valid (seconds)
MIN_THROW_VELOCITY = 1.5  # Minimum velocity to start tracking (m/s)
# Ground Y in ZED coords: +Y is up, camera is 0.6m above ground
GROUND_Y = -0.58  # Ground height relative to camera
DEBUG_TRAJECTORY = True  # Print debug info


# ==================== DATA CLASSES ====================

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


@dataclass
class Landing:
    """Landing prediction result"""
    x: float
    y: float
    z: float
    time: float
    confidence: float
    
    @property
    def pos(self):
        return np.array([self.x, self.y, self.z])


# ==================== PARAMETER LOADING ====================

def load_camera_parameters_from_files(camera_params_dir: str = '../camera_parameters'):
    """Load camera parameters from .dat files (REQUIRED - no SDK fallback)"""
    
    def read_intrinsics_file(filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Camera parameters file not found: {filepath}")
        
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
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Camera parameters file not found: {filepath}")
        
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
    
    # Load camera 0
    cam0_intrinsic_path = os.path.join(camera_params_dir, 'camera0_intrinsics.dat')
    cam0_rot_trans_path = os.path.join(camera_params_dir, 'camera0_rot_trans.dat')
    
    cam0_intrinsic, cam0_dist = read_intrinsics_file(cam0_intrinsic_path)
    cam0_rot, cam0_trans = read_rot_trans_file(cam0_rot_trans_path)
    
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
    
    print(f"✓ Loaded camera parameters from {camera_params_dir}/")
    print(f"  Camera 0: fx={cam0_params.fx:.2f}, fy={cam0_params.fy:.2f}")
    
    return cam0_params


# ==================== KALMAN FILTER ====================

class KalmanFilter:
    """Physics-based Kalman filter for ball trajectory."""
    
    def __init__(self, g=GRAVITY, q=0.01, r=0.02):
        self.g = g
        self.state = np.zeros(6)  # [x, y, z, vx, vy, vz]
        self.P = np.eye(6)
        self.Q_base = q
        self.R = np.eye(3) * r
        self.H = np.zeros((3, 6))
        self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = 1
        self.init = False
        self.n = 0
        self.t_last = None
    
    def update(self, m, t):
        m = np.asarray(m).flatten()[:3]
        
        if not self.init:
            self.state[:3] = m
            self.init = True
            self.t_last = t
            self.n = 1
            return self.state.copy()
        
        dt = t - self.t_last
        if dt > 0:
            # Predict step
            F = np.eye(6)
            F[0, 3] = F[1, 4] = F[2, 5] = dt
            B = np.zeros(6)
            B[1] = -0.5 * self.g * dt**2  # Gravity in -Y direction
            B[4] = -self.g * dt
            self.state = F @ self.state + B
            self.P = F @ self.P @ F.T + np.eye(6) * self.Q_base * dt
        
        self.t_last = t
        
        # Update step
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ (m - self.H @ self.state)
        self.P = (np.eye(6) - K @ self.H) @ self.P
        self.n += 1
        
        return self.state.copy()
    
    def pos(self):
        return self.state[:3].copy()
    
    def vel(self):
        return self.state[3:].copy()
    
    def reset(self):
        self.state = np.zeros(6)
        self.P = np.eye(6)
        self.init = False
        self.n = 0
        self.t_last = None


# ==================== TRAJECTORY PREDICTOR ====================

class TrajectoryPredictor:
    """Predicts ball landing position using raw position tracking and physics."""
    
    def __init__(self, g=GRAVITY, ground=GROUND_Y):
        self.g = g
        self.ground = ground
        
        # Track raw positions and times
        self.pos_hist = deque(maxlen=15)  # Buffer of (pos, time) tuples
        self.n = 0  # Total measurements
        
        # Prediction locking
        self.locked = False
        self.lock_pos = None
        self.lock_vel = None
        self.lock_land = None
        self.lock_time = 0
        self.current = None
    
    def _compute_velocity(self):
        """Compute velocity from recent position history using linear regression."""
        if len(self.pos_hist) < 3:
            return None
        
        # Use last 8 points for velocity estimation
        recent = list(self.pos_hist)[-8:]
        positions = np.array([p[0] for p in recent])
        times = np.array([p[1] for p in recent])
        
        # Center times for numerical stability
        t0 = times[0]
        times_centered = times - t0
        
        if times_centered[-1] - times_centered[0] < 0.05:  # Need at least 50ms span
            return None
        
        # Simple linear fit: pos = pos0 + vel * t
        # Use least squares: vel = sum((t - t_mean)(pos - pos_mean)) / sum((t - t_mean)^2)
        t_mean = np.mean(times_centered)
        pos_mean = np.mean(positions, axis=0)
        
        numerator = np.zeros(3)
        denominator = 0.0
        for i in range(len(recent)):
            dt = times_centered[i] - t_mean
            dpos = positions[i] - pos_mean
            numerator += dt * dpos
            denominator += dt * dt
        
        if denominator < 1e-6:
            return None
        
        velocity = numerator / denominator
        return velocity
    
    def update(self, pos, t):
        if self.locked:
            return self.lock_land
        
        # Store raw position
        pos = np.asarray(pos)
        self.pos_hist.append((pos.copy(), t))
        self.n += 1
        
        # Compute velocity from position history
        vel = self._compute_velocity()
        
        if vel is not None:
            speed = np.linalg.norm(vel)
            
            # Debug output
            if DEBUG_TRAJECTORY and self.n % 5 == 0:
                print(f"[DEBUG] Points: {self.n}, Speed: {speed:.2f} m/s, Pos: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}], Vel: [{vel[0]:.2f}, {vel[1]:.2f}, {vel[2]:.2f}]")
            
            # Need minimum points and velocity for prediction
            if self.n >= MIN_POINTS and speed >= MIN_THROW_VELOCITY:
                self.current = self._predict(pos, vel)
                if self.current:
                    if DEBUG_TRAJECTORY:
                        print(f"[DEBUG] Prediction: X={self.current.x:.2f}, Z={self.current.z:.2f}, T={self.current.time:.2f}s, Conf={self.current.confidence:.0%}")
                    
                    # Check if should lock
                    lock_result, reason = self._should_lock()
                    if lock_result:
                        self.locked = True
                        self.lock_pos = pos.copy()
                        self.lock_vel = vel.copy()
                        self.lock_land = self.current
                        self.lock_time = self.current.time
                        self._print_prediction()
                    elif DEBUG_TRAJECTORY:
                        print(f"[DEBUG] Not locking: {reason}")
                elif DEBUG_TRAJECTORY and self.n % 10 == 0:
                    # Show why prediction failed
                    a = -0.5 * self.g
                    b = vel[1]
                    c = pos[1] - self.ground
                    disc = b*b - 4*a*c
                    print(f"[DEBUG] No prediction: pos_y={pos[1]:.2f}, vel_y={vel[1]:.2f}, disc={disc:.2f}")
            elif DEBUG_TRAJECTORY and self.n % 10 == 0:
                print(f"[DEBUG] Waiting: speed={speed:.2f} m/s (need >= {MIN_THROW_VELOCITY})")
        
        return self.current
    
    def _should_lock(self):
        """Check if prediction should be locked. Returns (bool, reason)"""
        if self.current is None:
            return False, "No prediction"
        
        # Must have sufficient confidence
        if self.current.confidence < MIN_CONF_LOCK:
            return False, f"Low confidence: {self.current.confidence:.0%} < {MIN_CONF_LOCK:.0%}"
        
        # Must have reasonable flight time remaining
        if self.current.time < MIN_TIME_TO_LAND:
            return False, f"Short flight: {self.current.time:.2f}s < {MIN_TIME_TO_LAND}s"
        
        # Landing shouldn't be too far away (sanity check)
        if abs(self.current.x) > 10 or abs(self.current.z) > 15:
            return False, f"Too far: X={self.current.x:.1f}, Z={self.current.z:.1f}"
        
        return True, "OK"
    
    def _predict(self, pos, vel):
        """Predict landing position given current position and velocity."""
        # Solve y(t) = ground for t
        # y0 + vy*t - 0.5*g*t^2 = ground
        # -0.5*g*t^2 + vy*t + (y0 - ground) = 0
        a = -0.5 * self.g
        b = vel[1]
        c = pos[1] - self.ground
        
        disc = b*b - 4*a*c
        if disc < 0:
            return None
        
        t1 = (-b + np.sqrt(disc)) / (2*a)
        t2 = (-b - np.sqrt(disc)) / (2*a)
        ts = [t for t in [t1, t2] if t > MIN_TIME_TO_LAND]
        if not ts:
            return None
        
        t_land = min(ts)
        x_land = pos[0] + vel[0] * t_land
        z_land = pos[2] + vel[2] * t_land
        
        # Confidence based on number of points and flight time
        n_factor = min(1.0, self.n / 12)
        time_factor = min(1.0, t_land / 0.5)  # Full credit at 0.5s+
        
        conf = 0.5 * n_factor + 0.5 * time_factor
        
        return Landing(x_land, self.ground, z_land, t_land, float(np.clip(conf, 0, 1)))
    
    def _print_prediction(self):
        """Print and save prediction"""
        if self.lock_land is None:
            return
        
        land = self.lock_land
        print("\n" + "=" * 50)
        print("  TRAJECTORY PREDICTION (LOCKED)")
        print("=" * 50)
        print(f"  Initial Position: [{self.lock_pos[0]:.3f}, {self.lock_pos[1]:.3f}, {self.lock_pos[2]:.3f}] m")
        print(f"  Initial Velocity: [{self.lock_vel[0]:.3f}, {self.lock_vel[1]:.3f}, {self.lock_vel[2]:.3f}] m/s")
        print("-" * 50)
        print(f"  LANDING POSITION: X={land.x:+.3f}m  Z={land.z:+.3f}m")
        print(f"  TIME TO LANDING: {land.time:.3f} seconds")
        print(f"  CONFIDENCE: {land.confidence:.0%}")
        print("=" * 50)
        print("  Press [R] to reset and predict new throw")
        print("=" * 50 + "\n")
        
        # Save to file
        log_file = "predictions.txt"
        timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S")
        write_header = not os.path.exists(log_file)
        
        with open(log_file, 'a') as f:
            if write_header:
                f.write("=" * 80 + "\n")
                f.write("TRAJECTORY PREDICTION LOG\n")
                f.write("=" * 80 + "\n\n")
            
            f.write("-" * 80 + "\n")
            f.write(f"{timestamp_str}\n")
            f.write("-" * 80 + "\n")
            f.write(f"Initial Position (m): {self.lock_pos[0]:.4f}, {self.lock_pos[1]:.4f}, {self.lock_pos[2]:.4f}\n")
            f.write(f"Initial Velocity (m/s): {self.lock_vel[0]:.4f}, {self.lock_vel[1]:.4f}, {self.lock_vel[2]:.4f}\n")
            f.write(f"LANDING (m): X={land.x:.4f}, Y={land.y:.4f}, Z={land.z:.4f}\n")
            f.write(f"TIME TO LAND (s): {land.time:.4f}\n")
            f.write(f"CONFIDENCE: {land.confidence:.4f}\n\n")
        
        print(f"[SAVED] Appended to: {log_file}")
    
    def get_traj_pts(self, n=40):
        """Get trajectory points for visualization"""
        if self.locked:
            if self.lock_pos is None:
                return None
            p, v, t = self.lock_pos, self.lock_vel, self.lock_time
        else:
            if self.current is None or len(self.pos_hist) < 2:
                return None
            # Use latest position and computed velocity
            p = self.pos_hist[-1][0]
            v = self._compute_velocity()
            if v is None:
                return None
            t = self.current.time
        
        ts = np.linspace(0, t, n)
        pts = []
        for ti in ts:
            pts.append([p[0] + v[0]*ti, 
                       p[1] + v[1]*ti - 0.5*self.g*ti**2, 
                       p[2] + v[2]*ti])
        return np.array(pts)
    
    def get_land(self):
        return self.lock_land if self.locked else self.current
    
    def get_init_pos(self):
        return self.lock_pos if self.locked else None
    
    def reset(self):
        self.pos_hist.clear()
        self.n = 0
        self.locked = False
        self.lock_pos = self.lock_vel = self.lock_land = None
        self.lock_time = 0
        self.current = None
        print("[RESET] Ready for new throw")


# ==================== BALL DETECTOR (YOLO + HSV) ====================

class BallDetector:
    """Detects ball using YOLO with HSV fallback"""
    
    def __init__(self, hsv_config: dict, use_gpu: bool = True):
        print("Loading YOLO model...")
        from ultralytics import YOLO
        self.model = YOLO(YOLO_MODEL)
        
        if use_gpu:
            try:
                self.model.to("cuda")
                print("✓ YOLO using CUDA GPU")
            except Exception as e:
                print(f"  GPU unavailable, using CPU: {e}")
        
        # Warmup
        self.model.predict(np.zeros((320, 320, 3), dtype=np.uint8), verbose=False)
        
        self.lower = np.array(hsv_config['lower'])
        self.upper = np.array(hsv_config['upper'])
        self.kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
        self.last_method = None
        print("✓ Ball detector ready (YOLO + HSV fallback)")
    
    def set_hsv_range(self, lower: tuple, upper: tuple):
        self.lower = np.array(lower)
        self.upper = np.array(upper)
    
    def detect(self, frame: np.ndarray) -> Optional[Tuple[int, int, int]]:
        """Detect ball. Returns (x, y, radius) or None"""
        # Try YOLO first
        results = self.model.predict(frame, conf=YOLO_CONFIDENCE, classes=[32],
                                      verbose=False, imgsz=320)
        for result in results:
            if len(result.boxes) > 0:
                box = result.boxes[0]
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                radius = max(x2 - x1, y2 - y1) / 2
                if 5 < radius < 150:
                    self.last_method = 'YOLO'
                    return (int(cx), int(cy), int(radius))
        
        # HSV fallback
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, self.lower, self.upper)
        mask = cv.erode(mask, self.kernel, iterations=1)
        mask = cv.dilate(mask, self.kernel, iterations=2)
        
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        for cnt in sorted(contours, key=cv.contourArea, reverse=True)[:3]:
            area = cv.contourArea(cnt)
            if MIN_BALL_AREA < area < MAX_BALL_AREA:
                perim = cv.arcLength(cnt, True)
                if perim > 0 and 4 * np.pi * area / perim**2 > MIN_CIRCULARITY:
                    (cx, cy), radius = cv.minEnclosingCircle(cnt)
                    if 5 < radius < 150:
                        self.last_method = 'HSV'
                        return (int(cx), int(cy), int(radius))
        
        self.last_method = None
        return None


# ==================== DEPTH FUNCTIONS ====================

def get_robust_depth(depth_map: sl.Mat, point_cloud: sl.Mat,
                     x: int, y: int, intrinsics: CameraIntrinsics,
                     sample_radius: int = 5) -> Tuple[Optional[np.ndarray], float]:
    """Get robust 3D position using multi-point sampling
    
    ZED coordinate system:
    - +X: right
    - +Y: up  
    - +Z: into camera (so objects in front have NEGATIVE Z)
    
    For trajectory prediction, we negate Z so forward distance is positive.
    """
    h, w = depth_map.get_height(), depth_map.get_width()
    valid_points = []
    
    for dy in range(-sample_radius, sample_radius + 1, 2):
        for dx in range(-sample_radius, sample_radius + 1, 2):
            px, py = x + dx, y + dy
            if 0 <= px < w and 0 <= py < h:
                err, point = point_cloud.get_value(px, py)
                if err == sl.ERROR_CODE.SUCCESS and np.isfinite(point[2]):
                    # Negate Z: ZED has +Z into camera, we want +Z forward into scene
                    valid_points.append([point[0], point[1], -point[2]])
    
    if not valid_points:
        return None, 0.0
    
    valid_points = np.array(valid_points)
    median_point = np.median(valid_points, axis=0)
    confidence = len(valid_points) / ((2 * sample_radius + 1) ** 2)
    
    return median_point, confidence


# ==================== VISUALIZATION ====================

def project_to_2d(point_3d: np.ndarray, intrinsics: CameraIntrinsics) -> Optional[Tuple[int, int]]:
    """Project 3D point to 2D pixel coordinates"""
    if point_3d[2] <= 0:
        return None
    px = int(point_3d[0] * intrinsics.fx / point_3d[2] + intrinsics.cx)
    py = int(-point_3d[1] * intrinsics.fy / point_3d[2] + intrinsics.cy)
    if 0 <= px < intrinsics.width and 0 <= py < intrinsics.height:
        return (px, py)
    return None


def draw_overlay(frame: np.ndarray, predictor: TrajectoryPredictor,
                 current_pos: Optional[np.ndarray], detection_method: Optional[str],
                 intrinsics: CameraIntrinsics, udp_sender: Optional[UDPSender]):
    """Draw trajectory prediction overlay"""
    h, w = frame.shape[:2]
    y = 30
    
    # Header
    cv.putText(frame, "YOLO Trajectory Predictor", (10, y),
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
    
    # Detection status
    if current_pos is not None:
        method_str = f" ({detection_method})" if detection_method else ""
        cv.putText(frame, f"Ball: TRACKING{method_str}", (10, y),
                  cv.FONT_HERSHEY_SIMPLEX, 0.5, 
                  (0, 255, 0) if detection_method == 'YOLO' else (255, 255, 0), 1)
    else:
        cv.putText(frame, "Ball: SEARCHING", (10, y),
                  cv.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    y += 20
    
    # Prediction status
    if predictor.locked:
        cv.putText(frame, "Status: PREDICTION LOCKED", (10, y),
                  cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        y += 20
        cv.putText(frame, "Press [R] to reset for next throw", (10, y),
                  cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
    elif predictor.n > 0:
        cv.putText(frame, f"Status: TRACKING ({predictor.n} pts)", (10, y),
                  cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    else:
        cv.putText(frame, "Status: WAITING FOR THROW", (10, y),
                  cv.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    y += 25
    
    # Current position
    if current_pos is not None:
        cv.putText(frame, f"Pos: X={current_pos[0]:+.2f} Y={current_pos[1]:+.2f} Z={current_pos[2]:+.2f}m", 
                  (10, y), cv.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        y += 20
    
    # Velocity
    vel = predictor._compute_velocity()
    if vel is not None:
        speed = np.linalg.norm(vel)
        cv.putText(frame, f"Speed: {speed:.2f} m/s", (10, y),
                  cv.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        y += 25
    
    # Landing prediction
    land = predictor.get_land()
    if land:
        label = "LOCKED" if predictor.locked else "PRED"
        color = (255, 0, 255) if predictor.locked else (0, 255, 0)
        cv.putText(frame, f"{label}: X={land.x:+.2f}m Z={land.z:+.2f}m", (10, y),
                  cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        y += 20
        cv.putText(frame, f"Time: {land.time:.2f}s  Conf: {land.confidence:.0%}", (10, y),
                  cv.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    
    # Draw trajectory
    traj = predictor.get_traj_pts(30)
    if traj is not None:
        pts_2d = [project_to_2d(pt, intrinsics) for pt in traj]
        pts_2d = [p for p in pts_2d if p is not None]
        for i in range(1, len(pts_2d)):
            prog = i / len(pts_2d)
            cv.line(frame, pts_2d[i-1], pts_2d[i], 
                   (0, int(255*(1-prog)), int(255*prog)), 2)
    
    # Draw landing marker
    if land:
        lp = project_to_2d(land.pos, intrinsics)
        if lp:
            cv.circle(frame, lp, 15, (0, 0, 255), 2)
            cv.drawMarker(frame, lp, (0, 0, 255), cv.MARKER_CROSS, 20, 2)
    
    # Draw initial position
    ip = predictor.get_init_pos()
    if ip is not None:
        ip2d = project_to_2d(ip, intrinsics)
        if ip2d:
            cv.circle(frame, ip2d, 8, (0, 255, 0), 2)
    
    # Instructions
    cv.putText(frame, "[R] Reset  [T] Tennis  [P] Paper  [U] UDP  [Q] Quit",
              (10, h - 10), cv.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)


# ==================== MAIN ====================

def main():
    parser = argparse.ArgumentParser(description='YOLO Trajectory prediction')
    parser.add_argument('--robot-ip', type=str, default=DEFAULT_ROBOT_IP,
                       help=f'Robot IP address (default: {DEFAULT_ROBOT_IP})')
    parser.add_argument('--port', type=int, default=DEFAULT_ROBOT_PORT,
                       help=f'UDP port (default: {DEFAULT_ROBOT_PORT})')
    parser.add_argument('--no-udp', action='store_true',
                       help='Disable UDP sending')
    parser.add_argument('--no-gpu', action='store_true',
                       help='Disable GPU acceleration for YOLO')
    parser.add_argument('--send-prediction', action='store_true',
                       help='Send predicted landing position instead of current position')
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("  YOLO TRAJECTORY PREDICTOR - BALL LANDING ESTIMATION")
    print("  Camera parameters from: ../camera_parameters/")
    print("=" * 60)
    
    # Load camera parameters from files
    print("\n✓ Loading camera parameters from files...")
    try:
        cam_params = load_camera_parameters_from_files()
    except FileNotFoundError as e:
        print(f"\n✗ ERROR: {e}")
        print("  Make sure camera parameter files exist in ../camera_parameters/")
        return
    
    intrinsics = CameraIntrinsics(
        fx=cam_params.fx,
        fy=cam_params.fy,
        cx=cam_params.cx,
        cy=cam_params.cy,
        width=cam_params.width,
        height=cam_params.height
    )
    
    # Initialize UDP sender
    udp_sender = None
    if not args.no_udp:
        udp_sender = UDPSender(args.robot_ip, args.port, rate_limit=UDP_SEND_RATE)
        print(f"✓ UDP sender: {args.robot_ip}:{args.port}")
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
    print(f"✓ Camera: {intrinsics.width}x{intrinsics.height}")
    print(f"✓ Intrinsics: fx={intrinsics.fx:.2f}, fy={intrinsics.fy:.2f}")
    print("=" * 60)
    
    # Initialize components
    use_gpu = USE_GPU and not args.no_gpu
    detector = BallDetector(ACTIVE_HSV, use_gpu=use_gpu)
    predictor = TrajectoryPredictor(GRAVITY, GROUND_Y)
    
    # Runtime parameters
    runtime = sl.RuntimeParameters()
    runtime.confidence_threshold = 50
    
    # Allocate ZED mats
    image_left = sl.Mat()
    depth_map = sl.Mat()
    point_cloud = sl.Mat()
    
    print("\nStarting trajectory prediction...")
    print("Throw the ball to see landing prediction!\n")
    
    fps_q = deque(maxlen=30)
    
    try:
        while True:
            t0 = time.time()
            
            # Grab frame
            if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                continue
            
            current_time = time.time()
            
            # Retrieve images
            zed.retrieve_image(image_left, sl.VIEW.LEFT)
            zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH)
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
            
            frame = image_left.get_data()[:, :, :3].copy()
            
            # Detect ball
            detection = detector.detect(frame)
            
            current_pos = None
            
            if detection and not predictor.locked:
                x, y, radius = detection
                
                # Get 3D position
                world_pos, confidence = get_robust_depth(
                    depth_map, point_cloud, x, y, intrinsics
                )
                
                if world_pos is not None:
                    # Apply coordinate corrections
                    current_pos = world_pos.copy()
                    current_pos[1] += CAMERA_HEIGHT_ABOVE_GROUND
                    current_pos[2] += DEPTH_OFFSET
                    
                    # Update predictor
                    predictor.update(current_pos, current_time)
                    
                    # Send coordinates via UDP
                    if udp_sender and udp_sender.enabled:
                        land = predictor.get_land()
                        if args.send_prediction and land and predictor.locked:
                            udp_sender.send_coordinates(land.x, land.y, land.z)
                        else:
                            udp_sender.send_coordinates(
                                current_pos[0], current_pos[1], current_pos[2]
                            )
                    
                    # Draw detection
                    color = (0, 255, 0) if detector.last_method == 'YOLO' else (255, 255, 0)
                    cv.circle(frame, (x, y), radius, color, 2)
                    cv.circle(frame, (x, y), 3, color, -1)
                else:
                    cv.circle(frame, (x, y), radius, (0, 165, 255), 2)
            elif detection:
                x, y, radius = detection
                cv.circle(frame, (x, y), radius, (255, 0, 255), 2)
            
            # FPS
            fps_q.append(time.time() - t0)
            fps = 1.0 / (sum(fps_q) / len(fps_q))
            cv.putText(frame, f"FPS: {fps:.0f}", (frame.shape[1] - 80, 25),
                      cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Draw overlay
            draw_overlay(frame, predictor, current_pos, detector.last_method, 
                        intrinsics, udp_sender)
            
            # Display
            cv.imshow("YOLO Trajectory Predictor", frame)
            
            # Handle keyboard input
            key = cv.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('r'):
                predictor.reset()
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
        print("✓ Trajectory prediction stopped")
        print("=" * 60)


if __name__ == "__main__":
    main()
