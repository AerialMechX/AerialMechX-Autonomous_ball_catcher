#!/usr/bin/env python3
"""
Optimized Ball Catcher - STREAMING PREDICTIONS VERSION

Key change from previous version:
- Instead of waiting for a "locked" prediction, start sending predictions
  immediately after throw detection
- Continuously refine and send updated predictions as more data arrives
- Robot can start moving toward initial guess and correct course as predictions improve

UDP Packet Format (updated):
- Robot pose: 4 floats (x, y, z, yaw) - 16 bytes - unchanged
- Landing prediction: 5 floats (x, y, z, confidence, samples) - 20 bytes
  - confidence: 0.0-1.0 (higher = more reliable)
  - samples: number of trajectory samples used

The robot should:
1. Start moving toward prediction as soon as confidence > 0.3
2. Increase commitment as confidence grows
3. Final approach when confidence > 0.8
"""

import pyzed.sl as sl
import cv2 as cv
import numpy as np
import time
from datetime import datetime
from collections import deque
import random
import socket
import struct
from typing import Optional, Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor
import threading
from queue import Queue, Empty

# ===========================
# CONSTANTS - BALL / PHYSICS
# ===========================
G = 9.81
GROUND_Y = -0.52
MIN_SAMPLES = 3
RANSAC_ITERS = 50
RANSAC_INLIER_THRESH = 0.03
MAX_HISTORY = 60

# Velocity filters
VEL_WIN = 7
THROW_SPEED_THRESH = 0.80
THROW_UPWARD_VY_THRESH = 0.4
THROW_VERIFY_FRAMES = 3

# ===========================
# CAMERA CONFIG
# ===========================
INTR0_PATH = "./camera_parameters/camera0_intrinsics.dat"
INTR1_PATH = "./camera_parameters/camera1_intrinsics.dat"
EXTR1_PATH = "./camera_parameters/camera1_rot_trans.dat"

HSV_LOWER = np.array([0, 74, 0])
HSV_UPPER = np.array([94, 255, 255])

MIN_AREA = 80
MAX_AREA = 50000
MIN_CIRC = 0.35

# Relaxed detection for fast-moving ball (motion blur makes it less circular)
MIN_CIRC_FLIGHT = 0.15    # Much more lenient during flight
MAX_AREA_FLIGHT = 80000   # Motion blur can elongate the ball

SMOOTH_ALPHA = 0.4

# ===========================
# UDP CONFIG
# ===========================
DEFAULT_ROBOT_PORT = 5005
DEFAULT_ROBOT_IP = "192.168.0.248"
UDP_RATE_LIMIT_HZ = 60.0

# Use same port for predictions (robot listens on single port)
PREDICTION_PORT = 5005  # Same as DEFAULT_ROBOT_PORT - robot expects single port

# ===========================
# STREAMING PREDICTION CONFIG
# ===========================
MIN_CONFIDENCE_TO_SEND = 0.2      # Send predictions above this confidence
CONFIDENCE_LOCK_THRESHOLD = 0.8   # Consider "locked" above this
PREDICTION_SEND_RATE_HZ = 60.0    # How often to send prediction updates
MIN_SAMPLES_FOR_PREDICTION = 3    # Minimum samples before attempting prediction

# ===========================
# PERFORMANCE CONFIG
# ===========================
ARUCO_UPDATE_EVERY = 3
DETECT_SCALE = 0.5
SKIP_VIZ_DURING_FLIGHT = False  # Keep viz on for debugging streaming

# ===========================
# TRAJECTORY VISUALIZATION CONFIG
# ===========================
SHOW_TRAJECTORY_VIZ = True          # Master toggle for trajectory visualization
SHOW_REJECTED_TRAJECTORIES = True   # Show rejected RANSAC candidates in red
MAX_REJECTED_TO_SHOW = 5            # Limit rejected trajectories shown (performance)
TRAJECTORY_POINTS = 50              # Points to draw per trajectory curve
INLIER_POINT_COLOR = (255, 255, 0)  # Cyan (BGR)
OUTLIER_POINT_COLOR = (0, 165, 255) # Orange (BGR)
ACCEPTED_TRAJ_COLOR = (0, 255, 0)   # Green (BGR)
REJECTED_TRAJ_COLOR = (0, 0, 255)   # Red (BGR)
OBS_POINT_RADIUS = 5                # Radius for observation points


ROBOT_DISTANCE_OFFSET = 0.16


# ===========================
# UTIL: LOAD INTRINSICS / EXTRINSICS
# ===========================

def load_intrinsics_dat(path):
    with open(path, "r") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    K_rows, D_vals = [], []
    mode = None
    for line in lines:
        if line.startswith("intrinsic"):
            mode = "K"
            continue
        if line.startswith("distortion"):
            mode = "D"
            continue
        vals = [float(x) for x in line.replace(",", " ").split()]
        if mode == "K":
            K_rows.append(vals)
        else:
            D_vals.extend(vals)

    K = np.array(K_rows, dtype=np.float64)
    D = np.array(D_vals[:5], dtype=np.float64).reshape(-1, 1)
    return K, D


def load_extrinsics_dat(path):
    with open(path, "r") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    R_rows, T_vals = [], []
    mode = None
    for line in lines:
        if line.startswith("R"):
            mode = "R"
            continue
        if line.startswith("T"):
            mode = "T"
            continue
        vals = [float(x) for x in line.replace(",", " ").split()]
        if mode == "R":
            R_rows.append(vals)
        else:
            T_vals.extend(vals)

    R = np.array(R_rows, dtype=np.float64)
    T = np.array(T_vals, dtype=np.float64).reshape(3, 1)
    return R, T


# ===========================
# RANSAC CANDIDATE FOR VIZ
# ===========================

from dataclasses import dataclass

@dataclass
class RANSACCandidate:
    """Stores a RANSAC trajectory candidate for visualization."""
    params: Tuple[float, float, float, float, float, float]  # X0, Y0, Z0, Vx, Vy, Vz
    inlier_count: int
    total_count: int
    residual_mean: float
    is_accepted: bool = False
    
    @property
    def inlier_ratio(self) -> float:
        return self.inlier_count / max(1, self.total_count)


# ===========================
# TRAJECTORY VISUALIZATION HELPERS
# ===========================

def generate_trajectory_points(P: Tuple[float, ...], t_start: float, t_end: float, 
                                num_points: int = TRAJECTORY_POINTS) -> np.ndarray:
    """
    Generate 3D points along a ballistic trajectory.
    
    Args:
        P: (X0, Y0, Z0, Vx, Vy, Vz) trajectory parameters
        t_start: Start time (relative to trajectory start)
        t_end: End time
        num_points: Number of points to generate
        
    Returns:
        np.ndarray of shape (N, 3) with [X, Y, Z] positions
    """
    if P is None:
        return np.array([])
    
    X0, Y0, Z0, Vx, Vy, Vz = P
    times = np.linspace(t_start, t_end, num_points)
    
    points = []
    for t in times:
        x = X0 + Vx * t
        y = Y0 + Vy * t - 0.5 * G * t * t
        z = Z0 + Vz * t
        
        # Stop if below ground
        if y < GROUND_Y - 0.2:
            break
            
        points.append([x, y, z])
    
    return np.array(points) if points else np.array([])


def project_3d_to_pixel(point_3d: np.ndarray, stereo_model: 'StereoModel') -> Optional[Tuple[int, int]]:
    """
    Project a 3D point (in camera frame) to 2D pixel coordinates.
    
    Args:
        point_3d: [X, Y, Z] in camera coordinates (Y up, Z forward)
        stereo_model: StereoModel with camera intrinsics
        
    Returns:
        (u, v) pixel coordinates or None if behind camera
    """
    X, Y, Z = point_3d
    
    # Don't project points behind or too close to camera
    if Z <= 0.1:
        return None
    
    # Project using pinhole model
    # Note: In our coordinate system Y is up, but pixel Y increases downward
    u = stereo_model.fx * X / Z + stereo_model.cx
    v = -stereo_model.fy * Y / Z + stereo_model.cy  # Negative Y because Y-up -> pixel Y-down
    
    return (int(u), int(v))


def draw_trajectory_curve(frame: np.ndarray, trajectory_points: np.ndarray, 
                          stereo_model: 'StereoModel', color: Tuple[int, int, int],
                          thickness: int = 2, alpha: float = 1.0) -> None:
    """
    Draw a trajectory curve on the frame.
    
    Args:
        frame: Image to draw on
        trajectory_points: Array of 3D points (N, 3)
        stereo_model: For projection
        color: BGR color tuple
        thickness: Line thickness
        alpha: Opacity (1.0 = solid, 0.5 = semi-transparent)
    """
    if len(trajectory_points) < 2:
        return
    
    h, w = frame.shape[:2]
    pixels = []
    
    for pt in trajectory_points:
        px = project_3d_to_pixel(pt, stereo_model)
        if px is not None:
            u, v = px
            # Check if in frame bounds (with margin)
            if -100 < u < w + 100 and -100 < v < h + 100:
                pixels.append(px)
    
    if len(pixels) < 2:
        return
    
    # Draw the curve
    if alpha < 1.0:
        # Semi-transparent drawing using overlay
        overlay = frame.copy()
        for i in range(len(pixels) - 1):
            cv.line(overlay, pixels[i], pixels[i+1], color, thickness)
        cv.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    else:
        for i in range(len(pixels) - 1):
            cv.line(frame, pixels[i], pixels[i+1], color, thickness)


def draw_observation_points(frame: np.ndarray, positions: List[np.ndarray],
                            inlier_mask: Optional[np.ndarray],
                            stereo_model: 'StereoModel') -> None:
    """
    Draw observed trajectory points with inlier/outlier coloring.
    
    Args:
        frame: Image to draw on
        positions: List of 3D positions
        inlier_mask: Boolean mask (True = inlier)
        stereo_model: For projection
    """
    h, w = frame.shape[:2]
    
    for i, pos in enumerate(positions):
        px = project_3d_to_pixel(pos, stereo_model)
        if px is None:
            continue
            
        u, v = px
        if not (0 <= u < w and 0 <= v < h):
            continue
        
        # Color based on inlier/outlier status
        if inlier_mask is not None and i < len(inlier_mask):
            color = INLIER_POINT_COLOR if inlier_mask[i] else OUTLIER_POINT_COLOR
        else:
            color = (200, 200, 200)  # Gray if unknown
        
        cv.circle(frame, (u, v), OBS_POINT_RADIUS, color, -1)
        cv.circle(frame, (u, v), OBS_POINT_RADIUS, (0, 0, 0), 1)  # Black outline


def draw_landing_marker(frame: np.ndarray, landing_pos: np.ndarray,
                        stereo_model: 'StereoModel', is_locked: bool = False) -> None:
    """
    Draw an X marker at the predicted landing position.
    """
    px = project_3d_to_pixel(landing_pos, stereo_model)
    if px is None:
        return
        
    h, w = frame.shape[:2]
    u, v = px
    if not (0 <= u < w and 0 <= v < h):
        return
    
    # Draw X marker
    size = 15 if is_locked else 10
    color = (0, 255, 255) if is_locked else (0, 200, 200)  # Yellow/gold
    thickness = 3 if is_locked else 2
    
    cv.line(frame, (u - size, v - size), (u + size, v + size), color, thickness)
    cv.line(frame, (u - size, v + size), (u + size, v - size), color, thickness)
    
    # Add circle around locked prediction
    if is_locked:
        cv.circle(frame, (u, v), size + 5, color, 2)


# ===========================
# UDP SENDER (EXTENDED)
# ===========================

class UDPSender:
    def __init__(self, robot_ip: str = DEFAULT_ROBOT_IP, 
                 pose_port: int = DEFAULT_ROBOT_PORT,
                 prediction_port: int = PREDICTION_PORT,
                 rate_limit: Optional[float] = None):
        self.robot_ip = robot_ip
        self.pose_port = pose_port
        self.prediction_port = prediction_port
        
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)
        self.enabled = True
        self.packets_sent = 0
        self.predictions_sent = 0

        self.rate_limit = rate_limit
        self.last_send_time = 0.0
        self.min_send_interval = (1.0 / rate_limit) if rate_limit else 0.0
        
        # Separate rate limit for predictions (higher frequency)
        self.prediction_rate_limit = PREDICTION_SEND_RATE_HZ
        self.last_prediction_time = 0.0
        self.min_prediction_interval = 1.0 / PREDICTION_SEND_RATE_HZ

    def _rate_ok(self) -> bool:
        if not self.rate_limit:
            return True
        now = time.time()
        if (now - self.last_send_time) < self.min_send_interval:
            return False
        self.last_send_time = now
        return True

    def _prediction_rate_ok(self) -> bool:
        now = time.time()
        if (now - self.last_prediction_time) < self.min_prediction_interval:
            return False
        self.last_prediction_time = now
        return True

    def send_robot_pose(self, x: float, y: float, z: float, yaw: float) -> bool:
        """Send robot pose (4 floats: x, y, z, yaw)"""
        if not self.enabled or not self._rate_ok():
            return False
        try:
            packet = struct.pack('4f', x, y, z, yaw)
            self.sock.sendto(packet, (self.robot_ip, self.pose_port))
            self.packets_sent += 1
            return True
        except Exception:
            return False

    def send_predicted_landing(self, x: float, y: float, z: float, 
                               confidence: float = 1.0, 
                               num_samples: int = 0) -> bool:
        """
        Send predicted landing with confidence.
        
        Packet format: 5 floats (20 bytes)
        - x, y, z: landing coordinates
        - confidence: 0.0-1.0 reliability score
        - num_samples: number of trajectory points used (as float for packing)
        
        Robot should interpret:
        - confidence < 0.3: Very rough estimate, prepare to move
        - confidence 0.3-0.6: Initial estimate, start moving cautiously  
        - confidence 0.6-0.8: Good estimate, move with commitment
        - confidence > 0.8: High confidence, final approach
        """
        if not self.enabled:
            return False
        if not self._prediction_rate_ok():
            return False
        try:
            packet = struct.pack('5f', x, y, z, confidence, float(num_samples))
            self.sock.sendto(packet, (self.robot_ip, self.prediction_port))
            self.predictions_sent += 1
            return True
        except Exception:
            return False

    def send_no_prediction(self) -> bool:
        """Send zero prediction with zero confidence (no valid prediction)"""
        if not self.enabled:
            return False
        try:
            packet = struct.pack('5f', 0.0, 0.0, 0.0, 0.0, 0.0)
            self.sock.sendto(packet, (self.robot_ip, self.prediction_port))
            return True
        except Exception:
            return False

    def toggle(self) -> bool:
        self.enabled = not self.enabled
        print(f"[UDP] Sending {'ENABLED' if self.enabled else 'DISABLED'}")
        return self.enabled

    def close(self):
        self.sock.close()


# ===========================
# BALL DETECTOR
# ===========================

def find_ball_from_mask(mask: np.ndarray, min_radius: float = 3, max_radius: float = 100,
                        min_area: float = 50, min_circularity: float = 0.5) -> Optional[Tuple[int, int, int]]:
    """
    Given a binary mask, find the best ball candidate using contours + minEnclosingCircle.
    
    Uses area-based circularity scoring (from hsv_tuner.py approach):
    - circularity = contour_area / circle_area (ideally ~1.0 for perfect circle)
    - score = circularity * area (favors larger, more circular blobs)
    
    Args:
        mask: Binary mask image
        min_radius: Minimum ball radius in pixels
        max_radius: Maximum ball radius in pixels
        min_area: Minimum contour area to consider
        min_circularity: Minimum circularity ratio (0-1, higher = more strict)
        
    Returns:
        (x, y, radius) in image coords, or None if not found
    """
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    best_circle = None
    best_score = -1.0

    for cnt in contours:
        area = cv.contourArea(cnt)
        if area < min_area:
            continue

        (x, y), radius = cv.minEnclosingCircle(cnt)
        radius = float(radius)

        if radius < min_radius or radius > max_radius:
            continue

        circle_area = np.pi * (radius ** 2)
        if circle_area <= 0:
            continue

        # Area-based circularity: ratio of contour area to enclosing circle area
        circularity = area / circle_area  # ideally near 1 for perfect circle

        # Reject weird shapes (too elongated, cut-off, etc.)
        if circularity < min_circularity:
            continue

        # Score: favor larger + more circular blobs
        score = circularity * area
        if score > best_score:
            best_score = score
            best_circle = (int(x), int(y), int(radius))

    return best_circle


class HSVBallDetector:
    """
    Ball detector using HSV color filtering and the find_ball_from_mask() approach.
    
    Uses area-based circularity scoring which is more robust to:
    - Motion blur (ball appears elongated)
    - Partial occlusion
    - Lighting variations
    """
    
    def __init__(self, scale: float = 1.0):
        self.kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
        self.scale = scale
        self.scale_inv = 1.0 / scale if scale != 1.0 else 1.0
        
        # Scale-adjusted parameters
        self.min_area_scaled = MIN_AREA * (scale ** 2)
        self.max_area_scaled = MAX_AREA * (scale ** 2)
        self.max_area_flight_scaled = MAX_AREA_FLIGHT * (scale ** 2)
        
        # Radius limits (in scaled pixels)
        self.min_radius = 3 * scale
        self.max_radius = 150 * scale  # Reasonable max for a ball
        self.max_radius_flight = 200 * scale  # Larger during flight due to motion blur

    def detect(self, frame, flight_mode: bool = False):
        """
        Detect ball in frame using find_ball_from_mask() approach.
        
        Args:
            frame: BGR image
            flight_mode: If True, use relaxed parameters for motion-blurred ball
            
        Returns:
            ((cx, cy), radius) tuple or None if not found
        """
        # Optional downscaling for performance
        if self.scale != 1.0:
            small = cv.resize(frame, None, fx=self.scale, fy=self.scale,
                            interpolation=cv.INTER_LINEAR)
        else:
            small = frame

        # HSV color filtering
        hsv = cv.cvtColor(small, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, HSV_LOWER, HSV_UPPER)
        
        # Morphological cleanup
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, self.kernel)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, self.kernel)

        # Select parameters based on flight mode
        # Flight mode uses relaxed circularity for motion-blurred balls
        if flight_mode:
            min_circularity = MIN_CIRC_FLIGHT
            max_radius = self.max_radius_flight
        else:
            min_circularity = MIN_CIRC
            max_radius = self.max_radius

        # Use find_ball_from_mask for detection
        result = find_ball_from_mask(
            mask,
            min_radius=self.min_radius,
            max_radius=max_radius,
            min_area=self.min_area_scaled,
            min_circularity=min_circularity
        )

        if result is None:
            return None

        # Convert from (x, y, r) to ((cx, cy), radius) format
        # and scale back to original frame coordinates
        x, y, r = result
        cx = x * self.scale_inv
        cy = y * self.scale_inv
        radius = r * self.scale_inv

        return ((cx, cy), radius)


# ===========================
# STEREO MODEL
# ===========================

class StereoModel:
    def __init__(self, i0, i1, e1):
        self.K0, _ = load_intrinsics_dat(i0)
        self.K1, _ = load_intrinsics_dat(i1)
        self.R1, self.T1 = load_extrinsics_dat(e1)

        if np.linalg.norm(self.T1) > 1.0:
            self.T1 /= 100.0

        self.B = abs(self.T1[0, 0])
        self.fx = self.K0[0, 0]
        self.fy = self.K0[1, 1]
        self.cx = self.K0[0, 2]
        self.cy = self.K0[1, 2]

        print("[StereoModel] fx,fy,cx,cy =", self.fx, self.fy, self.cx, self.cy)
        print("[StereoModel] Baseline B =", self.B)


def triangulate_simple(uL, vL, uR, vR, M: StereoModel):
    d = (uL - uR)
    if d <= 0.5:
        return None

    Z = M.fx * M.B / d
    X = (uL - M.cx) * Z / M.fx
    Y = -(vL - M.cy) * Z / M.fy
    return np.array([X, Y, Z], float)


# ===========================
# TRAJECTORY ESTIMATOR (STREAMING)
# ===========================

class StreamingTrajectoryEstimator:
    """
    Trajectory estimator that provides predictions with confidence scores.
    Returns predictions as soon as MIN_SAMPLES are available, with
    confidence increasing as more samples are collected and fit quality improves.
    """
    
    def __init__(self):
        self.reset()

    def reset(self):
        self.t0 = None
        self.times = []
        self.pos = []
        self._t_array = None
        self._xyz_array = None
        self._dirty = True
        
        # Track prediction quality
        self.last_inlier_ratio = 0.0
        self.last_residual_mean = float('inf')
        self.predictions_made = 0
        
        # RANSAC visualization data
        self.ransac_candidates: List[RANSACCandidate] = []
        self.accepted_candidate: Optional[RANSACCandidate] = None
        self.last_inlier_mask: Optional[np.ndarray] = None

    def add(self, t_abs, p):
        if self.t0 is None:
            self.t0 = t_abs
        t = t_abs - self.t0
        self.times.append(t)
        self.pos.append(p.copy())
        self._dirty = True

        if len(self.times) > MAX_HISTORY:
            self.times.pop(0)
            self.pos.pop(0)

    def _update_arrays(self):
        if self._dirty and len(self.times) > 0:
            self._t_array = np.array(self.times)
            self._xyz_array = np.array(self.pos)
            self._dirty = False

    def _fit_ls_fast(self, t_sub, xyz_sub):
        n = len(t_sub)
        if n < 2:
            return None

        A = np.column_stack([np.ones(n), t_sub])
        t_sq = t_sub ** 2

        try:
            X0, Vx = np.linalg.lstsq(A, xyz_sub[:, 0], rcond=None)[0]
            Y_lin = xyz_sub[:, 1] + 0.5 * G * t_sq
            Y0, Vy = np.linalg.lstsq(A, Y_lin, rcond=None)[0]
            Z0, Vz = np.linalg.lstsq(A, xyz_sub[:, 2], rcond=None)[0]
            return (X0, Y0, Z0, Vx, Vy, Vz)
        except:
            return None

    def _compute_residuals(self, P, t, xyz):
        X0, Y0, Z0, Vx, Vy, Vz = P
        t_sq = t ** 2
        Xp = X0 + Vx * t
        Yp = Y0 + Vy * t - 0.5 * G * t_sq
        Zp = Z0 + Vz * t
        dx = Xp - xyz[:, 0]
        dy = Yp - xyz[:, 1]
        dz = Zp - xyz[:, 2]
        return np.sqrt(dx*dx + dy*dy + dz*dz)

    def estimate_with_confidence(self, store_candidates: bool = True):
        """
        Returns: (P, inlier_mask, confidence)
        - P: trajectory parameters or None
        - inlier_mask: boolean mask of inliers
        - confidence: 0.0-1.0 score
        
        If store_candidates=True, also stores RANSAC candidates for visualization.
        """
        n = len(self.times)
        if n < MIN_SAMPLES_FOR_PREDICTION:
            return None, None, 0.0

        self._update_arrays()
        t = self._t_array
        xyz = self._xyz_array

        bestP, bestIn, bestCount = None, None, -1
        idxAll = list(range(n))

        # Adaptive RANSAC iterations based on sample count
        iterations = min(RANSAC_ITERS, max(20, n * 5))
        
        # Clear previous candidates
        if store_candidates:
            self.ransac_candidates = []

        for _ in range(iterations):
            subset = random.sample(idxAll, min(3, n))
            t_sub = t[subset]
            xyz_sub = xyz[subset]
            
            P = self._fit_ls_fast(t_sub, xyz_sub)
            if P is None:
                continue

            r = self._compute_residuals(P, t, xyz)
            inl = r < RANSAC_INLIER_THRESH
            c = np.sum(inl)
            
            # Store candidate for visualization (sample some rejected ones)
            if store_candidates and len(self.ransac_candidates) < MAX_REJECTED_TO_SHOW + 1:
                candidate = RANSACCandidate(
                    params=P,
                    inlier_count=c,
                    total_count=n,
                    residual_mean=float(np.mean(r[inl])) if c > 0 else float('inf'),
                    is_accepted=False
                )
                self.ransac_candidates.append(candidate)
            
            if c > bestCount:
                bestCount = c
                bestP = P
                bestIn = inl

        if bestP is None or bestCount < MIN_SAMPLES_FOR_PREDICTION:
            return None, None, 0.0

        # Refit with all inliers
        finalIdx = np.where(bestIn)[0]
        P2 = self._fit_ls_fast(t[finalIdx], xyz[finalIdx])
        if P2 is not None:
            bestP = P2
            # Recompute residuals for final fit
            final_residuals = self._compute_residuals(bestP, t[finalIdx], xyz[finalIdx])
            self.last_residual_mean = np.mean(final_residuals)
        
        # Calculate confidence score
        self.last_inlier_ratio = bestCount / n
        confidence = self._calculate_confidence(n, bestCount, self.last_residual_mean)
        
        # Store the accepted candidate and inlier mask for visualization
        if store_candidates:
            self.accepted_candidate = RANSACCandidate(
                params=bestP,
                inlier_count=bestCount,
                total_count=n,
                residual_mean=self.last_residual_mean,
                is_accepted=True
            )
            self.last_inlier_mask = bestIn
        
        self.predictions_made += 1
        return bestP, bestIn, confidence

    def _calculate_confidence(self, num_samples: int, num_inliers: int, 
                             mean_residual: float) -> float:
        """
        Calculate confidence score based on multiple factors:
        1. Number of samples (more = better, up to ~15)
        2. Inlier ratio (higher = better)
        3. Fit quality / residual (lower = better)
        """
        # Sample count factor: ramps up from 0.3 at 3 samples to 1.0 at 15 samples
        sample_factor = min(1.0, 0.3 + 0.7 * (num_samples - 3) / 12)
        
        # Inlier ratio factor
        inlier_ratio = num_inliers / num_samples
        inlier_factor = inlier_ratio  # 0.0 to 1.0
        
        # Residual factor: lower residuals = higher confidence
        # Typical good residual is < 0.02m, bad is > 0.05m
        if mean_residual < 0.01:
            residual_factor = 1.0
        elif mean_residual < 0.03:
            residual_factor = 0.8
        elif mean_residual < 0.05:
            residual_factor = 0.5
        else:
            residual_factor = 0.3
        
        # Combine factors (weighted geometric mean)
        confidence = (sample_factor ** 0.4) * (inlier_factor ** 0.3) * (residual_factor ** 0.3)
        
        return min(1.0, max(0.0, confidence))

    def _solve_t_land(self, P):
        X0, Y0, Z0, Vx, Vy, Vz = P
        a = -0.5 * G
        b = Vy
        c = Y0 - GROUND_Y
        D = b * b - 4 * a * c
        if D < 0:
            return None
        sqrt_D = np.sqrt(D)
        r1 = (-b + sqrt_D) / (2 * a)
        r2 = (-b - sqrt_D) / (2 * a)
        cand = [t for t in (r1, r2) if t > 0]
        if not cand:
            return None
        return min(cand)

    def landing_point_with_confidence(self):
        """
        Returns: (landing_pos, P, tL, confidence, num_samples)
        All can be None/0 if no valid prediction
        """
        n = len(self.times)
        P, inl, confidence = self.estimate_with_confidence()
        
        if P is None:
            return None, None, None, 0.0, n

        tL = self._solve_t_land(P)
        if tL is None:
            # Can't solve landing time - trajectory might not intersect ground
            # Return low confidence
            return None, P, None, confidence * 0.3, n

        X0, Y0, Z0, Vx, Vy, Vz = P
        land = np.array([X0 + Vx * tL, GROUND_Y, Z0 + Vz * tL])
        
        return land, P, tL, confidence, n

    def get_sample_count(self) -> int:
        return len(self.times)

    def get_visualization_data(self) -> Dict:
        """
        Get data needed for trajectory visualization.
        
        Returns:
            dict with:
            - positions: list of observed 3D positions
            - inlier_mask: boolean mask for inliers
            - accepted_params: trajectory params for accepted fit
            - rejected_candidates: list of RANSACCandidate for rejected fits
            - t_end: time to use for trajectory endpoint (landing time or current)
        """
        tL = None
        if self.accepted_candidate and self.accepted_candidate.params:
            tL = self._solve_t_land(self.accepted_candidate.params)
        
        # Get time range for visualization
        t_end = tL if tL is not None and tL > 0 else (max(self.times) * 1.5 if self.times else 1.0)
        
        return {
            'positions': list(self.pos),
            'inlier_mask': self.last_inlier_mask,
            'accepted_params': self.accepted_candidate.params if self.accepted_candidate else None,
            'rejected_candidates': [c for c in self.ransac_candidates if not c.is_accepted][:MAX_REJECTED_TO_SHOW],
            't_end': t_end,
            't_start': 0.0,
        }


# ===========================
# ASYNC STREAMING TRAJECTORY ESTIMATOR
# ===========================

class AsyncStreamingEstimator:
    """
    Non-blocking wrapper that continuously computes predictions
    and makes the latest result available.
    """
    
    def __init__(self, base_estimator: StreamingTrajectoryEstimator):
        self.estimator = base_estimator
        self.result_queue = Queue(maxsize=1)
        self.last_result = (None, None, None, 0.0, 0)
        self._lock = threading.Lock()
        self._new_data = threading.Event()
        
        # Start thread AFTER all attributes are initialized
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        while self.running:
            # Wait for new data or timeout
            self._new_data.wait(timeout=0.02)
            self._new_data.clear()
            
            # Compute prediction
            with self._lock:
                if self.estimator.get_sample_count() >= MIN_SAMPLES_FOR_PREDICTION:
                    result = self.estimator.landing_point_with_confidence()
                else:
                    result = (None, None, None, 0.0, self.estimator.get_sample_count())
            
            # Update result (replace old)
            try:
                self.result_queue.get_nowait()
            except Empty:
                pass
            self.result_queue.put(result)

    def add(self, t, p):
        with self._lock:
            self.estimator.add(t, p)
        self._new_data.set()  # Signal new data available

    def get_result(self):
        """Non-blocking check for latest result"""
        try:
            self.last_result = self.result_queue.get_nowait()
        except Empty:
            pass
        return self.last_result

    def reset(self):
        with self._lock:
            self.estimator.reset()
        self.last_result = (None, None, None, 0.0, 0)
        try:
            self.result_queue.get_nowait()
        except Empty:
            pass

    def get_sample_count(self) -> int:
        with self._lock:
            return self.estimator.get_sample_count()

    def get_visualization_data(self) -> Dict:
        """Get visualization data from underlying estimator (thread-safe)."""
        with self._lock:
            return self.estimator.get_visualization_data()

    def stop(self):
        self.running = False
        self._new_data.set()  # Wake up thread
        self.thread.join(timeout=1.0)


# ===========================
# ARUCO DETECTION (unchanged from optimized version)
# ===========================

class ArucoCamera:
    def __init__(self, K: np.ndarray, D: np.ndarray):
        self.intrinsic = K
        self.distortion = D.reshape(-1, 1).astype(np.float64)


def create_aruco_detectors() -> Dict[str, cv.aruco.ArucoDetector]:
    detectors = {}
    dict_types = {
        'left':  cv.aruco.DICT_4X4_50,
        'front': cv.aruco.DICT_5X5_50,
        'right': cv.aruco.DICT_6X6_50,
        'back':  cv.aruco.DICT_7X7_50
    }
    for side_name, dict_type in dict_types.items():
        dictionary = cv.aruco.getPredefinedDictionary(dict_type)
        params = cv.aruco.DetectorParameters()
        params.adaptiveThreshWinSizeMin = 5
        params.adaptiveThreshWinSizeMax = 21
        params.adaptiveThreshWinSizeStep = 8
        params.cornerRefinementMethod = cv.aruco.CORNER_REFINE_SUBPIX
        params.cornerRefinementMaxIterations = 20
        detectors[side_name] = cv.aruco.ArucoDetector(dictionary, params)
    return detectors


def detect_aruco_corners_fast(frame, detectors):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    detected = {}
    for side_name in ['front', 'back', 'left', 'right']:
        corners, ids, _ = detectors[side_name].detectMarkers(gray)
        if ids is not None:
            for i, marker_id in enumerate(ids.flatten()):
                if marker_id == 0:
                    detected[side_name] = corners[i].reshape(4, 2)
                    break
    return detected


def get_marker_model_points(marker_size):
    half = marker_size / 2.0
    return np.array([[-half, half, 0.0], [half, half, 0.0],
                     [half, -half, 0.0], [-half, -half, 0.0]], dtype=np.float64)


def heading_yaw_from_marker(side_name, R_cam):
    n_cam = R_cam[:, 2].astype(np.float64)
    n_norm = np.linalg.norm(n_cam)
    if n_norm < 1e-8:
        return 0.0
    n_cam /= n_norm
    if side_name == 'back':
        H_cam = -n_cam
    elif side_name == 'front':
        H_cam = n_cam
    elif side_name == 'right':
        H_cam = np.array([-n_cam[2], n_cam[1], n_cam[0]])
    elif side_name == 'left':
        H_cam = np.array([n_cam[2], n_cam[1], -n_cam[0]])
    else:
        H_cam = -n_cam
    H_xz = np.array([H_cam[0], H_cam[2]], dtype=np.float64)
    h_norm = np.linalg.norm(H_xz)
    if h_norm < 1e-8:
        return 0.0
    H_xz /= h_norm
    return float(np.degrees(np.arctan2(H_xz[0], H_xz[1])))


def pose_from_aruco_pnp(side_name, corners_px, camera, marker_size_m):
    corners_input = corners_px.reshape(1, 1, 4, 2).astype(np.float32)
    rvecs, tvecs, _ = cv.aruco.estimatePoseSingleMarkers(
        corners_input, marker_size_m, camera.intrinsic, camera.distortion)
    rvec, tvec = rvecs[0, 0, :], tvecs[0, 0, :]
    R_cam, _ = cv.Rodrigues(rvec)
    t_cam = tvec.reshape(3, 1)
    yaw_deg = heading_yaw_from_marker(side_name, R_cam)
    S = np.diag([1.0, -1.0, 1.0])
    R_user, t_user = S @ R_cam, S @ t_cam
    obj_pts = get_marker_model_points(marker_size_m)
    corners_3d_user = np.array([(R_user @ p.reshape(3,1) + t_user).ravel() for p in obj_pts])
    return R_user, t_user.ravel(), corners_3d_user, yaw_deg


class DetectedMarker:
    __slots__ = ['side_name', 'corners_left', 'corners_right', 
                 'corners_3d', 'position', 'yaw_angle', 'rotation_matrix']
    def __init__(self, side_name, corners_left, corners_right,
                 corners_3d, position, yaw_angle, rotation_matrix):
        self.side_name = side_name
        self.corners_left = corners_left
        self.corners_right = corners_right
        self.corners_3d = corners_3d
        self.position = position
        self.yaw_angle = yaw_angle
        self.rotation_matrix = rotation_matrix


def process_aruco_frame(left_frame, right_frame, detectors, camera, marker_size_m):
    corners_left = detect_aruco_corners_fast(left_frame, detectors)
    corners_right = detect_aruco_corners_fast(right_frame, detectors)
    markers = []
    for side_name, pts_left in corners_left.items():
        R_user, t_user, corners_3d_user, yaw_deg = pose_from_aruco_pnp(
            side_name, pts_left, camera, marker_size_m)
        pts_right = corners_right.get(side_name, np.zeros((4, 2)))
        markers.append(DetectedMarker(side_name, pts_left, pts_right,
                                      corners_3d_user, t_user, yaw_deg, R_user))
    return markers


def compute_robot_pose(markers):
    if not markers:
        return None
    positions = np.array([m.position for m in markers])
    avg_position = np.mean(positions, axis=0)
    priority = {'back': 0, 'front': 1, 'left': 2, 'right': 3}
    best = min(markers, key=lambda m: priority.get(m.side_name, 99))
    return avg_position, best.yaw_angle, markers


# ===========================
# PARALLEL BALL DETECTION
# ===========================

class ParallelBallDetector:
    def __init__(self, scale: float = 1.0, max_workers: int = 2):
        self.detector = HSVBallDetector(scale=scale)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    def detect_stereo(self, frameL, frameR, flight_mode: bool = False):
        """Detect ball in both frames, with optional relaxed parameters for flight."""
        futureL = self.executor.submit(self.detector.detect, frameL, flight_mode)
        futureR = self.executor.submit(self.detector.detect, frameR, flight_mode)
        detL = futureL.result()
        detR = futureR.result() if detL else None
        return detL, detR
    
    def shutdown(self):
        self.executor.shutdown(wait=False)


# ===========================
# CONFIDENCE VISUALIZATION
# ===========================

def get_confidence_color(confidence: float) -> Tuple[int, int, int]:
    """Return BGR color based on confidence level."""
    if confidence < 0.3:
        return (0, 0, 255)      # Red - low confidence
    elif confidence < 0.5:
        return (0, 165, 255)    # Orange - medium-low
    elif confidence < 0.7:
        return (0, 255, 255)    # Yellow - medium
    elif confidence < 0.85:
        return (0, 255, 0)      # Green - good
    else:
        return (255, 0, 0)      # Blue - excellent (locked)


def draw_confidence_bar(frame, confidence: float, x: int, y: int, 
                        width: int = 200, height: int = 20):
    """Draw a confidence bar on the frame."""
    # Background
    cv.rectangle(frame, (x, y), (x + width, y + height), (50, 50, 50), -1)
    
    # Filled portion
    fill_width = int(width * confidence)
    color = get_confidence_color(confidence)
    if fill_width > 0:
        cv.rectangle(frame, (x, y), (x + fill_width, y + height), color, -1)
    
    # Border
    cv.rectangle(frame, (x, y), (x + width, y + height), (255, 255, 255), 1)
    
    # Text
    cv.putText(frame, f"{confidence*100:.0f}%", (x + width + 10, y + height - 5),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


# ===========================
# MAIN LOOP (STREAMING PREDICTIONS)
# ===========================

def main():
    cv.setUseOptimized(True)
    try:
        cv.setNumThreads(cv.getNumberOfCPUs())
    except:
        pass

    print("\n=== STREAMING PREDICTIONS BALL CATCHER ===")
    print(f"Min confidence to send: {MIN_CONFIDENCE_TO_SEND}")
    print(f"Lock threshold: {CONFIDENCE_LOCK_THRESHOLD}")
    print(f"Prediction rate: {PREDICTION_SEND_RATE_HZ} Hz")
    print(f"Detection scale: {DETECT_SCALE}")
    print()

    # Init modules
    parallel_detector = ParallelBallDetector(scale=DETECT_SCALE, max_workers=2)
    stereo_model = StereoModel(INTR0_PATH, INTR1_PATH, EXTR1_PATH)
    
    # Streaming trajectory estimator
    base_traj_est = StreamingTrajectoryEstimator()
    async_traj = AsyncStreamingEstimator(base_traj_est)

    # ArUco
    K0, D0 = load_intrinsics_dat(INTR0_PATH)
    aruco_cam = ArucoCamera(K0, D0)
    detectors = create_aruco_detectors()
    marker_size_m = 0.095

    # UDP (note: can use same port for both if robot expects it)
    udp = UDPSender(robot_ip=DEFAULT_ROBOT_IP,
                    pose_port=DEFAULT_ROBOT_PORT,
                    prediction_port=PREDICTION_PORT,
                    rate_limit=UDP_RATE_LIMIT_HZ)

    # ZED
    zed = sl.Camera()
    ip = sl.InitParameters()
    ip.camera_resolution = sl.RESOLUTION.HD720
    ip.camera_fps = 60
    ip.depth_mode = sl.DEPTH_MODE.NONE
    ip.coordinate_units = sl.UNIT.METER
    
    if zed.open(ip) != sl.ERROR_CODE.SUCCESS:
        print("ZED open FAILED.")
        return

    zed.set_camera_settings(sl.VIDEO_SETTINGS.AEC_AGC, 1)
    zed.set_camera_settings(sl.VIDEO_SETTINGS.WHITEBALANCE_TEMPERATURE, -1)

    left_mat = sl.Mat()
    right_mat = sl.Mat()

    # State
    last_smooth = None
    state = "IDLE"
    vel_buffer = deque(maxlen=VEL_WIN)
    last_raw = None
    last_time = None
    throw_verify = 0

    # Streaming prediction state
    current_prediction = None
    current_confidence = 0.0
    current_tL = None
    is_locked = False
    first_prediction = None  # Track first prediction for comparison
    throw_start_time = None  # Track when throw was detected
    flight_complete = False  # Flag to stop after first landing
    throw_detected = False  # Flag to only allow a single throw until reset
    
    frame_count = 0
    last_loop_time = None
    fps = 0.0

    # Robot pose cache
    aruco_frame_counter = 0
    last_robot_pose = None
    last_markers = []
    markers_currently_visible = False  # Track if markers are detected NOW
    
    # Debug: track detection success during flight
    flight_detections = 0
    flight_misses = 0
    
    # Visualization state (local vars that can be toggled)
    show_traj_viz = SHOW_TRAJECTORY_VIZ
    show_rejected = SHOW_REJECTED_TRAJECTORIES

    print("\nControls: 'q'=Quit, 'r'=Reset, 'u'=UDP toggle, 'v'=Toggle traj viz, 'x'=Toggle rejected traj\n")

    try:
        while True:
            now_loop = time.time()
            if last_loop_time is not None:
                dt_loop = now_loop - last_loop_time
                if dt_loop > 0:
                    fps = 1.0 / dt_loop
            last_loop_time = now_loop

            if zed.grab() != sl.ERROR_CODE.SUCCESS:
                continue

            zed.retrieve_image(left_mat, sl.VIEW.LEFT)
            zed.retrieve_image(right_mat, sl.VIEW.RIGHT)
            frameL = np.ascontiguousarray(left_mat.get_data()[:, :, :3])
            frameR = np.ascontiguousarray(right_mat.get_data()[:, :, :3])

            # Ball detection - use relaxed parameters during flight
            flight_mode = state in ["THROWN", "TRACKING"]
            detL, detR = parallel_detector.detect_stereo(frameL, frameR, flight_mode)

            # Check if flight is complete (ball should have landed)
            if is_locked and throw_start_time is not None and current_tL is not None:
                time_since_throw = now_loop - throw_start_time
                # Add small buffer (0.1s) to account for timing
                if time_since_throw > (current_tL + 0.1) and not flight_complete:
                    flight_complete = True
                    total_flight_frames = flight_detections + flight_misses
                    detection_rate = (flight_detections / total_flight_frames * 100) if total_flight_frames > 0 else 0
                    print(f"\n{'='*60}")
                    print(f"[FLIGHT COMPLETE] Ball should have landed")
                    print(f"  Time since throw: {time_since_throw:.3f}s")
                    print(f"  Predicted flight time: {current_tL:.3f}s")
                    if current_prediction is not None:
                        print(f"  Final landing: X={current_prediction[0]:+.3f} Z={current_prediction[2]:+.3f}")
                    print(f"  Detection stats: {flight_detections}/{total_flight_frames} frames ({detection_rate:.0f}%)")
                    if detection_rate < 50:
                        print(f"  WARNING: Low detection rate! Ball may have been missed during flight.")
                    print(f"{'='*60}\n")

            speed = 0.0
            vy = 0.0
            throw_cond = False

            if detL and detR:
                # Track detection during flight
                if state in ["THROWN", "TRACKING"]:
                    flight_detections += 1
                    
                (uL, vL), rL = detL
                (uR, vR), rR = detR
                raw = triangulate_simple(uL, vL, uR, vR, stereo_model)
                
                if raw is not None:
                    if last_smooth is None:
                        smooth = raw
                    else:
                        smooth = SMOOTH_ALPHA * raw + (1 - SMOOTH_ALPHA) * last_smooth
                    last_smooth = smooth

                    now = now_loop
                    if last_raw is not None and last_time is not None:
                        dt = now - last_time
                        if dt > 0:
                            v = (raw - last_raw) / dt
                            vel_buffer.append(v)

                    last_raw = raw
                    last_time = now

                    v_est = np.mean(vel_buffer, axis=0) if len(vel_buffer) > 0 else np.zeros(3)
                    speed = float(np.linalg.norm(v_est))
                    vy = float(v_est[1])

                    # State machine
                    if state == "IDLE":
                        state = "HOLD"
                        throw_verify = 0

                    elif state == "HOLD":
                        async_traj.reset()
                        current_prediction = None
                        current_confidence = 0.0
                        current_tL = None
                        is_locked = False
                        first_prediction = None  # Reset for new throw
                        throw_start_time = None
                        flight_complete = False
                        flight_detections = 0
                        flight_misses = 0

                        # Only allow throw detection if no throw has been detected yet
                        if not throw_detected:
                            throw_cond = (speed > THROW_SPEED_THRESH and
                                          vy > abs(THROW_UPWARD_VY_THRESH))
                            if throw_cond:
                                throw_verify += 1
                            else:
                                throw_verify = 0

                            if throw_verify >= THROW_VERIFY_FRAMES:
                                tic = time.time()
                                print(f"[THROW DETECTED] speed={speed:.2f}, vy={vy:.2f}, fps={fps:.1f}")
                                print(f"[INFO] Next throw will only be allowed after pressing 'r' to reset")
                                throw_detected = True  # Lock throw detection
                                state = "THROWN"
                                throw_start_time = now  # Record throw time
                                vel_buffer.clear()
                                async_traj.reset()
                                async_traj.add(now, smooth)
                                throw_verify = 0

                    elif state == "THROWN":
                        # Stop tracking if flight is complete
                        if flight_complete:
                            state = "LANDED"
                            continue
                            
                        # Add sample to trajectory
                        async_traj.add(now_loop, smooth)
                        
                        # Get latest prediction (non-blocking)
                        land, P, tL, confidence, num_samples = async_traj.get_result()
                        
                        if land is not None:
                            current_prediction = land
                            current_confidence = confidence
                            current_tL = tL
                            
                            # Check if we've reached lock threshold
                            if confidence >= CONFIDENCE_LOCK_THRESHOLD and not is_locked:
                                is_locked = True
                                print(f"\n{'='*60}")
                                print(f"[LOCKED] conf={confidence:.0%}, samples={num_samples}")
                                print(f"         Final: X={land[0]:+.3f} Z={land[2]:+.3f} tL={tL:.3f}s")
                                if first_prediction is not None:
                                    drift_x = land[0] - first_prediction[0]
                                    drift_z = land[2] - first_prediction[2]
                                    drift_total = np.sqrt(drift_x**2 + drift_z**2)
                                    print(f"         First: X={first_prediction[0]:+.3f} Z={first_prediction[2]:+.3f}")
                                    print(f"         Drift: dX={drift_x:+.3f} dZ={drift_z:+.3f} total={drift_total:.3f}m")
                                print(f"{'='*60}\n")
                            
                            # SEND PREDICTION if confidence above threshold
                            if confidence >= MIN_CONFIDENCE_TO_SEND:
                                sent = udp.send_predicted_landing(
                                    float(land[0]), float(land[1]), float(land[2]),
                                    float(confidence), num_samples
                                )
                                print(f"current time: {datetime.fromtimestamp(time.time())}")
                                if sent:
                                    # Track first prediction
                                    if first_prediction is None:
                                        first_prediction = land.copy()
                                        print(f"\n[PRED #{udp.predictions_sent:3d}] *** FIRST ESTIMATE ***")
                                        print(f"         X={land[0]:+.3f} Z={land[2]:+.3f} "
                                              f"conf={confidence:.0%} samples={num_samples} time={time.time() - tic}")
                                    else:
                                        print(f"[PRED #{udp.predictions_sent:3d}] "
                                              f"X={land[0]:+.3f} Z={land[2]:+.3f} "
                                              f"conf={confidence:.0%} samples={num_samples} time={time.time() - tic}")
                        
                        # Move to TRACKING state if locked (but keep updating)
                        if is_locked:
                            state = "TRACKING"

                    elif state == "TRACKING":
                        # Stop tracking if flight is complete
                        if flight_complete:
                            state = "LANDED"
                            continue
                            
                        # Continue tracking and refining even after lock
                        async_traj.add(now_loop, smooth)
                        
                        land, P, tL, confidence, num_samples = async_traj.get_result()
                        
                        if land is not None:
                            current_prediction = land
                            current_confidence = confidence
                            current_tL = tL
                            
                            # Keep sending updates
                            sent = udp.send_predicted_landing(
                                float(land[0]), float(land[1]), float(land[2]),
                                float(confidence), num_samples
                            )
                            if sent:
                                print(f"[PRED #{udp.predictions_sent:3d}] "
                                      f"X={land[0]:+.3f} Z={land[2]:+.3f} "
                                      f"conf={confidence:.0%} samples={num_samples} [TRACKING]")

                    elif state == "LANDED":
                        # Ball has landed - don't track anymore, wait for reset
                        pass

                    # Visualization
                    Xb, Yb, Zb = smooth
                    cv.circle(frameL, (int(uL), int(vL)), int(rL), (0, 255, 0), 2)
                    cv.putText(frameL, f"BALL X={Xb:.2f} Y={Yb:.2f} Z={Zb:.2f}",
                               (20, 40), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv.putText(frameL, f"speed={speed:.2f} vy={vy:.2f} STATE={state}",
                               (20, 70), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                    # Show prediction info
                    if current_prediction is not None:
                        LX, LY, LZ = current_prediction
                        tL_txt = f"{current_tL:.2f}" if current_tL is not None else "?"
                        color = get_confidence_color(current_confidence)
                        lock_txt = " [LOCKED]" if is_locked else ""
                        cv.putText(frameL, 
                                   f"PRED X={LX:.2f} Z={LZ:.2f} t={tL_txt}s{lock_txt}",
                                   (20, 100), cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        
                        # Confidence bar
                        draw_confidence_bar(frameL, current_confidence, 20, 130, 200, 20)
                        
                        # Sample count
                        samples = async_traj.get_sample_count()
                        cv.putText(frameL, f"Samples: {samples}",
                                   (240, 145), cv.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                        
                        # ========================================
                        # TRAJECTORY VISUALIZATION
                        # ========================================
                        if show_traj_viz and state in ["THROWN", "TRACKING", "LANDED"]:
                            viz_data = async_traj.get_visualization_data()
                            
                            # Draw rejected trajectories in RED (if enabled)
                            if show_rejected and viz_data['rejected_candidates']:
                                for candidate in viz_data['rejected_candidates']:
                                    if candidate.params:
                                        # Generate trajectory curve for this rejected candidate
                                        tL_rej = base_traj_est._solve_t_land(candidate.params) if hasattr(base_traj_est, '_solve_t_land') else None
                                        t_end_rej = tL_rej if tL_rej and tL_rej > 0 else viz_data['t_end']
                                        rej_pts = generate_trajectory_points(
                                            candidate.params, 0.0, t_end_rej, TRAJECTORY_POINTS // 2
                                        )
                                        if len(rej_pts) > 0:
                                            draw_trajectory_curve(frameL, rej_pts, stereo_model, 
                                                                REJECTED_TRAJ_COLOR, thickness=1, alpha=0.5)
                            
                            # Draw accepted trajectory in GREEN
                            if viz_data['accepted_params']:
                                accepted_pts = generate_trajectory_points(
                                    viz_data['accepted_params'], 
                                    viz_data['t_start'], 
                                    viz_data['t_end'],
                                    TRAJECTORY_POINTS
                                )
                                if len(accepted_pts) > 0:
                                    draw_trajectory_curve(frameL, accepted_pts, stereo_model,
                                                        ACCEPTED_TRAJ_COLOR, thickness=2, alpha=1.0)
                            
                            # Draw observation points (inliers=cyan, outliers=orange)
                            if viz_data['positions'] and len(viz_data['positions']) > 0:
                                draw_observation_points(frameL, viz_data['positions'],
                                                       viz_data['inlier_mask'], stereo_model)
                            
                            # Draw landing marker
                            if current_prediction is not None:
                                draw_landing_marker(frameL, current_prediction, stereo_model, is_locked)

                else:
                    cv.putText(frameL, "BAD TRIANG", (20, 40),
                               cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                # Track missed detections during flight
                if state in ["THROWN", "TRACKING"]:
                    flight_misses += 1
                    
                # No ball detected
                if state not in ["TRACKING", "LANDED"]:  # Don't reset if tracking or landed
                    state = "IDLE"
                    async_traj.reset()
                    vel_buffer.clear()
                    last_smooth = None
                    last_raw = None
                    throw_verify = 0
                    current_prediction = None
                    current_confidence = 0.0
                    current_tL = None
                    is_locked = False
                    first_prediction = None  # Reset for new throw
                    throw_start_time = None
                    flight_complete = False
                    flight_detections = 0
                    flight_misses = 0

                cv.putText(frameL, "BALL SEARCHING...", (20, 40),
                           cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Show last prediction if we were tracking or landed
                if current_prediction is not None and (is_locked or state == "LANDED"):
                    LX, LY, LZ = current_prediction
                    tL_txt = f"{current_tL:.2f}" if current_tL is not None else "?"
                    status = "[LANDED]" if state == "LANDED" else "[LOCKED]"
                    cv.putText(frameL, f"FINAL PRED X={LX:.2f} Z={LZ:.2f} t={tL_txt}s {status}",
                               (20, 70), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

            # ArUco (always run, even during flight - robot needs pose updates)
            aruco_frame_counter += 1
            if aruco_frame_counter % ARUCO_UPDATE_EVERY == 0:
                markers = process_aruco_frame(frameL, frameR, detectors, aruco_cam, marker_size_m)
                last_markers = markers
                if markers:
                    last_robot_pose = compute_robot_pose(markers)
                    markers_currently_visible = True
                else:
                    markers_currently_visible = False

            robot_pose = last_robot_pose
            robot_pose_text = "Robot: NO MARKER"
            if robot_pose is not None and markers_currently_visible:
                pos, yaw_deg, det_list = robot_pose
                rx, ry, rz = pos
                dist = float(np.sqrt(rx*rx + rz*rz))
                robot_pose_text = f"Robot X={rx:+.2f} Y={ry:+.2f} Z={rz - ROBOT_DISTANCE_OFFSET:+.2f} yaw={yaw_deg:+.1f}"

                # Draw markers
                for m in det_list:
                    colors = {'left': (255,0,0), 'front': (0,255,0), 
                              'right': (0,0,255), 'back': (255,255,0)}
                    color = colors.get(m.side_name, (255,255,255))
                    cL = m.corners_left.astype(int)
                    for i in range(4):
                        cv.line(frameL, tuple(cL[i]), tuple(cL[(i+1)%4]), color, 2)

                # Only send robot pose when markers are currently visible
                udp.send_robot_pose(float(rx), float(ry), float(rz) - ROBOT_DISTANCE_OFFSET, float(yaw_deg))

            cv.putText(frameL, robot_pose_text, (20, 170),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv.putText(frameL,
                       f"UDP {'ON' if udp.enabled else 'OFF'} pose:{udp.packets_sent} pred:{udp.predictions_sent}",
                       (20, 195), cv.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
            cv.putText(frameL, f"FPS={fps:.1f}", (20, 220),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Trajectory visualization status
            viz_status = f"TrajViz:{'ON' if show_traj_viz else 'OFF'} Rejected:{'ON' if show_rejected else 'OFF'}"
            cv.putText(frameL, viz_status, (20, 245),
                       cv.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

            cv.imshow("UnifiedView", frameL)
            key = cv.waitKey(1) & 0xFF

            if key == 27 or key == ord('q'):
                break
            elif key == ord('r'):
                async_traj.reset()
                current_prediction = None
                current_confidence = 0.0
                current_tL = None
                is_locked = False
                first_prediction = None  # Reset first prediction tracking
                throw_start_time = None
                flight_complete = False
                flight_detections = 0
                flight_misses = 0
                throw_detected = False  # Allow new throw detection
                vel_buffer.clear()
                state = "HOLD"
                throw_verify = 0
                last_smooth = None
                last_raw = None
                last_time = None
                print("[RESET] Ready for new throw detection")
            elif key == ord('u'):
                udp.toggle()
            elif key == ord('v'):
                # Toggle trajectory visualization
                show_traj_viz = not show_traj_viz
                print(f"[VIZ] Trajectory visualization {'ON' if show_traj_viz else 'OFF'}")
            elif key == ord('x'):
                # Toggle rejected trajectories
                show_rejected = not show_rejected
                print(f"[VIZ] Rejected trajectories {'ON' if show_rejected else 'OFF'}")

            frame_count += 1

    finally:
        print(f"\n[STATS] Pose packets: {udp.packets_sent}, Prediction packets: {udp.predictions_sent}")
        async_traj.stop()
        parallel_detector.shutdown()
        udp.close()
        zed.close()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()
