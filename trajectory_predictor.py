"""
Trajectory Predictor for Ball Landing Estimation

This script predicts the landing coordinates of a thrown ball using:
- RANSAC for robust outlier rejection
- Quadratic regression for ballistic trajectory fitting
- Kalman filtering for smooth position tracking

Designed to work with ball_tracker_params.py for ZED camera-based ball tracking.

Usage:
    python trajectory_predictor.py                    # Run with ZED camera
    python trajectory_predictor.py --debug            # Enable visualization
    python trajectory_predictor.py --robot-ip 192.168.0.51  # Set robot IP

Based on the approach from dhrumilp15/mind-quidditch
"""

import numpy as np
import cv2 as cv
import pyzed.sl as sl
import time
import os
import argparse
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from filterpy.kalman import KalmanFilter

# Import from ball_tracker_params
from ball_tracker_params import (
    BallDetector, CameraIntrinsics, 
    get_robust_depth, draw_detection,
    RESOLUTION, FPS, DEPTH_MODE, UNIT,
    HSV_TENNIS_BALL, HSV_PAPER_BALL, ACTIVE_HSV,
    CAMERA_HEIGHT_ABOVE_GROUND, DEPTH_OFFSET,
    MIN_BALL_AREA, MAX_BALL_AREA
)
from sender import UDPSender, DEFAULT_ROBOT_IP, DEFAULT_ROBOT_PORT


# ==================== CONFIGURATION ====================

# Import trajectory prediction configuration constants from config
from config import (
    # Physics constants
    GRAVITY,
    
    # Trajectory collection parameters
    MIN_POINTS_FOR_PREDICTION,
    MAX_TRAJECTORY_POINTS,
    TRAJECTORY_TIMEOUT,
    LANDING_HEIGHT,
    
    # RANSAC parameters
    RANSAC_MIN_SAMPLES,
    RANSAC_RESIDUAL_THRESHOLD,
    RANSAC_MAX_TRIALS,
    
    # Throw detection thresholds
    THROW_VELOCITY_THRESHOLD,
    THROW_CONFIRM_FRAMES,
    MIN_TRAJECTORY_DURATION,
    
    # Prediction confidence
    MIN_PREDICTION_CONFIDENCE,
    
    # Performance optimization
    PREDICTION_INTERVAL,
    
    # Logging
    PREDICTION_LOG_FILE
)


# ==================== DATA CLASSES ====================

@dataclass
class TrajectoryPoint:
    """Single point in the ball's trajectory"""
    position: np.ndarray  # [x, y, z] in meters
    timestamp: float
    velocity: Optional[np.ndarray] = None


@dataclass
class LandingPrediction:
    """Predicted landing information"""
    position: np.ndarray  # [x, y, z] landing coordinates
    time_to_land: float  # seconds until landing
    confidence: float  # 0-1 confidence score
    trajectory_points: List[np.ndarray] = field(default_factory=list)  # predicted path


# ==================== KALMAN FILTER ====================

class BallKalmanFilter:
    """
    6-state Kalman filter for 3D ball tracking
    State: [x, y, z, vx, vy, vz]
    """
    
    def __init__(self, dt: float = 1/30):
        self.dt = dt
        self.kf = KalmanFilter(dim_x=6, dim_z=3)
        
        # State transition matrix (constant velocity model with gravity)
        self.kf.F = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        
        # Measurement matrix (we only measure position)
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])
        
        # Measurement noise
        self.kf.R = np.eye(3) * 0.05  # 5cm measurement noise
        
        # Process noise (accounts for model uncertainty)
        q = 0.1
        self.kf.Q = np.eye(6) * q
        self.kf.Q[3:, 3:] *= 2  # Higher noise for velocity
        
        # Control input matrix (for gravity)
        self.kf.B = np.array([
            [0],
            [0.5 * dt**2],  # gravity affects y position
            [0],
            [0],
            [dt],  # gravity affects vy
            [0]
        ])
        
        # Initial covariance
        self.kf.P *= 1000
        
        self.initialized = False
    
    def initialize(self, position: np.ndarray):
        """Initialize filter with first measurement"""
        pos = np.asarray(position).flatten()
        # filterpy expects state as column vector (n, 1)
        self.kf.x = np.array([
            [pos[0]], [pos[1]], [pos[2]],
            [0.0], [0.0], [0.0]  # Initial velocity unknown
        ])
        self.initialized = True
    
    def predict(self) -> np.ndarray:
        """Predict next state with gravity"""
        u = np.array([[-GRAVITY]])  # gravity in -y direction
        self.kf.predict(u=u)
        # Extract position as 1D array (kf.x is column vector shape (6,1))
        return np.array([self.kf.x[0, 0], self.kf.x[1, 0], self.kf.x[2, 0]])
    
    def update(self, measurement: np.ndarray) -> np.ndarray:
        """Update with new measurement"""
        if not self.initialized:
            self.initialize(measurement)
        
        # filterpy expects measurement as 1D array or column vector
        meas = np.asarray(measurement).flatten()
        self.kf.update(meas)
        # Extract position as 1D array (kf.x is column vector shape (6,1))
        return np.array([self.kf.x[0, 0], self.kf.x[1, 0], self.kf.x[2, 0]])
    
    def get_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current position and velocity"""
        pos = np.array([self.kf.x[0, 0], self.kf.x[1, 0], self.kf.x[2, 0]])
        vel = np.array([self.kf.x[3, 0], self.kf.x[4, 0], self.kf.x[5, 0]])
        return pos, vel
    
    def reset(self):
        """Reset filter state"""
        self.kf.x = np.zeros((6, 1))  # Column vector
        self.kf.P = np.eye(6) * 1000
        self.initialized = False


# ==================== TRAJECTORY PREDICTOR ====================

class TrajectoryPredictor:
    """
    Predicts ball landing position using RANSAC and quadratic regression.
    
    The ball follows ballistic trajectory:
        x(t) = x0 + vx*t
        y(t) = y0 + vy*t - 0.5*g*t^2
        z(t) = z0 + vz*t
    
    We fit quadratic to y vs t, and linear to x,z vs t.
    """
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.trajectory_history: deque = deque(maxlen=MAX_TRAJECTORY_POINTS)
        self.kalman_filter = BallKalmanFilter(dt=1/FPS)
        
        self.is_throwing = False
        self.throw_start_time = 0
        self.last_detection_time = 0
        self.last_position = None
        self.last_velocity = None
        
        # Throw detection state
        self.high_velocity_frames = 0  # Count of consecutive high-velocity frames
        self.throw_confirmed = False  # True once throw is confirmed
        self.pre_throw_positions = deque(maxlen=5)  # Store positions before throw for reference
        
        # RANSAC regressors
        self.ransac_x = RANSACRegressor(
            min_samples=RANSAC_MIN_SAMPLES,
            residual_threshold=RANSAC_RESIDUAL_THRESHOLD,
            max_trials=RANSAC_MAX_TRIALS
        )
        self.ransac_y = RANSACRegressor(
            min_samples=RANSAC_MIN_SAMPLES,
            residual_threshold=RANSAC_RESIDUAL_THRESHOLD,
            max_trials=RANSAC_MAX_TRIALS
        )
        self.ransac_z = RANSACRegressor(
            min_samples=RANSAC_MIN_SAMPLES,
            residual_threshold=RANSAC_RESIDUAL_THRESHOLD,
            max_trials=RANSAC_MAX_TRIALS
        )
        
        # Fitted coefficients
        self.x_coeffs = None  # [x0, vx] linear
        self.y_coeffs = None  # [y0, vy, -0.5*g] quadratic
        self.z_coeffs = None  # [z0, vz] linear
        
        self.prediction_confidence = 0.0
        
        # Prediction tracking
        self.prediction_made = False
        self.prediction_locked = False
        self._prediction_count = 0
    
    def add_point(self, position: np.ndarray, timestamp: float):
        """Add a new trajectory point with improved throw detection"""
        # If prediction is already locked, just update position tracking for display
        # but don't process for new predictions
        if self.prediction_locked:
            position = np.asarray(position).flatten()
            if self.last_position is not None and self.last_detection_time > 0:
                dt = timestamp - self.last_detection_time
                if dt > 0 and dt < 0.5:
                    self.last_velocity = (position - self.last_position) / dt
            self.last_position = position.copy()
            self.last_detection_time = timestamp
            return
        
        # Ensure position is a flat 1D array
        position = np.asarray(position).flatten()
        
        # Update Kalman filter
        if self.kalman_filter.initialized:
            self.kalman_filter.predict()
        filtered_pos = self.kalman_filter.update(position)
        
        # Calculate velocity
        velocity = None
        speed = 0.0
        if self.last_position is not None and self.last_detection_time > 0:
            dt = timestamp - self.last_detection_time
            if dt > 0 and dt < 0.5:  # Ignore if too much time passed (detection gap)
                velocity = (position - self.last_position) / dt
                speed = np.linalg.norm(velocity)
        
        # Throw detection state machine
        if not self.throw_confirmed:
            # Store pre-throw positions
            self.pre_throw_positions.append((filtered_pos.copy(), timestamp))
            
            if velocity is not None:
                if speed > THROW_VELOCITY_THRESHOLD:
                    # High velocity detected
                    self.high_velocity_frames += 1
                    if self.debug:
                        print(f"[THROW CHECK] High velocity frame {self.high_velocity_frames}/{THROW_CONFIRM_FRAMES}, speed: {speed:.2f} m/s")
                    
                    if self.high_velocity_frames >= THROW_CONFIRM_FRAMES:
                        # Throw confirmed!
                        self.throw_confirmed = True
                        self.is_throwing = True
                        self.throw_start_time = timestamp
                        self.trajectory_history.clear()
                        
                        # Add the recent high-velocity points to trajectory
                        # (points from when we started seeing high velocity)
                        if self.debug:
                            print(f"[THROW CONFIRMED] Speed: {speed:.2f} m/s - starting trajectory collection")
                else:
                    # Low velocity - reset counter
                    if self.high_velocity_frames > 0 and self.debug:
                        print(f"[THROW CHECK] Reset - speed dropped to {speed:.2f} m/s")
                    self.high_velocity_frames = 0
        
        # Only add to trajectory history if throw is confirmed
        if self.throw_confirmed:
            point = TrajectoryPoint(
                position=filtered_pos.copy(),
                timestamp=timestamp,
                velocity=velocity.copy() if velocity is not None else None
            )
            self.trajectory_history.append(point)
        
        self.last_position = position.copy()
        self.last_velocity = velocity.copy() if velocity is not None else None
        self.last_detection_time = timestamp
    
    def can_predict(self) -> bool:
        """Check if we have enough data to make a prediction"""
        if not self.throw_confirmed:
            return False
        
        if len(self.trajectory_history) < MIN_POINTS_FOR_PREDICTION:
            return False
        
        # Check minimum trajectory duration
        if len(self.trajectory_history) >= 2:
            duration = self.trajectory_history[-1].timestamp - self.trajectory_history[0].timestamp
            if duration < MIN_TRAJECTORY_DURATION:
                if self.debug:
                    print(f"[PREDICT] Trajectory too short: {duration:.3f}s < {MIN_TRAJECTORY_DURATION}s")
                return False
        
        return True
    
    def fit_trajectory(self) -> bool:
        """Fit trajectory using RANSAC regression with fallback to simple least squares"""
        if len(self.trajectory_history) < MIN_POINTS_FOR_PREDICTION:
            if self.debug:
                print(f"[FIT] Not enough points: {len(self.trajectory_history)} < {MIN_POINTS_FOR_PREDICTION}")
            return False
        
        # Extract data
        times = []
        positions = []
        
        t0 = self.trajectory_history[0].timestamp
        for point in self.trajectory_history:
            t = point.timestamp - t0
            times.append(t)
            # Ensure position is exactly 3 elements
            pos = np.asarray(point.position).flatten()
            if len(pos) >= 3:
                positions.append(pos[:3])  # Take only first 3 elements
        
        if len(positions) < MIN_POINTS_FOR_PREDICTION:
            if self.debug:
                print(f"[FIT] Not enough valid positions: {len(positions)}")
            return False
        
        times = np.array(times[:len(positions)]).reshape(-1, 1)
        positions = np.array(positions)  # Should be (n, 3)
        
        if self.debug:
            print(f"[FIT] Fitting with {len(positions)} points, time span: {times[-1,0]:.3f}s")
        
        try:
            # Try RANSAC first, fall back to simple least squares if it fails
            try:
                # Fit x(t) = x0 + vx*t (linear)
                self.ransac_x.fit(times, positions[:, 0])
                self.x_coeffs = [
                    np.asarray(self.ransac_x.estimator_.intercept_).item(),
                    np.asarray(self.ransac_x.estimator_.coef_[0]).item()
                ]
            except Exception:
                # Fallback to simple least squares
                coeffs = np.polyfit(times.flatten(), positions[:, 0], 1)
                self.x_coeffs = [coeffs[1], coeffs[0]]  # [intercept, slope]
            
            try:
                # Fit y(t) = y0 + vy*t + a*t^2 (quadratic)
                times_quad = np.column_stack([times, times**2])
                self.ransac_y.fit(times_quad, positions[:, 1])
                self.y_coeffs = [
                    np.asarray(self.ransac_y.estimator_.intercept_).item(),
                    np.asarray(self.ransac_y.estimator_.coef_[0]).item(),
                    np.asarray(self.ransac_y.estimator_.coef_[1]).item()
                ]
            except Exception:
                # Fallback to simple quadratic fit
                coeffs = np.polyfit(times.flatten(), positions[:, 1], 2)
                self.y_coeffs = [coeffs[2], coeffs[1], coeffs[0]]  # [y0, vy, a]
            
            try:
                # Fit z(t) = z0 + vz*t (linear)
                self.ransac_z.fit(times, positions[:, 2])
                self.z_coeffs = [
                    np.asarray(self.ransac_z.estimator_.intercept_).item(),
                    np.asarray(self.ransac_z.estimator_.coef_[0]).item()
                ]
            except Exception:
                # Fallback to simple least squares
                coeffs = np.polyfit(times.flatten(), positions[:, 2], 1)
                self.z_coeffs = [coeffs[1], coeffs[0]]  # [intercept, slope]
            
            # Calculate predictions for confidence
            x_pred = self.x_coeffs[0] + self.x_coeffs[1] * times.flatten()
            y_pred = self.y_coeffs[0] + self.y_coeffs[1] * times.flatten() + self.y_coeffs[2] * times.flatten()**2
            z_pred = self.z_coeffs[0] + self.z_coeffs[1] * times.flatten()
            
            # Calculate R^2 for each dimension
            ss_res_x = np.sum((positions[:, 0] - x_pred)**2)
            ss_tot_x = np.sum((positions[:, 0] - np.mean(positions[:, 0]))**2)
            ss_res_y = np.sum((positions[:, 1] - y_pred)**2)
            ss_tot_y = np.sum((positions[:, 1] - np.mean(positions[:, 1]))**2)
            ss_res_z = np.sum((positions[:, 2] - z_pred)**2)
            ss_tot_z = np.sum((positions[:, 2] - np.mean(positions[:, 2]))**2)
            
            r2_x = 1 - ss_res_x / ss_tot_x if ss_tot_x > 0 else 0
            r2_y = 1 - ss_res_y / ss_tot_y if ss_tot_y > 0 else 0
            r2_z = 1 - ss_res_z / ss_tot_z if ss_tot_z > 0 else 0
            
            # Average R^2, clamped to [0, 1]
            self.prediction_confidence = max(0, min(1, (r2_x + r2_y + r2_z) / 3))
            
            if self.debug:
                print(f"[FIT] y_coeffs: y0={self.y_coeffs[0]:.3f}, vy={self.y_coeffs[1]:.3f}, a={self.y_coeffs[2]:.3f}")
                print(f"[FIT] Confidence: {self.prediction_confidence:.2f} (R2: x={r2_x:.2f}, y={r2_y:.2f}, z={r2_z:.2f})")
            
            return True
            
        except Exception as e:
            if self.debug:
                print(f"[FIT ERROR] {e}")
            return False
    
    def predict_landing(self) -> Optional[LandingPrediction]:
        """Predict where and when the ball will land"""
        # First check if we can predict (throw confirmed, enough points, enough duration)
        if not self.can_predict():
            return None
        
        if not self.fit_trajectory():
            return None
        
        if self.y_coeffs is None:
            if self.debug:
                print("[PREDICT] y_coeffs is None")
            return None
        
        if self.prediction_confidence < MIN_PREDICTION_CONFIDENCE:
            if self.debug:
                print(f"[PREDICT] Confidence too low: {self.prediction_confidence:.2f} < {MIN_PREDICTION_CONFIDENCE}")
            return None
        
        # Solve y(t) = LANDING_HEIGHT for t
        # y0 + vy*t + a*t^2 = LANDING_HEIGHT
        # a*t^2 + vy*t + (y0 - LANDING_HEIGHT) = 0
        a = self.y_coeffs[2]
        b = self.y_coeffs[1]
        c = self.y_coeffs[0] - LANDING_HEIGHT
        
        # Check if ball is actually falling (a should be negative for downward parabola)
        if self.debug:
            print(f"[PREDICT] Quadratic: a={a:.4f}, b={b:.4f}, c={c:.4f}")
        
        discriminant = b**2 - 4*a*c
        
        if discriminant < 0:
            if self.debug:
                print(f"[PREDICT] Negative discriminant: {discriminant:.4f} - ball won't reach ground")
            return None
        
        # Quadratic formula - take the positive root (future time)
        t0 = self.trajectory_history[0].timestamp
        current_t = self.last_detection_time - t0
        
        if abs(a) < 1e-6:
            # Nearly linear, use linear solution
            if abs(b) > 1e-6:
                t1 = t2 = -c / b
            else:
                if self.debug:
                    print("[PREDICT] Both a and b are near zero")
                return None
        else:
            t1 = (-b + np.sqrt(discriminant)) / (2*a)
            t2 = (-b - np.sqrt(discriminant)) / (2*a)
        
        # Choose future time that makes sense
        landing_times = [t for t in [t1, t2] if t > current_t]
        if not landing_times:
            return None
        
        landing_t = min(landing_times)
        
        # Predict landing position
        landing_x = float(self.x_coeffs[0] + self.x_coeffs[1] * landing_t)
        landing_y = float(LANDING_HEIGHT)
        landing_z = float(self.z_coeffs[0] + self.z_coeffs[1] * landing_t)
        
        landing_pos = np.array([landing_x, landing_y, landing_z], dtype=np.float64)
        time_to_land = float(landing_t - current_t)
        
        # Generate trajectory points for visualization
        trajectory_points = []
        num_points = 10  # Reduced from 20 for performance
        for i in range(num_points):
            t = current_t + (landing_t - current_t) * i / (num_points - 1)
            x = float(self.x_coeffs[0] + self.x_coeffs[1] * t)
            y = float(self.y_coeffs[0] + self.y_coeffs[1] * t + self.y_coeffs[2] * t**2)
            z = float(self.z_coeffs[0] + self.z_coeffs[1] * t)
            trajectory_points.append(np.array([x, y, z], dtype=np.float64))
        
        return LandingPrediction(
            position=landing_pos,
            time_to_land=time_to_land,
            confidence=self.prediction_confidence,
            trajectory_points=trajectory_points
        )
    
    def _save_and_print_prediction(self, prediction: LandingPrediction, 
                                     current_pos: np.ndarray, current_vel: np.ndarray):
        """Save prediction to file and print to console"""
        self._prediction_count += 1
        vel_mag = np.linalg.norm(current_vel)
        
        # Print to console
        print("\n" + "=" * 50)
        print(f"  TRAJECTORY PREDICTION #{self._prediction_count}")
        print("=" * 50)
        print(f"  Current Position: [{current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f}] m")
        print(f"  Current Velocity: [{current_vel[0]:.3f}, {current_vel[1]:.3f}, {current_vel[2]:.3f}] m/s")
        print(f"  Velocity Magnitude: {vel_mag:.3f} m/s")
        print("-" * 50)
        print(f"  LANDING POSITION: [{prediction.position[0]:.3f}, {prediction.position[1]:.3f}, {prediction.position[2]:.3f}] m")
        print(f"  TIME TO LANDING: {prediction.time_to_land:.3f} seconds")
        print(f"  CONFIDENCE: {prediction.confidence:.0%}")
        print("=" * 50)
        print("  Press [R] to reset and predict new throw")
        print("=" * 50 + "\n")
        
        # Save to predictions.txt
        timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S")
        write_header = not os.path.exists(PREDICTION_LOG_FILE)
        
        with open(PREDICTION_LOG_FILE, 'a') as f:
            if write_header:
                f.write("=" * 80 + "\n")
                f.write("TRAJECTORY PREDICTION LOG\n")
                f.write("=" * 80 + "\n\n")
            
            f.write("-" * 80 + "\n")
            f.write(f"Prediction #{self._prediction_count} | {timestamp_str}\n")
            f.write("-" * 80 + "\n")
            
            f.write(f"Initial Position (m): {current_pos[0]:.4f}, {current_pos[1]:.4f}, {current_pos[2]:.4f}\n")
            f.write(f"Initial Velocity (m/s): {current_vel[0]:.4f}, {current_vel[1]:.4f}, {current_vel[2]:.4f}\n")
            f.write(f"Speed (m/s): {vel_mag:.4f}\n\n")
            
            f.write(f"LANDING (m): {prediction.position[0]:.4f}, {prediction.position[1]:.4f}, {prediction.position[2]:.4f}\n")
            f.write(f"TIME TO LAND (s): {prediction.time_to_land:.4f}\n")
            f.write(f"CONFIDENCE: {prediction.confidence:.4f}\n\n")
            
            # Trajectory samples
            f.write("Trajectory samples:\n")
            n_points = len(prediction.trajectory_points)
            sample_indices = [0, n_points//4, n_points//2, 3*n_points//4, n_points-1]
            sample_indices = sorted(set([i for i in sample_indices if 0 <= i < n_points]))
            
            for i in sample_indices:
                pt = prediction.trajectory_points[i]
                # Calculate relative time from current position
                idx_ratio = i / max(1, n_points - 1)
                t = idx_ratio * prediction.time_to_land
                f.write(f"  t={t:.2f}s: ({pt[0]:.3f}, {pt[1]:.3f}, {pt[2]:.3f})\n")
            
            f.write("\n")
        
        print(f"[SAVED] Appended to: {PREDICTION_LOG_FILE}")
    
    def check_timeout(self, current_time: float) -> bool:
        """Check if trajectory should be cleared due to timeout (but keep prediction locked)"""
        if self.last_detection_time > 0:
            if current_time - self.last_detection_time > TRAJECTORY_TIMEOUT:
                # Only reset trajectory data, NOT the prediction lock
                self._soft_reset()
                return True
        return False
    
    def _soft_reset(self):
        """Reset trajectory tracking but keep prediction locked"""
        self.trajectory_history.clear()
        self.kalman_filter.reset()
        self.is_throwing = False
        self.throw_start_time = 0
        self.last_detection_time = 0
        self.last_position = None
        self.last_velocity = None
        self.x_coeffs = None
        self.y_coeffs = None
        self.z_coeffs = None
        
        # Reset throw detection state
        self.high_velocity_frames = 0
        self.throw_confirmed = False
        self.pre_throw_positions.clear()
        
        # Do NOT reset prediction_locked or prediction_made here
        if self.prediction_locked:
            print("[TIMEOUT] Trajectory cleared - prediction remains LOCKED (press R to reset)")
        else:
            print("[TIMEOUT] Trajectory cleared - ready for new throw")
    
    def reset(self):
        """Full reset - only called by user pressing 'R' key"""
        self.trajectory_history.clear()
        self.kalman_filter.reset()
        self.is_throwing = False
        self.throw_start_time = 0
        self.last_detection_time = 0
        self.last_position = None
        self.last_velocity = None
        self.x_coeffs = None
        self.y_coeffs = None
        self.z_coeffs = None
        self.prediction_confidence = 0.0
        
        # Reset throw detection state
        self.high_velocity_frames = 0
        self.throw_confirmed = False
        self.pre_throw_positions.clear()
        
        # Reset prediction tracking - ONLY on manual reset
        self.prediction_made = False
        self.prediction_locked = False
        
        print("[RESET] Full reset - ready for new throw detection")


# ==================== VISUALIZATION ====================

def draw_trajectory_overlay(frame: np.ndarray, 
                           predictor: TrajectoryPredictor,
                           prediction: Optional[LandingPrediction],
                           current_pos: Optional[np.ndarray],
                           udp_sender: Optional[UDPSender] = None):
    """Draw trajectory prediction overlay on frame"""
    h, w = frame.shape[:2]
    y = 30
    
    # Header
    cv.putText(frame, "Trajectory Predictor", (10, y),
              cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    y += 25
    
    # UDP Status
    if udp_sender:
        udp_status = f"UDP: {udp_sender.robot_ip}:{udp_sender.port}"
        udp_color = (0, 255, 0) if udp_sender.enabled else (0, 0, 255)
        status_text = "ON" if udp_sender.enabled else "OFF"
        cv.putText(frame, f"{udp_status} [{status_text}]", (10, y),
                  cv.FONT_HERSHEY_SIMPLEX, 0.45, udp_color, 1)
        y += 25
    
    # Throw status - more detailed
    if predictor.prediction_locked:
        throw_status = "PREDICTION LOCKED"
        throw_color = (255, 0, 255)  # Magenta - indicates locked state
    elif predictor.throw_confirmed:
        throw_status = "THROW CONFIRMED"
        throw_color = (0, 255, 0)  # Green
    elif predictor.high_velocity_frames > 0:
        throw_status = f"DETECTING ({predictor.high_velocity_frames}/{THROW_CONFIRM_FRAMES})"
        throw_color = (0, 255, 255)  # Yellow
    else:
        throw_status = "WAITING FOR THROW"
        throw_color = (150, 150, 150)  # Gray
    
    cv.putText(frame, f"Status: {throw_status}", (10, y),
              cv.FONT_HERSHEY_SIMPLEX, 0.5, throw_color, 1)
    y += 20
    
    # Show "Press R to reset" when locked
    if predictor.prediction_locked:
        cv.putText(frame, "Press [R] to reset for next throw", (10, y),
                  cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
        y += 20
    
    # Trajectory points count
    cv.putText(frame, f"Trajectory Points: {len(predictor.trajectory_history)}", (10, y),
              cv.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    y += 20
    
    # Show current speed
    if predictor.last_velocity is not None:
        speed = np.linalg.norm(predictor.last_velocity)
        speed_color = (0, 255, 0) if speed > THROW_VELOCITY_THRESHOLD else (200, 200, 200)
        cv.putText(frame, f"Speed: {speed:.2f} m/s (threshold: {THROW_VELOCITY_THRESHOLD})", (10, y),
                  cv.FONT_HERSHEY_SIMPLEX, 0.4, speed_color, 1)
        y += 20
    
    # Current position
    if current_pos is not None:
        pos = np.asarray(current_pos).flatten()
        cv.putText(frame, "Current Position (m):", (10, y),
                  cv.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y += 22
        cv.putText(frame, f"  X: {pos[0]:+.3f}", (10, y),
                  cv.FONT_HERSHEY_SIMPLEX, 0.45, (100, 200, 255), 1)
        y += 18
        cv.putText(frame, f"  Y: {pos[1]:+.3f}", (10, y),
                  cv.FONT_HERSHEY_SIMPLEX, 0.45, (100, 255, 100), 1)
        y += 18
        cv.putText(frame, f"  Z: {pos[2]:+.3f}", (10, y),
                  cv.FONT_HERSHEY_SIMPLEX, 0.45, (255, 100, 100), 1)
        y += 25
    
    # Prediction info
    if prediction:
        pred_pos = np.asarray(prediction.position).flatten()
        # Show different label if prediction is locked
        if predictor.prediction_locked:
            cv.putText(frame, "LOCKED PREDICTION:", (10, y),
                      cv.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2)  # Magenta for locked
        else:
            cv.putText(frame, "LANDING PREDICTION:", (10, y),
                      cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        y += 22
        pred_color = (255, 0, 255) if predictor.prediction_locked else (0, 255, 0)
        cv.putText(frame, f"  X: {pred_pos[0]:+.3f} m", (10, y),
                  cv.FONT_HERSHEY_SIMPLEX, 0.5, pred_color, 1)
        y += 20
        cv.putText(frame, f"  Z: {pred_pos[2]:+.3f} m", (10, y),
                  cv.FONT_HERSHEY_SIMPLEX, 0.5, pred_color, 1)
        y += 20
        cv.putText(frame, f"  Time: {prediction.time_to_land:.2f} s", (10, y),
                  cv.FONT_HERSHEY_SIMPLEX, 0.5, pred_color, 1)
        y += 20
        cv.putText(frame, f"  Confidence: {prediction.confidence:.0%}", (10, y),
                  cv.FONT_HERSHEY_SIMPLEX, 0.5, pred_color, 1)
    else:
        cv.putText(frame, "Prediction: N/A", (10, y),
                  cv.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
        y += 20
        # Show current confidence to help debug
        conf = predictor.prediction_confidence
        conf_color = (0, 255, 0) if conf >= MIN_PREDICTION_CONFIDENCE else (0, 128, 255)
        cv.putText(frame, f"  Confidence: {conf:.0%} (need {MIN_PREDICTION_CONFIDENCE:.0%})", (10, y),
                  cv.FONT_HERSHEY_SIMPLEX, 0.4, conf_color, 1)
    
    # Instructions at bottom
    cv.putText(frame, "[R] Reset  [T] Tennis  [P] Paper  [U] Toggle UDP  [Q] Quit",
              (10, h - 10), cv.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)


# ==================== MAIN ====================

def main():
    parser = argparse.ArgumentParser(description='Ball trajectory prediction')
    parser.add_argument('--debug', '-d', action='store_true',
                       help='Enable debug visualization')
    parser.add_argument('--robot-ip', type=str, default=DEFAULT_ROBOT_IP,
                       help=f'Robot IP address (default: {DEFAULT_ROBOT_IP})')
    parser.add_argument('--port', type=int, default=DEFAULT_ROBOT_PORT,
                       help=f'UDP port (default: {DEFAULT_ROBOT_PORT})')
    parser.add_argument('--no-udp', action='store_true',
                       help='Disable UDP sending')
    parser.add_argument('--send-prediction', action='store_true',
                       help='Send predicted landing position instead of current position')
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("  TRAJECTORY PREDICTOR - BALL LANDING ESTIMATION")
    print("=" * 60)
    
    # Initialize UDP sender
    udp_sender = None
    if not args.no_udp:
        udp_sender = UDPSender(args.robot_ip, args.port, rate_limit=30)
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
    
    # Get camera intrinsics (handle different ZED SDK versions)
    cam_info = zed.get_camera_information()
    
    # Try different attribute paths for calibration parameters
    try:
        calib = cam_info.camera_configuration.calibration_parameters.left_cam
    except AttributeError:
        try:
            calib = cam_info.calibration_parameters.left_cam
        except AttributeError:
            try:
                calib = cam_info.camera_configuration.calibration_parameters.left_cam
            except AttributeError:
                print("Warning: Could not get calibration from SDK, using defaults")
                calib = None
    
    # Try different attribute paths for resolution
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
                # Fall back to resolution-based defaults
                if RESOLUTION == sl.RESOLUTION.HD720:
                    res_width, res_height = 1280, 720
                elif RESOLUTION == sl.RESOLUTION.HD1080:
                    res_width, res_height = 1920, 1080
                else:
                    res_width, res_height = 1280, 720
    
    # Build intrinsics
    if calib is not None:
        intrinsics = CameraIntrinsics(
            fx=calib.fx, fy=calib.fy,
            cx=calib.cx, cy=calib.cy,
            width=res_width,
            height=res_height
        )
    else:
        # Default intrinsics for HD720
        intrinsics = CameraIntrinsics(
            fx=700.0, fy=700.0,
            cx=res_width / 2, cy=res_height / 2,
            width=res_width,
            height=res_height
        )
    
    print(f"✓ Camera: {intrinsics.width}x{intrinsics.height}")
    print("=" * 60)
    
    # Initialize components
    detector = BallDetector(ACTIVE_HSV)
    predictor = TrajectoryPredictor(debug=args.debug)
    
    # Runtime parameters
    runtime = sl.RuntimeParameters()
    runtime.confidence_threshold = 50
    
    # Allocate ZED mats
    image_left = sl.Mat()
    depth_map = sl.Mat()
    point_cloud = sl.Mat()
    
    print("\nStarting trajectory prediction...")
    print("Throw the ball to see landing prediction!\n")
    
    last_prediction = None
    frame_counter = 0  # For PREDICTION_INTERVAL optimization
    
    try:
        while True:
            # Grab frame
            if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                continue
            
            frame_counter += 1
            current_time = time.time()
            
            # Check for timeout
            predictor.check_timeout(current_time)
            
            # Retrieve images
            zed.retrieve_image(image_left, sl.VIEW.LEFT)
            zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH)
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
            
            frame = image_left.get_data()[:, :, :3].copy()
            hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            
            # Detect ball
            detection = detector.detect(hsv)
            
            current_pos = None
            prediction = None
            
            if detection:
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
                    
                    # Add to trajectory
                    predictor.add_point(current_pos, current_time)
                    
                    # Predict landing (only every PREDICTION_INTERVAL frames for performance)
                    # and only if no prediction has been locked yet
                    should_predict = (frame_counter % PREDICTION_INTERVAL == 0) and not predictor.prediction_locked
                    
                    if should_predict:
                        prediction = predictor.predict_landing()
                        if prediction:
                            last_prediction = prediction
                            # Mark prediction as made and call logging
                            predictor.prediction_made = True
                            predictor.prediction_locked = True
                            # Get velocity for logging
                            _, current_vel = predictor.kalman_filter.get_state()
                            predictor._save_and_print_prediction(prediction, current_pos, current_vel)
                        elif args.debug:
                            # Show why prediction failed
                            print(f"[MAIN] Throw confirmed: {predictor.throw_confirmed}, Points: {len(predictor.trajectory_history)}, High-vel frames: {predictor.high_velocity_frames}")
                    
                    # Send coordinates via UDP
                    if udp_sender and udp_sender.enabled:
                        if args.send_prediction and prediction:
                            # Send predicted landing position
                            udp_sender.send_coordinates(
                                prediction.position[0],
                                prediction.position[1],
                                prediction.position[2]
                            )
                        else:
                            # Send current position
                            udp_sender.send_coordinates(
                                current_pos[0],
                                current_pos[1],
                                current_pos[2]
                            )
                    
                    # Draw detection
                    draw_detection(frame, detection, "", (0, 255, 0))
                else:
                    draw_detection(frame, detection, "", (0, 165, 255))
            
            # Use last prediction if no new one
            display_prediction = prediction if prediction else last_prediction
            
            # Draw trajectory history on frame (project 3D to 2D)
            # Only draw every 2nd point for performance
            traj_len = len(predictor.trajectory_history)
            if traj_len > 1:
                step = max(1, traj_len // 15)  # Draw max ~15 points
                for i in range(0, traj_len, step):
                    point = predictor.trajectory_history[i]
                    pos = np.asarray(point.position).flatten()
                    if len(pos) >= 3 and pos[2] > 0:
                        px = int(pos[0].item() * intrinsics.fx / pos[2].item() + intrinsics.cx)
                        py = int(-pos[1].item() * intrinsics.fy / pos[2].item() + intrinsics.cy)
                        if 0 <= px < frame.shape[1] and 0 <= py < frame.shape[0]:
                            alpha = i / traj_len
                            color = (int(255 * (1 - alpha)), int(255 * alpha), 0)
                            cv.circle(frame, (px, py), 3, color, -1)
            
            # Draw predicted trajectory (only every 3rd segment for performance)
            if display_prediction and display_prediction.trajectory_points:
                pts = display_prediction.trajectory_points
                step = max(1, len(pts) // 7)  # Draw ~7 segments
                for i in range(0, len(pts) - 1, step):
                    pt = np.asarray(pts[i]).flatten()
                    if len(pt) >= 3 and pt[2] > 0:
                        px1 = int(pt[0].item() * intrinsics.fx / pt[2].item() + intrinsics.cx)
                        py1 = int(-pt[1].item() * intrinsics.fy / pt[2].item() + intrinsics.cy)
                        
                        next_idx = min(i + step, len(pts) - 1)
                        pt2 = np.asarray(pts[next_idx]).flatten()
                        if len(pt2) >= 3 and pt2[2] > 0:
                            px2 = int(pt2[0].item() * intrinsics.fx / pt2[2].item() + intrinsics.cx)
                            py2 = int(-pt2[1].item() * intrinsics.fy / pt2[2].item() + intrinsics.cy)
                            
                            if (0 <= px1 < frame.shape[1] and 0 <= py1 < frame.shape[0] and
                                0 <= px2 < frame.shape[1] and 0 <= py2 < frame.shape[0]):
                                cv.line(frame, (px1, py1), (px2, py2), (0, 255, 255), 2)
            
            # Draw landing marker
            if display_prediction:
                lp = np.asarray(display_prediction.position).flatten()
                if len(lp) >= 3 and lp[2] > 0:
                    lpx = int(lp[0].item() * intrinsics.fx / lp[2].item() + intrinsics.cx)
                    lpy = int(-lp[1].item() * intrinsics.fy / lp[2].item() + intrinsics.cy)
                    if 0 <= lpx < frame.shape[1] and 0 <= lpy < frame.shape[0]:
                        cv.drawMarker(frame, (lpx, lpy), (0, 0, 255), 
                                     cv.MARKER_CROSS, 30, 3)
                        cv.putText(frame, "LANDING", (lpx + 10, lpy - 10),
                                  cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Draw overlay
            draw_trajectory_overlay(frame, predictor, display_prediction, 
                                   current_pos, udp_sender)
            
            # Display
            cv.imshow("Trajectory Predictor", frame)
            
            # Handle keyboard input
            key = cv.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('r'):
                predictor.reset()
                last_prediction = None
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
