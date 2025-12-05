"""
Ball Catcher Integration System (OPTIMIZED)

Integrates trajectory prediction and robot tracking to enable a robot to catch thrown balls.
This module combines:
- Ball trajectory prediction from trajectory_predictor.py
- Robot pose tracking from aruco_robot_tracker.py  
- Combined UDP communication to send both ball landing and robot pose data

PERFORMANCE OPTIMIZATIONS:
- Configurable depth mode (PERFORMANCE vs ULTRA)
- Downscaled ball detection with coordinate rescaling
- Reduced depth sampling points
- Lazy visualization updates
- Pre-allocated buffers
- Moved imports to module level
- Configurable robot tracking frequency
- Optional threading for ArUco detection

COORDINATE SYSTEM (from camera's perspective):
    +X: Right
    +Y: Up
    +Z: Backward (into camera, away from scene)

UDP DATA FORMAT (40 bytes total):
    - Ball landing position: 3 floats (x, y, z) = 12 bytes
    - Ball prediction confidence: 1 float = 4 bytes
    - Robot pose: 6 floats (x, y, z, roll, pitch, yaw) = 24 bytes

Requirements:
    pip install opencv-contrib-python numpy pyzed filterpy scikit-learn

Usage:
    python ball_catcher.py --marker-size 95
    python ball_catcher.py --marker-size 95 --robot-ip 192.168.0.155 --robot-port 5005
    python ball_catcher.py --marker-size 95 --no-udp --debug
    python ball_catcher.py --marker-size 95 --fast-depth  # Use PERFORMANCE depth mode

Controls:
    q/ESC - Quit
    r - Reset trajectory prediction
    c - Calibrate/set robot origin
    u - Toggle UDP sending
    d - Toggle debug visualization
"""

import cv2 as cv
import numpy as np
import argparse
import time
import struct
from typing import Optional, Tuple
from threading import Thread, Lock
from collections import deque

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# Import ZED SDK
try:
    import pyzed.sl as sl
    ZED_AVAILABLE = True
except ImportError:
    ZED_AVAILABLE = False
    print("ERROR: ZED SDK not available. This module requires ZED camera.")
    exit(1)

# Import existing components - moved to module level for performance
from trajectory_predictor import TrajectoryPredictor, LandingPrediction
from ball_tracker_params import (
    BallDetector, 
    CameraIntrinsics as BallCameraIntrinsics,
    get_robust_depth,
    draw_detection
)
from aruco_robot_tracker import (
    MultiMarkerRobotTracker, 
    Pose as RobotPose,
    CameraIntrinsics as RobotCameraIntrinsics
)
from sender import UDPSender
from config import (
    RESOLUTION, FPS, DEPTH_MODE, UNIT,
    DEFAULT_ROBOT_IP, DEFAULT_ROBOT_PORT,
    ACTIVE_HSV, CAMERA_HEIGHT_ABOVE_GROUND, DEPTH_OFFSET,
    MIN_BALL_AREA, MAX_BALL_AREA, MIN_CIRCULARITY
)


# ==================== OPTIMIZED BALL DETECTOR ====================

class FastBallDetector:
    """
    Optimized ball detector with downscaling support.
    Processes images at reduced resolution for faster detection.
    """
    
    def __init__(self, hsv_config: dict, scale: float = 0.5):
        """
        Initialize fast ball detector.
        
        Args:
            hsv_config: HSV color range configuration
            scale: Downscale factor (0.5 = half resolution, 1.0 = full)
        """
        self.lower = np.array(hsv_config['lower'], dtype=np.uint8)
        self.upper = np.array(hsv_config['upper'], dtype=np.uint8)
        self.kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))  # Smaller kernel
        self.scale = scale
        self.inv_scale = 1.0 / scale
        
        # Scaled area thresholds
        self.min_area = int(MIN_BALL_AREA * scale * scale)
        self.max_area = int(MAX_BALL_AREA * scale * scale)
        
        # Pre-allocate arrays for reuse
        self._mask = None
        
    def set_hsv_range(self, lower: tuple, upper: tuple):
        self.lower = np.array(lower, dtype=np.uint8)
        self.upper = np.array(upper, dtype=np.uint8)
    
    def detect(self, frame_hsv: np.ndarray) -> Optional[Tuple[int, int, int]]:
        """
        Detect ball in HSV frame with optional downscaling.
        
        Args:
            frame_hsv: HSV frame (pre-converted for efficiency)
            
        Returns:
            (x, y, radius) in original frame coordinates, or None
        """
        # Downscale if needed
        if self.scale < 1.0:
            h, w = frame_hsv.shape[:2]
            new_w, new_h = int(w * self.scale), int(h * self.scale)
            hsv = cv.resize(frame_hsv, (new_w, new_h), interpolation=cv.INTER_LINEAR)
        else:
            hsv = frame_hsv
        
        # Create mask
        mask = cv.inRange(hsv, self.lower, self.upper)
        
        # Morphological operations (single pass each)
        mask = cv.erode(mask, self.kernel, iterations=1)
        mask = cv.dilate(mask, self.kernel, iterations=1)
        
        # Find contours
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        best_detection = None
        best_score = 0
        
        for contour in contours:
            area = cv.contourArea(contour)
            if area < self.min_area or area > self.max_area:
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
                # Scale coordinates back to original resolution
                best_detection = (
                    int(x * self.inv_scale),
                    int(y * self.inv_scale),
                    int(radius * self.inv_scale)
                )
        
        return best_detection


# ==================== OPTIMIZED DEPTH FUNCTION ====================

def get_fast_depth(point_cloud: sl.Mat, x: int, y: int, 
                   sample_radius: int = 3) -> Optional[np.ndarray]:
    """
    Fast 3D position retrieval with minimal sampling.
    
    Args:
        point_cloud: ZED point cloud
        x, y: Pixel coordinates
        sample_radius: Sampling radius (reduced from 5 to 3)
        
    Returns:
        3D position [x, y, z] or None
    """
    h, w = point_cloud.get_height(), point_cloud.get_width()
    
    # Try center point first (fastest path)
    err, point = point_cloud.get_value(x, y)
    if err == sl.ERROR_CODE.SUCCESS and np.isfinite(point[2]) and point[2] > 0:
        return np.array(point[:3], dtype=np.float32)
    
    # Sample sparse grid (9 points instead of 36)
    valid_points = []
    for dy in range(-sample_radius, sample_radius + 1, sample_radius):
        for dx in range(-sample_radius, sample_radius + 1, sample_radius):
            px, py = x + dx, y + dy
            if 0 <= px < w and 0 <= py < h:
                err, point = point_cloud.get_value(px, py)
                if err == sl.ERROR_CODE.SUCCESS and np.isfinite(point[2]) and point[2] > 0:
                    valid_points.append(point[:3])
    
    if not valid_points:
        return None
    
    # Use median for robustness
    return np.median(valid_points, axis=0).astype(np.float32)


# ==================== COMBINED UDP SENDER ====================

class CombinedUDPSender(UDPSender):
    """Enhanced UDP sender that sends combined ball + robot data"""
    
    def __init__(self, robot_ip: str = DEFAULT_ROBOT_IP, port: int = DEFAULT_ROBOT_PORT,
                 rate_limit: float = 60.0):
        super().__init__(robot_ip, port, rate_limit=rate_limit)
        # Pre-allocate pack buffer
        self._data_buffer = bytearray(40)
    
    def send_combined_data(self, 
                          ball_landing: Optional[np.ndarray] = None,
                          ball_confidence: float = 0.0,
                          robot_pose: Optional[RobotPose] = None) -> bool:
        """
        Send combined ball landing prediction and robot pose data.
        
        Packet format (40 bytes):
            - Ball landing (x, y, z): 3 floats (12 bytes)
            - Ball confidence: 1 float (4 bytes)
            - Robot pose (x, y, z, roll, pitch, yaw): 6 floats (24 bytes)
        """
        if not self.enabled:
            return False
        
        # Check rate limit
        current_time = time.time()
        if current_time - self.last_send_time < self.min_send_interval:
            return False
        
        try:
            # Prepare ball data
            if ball_landing is not None and len(ball_landing) >= 3:
                ball_x, ball_y, ball_z = float(ball_landing[0]), float(ball_landing[1]), float(ball_landing[2])
            else:
                ball_x, ball_y, ball_z = 0.0, 0.0, 0.0
                ball_confidence = 0.0
            
            # Prepare robot data
            if robot_pose is not None:
                robot_x, robot_y, robot_z = float(robot_pose.x), float(robot_pose.y), float(robot_pose.z)
                robot_roll, robot_pitch, robot_yaw = float(robot_pose.roll), float(robot_pose.pitch), float(robot_pose.yaw)
            else:
                robot_x, robot_y, robot_z = 0.0, 0.0, 0.0
                robot_roll, robot_pitch, robot_yaw = 0.0, 0.0, 0.0
            
            # Pack data: 10 floats = 40 bytes
            data = struct.pack('10f',
                             ball_x, ball_y, ball_z, float(ball_confidence),
                             robot_x, robot_y, robot_z, 
                             robot_roll, robot_pitch, robot_yaw)
            
            self.sock.sendto(data, (self.robot_ip, self.port))
            self.packets_sent += 1
            self.last_send_time = current_time
            return True
            
        except Exception as e:
            # Silently fail to avoid console spam
            return False


# ==================== THREADED ARUCO TRACKER ====================

class ThreadedArucoTracker:
    """
    Runs ArUco detection in a separate thread to avoid blocking main loop.
    """
    
    def __init__(self, tracker: MultiMarkerRobotTracker):
        self.tracker = tracker
        self.lock = Lock()
        self.latest_pose = None
        self.latest_detections = []
        self.frame_queue = None
        self.running = False
        self.thread = None
        
    def start(self):
        """Start background tracking thread"""
        self.running = True
        self.thread = Thread(target=self._run, daemon=True)
        self.thread.start()
        
    def stop(self):
        """Stop background tracking thread"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
    
    def update_frame(self, frame: np.ndarray):
        """Submit frame for processing (non-blocking)"""
        with self.lock:
            self.frame_queue = frame.copy()
    
    def _run(self):
        """Background processing loop"""
        while self.running:
            frame = None
            with self.lock:
                if self.frame_queue is not None:
                    frame = self.frame_queue
                    self.frame_queue = None
            
            if frame is not None:
                pose, detections = self.tracker.update(frame)
                with self.lock:
                    self.latest_pose = pose
                    self.latest_detections = detections
            else:
                time.sleep(0.001)  # Prevent busy-waiting
    
    def get_pose(self) -> Tuple[Optional[RobotPose], list]:
        """Get latest pose (non-blocking)"""
        with self.lock:
            return self.latest_pose, self.latest_detections


# ==================== BALL CATCHER SYSTEM ====================

class BallCatcherSystem:
    """Main integration class coordinating ball tracking and robot tracking"""
    
    def __init__(self, 
                 marker_size_mm: float,
                 left_id: int = 0, front_id: int = 0, right_id: int = 0, back_id: int = 0,
                 robot_ip: str = DEFAULT_ROBOT_IP,
                 robot_port: int = DEFAULT_ROBOT_PORT,
                 enable_udp: bool = True,
                 debug: bool = False,
                 fast_depth: bool = False,
                 detection_scale: float = 0.5,
                 robot_track_interval: int = 4,
                 use_threaded_aruco: bool = False):
        
        self.debug = debug
        self.detection_scale = detection_scale
        self.robot_track_interval = robot_track_interval
        self.use_threaded_aruco = use_threaded_aruco
        
        # Initialize ZED camera
        print("\nInitializing ZED camera...")
        self.zed = sl.Camera()
        
        init_params = sl.InitParameters()
        init_params.camera_resolution = RESOLUTION
        init_params.camera_fps = FPS
        init_params.coordinate_units = UNIT
        
        # Use faster depth mode if requested
        if fast_depth:
            init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
            print("  Using PERFORMANCE depth mode (faster)")
        else:
            init_params.depth_mode = DEPTH_MODE
            print(f"  Using {DEPTH_MODE} depth mode")
        
        status = self.zed.open(init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            print(f"ERROR: Failed to open ZED camera: {status}")
            exit(1)
        
        print("✓ ZED camera initialized")
        
        # Get camera intrinsics
        cam_info = self.zed.get_camera_information()
        
        try:
            calib = cam_info.camera_configuration.calibration_parameters.left_cam
        except AttributeError:
            try:
                calib = cam_info.calibration_parameters.left_cam
            except:
                print("ERROR: Failed to get camera calibration parameters")
                self.zed.close()
                exit(1)
        
        try:
            res = cam_info.camera_configuration.camera_resolution
            width, height = res.width, res.height
        except:
            width, height = 1280, 720
        
        fx, fy = calib.fx, calib.fy
        cx, cy = calib.cx, calib.cy
        
        try:
            dist = np.array(calib.disto[:5], dtype=np.float32)
        except:
            dist = np.zeros(5, dtype=np.float32)
        
        # Create camera intrinsics
        ball_intrinsics = BallCameraIntrinsics(
            fx=fx, fy=fy, cx=cx, cy=cy,
            width=width, height=height
        )
        
        robot_intrinsics = RobotCameraIntrinsics(
            fx=fx, fy=fy, cx=cx, cy=cy,
            width=width, height=height,
            dist_coeffs=dist
        )
        
        print(f"✓ Camera resolution: {width}x{height}")
        
        # Initialize FAST ball tracking components
        print("\nInitializing optimized ball tracker...")
        self.ball_detector = FastBallDetector(ACTIVE_HSV, scale=detection_scale)
        self.trajectory_predictor = TrajectoryPredictor(debug=debug)
        self.ball_intrinsics = ball_intrinsics
        print(f"✓ Ball tracker initialized (detection scale: {detection_scale})")
        
        # Initialize robot tracking
        print("\nInitializing robot tracker...")
        self.robot_tracker = MultiMarkerRobotTracker(
            intrinsics=robot_intrinsics,
            marker_size_mm=marker_size_mm,
            left_id=left_id,
            front_id=front_id,
            right_id=right_id,
            back_id=back_id
        )
        
        # Optionally use threaded ArUco detection
        self.threaded_tracker = None
        if use_threaded_aruco:
            self.threaded_tracker = ThreadedArucoTracker(self.robot_tracker)
            self.threaded_tracker.start()
            print(f"✓ Robot tracker initialized (threaded, marker size: {marker_size_mm}mm)")
        else:
            print(f"✓ Robot tracker initialized (interval: every {robot_track_interval} frames)")
        
        # Initialize combined UDP sender with higher rate limit
        self.udp_sender = CombinedUDPSender(robot_ip=robot_ip, port=robot_port, rate_limit=60.0)
        if not enable_udp:
            self.udp_sender.enabled = False
        
        print(f"\n{'✓' if self.udp_sender.enabled else '✗'} UDP: {robot_ip}:{robot_port}")
        
        # ZED image containers - pre-allocate
        self.image_left = sl.Mat()
        self.point_cloud = sl.Mat()
        self.runtime = sl.RuntimeParameters()
        self.runtime.confidence_threshold = 50
        
        # Pre-allocate processing buffers (reused each frame)
        self.hsv_frame = None  # Will be allocated on first frame
        
        # State
        self.current_prediction = None
        self.current_robot_pose = None
        self.current_detections = []
        self.frame_count = 0
        self.start_time = time.time()
        self.prediction_locked = False
        
        # Performance tracking
        self.fps_history = deque(maxlen=30)
        self.last_frame_time = time.time()
        
        # Visualization update interval (reduce text rendering overhead)
        self.viz_update_interval = 3
        self._cached_overlay_data = {}
        
    def process_frame(self) -> Optional[np.ndarray]:
        """Process one frame: detect ball, track robot, send UDP"""
        
        # Timing for FPS
        current_time = time.time()
        frame_dt = current_time - self.last_frame_time
        self.last_frame_time = current_time
        if frame_dt > 0:
            self.fps_history.append(1.0 / frame_dt)
        
        # Grab frame from ZED
        if self.zed.grab(self.runtime) != sl.ERROR_CODE.SUCCESS:
            return None
        
        # Retrieve image (always needed)
        self.zed.retrieve_image(self.image_left, sl.VIEW.LEFT)
        frame = self.image_left.get_data()
        
        if frame is None:
            return None
        
        # Get BGR frame (ZED returns BGRA) - make contiguous copy for OpenCV
        frame_bgr = frame[:, :, :3].copy()
        
        self.frame_count += 1
        
        # Convert to HSV once (reuse pre-allocated buffer if possible)
        if self.hsv_frame is None or self.hsv_frame.shape[:2] != frame_bgr.shape[:2]:
            self.hsv_frame = cv.cvtColor(frame_bgr, cv.COLOR_BGR2HSV)
        else:
            cv.cvtColor(frame_bgr, cv.COLOR_BGR2HSV, dst=self.hsv_frame)
        
        # ===== BALL TRACKING (OPTIMIZED) =====
        ball_pos = None
        detection = None
        
        # Detect ball using fast detector (pass HSV directly)
        detection = self.ball_detector.detect(self.hsv_frame)
        
        if detection:
            x, y, radius = detection
            
            # Only retrieve point cloud when ball is detected
            self.zed.retrieve_measure(self.point_cloud, sl.MEASURE.XYZRGBA)
            
            # Get 3D position using fast depth
            ball_pos = get_fast_depth(self.point_cloud, x, y, sample_radius=3)
            
            if ball_pos is not None:
                # Apply coordinate offset corrections
                ball_pos[1] += CAMERA_HEIGHT_ABOVE_GROUND
                ball_pos[2] += DEPTH_OFFSET
                
                # Only add to trajectory predictor if not locked
                if not self.prediction_locked:
                    self.trajectory_predictor.add_point(ball_pos, current_time)
        
        # Predict landing only if not locked (every 3 frames)
        if not self.prediction_locked and self.frame_count % 3 == 0:
            if self.trajectory_predictor.can_predict():
                new_prediction = self.trajectory_predictor.predict_landing()
                if new_prediction is not None:
                    self.current_prediction = new_prediction
                    self.prediction_locked = True
                    print("[LOCKED] Prediction locked. Press 'R' to reset.")
            
            # Check timeout
            self.trajectory_predictor.check_timeout(current_time)
        
        # ===== ROBOT TRACKING =====
        if self.use_threaded_aruco and self.threaded_tracker:
            # Submit frame for async processing
            if self.frame_count % self.robot_track_interval == 0:
                self.threaded_tracker.update_frame(frame_bgr)
            # Get latest result (non-blocking)
            self.current_robot_pose, self.current_detections = self.threaded_tracker.get_pose()
        else:
            # Synchronous tracking (every N frames)
            if self.frame_count % self.robot_track_interval == 0:
                self.current_robot_pose, self.current_detections = self.robot_tracker.update(frame_bgr)
        
        # ===== SEND COMBINED UDP =====
        ball_landing = None
        ball_confidence = 0.0
        
        if self.current_prediction is not None:
            ball_landing = self.current_prediction.position
            ball_confidence = self.current_prediction.confidence
        
        self.udp_sender.send_combined_data(
            ball_landing=ball_landing,
            ball_confidence=ball_confidence,
            robot_pose=self.current_robot_pose
        )
        
        # ===== VISUALIZATION (OPTIMIZED) =====
        # frame_bgr is already a copy from ZED, draw directly on it
        # No need for additional copy - saves memory bandwidth
        
        # Draw ball detection
        if detection and ball_pos is not None:
            x, y, radius = detection
            cv.circle(frame_bgr, (x, y), int(radius), (0, 255, 0), 2)
            cv.circle(frame_bgr, (x, y), 3, (0, 0, 255), -1)
        elif detection:
            x, y, radius = detection
            cv.circle(frame_bgr, (x, y), int(radius), (0, 165, 255), 2)
        
        # Update overlay only every N frames to reduce text rendering overhead
        if self.frame_count % self.viz_update_interval == 0:
            self._update_overlay_cache(ball_pos)
        
        self._draw_cached_overlay(frame_bgr, detection, ball_pos)
        
        return frame_bgr
    
    def _update_overlay_cache(self, ball_pos: Optional[np.ndarray]):
        """Update cached overlay data (called every N frames)"""
        # Calculate FPS
        if self.fps_history:
            fps = sum(self.fps_history) / len(self.fps_history)
        else:
            fps = 0.0
        
        self._cached_overlay_data = {
            'fps': fps,
            'packets': self.udp_sender.packets_sent,
            'udp_enabled': self.udp_sender.enabled,
            'ball_pos': ball_pos.copy() if ball_pos is not None else None,
            'prediction': self.current_prediction,
            'robot_pose': self.current_robot_pose,
            'prediction_locked': self.prediction_locked
        }
    
    def _draw_cached_overlay(self, frame: np.ndarray, detection, ball_pos: Optional[np.ndarray]):
        """Draw overlay using cached data"""
        h, w = frame.shape[:2]
        y_pos = 30
        cache = self._cached_overlay_data
        
        if not cache:
            return
        
        # Header
        cv.putText(frame, "Ball Catcher (Optimized)", (10, y_pos),
                  cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_pos += 30
        
        # FPS
        fps = cache.get('fps', 0)
        fps_color = (0, 255, 0) if fps >= 25 else (0, 165, 255) if fps >= 15 else (0, 0, 255)
        cv.putText(frame, f"FPS: {fps:.1f}", (10, y_pos),
                  cv.FONT_HERSHEY_SIMPLEX, 0.5, fps_color, 1)
        y_pos += 20
        
        # UDP Status
        udp_status = "ON" if cache.get('udp_enabled', False) else "OFF"
        udp_color = (0, 255, 0) if cache.get('udp_enabled', False) else (0, 0, 255)
        cv.putText(frame, f"UDP: {udp_status} ({cache.get('packets', 0)})", 
                  (10, y_pos), cv.FONT_HERSHEY_SIMPLEX, 0.5, udp_color, 1)
        y_pos += 30
        
        # Ball position (use live data for responsiveness)
        if ball_pos is not None:
            cv.putText(frame, f"Ball: [{ball_pos[0]:.2f}, {ball_pos[1]:.2f}, {ball_pos[2]:.2f}]", 
                      (10, y_pos), cv.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1)
            y_pos += 22
        
        # Landing prediction
        pred = cache.get('prediction')
        if pred is not None:
            landing = pred.position
            conf = pred.confidence
            pred_color = (0, 255, 255) if conf >= 0.8 else (0, 200, 255)
            status = "[LOCKED]" if cache.get('prediction_locked', False) else ""
            cv.putText(frame, f"Landing{status}: [{landing[0]:.2f}, {landing[2]:.2f}] ({conf*100:.0f}%)", 
                      (10, y_pos), cv.FONT_HERSHEY_SIMPLEX, 0.45, pred_color, 1)
            y_pos += 22
        
        # Robot pose (bottom left)
        robot_pose = cache.get('robot_pose')
        if robot_pose:
            bottom_y = h - 40
            cv.putText(frame, f"Robot: [{robot_pose.x:.2f}, {robot_pose.z:.2f}] yaw:{np.degrees(robot_pose.yaw):.0f}",
                      (10, bottom_y), cv.FONT_HERSHEY_SIMPLEX, 0.45, (255, 128, 0), 1)
        
        # Controls hint
        cv.putText(frame, "[R]Reset [C]Calibrate [U]UDP [D]Debug [Q]Quit",
                  (10, h - 10), cv.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
    
    def reset_trajectory(self):
        """Reset trajectory prediction"""
        self.trajectory_predictor.reset()
        self.current_prediction = None
        self.prediction_locked = False
        print("[RESET] Trajectory cleared and unlocked - ready for new throw")
    
    def calibrate_robot_origin(self):
        """Set current robot position as origin"""
        self.robot_tracker.set_origin()
        print("[CALIBRATE] Robot origin set")
    
    def reset_robot_origin(self):
        """Reset robot origin calibration"""
        self.robot_tracker.reset_origin()
        print("[RESET] Robot origin cleared")
    
    def toggle_udp(self):
        """Toggle UDP sending"""
        self.udp_sender.toggle()
    
    def close(self):
        """Clean up resources"""
        if self.threaded_tracker:
            self.threaded_tracker.stop()
        self.zed.close()
        self.udp_sender.close()
        cv.destroyAllWindows()


# ==================== MAIN ====================

def main():
    parser = argparse.ArgumentParser(description='Ball Catcher Integration System (Optimized)')
    
    # Robot tracking args
    parser.add_argument('--marker-size', type=float, required=True,
                       help='ArUco marker size in mm')
    parser.add_argument('--left-id', type=int, default=0,
                       help='Marker ID for left side (4x4 dict)')
    parser.add_argument('--front-id', type=int, default=0,
                       help='Marker ID for front side (5x5 dict)')
    parser.add_argument('--right-id', type=int, default=0,
                       help='Marker ID for right side (6x6 dict)')
    parser.add_argument('--back-id', type=int, default=0,
                       help='Marker ID for back side (7x7 dict)')
    
    # UDP args
    parser.add_argument('--robot-ip', type=str, default=DEFAULT_ROBOT_IP,
                       help='Robot IP address')
    parser.add_argument('--robot-port', type=int, default=DEFAULT_ROBOT_PORT,
                       help='Robot UDP port')
    parser.add_argument('--no-udp', action='store_true',
                       help='Disable UDP sending')
    
    # Performance args
    parser.add_argument('--fast-depth', action='store_true',
                       help='Use PERFORMANCE depth mode (faster but less accurate)')
    parser.add_argument('--detection-scale', type=float, default=0.5,
                       help='Ball detection scale factor (0.5 = half res, default: 0.5)')
    parser.add_argument('--robot-interval', type=int, default=4,
                       help='Robot tracking interval (frames, default: 4)')
    parser.add_argument('--threaded-aruco', action='store_true',
                       help='Use threaded ArUco detection')
    
    # Display args
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug visualization')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("  BALL CATCHER INTEGRATION SYSTEM (OPTIMIZED)")
    print("=" * 60)
    print(f"\nPerformance settings:")
    print(f"  - Detection scale: {args.detection_scale}")
    print(f"  - Fast depth: {args.fast_depth}")
    print(f"  - Robot tracking interval: every {args.robot_interval} frames")
    print(f"  - Threaded ArUco: {args.threaded_aruco}")
    
    # Initialize system
    system = BallCatcherSystem(
        marker_size_mm=args.marker_size,
        left_id=args.left_id,
        front_id=args.front_id,
        right_id=args.right_id,
        back_id=args.back_id,
        robot_ip=args.robot_ip,
        robot_port=args.robot_port,
        enable_udp=not args.no_udp,
        debug=args.debug,
        fast_depth=args.fast_depth,
        detection_scale=args.detection_scale,
        robot_track_interval=args.robot_interval,
        use_threaded_aruco=args.threaded_aruco
    )
    
    print("\n" + "=" * 60)
    print("Starting optimized tracking...")
    print("Controls:")
    print("  [R] Reset trajectory prediction")
    print("  [C] Calibrate robot origin")
    print("  [U] Toggle UDP sending")
    print("  [D] Toggle debug visualization")
    print("  [Q/ESC] Quit")
    print("=" * 60 + "\n")
    
    try:
        while True:
            # Process frame
            frame = system.process_frame()
            
            if frame is not None:
                cv.imshow("Ball Catcher System", frame)
            
            # Handle keyboard input (non-blocking)
            key = cv.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # Q or ESC
                break
            elif key == ord('r'):
                system.reset_trajectory()
            elif key == ord('c'):
                system.calibrate_robot_origin()
            elif key == ord('u'):
                system.toggle_udp()
            elif key == ord('d'):
                system.debug = not system.debug
                print(f"[DEBUG] Debug visualization: {'ON' if system.debug else 'OFF'}")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    finally:
        # Calculate final stats
        elapsed = time.time() - system.start_time
        avg_fps = system.frame_count / elapsed if elapsed > 0 else 0
        
        print("\n" + "=" * 60)
        print(f"✓ Total frames: {system.frame_count}")
        print(f"✓ Average FPS: {avg_fps:.1f}")
        print(f"✓ UDP packets sent: {system.udp_sender.packets_sent}")
        print("✓ Ball catcher system stopped")
        print("=" * 60)
        
        system.close()


if __name__ == "__main__":
    main()
