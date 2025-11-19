import numpy as np
import cv2 as cv
from scipy.optimize import least_squares
from scipy.interpolate import interp1d
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import time
from collections import deque
import threading
import pyzed.sl as sl


@dataclass
class Point3D:
    x: float
    y: float
    z: float
    timestamp: float
    
    def to_array(self):
        return np.array([self.x, self.y, self.z])
    
    def __sub__(self, other):
        return Point3D(
            self.x - other.x,
            self.y - other.y,
            self.z - other.z,
            self.timestamp
        )


@dataclass
class CameraCalibration:
    fx: float
    fy: float
    cx: float
    cy: float
    R: np.ndarray
    t: np.ndarray
    distortion: Optional[np.ndarray] = None
    
    @property
    def K(self):
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ])
    
    @property
    def P(self):
        return self.K @ np.hstack([self.R, self.t.reshape(-1, 1)])


class TrajectoryPredictor:
    
    def __init__(self, gravity: float = 9.81, drag_coefficient: float = 0.47):
        self.g = gravity
        self.Cd = drag_coefficient
        self.rho = 1.225  # Air density at sea level (kg/m^3)
        self.ball_radius = 0.035  # Typical paper ball radius (m)
        self.ball_mass = 0.005  # Typical paper ball mass (kg)
        self.history_size = 10
        self.position_history = deque(maxlen=self.history_size)
        
    def add_observation(self, point: Point3D):
        self.position_history.append(point)
    
    def estimate_initial_conditions(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if len(self.position_history) < 3:
            return None
        
        positions = np.array([p.to_array() for p in self.position_history])
        times = np.array([p.timestamp for p in self.position_history])
        
        # Fit polynomial to estimate velocity
        dt = times[-1] - times[0]
        if dt < 0.01:  # Need reasonable time difference
            return None
        
        # Estimate velocity using finite differences
        velocities = []
        for i in range(1, len(positions)):
            dt = times[i] - times[i-1]
            if dt > 0:
                v = (positions[i] - positions[i-1]) / dt
                velocities.append(v)
        
        if not velocities:
            return None
            
        # Use most recent position and average of recent velocities
        pos0 = positions[-1]
        vel0 = np.mean(velocities[-3:], axis=0) if len(velocities) >= 3 else velocities[-1]
        
        return pos0, vel0
    
    def predict_trajectory(self, 
                          initial_pos: Optional[np.ndarray] = None,
                          initial_vel: Optional[np.ndarray] = None,
                          time_span: float = 3.0,
                          dt: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
        if initial_pos is None or initial_vel is None:
            estimated = self.estimate_initial_conditions()
            if estimated is None:
                return np.array([]), np.array([])
            initial_pos, initial_vel = estimated
        
        # Calculate drag parameters
        A = np.pi * self.ball_radius**2  # Cross-sectional area
        drag_factor = 0.5 * self.Cd * self.rho * A / self.ball_mass
        
        num_steps = int(time_span / dt)
        times = np.linspace(0, time_span, num_steps)
        positions = np.zeros((num_steps, 3))
        velocities = np.zeros((num_steps, 3))
        
        positions[0] = initial_pos
        velocities[0] = initial_vel
        
        # Integrate using Euler method with drag
        for i in range(1, num_steps):
            v = velocities[i-1]
            v_mag = np.linalg.norm(v)
            
            # Acceleration due to gravity and drag
            a_gravity = np.array([0, 0, -self.g])
            a_drag = -drag_factor * v_mag * v if v_mag > 0 else np.zeros(3)
            a_total = a_gravity + a_drag
            
            # Update velocity and position
            velocities[i] = velocities[i-1] + a_total * dt
            positions[i] = positions[i-1] + velocities[i-1] * dt + 0.5 * a_total * dt**2
            
            # Stop if ball hits ground
            if positions[i, 2] <= 0:
                positions = positions[:i+1]
                times = times[:i+1]
                break
        
        return times, positions
    
    def predict_landing_point(self) -> Optional[Tuple[np.ndarray, float]]:
        times, positions = self.predict_trajectory()
        
        if len(positions) == 0:
            return None
        
        # Find when z crosses zero
        z_values = positions[:, 2]
        
        # Find the last point before ground and first point after
        above_ground = z_values > 0
        if not np.any(above_ground):
            return None
            
        last_above_idx = np.where(above_ground)[0][-1]
        
        if last_above_idx >= len(z_values) - 1:
            # Ball hasn't hit ground yet in prediction
            return None
        
        # Interpolate to find exact landing point
        z1, z2 = z_values[last_above_idx], z_values[last_above_idx + 1]
        t1, t2 = times[last_above_idx], times[last_above_idx + 1]
        pos1, pos2 = positions[last_above_idx], positions[last_above_idx + 1]
        
        # Linear interpolation for landing time
        t_land = t1 + (0 - z1) * (t2 - t1) / (z2 - z1)
        
        # Interpolate position at landing
        alpha = (t_land - t1) / (t2 - t1)
        pos_land = pos1 + alpha * (pos2 - pos1)
        pos_land[2] = 0  # Ensure z is exactly 0
        
        return pos_land, t_land


class RobotMotionPlanner:
    
    def __init__(self, 
                 max_velocity: float = 1.5,
                 max_acceleration: float = 2.0,
                 robot_radius: float = 0.2):
        self.v_max = max_velocity
        self.a_max = max_acceleration
        self.robot_radius = robot_radius
        self.current_position = np.array([0, 0, 0])
        self.current_velocity = np.array([0, 0, 0])
        self.catch_height = 0.3  # Height at which to catch (m)
        
    def update_robot_state(self, position: np.ndarray, velocity: Optional[np.ndarray] = None):
        self.current_position = position.copy()
        if velocity is not None:
            self.current_velocity = velocity.copy()
    
    def plan_interception(self, 
                         landing_point: np.ndarray,
                         time_to_landing: float,
                         safety_margin: float = 0.2) -> Optional[Dict]:
        target_pos = landing_point[:2]
        current_pos = self.current_position[:2]
        
        # Distance to travel
        distance = np.linalg.norm(target_pos - current_pos)
        
        # Time available (with safety margin)
        time_available = max(0, time_to_landing - safety_margin)
        
        if time_available <= 0:
            return None
        
        # Check if interception is possible
        # Using trapezoidal velocity profile
        t_accel = self.v_max / self.a_max
        d_accel = 0.5 * self.a_max * t_accel**2
        
        if 2 * d_accel > distance:
            # Short distance - triangular profile
            t_total = 2 * np.sqrt(distance / self.a_max)
            if t_total > time_available:
                return None
            v_peak = np.sqrt(distance * self.a_max)
        else:
            # Long distance - trapezoidal profile
            d_const = distance - 2 * d_accel
            t_const = d_const / self.v_max
            t_total = 2 * t_accel + t_const
            
            if t_total > time_available:
                # Try to reach with reduced max velocity
                v_reduced = distance / time_available + self.a_max * time_available / 2
                if v_reduced > self.v_max:
                    return None
                v_peak = min(v_reduced, self.v_max)
            else:
                v_peak = self.v_max
        
        # Generate trajectory waypoints
        direction = (target_pos - current_pos) / distance if distance > 0 else np.array([0, 0])
        
        waypoints = []
        velocities = []
        timestamps = []
        
        # Sample trajectory at regular intervals
        dt = 0.05  # 50ms intervals
        t = 0
        
        while t <= t_total:
            if t <= t_accel:
                # Acceleration phase
                s = 0.5 * self.a_max * t**2
                v = self.a_max * t
            elif t <= t_total - t_accel:
                # Constant velocity phase
                s = d_accel + v_peak * (t - t_accel)
                v = v_peak
            else:
                # Deceleration phase
                t_decel = t - (t_total - t_accel)
                s = distance - 0.5 * self.a_max * (t_accel - t_decel)**2
                v = v_peak - self.a_max * t_decel
            
            s = min(s, distance)  # Clamp to target
            pos = current_pos + direction * s
            vel = direction * v
            
            waypoints.append(np.array([pos[0], pos[1], self.current_position[2]]))
            velocities.append(np.array([vel[0], vel[1], 0]))
            timestamps.append(t)
            
            t += dt
        
        # Add final waypoint at exact target
        waypoints.append(np.array([target_pos[0], target_pos[1], self.current_position[2]]))
        velocities.append(np.array([0, 0, 0]))
        timestamps.append(t_total)
        
        return {
            'waypoints': np.array(waypoints),
            'velocities': np.array(velocities),
            'timestamps': np.array(timestamps),
            'total_time': t_total,
            'distance': distance,
            'feasible': True,
            'arrival_time': t_total,
            'time_margin': time_available - t_total
        }
    
    def compute_control_command(self, plan: Dict, current_time: float) -> np.ndarray:
        if plan is None or not plan['feasible']:
            return np.array([0, 0, 0])
        
        timestamps = plan['timestamps']
        velocities = plan['velocities']
        
        if current_time >= timestamps[-1]:
            # Plan completed
            return np.array([0, 0, 0])
        
        if current_time <= 0:
            return velocities[0]
        
        # Interpolate velocity command
        f_vx = interp1d(timestamps, velocities[:, 0], kind='linear', fill_value='extrapolate')
        f_vy = interp1d(timestamps, velocities[:, 1], kind='linear', fill_value='extrapolate')
        
        vx = float(f_vx(current_time))
        vy = float(f_vy(current_time))
        
        return np.array([vx, vy, 0])


class MultiCameraTracker:
    
    def __init__(self, calibrations: List[CameraCalibration]):
        self.calibrations = calibrations
        self.num_cameras = len(calibrations)
        
    def triangulate_point(self, 
                         observations: List[Tuple[float, float]], 
                         camera_indices: List[int]) -> Optional[np.ndarray]:
        if len(observations) < 2:
            return None
        
        # Build system of equations for DLT
        A = []
        for (x, y), cam_idx in zip(observations, camera_indices):
            P = self.calibrations[cam_idx].P
            A.append(x * P[2] - P[0])
            A.append(y * P[2] - P[1])
        
        A = np.array(A)
        
        # Solve using SVD
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        X = X[:3] / X[3]  # Convert from homogeneous
        
        return X
    
    def triangulate_with_ransac(self,
                               observations: List[Tuple[float, float]],
                               camera_indices: List[int],
                               max_iterations: int = 100,
                               inlier_threshold: float = 5.0) -> Optional[np.ndarray]:
        if len(observations) < 2:
            return None
        
        best_point = None
        best_inliers = 0
        
        for _ in range(max_iterations):
            # Random sample of 2 observations
            if len(observations) == 2:
                sample_idx = [0, 1]
            else:
                sample_idx = np.random.choice(len(observations), 2, replace=False)
            
            sample_obs = [observations[i] for i in sample_idx]
            sample_cams = [camera_indices[i] for i in sample_idx]
            
            # Triangulate with sample
            point = self.triangulate_point(sample_obs, sample_cams)
            if point is None:
                continue
            
            # Count inliers
            inliers = 0
            for (x, y), cam_idx in zip(observations, camera_indices):
                # Project point to camera
                P = self.calibrations[cam_idx].P
                X_h = np.append(point, 1)
                x_proj = P @ X_h
                x_proj = x_proj[:2] / x_proj[2]
                
                # Check reprojection error
                error = np.linalg.norm(x_proj - np.array([x, y]))
                if error < inlier_threshold:
                    inliers += 1
            
            if inliers > best_inliers:
                best_inliers = inliers
                best_point = point
        
        return best_point


class BallDetector:
    
    def __init__(self, hsv_lower=(0, 0, 200), hsv_upper=(180, 30, 255)):
        self.hsv_lower = np.array(hsv_lower)
        self.hsv_upper = np.array(hsv_upper)
        self.min_area = 100
        self.max_area = 10000
        
    def detect_ball(self, image: np.ndarray) -> Optional[Tuple[float, float, float]]:
        # Convert to HSV
        hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        
        # Threshold for white/light colored ball
        mask = cv.inRange(hsv, self.hsv_lower, self.hsv_upper)
        
        # Morphological operations to clean up
        kernel = np.ones((5, 5), np.uint8)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Find the most circular contour
        best_circle = None
        best_circularity = 0
        
        for contour in contours:
            area = cv.contourArea(contour)
            if area < self.min_area or area > self.max_area:
                continue
            
            perimeter = cv.arcLength(contour, True)
            if perimeter == 0:
                continue
            
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            if circularity > best_circularity and circularity > 0.7:
                # Fit circle to contour
                (x, y), radius = cv.minEnclosingCircle(contour)
                best_circle = (x, y, radius)
                best_circularity = circularity
        
        return best_circle


def main_tracking_loop():
    # Initialize components
    predictor = TrajectoryPredictor()
    planner = RobotMotionPlanner()
    detector = BallDetector()
    
    # Initialize ZED cameras (example for 2 cameras)
    cameras = []
    calibrations = []
    
    for i in range(2):
        zed = sl.Camera()
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD720
        init_params.camera_fps = 60
        init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
        init_params.coordinate_units = sl.UNIT.METER
        
        if zed.open(init_params) == sl.ERROR_CODE.SUCCESS:
            cameras.append(zed)
            
            info = zed.get_camera_information()
            calib_params = info.calibration_parameters
            
            calib = CameraCalibration(
                fx=calib_params.left_cam.fx,
                fy=calib_params.left_cam.fy,
                cx=calib_params.left_cam.cx,
                cy=calib_params.left_cam.cy,
                R=np.eye(3),  # Identity for first camera
                t=np.array([i * 2.0, 0, 1.5])  # Example positioning
            )
            calibrations.append(calib)
    
    if len(cameras) < 2:
        print("Need at least 2 cameras for multi-view tracking")
        return
    
    tracker = MultiCameraTracker(calibrations)
    
    print("Starting tracking loop...")
    last_plan = None
    plan_start_time = None
    
    try:
        while True:
            # Grab frames from all cameras
            observations = []
            camera_indices = []
            
            for i, zed in enumerate(cameras):
                image = sl.Mat()
                if zed.grab() == sl.ERROR_CODE.SUCCESS:
                    zed.retrieve_image(image, sl.VIEW.LEFT)
                    frame = image.get_data()[:, :, :3]
                    
                    # Detect ball
                    detection = detector.detect_ball(frame)
                    if detection:
                        x, y, radius = detection
                        observations.append((x, y))
                        camera_indices.append(i)
                        
                        # Draw detection
                        cv.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)
                    
                    cv.imshow(f"Camera {i}", frame)
            
            # Triangulate if we have observations
            if len(observations) >= 2:
                point_3d = tracker.triangulate_with_ransac(observations, camera_indices)
                
                if point_3d is not None:
                    # Add observation to predictor
                    predictor.add_observation(Point3D(
                        point_3d[0], point_3d[1], point_3d[2],
                        time.time()
                    ))
                    
                    # Try to predict landing
                    landing = predictor.predict_landing_point()
                    
                    if landing is not None:
                        landing_pos, time_to_land = landing
                        print(f"Predicted landing: {landing_pos[:2]} in {time_to_land:.2f}s")
                        
                        # Plan interception
                        plan = planner.plan_interception(landing_pos, time_to_land)
                        
                        if plan and plan['feasible']:
                            print(f"Interception planned: {plan['distance']:.2f}m in {plan['total_time']:.2f}s")
                            last_plan = plan
                            plan_start_time = time.time()
            
            # Execute plan if we have one
            if last_plan and plan_start_time:
                elapsed = time.time() - plan_start_time
                command = planner.compute_control_command(last_plan, elapsed)
                print(f"Robot command: vx={command[0]:.2f}, vy={command[1]:.2f} m/s")
            
            if cv.waitKey(1) & 0xFF == 27:  # ESC to quit
                break
                
    finally:
        cv.destroyAllWindows()
        for zed in cameras:
            zed.close()


if __name__ == "__main__":
    main_tracking_loop()
