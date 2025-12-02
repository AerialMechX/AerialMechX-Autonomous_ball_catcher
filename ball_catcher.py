"""
Ball Catcher Integration System

Integrates trajectory prediction and robot tracking to enable a robot to catch thrown balls.
This module combines:
- Ball trajectory prediction from trajectory_predictor.py
- Robot pose tracking from aruco_robot_tracker.py  
- Combined UDP communication to send both ball landing and robot pose data

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
from typing import Optional

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

# Import existing components
from trajectory_predictor import TrajectoryPredictor, LandingPrediction
from ball_tracker_params import BallDetector, CameraIntrinsics as BallCameraIntrinsics
from aruco_robot_tracker import (
    MultiMarkerRobotTracker, 
    Pose as RobotPose,
    CameraIntrinsics as RobotCameraIntrinsics
)
from sender import UDPSender
from config import (
    RESOLUTION, FPS, DEPTH_MODE, UNIT,
    DEFAULT_ROBOT_IP, DEFAULT_ROBOT_PORT
)


# ==================== COMBINED UDP SENDER ====================

class CombinedUDPSender(UDPSender):
    """Enhanced UDP sender that sends combined ball + robot data"""
    
    def send_combined_data(self, 
                          ball_landing: Optional[np.ndarray] = None,
                          ball_confidence: float = 0.0,
                          robot_pose: Optional[RobotPose] = None):
        """
        Send combined ball landing prediction and robot pose data.
        
        Packet format (40 bytes):
            - Ball landing (x, y, z): 3 floats (12 bytes)
            - Ball confidence: 1 float (4 bytes)
            - Robot pose (x, y, z, roll, pitch, yaw): 6 floats (24 bytes)
        
        If ball or robot data is missing, zeros are sent for that component.
        """
        if not self.enabled:
            return
        
        # Check rate limit
        current_time = time.time()
        if current_time - self.last_send_time < self.min_send_interval:
            return
        
        try:
            # Prepare ball data (default to zeros if no prediction)
            if ball_landing is not None and len(ball_landing) >= 3:
                ball_x, ball_y, ball_z = float(ball_landing[0]), float(ball_landing[1]), float(ball_landing[2])
            else:
                ball_x, ball_y, ball_z = 0.0, 0.0, 0.0
                ball_confidence = 0.0
            
            # Prepare robot data (default to zeros if no pose)
            if robot_pose is not None:
                robot_x, robot_y, robot_z = float(robot_pose.x), float(robot_pose.y), float(robot_pose.z)
                robot_roll, robot_pitch, robot_yaw = float(robot_pose.roll), float(robot_pose.pitch), float(robot_pose.yaw)
            else:
                robot_x, robot_y, robot_z = 0.0, 0.0, 0.0
                robot_roll, robot_pitch, robot_yaw = 0.0, 0.0, 0.0
            
            # Pack data: 10 floats = 40 bytes
            data = struct.pack('10f',
                             ball_x, ball_y, ball_z, ball_confidence,
                             robot_x, robot_y, robot_z, 
                             robot_roll, robot_pitch, robot_yaw)
            
            self.sock.sendto(data, (self.robot_ip, self.port))
            self.packets_sent += 1
            self.last_send_time = current_time
            
        except Exception as e:
            print(f"UDP send error: {e}")


# ==================== BALL CATCHER SYSTEM ====================

class BallCatcherSystem:
    """Main integration class coordinating ball tracking and robot tracking"""
    
    def __init__(self, 
                 marker_size_mm: float,
                 left_id: int = 0, front_id: int = 0, right_id: int = 0, back_id: int = 0,
                 robot_ip: str = DEFAULT_ROBOT_IP,
                 robot_port: int = DEFAULT_ROBOT_PORT,
                 enable_udp: bool = True,
                 debug: bool = False):
        
        self.debug = debug
        
        # Initialize ZED camera
        print("\nInitializing ZED camera...")
        self.zed = sl.Camera()
        
        init_params = sl.InitParameters()
        init_params.camera_resolution = RESOLUTION
        init_params.camera_fps = FPS
        init_params.depth_mode = DEPTH_MODE
        init_params.coordinate_units = UNIT
        
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
            # Fallback to default resolution
            width, height = 1280, 720
        
        fx, fy = calib.fx, calib.fy
        cx, cy = calib.cx, calib.cy
        
        # Get distortion coefficients
        try:
            dist = np.array(calib.disto[:5], dtype=np.float32)
        except:
            dist = np.zeros(5, dtype=np.float32)
        
        # Create camera intrinsics for ball detector
        ball_intrinsics = BallCameraIntrinsics(
            fx=fx, fy=fy, cx=cx, cy=cy,
            width=width, height=height
        )
        
        # Create camera intrinsics for robot tracker
        robot_intrinsics = RobotCameraIntrinsics(
            fx=fx, fy=fy, cx=cx, cy=cy,
            width=width, height=height,
            dist_coeffs=dist
        )
        
        print(f"✓ Camera resolution: {width}x{height}")
        print(f"✓ Camera intrinsics: fx={fx:.2f}, fy={fy:.2f}")
        
        # Initialize ball tracking components
        print("\nInitializing ball tracker...")
        from config import ACTIVE_HSV  # Import HSV configuration
        self.ball_detector = BallDetector(ACTIVE_HSV)
        self.trajectory_predictor = TrajectoryPredictor(debug=debug)
        self.ball_intrinsics = ball_intrinsics
        print("✓ Ball tracker initialized")
        
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
        print(f"✓ Robot tracker initialized (marker size: {marker_size_mm}mm)")
        
        # Initialize combined UDP sender
        self.udp_sender = CombinedUDPSender(robot_ip=robot_ip, port=robot_port)
        if not enable_udp:
            self.udp_sender.enabled = False
        
        print(f"\n{'✓' if self.udp_sender.enabled else '✗'} UDP: {robot_ip}:{robot_port}")
        
        # ZED image containers
        self.image_left = sl.Mat()
        self.depth_map = sl.Mat()
        self.point_cloud = sl.Mat()
        self.runtime = sl.RuntimeParameters()
        
        # State
        self.current_prediction = None
        self.current_robot_pose = None
        self.frame_count = 0
        self.start_time = time.time()
        self.prediction_locked = False  # Lock prediction after first successful throw
        
    def process_frame(self):
        """Process one frame: detect ball, track robot, send UDP"""
        
        # Grab frame from ZED
        if self.zed.grab(self.runtime) != sl.ERROR_CODE.SUCCESS:
            return None
        
        # Always retrieve image
        self.zed.retrieve_image(self.image_left, sl.VIEW.LEFT)
        frame = self.image_left.get_data()
        
        if frame is None:
            return None
        
        self.frame_count += 1
        current_time = time.time()
        
        # ===== BALL TRACKING (OPTIMIZED) =====
        ball_pos = None
        detection = None
        
        # Always detect ball for visualization, even when locked
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        detection = self.ball_detector.detect(hsv)
        
        if detection:
            x, y, radius = detection
            
            # Only retrieve depth/point cloud when ball is detected (MAJOR OPTIMIZATION)
            self.zed.retrieve_measure(self.depth_map, sl.MEASURE.DEPTH)
            self.zed.retrieve_measure(self.point_cloud, sl.MEASURE.XYZRGBA)
            
            # Get 3D position from point cloud/depth
            from ball_tracker_params import get_robust_depth
            ball_pos, depth_conf = get_robust_depth(
                self.depth_map, self.point_cloud, x, y, self.ball_intrinsics
            )
            
            if ball_pos is not None:
                # Apply coordinate offset corrections
                from config import CAMERA_HEIGHT_ABOVE_GROUND, DEPTH_OFFSET
                ball_pos[1] += CAMERA_HEIGHT_ABOVE_GROUND
                ball_pos[2] += DEPTH_OFFSET
                
                # Only add to trajectory predictor if not locked
                if not self.prediction_locked:
                    self.trajectory_predictor.add_point(ball_pos, current_time)
        
        # Predict landing only if not locked
        if not self.prediction_locked:
            # Predict landing (every 3 frames)
            if self.trajectory_predictor.can_predict() and self.frame_count % 3 == 0:
                new_prediction = self.trajectory_predictor.predict_landing()
                if new_prediction is not None:
                    self.current_prediction = new_prediction
                    self.prediction_locked = True
                    print("[LOCKED] Prediction locked. Press 'R' to reset.")
            
            # Check timeout
            self.trajectory_predictor.check_timeout(current_time)
        
        # ===== ROBOT TRACKING (OPTIMIZED - every 2 frames) =====
        if self.frame_count % 2 == 0:
            self.current_robot_pose, detections = self.robot_tracker.update(frame)
        else:
            detections = []
        
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
        
        # ===== VISUALIZATION =====
        h, w = frame.shape[:2]
        y_pos = 30
        
        # --------------- LEFT SIDE ---------------
        # Header
        cv.putText(frame, "Ball Catcher System", (10, y_pos),
                  cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_pos += 30
        
        # FPS and UDP status
        elapsed = current_time - self.start_time
        fps = self.frame_count / elapsed if elapsed > 0 else 0
        cv.putText(frame, f"FPS: {fps:.1f}", (10, y_pos),
                  cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        y_pos += 20
        
        udp_status = "ON" if self.udp_sender.enabled else "OFF"
        udp_color = (0, 255, 0) if self.udp_sender.enabled else (0, 0, 255)
        cv.putText(frame, f"UDP: {udp_status} ({self.udp_sender.packets_sent})", 
                  (10, y_pos), cv.FONT_HERSHEY_SIMPLEX, 0.5, udp_color, 1)
        y_pos += 30
        
        # Current ball position
        if ball_pos is not None:
            cv.putText(frame, f"Ball Position:", (10, y_pos),
                      cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
            y_pos += 20
            cv.putText(frame, f"  X: {ball_pos[0]:.3f} m", (10, y_pos),
                      cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1)
            y_pos += 18
            cv.putText(frame, f"  Y: {ball_pos[1]:.3f} m", (10, y_pos),
                      cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1)
            y_pos += 18
            cv.putText(frame, f"  Z: {ball_pos[2]:.3f} m", (10, y_pos),
                      cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1)
            y_pos += 25
        
        # Landing prediction
        if self.current_prediction is not None:
            landing = self.current_prediction.position
            confidence = self.current_prediction.confidence
            ttl = self.current_prediction.time_to_land
            
            pred_color = (0, 255, 255) if confidence >= 0.8 else (0, 200, 255)
            cv.putText(frame, f"Landing Prediction:", (10, y_pos),
                      cv.FONT_HERSHEY_SIMPLEX, 0.5, pred_color, 1)
            y_pos += 20
            cv.putText(frame, f"  X: {landing[0]:.3f} m", (10, y_pos),
                      cv.FONT_HERSHEY_SIMPLEX, 0.4, pred_color, 1)
            y_pos += 18
            cv.putText(frame, f"  Y: {landing[1]:.3f} m", (10, y_pos),
                      cv.FONT_HERSHEY_SIMPLEX, 0.4, pred_color, 1)
            y_pos += 18
            cv.putText(frame, f"  Z: {landing[2]:.3f} m", (10, y_pos),
                      cv.FONT_HERSHEY_SIMPLEX, 0.4, pred_color, 1)
            y_pos += 18
            cv.putText(frame, f"  Conf: {confidence*100:.0f}%  TTL: {ttl:.2f}s", (10, y_pos),
                      cv.FONT_HERSHEY_SIMPLEX, 0.4, pred_color, 1)
        
        # --------------- BOTTOM LEFT ---------------
        # Robot position and orientation
        if self.current_robot_pose:
            pose_deg = self.current_robot_pose.to_degrees()
            bottom_y = h - 100
            
            cv.putText(frame, f"Robot Position:", (10, bottom_y),
                      cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 0), 1)
            bottom_y += 20
            cv.putText(frame, f"  X: {pose_deg.x:.3f} m  Y: {pose_deg.y:.3f} m  Z: {pose_deg.z:.3f} m",
                      (10, bottom_y), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 128, 0), 1)
            bottom_y += 20
            cv.putText(frame, f"  Yaw: {pose_deg.yaw:.1f} deg", (10, bottom_y),
                      cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 128, 0), 1)
            
            if self.debug:
                bottom_y += 20
                cv.putText(frame, f"  Roll: {pose_deg.roll:.1f} deg  Pitch: {pose_deg.pitch:.1f} deg",
                          (10, bottom_y), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 128, 0), 1)
                bottom_y += 20
                cv.putText(frame, f"  Markers: {len(detections) if self.frame_count % 2 == 0 else 'N/A'}",
                          (10, bottom_y), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Draw ball circle on frame
        if detection and ball_pos is not None:
            x, y, radius = detection
            cv.circle(frame, (x, y), int(radius), (0, 255, 0), 2)
            cv.circle(frame, (x, y), 3, (0, 0, 255), -1)
        

        return frame
    
    def reset_trajectory(self):
        """Reset trajectory prediction"""
        self.trajectory_predictor.reset()
        self.current_prediction = None
        self.prediction_locked = False  # Unlock to allow new prediction
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
        self.zed.close()
        self.udp_sender.close()
        cv.destroyAllWindows()


# ==================== MAIN ====================

def main():
    parser = argparse.ArgumentParser(description='Ball Catcher Integration System')
    
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
    
    # Display args
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug visualization')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("  BALL CATCHER INTEGRATION SYSTEM")
    print("=" * 60)
    
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
        debug=args.debug
    )
    
    print("\n" + "=" * 60)
    print("Starting integrated tracking...")
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
            
            # Handle keyboard input
            key = cv.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # Q or ESC
                break
            elif key == ord('r'):  # Reset trajectory
                system.reset_trajectory()
            elif key == ord('c'):  # Calibrate robot origin
                system.calibrate_robot_origin()
            elif key == ord('u'):  # Toggle UDP
                system.toggle_udp()
            elif key == ord('d'):  # Toggle debug
                system.debug = not system.debug
                print(f"[DEBUG] Debug visualization: {'ON' if system.debug else 'OFF'}")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    finally:
        print("\n" + "=" * 60)
        print(f"✓ Total frames: {system.frame_count}")
        print(f"✓ UDP packets sent: {system.udp_sender.packets_sent}")
        print("✓ Ball catcher system stopped")
        print("=" * 60)
        
        system.close()


if __name__ == "__main__":
    main()
