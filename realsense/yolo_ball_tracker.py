"""
Dual-Camera Tennis Ball Tracker (Custom Stereo Implementation)
==============================================================
Camera 0 (Left/Origin): 341522302002
Camera 1 (Right/Rel):   213522253879

This script performs custom stereo triangulation using calibration files
(stereo_calib.npz) instead of the RealSense SDK's internal depth engine.

It calculates 3D coordinates (X, Y, Z) by detecting the ball in both
camera frames and triangulating the position mathematically.

NOTE: Resolution set to 1280x720 to match provided calibration.
"""

import numpy as np
import pyrealsense2 as rs
import cv2
import os
import time
import threading
import argparse
from ultralytics import YOLO
from collections import deque
from typing import Optional, Tuple
from dataclasses import dataclass

# ===================== CONFIGURATION =====================
# Serial numbers
CAMERA0_SERIAL = "341522302002"
CAMERA1_SERIAL = "213522253879"

# Model
YOLO_MODEL = "yolov8n.pt"
CONFIDENCE = 0.35
# UPDATED: Resolution must match calibration (cx~632 implies 1280 width)
RESOLUTION = (1280, 720) 
FPS = 30

# Calibration Path
CALIB_FILE = "../calib_output/stereo_calib.npz"

# HSV Fallback
HSV_LOWER = np.array([29, 86, 6])
HSV_UPPER = np.array([64, 255, 255])

# Optimization
USE_GPU = True
SKIP_CAM1_FRAMES = 2  # Detect on Cam1 every N frames

# ===================== DATA STRUCTURES =====================

@dataclass
class Detection:
    center: Tuple[float, float] # (x, y)
    radius: float
    confidence: float
    method: str
    camera_id: int

class StereoCalibration:
    """Holds calibration data and projection matrices."""
    def __init__(self, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Calibration file not found: {filepath}")
        
        print(f"Loading calibration from {filepath}...")
        data = np.load(filepath)
        
        # Intrinsic Matrices (3x3)
        self.K0 = data['K0']
        self.K1 = data['K1']
        
        # Distortion Coefficients
        self.D0 = data['dist0']
        self.D1 = data['dist1']
        
        # Extrinsics (Rotation and Translation of Cam1 relative to Cam0)
        self.R = data['R']
        self.T = data['T']
        
        # Construct Projection Matrices (3x4)
        # P = K * [R | t]
        
        # Camera 0 is the origin (Identity rotation, Zero translation)
        # P0 = K0 * [I | 0]
        self.P0 = self.K0 @ np.hstack((np.eye(3), np.zeros((3, 1))))
        
        # Camera 1 is relative to Camera 0
        # P1 = K1 * [R | T]
        self.P1 = self.K1 @ np.hstack((self.R, self.T))
        
        print("✓ Calibration loaded successfully")
        print(f"  Baseline: {np.linalg.norm(self.T):.4f} meters")
        print(f"  Principal Point Cam0: ({self.K0[0,2]:.1f}, {self.K0[1,2]:.1f})")

# ===================== BALL DETECTOR =====================

class BallDetector:
    def __init__(self):
        print("Loading YOLO...")
        self.model = YOLO(YOLO_MODEL)
        if USE_GPU:
            try:
                self.model.to("cuda")
                print("✓ YOLO using CUDA GPU")
            except Exception as e:
                print(f"  GPU unavailable, using CPU: {e}")
        
        # Warmup
        self.model.predict(np.zeros((320, 320, 3), dtype=np.uint8), verbose=False)
    
    def detect(self, image: np.ndarray, cam_id: int) -> Optional[Detection]:
        # 1. Try YOLO
        # We process at smaller size for speed, but coordinates are returned in original scale
        results = self.model.predict(image, conf=CONFIDENCE, classes=[32], 
                                      verbose=False, imgsz=320)
        
        for result in results:
            if len(result.boxes) > 0:
                box = result.boxes[0]
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                r = max(x2 - x1, y2 - y1) / 2
                if 5 < r < 300: # Increased max radius for higher res
                    return Detection((cx, cy), r, float(box.conf[0]), 'YOLO', cam_id)
        
        # 2. HSV Fallback
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:3]:
            area = cv2.contourArea(cnt)
            if 200 < area < 50000:
                perim = cv2.arcLength(cnt, True)
                if perim > 0 and 4 * np.pi * area / perim**2 > 0.6:
                    (cx, cy), r = cv2.minEnclosingCircle(cnt)
                    if 5 < r < 200:
                        return Detection((cx, cy), r, 0.8, 'HSV', cam_id)
        return None

# ===================== CAMERA HANDLING =====================

class ThreadedCamera:
    """Background thread for Secondary Camera frame capture."""
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
    
    def _capture_loop(self):
        while self.running:
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=100)
                color = frames.get_color_frame()
                if color:
                    frame_data = np.asanyarray(color.get_data())
                    with self.lock:
                        self.frame = frame_data
            except Exception:
                pass
    
    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None
    
    def stop(self):
        self.running = False
        self.thread.join(timeout=1.0)

class DualCameraSystem:
    def __init__(self):
        print("Initializing cameras...")
        
        # Load Calibration
        self.calib = StereoCalibration(CALIB_FILE)
        self.detector = BallDetector()
        
        self._init_cameras()
        
        self.position_history = deque(maxlen=50)
        self.frame_count = 0
        self.last_det1 = None
    
    def _init_cameras(self):
        ctx = rs.context()
        devices = {d.get_info(rs.camera_info.serial_number): d for d in ctx.query_devices()}
        
        if CAMERA0_SERIAL not in devices or CAMERA1_SERIAL not in devices:
            found = list(devices.keys())
            raise RuntimeError(f"Required cameras not found.\nLooking for: {CAMERA0_SERIAL}, {CAMERA1_SERIAL}\nFound: {found}")
        
        # Camera 0 (Main)
        self.pipe0 = rs.pipeline()
        cfg0 = rs.config()
        cfg0.enable_device(CAMERA0_SERIAL)
        cfg0.enable_stream(rs.stream.color, RESOLUTION[0], RESOLUTION[1], rs.format.bgr8, FPS)
        self.pipe0.start(cfg0)
        print(f"Cam0 ({CAMERA0_SERIAL}) started at {RESOLUTION}")
        
        # Camera 1 (Secondary)
        pipe1 = rs.pipeline()
        cfg1 = rs.config()
        cfg1.enable_device(CAMERA1_SERIAL)
        cfg1.enable_stream(rs.stream.color, RESOLUTION[0], RESOLUTION[1], rs.format.bgr8, FPS)
        pipe1.start(cfg1)
        print(f"Cam1 ({CAMERA1_SERIAL}) started (threaded) at {RESOLUTION}")
        
        self.cam1_thread = ThreadedCamera(pipe1)
        
    def get_frames(self):
        # Cam0 is blocking
        frames0 = self.pipe0.wait_for_frames()
        c0 = np.asanyarray(frames0.get_color_frame().get_data())
        c1 = self.cam1_thread.get_frame()
        return c0, c1
    
    def triangulate_position(self, det0: Detection, det1: Detection) -> Optional[np.ndarray]:
        if not det0 or not det1:
            return None
            
        # 1. Prepare points (N, 1, 2)
        pt0 = np.array([[[det0.center[0], det0.center[1]]]], dtype=np.float32)
        pt1 = np.array([[[det1.center[0], det1.center[1]]]], dtype=np.float32)
        
        # 2. Undistort
        undist0 = cv2.undistortPoints(pt0, self.calib.K0, self.calib.D0, P=self.calib.K0)
        undist1 = cv2.undistortPoints(pt1, self.calib.K1, self.calib.D1, P=self.calib.K1)
        
        # 3. Triangulate
        point_4d = cv2.triangulatePoints(self.calib.P0, self.calib.P1, undist0, undist1)
        point_3d = point_4d[:3] / point_4d[3]
        
        return point_3d.flatten() 

    def stop(self):
        self.cam1_thread.stop()
        self.pipe0.stop()

    def run(self):
        print(f"\nSTARTING CUSTOM STEREO TRACKER")
        print(f"Cameras: {CAMERA0_SERIAL} (L) / {CAMERA1_SERIAL} (R)")
        print(f"Resolution: {RESOLUTION} (Matched to Calibration)")
        
        fps_dq = deque(maxlen=30)
        
        try:
            while True:
                t0 = time.time()
                self.frame_count += 1
                
                frame0, frame1 = self.get_frames()
                if frame0 is None: continue
                
                det0 = self.detector.detect(frame0, 0)
                
                if frame1 is not None and self.frame_count % SKIP_CAM1_FRAMES == 0:
                    det1 = self.detector.detect(frame1, 1)
                    self.last_det1 = det1
                else:
                    det1 = self.last_det1
                
                pos_3d = self.triangulate_position(det0, det1)
                
                fps_dq.append(time.time() - t0)
                fps = 1.0 / (sum(fps_dq) / len(fps_dq))
                
                # --- VISUALIZATION ---
                
                if det0:
                    cv2.circle(frame0, (int(det0.center[0]), int(det0.center[1])), 
                              int(det0.radius), (0, 255, 0), 2)
                    cv2.putText(frame0, f"{det0.method}", (int(det0.center[0])+10, int(det0.center[1])),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # UI Overlay
                cv2.rectangle(frame0, (0, 0), (640, 100), (0, 0, 0), -1)
                cv2.putText(frame0, "STEREO TRACKER (1280x720)", (10, 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                status_color = (0, 255, 0) if (det0 and det1) else (0, 0, 255)
                status_text = "TRACKING 3D" if (det0 and det1) else "SEARCHING"
                
                if pos_3d is not None:
                    x, y, z = pos_3d
                    # Display X, Y, Z
                    cv2.putText(frame0, f"X: {x:.3f} m (Right)", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    cv2.putText(frame0, f"Y: {y:.3f} m (Down)", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    cv2.putText(frame0, f"Z: {z:.3f} m (Depth)", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                else:
                    cv2.putText(frame0, status_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
                
                cv2.putText(frame0, f"FPS: {fps:.1f}", (10, 700), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                # Resize for display if screen is small
                display_frame = cv2.resize(frame0, (960, 540))
                cv2.imshow("Custom Stereo Tracker", display_frame)
                
                k = cv2.waitKey(1) & 0xFF
                if k == ord('q'):
                    break
                elif k == ord('r'):
                    self.position_history.clear()
                    print("History Reset")
                    
        finally:
            self.stop()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = DualCameraSystem()
    tracker.run()
