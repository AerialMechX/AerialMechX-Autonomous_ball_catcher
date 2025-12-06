"""
Dual-Camera Tennis Ball Tracker (Optimized)
============================================
Camera 0 (origin): 341522302002
Camera 1 (secondary): 213522253879

Run: python dual_ball_tracker.py
Keys: q=quit, r=reset
"""

import numpy as np
import pyrealsense2 as rs
import cv2
from ultralytics import YOLO
from collections import deque
import time
from typing import Optional, Tuple
from dataclasses import dataclass

from stereo_calib_loader import load_calibration, CAMERA0_SERIAL, CAMERA1_SERIAL

# ===================== CONFIGURATION =====================
YOLO_MODEL = "yolov8n.pt"
CONFIDENCE = 0.35
RESOLUTION = (640, 480)
FPS = 30

HSV_LOWER = np.array([29, 86, 6])
HSV_UPPER = np.array([64, 255, 255])

MIN_DEPTH = 0.3
MAX_DEPTH = 5.0
CALIB_DIR = "calib_output"
MAX_FUSION_DISTANCE = 0.3
# =========================================================


@dataclass
class Detection:
    center: Tuple[float, float]
    radius: float
    confidence: float
    method: str
    camera_id: int


class BallDetector:
    def __init__(self):
        print("Loading YOLO...")
        self.model = YOLO(YOLO_MODEL)
        self.model.predict(np.zeros((320, 320, 3), dtype=np.uint8), verbose=False)
        print("Ready!")
    
    def detect(self, image: np.ndarray, cam_id: int) -> Optional[Detection]:
        # YOLO with smaller input
        results = self.model.predict(image, conf=CONFIDENCE, classes=[32], 
                                      verbose=False, imgsz=320)
        
        h, w = image.shape[:2]
        for result in results:
            if len(result.boxes) > 0:
                box = result.boxes[0]
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                r = max(x2 - x1, y2 - y1) / 2
                if 5 < r < 150:
                    return Detection((cx, cy), r, float(box.conf[0]), 'YOLO', cam_id)
        
        # HSV fallback
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
                    if 5 < r < 150:
                        return Detection((cx, cy), r, 0.8, 'HSV', cam_id)
        return None


class DualCameraTracker:
    def __init__(self):
        print("Initializing cameras...")
        self.calib = load_calibration(CALIB_DIR)
        self._init_cameras()
        self.detector = BallDetector()
        self.position_history = deque(maxlen=50)
        self.time_history = deque(maxlen=50)
    
    def _init_cameras(self):
        ctx = rs.context()
        devices = {d.get_info(rs.camera_info.serial_number): d for d in ctx.query_devices()}
        
        if CAMERA0_SERIAL not in devices or CAMERA1_SERIAL not in devices:
            raise RuntimeError(f"Cameras not found. Need {CAMERA0_SERIAL} and {CAMERA1_SERIAL}")
        
        self.pipelines = []
        self.intrinsics = []
        self.depth_scales = []
        self.spatial = rs.spatial_filter()
        
        for i, serial in enumerate([CAMERA0_SERIAL, CAMERA1_SERIAL]):
            pipe = rs.pipeline()
            cfg = rs.config()
            cfg.enable_device(serial)
            cfg.enable_stream(rs.stream.color, RESOLUTION[0], RESOLUTION[1], rs.format.bgr8, FPS)
            cfg.enable_stream(rs.stream.depth, RESOLUTION[0], RESOLUTION[1], rs.format.z16, FPS)
            
            profile = pipe.start(cfg)
            self.depth_scales.append(profile.get_device().first_depth_sensor().get_depth_scale())
            self.intrinsics.append(profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics())
            self.pipelines.append(pipe)
            print(f"Cam{i} ({serial}) started")
    
    def get_frames(self, cam_id: int):
        try:
            frames = self.pipelines[cam_id].wait_for_frames(timeout_ms=30)
            color = frames.get_color_frame()
            depth = self.spatial.process(frames.get_depth_frame())
            if color and depth:
                return np.asanyarray(color.get_data()), np.asanyarray(depth.get_data())
        except:
            pass
        return None, None
    
    def pixel_to_3d(self, u, v, depth_img, cam_id):
        h, w = depth_img.shape
        roi = depth_img[max(0,int(v)-5):min(h,int(v)+5), max(0,int(u)-5):min(w,int(u)+5)]
        roi_m = roi.astype(np.float32) * self.depth_scales[cam_id]
        valid = roi_m[(roi_m > MIN_DEPTH) & (roi_m < MAX_DEPTH)]
        if len(valid) == 0:
            return None
        depth = np.median(valid)
        return np.array(rs.rs2_deproject_pixel_to_point(self.intrinsics[cam_id], [u, v], depth))
    
    def fuse(self, det0, det1, depth0, depth1):
        p0 = self.pixel_to_3d(det0.center[0], det0.center[1], depth0, 0) if det0 and depth0 is not None else None
        p1 = None
        if det1 and depth1 is not None:
            p1_local = self.pixel_to_3d(det1.center[0], det1.center[1], depth1, 1)
            if p1_local is not None:
                p1 = self.calib.cam1_to_cam0(p1_local)
        
        if p0 is None and p1 is None:
            return None
        if p0 is None:
            return p1
        if p1 is None:
            return p0
        
        if np.linalg.norm(p0 - p1) > MAX_FUSION_DISTANCE:
            return p0 if det0.confidence >= det1.confidence else p1
        
        w0, w1 = det0.confidence / (p0[2] + 0.1), det1.confidence / (p1[2] + 0.1)
        return (w0 * p0 + w1 * p1) / (w0 + w1)
    
    def reset(self):
        self.position_history.clear()
        self.time_history.clear()
    
    def stop(self):
        for p in self.pipelines:
            p.stop()
    
    def run(self):
        fps_q = deque(maxlen=20)
        
        print(f"\nDUAL-CAMERA BALL TRACKER | Baseline: {self.calib.baseline*100:.1f}cm")
        print("q=quit r=reset\n")
        
        try:
            while True:
                t0 = time.time()
                
                c0, d0 = self.get_frames(0)
                c1, d1 = self.get_frames(1)
                
                if c0 is None:
                    continue
                
                det0 = self.detector.detect(c0, 0)
                det1 = self.detector.detect(c1, 1) if c1 is not None else None
                fused = self.fuse(det0, det1, d0, d1)
                
                if fused is not None:
                    self.position_history.append(fused)
                    self.time_history.append(time.time())
                
                fps_q.append(time.time() - t0)
                fps = 1.0 / (sum(fps_q) / len(fps_q))
                
                # Draw
                frame = c0.copy()
                h, w = frame.shape[:2]
                
                if det0:
                    cv2.circle(frame, (int(det0.center[0]), int(det0.center[1])), 
                              int(det0.radius), (0,255,0) if det0.method=='YOLO' else (255,255,0), 2)
                
                cv2.putText(frame, f"C1:{'Y' if det1 else 'N'}", (w-50,20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0) if det1 else (100,100,100), 1)
                
                if fused is not None:
                    lbl = "F" if det0 and det1 else "0" if det0 else "1"
                    cv2.putText(frame, f"[{lbl}] X:{fused[0]:+.2f} Y:{fused[1]:+.2f} Z:{fused[2]:+.2f}",
                               (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)
                
                cv2.putText(frame, f"FPS:{fps:.0f}", (10,h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                cv2.imshow("Ball Tracker", frame)
                
                k = cv2.waitKey(1) & 0xFF
                if k == ord('q'):
                    break
                elif k == ord('r'):
                    self.reset()
        finally:
            self.stop()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    # Create window FIRST for responsiveness
    cv2.namedWindow("Ball Tracker", cv2.WINDOW_AUTOSIZE)
    cv2.waitKey(1)
    
    print("Initializing (this may take a few seconds)...")
    tracker = DualCameraTracker()
    tracker.run()
