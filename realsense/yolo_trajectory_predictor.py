"""
Dual-Camera Trajectory Predictor (Optimized)
=============================================
Camera 0 (origin): 341522302002
Camera 1 (secondary): 213522253879

Optimizations:
- GPU inference for YOLO (CUDA)
- Threaded Camera 1 frame capture
- Alternate-frame Camera 1 detection
- Reduced frame copies

Run: python yolo_trajectory_predictor.py
Keys: q=quit, r=reset
"""

import numpy as np
import pyrealsense2 as rs
import cv2
from ultralytics import YOLO
from collections import deque
import time
import threading
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

GRAVITY = 9.81
GROUND_Y = 0.96

MIN_POINTS = 3
MIN_CONF_LOCK = 0.5

CALIB_DIR = "../calib_output"
MAX_FUSION_DIST = 0.3

# Optimization settings
USE_GPU = True  # Set to False to force CPU
SKIP_CAM1_FRAMES = 2  # Detect on Cam1 every N frames (1 = every frame)
# =========================================================


@dataclass
class Detection:
    center: Tuple[float, float]
    radius: float
    confidence: float
    method: str
    camera_id: int


@dataclass
class Landing:
    x: float
    y: float
    z: float
    time: float
    confidence: float
    
    @property
    def pos(self):
        return np.array([self.x, self.y, self.z])


class KalmanFilter:
    """Physics-based Kalman filter for ball trajectory."""
    
    def __init__(self, g=GRAVITY, q=0.01, r=0.02):
        self.g = g
        self.state = np.zeros(6)  # [x,y,z,vx,vy,vz]
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
            # Predict
            F = np.eye(6)
            F[0,3] = F[1,4] = F[2,5] = dt
            B = np.zeros(6)
            B[1] = 0.5 * self.g * dt**2
            B[4] = self.g * dt
            self.state = F @ self.state + B
            self.P = F @ self.P @ F.T + np.eye(6) * self.Q_base * dt
        
        self.t_last = t
        
        # Update
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


class TrajectoryPredictor:
    def __init__(self, g=GRAVITY, ground=GROUND_Y):
        self.g = g
        self.ground = ground
        self.kf = KalmanFilter(g=g)
        self.vel_hist = deque(maxlen=10)
        
        self.locked = False
        self.lock_pos = None
        self.lock_vel = None
        self.lock_land = None
        self.lock_time = 0
        self.current = None
    
    def update(self, pos, t):
        if self.locked:
            return self.lock_land
        
        self.kf.update(pos, t)
        self.vel_hist.append(self.kf.vel())
        
        if self.kf.n >= MIN_POINTS:
            self.current = self._predict()
            if self.current and self.current.confidence >= MIN_CONF_LOCK:
                self.locked = True
                self.lock_pos = self.kf.pos()
                self.lock_vel = self.kf.vel()
                self.lock_land = self.current
                self.lock_time = self.current.time
                print(f"\n[LOCKED] X:{self.lock_land.x:+.3f} Z:{self.lock_land.z:+.3f} T:{self.lock_time:.2f}s")
        
        return self.current
    
    def _predict(self):
        p, v = self.kf.pos(), self.kf.vel()
        a = 0.5 * self.g
        b = v[1]
        c = p[1] - self.ground
        
        disc = b*b - 4*a*c
        if disc < 0:
            return None
        
        t1 = (-b + np.sqrt(disc)) / (2*a)
        t2 = (-b - np.sqrt(disc)) / (2*a)
        ts = [t for t in [t1, t2] if t > 0]
        if not ts:
            return None
        
        t_land = min(ts)
        x_land = p[0] + v[0] * t_land
        z_land = p[2] + v[2] * t_land
        
        # Confidence
        n_factor = min(1.0, self.kf.n / 8)
        if len(self.vel_hist) >= 3:
            vels = np.array(list(self.vel_hist))
            stability = 1.0 - np.mean(np.std(vels, axis=0) / (np.abs(np.mean(vels, axis=0)) + 1e-6)).clip(0, 1)
        else:
            stability = 0.5
        conf = 0.5 * n_factor + 0.5 * stability
        
        return Landing(x_land, self.ground, z_land, t_land, float(np.clip(conf, 0, 1)))
    
    def get_traj_pts(self, n=40):
        if self.locked:
            if self.lock_pos is None:
                return None
            p, v, t = self.lock_pos, self.lock_vel, self.lock_time
        else:
            if self.current is None:
                return None
            p, v, t = self.kf.pos(), self.kf.vel(), self.current.time
        
        ts = np.linspace(0, t, n)
        pts = []
        for ti in ts:
            pts.append([p[0] + v[0]*ti, p[1] + v[1]*ti + 0.5*self.g*ti**2, p[2] + v[2]*ti])
        return np.array(pts)
    
    def get_land(self):
        return self.lock_land if self.locked else self.current
    
    def get_init_pos(self):
        return self.lock_pos if self.locked else None
    
    def reset(self):
        self.kf.reset()
        self.vel_hist.clear()
        self.locked = False
        self.lock_pos = self.lock_vel = self.lock_land = None
        self.lock_time = 0
        self.current = None


class BallDetector:
    def __init__(self):
        print("Loading YOLO...")
        self.model = YOLO(YOLO_MODEL)
        
        # Enable GPU if available
        if USE_GPU:
            try:
                self.model.to("cuda")
                print("YOLO using CUDA GPU")
            except Exception as e:
                print(f"GPU unavailable, using CPU: {e}")
        
        # Warmup inference
        self.model.predict(np.zeros((320, 320, 3), dtype=np.uint8), verbose=False)
        print("Ready!")
    
    def detect(self, img, cam_id):
        results = self.model.predict(img, conf=CONFIDENCE, classes=[32], verbose=False, imgsz=320)
        for r in results:
            if len(r.boxes) > 0:
                b = r.boxes[0]
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
                cx, cy, rad = (x1+x2)/2, (y1+y2)/2, max(x2-x1, y2-y1)/2
                if 5 < rad < 150:
                    return Detection((cx, cy), rad, float(b.conf[0]), 'YOLO', cam_id)
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in sorted(contours, key=cv2.contourArea, reverse=True)[:3]:
            area = cv2.contourArea(c)
            if 200 < area < 50000:
                perim = cv2.arcLength(c, True)
                if perim > 0 and 4*np.pi*area/perim**2 > 0.6:
                    (cx, cy), rad = cv2.minEnclosingCircle(c)
                    if 5 < rad < 150:
                        return Detection((cx, cy), rad, 0.8, 'HSV', cam_id)
        return None


class ThreadedCamera:
    """Background thread for Camera 1 frame capture."""
    
    def __init__(self, pipeline, spatial_filter, depth_scale):
        self.pipeline = pipeline
        self.spatial = spatial_filter
        self.depth_scale = depth_scale
        self.color = None
        self.depth = None
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
    
    def _capture_loop(self):
        while self.running:
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=50)
                color = frames.get_color_frame()
                depth = self.spatial.process(frames.get_depth_frame())
                if color and depth:
                    with self.lock:
                        self.color = np.asanyarray(color.get_data())
                        self.depth = np.asanyarray(depth.get_data())
            except:
                pass
    
    def get_frames(self):
        with self.lock:
            return self.color, self.depth
    
    def stop(self):
        self.running = False
        self.thread.join(timeout=1.0)


class DualCamSystem:
    def __init__(self, calib):
        self.calib = calib
        ctx = rs.context()
        devs = {d.get_info(rs.camera_info.serial_number): d for d in ctx.query_devices()}
        
        if CAMERA0_SERIAL not in devs or CAMERA1_SERIAL not in devs:
            raise RuntimeError("Cameras not found")
        
        self.intrin = []
        self.scales = []
        self.spatial = rs.spatial_filter()
        
        # Camera 0 - main pipeline (blocking)
        self.pipe0 = rs.pipeline()
        cfg0 = rs.config()
        cfg0.enable_device(CAMERA0_SERIAL)
        cfg0.enable_stream(rs.stream.color, RESOLUTION[0], RESOLUTION[1], rs.format.bgr8, FPS)
        cfg0.enable_stream(rs.stream.depth, RESOLUTION[0], RESOLUTION[1], rs.format.z16, FPS)
        prof0 = self.pipe0.start(cfg0)
        self.scales.append(prof0.get_device().first_depth_sensor().get_depth_scale())
        self.intrin.append(prof0.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics())
        print(f"Cam0 ({CAMERA0_SERIAL}) started")
        
        # Camera 1 - threaded pipeline (non-blocking)
        pipe1 = rs.pipeline()
        cfg1 = rs.config()
        cfg1.enable_device(CAMERA1_SERIAL)
        cfg1.enable_stream(rs.stream.color, RESOLUTION[0], RESOLUTION[1], rs.format.bgr8, FPS)
        cfg1.enable_stream(rs.stream.depth, RESOLUTION[0], RESOLUTION[1], rs.format.z16, FPS)
        prof1 = pipe1.start(cfg1)
        self.scales.append(prof1.get_device().first_depth_sensor().get_depth_scale())
        self.intrin.append(prof1.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics())
        print(f"Cam1 ({CAMERA1_SERIAL}) started (threaded)")
        
        self.cam1_thread = ThreadedCamera(pipe1, self.spatial, self.scales[1])
    
    def get_frames_cam0(self):
        try:
            fr = self.pipe0.wait_for_frames(timeout_ms=50)
            c, d = fr.get_color_frame(), self.spatial.process(fr.get_depth_frame())
            if c and d:
                return np.asanyarray(c.get_data()), np.asanyarray(d.get_data())
        except:
            pass
        return None, None
    
    def get_frames_cam1(self):
        return self.cam1_thread.get_frames()
    
    def pix_to_3d(self, u, v, dep, cam):
        h, w = dep.shape
        v_int, u_int = int(v), int(u)
        y1, y2 = max(0, v_int - 5), min(h, v_int + 5)
        x1, x2 = max(0, u_int - 5), min(w, u_int + 5)
        roi = dep[y1:y2, x1:x2]
        roi_m = roi.astype(np.float32) * self.scales[cam]
        valid = roi_m[(roi_m > MIN_DEPTH) & (roi_m < MAX_DEPTH)]
        if len(valid) == 0:
            return None
        return np.array(rs.rs2_deproject_pixel_to_point(self.intrin[cam], [u, v], np.median(valid)))
    
    def stop(self):
        self.cam1_thread.stop()
        self.pipe0.stop()


class Visualizer:
    def __init__(self, intrin):
        self.intrin = intrin
    
    def proj(self, pt):
        if pt[2] <= 0:
            return None
        px = rs.rs2_project_point_to_pixel(self.intrin, pt.tolist())
        u, v = int(px[0]), int(px[1])
        if 0 <= u < RESOLUTION[0] and 0 <= v < RESOLUTION[1]:
            return (u, v)
        return None
    
    def draw(self, frame, det0, det1, pred, fused, fps):
        # Draw directly on frame (no copy needed)
        h, w = frame.shape[:2]
        
        if det0:
            cv2.circle(frame, (int(det0.center[0]), int(det0.center[1])), int(det0.radius),
                      (0,255,0) if det0.method=='YOLO' else (255,255,0), 2)
        
        cv2.putText(frame, f"C1:{'Y' if det1 else 'N'}", (w-50,20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0) if det1 else (100,100,100), 1)
        
        if pred.kf.init:
            p, v = pred.kf.pos(), pred.kf.vel()
            lbl = "F" if det0 and det1 else "0" if det0 else "1"
            cv2.putText(frame, f"[{lbl}] ({p[0]:.2f},{p[1]:.2f},{p[2]:.2f})m v={np.linalg.norm(v):.1f}m/s",
                       (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,255), 2)
        
        land = pred.get_land()
        if land:
            traj = pred.get_traj_pts(40)
            if traj is not None:
                pts2d = [self.proj(pt) for pt in traj]
                pts2d = [p for p in pts2d if p]
                for i in range(1, len(pts2d)):
                    prog = i / len(pts2d)
                    cv2.line(frame, pts2d[i-1], pts2d[i], (0, int(255*(1-prog)), int(255*prog)), 2)
            
            lp = self.proj(land.pos)
            if lp:
                cv2.circle(frame, lp, 15, (0,0,255), 2)
                cv2.drawMarker(frame, lp, (0,0,255), cv2.MARKER_CROSS, 20, 2)
            
            ip = pred.get_init_pos()
            if ip is not None:
                ip2d = self.proj(ip)
                if ip2d:
                    cv2.circle(frame, ip2d, 8, (0,255,0), 2)
            
            # Info box
            bx = w - 200
            cv2.rectangle(frame, (bx, 40), (w-5, 120), (30,30,30), -1)
            cv2.rectangle(frame, (bx, 40), (w-5, 120), (0,0,255), 1)
            lbl = "LOCKED" if pred.locked else "PRED"
            cv2.putText(frame, lbl, (bx+5, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 
                       (0,255,0) if pred.locked else (0,200,255), 1)
            cv2.putText(frame, f"X:{land.x:+.2f} Z:{land.z:+.2f}", (bx+5, 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
            cv2.putText(frame, f"T:{land.time:.2f}s C:{land.confidence:.0%}", (bx+5, 95),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)
        else:
            if pred.kf.init:
                cv2.putText(frame, f"Need {MIN_POINTS - pred.kf.n} more pts", (10, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100,100,255), 1)
        
        cv2.putText(frame, f"FPS:{fps:.0f}", (10,h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        return frame


def fuse(calib, cams, det0, det1, d0, d1):
    p0 = cams.pix_to_3d(det0.center[0], det0.center[1], d0, 0) if det0 and d0 is not None else None
    p1 = None
    if det1 and d1 is not None:
        p1_loc = cams.pix_to_3d(det1.center[0], det1.center[1], d1, 1)
        if p1_loc is not None:
            p1 = calib.cam1_to_cam0(p1_loc)
    
    if p0 is None and p1 is None:
        return None
    if p0 is None:
        return p1
    if p1 is None:
        return p0
    if np.linalg.norm(p0 - p1) > MAX_FUSION_DIST:
        return p0 if det0.confidence >= det1.confidence else p1
    w0, w1 = det0.confidence / (p0[2]+0.1), det1.confidence / (p1[2]+0.1)
    return (w0*p0 + w1*p1) / (w0+w1)


def main():
    calib = load_calibration(CALIB_DIR)
    cams = DualCamSystem(calib)
    det = BallDetector()
    pred = TrajectoryPredictor(GRAVITY, GROUND_Y)
    vis = Visualizer(cams.intrin[0])
    
    cv2.namedWindow("Trajectory", cv2.WINDOW_AUTOSIZE)
    fps_q = deque(maxlen=30)
    
    # Optimization state
    frame_count = 0
    last_det1 = None
    last_d1 = None
    
    print(f"\nDUAL-CAM TRAJECTORY | Ground:{GROUND_Y}m | Baseline:{calib.baseline*100:.1f}cm")
    print(f"GPU: {'CUDA' if USE_GPU else 'CPU'} | Cam1 skip: {SKIP_CAM1_FRAMES}")
    print("q=quit r=reset\n")
    
    try:
        while True:
            t0 = time.time()
            frame_count += 1
            
            # Camera 0: blocking wait
            c0, d0 = cams.get_frames_cam0()
            if c0 is None:
                continue
            
            # Camera 1: non-blocking (threaded)
            c1, d1 = cams.get_frames_cam1()
            
            # Always detect on Camera 0
            det0 = det.detect(c0, 0)
            
            # Detect on Camera 1 every N frames (use cached otherwise)
            if c1 is not None and frame_count % SKIP_CAM1_FRAMES == 0:
                det1 = det.detect(c1, 1)
                last_det1 = det1
                last_d1 = d1
            else:
                det1 = last_det1
                d1 = last_d1
            
            fused_pos = fuse(calib, cams, det0, det1, d0, d1)
            
            if fused_pos is not None and not pred.locked:
                pred.update(fused_pos, time.time())
            
            fps_q.append(time.time() - t0)
            fps = 1.0 / (sum(fps_q) / len(fps_q))
            
            # Draw directly on c0 (no copy)
            frame = vis.draw(c0, det0, det1, pred, fused_pos, fps)
            cv2.imshow("Trajectory", frame)
            
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            elif k == ord('r'):
                pred.reset()
                last_det1 = None
                print("[RESET]")
    finally:
        cams.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Create window FIRST for responsiveness
    cv2.namedWindow("Trajectory", cv2.WINDOW_AUTOSIZE)
    cv2.waitKey(1)
    
    print("Initializing (this may take a few seconds)...")
    main()

