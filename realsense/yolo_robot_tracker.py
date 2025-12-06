"""
Dual-Camera Robot Pose Estimator (Optimized)
=============================================
Camera 0 (origin): 341522302002
Camera 1 (secondary): 213522253879

Markers: 4x4=LEFT, 5x5=FRONT, 6x6=RIGHT, 7x7=BACK (ID=0, 95mm)

Run: python dual_robot_pose_estimator.py
Keys: q=quit, r=reset
"""

import numpy as np
import pyrealsense2 as rs
import cv2
from cv2 import aruco
from collections import deque
import time
from typing import Optional, List, Tuple
from dataclasses import dataclass

from stereo_calib_loader import load_calibration, CAMERA0_SERIAL, CAMERA1_SERIAL

# ===================== CONFIGURATION =====================
RESOLUTION = (640, 480)
FPS = 30

MARKER_SIZE = 0.095  # 95mm
MARKER_ID = 0

MARKER_CONFIGS = {
    'LEFT':  aruco.DICT_4X4_50,
    'FRONT': aruco.DICT_5X5_50,
    'RIGHT': aruco.DICT_6X6_50,
    'BACK':  aruco.DICT_7X7_50,
}

SIDE_YAW_OFFSET = {'LEFT': 90, 'FRONT': 0, 'RIGHT': -90, 'BACK': 180}
MARKER_TO_CENTER = {'LEFT': 0.15, 'FRONT': 0.15, 'RIGHT': 0.15, 'BACK': 0.15}
SIDE_COLORS = {'LEFT': (255,0,0), 'FRONT': (0,255,0), 'RIGHT': (0,0,255), 'BACK': (0,255,255)}

CALIB_DIR = "calib_output"
# =========================================================


@dataclass
class MarkerDet:
    side: str
    tvec: np.ndarray
    rvec: np.ndarray
    yaw: float
    dist: float
    corners: np.ndarray
    cam_id: int


@dataclass
class RobotPose:
    x: float
    y: float
    z: float
    yaw: float
    conf: float
    sides: List[str]
    n_det: int


class PoseKF:
    """Kalman filter for x, z, yaw."""
    
    def __init__(self, pn=0.005, mn=0.02):
        self.state = np.zeros(6)  # x, z, yaw, vx, vz, vyaw
        self.P = np.eye(6) * 0.1
        self.Q = np.eye(6) * pn
        self.R = np.eye(3) * mn
        self.R[2, 2] *= 2
        self.H = np.zeros((3, 6))
        self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = 1
        self.init = False
        self.t_last = None
    
    def update(self, x, z, yaw, t):
        m = np.array([x, z, yaw])
        
        if not self.init:
            self.state[:3] = m
            self.init = True
            self.t_last = t
            return
        
        dt = t - self.t_last
        if dt > 0:
            F = np.eye(6)
            F[0, 3] = F[1, 4] = F[2, 5] = dt
            self.state = F @ self.state
            self.P = F @ self.P @ F.T + self.Q * dt
        self.t_last = t
        
        # Yaw wraparound
        yd = m[2] - self.state[2]
        if yd > 180:
            m[2] -= 360
        elif yd < -180:
            m[2] += 360
        
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ (m - self.H @ self.state)
        
        while self.state[2] > 180:
            self.state[2] -= 360
        while self.state[2] < -180:
            self.state[2] += 360
        
        self.P = (np.eye(6) - K @ self.H) @ self.P
    
    def get(self):
        return self.state[0], self.state[1], self.state[2]
    
    def reset(self):
        self.state = np.zeros(6)
        self.P = np.eye(6) * 0.1
        self.init = False
        self.t_last = None


class ArUcoDetector:
    def __init__(self):
        self.detectors = {}
        params = aruco.DetectorParameters()
        params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        
        for side, dict_type in MARKER_CONFIGS.items():
            self.detectors[side] = aruco.ArucoDetector(
                aruco.getPredefinedDictionary(dict_type), params)
    
    def detect(self, img, K, dist, cam_id) -> List[MarkerDet]:
        dets = []
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        for side, detector in self.detectors.items():
            corners, ids, _ = detector.detectMarkers(gray)
            
            if ids is not None and MARKER_ID in ids.flatten():
                idx = list(ids.flatten()).index(MARKER_ID)
                rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                    [corners[idx]], MARKER_SIZE, K, dist)
                
                rvec = rvecs[0][0]
                tvec = tvecs[0][0]
                
                R, _ = cv2.Rodrigues(rvec)
                mz = R[:, 2]
                yaw = np.degrees(np.arctan2(mz[0], mz[2])) + SIDE_YAW_OFFSET[side]
                while yaw > 180:
                    yaw -= 360
                while yaw < -180:
                    yaw += 360
                
                dets.append(MarkerDet(side, tvec, rvec, yaw, np.linalg.norm(tvec), 
                                      corners[idx][0], cam_id))
        return dets


class DualCamSystem:
    def __init__(self, calib):
        self.calib = calib
        ctx = rs.context()
        devs = {d.get_info(rs.camera_info.serial_number): d for d in ctx.query_devices()}
        
        if CAMERA0_SERIAL not in devs or CAMERA1_SERIAL not in devs:
            raise RuntimeError(f"Cameras not found. Need {CAMERA0_SERIAL} and {CAMERA1_SERIAL}")
        
        self.pipes = []
        self.Ks = [calib.K0, calib.K1]
        self.dists = [calib.dist0, calib.dist1]
        
        for i, ser in enumerate([CAMERA0_SERIAL, CAMERA1_SERIAL]):
            p = rs.pipeline()
            cfg = rs.config()
            cfg.enable_device(ser)
            cfg.enable_stream(rs.stream.color, RESOLUTION[0], RESOLUTION[1], rs.format.bgr8, FPS)
            p.start(cfg)
            self.pipes.append(p)
            print(f"Cam{i} ({ser}) started")
    
    def get_frame(self, cam):
        try:
            fr = self.pipes[cam].wait_for_frames(timeout_ms=30)
            c = fr.get_color_frame()
            if c:
                return np.asanyarray(c.get_data())
        except:
            pass
        return None
    
    def stop(self):
        for p in self.pipes:
            p.stop()


class PoseEstimator:
    def __init__(self, calib):
        self.calib = calib
        self.kf = PoseKF()
        self.last_pose = None
    
    def estimate(self, det0: List[MarkerDet], det1: List[MarkerDet]) -> Optional[RobotPose]:
        # Transform cam1 detections to cam0 frame
        all_dets = list(det0)
        
        for d in det1:
            tvec_c0 = self.calib.cam1_to_cam0(d.tvec)
            R_m, _ = cv2.Rodrigues(d.rvec)
            R_m_c0 = self.calib.rotation_cam1_to_cam0(R_m)
            mz = R_m_c0[:, 2]
            yaw = np.degrees(np.arctan2(mz[0], mz[2])) + SIDE_YAW_OFFSET[d.side]
            while yaw > 180:
                yaw -= 360
            while yaw < -180:
                yaw += 360
            
            rvec_c0, _ = cv2.Rodrigues(R_m_c0)
            all_dets.append(MarkerDet(d.side, tvec_c0, rvec_c0.flatten(), yaw,
                                      np.linalg.norm(tvec_c0), d.corners, 0))
        
        if not all_dets:
            return self.last_pose
        
        positions, yaws, weights, sides = [], [], [], []
        
        for d in all_dets:
            R, _ = cv2.Rodrigues(d.rvec)
            mz = R[:, 2]
            offset = MARKER_TO_CENTER.get(d.side, 0.15)
            robot_pos = d.tvec - mz * offset
            
            positions.append(robot_pos)
            yaws.append(d.yaw)
            weights.append(1.0 / (d.dist + 0.1))
            if d.side not in sides:
                sides.append(d.side)
        
        weights = np.array(weights)
        weights /= weights.sum()
        positions = np.array(positions)
        avg_pos = np.average(positions, axis=0, weights=weights)
        
        # Circular mean for yaw
        yaws_rad = np.radians(yaws)
        avg_yaw = np.degrees(np.arctan2(
            np.sum(weights * np.sin(yaws_rad)),
            np.sum(weights * np.cos(yaws_rad))))
        
        t = time.time()
        self.kf.update(avg_pos[0], avg_pos[2], avg_yaw, t)
        x_f, z_f, yaw_f = self.kf.get()
        
        pose = RobotPose(x_f, avg_pos[1], z_f, yaw_f, 
                         min(1.0, len(all_dets)/2.0), sides, len(all_dets))
        self.last_pose = pose
        return pose
    
    def reset(self):
        self.kf.reset()
        self.last_pose = None


def draw(img, dets, pose, K, dist, baseline, fps):
    h, w = img.shape[:2]
    
    for d in dets:
        clr = SIDE_COLORS[d.side]
        cv2.polylines(img, [d.corners.astype(int)], True, clr, 2)
        cv2.drawFrameAxes(img, K, dist, d.rvec, d.tvec, MARKER_SIZE * 0.5)
        ctr = d.corners.mean(axis=0).astype(int)
        cv2.putText(img, d.side, (ctr[0]-15, ctr[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, clr, 1)
    
    if pose:
        cv2.rectangle(img, (5, 5), (200, 110), (30, 30, 30), -1)
        cv2.putText(img, "ROBOT POSE", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(img, f"X:{pose.x:+.3f}m", (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        cv2.putText(img, f"Z:{pose.z:+.3f}m", (100, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        cv2.putText(img, f"YAW:{pose.yaw:+.1f}deg", (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,255), 2)
        cv2.putText(img, f"{','.join(pose.sides)} ({pose.n_det})", (10, 82),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
        
        # Mini top-down view
        ox, oy = w - 80, h - 80
        cv2.rectangle(img, (ox-60, oy-60), (ox+60, oy+60), (40, 40, 40), -1)
        cv2.circle(img, (ox, oy), 3, (255, 255, 0), -1)
        
        rx = int(ox + pose.x * 30)
        ry = int(oy - pose.z * 30)
        cv2.circle(img, (rx, ry), 8, (0, 255, 0), 2)
        yr = np.radians(pose.yaw)
        ax, ay = int(rx + 12*np.sin(yr)), int(ry - 12*np.cos(yr))
        cv2.arrowedLine(img, (rx, ry), (ax, ay), (0, 255, 0), 2)
    else:
        cv2.putText(img, "No robot", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    cv2.putText(img, f"FPS:{fps:.0f}", (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    return img


def main():
    calib = load_calibration(CALIB_DIR)
    cams = DualCamSystem(calib)
    detector = ArUcoDetector()
    estimator = PoseEstimator(calib)
    
    fps_q = deque(maxlen=20)
    
    print(f"\nDUAL-CAM ROBOT POSE | Baseline:{calib.baseline*100:.1f}cm")
    print("q=quit r=reset\n")
    
    try:
        while True:
            t0 = time.time()
            
            f0 = cams.get_frame(0)
            f1 = cams.get_frame(1)
            
            if f0 is None:
                continue
            
            det0 = detector.detect(f0, cams.Ks[0], cams.dists[0], 0)
            det1 = detector.detect(f1, cams.Ks[1], cams.dists[1], 1) if f1 is not None else []
            
            pose = estimator.estimate(det0, det1)
            
            fps_q.append(time.time() - t0)
            fps = 1.0 / (sum(fps_q) / len(fps_q))
            
            frame = draw(f0.copy(), det0, pose, cams.Ks[0], cams.dists[0], calib.baseline, fps)
            cv2.putText(frame, f"C1:{len(det1)}m", (frame.shape[1]-60, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0) if det1 else (100,100,100), 1)
            
            cv2.imshow("Robot Pose", frame)
            
            if pose:
                print(f"\rX:{pose.x:+.3f} Z:{pose.z:+.3f} YAW:{pose.yaw:+6.1f}° [{','.join(pose.sides)}]  ", end="")
            
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            elif k == ord('r'):
                estimator.reset()
                print("\n[RESET]")
    finally:
        cams.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Create window FIRST for responsiveness
    cv2.namedWindow("Robot Pose", cv2.WINDOW_AUTOSIZE)
    cv2.waitKey(1)
    
    print("Initializing (this may take a few seconds)...")
    main()
