#!/usr/bin/env python3
import pyzed.sl as sl
import cv2 as cv
import numpy as np
import time
from collections import deque
import random
import socket
import struct
from typing import Optional, Dict, List, Tuple

# ===========================
# CONSTANTS - BALL / PHYSICS
# ===========================
G = 9.81
GROUND_Y = -0.52
MIN_SAMPLES = 3
RANSAC_ITERS = 150
RANSAC_INLIER_THRESH = 0.03
MAX_HISTORY = 60

# Velocity filters
VEL_WIN = 7                  # frames to average velocity over
THROW_SPEED_THRESH = 0.80    # relaxed from 1.4 m/s
THROW_UPWARD_VY_THRESH = 0.4 # relaxed from 0.7 m/s
THROW_VERIFY_FRAMES = 3      # relaxed from 4 frames

# ===========================
# CAMERA CONFIG
# ===========================
INTR0_PATH = "./camera_parameters/camera0_intrinsics.dat"
INTR1_PATH = "./camera_parameters/camera1_intrinsics.dat"
EXTR1_PATH = "./camera_parameters/camera1_rot_trans.dat"

HSV_LOWER = np.array([29, 86, 6])
HSV_UPPER = np.array([64, 255, 255])

MIN_AREA = 80
MAX_AREA = 50000
MIN_CIRC = 0.35

SMOOTH_ALPHA = 0.4

# ===========================
# UDP CONFIG
# ===========================
DEFAULT_ROBOT_PORT = 5005
DEFAULT_ROBOT_IP = "192.168.0.155"
UDP_RATE_LIMIT_HZ = 30.0

# ===========================
# PERFORMANCE CONFIG
# ===========================
ARUCO_UPDATE_EVERY = 3   # run ArUco only every N frames (reuse last pose between updates)


# ===========================
# UTIL: LOAD INTRINSICS / EXTRINSICS (.dat)
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
# UDP SENDER
# ===========================

class UDPSender:
    def __init__(self, robot_ip: str = DEFAULT_ROBOT_IP, port: int = DEFAULT_ROBOT_PORT,
                 rate_limit: Optional[float] = None):
        self.robot_ip = robot_ip
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.enabled = True
        self.packets_sent = 0

        self.rate_limit = rate_limit
        self.last_send_time = 0.0
        self.min_send_interval = (1.0 / rate_limit) if rate_limit else 0.0

    def _rate_ok(self) -> bool:
        if not self.rate_limit:
            return True
        now = time.time()
        if (now - self.last_send_time) < self.min_send_interval:
            return False
        self.last_send_time = now
        return True

    def send_robot_pose(self, x: float, y: float, z: float, yaw: float) -> bool:
        if not self.enabled:
            return False
        if not self._rate_ok():
            return False
        try:
            packet = struct.pack('4f', x, y, z, yaw)
            self.sock.sendto(packet, (self.robot_ip, self.port))
            self.packets_sent += 1
            return True
        except Exception as e:
            print(f"[UDP ERROR] Failed to send robot pose: {e}")
            return False

    def send_predicted_landing(self, x: float, y: float, z: float) -> bool:
        if not self.enabled:
            return False
        # no rate limit for predicted landing
        try:
            packet = struct.pack('3f', x, y, z)
            self.sock.sendto(packet, (self.robot_ip, self.port))
            self.packets_sent += 1
            return True
        except Exception as e:
            print(f"[UDP ERROR] Failed to send predicted landing: {e}")
            return False

    def toggle(self) -> bool:
        self.enabled = not self.enabled
        status = "ENABLED" if self.enabled else "DISABLED"
        print(f"[UDP] Sending {status}")
        return self.enabled

    def close(self):
        self.sock.close()


# ===========================
# BALL DETECTOR
# ===========================

class HSVBallDetector:
    def __init__(self):
        self.kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))

    def detect(self, frame):
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, HSV_LOWER, HSV_UPPER)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, self.kernel)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, self.kernel)

        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        best, best_area = None, 0
        for c in contours:
            area = cv.contourArea(c)
            if not (MIN_AREA < area < MAX_AREA):
                continue

            peri = cv.arcLength(c, True)
            if peri == 0:
                continue

            circ = 4 * np.pi * area / (peri * peri)
            if circ < MIN_CIRC:
                continue

            (cx, cy), radius = cv.minEnclosingCircle(c)
            if radius < 4:
                continue

            if area > best_area:
                best_area = area
                best = ((cx, cy), radius)

        return best


# ===========================
# STEREO MODEL FOR BALL TRIANGULATION
# ===========================

class StereoModel:
    def __init__(self, i0, i1, e1):
        self.K0, _ = load_intrinsics_dat(i0)
        self.K1, _ = load_intrinsics_dat(i1)
        self.R1, self.T1 = load_extrinsics_dat(e1)

        # Heuristic: convert T1 to meters (cm or mm)
        if np.linalg.norm(self.T1) > 1.0:
            self.T1 /= 100.0  # assume cm -> m

        self.B = abs(self.T1[0, 0])   # baseline in meters
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
# BALL TRAJECTORY ESTIMATOR
# ===========================

class BallTrajectoryEstimator:
    def __init__(self):
        self.reset()

    def reset(self):
        self.t0 = None
        self.times = []
        self.pos = []

    def add(self, t_abs, p):
        if self.t0 is None:
            self.t0 = t_abs
        t = t_abs - self.t0
        self.times.append(t)
        self.pos.append(p.copy())

        if len(self.times) > MAX_HISTORY:
            self.times.pop(0)
            self.pos.pop(0)

    def _fit_ls(self, idx):
        if len(idx) < 2:
            return None

        t = np.array([self.times[i] for i in idx])
        xyz = np.array([self.pos[i] for i in idx])

        A = np.column_stack([np.ones_like(t), t])

        try:
            X0, Vx = np.linalg.lstsq(A, xyz[:, 0], rcond=None)[0]
            Y_lin = xyz[:, 1] + 0.5 * G * (t ** 2)
            Y0, Vy = np.linalg.lstsq(A, Y_lin, rcond=None)[0]
            Z0, Vz = np.linalg.lstsq(A, xyz[:, 2], rcond=None)[0]
            return X0, Y0, Z0, Vx, Vy, Vz
        except:
            return None

    def _res(self, P):
        X0, Y0, Z0, Vx, Vy, Vz = P
        t = np.array(self.times)
        xyz = np.array(self.pos)
        Xp = X0 + Vx * t
        Yp = Y0 + Vy * t - 0.5 * G * t * t
        Zp = Z0 + Vz * t
        return np.linalg.norm(np.column_stack([Xp, Yp, Zp]) - xyz, axis=1)

    def estimate(self):
        n = len(self.times)
        if n < MIN_SAMPLES:
            # debug
            print(f"[ESTIMATE] Not enough samples: n={n} < MIN_SAMPLES={MIN_SAMPLES}")
            return None, None

        bestP, bestIn, bestCount = None, None, -1
        idxAll = list(range(n))

        for _ in range(RANSAC_ITERS):
            subset = random.sample(idxAll, 3)
            P = self._fit_ls(subset)
            if P is None:
                continue

            r = self._res(P)
            inl = r < RANSAC_INLIER_THRESH
            c = np.sum(inl)
            if c > bestCount:
                bestCount = c
                bestP = P
                bestIn = inl

        if bestP is None or bestCount < MIN_SAMPLES:
            print(f"[ESTIMATE] RANSAC failed or too few inliers: "
                  f"bestCount={bestCount}, MIN_SAMPLES={MIN_SAMPLES}, n={n}")
            return None, None

        finalIdx = [i for i, v in enumerate(bestIn) if v]
        P2 = self._fit_ls(finalIdx)
        if P2 is None:
            return bestP, bestIn
        return P2, bestIn

    def _solve_t_land(self, P):
        X0, Y0, Z0, Vx, Vy, Vz = P
        a = -0.5 * G
        b = Vy
        c = Y0 - GROUND_Y
        D = b * b - 4 * a * c
        if D < 0:
            print(f"[LAND] Negative discriminant: D={D:.4f}, Y0={Y0:.3f}, Vy={Vy:.3f}")
            return None
        r1 = (-b + np.sqrt(D)) / (2 * a)
        r2 = (-b - np.sqrt(D)) / (2 * a)
        cand = [t for t in (r1, r2) if t > 0]
        if not cand:
            print(f"[LAND] No positive root: r1={r1:.3f}, r2={r2:.3f}")
            return None
        return min(cand)

    def landing_point(self):
        """
        Returns:
            landing_pos: np.array([X, Y, Z]) at ground
            P: trajectory params
            tL: time-to-land (s, relative to first sample)
        """
        n = len(self.times)
        P, inl = self.estimate()
        if P is None:
            # already logged in estimate()
            return None, None, None

        tL = self._solve_t_land(P)
        if tL is None:
            # already logged in _solve_t_land
            return None, P, None

        X0, Y0, Z0, Vx, Vy, Vz = P
        land = np.array([X0 + Vx * tL, GROUND_Y, Z0 + Vz * tL])
        print(f"[LAND] Success: n={n}, land={land}, tL={tL:.3f}")
        return land, P, tL


# ===========================
# ARUCO ROBOT POSE (SINGLE CAMERA, HEADING YAW)
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
        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = 23
        params.adaptiveThreshWinSizeStep = 10
        params.cornerRefinementMethod = cv.aruco.CORNER_REFINE_SUBPIX
        detectors[side_name] = cv.aruco.ArucoDetector(dictionary, params)
    return detectors


def detect_aruco_corners(
    frame: np.ndarray,
    detectors: Dict[str, cv.aruco.ArucoDetector]
) -> Dict[str, np.ndarray]:
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    detected = {}
    for side_name, detector in detectors.items():
        corners, ids, _ = detector.detectMarkers(gray)
        if ids is not None:
            for i, marker_id in enumerate(ids.flatten()):
                if marker_id == 0:
                    detected[side_name] = corners[i].reshape(4, 2)
                    break
    return detected


def get_marker_model_points(marker_size: float) -> np.ndarray:
    half = marker_size / 2.0
    return np.array([
        [-half,  half, 0.0],  # top-left
        [ half,  half, 0.0],  # top-right
        [ half, -half, 0.0],  # bottom-right
        [-half, -half, 0.0],  # bottom-left
    ], dtype=np.float64)


def heading_yaw_from_marker(side_name: str, R_cam: np.ndarray) -> float:
    # marker normal in camera frame (OpenCV: +X right, +Y down, +Z forward)
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
        H_cam = np.array([-n_cam[2], n_cam[1], n_cam[0]])   # R_y(-90°)*n
    elif side_name == 'left':
        H_cam = np.array([n_cam[2], n_cam[1], -n_cam[0]])   # R_y(+90°)*n
    else:
        H_cam = -n_cam

    H_xz = np.array([H_cam[0], H_cam[2]], dtype=np.float64)
    h_norm = np.linalg.norm(H_xz)
    if h_norm < 1e-8:
        return 0.0
    H_xz /= h_norm

    yaw_rad = np.arctan2(H_xz[0], H_xz[1])  # atan2(X, Z)
    return float(np.degrees(yaw_rad))


def pose_from_aruco_pnp(
    side_name: str,
    corners_px: np.ndarray,
    camera: ArucoCamera,
    marker_size_m: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    corners_input = corners_px.reshape(1, 1, 4, 2).astype(np.float32)

    rvecs, tvecs, _ = cv.aruco.estimatePoseSingleMarkers(
        corners_input,
        marker_size_m,
        camera.intrinsic,
        camera.distortion
    )

    rvec = rvecs[0, 0, :]
    tvec = tvecs[0, 0, :]

    R_cam, _ = cv.Rodrigues(rvec)
    t_cam = tvec.reshape(3, 1)

    yaw_deg = heading_yaw_from_marker(side_name, R_cam)

    # camera OpenCV (+Y down) -> USER (+Y up)
    S = np.diag([1.0, -1.0, 1.0])
    R_user = S @ R_cam
    t_user = S @ t_cam

    obj_pts = get_marker_model_points(marker_size_m)
    corners_3d_user = []
    for p in obj_pts:
        p_cam = R_user @ p.reshape(3, 1) + t_user
        corners_3d_user.append(p_cam.ravel())
    corners_3d_user = np.array(corners_3d_user)

    return R_user, t_user.ravel(), corners_3d_user, yaw_deg


class DetectedMarker:
    def __init__(self, side_name, corners_left, corners_right,
                 corners_3d, position, yaw_angle, rotation_matrix):
        self.side_name = side_name
        self.corners_left = corners_left
        self.corners_right = corners_right
        self.corners_3d = corners_3d
        self.position = position
        self.yaw_angle = yaw_angle
        self.rotation_matrix = rotation_matrix


def process_aruco_frame(
    left_frame: np.ndarray,
    right_frame: np.ndarray,
    detectors: Dict[str, cv.aruco.ArucoDetector],
    camera: ArucoCamera,
    marker_size_m: float
) -> List[DetectedMarker]:
    corners_left = detect_aruco_corners(left_frame, detectors)
    corners_right = detect_aruco_corners(right_frame, detectors)

    markers: List[DetectedMarker] = []
    for side_name, pts_left in corners_left.items():
        R_user, t_user, corners_3d_user, yaw_deg = pose_from_aruco_pnp(
            side_name,
            pts_left,
            camera,
            marker_size_m
        )
        position = t_user
        pts_right = corners_right.get(side_name, np.zeros((4, 2)))
        markers.append(DetectedMarker(
            side_name, pts_left, pts_right,
            corners_3d_user, position, yaw_deg, R_user
        ))
    return markers


def compute_robot_pose(
    markers: List[DetectedMarker]
) -> Optional[Tuple[np.ndarray, float, List[DetectedMarker]]]:
    if not markers:
        return None

    positions = np.array([m.position for m in markers])
    avg_position = np.mean(positions, axis=0)

    priority = {'back': 0, 'front': 1, 'left': 2, 'right': 3}
    best = min(markers, key=lambda m: priority.get(m.side_name, 99))
    yaw = best.yaw_angle

    return avg_position, yaw, markers


# ===========================
# MAIN UNIFIED LOOP
# ===========================

def main():
    # Enable OpenCV optimizations (multi-threaded C++)
    cv.setUseOptimized(True)
    try:
        cv.setNumThreads(cv.getNumberOfCPUs())
    except Exception:
        pass

    # -------------
    # Init modules
    # -------------
    detector = HSVBallDetector()
    stereo_model = StereoModel(INTR0_PATH, INTR1_PATH, EXTR1_PATH)
    traj_est = BallTrajectoryEstimator()

    # intrinsics for ArUco camera 0
    K0, D0 = load_intrinsics_dat(INTR0_PATH)
    aruco_cam = ArucoCamera(K0, D0)
    detectors = create_aruco_detectors()
    marker_size_m = 0.095  # 95 mm

    # UDP
    udp = UDPSender(robot_ip=DEFAULT_ROBOT_IP,
                    port=DEFAULT_ROBOT_PORT,
                    rate_limit=UDP_RATE_LIMIT_HZ)

    # -------------
    # Init ZED
    # -------------
    zed = sl.Camera()
    ip = sl.InitParameters()
    ip.camera_resolution = sl.RESOLUTION.HD720
    ip.camera_fps = 30
    ip.depth_mode = sl.DEPTH_MODE.NONE
    ip.coordinate_units = sl.UNIT.METER
    if zed.open(ip) != sl.ERROR_CODE.SUCCESS:
        print("ZED open FAILED.")
        return

    left_mat = sl.Mat()
    right_mat = sl.Mat()

    # -------------
    # Ball tracker state
    # -------------
    last_smooth = None
    state = "IDLE"
    locked = None
    locked_tL = None

    vel_buffer = deque(maxlen=VEL_WIN)
    last_raw = None
    last_time = None

    throw_verify = 0
    prediction_done = False
    landing_udp_sent = False

    frame_count = 0
    last_loop_time = None
    fps = 0.0

    # Robot pose cache for subsampling ArUco
    aruco_frame_counter = 0
    last_robot_pose = None
    last_markers = []

    print("\nControls: 'q'/ESC=Quit, 'r'=Reset ball, 'u'=UDP toggle\n")

    while True:
        # -------- FPS measurement --------
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

        # ====================================================
        # BALL TRACKING + TRAJECTORY
        # ====================================================
        detL = detector.detect(frameL)
        detR = detector.detect(frameR) if detL else None

        speed = 0.0
        vy = 0.0
        throw_cond = False

        if detL and detR:
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

                # ---------------------------
                #   STATE MACHINE
                # ---------------------------
                if state == "IDLE":
                    state = "HOLD"
                    throw_verify = 0

                elif state == "HOLD":
                    traj_est.reset()
                    if not prediction_done:
                        locked = None
                        locked_tL = None
                        landing_udp_sent = False

                        throw_cond = (speed > THROW_SPEED_THRESH and
                                      vy > abs(THROW_UPWARD_VY_THRESH))
                        if throw_cond:
                            throw_verify += 1
                        else:
                            throw_verify = 0

                        if throw_verify >= THROW_VERIFY_FRAMES:
                            print(f"[THROW DETECTED] speed={speed:.2f}, vy={vy:.2f}, fps={fps:.1f}")
                            state = "THROWN"
                            vel_buffer.clear()
                            traj_est.reset()
                            traj_est.add(now, smooth)
                            throw_verify = 0

                elif state == "THROWN":
                    if not prediction_done:
                        traj_est.add(now_loop, smooth)
                        # debug
                        # print(f"[THROWN] samples={len(traj_est.times)}")
                        if locked is None:
                            land, P, tL = traj_est.landing_point()
                            if land is not None and tL is not None:
                                locked = land
                                locked_tL = tL
                                prediction_done = True
                                state = "LOCKED_STATE"
                                print(f"[LOCKED] Landing = {land}, tL={tL:.3f} s")

                                if not landing_udp_sent:
                                    landing_udp_sent = True

                elif state == "LOCKED_STATE":
                    pass

                # draw ball debug
                Xb, Yb, Zb = smooth
                cv.circle(frameL, (int(uL), int(vL)), int(rL), (0, 255, 0), 2)
                cv.putText(frameL, f"BALL X={Xb:.2f} Y={Yb:.2f} Z={Zb:.2f}",
                           (20, 40), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv.putText(frameL, f"speed={speed:.2f} vy={vy:.2f} STATE={state}",
                           (20, 70), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv.putText(frameL,
                           f"throw_cond={throw_cond} verify={throw_verify}/{THROW_VERIFY_FRAMES}",
                           (20, 100), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

                if locked is not None:
                    LX, LY, LZ = locked
                    tL_txt = f"{locked_tL:.2f}" if locked_tL is not None else "?"
                    cv.putText(frameL, f"LAND LOCK X={LX:.2f} Z={LZ:.2f} t={tL_txt}s",
                               (20, 130), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

            else:
                cv.putText(frameL, "BAD TRIANG", (20, 40),
                           cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            if not prediction_done:
                state = "IDLE"
                traj_est.reset()
                vel_buffer.clear()
                last_smooth = None
                last_raw = None
                locked = None
                locked_tL = None
                throw_verify = 0
                landing_udp_sent = False

            cv.putText(frameL, "BALL SEARCHING...", (20, 40),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if locked is not None and prediction_done:
                LX, LY, LZ = locked
                tL_txt = f"{locked_tL:.2f}" if locked_tL is not None else "?"
                cv.putText(frameL, f"LAND LOCK X={LX:.2f} Z={LZ:.2f} t={tL_txt}s",
                           (20, 70), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        # ====================================================
        # ROBOT POSE VIA ARUCO (subsampled, but skip while THROWN)
        # ====================================================
        # While ball is in flight and prediction not done, skip ArUco to keep FPS high
        if not (state == "THROWN" and not prediction_done):
            aruco_frame_counter += 1
            if aruco_frame_counter % ARUCO_UPDATE_EVERY == 0:
                markers = process_aruco_frame(frameL, frameR, detectors, aruco_cam, marker_size_m)
                last_markers = markers
                if markers:
                    last_robot_pose = compute_robot_pose(markers)

        markers = last_markers
        robot_pose = last_robot_pose

        robot_pose_text = "Robot: NO MARKER"
        if robot_pose is not None:
            pos, yaw_deg, det_list = robot_pose
            rx, ry, rz = pos
            dist = float(np.sqrt(rx*rx + rz*rz))
            robot_pose_text = (f"Robot X={rx:+.2f} Y={ry:+.2f} Z={rz:+.2f} "
                               f"dist={dist:.2f} yaw={yaw_deg:+.1f}")

            colors = {
                'left':  (255, 0, 0),
                'front': (0, 255, 0),
                'right': (0, 0, 255),
                'back':  (255, 255, 0)
            }
            for m in det_list:
                color = colors.get(m.side_name, (255, 255, 255))
                cL = m.corners_left.astype(int)
                for i in range(4):
                    cv.line(frameL, tuple(cL[i]), tuple(cL[(i+1) % 4]), color, 2)
                center_l = cL.mean(axis=0).astype(int)
                cv.putText(frameL, m.side_name.upper(),
                           (center_l[0]-20, center_l[1]-10),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            udp.send_robot_pose(float(rx), float(ry), float(rz), float(yaw_deg))

        cv.putText(frameL, robot_pose_text, (20, 170),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv.putText(frameL,
                   f"UDP {'ON' if udp.enabled else 'OFF'} sent={udp.packets_sent}",
                   (20, 200),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)

        # Send predicted landing coordinates (0,0,0 until locked)
        if locked is not None:
            udp.send_predicted_landing(float(locked[0]), float(locked[1]), float(locked[2]))
        else:
            udp.send_predicted_landing(0.0, 0.0, 0.0)

        cv.putText(frameL, f"FPS={fps:4.1f}", (20, 230),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv.imshow("UnifiedView", frameL)
        key = cv.waitKey(1) & 0xFF

        if key == 27 or key == ord('q'):
            break
        elif key == ord('r'):
            traj_est.reset()
            locked = None
            locked_tL = None
            prediction_done = False
            vel_buffer.clear()
            state = "HOLD"
            throw_verify = 0
            last_smooth = None
            last_raw = None
            last_time = None
            landing_udp_sent = False
            print("[RESET BALL]")
        elif key == ord('u'):
            udp.toggle()

        frame_count += 1

    udp.close()
    zed.close()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
