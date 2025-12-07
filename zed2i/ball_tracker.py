import pyzed.sl as sl
import cv2 as cv
import numpy as np
import time
from collections import deque
import random

# ===========================
# CONSTANTS
# ===========================
G = 9.81
GROUND_Y = -0.52
MIN_SAMPLES = 5
RANSAC_ITERS = 200
RANSAC_INLIER_THRESH = 0.03
MAX_HISTORY = 60

# Velocity filters
VEL_WIN = 7                  # frames to average velocity over (more stable)
THROW_SPEED_THRESH = 1.4     # >1.4 m/s triggers throw
THROW_UPWARD_VY_THRESH = 0.7 # upward velocity threshold
THROW_VERIFY_FRAMES = 4      # consecutive frames to confirm throw

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

VIS_SCALE = 1.0
SMOOTH_ALPHA = 0.4


# ===========================
# LOAD INTRINSICS
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

    return np.array(K_rows), np.array(D_vals[:5]).reshape(-1, 1)


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

    return np.array(R_rows), np.array(T_vals).reshape(3, 1)


# ===========================
# DETECTOR
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
# STEREO
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
# TRAJECTORY ESTIMATOR
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
            return None
        r1 = (-b + np.sqrt(D)) / (2 * a)
        r2 = (-b - np.sqrt(D)) / (2 * a)
        cand = [t for t in (r1, r2) if t > 0]
        if not cand:
            return None
        return min(cand)

    def landing_point(self):
        P, inl = self.estimate()
        if P is None:
            return None, None
        tL = self._solve_t_land(P)
        if tL is None:
            return None, P
        X0, Y0, Z0, Vx, Vy, Vz = P
        return np.array([X0 + Vx * tL, GROUND_Y, Z0 + Vz * tL]), P


# ===========================
# MAIN
# ===========================

def main():
    detector = HSVBallDetector()
    model = StereoModel(INTR0_PATH, INTR1_PATH, EXTR1_PATH)
    est = BallTrajectoryEstimator()

    zed = sl.Camera()
    ip = sl.InitParameters()
    ip.camera_resolution = sl.RESOLUTION.HD720
    ip.camera_fps = 30
    ip.depth_mode = sl.DEPTH_MODE.NONE
    ip.coordinate_units = sl.UNIT.METER
    if zed.open(ip) != sl.ERROR_CODE.SUCCESS:
        print("ZED failed.")
        return

    left = sl.Mat()
    right = sl.Mat()

    last_smooth = None
    state = "IDLE"
    locked = None

    vel_buffer = deque(maxlen=VEL_WIN)
    last_raw = None
    last_time = None

    throw_verify = 0          # consecutive frames satisfying throw condition
    prediction_done = False   # True after first landing prediction; block further predictions

    while True:
        if zed.grab() != sl.ERROR_CODE.SUCCESS:
            continue

        zed.retrieve_image(left, sl.VIEW.LEFT)
        zed.retrieve_image(right, sl.VIEW.RIGHT)
        frameL = left.get_data()[:, :, :3].copy()

        detL = detector.detect(frameL)
        detR = detector.detect(right.get_data()[:, :, :3])

        if detL and detR:
            (uL, vL), rL = detL
            (uR, vR), rR = detR
            raw = triangulate_simple(uL, vL, uR, vR, model)
            if raw is None:
                cv.putText(frameL, "BAD TRIANGULATION", (20, 40),
                           cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv.imshow("Tracker", frameL)
                if cv.waitKey(1) in [27, ord('q')]:
                    break
                continue

            # -------- Smooth (for display & fitting) --------
            if last_smooth is None:
                smooth = raw
            else:
                smooth = SMOOTH_ALPHA * raw + (1 - SMOOTH_ALPHA) * last_smooth
            last_smooth = smooth

            # -------- Velocity estimation (raw) --------
            now = time.time()
            dt = 0
            if last_raw is not None and last_time is not None:
                dt = now - last_time
                if dt > 0:
                    v = (raw - last_raw) / dt
                    vel_buffer.append(v)

            last_raw = raw
            last_time = now

            v_est = np.mean(vel_buffer, axis=0) if len(vel_buffer) > 0 else np.zeros(3)
            speed = np.linalg.norm(v_est)
            vy = v_est[1]

            # =====================================
            #   STATE MACHINE
            # =====================================

            if state == "IDLE":
                state = "HOLD"
                throw_verify = 0

            elif state == "HOLD":
                # Only allow new throws if no prediction has been done yet
                est.reset()
                if not prediction_done:
                    locked = None

                    # Throw detection with consecutive-frame gating
                    if speed > THROW_SPEED_THRESH and vy > THROW_UPWARD_VY_THRESH:
                        throw_verify += 1
                    else:
                        throw_verify = 0

                    if throw_verify >= THROW_VERIFY_FRAMES:
                        print("[THROW DETECTED]")
                        state = "THROWN"
                        vel_buffer.clear()
                        est.reset()
                        est.add(now, smooth)
                        throw_verify = 0
                # if prediction_done == True, we stay here but do not trigger new throws

            elif state == "THROWN":
                # If we've already predicted once, do not update estimator again
                if not prediction_done:
                    est.add(now, smooth)
                    if locked is None:
                        land, _ = est.landing_point()
                        if land is not None:
                            locked = land
                            prediction_done = True
                            state = "LOCKED_STATE"
                            print("[LOCKED] Landing =", land)

            elif state == "LOCKED_STATE":
                # Completely frozen: no more prediction, no more updates
                pass

            # -------- Draw debug --------
            X, Y, Z = smooth
            cv.circle(frameL, (int(uL), int(vL)), int(rL), (0, 255, 0), 2)
            cv.putText(frameL, f"X={X:.2f} Y={Y:.2f} Z={Z:.2f}", (20, 40),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv.putText(frameL, f"speed={speed:.2f} vy={vy:.2f}", (20, 70),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv.putText(frameL, f"STATE={state}", (20, 100),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            if locked is not None:
                LX, LY, LZ = locked
                cv.putText(frameL, f"LAND LOCKED X={LX:.2f} Z={LZ:.2f}", (20, 130),
                           cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

        else:
            # Lost detection
            # If prediction is already done, DO NOT reset locked or state.
            if not prediction_done:
                state = "IDLE"
                est.reset()
                vel_buffer.clear()
                last_smooth = None
                last_raw = None
                locked = None
                throw_verify = 0

            cv.putText(frameL, "SEARCHING...", (20, 40),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if locked is not None and prediction_done:
                # Still show locked landing even if ball not visible
                LX, LY, LZ = locked
                cv.putText(frameL, f"LAND LOCKED X={LX:.2f} Z={LZ:.2f}", (20, 70),
                           cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

        cv.imshow("Tracker", frameL)
        k = cv.waitKey(1)
        if k == 27 or k == ord('q'):
            break
        elif k == ord('r'):
            # Manual reset clears everything including prediction lock
            est.reset()
            locked = None
            vel_buffer.clear()
            state = "HOLD"
            throw_verify = 0
            prediction_done = False
            last_smooth = None
            last_raw = None
            last_time = None
            print("[RESET]")

    zed.close()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
