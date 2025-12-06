"""
RealSense Stereo Ball Tracking with Triangulation and UDP Output

- Two calibrated RealSense RGB cameras
- Loads intrinsics/extrinsics from calib_output/stereo_calib.npz or stereo_calib.yaml
- Rectifies stereo pair, detects ball in both images, triangulates 3D position
- Sends (X, Y, Z) to robot over UDP using sender.UDPSender

Usage example:
    python track_ball.py \
        --calib-dir calib_output \
        --serial0 341522302002 \
        --serial1 213522253879 \
        --width 1280 --height 720 --fps 30 \
        --robot-ip 192.168.0.155 --port 5005

Options:
    --no-udp   Disable UDP sending

Keys:
    q/ESC  - Quit
    r      - Reset tracking history
    t      - Switch to tennis ball HSV
    p      - Switch to paper ball HSV
    u      - Toggle UDP sending on/off
"""

import argparse
import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2 as cv
import numpy as np
import pyrealsense2 as rs

from config import (
    HSV_TENNIS_BALL, HSV_PAPER_BALL, ACTIVE_HSV,
    MIN_BALL_AREA, MAX_BALL_AREA, MIN_CIRCULARITY,
    CAMERA_HEIGHT_ABOVE_GROUND, DEPTH_OFFSET,
    DEFAULT_ROBOT_IP, DEFAULT_ROBOT_PORT, UDP_SEND_RATE,
)

from sender import UDPSender


# ==================== DATA CLASSES ====================

@dataclass
class StereoCalibration:
    K0: np.ndarray
    dist0: np.ndarray
    K1: np.ndarray
    dist1: np.ndarray
    R: np.ndarray
    T: np.ndarray
    R0_rect: np.ndarray
    R1_rect: np.ndarray
    P0_rect: np.ndarray
    P1_rect: np.ndarray
    Q: np.ndarray
    image_size: Tuple[int, int]
    baseline: float
    fx_rect: float
    cx_rect: float
    cy_rect: float
    rms_mono0: float
    rms_mono1: float
    rms_stereo: float


# ==================== BALL DETECTOR ====================

class BallDetector:
    """Detects colored ball using HSV thresholding."""

    def __init__(self, hsv_config: dict):
        self.lower = np.array(hsv_config["lower"])
        self.upper = np.array(hsv_config["upper"])
        self.kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))

    def set_hsv_range(self, lower: tuple, upper: tuple):
        self.lower = np.array(lower)
        self.upper = np.array(upper)

    def detect(self, hsv_frame: np.ndarray) -> Optional[Tuple[int, int, int]]:
        mask = cv.inRange(hsv_frame, self.lower, self.upper)
        mask = cv.erode(mask, self.kernel, iterations=1)
        mask = cv.dilate(mask, self.kernel, iterations=2)

        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        best_detection = None
        best_score = 0.0

        for contour in contours:
            area = cv.contourArea(contour)
            if area < MIN_BALL_AREA or area > MAX_BALL_AREA:
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
                best_detection = (int(x), int(y), int(radius))

        return best_detection


# ==================== DRAWING / HUD ====================

def draw_detection(frame: np.ndarray, detection: Tuple[int, int, int],
                   label: str, color: Tuple[int, int, int]):
    x, y, radius = detection
    cv.circle(frame, (x, y), radius, color, 2)
    cv.circle(frame, (x, y), 3, color, -1)
    if label:
        cv.putText(frame, label, (x + radius + 5, y),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def draw_tracking_overlay(frame: np.ndarray,
                          detection_left: Optional[Tuple[int, int, int]],
                          detection_right: Optional[Tuple[int, int, int]],
                          pos_3d: Optional[np.ndarray],
                          param_source: str,
                          triangulation_success: bool,
                          udp_sender: Optional[UDPSender]):
    h, w = frame.shape[:2]
    y = 25

    cv.putText(frame, "RealSense Stereo Ball Tracker", (10, y),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    y += 25

    if udp_sender:
        udp_status = f"UDP: {udp_sender.robot_ip}:{udp_sender.port}"
        udp_color = (0, 255, 0) if udp_sender.enabled else (0, 0, 255)
        status_text = "ON" if udp_sender.enabled else "OFF"
        cv.putText(frame, f"{udp_status} [{status_text}]", (10, y),
                   cv.FONT_HERSHEY_SIMPLEX, 0.45, udp_color, 1)
        y += 20
        cv.putText(frame, f"Packets sent: {udp_sender.packets_sent}", (10, y),
                   cv.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        y += 25

    ball_detected = (detection_left is not None) and (detection_right is not None)
    status = "TRACKING" if ball_detected else "SEARCHING"
    color = (0, 255, 0) if ball_detected else (0, 165, 255)
    cv.putText(frame, f"Ball: {status}", (10, y),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    y += 25

    tri_status = "OK" if triangulation_success else "FAILED"
    tri_color = (0, 255, 0) if triangulation_success else (0, 0, 255)
    cv.putText(frame, f"3D: {tri_status} ({param_source})", (10, y),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, tri_color, 1)
    y += 30

    if pos_3d is not None:
        cv.putText(frame, "3D Position (m):", (10, y),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y += 22
        cv.putText(frame, f"  X: {pos_3d[0]:+.3f}", (10, y),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1)
        y += 22
        cv.putText(frame, f"  Y: {pos_3d[1]:+.3f}", (10, y),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
        y += 22
        cv.putText(frame, f"  Z: {pos_3d[2]:+.3f}", (10, y),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 1)

    cv.putText(
        frame,
        "[R] Reset  [T] Tennis  [P] Paper  [U] Toggle UDP  [Q] Quit",
        (10, h - 10),
        cv.FONT_HERSHEY_SIMPLEX,
        0.4,
        (150, 150, 150),
        1,
    )


# ==================== CALIBRATION LOADING ====================

def load_stereo_calibration(calib_dir: str,
                            image_size: Tuple[int, int]) -> StereoCalibration:
    """
    Load stereo calibration from calib_output:
    - Prefer stereo_calib.npz
    - Fallback to stereo_calib.yaml
    """
    w, h = image_size
    size = (w, h)

    npz_path = os.path.join(calib_dir, "stereo_calib.npz")
    yaml_path = os.path.join(calib_dir, "stereo_calib.yaml")

    if os.path.exists(npz_path):
        print(f"[CALIB] Loading NPZ calibration: {npz_path}")
        data = np.load(npz_path)
        K0 = data["K0"]
        dist0 = data["dist0"]
        K1 = data["K1"]
        dist1 = data["dist1"]
        R = data["R"]
        T = data["T"]
        rms_mono0 = float(data["rms_mono0"])
        rms_mono1 = float(data["rms_mono1"])
        rms_stereo = float(data["rms_stereo"])
        param_source = "NPZ"
    elif os.path.exists(yaml_path):
        print(f"[CALIB] Loading YAML calibration: {yaml_path}")
        import yaml
        with open(yaml_path, "r") as f:
            ydata = yaml.safe_load(f)

        K0 = np.array(ydata["camera0"]["K"], dtype=np.float64)
        dist0 = np.array(ydata["camera0"]["dist"], dtype=np.float64)
        K1 = np.array(ydata["camera1"]["K"], dtype=np.float64)
        dist1 = np.array(ydata["camera1"]["dist"], dtype=np.float64)
        R = np.array(ydata["stereo"]["R"], dtype=np.float64)
        T = np.array(ydata["stereo"]["T"], dtype=np.float64)
        rms_mono0 = float(ydata["camera0"]["mono_rms"])
        rms_mono1 = float(ydata["camera1"]["mono_rms"])
        rms_stereo = float(ydata["stereo"]["rms"])
        param_source = "YAML"
    else:
        raise FileNotFoundError(
            f"No stereo calibration found in {calib_dir} "
            f"(expected stereo_calib.npz or stereo_calib.yaml)"
        )

    R0_rect, R1_rect, P0_rect, P1_rect, Q, _, _ = cv.stereoRectify(
        K0, dist0, K1, dist1, size, R, T,
        flags=cv.CALIB_ZERO_DISPARITY, alpha=0
    )

    fx_rect = P0_rect[0, 0]
    cx_rect = P0_rect[0, 2]
    cy_rect = P0_rect[1, 2]
    baseline = -P1_rect[0, 3] / fx_rect  # meters

    print(f"[CALIB] Source = {param_source}")
    print(f"[CALIB] mono0 RMS = {rms_mono0:.4f} px, mono1 RMS = {rms_mono1:.4f} px")
    print(f"[CALIB] stereo RMS = {rms_stereo:.4f} px")
    print(f"[CALIB] baseline = {baseline:.4f} m, fx_rect = {fx_rect:.2f} px")

    return StereoCalibration(
        K0=K0, dist0=dist0,
        K1=K1, dist1=dist1,
        R=R, T=T,
        R0_rect=R0_rect,
        R1_rect=R1_rect,
        P0_rect=P0_rect,
        P1_rect=P1_rect,
        Q=Q,
        image_size=size,
        baseline=float(baseline),
        fx_rect=float(fx_rect),
        cx_rect=float(cx_rect),
        cy_rect=float(cy_rect),
        rms_mono0=rms_mono0,
        rms_mono1=rms_mono1,
        rms_stereo=rms_stereo,
    )


def create_rectification_maps(calib: StereoCalibration):
    w, h = calib.image_size
    map10, map11 = cv.initUndistortRectifyMap(
        calib.K0, calib.dist0,
        calib.R0_rect, calib.P0_rect,
        (w, h), cv.CV_32FC1
    )
    map20, map21 = cv.initUndistortRectifyMap(
        calib.K1, calib.dist1,
        calib.R1_rect, calib.P1_rect,
        (w, h), cv.CV_32FC1
    )
    return map10, map11, map20, map21


# ==================== REALSENSE HELPERS ====================

def open_realsense_camera(serial: str, width: int, height: int, fps: int) -> rs.pipeline:
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(str(serial))
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

    print(f"[INFO] Opening RealSense {serial} at {width}x{height}@{fps}...")
    profile = pipeline.start(config)
    print(f"[INFO] RealSense {serial} opened.")

    # Load per-camera preset
    try:
        dev = profile.get_device()

        # if you have separate presets, choose based on serial
        if serial == "341522302002":   # your left cam
            preset_path = "realsense1.json"
        elif serial == "213522253879": # your right cam
            preset_path = "realsense2.json"
        else:
            preset_path = "realsense3.json"

        if os.path.exists(preset_path):
            with open(preset_path, "r") as f:
                js = f.read()
            dev.load_json(js)
            print(f"[INFO] Loaded RealSense preset from {preset_path}")
        else:
            print(f"[WARN] Preset JSON not found: {preset_path}")

    except Exception as e:
        print(f"[WARN] Could not load RealSense JSON preset: {e}")

    return pipeline


# ==================== TRIANGULATION ====================

def triangulate_point_rectified(u0: float, v0: float,
                                u1: float, v1: float,
                                calib: StereoCalibration) -> Optional[np.ndarray]:
    """
    Triangulate from rectified pixel coordinates using OpenCV's triangulatePoints.
    """
    pts0 = np.array([[u0], [v0]], dtype=np.float64)
    pts1 = np.array([[u1], [v1]], dtype=np.float64)

    homog = cv.triangulatePoints(calib.P0_rect, calib.P1_rect, pts0, pts1)
    w = homog[3, 0]
    if abs(w) < 1e-8:
        return None

    X = homog[:3, 0] / w  # [X, Y, Z]
    if not np.isfinite(X).all():
        return None

    Z = X[2]
    if Z <= 0 or Z > 50.0:  # discard behind camera or absurdly far
        return None

    return X.astype(np.float32)


# ==================== MAIN LOOP ====================

def main():
    parser = argparse.ArgumentParser(description="RealSense stereo ball tracking")
    parser.add_argument("--calib-dir", type=str, default="calib_output",
                        help="Directory containing stereo_calib.npz or stereo_calib.yaml")
    parser.add_argument("--serial0", type=str, default=341522302002,
                        help="RealSense serial number for left camera (camera0)")
    parser.add_argument("--serial1", type=str, default=213522253879,
                        help="RealSense serial number for right camera (camera1)")
    parser.add_argument("--width", type=int, default=1280,
                        help="Color frame width (must match calibration)")
    parser.add_argument("--height", type=int, default=720,
                        help="Color frame height (must match calibration)")
    parser.add_argument("--fps", type=int, default=30,
                        help="Color frame FPS")
    parser.add_argument("--robot-ip", type=str, default=DEFAULT_ROBOT_IP,
                        help=f"Robot IP (default: {DEFAULT_ROBOT_IP})")
    parser.add_argument("--port", type=int, default=DEFAULT_ROBOT_PORT,
                        help=f"Robot UDP port (default: {DEFAULT_ROBOT_PORT})")
    parser.add_argument("--no-udp", action="store_true",
                        help="Disable UDP sending")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print(" REALSENSE STEREO BALL TRACKING")
    print("=" * 60)

    width, height, fps = args.width, args.height, args.fps
    image_size = (width, height)

    calib = load_stereo_calibration(args.calib_dir, image_size)
    map10, map11, map20, map21 = create_rectification_maps(calib)

    udp_sender = None
    if not args.no_udp:
        udp_sender = UDPSender(args.robot_ip, args.port, rate_limit=UDP_SEND_RATE)
        print(f"[UDP] Sender to {args.robot_ip}:{args.port}, rate {UDP_SEND_RATE} Hz")
    else:
        print("[UDP] UDP sending disabled")

    detector = BallDetector(ACTIVE_HSV)

    pipe0 = open_realsense_camera(args.serial0, width, height, fps)
    pipe1 = open_realsense_camera(args.serial1, width, height, fps)

    last_valid_pos = None
    last_valid_time = 0.0
    POSITION_TIMEOUT = 0.5  # seconds

    # For temporal filtering of 2D detections
    last_det0 = None
    last_det1 = None
    last_det_time0 = 0.0
    last_det_time1 = 0.0
    DET_TIMEOUT = 0.3        # seconds for 2D persistence
    MAX_JUMP_PIXELS = 200.0  # max allowed jump in pixels

    param_source = "Stereo calib"
    epipolar_tol = 5.0       # allowed |v0 - v1| in rectified images

    try:
        while True:
            frames0 = pipe0.wait_for_frames()
            frames1 = pipe1.wait_for_frames()
            color0 = frames0.get_color_frame()
            color1 = frames1.get_color_frame()
            if not color0 or not color1:
                continue

            frame0 = np.asanyarray(color0.get_data())
            frame1 = np.asanyarray(color1.get_data())

            rect0 = cv.remap(frame0, map10, map11, cv.INTER_LINEAR)
            rect1 = cv.remap(frame1, map20, map21, cv.INTER_LINEAR)

            hsv0 = cv.cvtColor(rect0, cv.COLOR_BGR2HSV)
            hsv1 = cv.cvtColor(rect1, cv.COLOR_BGR2HSV)

            det0_raw = detector.detect(hsv0)
            det1_raw = detector.detect(hsv1)
            now = time.time()

            # Temporal filter on 2D detections: reject wild jumps
            det0 = det0_raw
            if det0_raw is not None:
                if last_det0 is not None and (now - last_det_time0) < DET_TIMEOUT:
                    x0, y0, r0 = det0_raw
                    x_prev, y_prev, _ = last_det0
                    jump = np.hypot(x0 - x_prev, y0 - y_prev)
                    if jump > MAX_JUMP_PIXELS:
                        # treat as noise, keep previous if still fresh
                        det0 = last_det0
                    else:
                        last_det0 = det0_raw
                        last_det_time0 = now
                else:
                    last_det0 = det0_raw
                    last_det_time0 = now
            else:
                if last_det0 is not None and (now - last_det_time0) < DET_TIMEOUT:
                    det0 = last_det0
                else:
                    det0 = None
                    last_det0 = None

            det1 = det1_raw
            if det1_raw is not None:
                if last_det1 is not None and (now - last_det_time1) < DET_TIMEOUT:
                    x1, y1, r1 = det1_raw
                    x_prev, y_prev, _ = last_det1
                    jump = np.hypot(x1 - x_prev, y1 - y_prev)
                    if jump > MAX_JUMP_PIXELS:
                        det1 = last_det1
                    else:
                        last_det1 = det1_raw
                        last_det_time1 = now
                else:
                    last_det1 = det1_raw
                    last_det_time1 = now
            else:
                if last_det1 is not None and (now - last_det_time1) < DET_TIMEOUT:
                    det1 = last_det1
                else:
                    det1 = None
                    last_det1 = None

            pos_3d = None
            tri_success = False

            if det0 is not None and det1 is not None:
                x0, y0, r0 = det0
                x1, y1, r1 = det1

                # Epipolar sanity: in rectified images v coordinates should be close
                if abs(y0 - y1) < epipolar_tol:
                    draw_detection(rect0, det0, "L", (0, 255, 0))
                    draw_detection(rect1, det1, "R", (0, 255, 0))

                    world = triangulate_point_rectified(x0, y0, x1, y1, calib)
                    if world is not None:
                        pos_3d = world.copy()
                        pos_3d[1] += CAMERA_HEIGHT_ABOVE_GROUND
                        pos_3d[2] += DEPTH_OFFSET

                        tri_success = True
                        last_valid_pos = pos_3d.copy()
                        last_valid_time = now

                        if udp_sender and udp_sender.enabled:
                            udp_sender.send_coordinates(
                                float(pos_3d[0]),
                                float(pos_3d[1]),
                                float(pos_3d[2]),
                            )
                else:
                    # v0-v1 not consistent -> likely mismatch or bogus detection
                    if last_valid_pos is not None and (now - last_valid_time) < POSITION_TIMEOUT:
                        pos_3d = last_valid_pos
                        tri_success = False
                    else:
                        pos_3d = None
                        tri_success = False
            else:
                # No valid detection in one/both views
                if last_valid_pos is not None and (time.time() - last_valid_time) >= POSITION_TIMEOUT:
                    last_valid_pos = None

            display = rect0.copy()
            draw_tracking_overlay(
                display,
                det0,
                det1,
                pos_3d,
                param_source,
                tri_success,
                udp_sender,
            )

            right_small = cv.resize(rect1, None, fx=0.5, fy=0.5)
            cv.imshow("Right Rectified", right_small)
            cv.imshow("Left Rectified + HUD", display)

            key = cv.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("r"):
                last_valid_pos = None
                last_det0 = None
                last_det1 = None
                print("[RESET] Cleared history")
            elif key == ord("t"):
                detector.set_hsv_range(
                    HSV_TENNIS_BALL["lower"], HSV_TENNIS_BALL["upper"]
                )
                print("[HSV] Tennis ball range")
            elif key == ord("p"):
                detector.set_hsv_range(
                    HSV_PAPER_BALL["lower"], HSV_PAPER_BALL["upper"]
                )
                print("[HSV] Paper ball range")
            elif key == ord("u"):
                if udp_sender:
                    udp_sender.toggle()

    finally:
        cv.destroyAllWindows()
        pipe0.stop()
        pipe1.stop()
        if udp_sender:
            print(f"\n[UDP] Total packets sent: {udp_sender.packets_sent}")
            udp_sender.close()
        print("[INFO] Tracking stopped")


if __name__ == "__main__":
    main()
