#!/usr/bin/env python3
"""
Robot Pose Estimator using ArUco PnP (single calibrated camera)

Camera 0 frame (USER frame used for all outputs):
    +X: right
    +Y: up
    +Z: forward (into the scene, away from camera)

Robot moves in X-Z plane. We estimate:
    - Position of robot in camera frame (from marker centers)
    - Heading yaw of robot (orientation of robot's FRONT in camera frame).

Yaw definition:
    yaw = atan2(H_x, H_z)
    where H is the ROBOT FRONT direction in camera frame.

So:
    - yaw ≈ 0°  → robot front points away from camera (+Z)
    - yaw ≈ +90° → robot front points to camera's RIGHT (+X)
    - yaw ≈ -90° → robot front points to camera's LEFT (-X)

FPS optimization:
    - Pose is computed ONLY from LEFT camera (unchanged).
    - RIGHT camera is shown raw, without ArUco detection or drawing.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import argparse
import os

# Marker configuration
MARKER_SIZE_MM = 95.0
MARKER_SIZE_M = MARKER_SIZE_MM / 1000.0


@dataclass
class CameraParams:
    """Camera intrinsics and extrinsics (extr not used for pose here)."""
    intrinsic: np.ndarray
    distortion: np.ndarray
    rotation: np.ndarray
    translation: np.ndarray


@dataclass
class StereoSystem:
    """Stereo system container; only cam0 intrinsics/distortion matter for pose."""
    cam0: CameraParams
    cam1: CameraParams  # kept for completeness


@dataclass
class DetectedMarker:
    """Detected marker with pose from left camera."""
    side_name: str
    corners_left: np.ndarray            # 4x2 corners in left image (pixels)
    corners_right: Optional[np.ndarray] # 4x2 corners in right image, or None
    corners_3d: np.ndarray              # 4x3 3D points (meters, USER frame)
    position: np.ndarray                # 3D center (meters, USER frame)
    yaw_angle: float                    # robot HEADING yaw (deg, about +Y)
    rotation_matrix: np.ndarray         # marker->camera rotation (USER frame)


# ---------------------------------------------------------------------------
# Calibration loading
# ---------------------------------------------------------------------------

def _interpret_translation_units(translation_raw: np.ndarray) -> np.ndarray:
    """
    Heuristic to interpret ZED-style T units and convert to meters.

    - If norm < 1   -> assume already meters
    - If 1 < norm < 100  -> assume centimeters -> divide by 100
    - If norm >= 100     -> assume millimeters -> divide by 1000

    This is only used for completeness; extrinsics are NOT used for pose.
    """
    t = translation_raw.astype(np.float64)
    n = np.linalg.norm(t)
    if n < 1.0:
        return t
    elif n < 100.0:
        return t / 100.0
    else:
        return t / 1000.0


def load_camera_params(intrinsics_path: str, extrinsics_path: str) -> CameraParams:
    """Load camera parameters from your existing .dat files."""

    # ---- Intrinsics ----
    with open(intrinsics_path, 'r') as f:
        lines = f.readlines()

    intrinsic_rows = []
    distortion = None

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line == 'intrinsic:':
            for j in range(3):
                row = [float(x) for x in lines[i + 1 + j].strip().split()]
                intrinsic_rows.append(row)
            i += 4
        elif line == 'distortion:':
            distortion = np.array([float(x) for x in lines[i + 1].strip().split()])
            i += 2
        else:
            i += 1

    intrinsic = np.array(intrinsic_rows, dtype=np.float64)

    # ---- Extrinsics ----
    with open(extrinsics_path, 'r') as f:
        lines = f.readlines()

    rotation_rows = []
    translation = []

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line == 'R:':
            for j in range(3):
                row = [float(x) for x in lines[i + 1 + j].strip().split()]
                rotation_rows.append(row)
            i += 4
        elif line == 'T:':
            for j in range(3):
                translation.append(float(lines[i + 1 + j].strip()))
            i += 4
        else:
            i += 1

    rotation = np.array(rotation_rows, dtype=np.float64)
    translation_raw = np.array(translation, dtype=np.float64)
    translation_m = _interpret_translation_units(translation_raw)

    return CameraParams(
        intrinsic=intrinsic,
        distortion=distortion,
        rotation=rotation,
        translation=translation_m
    )


def load_stereo_system(calibration_dir: str) -> StereoSystem:
    """
    Load both camera params. For pose, we only use cam0 intrinsics/distortion.
    """

    cam0 = load_camera_params(
        os.path.join(calibration_dir, 'camera0_intrinsics.dat'),
        os.path.join(calibration_dir, 'camera0_rot_trans.dat')
    )

    cam1 = load_camera_params(
        os.path.join(calibration_dir, 'camera1_intrinsics.dat'),
        os.path.join(calibration_dir, 'camera1_rot_trans.dat')
    )

    return StereoSystem(cam0=cam0, cam1=cam1)


# ---------------------------------------------------------------------------
# ArUco creation and detection
# ---------------------------------------------------------------------------

def create_aruco_detectors() -> Dict[str, cv2.aruco.ArucoDetector]:
    """Create ArUco detectors for each side (using different dictionaries)."""

    detectors = {}
    dict_types = {
        'left':  cv2.aruco.DICT_4X4_50,
        'front': cv2.aruco.DICT_5X5_50,
        'right': cv2.aruco.DICT_6X6_50,
        'back':  cv2.aruco.DICT_7X7_50
    }

    for side_name, dict_type in dict_types.items():
        dictionary = cv2.aruco.getPredefinedDictionary(dict_type)
        parameters = cv2.aruco.DetectorParameters()
        parameters.adaptiveThreshWinSizeMin = 3
        parameters.adaptiveThreshWinSizeMax = 23
        parameters.adaptiveThreshWinSizeStep = 10
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        detectors[side_name] = detector

    return detectors


def detect_corners(
    frame: np.ndarray,
    detectors: Dict[str, cv2.aruco.ArucoDetector]
) -> Dict[str, np.ndarray]:
    """Detect ArUco marker corners in a frame for each side (ID=0)."""

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected: Dict[str, np.ndarray] = {}

    for side_name, detector in detectors.items():
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is not None:
            for i, marker_id in enumerate(ids.flatten()):
                if marker_id == 0:
                    detected[side_name] = corners[i].reshape(4, 2)  # (4,2)
                    break

    return detected


# ---------------------------------------------------------------------------
# Marker model and pose from PnP
# ---------------------------------------------------------------------------

def get_marker_model_points(marker_size: float) -> np.ndarray:
    """
    3D coordinates of marker corners in the marker's local frame.

    Marker lies in Z=0 plane, centered at origin.
    Corner order matches ArUco: TL, TR, BR, BL.

    We define Y axis positive UP in marker frame here.
    """
    half = marker_size / 2.0
    model_points = np.array([
        [-half,  half, 0.0],  # top-left
        [ half,  half, 0.0],  # top-right
        [ half, -half, 0.0],  # bottom-right
        [-half, -half, 0.0],  # bottom-left
    ], dtype=np.float64)
    return model_points


def heading_yaw_from_marker(
    side_name: str,
    R_cam: np.ndarray
) -> float:
    """
    Compute robot heading yaw (deg) from marker orientation and side name.

    R_cam: 3x3 rotation from marker frame to CAMERA frame (OpenCV camera frame).
    We extract marker normal n_cam and map to robot front direction H_cam
    depending on which side the marker is mounted on.

    yaw = atan2(H_x, H_z) (in camera frame)
    """

    # z-axis of marker in camera frame: marker normal
    n_cam = R_cam[:, 2].astype(np.float64)
    n_norm = np.linalg.norm(n_cam)
    if n_norm < 1e-8:
        return 0.0
    n_cam /= n_norm

    # Map marker normal to robot FRONT direction in camera frame
    if side_name == 'back':
        # back face: normal points towards camera when yaw ≈ 0
        # robot front is opposite of that
        H_cam = -n_cam
    elif side_name == 'front':
        # front face: normal aligns with robot front direction
        H_cam = n_cam
    elif side_name == 'right':
        # right side: robot front is -90° yaw from normal
        # R_y(-90°) * n_cam
        H_cam = np.array([-n_cam[2], n_cam[1], n_cam[0]])
    elif side_name == 'left':
        # left side: robot front is +90° yaw from normal
        # R_y(+90°) * n_cam
        H_cam = np.array([n_cam[2], n_cam[1], -n_cam[0]])
    else:
        # Fallback: treat as back
        H_cam = -n_cam

    # Project to XZ plane
    H_xz = np.array([H_cam[0], H_cam[2]], dtype=np.float64)
    h_norm = np.linalg.norm(H_xz)
    if h_norm < 1e-8:
        return 0.0
    H_xz /= h_norm

    yaw_rad = np.arctan2(H_xz[0], H_xz[1])  # atan2(X, Z)
    yaw_deg = np.degrees(yaw_rad)

    return float(yaw_deg)


def  pose_from_aruco_pnp(
    side_name: str,
    corners_px: np.ndarray,
    camera: CameraParams,
    marker_size_m: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Estimate marker pose from a single calibrated camera using ArUco PnP.

    Args:
        side_name: 'front', 'back', 'left', 'right'
        corners_px: (4,2) pixel coords in left image
        camera: CameraParams for cam0 (intrinsic + distortion)
        marker_size_m: size in meters

    Returns:
        R_user: 3x3 marker->camera rotation (USER frame, +Y up)
        t_user: 3-vector translation (meters, USER frame)
        corners_3d_user: (4,3) 3D points in USER frame
        yaw_deg: robot HEADING yaw (deg) about +Y, as defined above
    """

    # ArUco expects corners shape (N,1,4,2)
    corners_input = corners_px.reshape(1, 1, 4, 2).astype(np.float32)

    # estimatePoseSingleMarkers: marker size in meters
    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
        corners_input,
        marker_size_m,
        camera.intrinsic,
        camera.distortion
    )

    rvec = rvecs[0, 0, :]  # (3,)
    tvec = tvecs[0, 0, :]  # (3,)

    # Rotation from marker frame to CAMERA frame (OpenCV camera: +X right, +Y down, +Z forward)
    R_cam, _ = cv2.Rodrigues(rvec)  # 3x3
    t_cam = tvec.reshape(3, 1)      # 3x1

    # Robot heading yaw from this marker (in CAMERA frame)
    yaw_deg = heading_yaw_from_marker(side_name, R_cam)

    # Convert to USER frame: +Y up (flip Y)
    S = np.diag([1.0, -1.0, 1.0])  # camera -> user
    R_user = S @ R_cam
    t_user = S @ t_cam

    # Marker corners in marker frame
    obj_pts = get_marker_model_points(marker_size_m)  # 4x3
    corners_3d_user = []
    for p in obj_pts:
        p_cam = R_user @ p.reshape(3, 1) + t_user  # 3x1
        corners_3d_user.append(p_cam.ravel())
    corners_3d_user = np.array(corners_3d_user)  # (4,3)

    return R_user, t_user.ravel(), corners_3d_user, yaw_deg


# ---------------------------------------------------------------------------
# Frame processing and robot pose aggregation
# ---------------------------------------------------------------------------

def process_stereo_frame(
    left_frame: np.ndarray,
    right_frame: np.ndarray,
    detectors: Dict[str, cv2.aruco.ArucoDetector],
    stereo: StereoSystem,
    marker_size_m: float,
    marker_tilt: float  # unused, kept for CLI compatibility
) -> List[DetectedMarker]:
    """
    Process a stereo frame pair:

    - Detection + pose ONLY on LEFT (to keep logic and pose identical).
    - RIGHT frame is passed through just for display (no detection).
    """

    corners_left = detect_corners(left_frame, detectors)
    # FPS optimization: no ArUco detection on right frame
    # corners_right = detect_corners(right_frame, detectors)

    detected_markers: List[DetectedMarker] = []

    for side_name, pts_left in corners_left.items():
        # pose from left camera only (unchanged logic)
        R_user, t_user, corners_3d_user, yaw_deg = pose_from_aruco_pnp(
            side_name,
            pts_left,
            stereo.cam0,
            marker_size_m
        )

        # position: marker origin (center) in USER camera frame
        position = t_user  # 3-vector

        # For right image: we don't draw markers anymore (no detection).
        pts_right = None

        detected_markers.append(DetectedMarker(
            side_name=side_name,
            corners_left=pts_left,
            corners_right=pts_right,
            corners_3d=corners_3d_user,
            position=position,
            yaw_angle=yaw_deg,
            rotation_matrix=R_user
        ))

    return detected_markers


def compute_robot_pose(
    markers: List[DetectedMarker]
) -> Optional[Tuple[np.ndarray, float, List[DetectedMarker]]]:
    """
    Compute robot pose from detected markers.

    Position:
        average of all marker centers in camera-0 USER frame.

    Heading yaw:
        take yaw from a priority marker (uses orientation-based yaw).

        Priority: back > front > left > right
        - When robot front points away from camera (yaw ≈ 0), back marker is dominant.
        - When robot front faces camera, front marker yaw ≈ ±180°.
        - When robot side-on, left/right give yaw ≈ ±90°.
    """

    if not markers:
        return None

    # Average position
    positions = np.array([m.position for m in markers])  # (N,3)
    avg_position = np.mean(positions, axis=0)

    # Choose yaw from priority marker
    priority = {'back': 0, 'front': 1, 'left': 2, 'right': 3}
    best = min(markers, key=lambda m: priority.get(m.side_name, 99))
    yaw = best.yaw_angle

    return avg_position, yaw, markers


# ---------------------------------------------------------------------------
# Visualization / printing
# ---------------------------------------------------------------------------

def draw_results(
    left_frame: np.ndarray,
    right_frame: np.ndarray,
    markers: List[DetectedMarker],
    robot_pose: Optional[Tuple[np.ndarray, float, List[DetectedMarker]]]
) -> np.ndarray:
    """
    Draw detection results on the stereo frames.

    - Left: markers + pose overlay.
    - Right: raw image only (no detection) for FPS.
    """

    left_result = left_frame.copy()
    right_result = right_frame.copy()

    colors = {
        'left':  (255, 0, 0),
        'front': (0, 255, 0),
        'right': (0, 0, 255),
        'back':  (255, 255, 0)
    }

    # Draw markers on LEFT only
    for marker in markers:
        color = colors.get(marker.side_name, (255, 255, 255))

        corners_l = marker.corners_left.astype(int)
        for i in range(4):
            cv2.line(left_result,
                     tuple(corners_l[i]),
                     tuple(corners_l[(i + 1) % 4]),
                     color, 2)
        center_l = corners_l.mean(axis=0).astype(int)
        cv2.putText(left_result, marker.side_name.upper(),
                    (center_l[0] - 30, center_l[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Info panel on LEFT
    if robot_pose is not None:
        position, yaw, detected = robot_pose
        distance = np.sqrt(position[0] ** 2 + position[2] ** 2)

        info_lines = [
            "ROBOT POSE (Camera 0 USER Frame)",
            "Frame: +X right, +Y up, +Z forward",
            f"Position (m):",
            f"  X: {position[0]:+.3f} (right)",
            f"  Y: {position[1]:+.3f} (up)",
            f"  Z: {position[2]:+.3f} (forward)",
            f"Distance (XZ): {distance:.3f} m",
            f"Heading yaw: {yaw:+.1f} deg",
            f"Markers: {', '.join(m.side_name for m in detected)}"
        ]

        y_offset = 30
        for line in info_lines:
            cv2.putText(left_result, line, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            y_offset += 22
    else:
        cv2.putText(left_result, "No markers detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    combined = np.hstack([left_result, right_result])
    cv2.putText(combined, "LEFT (Camera 0)", (10, combined.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(combined, "RIGHT (Camera 1)",
                (left_result.shape[1] + 10, combined.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return combined


def print_pose_info(robot_pose: Tuple[np.ndarray, float, List[DetectedMarker]]):
    """Print robot pose (Camera 0 USER frame)."""

    position, yaw, markers = robot_pose
    distance = np.sqrt(position[0] ** 2 + position[2] ** 2)

    print("\n" + "=" * 60)
    print("ROBOT POSE (w.r.t. Camera 0 USER Frame)")
    print("=" * 60)
    print("Frame: +X right, +Y up, +Z forward\n")

    print(f"Detected Markers ({len(markers)}):")
    for m in markers:
        print(f"  {m.side_name.upper():6}: "
              f"X={m.position[0]:+.3f}, "
              f"Y={m.position[1]:+.3f}, "
              f"Z={m.position[2]:+.3f} m, "
              f"marker yaw: {m.yaw_angle:+.1f}°")

    print("\nRobot Position (m):")
    print(f"  X: {position[0]:+.3f} (right)")
    print(f"  Y: {position[1]:+.3f} (up)")
    print(f"  Z: {position[2]:+.3f} (forward)")
    print(f"  Distance (XZ plane): {distance:.3f} m")

    print("\nRobot Heading:")
    print("  yaw = atan2(H_x, H_z), where H is robot FRONT direction in camera frame")
    print(f"  Heading yaw: {yaw:+.1f}° "
          f"(0°: away from camera, +90°: right, -90°: left)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Robot Pose Estimator using ArUco PnP (single camera)'
    )
    parser.add_argument('--calibration-dir', type=str, default='./camera_parameters',
                        help='Directory containing calibration files')
    parser.add_argument('--marker-size', type=float, default=95.0,
                        help='ArUco marker size in mm')
    parser.add_argument('--marker-tilt', type=float, default=45.0,
                        help='Marker tilt angle (info only)')
    parser.add_argument('--camera-id', type=int, default=0,
                        help='Camera device ID')
    parser.add_argument('--image', type=str, default=None,
                        help='Path to stereo image (side-by-side)')
    parser.add_argument('--image-left', type=str, default=None,
                        help='Path to left camera image')
    parser.add_argument('--image-right', type=str, default=None,
                        help='Path to right camera image')
    parser.add_argument('--save-output', type=str, default=None,
                        help='Path to save output image')
    parser.add_argument('--no-display', action='store_true',
                        help='Disable display window')
    args = parser.parse_args()

    marker_size_m = args.marker_size / 1000.0

    print(f"Loading camera parameters from: {args.calibration_dir}")
    stereo = load_stereo_system(args.calibration_dir)

    print("\nCamera 0 (Left):")
    print(f"  fx={stereo.cam0.intrinsic[0, 0]:.2f}, "
          f"fy={stereo.cam0.intrinsic[1, 1]:.2f}")
    print(f"  cx={stereo.cam0.intrinsic[0, 2]:.2f}, "
          f"cy={stereo.cam0.intrinsic[1, 2]:.2f}")

    print("\nArUco Configuration:")
    print("  4x4 (ID=0) → LEFT side")
    print("  5x5 (ID=0) → FRONT")
    print("  6x6 (ID=0) → RIGHT side")
    print("  7x7 (ID=0) → BACK")
    detectors = create_aruco_detectors()

    # ---- Image mode ----
    if args.image or (args.image_left and args.image_right):
        if args.image:
            frame = cv2.imread(args.image)
            if frame is None:
                print(f"Error: Could not load image {args.image}")
                return
            h, w = frame.shape[:2]
            left_frame = frame[:, :w // 2]
            right_frame = frame[:, w // 2:]
        else:
            left_frame = cv2.imread(args.image_left)
            right_frame = cv2.imread(args.image_right)
            if left_frame is None or right_frame is None:
                print("Error: Could not load image pair")
                return

        markers = process_stereo_frame(
            left_frame, right_frame, detectors, stereo,
            marker_size_m, args.marker_tilt
        )
        robot_pose = compute_robot_pose(markers)

        if robot_pose:
            print_pose_info(robot_pose)
        else:
            print("\nNo markers detected in left view!")

        result = draw_results(left_frame, right_frame, markers, robot_pose)

        if args.save_output:
            cv2.imwrite(args.save_output, result)
            print(f"\nOutput saved to: {args.save_output}")

        if not args.no_display:
            cv2.imshow('Robot Pose (ArUco PnP)', result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    # ---- Live mode ----
    else:
        print(f"\nOpening camera {args.camera_id}...")
        cap = cv2.VideoCapture(args.camera_id)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        print("\nControls: 'q'=Quit, 's'=Save frame, 'p'=Print pose")
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            left_frame = frame[:, :w // 2]
            right_frame = frame[:, w // 2:]

            markers = process_stereo_frame(
                left_frame, right_frame, detectors, stereo,
                marker_size_m, args.marker_tilt
            )
            robot_pose = compute_robot_pose(markers)

            result = draw_results(left_frame, right_frame, markers, robot_pose)
            cv2.imshow('Robot Pose (ArUco PnP)', result)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                fname = f'robot_pose_{frame_count}.png'
                cv2.imwrite(fname, result)
                print(f"\nSaved: {fname}")
            elif key == ord('p') and robot_pose:
                print_pose_info(robot_pose)

            frame_count += 1

        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
