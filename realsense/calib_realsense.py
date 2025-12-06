import os
import sys
import time
import glob
import yaml

import numpy as np
import cv2 as cv
import pyrealsense2 as rs


# -------------------------------------------------------------
# Global settings
# -------------------------------------------------------------
calibration_settings = {}


# -------------------------------------------------------------
# YAML parsing
# -------------------------------------------------------------
def parse_calibration_settings_file(filename):
    global calibration_settings

    if not os.path.exists(filename):
        print(f"[ERROR] Settings file does not exist: {filename}")
        sys.exit(1)

    print(f"[INFO] Using calibration settings: {filename}")
    with open(filename, "r") as f:
        calibration_settings = yaml.safe_load(f)

    for key in ["realsense_serial_0", "realsense_serial_1"]:
        if key not in calibration_settings:
            print(f"[ERROR] '{key}' key not found in YAML.")
            sys.exit(1)


# -------------------------------------------------------------
# RealSense helpers
# -------------------------------------------------------------
def open_realsense_camera(serial_number):
    """
    Open a RealSense camera color stream by serial number.
    """
    pipeline = rs.pipeline()
    config = rs.config()

    width = calibration_settings.get("frame_width", 1280)
    height = calibration_settings.get("frame_height", 720)
    fps = calibration_settings.get("fps", 30)

    config.enable_device(str(serial_number))
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

    print(f"[INFO] Opening RealSense {serial_number}...")
    try:
        profile = pipeline.start(config)
    except RuntimeError as e:
        print(f"[ERROR] Failed to open RealSense {serial_number}")
        print(f"        {e}")
        sys.exit(1)

    # Optional: try to disable some auto stuff for calibration
    try:
        device = profile.get_device()
        color_sensor = device.first_color_sensor()
        # Turn off auto-exposure priority
        if color_sensor.supports(rs.option.auto_exposure_priority):
            color_sensor.set_option(rs.option.auto_exposure_priority, 0)
    except Exception as e:
        print(f"[WARN] Could not adjust RealSense options: {e}")

    print(f"[INFO] Camera {serial_number} opened.")
    return pipeline


# -------------------------------------------------------------
# Capture frames for mono calibration
# -------------------------------------------------------------
def save_frames_single_camera_realsense(serial_number, cam_name="cam0"):
    """
    Capture frames from a specific RealSense camera for mono calibration.
    """
    out_dir = "mono_frames"
    os.makedirs(out_dir, exist_ok=True)

    n_frames = calibration_settings["mono_calibration_frames"]
    view_resize = calibration_settings["view_resize"]
    cooldown_target = calibration_settings["cooldown"]

    pipeline = open_realsense_camera(serial_number)
    print(f"[INFO] Ready to capture {n_frames} mono frames for {cam_name}")

    start = False
    cooldown = cooldown_target
    saved = 0

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())
            display = cv.resize(frame, None, fx=1.0/view_resize, fy=1.0/view_resize)

            if not start:
                cv.putText(display, "Press SPACE to start capture", (30, 40),
                           cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                cv.putText(display, f"Camera: {cam_name}", (30, 80),
                           cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            else:
                cooldown -= 1
                cv.putText(display, f"Cooldown: {cooldown}", (30, 40),
                           cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                cv.putText(display, f"Frames: {saved}/{n_frames}", (30, 80),
                           cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

                if cooldown <= 0:
                    fname = os.path.join(out_dir, f"mono_{cam_name}_{saved:03d}.png")
                    cv.imwrite(fname, frame)
                    print(f"[INFO] Saved {fname}")
                    saved += 1
                    cooldown = cooldown_target

            cv.imshow(f"Mono {cam_name}", display)
            key = cv.waitKey(1) & 0xFF

            if key == 27:  # ESC
                print("[INFO] ESC pressed. Aborting mono capture.")
                break
            if key == 32 and not start:  # SPACE
                print("[INFO] Starting mono capture.")
                start = True

            if saved >= n_frames:
                print(f"[INFO] Completed mono capture for {cam_name}")
                break

    finally:
        pipeline.stop()
        cv.destroyAllWindows()
        time.sleep(1.0)


# -------------------------------------------------------------
# Capture stereo pairs
# -------------------------------------------------------------
def save_frames_two_cams_realsense():
    """
    Capture synchronized stereo pairs from two RealSense RGB cameras.
    For calibration, we assume static board (so sequential grab is OK).
    """
    out_dir = "stereo_frames"
    os.makedirs(out_dir, exist_ok=True)

    view_resize = calibration_settings["view_resize"]
    cooldown_target = calibration_settings["cooldown"]
    n_pairs = calibration_settings["stereo_calibration_frames"]

    serial_0 = calibration_settings["realsense_serial_0"]
    serial_1 = calibration_settings["realsense_serial_1"]

    pipe0 = open_realsense_camera(serial_0)
    pipe1 = open_realsense_camera(serial_1)

    print(f"[INFO] Ready to capture {n_pairs} stereo pairs")

    start = False
    cooldown = cooldown_target
    saved = 0

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

            disp0 = cv.resize(frame0, None, fx=1.0/view_resize, fy=1.0/view_resize)
            disp1 = cv.resize(frame1, None, fx=1.0/view_resize, fy=1.0/view_resize)

            if not start:
                cv.putText(disp0, "Press SPACE to start stereo capture", (30, 40),
                           cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                cv.putText(disp0, "Make sure BOTH see the board", (30, 80),
                           cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            else:
                cooldown -= 1
                cv.putText(disp0, f"Cooldown: {cooldown}", (30, 40),
                           cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                cv.putText(disp0, f"Pairs: {saved}/{n_pairs}", (30, 80),
                           cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

                cv.putText(disp1, f"Cooldown: {cooldown}", (30, 40),
                           cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                cv.putText(disp1, f"Pairs: {saved}/{n_pairs}", (30, 80),
                           cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

                if cooldown <= 0:
                    fname0 = os.path.join(out_dir, f"stereo_cam0_{saved:03d}.png")
                    fname1 = os.path.join(out_dir, f"stereo_cam1_{saved:03d}.png")
                    cv.imwrite(fname0, frame0)
                    cv.imwrite(fname1, frame1)
                    print(f"[INFO] Saved stereo pair {saved}: {fname0}, {fname1}")
                    saved += 1
                    cooldown = cooldown_target

            cv.imshow("Stereo cam0", disp0)
            cv.imshow("Stereo cam1", disp1)
            key = cv.waitKey(1) & 0xFF

            if key == 27:  # ESC
                print("[INFO] ESC pressed. Aborting stereo capture.")
                break
            if key == 32 and not start:  # SPACE
                print("[INFO] Starting stereo capture.")
                start = True

            if saved >= n_pairs:
                print("[INFO] Completed stereo capture.")
                break

    finally:
        pipe0.stop()
        pipe1.stop()
        cv.destroyAllWindows()
        time.sleep(1.0)


# -------------------------------------------------------------
# Checkerboard object points
# -------------------------------------------------------------
def create_object_points():
    rows = calibration_settings["checkerboard_rows"]
    cols = calibration_settings["checkerboard_cols"]
    sq = calibration_settings["square_size_m"]

    objp = np.zeros((rows * cols, 3), np.float32)
    # patternSize in findChessboardCorners is (cols, rows)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= sq
    return objp


# -------------------------------------------------------------
# Mono calibration
# -------------------------------------------------------------
def calibrate_camera_for_intrinsic_parameters(images_pattern, cam_name="cam"):
    img_names = sorted(glob.glob(images_pattern))
    if len(img_names) == 0:
        print(f"[ERROR] No mono images found for pattern: {images_pattern}")
        return None, None, None

    print(f"[INFO] Found {len(img_names)} images for {cam_name} mono calibration")

    obj_template = create_object_points()
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

    objpoints = []
    imgpoints = []

    for fname in img_names:
        img = cv.imread(fname, cv.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] Could not read {fname}, skipping.")
            continue

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        rows = calibration_settings["checkerboard_rows"]
        cols = calibration_settings["checkerboard_cols"]

        ret, corners = cv.findChessboardCorners(gray, (cols, rows), None)
        if not ret:
            print(f"[WARN] Chessboard NOT found in {fname}, skipping.")
            continue

        corners_refined = cv.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), criteria
        )

        cv.drawChessboardCorners(img, (cols, rows), corners_refined, ret)
        cv.imshow(f"Mono corners {cam_name}", img)
        print(f"[INFO] Press 's' to skip this image, any other key to accept.")
        key = cv.waitKey(0) & 0xFF
        if key == ord('s'):
            print("[INFO] Skipping frame.")
            continue

        objpoints.append(obj_template)
        imgpoints.append(corners_refined)

    cv.destroyAllWindows()

    if len(objpoints) < 5:
        print("[ERROR] Too few valid mono frames, need at least 5.")
        return None, None, None

    image_shape = (gray.shape[1], gray.shape[0])
    print(f"[INFO] Running mono calibrateCamera for {cam_name}...")
    rms, K, dist, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, image_shape, None, None
    )

    print(f"[RESULT] {cam_name} mono RMS reprojection error = {rms:.4f} px")
    print(f"[RESULT] {cam_name} camera matrix:\n{K}")
    print(f"[RESULT] {cam_name} dist coeffs:\n{dist.ravel()}")

    return (K, dist, rms)


# -------------------------------------------------------------
# Stereo calibration
# -------------------------------------------------------------
def stereo_calibrate(K0, dist0, K1, dist1):
    """
    Stereo calibration with relaxed intrinsics.
    Uses mono intrinsics as a starting point but allows them to adjust.
    """
    pattern0 = os.path.join("stereo_frames", "stereo_cam0_*.png")
    pattern1 = os.path.join("stereo_frames", "stereo_cam1_*.png")

    names0 = sorted(glob.glob(pattern0))
    names1 = sorted(glob.glob(pattern1))

    if len(names0) == 0 or len(names1) == 0:
        print("[ERROR] No stereo frames found.")
        return None

    if len(names0) != len(names1):
        print("[ERROR] Number of left/right stereo images differ.")
        return None

    obj_template = create_object_points()
    rows = calibration_settings["checkerboard_rows"]
    cols = calibration_settings["checkerboard_cols"]
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

    objpoints = []
    imgpoints0 = []
    imgpoints1 = []

    for f0, f1 in zip(names0, names1):
        im0 = cv.imread(f0, cv.IMREAD_COLOR)
        im1 = cv.imread(f1, cv.IMREAD_COLOR)
        if im0 is None or im1 is None:
            print(f"[WARN] Could not read {f0} or {f1}, skipping.")
            continue

        gray0 = cv.cvtColor(im0, cv.COLOR_BGR2GRAY)
        gray1 = cv.cvtColor(im1, cv.COLOR_BGR2GRAY)

        ret0, corners0 = cv.findChessboardCorners(gray0, (cols, rows), None)
        ret1, corners1 = cv.findChessboardCorners(gray1, (cols, rows), None)

        if not ret0 or not ret1:
            print(f"[WARN] Chessboard not detected in pair {f0}, {f1}, skipping.")
            continue

        corners0 = cv.cornerSubPix(gray0, corners0, (11, 11), (-1, -1), criteria)
        corners1 = cv.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)

        # Optional visualization
        vis0 = im0.copy()
        vis1 = im1.copy()
        cv.drawChessboardCorners(vis0, (cols, rows), corners0, ret0)
        cv.drawChessboardCorners(vis1, (cols, rows), corners1, ret1)
        stacked = np.hstack([vis0, vis1])
        cv.imshow("Stereo corners (left | right)", stacked)
        print("[INFO] Press 's' to skip this pair, any other key to accept.")
        key = cv.waitKey(0) & 0xFF
        if key == ord('s'):
            print("[INFO] Skipping pair.")
            continue

        objpoints.append(obj_template)
        imgpoints0.append(corners0)
        imgpoints1.append(corners1)

    cv.destroyAllWindows()

    if len(objpoints) < 5:
        print("[ERROR] Too few valid stereo pairs, need at least 5.")
        return None

    image_shape = (gray0.shape[1], gray0.shape[0])
    print(f"[INFO] Running stereoCalibrate on {len(objpoints)} pairs...")

    flags = 0
    # Use mono intrinsics as initial guess, but allow refinement
    flags |= cv.CALIB_USE_INTRINSIC_GUESS
    # You can optionally fix some distortion terms if needed
    # flags |= cv.CALIB_FIX_K3

    rms, K0_refined, dist0_refined, K1_refined, dist1_refined, R, T, E, F = cv.stereoCalibrate(
        objpoints,
        imgpoints0,
        imgpoints1,
        K0, dist0,
        K1, dist1,
        image_shape,
        criteria=criteria,
        flags=flags
    )

    print(f"[RESULT] Stereo RMS reprojection error = {rms:.4f} px")
    print("[RESULT] Refined camera0 matrix:\n", K0_refined)
    print("[RESULT] Refined camera1 matrix:\n", K1_refined)
    print("[RESULT] Baseline translation vector T (cam1 in cam0 frame):\n", T.ravel())
    print("[RESULT] Rotation R:\n", R)

    result = {
        "K0": K0_refined,
        "dist0": dist0_refined,
        "K1": K1_refined,
        "dist1": dist1_refined,
        "R": R,
        "T": T,
        "E": E,
        "F": F,
        "rms": rms,
    }
    return result


# -------------------------------------------------------------
# Saving parameters
# -------------------------------------------------------------
def save_calibration_results(mono0, mono1, stereo_result, out_dir="../calib_output"):
    os.makedirs(out_dir, exist_ok=True)

    K0, dist0, rms0 = mono0
    K1, dist1, rms1 = mono1

    K0s = stereo_result["K0"]
    dist0s = stereo_result["dist0"]
    K1s = stereo_result["K1"]
    dist1s = stereo_result["dist1"]
    R = stereo_result["R"]
    T = stereo_result["T"]
    E = stereo_result["E"]
    F = stereo_result["F"]
    rms_stereo = stereo_result["rms"]

    # Save as npz
    npz_path = os.path.join(out_dir, "stereo_calib.npz")
    np.savez(
        npz_path,
        K0=K0s, dist0=dist0s, K1=K1s, dist1=dist1s,
        R=R, T=T, E=E, F=F,
        rms_mono0=rms0, rms_mono1=rms1, rms_stereo=rms_stereo
    )
    print(f"[INFO] Saved NPZ calibration to: {npz_path}")

    # Save YAML for human inspection
    yaml_path = os.path.join(out_dir, "stereo_calib.yaml")
    data = {
        "camera0": {
            "K": K0s.tolist(),
            "dist": dist0s.tolist(),
            "mono_rms": float(rms0),
        },
        "camera1": {
            "K": K1s.tolist(),
            "dist": dist1s.tolist(),
            "mono_rms": float(rms1),
        },
        "stereo": {
            "R": R.tolist(),
            "T": T.tolist(),
            "E": E.tolist(),
            "F": F.tolist(),
            "rms": float(rms_stereo),
        },
    }
    with open(yaml_path, "w") as f:
        yaml.dump(data, f)
    print(f"[INFO] Saved YAML calibration to: {yaml_path}")


# -------------------------------------------------------------
# Main
# -------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python calib_realsense.py calibration_settings_realsense.yaml")
        sys.exit(1)

    settings_file = sys.argv[1]
    parse_calibration_settings_file(settings_file)

    serial_0 = calibration_settings["realsense_serial_0"]
    serial_1 = calibration_settings["realsense_serial_1"]

    print("========================================")
    print(" DUAL REALSENSE STEREO CALIBRATION")
    print("========================================")
    print(f" Camera 0 serial: {serial_0}")
    print(f" Camera 1 serial: {serial_1}")
    print(" Make sure:")
    print("  - Baseline ~20–40 cm, cameras almost parallel")
    print("  - Rigid mount, no flex")
    print("  - Checkerboard flat & rigid")
    print("========================================\n")

    # Step 1: Mono frames
    print("\n[STEP 1] Capture mono frames for camera 0")
    save_frames_single_camera_realsense(serial_0, cam_name="cam0")

    print("\n[STEP 1] Capture mono frames for camera 1")
    save_frames_single_camera_realsense(serial_1, cam_name="cam1")

    # Step 2: Mono calibration
    print("\n[STEP 2] Mono calibration camera 0")
    mono0 = calibrate_camera_for_intrinsic_parameters("mono_frames/mono_cam0_*.png", cam_name="cam0")
    if mono0[0] is None:
        sys.exit(1)

    print("\n[STEP 2] Mono calibration camera 1")
    mono1 = calibrate_camera_for_intrinsic_parameters("mono_frames/mono_cam1_*.png", cam_name="cam1")
    if mono1[0] is None:
        sys.exit(1)

    K0, dist0, rms0 = mono0
    K1, dist1, rms1 = mono1

    # Step 3: Stereo pairs
    print("\n[STEP 3] Capture stereo pairs")
    save_frames_two_cams_realsense()

    # Step 4: Stereo calibration
    print("\n[STEP 4] Stereo calibration")
    stereo_result = stereo_calibrate(K0, dist0, K1, dist1)
    if stereo_result is None:
        sys.exit(1)

    # Step 5: Save results
    print("\n[STEP 5] Saving calibration results")
    save_calibration_results(mono0, mono1, stereo_result)

    print("\n========================================")
    print(" CALIBRATION COMPLETE")
    print("========================================")


if __name__ == "__main__":
    main()
