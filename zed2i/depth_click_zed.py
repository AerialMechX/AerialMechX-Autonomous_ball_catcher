import pyzed.sl as sl
import cv2 as cv
import numpy as np
import sys

RESOLUTION = sl.RESOLUTION.HD720
FPS        = 30
DEPTH_MODE = sl.DEPTH_MODE.ULTRA        # HIGH, ULTRA, or PERFORMANCE
UNIT       = sl.UNIT.METER              # METER, MILLIMETER, etc.
ROI_SIZE   = 5                          

last_click = None
show_text  = ""

def on_mouse(event, x, y, flags, param):
    global last_click
    if event == cv.EVENT_LBUTTONDOWN:
        last_click = (x, y)

def median_depth(depth_mat, x, y, roi):
    h, w = depth_mat.get_height(), depth_mat.get_width()
    r = roi // 2
    x0, x1 = max(0, x - r), min(w - 1, x + r)
    y0, y1 = max(0, y - r), min(h - 1, y + r)

    vals = []
    for yy in range(y0, y1 + 1):
        for xx in range(x0, x1 + 1):
            try:
                err, d = depth_mat.get_value(xx, yy)
            except Exception:
                continue
            if err == sl.ERROR_CODE.SUCCESS and np.isfinite(d):
                vals.append(d)
    if not vals:
        return np.nan
    return float(np.median(vals))

def find_intrinsics_object(container):
    for name in dir(container):
        if name.startswith("_"):
            continue
        try:
            attr = getattr(container, name)
        except Exception:
            continue
        if callable(attr):
            continue
        if hasattr(attr, "fx") and hasattr(attr, "cx"):
            return name, attr
    return None, None

def get_intrinsics_and_baseline(zed, info):
    tried = []
    cal = None
    path_used = None

    attempts = [
        ("info.calibration_parameters", lambda i: getattr(i, "calibration_parameters", None)),
        ("info.camera_configuration.calibration_parameters", lambda i: getattr(getattr(i, "camera_configuration", None), "calibration_parameters", None)),
        ("info.camera_configuration", lambda i: getattr(i, "camera_configuration", None)),
        ("info", lambda i: i),
    ]

    for pname, fn in attempts:
        try:
            candidate = fn(info)
        except Exception:
            candidate = None
        tried.append(pname)
        if candidate is None:
            continue

        for intr_name in ("left_cam", "left_camera", "left_cam_parameters", "left_camera_parameters", "left_cam_info", "left"):
            if hasattr(candidate, intr_name):
                intr = getattr(candidate, intr_name)
                if hasattr(intr, "fx") and hasattr(intr, "cx"):
                    cal = candidate
                    intr_obj = intr
                    path_used = f"{pname}.{intr_name}"
                    break
        if cal is not None:
            break

        nm, intr_obj = find_intrinsics_object(candidate)
        if intr_obj is not None:
            cal = candidate
            path_used = f"{pname}.{nm}"
            break

    if cal is None or intr_obj is None:
        msg = (
            "Could not find camera intrinsics in CameraInformation object.\n"
            "Tried paths: " + ", ".join(tried) + "\n"
            "CameraInformation dir():\n" + ", ".join([d for d in dir(info) if not d.startswith('_')][:200])
        )
        raise RuntimeError(msg)

    try:
        fx = float(getattr(intr_obj, "fx"))
        fy = float(getattr(intr_obj, "fy", fx))
        cx = float(getattr(intr_obj, "cx"))
        cy = float(getattr(intr_obj, "cy"))
    except Exception as e:
        raise RuntimeError(f"Found intrinsics object but couldn't read fx/fy/cx/cy: {e}")

    baseline = None
    baseline_attempts = []
    baseline_names = [
        "get_camera_baseline", "get_baseline", "baseline", "get_camera_baseline_meters", "getCameraBaseline"
    ]
    for bn in baseline_names:
        if hasattr(cal, bn):
            attr = getattr(cal, bn)
            try:
                if callable(attr):
                    baseline = float(attr())
                else:
                    baseline = float(attr)
                baseline_attempts.append(bn)
                break
            except Exception:
                baseline_attempts.append(bn + "(call-failed)")
                continue

    try:
        cfg = getattr(info, "camera_configuration", None)
        if cfg is not None and baseline is None:
            if hasattr(cfg, "get_camera_baseline"):
                baseline = float(cfg.get_camera_baseline())
                baseline_attempts.append("camera_configuration.get_camera_baseline")
    except Exception:
        pass

    if baseline is None:
        for name in dir(cal):
            if name.lower().startswith("baseline") or name.lower() == "b":
                try:
                    val = float(getattr(cal, name))
                    baseline = val
                    baseline_attempts.append(name)
                    break
                except Exception:
                    continue

    if baseline is None:
        raise RuntimeError("Could not find stereo baseline from CameraInformation/calibration. Attempts: " + ", ".join(baseline_attempts))

    return {
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "baseline_m": baseline,
        "path_used": path_used,
        "baseline_attempts": baseline_attempts,
    }

def main():
    global show_text, last_click

    zed = sl.Camera()
    init = sl.InitParameters()
    init.camera_resolution = RESOLUTION
    init.camera_fps = FPS
    init.depth_mode = DEPTH_MODE
    init.coordinate_units = UNIT
    init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP

    err = zed.open(init)
    if err != sl.ERROR_CODE.SUCCESS:
        print("Failed to open ZED:", err)
        return

    runtime = sl.RuntimeParameters()
    runtime.confidence_threshold = 50

    image_left = sl.Mat()
    depth_map  = sl.Mat()
    point_cloud = sl.Mat()

    info = zed.get_camera_information()
    try:
        calib = get_intrinsics_and_baseline(zed, info)
    except RuntimeError as e:
        print("ERROR extracting calibration info:\n", e)
        for name in dir(info):
            if name.startswith("_"):
                continue
            try:
                print(name, ":", getattr(info, name))
            except Exception:
                print(name, ": <unprintable>")
        zed.close()
        sys.exit(1)

    fx, fy, cx, cy = calib["fx"], calib["fy"], calib["cx"], calib["cy"]
    baseline = calib["baseline_m"]
    print(f"Using intrinsics path: {calib['path_used']}")
    print(f"Left cam fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
    print(f"Stereo baseline (m): {baseline:.6f}")

    try:
        cfg = info.camera_configuration
        print(f"Stream: {cfg.camera_fps} FPS @ {cfg.resolution.width}x{cfg.resolution.height}")
    except Exception:
        pass

    cv.namedWindow("ZED Left")
    cv.setMouseCallback("ZED Left", on_mouse)

    try:
        while True:
            if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                continue

            zed.retrieve_image(image_left, sl.VIEW.LEFT)
            zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH)
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)

            frame = image_left.get_data()[:, :, :3].copy()

            if last_click is not None:
                x, y = last_click
                try:
                    err_d, d = depth_map.get_value(x, y)
                except Exception:
                    err_d, d = None, None

                d_med = median_depth(depth_map, x, y, ROI_SIZE)

                try:
                    err_pc, xyzrgba = point_cloud.get_value(x, y)
                except Exception:
                    err_pc, xyzrgba = None, None

                if err_d == sl.ERROR_CODE.SUCCESS and err_pc == sl.ERROR_CODE.SUCCESS and xyzrgba is not None:
                    X, Y, Z = float(xyzrgba[0]), float(xyzrgba[1]), float(xyzrgba[2])
                    show_text = (
                        f"(x={x}, y={y})  Z={d:.3f} {UNIT.name.lower()}  "
                        f"median({ROI_SIZE}x{ROI_SIZE})={d_med:.3f}  "
                        f"3D=({X:.3f}, {Y:.3f}, {Z:.3f})"
                    )
                elif err_d == sl.ERROR_CODE.SUCCESS:
                    show_text = f"(x={x}, y={y})  Z={d:.3f} {UNIT.name.lower()}  median={d_med:.3f}"
                else:
                    show_text = "Invalid depth at this pixel (no data / occlusion)."
                cv.circle(frame, (x, y), 4, (0, 255, 255), -1)
                last_click = None

            if show_text:
                cv.rectangle(frame, (8, 8), (8 + 980, 40), (0, 0, 0), -1)
                cv.putText(frame, show_text, (12, 34), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv.putText(frame, "Click anywhere to query depth / 3D.  ESC to quit.",
                       (10, frame.shape[0]-12), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            cv.imshow("ZED Left", frame)

            if (cv.waitKey(1) & 0xFF) == 27:
                break
    finally:
        cv.destroyAllWindows()
        zed.close()

if __name__ == "__main__":
    main()
