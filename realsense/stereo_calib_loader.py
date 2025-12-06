"""
Stereo Calibration Loader
=========================
Shared module for loading stereo calibration and transforming between camera frames.

Camera Serial Numbers:
- Camera 0 (origin): 341522302002
- Camera 1 (secondary): 213522253879

Calibration Convention (OpenCV standard):
- R, T transform points FROM camera0 TO camera1: P1 = R @ P0 + T
- To transform FROM camera1 TO camera0: P0 = R.T @ P1 - R.T @ T

Coordinate Frame (RealSense):
- +X: right
- +Y: down  
- +Z: forward (away from camera)
"""

import numpy as np
import yaml
import os
from dataclasses import dataclass
from typing import Optional, Tuple

# Camera serial numbers (fixed order)
CAMERA0_SERIAL = "341522302002"  # Primary (origin)
CAMERA1_SERIAL = "213522253879"  # Secondary


@dataclass
class StereoCalibration:
    """Stereo calibration data."""
    # Camera 0 (origin) intrinsics
    K0: np.ndarray          # 3x3 camera matrix
    dist0: np.ndarray       # Distortion coefficients
    
    # Camera 1 (secondary) intrinsics
    K1: np.ndarray
    dist1: np.ndarray
    
    # Stereo extrinsics (cam0 -> cam1)
    R: np.ndarray           # 3x3 rotation
    T: np.ndarray           # 3x1 translation
    
    # Optional
    E: Optional[np.ndarray] = None  # Essential matrix
    F: Optional[np.ndarray] = None  # Fundamental matrix
    rms: float = 0.0
    
    def cam1_to_cam0(self, point: np.ndarray) -> np.ndarray:
        """
        Transform 3D point from camera1 frame to camera0 (origin) frame.
        
        P0 = R.T @ P1 - R.T @ T = R.T @ (P1 - T)
        """
        point = np.asarray(point).reshape(3)
        return self.R.T @ point - self.R.T @ self.T.flatten()
    
    def cam0_to_cam1(self, point: np.ndarray) -> np.ndarray:
        """Transform 3D point from camera0 frame to camera1 frame."""
        point = np.asarray(point).reshape(3)
        return self.R @ point + self.T.flatten()
    
    def rotation_cam1_to_cam0(self, R_cam1: np.ndarray) -> np.ndarray:
        """Transform rotation matrix from camera1 frame to camera0 frame."""
        return self.R.T @ R_cam1
    
    @property
    def baseline(self) -> float:
        """Baseline distance between cameras in meters."""
        return float(np.linalg.norm(self.T))


def load_calibration(calib_dir: str = "../calib_output") -> StereoCalibration:
    """
    Load stereo calibration from directory.
    Tries YAML first, then NPZ.
    """
    yaml_path = os.path.join(calib_dir, "stereo_calib.yaml")
    npz_path = os.path.join(calib_dir, "stereo_calib.npz")
    
    if os.path.exists(yaml_path):
        return load_calibration_yaml(yaml_path)
    elif os.path.exists(npz_path):
        return load_calibration_npz(npz_path)
    else:
        raise FileNotFoundError(
            f"No calibration file found in {calib_dir}. "
            f"Expected stereo_calib.yaml or stereo_calib.npz"
        )


def load_calibration_yaml(filepath: str) -> StereoCalibration:
    """Load stereo calibration from YAML file."""
    print(f"Loading calibration from {filepath}")
    
    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)
    
    calib = StereoCalibration(
        K0=np.array(data['camera0']['K']),
        dist0=np.array(data['camera0']['dist']).flatten(),
        K1=np.array(data['camera1']['K']),
        dist1=np.array(data['camera1']['dist']).flatten(),
        R=np.array(data['stereo']['R']),
        T=np.array(data['stereo']['T']).flatten(),
        E=np.array(data['stereo']['E']) if 'E' in data['stereo'] else None,
        F=np.array(data['stereo']['F']) if 'F' in data['stereo'] else None,
        rms=data['stereo'].get('rms', 0.0)
    )
    
    print(f"  Baseline: {calib.baseline*100:.1f} cm")
    print(f"  Stereo RMS: {calib.rms:.4f}")
    
    return calib


def load_calibration_npz(filepath: str) -> StereoCalibration:
    """Load stereo calibration from NPZ file."""
    print(f"Loading calibration from {filepath}")
    
    data = np.load(filepath)
    
    # Handle different key naming conventions
    K0 = data['K0'] if 'K0' in data else data['camera0_K']
    dist0 = data['dist0'] if 'dist0' in data else data['camera0_dist']
    K1 = data['K1'] if 'K1' in data else data['camera1_K']
    dist1 = data['dist1'] if 'dist1' in data else data['camera1_dist']
    
    calib = StereoCalibration(
        K0=K0,
        dist0=dist0.flatten(),
        K1=K1,
        dist1=dist1.flatten(),
        R=data['R'],
        T=data['T'].flatten(),
        E=data['E'] if 'E' in data else None,
        F=data['F'] if 'F' in data else None
    )
    
    print(f"  Baseline: {calib.baseline*100:.1f} cm")
    
    return calib


def triangulate_point(
    pt0: Tuple[float, float],  # (u, v) in camera 0
    pt1: Tuple[float, float],  # (u, v) in camera 1
    calib: StereoCalibration
) -> np.ndarray:
    """
    Triangulate 3D point from corresponding 2D points in both cameras.
    Returns point in camera0 (origin) frame.
    """
    import cv2
    
    # Projection matrices
    P0 = calib.K0 @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P1 = calib.K1 @ np.hstack([calib.R, calib.T.reshape(3, 1)])
    
    # Triangulate
    pts0 = np.array([[pt0[0]], [pt0[1]]], dtype=np.float64)
    pts1 = np.array([[pt1[0]], [pt1[1]]], dtype=np.float64)
    
    point_4d = cv2.triangulatePoints(P0, P1, pts0, pts1)
    point_3d = point_4d[:3] / point_4d[3]
    
    return point_3d.flatten()


if __name__ == "__main__":
    # Test loading
    try:
        calib = load_calibration("../calib_output")
        print(f"\nCalibration loaded successfully!")
        print(f"K0:\n{calib.K0}")
        print(f"K1:\n{calib.K1}")
        print(f"R:\n{calib.R}")
        print(f"T: {calib.T}")
        
        # Test transformation
        test_point = np.array([0.5, 0.3, 1.5])  # Point in cam1
        transformed = calib.cam1_to_cam0(test_point)
        print(f"\nTest: Point in cam1: {test_point}")
        print(f"      Point in cam0: {transformed}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
