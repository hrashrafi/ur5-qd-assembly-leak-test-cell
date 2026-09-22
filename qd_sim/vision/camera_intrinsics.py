"""
Pinhole camera intrinsics derived from a MuJoCo camera's own fovy - no
separate calibration step needed since we're rendering, not photographing.
"""

import numpy as np


def intrinsics_from_mj_camera(model, cam_name, width, height):
    """Returns the 3x3 intrinsic matrix K for the given MuJoCo camera at the
    given render resolution.

    MuJoCo's `fovy` is the full vertical field of view in degrees; pixels
    are square (the renderer doesn't introduce anisotropic scaling), so
    fx == fy follows directly from fy and the image height. The principal
    point is assumed at the image center - true for an ideal pinhole render
    with no cropping/offset, which is what mujoco.Renderer produces.

    No distortion coefficients: the renderer has no lens distortion model,
    so every detected pixel is already distortion-free.
    """
    cam_id = model.camera(cam_name).id
    fovy_deg = model.cam_fovy[cam_id]
    fovy_rad = np.deg2rad(fovy_deg)

    fy = height / (2 * np.tan(fovy_rad / 2))
    fx = fy
    cx = width / 2
    cy = height / 2

    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1],
    ], dtype=np.float64)
    return K
