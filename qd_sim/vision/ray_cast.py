"""Pixel -> world-position back-projection for a known Z-height plane.

Unlike a solvePnP path (which recovers a full 6-DOF pose from several known
3D-to-2D point correspondences - the right tool for a printed marker's
precisely-known square), the shape detectors in shape_detector.py only give a
single 2D pixel centroid per detected feature, with no correspondence structure
to solve a full pose from. That's fine here: the resting/insertion surface
heights in this scene are fixed and already known (QDs rest on a known floor
height; the manifold's top face height is fixed) - vision only needs to refine
X/Y, not re-derive a Z the scene doesn't actually leave uncertain. This
ray-casts a detected pixel back through the camera to the one known world-Z
plane it must lie on.
"""

import numpy as np


def pixel_to_world_on_plane(px, py, K, cam_world_pos, cam_world_mat_mj, target_z):
    """(px, py): detected pixel (OpenCV convention - origin top-left, +y
    down). K: 3x3 intrinsics (qd_sim/vision/camera_intrinsics.py).
    cam_world_pos/cam_world_mat_mj: the camera's world pose (data.cam_xpos,
    data.cam_xmat.reshape(3,3) - MuJoCo's own OpenGL-style camera
    convention: local -Z forward, +X right, +Y up). target_z: the known
    world Z-height the real feature lies on.

    Returns the 3D world point where the ray through that pixel crosses
    target_z.
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Pixel -> normalized camera-local direction. Forward is -Z (MuJoCo
    # convention), and pixel-Y increasing downward is camera-local -Y (up),
    # hence the sign flip.
    x_ndc = (px - cx) / fx
    y_ndc = -(py - cy) / fy
    dir_cam = np.array([x_ndc, y_ndc, -1.0])

    R_cam_world = np.asarray(cam_world_mat_mj).reshape(3, 3)
    dir_world = R_cam_world @ dir_cam
    cam_pos = np.asarray(cam_world_pos, dtype=float)

    if abs(dir_world[2]) < 1e-9:
        raise ValueError("camera ray is parallel to the target Z-plane - can't intersect")
    t = (target_z - cam_pos[2]) / dir_world[2]
    if t <= 0:
        raise ValueError(f"target Z-plane ({target_z}) is behind the camera along this ray (t={t:.4f})")
    return cam_pos + t * dir_world


def world_to_pixel(world_point, K, cam_world_pos, cam_world_mat_mj):
    """The exact inverse of pixel_to_world_on_plane() above - given a real
    3D world point (not a Z-plane intersection; any point at all), returns
    the (px, py) pixel it projects to, or None if it's behind the camera.
    Used for qd_sim/tasks/leak_test_sequence.py's own --live dashboard
    overlay: the locked ring and the orbit's own planned path
    are computed and stored in WORLD coordinates (the same frame the arm's
    own IK targets live in), but qd_sim/vision/annotate.py's annotate()
    draws shapes in PIXEL coordinates - this is what turns one into the
    other, reusing the exact same pinhole model (same K, same MuJoCo
    camera-pose convention) pixel_to_world_on_plane() already does, just
    run forward instead of back-projected.

    Derivation (mirrors pixel_to_world_on_plane()'s own comments): a world
    point P, expressed in the camera's own local frame, is
    p_cam = R_cam_world^T @ (P - cam_pos) (R's transpose = its inverse,
    since it's a pure rotation). MuJoCo's camera convention has local -Z
    as forward, so a point actually in front of the camera has a NEGATIVE
    local Z - the normalized direction toward it is p_cam / (-p_cam.z),
    whose first two components are exactly the (x_ndc, y_ndc) the forward
    function itself defines pixels in terms of - inverting those two
    lines directly gives the pixel."""
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    R_cam_world = np.asarray(cam_world_mat_mj).reshape(3, 3)
    p_cam = R_cam_world.T @ (np.asarray(world_point, dtype=float) - np.asarray(cam_world_pos, dtype=float))
    if p_cam[2] >= 0:
        return None  # behind the camera - nothing sensible to draw
    x_ndc = p_cam[0] / -p_cam[2]
    y_ndc = p_cam[1] / -p_cam[2]
    px = x_ndc * fx + cx
    py = cy - y_ndc * fy
    return (px, py)
