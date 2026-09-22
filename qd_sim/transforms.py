"""
SE(3) pose helpers shared across the vision and control code.

Convention: poses are (pos, quat) pairs, pos a length-3 array in meters and
quat a length-4 array in MuJoCo's (w, x, y, z) order - NOT OpenCV/ROS's
(x, y, z, w). Getting this backwards silently produces a subtly wrong
rotation (still unit-norm, still "a" rotation, just the wrong one), so it's
worth being explicit about everywhere a quat crosses a library boundary.
"""

import numpy as np


def quat_to_mat(quat_wxyz):
    """(w,x,y,z) quaternion -> 3x3 rotation matrix."""
    w, x, y, z = quat_wxyz
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        raise ValueError(f"Degenerate (near-zero) quaternion: {quat_wxyz}")
    s = 2.0 / n
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
        [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
    ])


def mat_to_quat(R):
    """3x3 rotation matrix -> (w,x,y,z) quaternion."""
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def pose_to_matrix(pos, quat_wxyz):
    """(pos, quat) -> 4x4 homogeneous transform."""
    T = np.eye(4)
    T[:3, :3] = quat_to_mat(quat_wxyz)
    T[:3, 3] = pos
    return T


def matrix_to_pose(T):
    """4x4 homogeneous transform -> (pos, quat_wxyz)."""
    return T[:3, 3].copy(), mat_to_quat(T[:3, :3])


def invert(T):
    """Inverse of a 4x4 rigid transform (exploits orthogonality of R, cheaper
    and more numerically stable than a general np.linalg.inv)."""
    R = T[:3, :3]
    p = T[:3, 3]
    Tinv = np.eye(4)
    Tinv[:3, :3] = R.T
    Tinv[:3, 3] = -R.T @ p
    return Tinv


def compose(T_a_b, T_b_c):
    """Chain two transforms: T_a_c such that a point in frame c maps to
    frame a. E.g. compose(world_T_site, site_T_qd) = world_T_qd."""
    return T_a_b @ T_b_c


def slerp(quat0_wxyz, quat1_wxyz, s):
    """Spherical linear interpolation between two (w,x,y,z) quaternions,
    s in [0,1] (0 -> quat0, 1 -> quat1) - for Cartesian trajectory
    planning (qd_sim/robot/arm_controller.py's set_smooth_target()), so
    orientation eases smoothly alongside position instead of snapping.
    Picks the shorter of the two paths on the unit hypersphere (flips
    sign if the dot product is negative - q and -q represent the same
    rotation, and interpolating toward the "wrong-signed" copy would
    take the long way around)."""
    q0 = np.asarray(quat0_wxyz, dtype=float)
    q1 = np.asarray(quat1_wxyz, dtype=float)
    dot = np.dot(q0, q1)
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = np.clip(dot, -1.0, 1.0)
    if dot > 0.9995:  # nearly identical - linear interp is numerically safer
        q = q0 + s * (q1 - q0)
        return q / np.linalg.norm(q)
    theta0 = np.arccos(dot)
    theta = theta0 * s
    q_perp = q1 - q0 * dot
    q_perp /= np.linalg.norm(q_perp)
    return q0 * np.cos(theta) + q_perp * np.sin(theta)


def pose_error(pos_a, quat_a, pos_b, quat_b):
    """Position error (m) and angle error (deg) between two poses - the
    arm controller's convergence checks are built on this."""
    pos_err_m = float(np.linalg.norm(np.asarray(pos_a) - np.asarray(pos_b)))
    R_a = quat_to_mat(quat_a)
    R_b = quat_to_mat(quat_b)
    R_diff = R_a.T @ R_b
    # rotation angle of R_diff via its trace: trace = 1 + 2*cos(theta)
    cos_theta = np.clip((np.trace(R_diff) - 1.0) / 2.0, -1.0, 1.0)
    angle_err_deg = float(np.degrees(np.arccos(cos_theta)))
    return pos_err_m, angle_err_deg
