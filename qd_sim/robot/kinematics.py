"""
Small kinematics helpers shared by the IK solver and controllers: reading a
site's current world pose, and computing a 6D Cartesian error (position +
orientation) between a current and target pose.
"""

import numpy as np

from qd_sim import transforms as tf

ARM_JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]


def arm_dof_indices(model, joint_names=None):
    """The 6 arm joints' dof (qvel) indices - for slicing a full Jacobian
    down to just the arm, excluding the gripper's own DOFs.

    joint_names defaults to the module-level ARM_JOINT_NAMES (the pick
    arm's own bare names). A second arm (e.g. the leak-test arm, namespaced
    "leak/" by mujoco.MjSpec.attach()) passes its own prefixed names
    instead, e.g. [f"leak/{n}" for n in ARM_JOINT_NAMES]."""
    names = joint_names if joint_names is not None else ARM_JOINT_NAMES
    return [model.joint(n).dofadr[0] for n in names]


def arm_qpos_indices(model, joint_names=None):
    names = joint_names if joint_names is not None else ARM_JOINT_NAMES
    return [model.joint(n).qposadr[0] for n in names]


def site_pose(data, site_id):
    """Current world (pos, quat_wxyz) of a site."""
    pos = data.site_xpos[site_id].copy()
    quat = tf.mat_to_quat(data.site_xmat[site_id].reshape(3, 3))
    return pos, quat


def pose_error_twist(pos_current, quat_current, pos_target, quat_target):
    """6D twist (3 linear + 3 angular, world frame) that would move the
    current pose toward the target pose - the standard input to a
    resolved-rate / damped-least-squares IK step.

    The angular part is the rotation vector (axis * angle) of the relative
    rotation R_current^T @ R_target, expressed back in world frame.
    """
    pos_err = np.asarray(pos_target) - np.asarray(pos_current)

    R_cur = tf.quat_to_mat(quat_current)
    R_tgt = tf.quat_to_mat(quat_target)
    R_err = R_cur.T @ R_tgt  # rotation from current to target, in current's local frame

    # matrix -> axis-angle (rotation vector), via the same trace identity
    # used in transforms.pose_error, but keeping the vector (not just angle)
    cos_theta = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if theta < 1e-8:
        axis_local = np.zeros(3)
    else:
        axis_local = np.array([
            R_err[2, 1] - R_err[1, 2],
            R_err[0, 2] - R_err[2, 0],
            R_err[1, 0] - R_err[0, 1],
        ]) / (2 * np.sin(theta))
    rotvec_local = axis_local * theta
    rotvec_world = R_cur @ rotvec_local  # rotate the local-frame error back into world frame

    return np.concatenate([pos_err, rotvec_world])
