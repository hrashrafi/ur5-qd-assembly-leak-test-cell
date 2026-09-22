"""
Damped-least-squares differential IK for the 6-DOF arm, driving a chosen
site (normally "gripper/pinch") toward a target Cartesian pose.

This is deliberately a velocity-level (resolved-rate) solver, not a
full-pose analytic/numeric IK solve to convergence in one shot: it's called
every control tick with the current pose error, which is the natural fit
for a controller that tracks a moving target - the screw motion, for
instance, moves the target along a helix and this solver simply follows.
"""

import mujoco
import numpy as np

from qd_sim.robot.kinematics import arm_dof_indices, pose_error_twist, site_pose


class DampedLeastSquaresIK:
    def __init__(self, model, site_name, damping=0.05, max_joint_speed=4.0, joint_names=None):
        self.model = model
        self.site_id = model.site(site_name).id
        # joint_names: see arm_dof_indices()'s own docstring - defaults to
        # the pick arm's bare ARM_JOINT_NAMES; a second arm passes its own
        # prefixed names.
        self.dof_idx = arm_dof_indices(model, joint_names)
        self.damping = damping
        # rad/s, clamps a single step. It doesn't need to be gentle for grip
        # reasons: the QD is a kinematically-welded mocap body, immune to
        # being shaken by arm motion at any speed (see
        # qd_sim/tasks/kinematic_weld.py). This is the one cap that also
        # governs the REACTIVE phases (track_descend(), the arm's own
        # tracking during screwing), not just planned Cartesian moves.
        self.max_joint_speed = max_joint_speed

        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

    def compute_joint_velocities(self, data, target_pos, target_quat):
        """Returns a 6-vector of arm joint velocities (rad/s) that drives
        the site toward (target_pos, target_quat)."""
        cur_pos, cur_quat = site_pose(data, self.site_id)
        twist = pose_error_twist(cur_pos, cur_quat, target_pos, target_quat)

        mujoco.mj_jacSite(self.model, data, self._jacp, self._jacr, self.site_id)
        J_full = np.vstack([self._jacp, self._jacr])  # (6, nv)
        J = J_full[:, self.dof_idx]  # (6, 6) - arm columns only

        # damped least squares: dq = J^T (J J^T + lambda^2 I)^-1 twist
        lam2 = self.damping ** 2
        JJt = J @ J.T
        dq = J.T @ np.linalg.solve(JJt + lam2 * np.eye(6), twist)

        speed = np.max(np.abs(dq))
        if speed > self.max_joint_speed:
            dq *= self.max_joint_speed / speed
        return dq
