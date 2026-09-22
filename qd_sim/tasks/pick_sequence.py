"""
Pick sequence: approach above a QD's grasp point (the hex flange), descend,
close the gripper, lift. States: APPROACH -> DESCEND -> GRASP -> LIFT -> DONE.

The flange has hex flats, and the gripper must align its closing axis with
a pair of opposite flats, not just point straight down at an arbitrary yaw:
two flat pads closing on a hex at a corner-catching yaw is a genuinely
unstable grip, and the QD tips ~30deg the moment lifting starts. That is a
geometry problem, not a friction one (contact friction is 1.0, MuJoCo's
default, plenty) - more grip-settle time, a slower lift or better
Z-centering don't fix it on their own.

GRIPPER_DOWN_QUAT bakes in both the "point straight down" 180deg-about-X
flip AND a 40deg yaw that lines the pads up with a pair of opposite flats.
The hex's 6 corners sit at 0/60/120/180/-60/-120deg in the QD's own frame;
composed through the QD's 180deg flip into the world frame, 40deg puts the
closing axis across two flats rather than two corners.

Result: 0.51deg tilt, at yaw=40deg and a grasp height near the flange's
lower edge (35.0mm in the QD's own frame, vs the flange's vertical center
at 38.9mm, which also helps).
"""

import numpy as np

from qd_sim.control.state_machine import StateMachine
from qd_sim.robot.arm_controller import DEFAULT_CRUISE_SPEED_MPS as CRUISE_SPEED_MPS
from qd_sim import transforms as tf


def _gripper_down_quat(yaw_deg):
    """Point the gripper straight down (180deg about X: local +Z, toward
    the fingers, maps to world -Z), then yaw about world Z by yaw_deg to
    align the closing axis with a pair of hex flats."""
    R_flip = np.diag([1.0, -1.0, -1.0])
    c, s = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
    R_yaw = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return tf.mat_to_quat(R_yaw @ R_flip)


GRASP_YAW_DEG = 40.0  # closing axis across a pair of the flange's hex flats
GRIPPER_DOWN_QUAT = _gripper_down_quat(GRASP_YAW_DEG)

APPROACH_CLEARANCE_M = 0.08  # height above the grasp point to approach from

# Loose tolerance for the APPROACH->DESCEND handoff. Requiring full tight
# convergence (is_converged()'s default 3mm/2deg) before starting DESCEND
# would make the arm fully decelerate to a near-stop at approach_point
# every time, a visible stop-then-go. DESCEND's own convergence (into the
# actual grasp) stays tight - this only loosens the handoff INTO it, so the
# two legs blend into one continuous motion instead of stopping in between.
APPROACH_LOOSE_POS_TOL_M = 0.03
APPROACH_LOOSE_ANGLE_TOL_DEG = 8.0

# Loose tolerance for LIFT's own completion check. The arm should never
# come to a zero-Cartesian-velocity moment except where something physical
# requires it: DESCEND->GRASP is the real "must be stationary" moment, and
# stays tight, but LIFT ending in a full stop before the caller flies off
# to the manifold would be an unnecessary stop. Same mechanism as
# APPROACH_LOOSE_*: is_done() firing while still moving lets the caller's
# next move (the session's fly_safe(), via ArmController.set_smooth_path())
# carry over whatever velocity the arm still has, since switching targets
# never resets it (see ArmController.is_near_final()'s own docstring) - only
# a real stop (hold_here()) would.
LIFT_LOOSE_POS_TOL_M = 0.03
LIFT_LOOSE_ANGLE_TOL_DEG = 8.0
GRASP_SETTLE_TICKS = 440     # ticks to hold the close command before lifting
GRASP_RAMP_TICKS = 400       # of those, ticks spent ramping ctrl open->closed
                              # (0.8s at the sim's 500Hz) rather than snapping
                              # to full close in one step - a single-step
                              # close commands the actuator's full force/
                              # speed instantly, which can jolt the QD out of
                              # alignment right as it's grasped (rather than
                              # a fixed offset error, this shows up as
                              # instability later, once the arm starts moving
                              # the QD around).
                              # The remaining 40 ticks are a short
                              # post-full-close settle beat. The ramp IS the
                              # grasp-stability-sensitive part per the
                              # docstring above; the settle after it only
                              # needs to be long enough to let the fingers
                              # come to rest.
                              # itself - the ramp IS the grasp-stability-
                              # sensitive part per the docstring above, this
                              # only shortens the idle hold after it finishes.

# The "pinch" site (145mm from the gripper base along its approach axis) is NOT
# where the pads actually make contact: the pad collision geoms (the
# pad_box1/pad_box2 classes in scene/ur5e_robotiq.xml) span 111.9-149.4mm from
# base, centered at ~130.7mm.
# Targeting pinch==grasp_point directly would put the flange 14.3mm too deep
# into the gripper (past the pads' real contact center), producing a poor,
# off-center grip. This offset corrects for it by moving the commanded pinch position back
# (further from the target) by the same 14.3mm, so the pad center - not the
# pinch site - lands on the flange.
PINCH_TO_PAD_CENTER_OFFSET_M = 0.0143


class PickSequence:
    def __init__(self, arm_ctrl, gripper_ctrl, grasp_point_world):
        self.arm = arm_ctrl
        self.gripper = gripper_ctrl
        self.grasp_point = np.asarray(grasp_point_world, dtype=float)
        # What we actually command the pinch site to (not grasp_point itself
        # - see PINCH_TO_PAD_CENTER_OFFSET_M) so the pads' real contact
        # center, not the pinch site, lands on the flange.
        self._pinch_target_at_grasp = self.grasp_point + np.array([0, 0, -PINCH_TO_PAD_CENTER_OFFSET_M])
        self.approach_point = self._pinch_target_at_grasp + np.array([0, 0, APPROACH_CLEARANCE_M])
        self._grasp_ticks = 0
        self._dt = None  # set every tick() call - see set_smooth_target_paced() call sites below

        self.fsm = StateMachine(
            handlers={
                "APPROACH": self._h_approach,
                "DESCEND": self._h_descend,
                "GRASP": self._h_grasp,
                "LIFT": self._h_lift,
            },
            start_state="APPROACH",
        )

    def start(self, data, dt):
        # NOT self.arm.reset(data): ArmController.reset() zeroes
        # _cur_velocity/_prev_dq, which would silently defeat the
        # set_smooth_target_paced() call right below - the velocity-matched
        # start tangent it depends on would already be wiped, on every
        # pick. Each caller does its own one-time arm.reset() instead - see
        # the session's setup.
        self._dt = dt
        self.gripper.open(data)
        # set_smooth_target_paced() - this is the live-tracking-to-grasp
        # handoff. A plain set_target_pose() here, with no continuity
        # treatment, would be the largest velocity-direction discontinuity
        # in the whole session (~15-23deg tick-to-tick). The spline's start tangent comes
        # from the arm's actual _cur_velocity (tracked continuously by
        # ArmController, not reset by this call), so it continues smoothly
        # from whatever direction the session's track_descend() left it
        # moving in.
        self.arm.set_smooth_target_paced(data, self.approach_point, GRIPPER_DOWN_QUAT, dt,
                                          cruise_speed_mps=CRUISE_SPEED_MPS)
        self.fsm.reset()

    def _h_approach(self, data):
        # is_near_final(), not is_converged(): APPROACH's own entry (see
        # start()) is a smooth trajectory, and is_converged() refuses
        # to fire at all while ANY trajectory is still active (by design -
        # see its own docstring) - using it here would mean waiting for
        # APPROACH's planned decelerate-to-rest to fully finish before
        # even checking the loose tolerance, defeating the "hand off while
        # still moving" this loose check exists for.
        if self.arm.is_near_final(data, pos_tol_m=APPROACH_LOOSE_POS_TOL_M,
                                   angle_tol_deg=APPROACH_LOOSE_ANGLE_TOL_DEG):
            self.arm.set_smooth_target_paced(data, self._pinch_target_at_grasp, GRIPPER_DOWN_QUAT,
                                              self._dt, cruise_speed_mps=CRUISE_SPEED_MPS)
            return "DESCEND"
        return "APPROACH"

    def _h_descend(self, data):
        if self.arm.is_converged(data, pos_tol_m=0.002):
            self._grasp_ticks = 0
            return "GRASP"
        return "DESCEND"

    def _h_grasp(self, data):
        self._grasp_ticks += 1
        # Ramp the close command open->closed over GRASP_RAMP_TICKS instead
        # of commanding full close in one step, then hold fully closed for
        # the rest of the settle window.
        ramp_frac = min(self._grasp_ticks / GRASP_RAMP_TICKS, 1.0)
        ctrl = self.gripper.OPEN_CTRL + ramp_frac * (self.gripper.CLOSED_CTRL - self.gripper.OPEN_CTRL)
        self.gripper.close(data, ctrl_value=ctrl)
        if self._grasp_ticks >= GRASP_SETTLE_TICKS:
            self.arm.set_smooth_target_paced(data, self.approach_point, GRIPPER_DOWN_QUAT,
                                              self._dt, cruise_speed_mps=CRUISE_SPEED_MPS)
            return "LIFT"
        return "GRASP"

    def _h_lift(self, data):
        # is_near_final(), not is_converged() - same reasoning as
        # _h_approach: LIFT's own entry (above) is a smooth
        # trajectory too.
        if self.arm.is_near_final(data, pos_tol_m=LIFT_LOOSE_POS_TOL_M,
                                   angle_tol_deg=LIFT_LOOSE_ANGLE_TOL_DEG):
            return "DONE"
        return "LIFT"

    def tick(self, data, dt):
        """Advance the arm/gripper controllers by one control tick and run
        one FSM transition check. Call every physics step."""
        self._dt = dt
        self.arm.step(data, dt)
        self.fsm.tick(data)
        return self.fsm.state

    @property
    def state(self):
        return self.fsm.state

    def is_done(self):
        return self.fsm.is_done()
