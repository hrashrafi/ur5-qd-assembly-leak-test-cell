"""
Transport sequence: carry a grasped QD from wherever the port-tracking
descent left it to a pre-insertion pose above the target port, ready for the
screw controller to take over.
States: TO_PORT_ABOVE -> DESCEND_TO_PRE_INSERT -> DONE.

The QD is carried by a kinematic weld (qd_sim/tasks/kinematic_weld.py), not
friction/contact - the arm just needs to move, the weld keeps the QD's pose
exactly right relative to the gripper regardless of where that is.

pinch_to_thread_face_offset_m must be measured by the caller (pinch_z minus
thread_face_z, right after the grasp - constant from then on as long as
orientation doesn't change, true through transport and insertion both) and
passed in - this class doesn't assume any particular relationship between
the pinch site and the QD. Assuming the pinch is concentric with the flange
and hardcoding the offset would silently break whenever the grasp point
moves (it is set by qd_cell/constants.py's GRASP_OFFSET), landing
pre_insert_point tens of mm off. Because the screw motion starts directly
from wherever transport leaves off, that error would show up as a screw
phase that starts only a few degrees before finishing, with transport
already landing at (or past) the target depth.
"""

import numpy as np

from qd_sim.control.state_machine import StateMachine
from qd_sim.robot.arm_controller import DEFAULT_CRUISE_SPEED_MPS as CRUISE_SPEED_MPS
from qd_sim.tasks.pick_sequence import GRIPPER_DOWN_QUAT

TRANSIT_CLEARANCE_M = 0.10   # height above the port to fly over at during transit
PRE_INSERT_CLEARANCE_M = 0.020  # height above full-seat pose to hand off to
                                  # screwing. Screwing covers this whole
                                  # descent smoothly and gradually
                                  # (screw_controller.py's per-tick advance,
                                  # not a single jump), so there's no
                                  # collision risk in a close hand-off
                                  # point; this still leaves the thread tip
                                  # (12mm below the tracked face) about 8mm
                                  # above the port surface at the start,
                                  # clear of touching.

# Loose tolerance for the TO_PORT_ABOVE->DESCEND_TO_PRE_INSERT handoff -
# the same idea as pick_sequence.py's APPROACH_LOOSE_POS_TOL_M, mirrored
# here for the manifold/insert side: requiring full tight convergence
# (is_converged()'s default 3mm/2deg) above the port before starting the
# descent into it would cause a visible stop-then-go.
# DESCEND_TO_PRE_INSERT's own convergence stays tight (pos_tol_m=0.002) -
# this only loosens the handoff INTO it.
TRANSIT_LOOSE_POS_TOL_M = 0.03
TRANSIT_LOOSE_ANGLE_TOL_DEG = 8.0


class TransportSequence:
    def __init__(self, arm_ctrl, port_target_world, pinch_to_thread_face_offset_m):
        self.arm = arm_ctrl
        port = np.asarray(port_target_world, dtype=float)
        # port_target_world[2] IS the full-seat thread_face_z target
        # already (see the session's screw stop condition) - no
        # separate "seated pose" reference point needed.
        pinch_offset = np.array([0, 0, pinch_to_thread_face_offset_m])
        self.transit_point = port + np.array([0, 0, TRANSIT_CLEARANCE_M]) + pinch_offset
        self.pre_insert_point = port + np.array([0, 0, PRE_INSERT_CLEARANCE_M]) + pinch_offset
        self._dt = None  # set every tick() call - see set_smooth_target_paced() call sites below

        self.fsm = StateMachine(
            handlers={
                "TO_PORT_ABOVE": self._h_to_port_above,
                "DESCEND_TO_PRE_INSERT": self._h_descend,
            },
            start_state="TO_PORT_ABOVE",
        )

    def start(self, data, dt):
        # Go straight to the transit point (well above the port) first, so
        # the grasped QD doesn't get dragged sideways through anything at
        # grasp height on the way there. set_smooth_target_paced() - this
        # is the live-tracking-to-transport handoff, handled the same way
        # as the pick side (PickSequence.start()) - its start tangent
        # comes from the arm's actual _cur_velocity, so it continues
        # smoothly from wherever the session's track_descend() left it.
        self._dt = dt
        self.arm.set_smooth_target_paced(data, self.transit_point, GRIPPER_DOWN_QUAT, dt,
                                          cruise_speed_mps=CRUISE_SPEED_MPS)
        self.fsm.reset()

    def _h_to_port_above(self, data):
        # is_near_final(), not is_converged() - TO_PORT_ABOVE's own entry
        # (above) is a smooth trajectory, and is_converged() refuses
        # to fire while any trajectory is active (see its own docstring) -
        # same reasoning as pick_sequence.py's _h_approach.
        if self.arm.is_near_final(data, pos_tol_m=TRANSIT_LOOSE_POS_TOL_M,
                                   angle_tol_deg=TRANSIT_LOOSE_ANGLE_TOL_DEG):
            self.arm.set_smooth_target_paced(data, self.pre_insert_point, GRIPPER_DOWN_QUAT,
                                              self._dt, cruise_speed_mps=CRUISE_SPEED_MPS)
            return "DESCEND_TO_PRE_INSERT"
        return "TO_PORT_ABOVE"

    def _h_descend(self, data):
        if self.arm.is_converged(data, pos_tol_m=0.002):
            return "DONE"
        return "DESCEND_TO_PRE_INSERT"

    def tick(self, data, dt):
        self._dt = dt
        self.arm.step(data, dt)
        self.fsm.tick(data)
        return self.fsm.state

    @property
    def state(self):
        return self.fsm.state

    def is_done(self):
        return self.fsm.is_done()
