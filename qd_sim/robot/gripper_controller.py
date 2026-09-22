"""
Thin wrapper around the Robotiq 2F-85's single actuator. ctrl range is 0-255
(0=open, 255=fully closed - see scene/ur5e_robotiq.xml's actuator), not a
joint angle directly, since the driver joint is tendon-coupled across both
fingers via a single "split" tendon.
"""

import numpy as np


class GripperController:
    OPEN_CTRL = 0.0
    CLOSED_CTRL = 255.0

    def __init__(self, model, actuator_name="gripper/fingers_actuator"):
        self.model = model
        self.actuator_id = model.actuator(actuator_name).id

    def open(self, data):
        data.ctrl[self.actuator_id] = self.OPEN_CTRL

    def close(self, data, ctrl_value=None):
        """ctrl_value lets a caller command any value in between - the pick's
        closing ramp and the session's opening ramp both do; defaults to
        fully closed."""
        data.ctrl[self.actuator_id] = self.CLOSED_CTRL if ctrl_value is None else ctrl_value

    def hold_here(self, data):
        """Freeze the actuator's target at wherever it is CURRENTLY,
        physically, holding - not the open/close ctrl value it was last
        commanded toward. close() always commands CLOSED_CTRL (255)
        regardless of contact (a soft position-servo grip: while an
        object resists, the real steady-state position sits well short of
        the ideal 255 target, held back by the contact force balancing
        the actuator's own pull). The instant that resisting contact
        disappears - e.g. a grasped object's own collision being turned
        off right as a kinematic weld takes over its pose (see
        kinematic_weld.py's engage() call site) - the same still-255
        ctrl has nothing left to resist it and keeps driving fully
        closed, visibly closing the fingers straight through the (now
        collision-free) object's mesh. This must be called that same
        tick to prevent that.

        Solves the actuator's own affine force equation (force =
        gainprm[0]*ctrl + biasprm[0] + biasprm[1]*length +
        biasprm[2]*velocity, MuJoCo's standard <general biastype="affine">
        form - see gripper/fingers_actuator in scene/ur5e_robotiq.xml)
        for the ctrl that makes force ~0 at the CURRENT actuator length,
        rather than hardcoding this actuator's specific gainprm/biasprm
        numbers - stays correct if they're ever retuned. Velocity's
        contribution is dropped (not just assumed zero): by the time
        this is called the ramp-then-settle hold in PickSequence's GRASP
        state has already run for hundreds of ticks, so qvel is already
        ~0."""
        gainprm = self.model.actuator_gainprm[self.actuator_id]
        biasprm = self.model.actuator_biasprm[self.actuator_id]
        length = data.actuator_length[self.actuator_id]
        ctrl = -(biasprm[0] + biasprm[1] * length) / gainprm[0]
        ctrl = float(np.clip(ctrl, self.OPEN_CTRL, self.CLOSED_CTRL))
        data.ctrl[self.actuator_id] = ctrl
        return ctrl
