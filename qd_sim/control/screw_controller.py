"""
Kinematic thread-advance screwing motion: rotates a dedicated rotary tool
adapter (the "screw_adapter" joint) - and, via the kinematic weld, the QD
mounted through it - while the arm descends at the exact helical
relationship v = pitch/(2*pi) * omega, for the QD's real engagement depth
(12.0mm). This is a kinematics-first design, not a force-first one: the
helical motion itself IS the control law, not something a force loop
discovers by feel. On a real cell, force sensing would sit on top of this
as a supervisory check (contact confirm, seating spike, jam abort), never as
what generates the motion.

Thread pitch comes from the connector's CAD/datasheet, not the mesh (the
mesh's thread region is a coarse cosmetic ridge pattern, not a helix): a
7/8"-14 UNF male straight thread (ISO 11926-3 / SAE J1926 ORB), so pitch is
25.4mm/14 = 1.8143mm/turn, ~6.61 turns over the 12.0mm depth. That is not
rounded to an integer - expected_turns is only ever used as a divisor below,
see __init__.

No collision/dimension interaction with the manifold bore, and no force/
torque-based fault detection (jam, seating-spike): the manifold's collision
is a deliberately wide-open slot, not a snug bore, and even if it
weren't, the QD is a kinematically-welded mocap body
(qd_sim/tasks/kinematic_weld.py) which by construction can't generate or
resist any physical force at all. "Seated" here just means "the real
measured depth was reached" (the caller checks actual position, not a
commanded step count), not a force-sensed event.

Why a dedicated joint instead of driving the arm's own wrist_3:
  1. wrist_3_joint has a hard +-2*pi limit (matches real UR5e hardware, not
     a modeling artifact) - nowhere near enough for several full turns from
     most starting angles.
  2. Independent of the limit: commanding rotation about the tool's own
     approach axis through the Cartesian differential IK while pointing
     straight down sits right at this arm's wrist singularity (wrist_1/
     wrist_3 axes co-axial there) - even a small roll correction through
     that route produces tens of degrees of error that never converges.
screw_adapter sidesteps both: it has no range limit at all (a dedicated
tool joint, not a robot arm joint with a real mechanical stop), and because
it's a separate DOF the arm's own 6 joints never need to change orientation
for screwing in the first place. The weld (qd_sim/tasks/kinematic_weld.py)
picks up this joint's rotation for free, the same way it picks up any
other arm motion - it just reads the pinch site's actual world pose each
tick, wherever that ends up.

The pinch site still sits downstream of screw_adapter in the kinematic
chain though, so its ACTUAL orientation keeps rotating as screw_adapter
turns even though the arm's own 6 joints aren't the ones doing that
turning. With a fixed target_quat, the arm's IK would see that rotation as
a growing error and fight it - wrist_3 turns ~341deg over a single screw
phase doing so. Instead, target_quat is computed analytically each tick to
follow the intended rotation (see _advance() below). Simply locking wrist_3
is not enough on its own: the other 5 joints still react to the
orientation error it can't correct, leaving more residual tilt on the QD.

Rotation speed is ramped (accel-limited), not commanded at full speed from
the first tick: screw_adapter is a real actuator turning a real (if small)
mass against damping/armature, so an instant jump to full angular velocity
produces a reaction-torque spike on the mount, visible as the arm
shuddering as screwing starts and again as it stops. Ramping both the start
(tick()) and the end (decelerate_tick(), called a few times once the
caller's real-depth stop condition fires) removes that at the source.
Descent speed is derived from the SAME ramped angular speed (not commanded
separately), so the helix's pitch relationship holds exactly even while
ramping, not just at full speed.
"""

import numpy as np

from qd_sim.robot.kinematics import site_pose
from qd_sim import transforms as tf


class ScrewController:
    def __init__(self, arm_ctrl, model, engagement_depth_m, expected_turns,
                 angular_speed_rad_s=2 * np.pi, angular_accel_rad_s2=4 * np.pi, reverse=False,
                 unwind_speed_rad_s=None, unwind_accel_rad_s2=None):
        self.arm = arm_ctrl
        self.pitch_m = engagement_depth_m / expected_turns
        self.omega_mag = abs(angular_speed_rad_s)
        self.angular_accel = angular_accel_rad_s2
        # Unwind (see unwind_tick() below) is a purely cosmetic derotation,
        # not constrained by real thread engagement the way tick()'s actual
        # screwing speed is - there's no physical reason it has to move at
        # the same pace. Both default to the screwing values, but a caller
        # can pass faster ones to speed up just this part. That is safe
        # because it only changes how fast screw_adapter's OWN dedicated
        # actuator moves, with no interaction with the arm's 6 joints at all
        # (unlike running unwind concurrently with flying elsewhere, which
        # disturbs the arm's tracking - see the session).
        #
        # unwind_accel_rad_s2 matters more than unwind_speed_rad_s in
        # practice: unwind's "remainder" (see start_unwind()'s docstring -
        # less than one full turn) is small enough that unwind never
        # reaches its speed ceiling at all. It is a pure accel-limited
        # triangular ramp up-then-down, so the ceiling is irrelevant and
        # only the acceleration shortens it.
        self.unwind_omega_mag = abs(unwind_speed_rad_s) if unwind_speed_rad_s is not None else self.omega_mag
        self.unwind_accel = abs(unwind_accel_rad_s2) if unwind_accel_rad_s2 is not None else self.angular_accel
        # +1/-1: which way the tool visually spins. Independent of descent
        # direction (always downward/inserting, regardless of spin sense -
        # a real screw can be left- or right-handed without changing which
        # way it's being driven in).
        self.sign = -1 if reverse else 1
        self.screw_qpos_adr = model.joint("screw_adapter_joint").qposadr[0]
        self.screw_actuator_id = model.actuator("screw_adapter").id
        self._current_omega_mag = 0.0

    def start(self, data):
        # hold_here(), not set_target_pose(): also clears the arm's
        # residual commanded joint velocity, not just the target position -
        # see its docstring for the dip-then-recover this prevents (transport
        # can still be descending, in the ArmController's own rate-limited
        # sense, at the instant it hands off here).
        self.arm.hold_here(data)
        _, quat = site_pose(data, self.arm.site_id)
        self._initial_target_quat = quat.copy()
        self._screw_target = data.qpos[self.screw_qpos_adr]
        self._start_angle = self._screw_target

    def begin_stop(self, data):
        """Call exactly once, right after tick()'s real-depth condition
        fires and before the decelerate_tick() loop - resyncs
        arm.target_pos to the arm's ACTUAL current position rather than
        leaving it at whatever the commanded Z target happened to be at
        that instant.

        Those two aren't the same: the arm's own Z-tracking runs a
        steady-state lag of ~1.2mm under sustained descent (the stock UR5e
        gains are tuned for slow repositioning, not continuous tracking),
        with the real position consistently behind the commanded target
        throughout cruise. decelerate_tick() doesn't command further
        descent (advance_z=False), but simply freezing target_pos at its
        current, already-1.2mm-ahead value would still leave the arm
        something to chase: over the deceleration ramp it would keep
        closing that gap, descending further and overshooting a depth that
        was already correct the moment tick() stopped (by nearly 1mm over a
        249-tick decel). Snapping target_pos to the real, current position
        here removes that stale gap at the source.
        """
        pos, _ = site_pose(data, self.arm.site_id)
        self.arm.target_pos = pos

    def _advance(self, data, dt, target_omega_mag, direction_sign, track_arm=True, advance_z=True, accel=None):
        max_delta = (accel if accel is not None else self.angular_accel) * dt
        self._current_omega_mag += np.clip(target_omega_mag - self._current_omega_mag, -max_delta, max_delta)
        self._screw_target += direction_sign * self._current_omega_mag * dt
        data.ctrl[self.screw_actuator_id] = self._screw_target
        if track_arm:
            if advance_z:
                dz = self.pitch_m / (2 * np.pi) * self._current_omega_mag * dt
                self.arm.target_pos = self.arm.target_pos + np.array([0.0, 0.0, -dz])
            # Keep the arm's own orientation target in sync with reality,
            # not fixed - see module docstring for why (the 341deg fight
            # this avoids). Computed analytically from the intended
            # (commanded) rotation, not read back from the actual site
            # pose - reading actual pose would let any small tracking sag
            # get accepted as the new "correct" target every tick, with
            # nothing left to pull it back (leaving ~2.6deg of QD tilt by
            # the end of a full screw). The analytical version
            # still gives the IK a real target to correct sag against,
            # while never fighting screw_adapter's own rotation, since it
            # already accounts for exactly that rotation.
            total_dtheta = -(self._screw_target - self._start_angle)
            c, s = np.cos(total_dtheta), np.sin(total_dtheta)
            Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            self.arm.target_quat = tf.mat_to_quat(Rz @ tf.quat_to_mat(self._initial_target_quat))
            self.arm.step(data, dt)

    def tick(self, data, dt):
        """Advance the helix by one tick (ramping up to full speed, not
        snapping to it) and drive the arm toward it - call every physics
        step until a real measured depth (e.g. `thread_face_z() <=
        target_z`) says to stop, then switch to decelerate_tick() rather
        than just stopping cold. This is feed-forward (open-loop position
        tracking, not closed-loop on depth itself), so it can run a little
        ahead of or behind its own commanded pace under the arm's speed/
        accel limits - checking the real result is what matters, not how
        many ticks were commanded to get there."""
        self._advance(data, dt, self.omega_mag, self.sign)

    def decelerate_tick(self, data, dt):
        """Ramp rotation down to a stop - call a few times right after the
        real-depth stop condition triggers, instead of freezing the target
        instantly (see module docstring for why that matters: a real
        reaction-torque jolt, not cosmetic). Does NOT keep advancing the
        commanded depth (advance_z=False) - by the time this is called,
        tick()'s real-depth check has already fired, meaning the target
        depth was already reached. Feeding the arm a deeper target through
        the whole ramp-down would keep telling it to descend for another
        couple hundred ticks, overshooting the seat depth by up to ~1mm -
        and by a varying amount, depending on how much else the IK is
        correcting at the same time.
        Orientation is still tracked and the arm is still stepped
        (track_arm=True) so the arm doesn't fight the tool's continuing
        (decelerating) spin - only the Z target freezes."""
        self._advance(data, dt, 0.0, self.sign, advance_z=False)

    def is_stopped(self, tol=0.01):
        return self._current_omega_mag < tol

    def start_unwind(self, data):
        """Prepare to spin the tool back - not all the way back to its
        angle at start() (i.e. not literally undoing every commanded turn),
        just to the NEAREST angle that looks the same (a whole number of
        turns short of it): the tool is rotationally symmetric and already
        released the QD by this point (see unwind_tick's docstring), so
        anything a multiple of a full turn away from the start angle is
        visually and mechanically identical - unwinding the full amount
        would be slower than necessary and, for a part done with many
        turns, could take longer than the screw itself."""
        self._current_omega_mag = 0.0
        current = data.qpos[self.screw_qpos_adr]
        total_forward = (current - self._start_angle) * self.sign  # >= 0, in "turns forward" units
        remainder = total_forward % (2 * np.pi)
        self._unwind_target_qpos = current - self.sign * remainder
        self._screw_target = current

    def unwind_tick(self, data, dt):
        """Rotate the tool joint back toward the nearest equivalent angle
        computed in start_unwind() - call after releasing the QD
        (weld.release()) and before flying elsewhere, until
        unwind_done(data). The QD is already released by then (its mocap
        pose stays put regardless), so this only affects the empty
        gripper - but without it, the pinch site's actual orientation keeps
        whatever roll screwing left behind, and the next pick's approach
        would need the arm's own IK to correct that roll itself on its own
        6 joints - hitting the exact same wrist singularity all over again,
        just one step removed from the screw motion itself (see module
        docstring). Doesn't touch the arm - position holds on its own (the
        position servos keep whatever ctrl they were last given).

        Speed ramps down automatically as it nears the target (a standard
        stopping-distance switch: cruise at full speed until the remaining
        distance drops to what the accel limit needs to brake to zero
        exactly there, then brake - not a separate decelerate call, since
        the exact target is known in advance here, unlike the forward
        screw's real-depth-driven stop) for the same reaction-jolt reason
        as tick()/decelerate_tick(). It is a binary switch rather than a
        "safe speed" recomputed from the remaining distance every tick,
        because that is numerically unstable for small distances: the
        target speed and the remaining distance shrink together each tick
        and the ramp never reaches a cruise phase (a 0.16 rad remainder
        would not finish within 16s)."""
        remaining = abs(self._unwind_target_qpos - data.qpos[self.screw_qpos_adr])
        stopping_distance = self._current_omega_mag ** 2 / (2 * self.unwind_accel) if self.unwind_accel > 0 else 0.0
        target_omega = 0.0 if remaining <= stopping_distance else self.unwind_omega_mag
        self._advance(data, dt, target_omega, -self.sign, track_arm=False, accel=self.unwind_accel)

    def unwind_done(self, data, tol_rad=0.01):
        return abs(data.qpos[self.screw_qpos_adr] - self._unwind_target_qpos) < tol_rad
