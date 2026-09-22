"""
Position-tracking controller for the 6-DOF arm: holds a target Cartesian
pose for the gripper's pinch site, and each tick converts the current pose
error into a joint velocity (via IK) that's integrated into a joint-position
command sent to the arm's actuators.

The UR5e's actuators are position servos (ctrl = desired joint angle, see
scene/ur5e_robotiq.xml's <general biastype="affine">), so "control" here means maintaining
our own running qpos target and writing it to ctrl every tick - not sending
raw torques or velocities directly.
"""

import numpy as np

from qd_sim.robot.ik_solver import DampedLeastSquaresIK
from qd_sim.robot.kinematics import ARM_JOINT_NAMES, arm_qpos_indices, site_pose
from qd_sim import transforms as tf


def _smootherstep(t):
    """Ken Perlin's quintic ease: zero velocity AND zero acceleration at
    both t=0 and t=1 (smoothstep's cubic 3t^2-2t^3 only zeroes velocity) -
    used by set_smooth_target()/set_smooth_path() so a planned trajectory
    starts and ends completely at rest, with no jerk at the handoff to
    whatever comes next."""
    return t * t * t * (t * (t * 6 - 15) + 10)


# Default cruise pace for set_smooth_target_paced() - centralized here as
# the ONE default every caller (the session, PickSequence,
# TransportSequence) shares, rather than each picking its own. Sized
# together with max_joint_speed/max_joint_accel below, so the planned moves
# and the reactive phases run at a consistent pace rather than one of them
# lagging the other.
DEFAULT_CRUISE_SPEED_MPS = 0.30
DEFAULT_CRUISE_ANGULAR_SPEED_DEG_S = 260.0
DEFAULT_MIN_MOVE_STEPS = 60


def _catmull_rom(p0, p1, p2, p3, t):
    """Position at parameter t in [0,1] between control points p1 and p2,
    given the neighbors p0/p3 on either side - the standard uniform
    Catmull-Rom formula. Continuous in both POSITION and TANGENT
    (direction + magnitude of dP/dt) at every control point, since each
    segment's boundary tangent is defined from its actual neighbors
    (0.5*(p2-p0) at the p1 end, 0.5*(p3-p1) at the p2 end) rather than
    each segment picking its own independent straight-line direction -
    that's what makes the path CURVE through a waypoint instead of
    cornering at it."""
    t2 = t * t
    t3 = t2 * t
    return 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                  + (-p0 + 3 * p1 - 3 * p2 + p3) * t3)


class ArmController:
    def __init__(self, model, site_name="gripper/pinch", max_joint_accel=14.0, joint_names=None):
        self.model = model
        self.site_id = model.site(site_name).id
        # joint_names: see kinematics.arm_dof_indices()'s own docstring -
        # defaults to the pick arm's bare ARM_JOINT_NAMES. The leak-test arm,
        # whose joints and actuators all carry a "leak/" prefix, passes its
        # own list, e.g. [f"leak/{n}" for n in ARM_JOINT_NAMES], to get an
        # ArmController wired to ITS joints/actuators instead.
        names = joint_names if joint_names is not None else ARM_JOINT_NAMES
        self.qpos_idx = arm_qpos_indices(model, joint_names)
        self.actuator_idx = [model.actuator(n.replace("_joint", "")).id for n in names]
        self.ik = DampedLeastSquaresIK(model, site_name, joint_names=joint_names)
        # rad/s^2, caps how fast commanded joint velocity itself can change
        # per tick - the IK's own max_joint_speed clamps speed but not how
        # quickly that speed is reached, so without this a fresh
        # set_target_pose() (a step change in target, e.g. between task
        # phases) would make dq jump straight to max_joint_speed on the very
        # first tick: a velocity step, i.e. infinite instantaneous
        # acceleration. Rate-limiting gives a visibly smooth ramped accel/
        # decel rather than a snap-to-speed motion. The value doesn't need
        # to be gentle for grip reasons: the QD is a kinematically-welded
        # mocap body (qd_sim/tasks/kinematic_weld.py) that arm motion can't
        # shake loose, so whole-session pacing is the constraint that sets
        # it, not grip safety.
        self.max_joint_accel = max_joint_accel

        self.target_pos = None
        self.target_quat = None
        self._target_qpos = None  # our running joint-position command
        self._prev_dq = None
        self._traj = None  # active smooth Cartesian trajectory, if any - see set_smooth_path()
        self._last_pos = None       # site position as of the previous step() call
        self._cur_velocity = np.zeros(3)  # finite-difference estimate of ACTUAL Cartesian
                                            # velocity, updated every step() - used by
                                            # set_smooth_path() as the new trajectory's start
                                            # tangent, so a fresh plan continues in whatever
                                            # direction the arm is ALREADY moving instead of
                                            # assuming it starts from rest.

    def reset(self, data):
        """Snap the internal target to the arm's current actual joint
        position - call this once after any qpos reset (e.g. a keyframe
        load) so the first control tick doesn't jump."""
        self._target_qpos = data.qpos[self.qpos_idx].copy()
        cur_pos, cur_quat = site_pose(data, self.site_id)
        self.target_pos, self.target_quat = cur_pos, cur_quat
        self._prev_dq = np.zeros(6)
        self._traj = None
        self._last_pos = cur_pos
        self._cur_velocity = np.zeros(3)

    def set_target_pose(self, pos, quat):
        """Set a single fixed target - the arm chases it REACTIVELY every
        tick via velocity IK (step()'s own accel/speed limiting shapes how
        fast, but the PATH taken is whatever the IK's per-tick solution
        happens to produce, not a planned curve). Cancels any in-progress
        smooth trajectory (set_smooth_target()) - this is an immediate,
        unplanned retarget."""
        self.target_pos = np.asarray(pos, dtype=float)
        self.target_quat = np.asarray(quat, dtype=float)
        self._traj = None

    def set_smooth_target(self, data, pos, quat, n_steps):
        """Plan a smooth Cartesian trajectory from the arm's CURRENT pose
        straight to (pos, quat) - the single-destination case of
        set_smooth_path() below; see that docstring for the full
        reasoning (time profile, why it fixes per-axis wobble, etc.)."""
        self.set_smooth_path(data, [(pos, quat)], n_steps)

    def set_smooth_target_paced(self, data, pos, quat, dt, cruise_speed_mps=DEFAULT_CRUISE_SPEED_MPS,
                                 cruise_angular_speed_deg_s=DEFAULT_CRUISE_ANGULAR_SPEED_DEG_S,
                                 min_steps=DEFAULT_MIN_MOVE_STEPS):
        """set_smooth_target(), but picking n_steps automatically from the
        distance/angle to cover and a cruise pace, instead of the caller
        computing it. Centralizes the ONE pacing formula every smooth-
        target caller in this codebase uses - one set of default cruise
        constants, in one place, so every caller moves at a consistent,
        comparable pace rather than silently drifting apart."""
        pos, quat = np.asarray(pos, dtype=float), np.asarray(quat, dtype=float)
        cur_pos, cur_quat = site_pose(data, self.site_id)
        dist_m, angle_deg = tf.pose_error(cur_pos, cur_quat, pos, quat)
        duration_s = max(dist_m / cruise_speed_mps, angle_deg / cruise_angular_speed_deg_s)
        n_steps = max(min_steps, int(duration_s / dt))
        self.set_smooth_target(data, pos, quat, n_steps)

    def set_smooth_path(self, data, waypoints, n_steps):
        """Plan a smooth Cartesian trajectory from the arm's CURRENT pose
        through an ORDERED sequence of via-points to a final destination
        (waypoints = [(pos, quat), ...], last entry is the destination),
        and start tracking it - each step() call advances one sample
        along this pre-planned path and uses THAT as target_pos/
        target_quat, instead of jumping straight to a fixed point and
        leaving velocity-IK to reactively chase it from far away.

        ONE continuous "smootherstep" time profile (6t^5-15t^4+10t^3,
        zero velocity AND zero acceleration at both t=0 and t=1) governs
        progress along the WHOLE path - not one profile per leg - so
        intermediate via-points (e.g. fly_safe()'s safety rise-before-
        translate-before-descend) are passed through at cruising speed,
        no stop in between, while still visiting them in the required
        order (unlike a single straight-line move, which would cut
        diagonally through whatever the via-points exist to avoid). Only
        the FINAL destination gets the profile's zero-velocity ending
        (callers wanting to hand off earlier, mid-cruise, use
        is_near_final() instead of is_converged() - see that method).

        Position follows a Catmull-Rom SPLINE through all the waypoints,
        not a piecewise-straight-line path: a straight-line path is smooth
        in SPEED (the timing profile) but still CORNERS in DIRECTION at
        every via-point (like the corner of an L), since each leg picks
        its own independent straight direction. Catmull-Rom instead
        derives each via-point's tangent from its actual neighbors, so the
        path curves through every via-point with continuous direction AND
        magnitude, not just continuous speed - see _catmull_rom()'s own
        docstring. The spline's START tangent is derived from
        _cur_velocity (the arm's actual measured Cartesian velocity as of
        the last step() call), not from a duplicated/zero phantom point -
        so a NEW trajectory continues in whatever direction the arm is
        already physically moving (e.g. handed off mid-cruise via
        is_near_final()) instead of implying a fresh direction change at
        the handoff. Arc length for progress-allocation purposes is still
        measured along the straight-line control polygon (not the true
        curve length) - a standard, cheap approximation; the curve bulges
        slightly off that polygon between via-points, which very slightly
        perturbs pacing but never speed continuity. Orientation still
        SLERPs linearly per leg (qd_sim/transforms.py's slerp()) -
        orientation changes in this codebase are rare/small enough that
        this doesn't need the same treatment.

        This also removes per-axis wobble a plain set_target_pose() move
        can show: since the REFERENCE itself follows an explicit path and
        never backtracks, the IK's own multi-DOF (position+orientation)
        resolution choosing a non-monotonic joint-space route can't appear
        in the reference (the arm's own following error against it is
        still whatever step() naturally produces, but has nothing
        non-monotonic to inherit from the target itself).

        n_steps is in TICKS (not seconds) - the caller picks it from
        whatever cruise speed / dt convention it's using (see
        set_smooth_target_paced()); is_converged() will not report done
        while a trajectory is still active (see its own docstring),
        regardless of momentary position-matching early in the ramp."""
        # A RE-PLAN (a path is already active) starts from the current
        # REFERENCE point, not the arm's measured pose: the reference
        # legitimately leads the arm by however much the controller is
        # lagging, and restarting at the arm's actual position would throw
        # that lead away every re-plan - which caps the achievable speed,
        # since the arm can only move as fast as its target leads it. A
        # first plan (no active path) starts from the measured pose, which
        # is the only meaningful reference there.
        if self._traj is not None:
            cur_pos, cur_quat = np.asarray(self.target_pos, dtype=float), np.asarray(self.target_quat, dtype=float)
        else:
            cur_pos, cur_quat = site_pose(data, self.site_id)
        positions = [cur_pos] + [np.asarray(p, dtype=float) for p, _ in waypoints]
        quats = [cur_quat] + [np.asarray(q, dtype=float) for _, q in waypoints]
        seg_lengths = [float(np.linalg.norm(positions[i + 1] - positions[i]))
                       for i in range(len(positions) - 1)]
        total = sum(seg_lengths) or 1e-9  # all-zero-length edge case (pure reorientation)
        # Phantom point BEHIND the start, placed so the spline's initial
        # tangent (0.5*(positions[1]-phantom)) matches the arm's ACTUAL
        # current velocity direction, at the first leg's own natural
        # scale (its own arc length) rather than an arbitrary dt-based
        # distance - avoids either negligible influence (too close) or
        # overshoot/bulging (too far). Falls back to duplicating the
        # start point (a zero-influence phantom) when the arm is genuinely
        # at rest (speed ~0 - e.g. the very first move of the whole
        # session).
        speed = float(np.linalg.norm(self._cur_velocity))
        if speed > 1e-6:
            phantom_start = cur_pos - (self._cur_velocity / speed) * seg_lengths[0]
        else:
            phantom_start = cur_pos
        self._traj = {
            "positions": positions, "quats": quats, "seg_lengths": seg_lengths, "total": total,
            "phantom_start": phantom_start, "n": max(1, int(n_steps)), "i": 0,
        }

    def _sample_path(self, s):
        """Position/orientation at fractional progress s in [0,1] along
        the active trajectory's Catmull-Rom spline (see
        set_smooth_path())."""
        traj = self._traj
        traveled = s * traj["total"]
        acc = 0.0
        positions, quats, seg_lengths = traj["positions"], traj["quats"], traj["seg_lengths"]
        n_segs = len(seg_lengths)
        for i, seg_len in enumerate(seg_lengths):
            is_last = i == n_segs - 1
            if traveled <= acc + seg_len or is_last:
                frac = 0.0 if seg_len < 1e-9 else np.clip((traveled - acc) / seg_len, 0.0, 1.0)
                p0 = positions[i - 1] if i > 0 else traj["phantom_start"]
                p1, p2 = positions[i], positions[i + 1]
                p3 = positions[i + 2] if i + 2 < len(positions) else positions[-1]
                pos = _catmull_rom(p0, p1, p2, p3, frac)
                quat = tf.slerp(quats[i], quats[i + 1], frac)
                return pos, quat
            acc += seg_len
        return positions[-1], quats[-1]

    def hold_here(self, data):
        """Freeze the target at the arm's current actual pose AND clear any
        residual commanded joint velocity (_prev_dq) - unlike
        set_target_pose() alone, which leaves _prev_dq untouched so a target
        change ramps in smoothly from whatever velocity was already
        commanded (see max_joint_accel's docstring above - deliberate, for
        genuine moves to a new distant target).

        That's wrong specifically when the new target IS the current pose,
        i.e. "stop exactly here": zero pose error means dq_desired is ~0,
        but the accel-limited chase toward it (step()'s dq = prev_dq +
        clip(...)) still takes several ticks to bring a nonzero prev_dq down
        to 0, so _target_qpos keeps coasting in the old direction for those
        ticks regardless of the target already being reached - overshooting
        past the frozen target, then correcting back. This shows as a
        visible dip-then-recover right as screwing starts if it's not
        cleared: transport's final descend phase can still have residual
        downward joint velocity at the instant its position-only
        convergence check passes, which is why ScrewController.start()
        calls this rather than set_target_pose()."""
        cur_pos, cur_quat = site_pose(data, self.site_id)
        self.target_pos, self.target_quat = cur_pos, cur_quat
        self._prev_dq = np.zeros(6)
        self._traj = None
        self._cur_velocity = np.zeros(3)

    def step(self, data, dt):
        """Advance the joint-position command by one control tick and write
        it to data.ctrl. Must be called after reset()."""
        if self._target_qpos is None:
            raise RuntimeError("ArmController.reset(data) must be called before step()")

        # Finite-difference estimate of ACTUAL Cartesian velocity, updated
        # every tick - see set_smooth_path()'s use of this as a new
        # trajectory's start tangent. Read at the TOP of step() (i.e.
        # reflects the result of the previous tick's env.step(), which
        # runs between this call and the last) - a one-tick lag at 500Hz
        # (2ms) is immaterial for this purpose.
        cur_pos, _ = site_pose(data, self.site_id)
        if self._last_pos is not None:
            self._cur_velocity = (cur_pos - self._last_pos) / dt
        self._last_pos = cur_pos

        if self._traj is not None:
            traj = self._traj
            s = _smootherstep(min(traj["i"] / traj["n"], 1.0))
            self.target_pos, self.target_quat = self._sample_path(s)
            traj["i"] += 1
            if traj["i"] > traj["n"]:
                self._traj = None  # trajectory consumed - target_pos/quat now sit exactly at the destination

        dq_desired = self.ik.compute_joint_velocities(data, self.target_pos, self.target_quat)
        max_delta = self.max_joint_accel * dt
        dq = self._prev_dq + np.clip(dq_desired - self._prev_dq, -max_delta, max_delta)
        self._prev_dq = dq

        self._target_qpos = self._target_qpos + dq * dt
        data.ctrl[self.actuator_idx] = self._target_qpos

    def pose_error(self, data):
        """(position error in m, angle error in deg) between the site's
        current actual pose and the target - for convergence checks."""
        cur_pos, cur_quat = site_pose(data, self.site_id)
        return tf.pose_error(cur_pos, cur_quat, self.target_pos, self.target_quat)

    def is_converged(self, data, pos_tol_m=0.003, angle_tol_deg=2.0):
        """True once the arm has actually arrived at target_pos/target_quat
        (within tolerance). While a smooth trajectory (set_smooth_target())
        is still active, this ALWAYS returns False regardless of momentary
        position-matching - necessary because target_pos is, during a
        trajectory, a MOVING intermediate sample, not the final
        destination; early in the ramp it sits very close to the arm's
        starting pose (smootherstep's own derivative is ~0 near t=0), which
        would otherwise let this return a false "converged" on the very
        first tick, before the arm has gone anywhere. Because is_converged()
        also waits for the trajectory's own built-in decelerate-to-REST
        ending, a caller using it as a completion signal always gets a full
        stop at the destination - correct when that stop is actually wanted
        (see hold_here()'s docstring for the class of bug a real stop
        avoids), wrong when the caller wants to hand off to whatever comes
        next WHILE STILL MOVING (see is_near_final() below)."""
        if self._traj is not None:
            return False
        pos_err, angle_err = self.pose_error(data)
        return pos_err < pos_tol_m and angle_err < angle_tol_deg

    def final_pose_error(self, data):
        """Like pose_error(), but against the trajectory's ULTIMATE
        destination even while it's still in progress (not the current
        moving intermediate sample) - for callers that want to know "are
        we basically there" without waiting for the built-in decelerate-
        to-rest ending. Falls back to plain target_pos/target_quat when no
        trajectory is active (identical to pose_error() in that case)."""
        cur_pos, cur_quat = site_pose(data, self.site_id)
        if self._traj is not None:
            final_pos, final_quat = self._traj["positions"][-1], self._traj["quats"][-1]
        else:
            final_pos, final_quat = self.target_pos, self.target_quat
        return tf.pose_error(cur_pos, cur_quat, final_pos, final_quat)

    def is_near_final(self, data, pos_tol_m=0.03, angle_tol_deg=8.0):
        """True once the arm is CLOSE to its trajectory's ultimate
        destination, whether or not the trajectory has finished its own
        decelerate-to-rest ending yet - so a caller flying to a coarse
        hover position (or any other cruise-through handoff) can hand off
        to whatever tracks/moves next WHILE STILL MOVING, not wait for a
        full stop it doesn't actually need. Handing off here mid-
        trajectory is safe: switching to a new target
        (set_target_pose()/set_smooth_target(), even mid-ramp) never
        resets _prev_dq (the actual commanded joint velocity) - only
        hold_here() does that, deliberately, for the cases that DO need a
        real stop - so whatever velocity the arm already has carries
        smoothly into the next target instead of decaying to zero first."""
        pos_err, angle_err = self.final_pose_error(data)
        return pos_err < pos_tol_m and angle_err < angle_tol_deg
