"""One continuous assembly + leak-test session, and the twelve gates it
stops at.

`CellSession` owns everything a run needs - the physics, both arms, the two
renderers, the video writer, the vision display, and (with --plc) the Modbus
slave the supervisor polls. Its methods are grouped in the order a reader
needs them:

    setup        building the rig, and the state a run accumulates
    the tick     one physics step, recording, gates, hold and abort
    motion       planned moves and the safe-height detour between tasks
    vision       what the cameras see, and which candidate is next
    display      what the vision window draws on top of the camera feed
    the session  wait for start, install a QD, park, leak-test, finish

Between gates the PLC is an observer, not a controller: the motion and vision
loops never leave this process, exactly as a real robot controller keeps its
control loop off the fieldbus.
"""

import csv
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np

from qd_sim.control.screw_controller import ScrewController
from qd_sim.robot import arm_controller
from qd_sim.robot.arm_controller import ArmController
from qd_sim.robot.gripper_controller import GripperController
from qd_sim.robot.kinematics import ARM_JOINT_NAMES, site_pose
from qd_sim.env import SimEnv
from qd_sim.tasks.kinematic_weld import KinematicWeld
from qd_sim.tasks.leak_test_sequence import (LEAK_CAM_OPTICAL_OFFSET_XY_M, PROBE_DOWN_QUAT,
                                             LeakTestSequence, set_qd_collision_for_inspection,
                                             survey_installed_flanges)
from qd_sim.tasks.pick_sequence import GRIPPER_DOWN_QUAT, PickSequence
from qd_sim.tasks.transport_sequence import TransportSequence
from qd_sim import transforms as tf
from qd_sim.vision.annotate import DIM_COLOR, HEX_COLOR, HOLE_COLOR, OCCUPIED_COLOR
from qd_sim.vision.camera_intrinsics import intrinsics_from_mj_camera
from qd_sim.vision.ray_cast import pixel_to_world_on_plane, world_to_pixel
from qd_sim.vision.shape_detector import detect_hexagons, detect_holes
from qd_sim.live_dashboard import PANEL_HEADING, PANEL_TEXT, LiveDashboard

from qd_plc.gate import PhaseGate, SimAborted
from qd_plc.server import SlaveServer
from qd_plc.signals import SimSignals
from qd_plc.tags import MAP_A, Phase

from qd_cell.constants import (ENGAGEMENT_DEPTH_M, EXPECTED_TURNS, FPS, GRASP_OFFSET,
                               GRIPPER_OPEN_RAMP_TICKS, HEIGHT, LEAK_ARM_PREFIX,
                               MANIFOLD_HOVER_XY, MANIFOLD_HOVER_Z, PORTS_HARDCODED,
                               PORT_MATCH_RADIUS_M, PORT_TRACK_FLOOR_Z, PORT_Z,
                               QD_FLANGE_TOP_Z, QD_GRASP_Z, QD_MATCH_RADIUS_M,
                               QD_NEST_HARDCODED, QD_TRACK_FLOOR_Z, SAFE_Z,
                               SCREW_ANGULAR_ACCEL_RAD_S2, SCREW_ANGULAR_SPEED_RAD_S, SESSION,
                               STEPS_PER_FRAME, THIRD_PERSON_SCENE_OPTION,
                               THREAD_FACE_LOCAL_OFFSET, TRAY_HOVER_XY, TRAY_HOVER_Z,
                               UNWIND_ANGULAR_ACCEL_RAD_S2, UNWIND_ANGULAR_SPEED_RAD_S,
                               VISION_HEIGHT, VISION_WIDTH, WIDTH,
                               WRIST_CAM_OPTICAL_OFFSET_XY_M)

#: Keep publishing this long after the session finishes, so the supervisor can
#: observe COMPLETE instead of watching the slave device disappear. Generous:
#: the panel polls at 250ms and a person may take a while to come and look.
PLC_IDLE_HOLD_S = 300.0

#: Bodies that count as "the arm" when looking for a self-collision.
ARM_BODY_PREFIXES = ("shoulder", "upper_arm", "forearm", "wrist", "gripper", "base")

#: Outside a tracking step the dashboard falls back to a plain wrist_cam
#: render, which is work nothing else needs. Rendering it every other frame
#: halves that cost and is invisible: the wrist view barely changes tick to
#: tick when nothing is being tracked.
LIVE_PLAIN_WRIST_STRIDE = 2

#: Used by the tracking descent's exit handoff, which is a plain reactive
#: set_target_pose(). Whatever runs next sets its own fresh target the moment
#: tracking returns, so tracking has no reason to come to a stop first.
FLY_LOOSE_POS_TOL_M = 0.03
FLY_LOOSE_ANGLE_TOL_DEG = 8.0

#: "Did I already do this one?" Under half the 45mm port spacing, so it can
#: never confuse two neighbours.
DONE_MATCH_RADIUS_M = 0.020
#: Detected X values closer than this count as one column when ordering the
#: roster: well above detection noise (~0.1mm), well below either grid pitch.
ROSTER_COLUMN_TOL_M = 0.020

#: Cruise pace for the Cartesian trajectory planner, centralised in
#: ArmController.
CRUISE_SPEED_MPS = arm_controller.DEFAULT_CRUISE_SPEED_MPS
CRUISE_ANGULAR_SPEED_DEG_S = arm_controller.DEFAULT_CRUISE_ANGULAR_SPEED_DEG_S
MIN_MOVE_STEPS = arm_controller.DEFAULT_MIN_MOVE_STEPS

#: How high above the port row the leak arm surveys from.
LEAK_SURVEY_HOVER_M = 0.20


class LiveSessionAborted(Exception):
    """Raised when the operator presses 'q'/Esc in a live window. Caught by
    run() so the video writer and the windows are closed properly, rather
    than leaving a half-written file behind."""


def disable_qd_collision(model, qd_name):
    """Turn off collision for a placed QD's geoms - call once it's seated and
    released. A placed QD is done: it only needs to look right from here on,
    and leaving its collision on is a real problem, not a cosmetic one.
    Later QDs sit on ports only 45mm apart, close enough that the gripper
    installing a neighbour genuinely clips an already-placed one. The QD
    itself cannot be moved by that contact, being a kinematically-welded mocap
    body - but the GRIPPER very much can, because a mocap body still exerts a
    real reaction force on whatever touches it."""
    for suffix in ("col_far", "col_flange", "col_thread"):
        geom_id = model.geom(f"{qd_name}_{suffix}").id
        model.geom_contype[geom_id] = 0
        model.geom_conaffinity[geom_id] = 0


def tilt_deg(data, body_name):
    """How far a body's own -Z axis has tipped away from straight down."""
    R = data.body(body_name).xmat.reshape(3, 3)
    return np.degrees(np.arccos(np.clip((R @ np.array([0, 0, -1]))[2], -1, 1)))


def quat_close(q1, q2, tol_deg=1.0):
    _, angle_err = tf.pose_error(np.zeros(3), q1, np.zeros(3), q2)
    return angle_err < tol_deg


class CellSession:
    """One run of the cell, from the ready pose back to the ready pose."""

    # ---- setup ----------------------------------------------------------

    def __init__(self, args):
        self.args = args
        self.out_path = Path(args.out)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)

        self.env = SimEnv()
        self.arm = ArmController(self.env.model, "gripper/pinch")
        self.gripper = GripperController(self.env.model)
        self.weld = KinematicWeld(self.env.model)
        leak_joints = [f"{LEAK_ARM_PREFIX}{n}" for n in ARM_JOINT_NAMES]
        self.leak_arm = ArmController(self.env.model, "leak/probe_tip",
                                      joint_names=leak_joints)

        self.renderer = mujoco.Renderer(self.env.model, height=HEIGHT, width=WIDTH)
        self.vision_renderer = mujoco.Renderer(self.env.model, height=VISION_HEIGHT,
                                               width=VISION_WIDTH)
        self.writer = cv2.VideoWriter(str(self.out_path), cv2.VideoWriter_fourcc(*"avc1"),
                                      FPS, (WIDTH, HEIGHT))

        live_out_path = None
        if args.live:
            live_out_path = Path(args.live_out)
            live_out_path.parent.mkdir(parents=True, exist_ok=True)
        # This window IS the vision system's own display - wrist camera, live
        # detection overlays, coasting indicator - the screen a Cognex or
        # Keyence controller ships with, separate from the operator panel.
        want_dashboard = args.live or (args.plc and not args.no_vision_display)
        self.dash = (LiveDashboard(record_path=live_out_path, record_fps=FPS,
                                   exit_button=args.plc)
                     if want_dashboard else None)

        self.modbus = None
        self.gate = None
        self.sig = None
        if args.plc:
            self._start_modbus_slave()

        # --- what a run accumulates ---------------------------------------
        self.frame_count = 0
        self.phase = "init"
        self.tick_log = []      # (tick, phase, ax, ay, az, tx, ty, tz)
        self.vision_events = []  # (tick, phase, status, dist_mm, x, y)
        self.self_collision_peak = 0
        self.self_collision_pairs = set()
        self.leak_contact_peak = 0
        self.leak_contact_pairs = set()
        self.picked_qds_xy = []   # the pick arm's half of the roster
        self.filled_ports_xy = []  # ports already carrying a QD
        self.tested_ports_xy = []  # ports already leak-tested

        # --- display bookkeeping, read by nothing that affects the robot ---
        self.wrist_view = {"img": None, "shapes": [], "labels": [], "coast_text": None}
        self.status = {"qd": None, "port": None, "diff_mm": None, "tilt_deg": None,
                       "completed": 0, "leak_port": None}
        self._plain_wrist = None
        self.dashboard_cam = "wrist"   # or "leak", switched before that phase
        self._gate_frame_tick = 0
        self._last_seen = []           # most recent leak survey, for the overlay

        self.arm.reset(self.env.data)
        self.leak_arm.reset(self.env.data)
        self.ready_pos, self.ready_quat = site_pose(self.env.data, self.arm.site_id)
        self.ready_arm_qpos = self.env.data.qpos[self.arm.qpos_idx].copy()

    def _start_modbus_slave(self):
        """Publish Map A and open the gate handshake, so the PLC can see the
        robot and hold it at G0 until the operator presses START."""
        args = self.args
        self.modbus = SlaveServer(MAP_A, host=args.modbus_host, port=args.modbus_port)
        self.modbus.start()
        self.gate = PhaseGate(self.modbus.data, auto_permit=args.no_plc_gating)
        self.sig = SimSignals(self.modbus.data, self.gate)
        self.modbus.data.set("SIM_READY", True)
        # Means "the operator has not closed the vision display", not "a
        # vision display exists". Running headless is a configuration, not a
        # fault: publishing False would trip fault 17 the instant the cell
        # started. Cleared only by an actual quit.
        self.modbus.data.set_raw("LIVE_WINDOW_OPEN", True)
        print(f"  Modbus slave on {args.modbus_host}:{args.modbus_port}"
              + ("  (--no-plc-gating: every gate auto-releases)"
                 if args.no_plc_gating else ""))

    # ---- the tick: physics, recording, gates, hold and abort -------------

    def set_phase(self, label):
        """Name the step the robot is on. One hook for every phase boundary,
        so the panel tracks the robot instead of lagging to the next gate."""
        self.phase = label
        if self.sig:
            self.sig.publish_phase_label(label)

    def tick_once(self):
        """One physics step, as run_until() and the gate pump take it: step,
        carry the welded QD, count contacts, and publish the heartbeat."""
        self.env.step(1)
        self.weld.update(self.env.data)
        self.check_self_collision()
        self.check_leak_contacts()
        if self.sig:
            self.sig.tick()

    def untouched_tick(self, i):
        """One physics step for the three loops that step without run_until()
        - the post-screw dead hold, the gripper-open ramp and the ready-pose
        snap. Nothing commands the arm here; i counts the loop's own ticks, so
        a frame is recorded every STEPS_PER_FRAME of them. The heartbeat is
        published with each frame: an 800-tick silent stretch is ~1.6s, close
        enough to the PLC's 2s comms watchdog to matter."""
        self.env.step(1)
        self.weld.update(self.env.data)
        self.check_self_collision()
        self.check_leak_contacts()
        if i % STEPS_PER_FRAME == 0:
            self.record_frame()
            if self.sig:
                self.sig.tick()

    def record_frame(self):
        """Render the third-person view, write it to the video, and refresh
        the vision display."""
        self.renderer.update_scene(self.env.data, camera="third_person_cam",
                                   scene_option=THIRD_PERSON_SCENE_OPTION)
        img = self.renderer.render()
        self.writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        self.frame_count += 1
        if self.dash is None:
            return
        wrist_img = self.wrist_view["img"]
        if wrist_img is None:
            if self._plain_wrist is None or self.frame_count % LIVE_PLAIN_WRIST_STRIDE == 0:
                self._plain_wrist = (self.render_leak_cam() if self.dashboard_cam == "leak"
                                     else self.render_wrist_cam())
            wrist_img = self._plain_wrist
        still_running = self.dash.update(img, wrist_img, self.hud_lines(),
                                         shapes=self.wrist_view["shapes"],
                                         labels=self.wrist_view["labels"],
                                         coast_text=self.wrist_view["coast_text"])
        if not still_running:
            raise LiveSessionAborted("User pressed quit in the live window.")

    def check_self_collision(self):
        """Arm against itself, every physics step for the whole run - so the
        final number measures the session, not just the transitions this was
        tuned against."""
        n = 0
        for c in self.env.data.contact[: self.env.data.ncon]:
            b1 = self.env.model.body(self.env.model.geom_bodyid[c.geom1]).name
            b2 = self.env.model.body(self.env.model.geom_bodyid[c.geom2]).name
            if b1.startswith(ARM_BODY_PREFIXES) and b2.startswith(ARM_BODY_PREFIXES) and b1 != b2:
                n += 1
                self.self_collision_pairs.add(tuple(sorted((b1, b2))))
        if n > self.self_collision_peak:
            self.self_collision_peak = n
            if self.sig:
                self.sig.note_self_collision_peak(n)

    def check_leak_contacts(self):
        """Leak arm against any QD or the manifold. Also checked for the whole
        run, not just the leak phase, so a transit that clipped the parked
        leak arm would show up here too."""
        n = 0
        for c in self.env.data.contact[: self.env.data.ncon]:
            b1 = self.env.model.body(self.env.model.geom_bodyid[c.geom1]).name
            b2 = self.env.model.body(self.env.model.geom_bodyid[c.geom2]).name
            is_leak = b1.startswith(LEAK_ARM_PREFIX) or b2.startswith(LEAK_ARM_PREFIX)
            is_target = (b1.startswith("qd_") or b1 == "manifold"
                         or b2.startswith("qd_") or b2 == "manifold")
            if is_leak and is_target and b1 != b2:
                n += 1
                self.leak_contact_pairs.add(tuple(sorted((b1, b2))))
        if n > self.leak_contact_peak:
            self.leak_contact_peak = n
            if self.sig:
                self.sig.note_leak_contact_peak(n)

    def plc_interrupt(self):
        """Stop has to reach the robot BETWEEN gates, not only at them.

        A gate is where the supervisor decides something, and gates sit only
        where motion has already come to rest. A pause is not a decision,
        though, so it needs no resting point: nothing is stepped while held,
        so qpos and qvel are preserved untouched and resuming continues the
        same trajectory rather than restarting it. A freeze frame, not a stop
        - which is why this is safe mid-move where releasing a gate would not
        be.

        The heartbeat must keep going while held, or the supervisor's 2s comms
        watchdog faults the cell for doing exactly what it was told."""
        if self.gate is None:
            return
        if self.modbus.data.get("ABORT"):
            raise SimAborted("PLC commanded abort mid-phase")
        while self.modbus.data.get("HOLD"):
            if self.modbus.data.get("ABORT"):
                raise SimAborted("PLC commanded abort while held")
            self.sig.tick()
            time.sleep(0.005)

    def run_until(self, step_fn, is_done_fn, max_steps):
        """Step the sim until is_done_fn(), recording and publishing as it
        goes. Returns the step count reached."""
        for i in range(max_steps):
            self.plc_interrupt()
            step_fn()
            self.tick_once()
            cur_pos, _ = site_pose(self.env.data, self.arm.site_id)
            tgt = self.arm.target_pos
            self.tick_log.append((len(self.tick_log), self.phase,
                                  cur_pos[0], cur_pos[1], cur_pos[2],
                                  tgt[0], tgt[1], tgt[2]))
            if i % STEPS_PER_FRAME == 0:
                self.record_frame()
            if is_done_fn():
                return i
        if self.sig:
            # Without this a hung phase is indistinguishable from a finished
            # one - run_until() just returns either way.
            self.sig.note_maxsteps_expired()
        return max_steps

    def _gate_pump(self):
        """What keeps running while parked at a gate. Never calls arm.step():
        every gate sits where the arm has already come to rest, so the pump
        only has to keep physics, recording and comms alive."""
        self.tick_once()
        self._gate_frame_tick += 1
        if self._gate_frame_tick % STEPS_PER_FRAME == 0:
            self.record_frame()

    def plc_gate(self, phase, label=""):
        """Wait here for the supervisor's permit. A no-op without --plc.
        Raises SimAborted the same way quitting the vision display does."""
        if self.gate is not None:
            self.gate.gate(phase, self._gate_pump, label=label)

    def honour_exit_request(self):
        """EXIT asks for the cell to STOP, not to restart.

        The supervisor relaunches on any exit it has not been told about, so
        leave it the same note scripts/stop_cell.sh leaves - one mechanism,
        not two that can disagree."""
        if self.dash is not None and self.dash.exit_requested:
            Path("out/.robot_stop").touch()
            print("\n[EXIT] pressed - the cell will stay down "
                  "(run ./scripts/start_cell.sh to bring it back)")

    # ---- motion ----------------------------------------------------------

    def plan_steps(self, dist_m, angle_deg):
        duration_s = max(dist_m / CRUISE_SPEED_MPS, angle_deg / CRUISE_ANGULAR_SPEED_DEG_S)
        return max(MIN_MOVE_STEPS, int(duration_s / self.env.dt))

    def move_to(self, pos, quat, max_steps=8000, pos_tol_m=None, angle_tol_deg=None):
        """A single planned move to one pose, run to convergence."""
        self.arm.set_smooth_target_paced(self.env.data, pos, quat, self.env.dt,
                                         cruise_speed_mps=CRUISE_SPEED_MPS,
                                         cruise_angular_speed_deg_s=CRUISE_ANGULAR_SPEED_DEG_S,
                                         min_steps=MIN_MOVE_STEPS)
        tol = {}
        if pos_tol_m is not None:
            tol["pos_tol_m"] = pos_tol_m
        if angle_tol_deg is not None:
            tol["angle_tol_deg"] = angle_tol_deg
        return self.run_until(lambda: self.arm.step(self.env.data, self.env.dt),
                              lambda: self.arm.is_converged(self.env.data, **tol), max_steps)

    def fly_safe(self, target_xy, target_z, quat=GRIPPER_DOWN_QUAT, safe_z=SAFE_Z):
        """Reorient in place if needed, rise straight up to safe_z, translate
        across at safe_z, then descend - never a direct diagonal. The rise
        guarantees no horizontal motion happens until the arm has cleared
        every installed QD (~93mm at most, against SAFE_Z).

        Rise, translate and descend are one continuous trajectory: a single
        accel-cruise-decel profile governs the whole detour, so the arm passes
        through the via-points at cruising speed while still visiting them in
        the required order. Only the reorientation, which is rare, is a
        separate converging move.

        Completion is is_near_final(), not is_converged(): this returns while
        the arm is still cruising, and the caller immediately sets its next
        target, so the velocity carries through the handoff. The hover is a
        geometric waypoint, not a place to stop.

        When the destination is at or above safe_z the mid-height waypoint is
        skipped. Translating across first would leave a purely vertical climb
        into the hover that the tracking descent then immediately reverses - a
        direction reversal with no horizontal motion to carry through it, so
        it has to pass through zero speed (measured: ~240mm/s to ~6mm/s at the
        apex). The diagonal keeps moving and is just as safe, being entirely
        at or above safe_z."""
        cur_pos, cur_quat = site_pose(self.env.data, self.arm.site_id)
        if not quat_close(cur_quat, quat):
            self.move_to(cur_pos, quat)
        cur_pos, _ = site_pose(self.env.data, self.arm.site_id)
        waypoints = [([cur_pos[0], cur_pos[1], safe_z], quat)]
        if target_z < safe_z:
            # Below the safe height, so translate up at safe_z and descend
            # only afterwards - that ordering is the point of this function.
            waypoints.append(([target_xy[0], target_xy[1], safe_z], quat))
        waypoints.append(([target_xy[0], target_xy[1], target_z], quat))
        total_dist = sum(
            np.linalg.norm(np.array(waypoints[i][0])
                           - (cur_pos if i == 0 else np.array(waypoints[i - 1][0])))
            for i in range(len(waypoints)))
        self.arm.set_smooth_path(self.env.data, waypoints, self.plan_steps(total_dist, 0.0))
        self.run_until(lambda: self.arm.step(self.env.data, self.env.dt),
                       lambda: self.arm.is_near_final(self.env.data), 12000)

    def thread_face_z(self, qd_name):
        """World height of the QD's thread-side face - the number the seating
        check is made against."""
        qd_pos = self.env.data.body(qd_name).xpos.copy()
        qd_R = self.env.data.body(qd_name).xmat.reshape(3, 3).copy()
        return (qd_pos + qd_R @ THREAD_FACE_LOCAL_OFFSET)[2]

    # ---- vision ----------------------------------------------------------

    def render_wrist_cam(self):
        self.vision_renderer.update_scene(self.env.data, camera="wrist_cam")
        return cv2.cvtColor(self.vision_renderer.render(), cv2.COLOR_RGB2BGR)

    def render_leak_cam(self):
        self.vision_renderer.update_scene(self.env.data, camera="leak/leak_cam")
        return cv2.cvtColor(self.vision_renderer.render(), cv2.COLOR_RGB2BGR)

    def detect_qds(self):
        """Render wrist_cam from wherever the arm is now and return
        (world_xy_list, raw_img, shape_pairs, occupied): the world X/Y of
        every visible hex flange, ray-cast to the flange's own top height
        regardless of how high the camera is, plus the polygons behind them
        for the display overlay. occupied is always empty here - a QD in the
        tray has no such state - and exists only so this and detect_ports()
        share one return shape.

        The polygons come back even with no display: detect_hexagons()
        computes the approximation anyway for its own vertex-count check, so
        returning it costs nothing. Using it here saves a separate
        render-and-detect pass purely for the overlay on every tracking
        tick."""
        img = self.render_wrist_cam()
        cam_id = self.env.model.camera("wrist_cam").id
        K = intrinsics_from_mj_camera(self.env.model, "wrist_cam", VISION_WIDTH, VISION_HEIGHT)
        shape_pairs = []
        for (px, py), polygon in detect_hexagons(img):
            world_xy = pixel_to_world_on_plane(px, py, K, self.env.data.cam_xpos[cam_id],
                                               self.env.data.cam_xmat[cam_id],
                                               QD_FLANGE_TOP_Z)[:2]
            shape_pairs.append((world_xy, polygon))
        return [xy for xy, _ in shape_pairs], img, shape_pairs, []

    def detect_ports(self):
        """The same for manifold holes: the world X/Y of every currently EMPTY
        hole, ray-cast to the manifold's known top face.

        "Empty" is decided from filled_ports_xy - what this session has
        actually installed - not from the image. Reading it out of the image
        would mean absolute Canny thresholds, which only hold at the exact
        scene brightness they were tuned against: after a lighting change an
        image-based filter drops port tracking to 130 coasting ticks against
        22 tracked, and finds the tracked port at only 3/9 descent heights.
        Answering it from fact finds it at 9/9."""
        img = self.render_wrist_cam()
        cam_id = self.env.model.camera("wrist_cam").id
        K = intrinsics_from_mj_camera(self.env.model, "wrist_cam", VISION_WIDTH, VISION_HEIGHT)
        shape_pairs, occupied = [], []
        for px, py, r in detect_holes(img):
            world_xy = pixel_to_world_on_plane(px, py, K, self.env.data.cam_xpos[cam_id],
                                               self.env.data.cam_xmat[cam_id], PORT_Z)[:2]
            if any(np.linalg.norm(world_xy - np.asarray(f)) < DONE_MATCH_RADIUS_M
                   for f in self.filled_ports_xy):
                occupied.append((px, py, r))
            else:
                shape_pairs.append((world_xy, (px, py, r)))
        return [xy for xy, _ in shape_pairs], img, shape_pairs, occupied

    @staticmethod
    def choose_next(detected_xy, done_xy, radius=DONE_MATCH_RADIUS_M):
        """The edge-most not-yet-finished candidate, or None - so the session
        works along the row in a predictable direction rather than in
        detection order.

        Both arms schedule this way: survey once from a fixed pose that frames
        the whole group, then pick the next by excluding what is already done.
        The point is which side decides. Walking a hardcoded order means
        handing the tracker that object's hardcoded XY as its seed, so vision
        only ever refines a position it was already given. Here vision decides
        which objects exist and which is next.

        The Y tie-break is not cosmetic. The ports are a single row along X,
        but the tray is a 2x2 grid, so X alone leaves two candidates tied every
        round and min() resolves that arbitrarily. Plain lexicographic (X, then
        Y) does not fix it either: these X values are detected, not nominal, so
        a nominally tied column comes back ~0.1mm apart and Y is never
        consulted at all - the order ends up decided by sub-millimetre noise.
        So candidates within ROSTER_COLUMN_TOL_M of the leading X count as one
        column and are ordered by Y within it."""
        remaining = [np.asarray(p, dtype=float) for p in detected_xy
                     if all(np.linalg.norm(np.asarray(p) - np.asarray(d)) >= radius
                            for d in done_xy)]
        if not remaining:
            return None
        lead_x = min(p[0] for p in remaining)
        column = [p for p in remaining if p[0] - lead_x < ROSTER_COLUMN_TOL_M]
        return min(column, key=lambda p: p[1])

    def nearest_qd_body(self, xy):
        """Which qd_N body is at this world XY - SIMULATION BOOKKEEPING ONLY,
        not part of the perception story. The weld, the tilt readout and the
        collision toggles all need to name the specific body the gripper is
        about to hold, which a real cell would know from having grasped it
        rather than by looking it up. Kept explicit, and separate from
        anything vision-derived, so it cannot be mistaken for the detector
        knowing which part is which."""
        best, best_d = None, None
        for qd_name, _ in SESSION:
            qd_xy = self.env.data.body(qd_name).xpos[:2]
            d = np.linalg.norm(np.asarray(xy) - qd_xy)
            if best_d is None or d < best_d:
                best, best_d = qd_name, d
        return best

    def track_descend(self, hover_xy, hover_z, track_floor_z, seed_xy, detect_fn,
                      match_radius_m, label, max_steps=12000,
                      shape_kind=None, active_color=None):
        """Fly to the coarse hover, then drive smoothly and CONTINUOUSLY down
        to track_floor_z in one motion - the arm's own normal accel- and
        speed-limited move, not a series of discrete stop-and-recheck
        waypoints.

        Vision runs continuously alongside the motion: once per rendered frame
        (the same cadence as recording, and the rate a real camera feed would
        actually be processed) re-detect and refine the live X/Y target,
        feeding it straight into the arm's current Cartesian target. Z is
        commanded straight to track_floor_z from the start, so the arm's own
        accel/decel profile governs the descent and the camera still sees
        every real intermediate height along the way.

        Whichever detected candidate is nearest the CURRENT target is adopted,
        but only within match_radius_m - a farther detection is a different,
        still-visible QD or hole, not the one being tracked, and is ignored
        outright. When nothing is detected at all, below the detector's
        reliable range, the target is simply left unchanged: coasting on the
        last good fix rather than guessing or stalling. seed_xy anchors the
        first match, before any tracking history exists.

        Returns the final tracked X/Y, which is what the pick or transport
        sequence takes over from."""
        self.set_phase(f"fly_hover:{label}")
        self.fly_safe(hover_xy, hover_z)
        self.set_phase(f"track:{label}")
        current_xy = np.array(seed_xy, dtype=float)
        tick = 0

        def step_fn():
            nonlocal current_xy, tick
            if tick % STEPS_PER_FRAME == 0:
                candidates, raw_img, shape_pairs, occupied = detect_fn()
                if candidates:
                    best = min(candidates,
                               key=lambda p: np.linalg.norm(np.asarray(p) - current_xy))
                    dist = np.linalg.norm(np.asarray(best) - current_xy)
                    if dist < match_radius_m:
                        current_xy = np.asarray(best, dtype=float)
                        status = f"tracked (moved {dist*1000:.1f}mm)"
                        self.vision_events.append((len(self.tick_log), self.phase, "tracked",
                                                   dist * 1000, current_xy[0], current_xy[1]))
                        if self.sig:
                            self.sig.vision_tracked()
                    else:
                        status = (f"ignored a detection {dist*1000:.1f}mm away "
                                  "(not the tracked object)")
                        self.vision_events.append((len(self.tick_log), self.phase, "ignored",
                                                   dist * 1000, best[0], best[1]))
                else:
                    status = "no detection - coasting on last known position"
                    self.vision_events.append((len(self.tick_log), self.phase, "coasting",
                                               None, current_xy[0], current_xy[1]))
                    if self.sig:
                        self.sig.vision_coasting()
                cur_z, _ = site_pose(self.env.data, self.arm.site_id)
                print(f"[vision] {label} live-track z={cur_z[2]:.3f}: {status} "
                      f"-> ({current_xy[0]:.4f}, {current_xy[1]:.4f})")
                # Target the pinch at (current_xy - the mount offset), not at
                # current_xy, so the CAMERA stays centred over the target - the
                # same compensation the hover applies once up front. Applying
                # it only at the hover drifts ~2.6mm and tilts the grasp
                # 0.37deg instead of 0.1deg.
                cam_target = current_xy - np.array(WRIST_CAM_OPTICAL_OFFSET_XY_M)
                self.arm.set_target_pose(
                    np.array([cam_target[0], cam_target[1], track_floor_z]), GRIPPER_DOWN_QUAT)
                if self.dash is not None and shape_kind is not None:
                    self._draw_tracking(shape_pairs, occupied, raw_img, current_xy,
                                        match_radius_m, label, shape_kind, active_color)
            self.arm.step(self.env.data, self.env.dt)
            tick += 1

        self.run_until(step_fn,
                       lambda: self.arm.is_converged(self.env.data,
                                                     pos_tol_m=FLY_LOOSE_POS_TOL_M,
                                                     angle_tol_deg=FLY_LOOSE_ANGLE_TOL_DEG),
                       max_steps)
        return current_xy

    # ---- what the vision display draws -----------------------------------

    def clear_wrist_overlay(self):
        self.wrist_view.update(img=None, shapes=[], labels=[], coast_text=None)
        self._plain_wrist = None

    def _draw_tracking(self, shape_pairs, occupied, raw_img, current_xy,
                       match_radius_m, label, shape_kind, active_color):
        """Mark the candidate being tracked in its bright colour and every
        other detection dim, on the same frame the detection already used."""
        shapes, labels, found_active = [], [], False
        for world_xy, shape_repr in shape_pairs:
            d = np.linalg.norm(np.asarray(world_xy) - current_xy)
            if d < match_radius_m and not found_active:
                shapes.append((shape_kind, shape_repr, active_color))
                anchor = (np.mean(shape_repr, axis=0) if shape_kind == "polygon"
                          else shape_repr[:2])
                labels.append((anchor[0], anchor[1],
                               [f"TRACKING {label}",
                                f"x={world_xy[0]*1000:+.1f}mm y={world_xy[1]*1000:+.1f}mm"],
                               active_color))
                found_active = True
            else:
                shapes.append((shape_kind, shape_repr, DIM_COLOR))
        for px, py, r in occupied:
            shapes.append(("circle", (px, py, r), OCCUPIED_COLOR))
        self.wrist_view["img"] = raw_img
        self.wrist_view["shapes"] = shapes
        self.wrist_view["labels"] = labels
        self.wrist_view["coast_text"] = (
            None if found_active else f"{label}: not visible this step - coasting")

    def _leak_roster_shapes(self, K, cam_pos, cam_mat, seen_xy, tested_xy, active_xy=None):
        """Every flange the manifold survey saw, colour-coded by roster
        status: orange-red for already tested, green for the one under test,
        dim for still pending - the same colour vocabulary the install phase
        uses for filled vs. empty ports.

        Unlike the install phase this status is not visible in the image - a
        tested QD looks identical to a pending one - so it comes from the
        session's own done-set."""
        shapes, labels = [], []
        for xy in seen_xy:
            px = world_to_pixel(np.array([xy[0], xy[1], PORT_Z]), K, cam_pos, cam_mat)
            if px is None:
                continue
            done = any(np.linalg.norm(np.asarray(xy) - np.asarray(t)) < DONE_MATCH_RADIUS_M
                       for t in tested_xy)
            is_active = (active_xy is not None
                         and np.linalg.norm(np.asarray(xy) - np.asarray(active_xy))
                         < DONE_MATCH_RADIUS_M)
            if is_active:
                colour, text = HEX_COLOR, "TESTING"
            elif done:
                colour, text = OCCUPIED_COLOR, "tested"
            else:
                colour, text = DIM_COLOR, "pending"
            shapes.append(("circle", (px[0], px[1], 16), colour))
            labels.append((px[0], px[1], [text], colour))
        return shapes, labels

    def _draw_leak(self, leak_seq, port_num, seen_xy=(), tested_xy=(), active_xy=None):
        """The leak phase's two interesting moments, drawn on the leak camera:

          - HOVER_DETECT: the most recent successful detection, marked with a
            dot and its detected yaw (mod 60deg, a hex being 6-fold symmetric).
          - ORBIT_SEAM: the whole locked ring drawn dim as the planned path,
            with the edge actually being traced picked out bright.

        Both are computed in world coordinates by LeakTestSequence itself -
        the same frame its IK targets live in - and projected through the
        camera's current pose to get pixels. With nothing to show yet, the
        plain frame stands on its own."""
        img = self.render_leak_cam()
        cam_id = self.env.model.camera("leak/leak_cam").id
        cam_pos, cam_mat = self.env.data.cam_xpos[cam_id], self.env.data.cam_xmat[cam_id]
        K = intrinsics_from_mj_camera(self.env.model, "leak/leak_cam",
                                      VISION_WIDTH, VISION_HEIGHT)
        # Roster status underneath, so which ports are already done stays
        # visible for the whole phase, not only during the survey.
        shapes, labels = self._leak_roster_shapes(K, cam_pos, cam_mat, seen_xy,
                                                  tested_xy, active_xy)
        if leak_seq.state == "HOVER_DETECT" and leak_seq.last_detection is not None:
            center_xy, yaw = leak_seq.last_detection
            px = world_to_pixel(np.array([center_xy[0], center_xy[1], leak_seq.seam_z]),
                                K, cam_pos, cam_mat)
            if px is not None:
                shapes.append(("circle", (px[0], px[1], 6), HEX_COLOR))
                labels.append((px[0], px[1],
                               [f"port {port_num} - detected",
                                f"yaw={np.degrees(yaw) % 60:.1f}deg"], HEX_COLOR))
        elif leak_seq.state == "ORBIT_SEAM" and leak_seq.ring_points is not None:
            ring_px = [world_to_pixel(p, K, cam_pos, cam_mat) for p in leak_seq.ring_points]
            if all(p is not None for p in ring_px):
                shapes.append(("polygon", ring_px, DIM_COLOR))
                i = leak_seq.current_edge_index
                if i + 1 < len(ring_px):
                    shapes.append(("polygon", [ring_px[i], ring_px[i + 1]], HEX_COLOR))
                    labels.append((ring_px[i][0], ring_px[i][1],
                                   [f"port {port_num} - edge {i + 1}/{len(ring_px) - 1}"],
                                   HEX_COLOR))
        self.wrist_view.update(img=img, shapes=shapes, labels=labels, coast_text=None)

    def hud_lines(self):
        """The telemetry panel under the two camera views."""
        lines = [(f"Phase: {self.phase}", PANEL_HEADING)]
        st = self.status
        if st["qd"] is not None:
            lines.append((f"Current QD/port: {st['qd']} -> port {st['port']}", PANEL_TEXT))
        if st["diff_mm"] is not None:
            lines.append((f"Last tracked vs. reference: {st['diff_mm']:.2f}mm", PANEL_TEXT))
        if st["tilt_deg"] is not None:
            lines.append((f"Last grasp tilt: {st['tilt_deg']:.2f}deg", PANEL_TEXT))
        lines.append((f"QDs installed so far: {st['completed']}/{len(SESSION)}", PANEL_TEXT))
        lines.append((f"Self-collision contacts (peak): {self.self_collision_peak}", PANEL_TEXT))
        if st["leak_port"] is not None:
            lines.append((f"Leak-test port: {st['leak_port']}/{len(SESSION)}", PANEL_TEXT))
            lines.append((f"Leak-arm/QD/manifold contacts (peak): {self.leak_contact_peak}",
                          PANEL_TEXT))
        lines.append((f"Frame: {self.frame_count}  ({self.frame_count / FPS:.1f}s)", PANEL_TEXT))
        return lines

    # ---- the session -----------------------------------------------------

    def run(self):
        """The whole session. Returns once the robot is parked and the video
        is flushed, however it ended."""
        if not self.wait_for_start():
            return
        try:
            for round_num in range(len(SESSION)):
                self.install_qd(round_num)
            self.return_to_ready()
            self.plc_gate_if_plc(Phase.LEAK_PREP, "G8 LEAK_PHASE_PERMIT")
            self.leak_test_all()
            self.plc_gate_if_plc(Phase.SESSION_COMPLETE, "G11 SESSION_END")
            self.finish()
        except LiveSessionAborted as e:
            self.honour_exit_request()
            self._report_abort("Live session aborted", e)
            if self.sig:
                # The one case this bit actually reports: the operator quit
                # the display.
                self.sig.data.set_raw("LIVE_WINDOW_OPEN", False)
        except SimAborted as e:
            # The PLC commanded ABORT - see qd_plc/gate.py.
            self._report_abort("[PLC] session aborted", e)
        finally:
            self.close()

    def plc_gate_if_plc(self, phase, label):
        """A gate that only exists when the supervisor does."""
        if self.args.plc:
            self.plc_gate(phase, label=label)

    def wait_for_start(self):
        """Park and wait to be started. Returns False if the operator gave up
        before the session began.

        With --plc that is the panel's job: its START button writes HMI_START,
        OpenPLC checks its interlocks and permits G0, and this gate releases.
        Both abort paths are caught, not just SimAborted - the gate's pump
        calls record_frame(), which is what raises LiveSessionAborted when the
        vision display is quit."""
        if self.args.plc:
            self.set_phase("idle")
            try:
                self.plc_gate(Phase.WAIT_START, label="G0 WAIT_START")
            except (SimAborted, LiveSessionAborted) as exc:
                print(f"\n[PLC] {exc}")
                self.honour_exit_request()
                self.writer.release()
                if self.dash is not None:
                    self.dash.close()
                self.modbus.data.set_raw("ABORT_COMPLETE", True)
                self.modbus.stop()
                self.out_path.unlink(missing_ok=True)
                return False
        elif self.dash is not None and not self.args.auto_start:
            self.set_phase("idle")
            self.renderer.update_scene(self.env.data, camera="third_person_cam",
                                       scene_option=THIRD_PERSON_SCENE_OPTION)
            preview_third = self.renderer.render()
            preview_wrist = self.render_wrist_cam()
            subtitle = ["Queued: " + ", ".join(f"{q}->port{p}" for q, p in SESSION)]
            if not self.dash.wait_for_start(preview_third, preview_wrist, subtitle):
                self.writer.release()
                self.dash.close()
                # Nothing real was recorded - don't leave an empty .mp4.
                self.out_path.unlink(missing_ok=True)
                print("Live session cancelled before start - no video written.")
                return False
        return True

    def install_qd(self, round_num):
        """One full cycle: find a QD, pick it, find a port, screw it in, let
        go, retract."""
        if self.args.plc:
            # G1 - the arm is stationary here either way: just past G0 on the
            # first round, past G7 CYCLE_END on the later ones.
            self.plc_gate(Phase.SURVEY_TRAY, label="G1 CYCLE_START")
            self.sig.note_port_id(0)

        qd_name = self._pick_a_qd(round_num)
        port = self._choose_and_track_port(round_num, qd_name)
        screw = self._insert(qd_name, port)
        self._release_and_retract(qd_name, screw)

        self.status["completed"] += 1
        if self.sig:
            self.sig.note_cycle_complete(self.status["completed"])
        if self.args.plc:  # G7 - the counting and reporting gate
            self.plc_gate(Phase.CYCLE_END, label="G7 CYCLE_END")

    def _pick_a_qd(self, round_num):
        """Survey the tray, track the chosen QD down, and grasp it."""
        # Survey from the one fixed tray hover every round: see the QDs still
        # there, drop the ones already picked, take the edge-most of what is
        # left. The hover pose never changes between rounds.
        self.set_phase(f"survey:tray:{round_num + 1}")
        self.fly_safe(TRAY_HOVER_XY, TRAY_HOVER_Z)
        seen, _img, _pairs, _occ = self.detect_qds()
        if self.sig:
            self.sig.vision_part_detected(len(seen))
        target_xy = self.choose_next(seen, self.picked_qds_xy)
        if target_xy is None:
            # Nothing visible that the roster hasn't already done. Fall back
            # to the taught nest rather than stopping - and say so plainly,
            # since a silent fallback would read like a real detection.
            fallback = SESSION[round_num][0]
            print(f"[vision] tray survey {round_num + 1}: saw {len(seen)} QD(s), none "
                  f"un-picked - falling back to hardcoded {fallback}")
            target_xy = np.asarray(QD_NEST_HARDCODED[fallback][:2], dtype=float)
            if self.sig:
                self.sig.vision_fallback()
        qd_name = self.nearest_qd_body(target_xy)
        print(f"[vision] tray survey {round_num + 1}: saw {len(seen)} QD(s), "
              f"{len(self.picked_qds_xy)} already picked -> chose {qd_name} at "
              f"({target_xy[0]:.4f}, {target_xy[1]:.4f})")
        self.status["qd"] = qd_name
        # Which port this QD goes into is decided later, by the manifold's own
        # survey - deliberately not derived from which QD this is. Tying the
        # two together would put the first QD in a middle port: the tray
        # choice is made on the tray's geometry, and a hardcoded qd_N -> port N
        # map would carry that answer across, so "start from an edge" would
        # hold on the tray and not on the manifold.
        if self.args.plc:  # G2 - arm converged at the tray hover
            self.plc_gate(Phase.TRACK_QD, label="G2 PICK_PERMIT")
        qd_xy = self.track_descend(TRAY_HOVER_XY, TRAY_HOVER_Z, QD_TRACK_FLOOR_Z,
                                   target_xy, self.detect_qds, QD_MATCH_RADIUS_M, qd_name,
                                   shape_kind="polygon", active_color=HEX_COLOR)
        self.picked_qds_xy.append(np.asarray(qd_xy, dtype=float))
        self.clear_wrist_overlay()
        grasp_point = np.array([qd_xy[0], qd_xy[1], QD_GRASP_Z])
        diff_mm = np.linalg.norm(
            grasp_point - (QD_NEST_HARDCODED[qd_name] + GRASP_OFFSET)) * 1000
        self.status["diff_mm"] = diff_mm
        if self.sig:
            self.sig.note_track_diff(diff_mm)
        print(f"[vision] {qd_name} final tracked grasp point: {grasp_point} "
              f"(vs hardcoded, diff={diff_mm:.2f}mm)")

        # No fly_safe() to the grasp point: tracking already brought the arm
        # down to QD_TRACK_FLOOR_Z directly above the tracked X/Y, and
        # PickSequence handles the remaining descent itself.
        pick = PickSequence(self.arm, self.gripper, grasp_point)
        pick.start(self.env.data, self.env.dt)
        prev_state = pick.state

        def pick_step():
            nonlocal prev_state
            self.set_phase(f"pick:{pick.state}")
            pick.tick(self.env.data, self.env.dt)
            if prev_state == "GRASP" and pick.state == "LIFT":
                self.weld.engage(self.env.data, qd_name, self.arm.site_id)
                # Kill the QD's collision the instant it is welded, not just
                # once it is placed. From here the weld provides all of its
                # positional control, so contact with the gripper holding it
                # serves no purpose and is actively dangerous: a mocap body
                # cannot be pushed, so if the closed pads penetrate the flange
                # at all the solver cannot resolve it by moving the QD and
                # escalates the correction force every tick instead. Measured
                # climbing from ~36N to over 14,000N in about 20 ticks during
                # screwing, as the hex corners swept past the pads.
                disable_qd_collision(self.env.model, qd_name)
                # Freeze the gripper's ctrl target at its true, contact-limited
                # position the same tick. The grasp hold commands fully-closed
                # regardless of contact, and contact with the QD was the only
                # thing keeping it from getting there - which the line above
                # just removed. Without this the fingers close through the
                # flange.
                self.gripper.hold_here(self.env.data)
            prev_state = pick.state

        steps = self.run_until(pick_step, pick.is_done, 15000)
        self.status["tilt_deg"] = tilt_deg(self.env.data, qd_name)
        if self.sig:
            self.sig.note_tilt(self.status["tilt_deg"])
            self.sig.note_qd_grasped(True)
        print(f"[{qd_name}] Pick done in {steps} steps, "
              f"tilt={self.status['tilt_deg']:.2f}deg")
        return qd_name

    def _choose_and_track_port(self, round_num, qd_name):
        """Survey the manifold, choose the edge-most empty port, track it
        down, and carry the QD to it."""
        # The same roster rule as the tray, applied to the destination.
        # detect_ports() already drops anything in filled_ports_xy.
        self.set_phase(f"survey:ports:{round_num + 1}")
        self.fly_safe(MANIFOLD_HOVER_XY, MANIFOLD_HOVER_Z)
        holes, _img, _pairs, _occ = self.detect_ports()
        if self.sig:
            self.sig.vision_part_detected(len(holes))
        port_target_xy = self.choose_next(holes, self.filled_ports_xy)
        if port_target_xy is None:
            fallback = SESSION[round_num][1]
            print(f"[vision] port survey {round_num + 1}: saw {len(holes)} empty hole(s), "
                  f"none usable - falling back to hardcoded port {fallback}")
            port_target_xy = np.asarray(PORTS_HARDCODED[fallback][:2], dtype=float)
            if self.sig:
                self.sig.vision_fallback()
        port_num = min(PORTS_HARDCODED,
                       key=lambda k: np.linalg.norm(port_target_xy - PORTS_HARDCODED[k][:2]))
        self.status["port"] = port_num
        if self.sig:
            self.sig.note_port_id(port_num)
        print(f"[vision] port survey {round_num + 1}: saw {len(holes)} empty hole(s), "
              f"{len(self.filled_ports_xy)} already filled -> chose port {port_num} at "
              f"({port_target_xy[0]:.4f}, {port_target_xy[1]:.4f})")

        if self.args.plc:  # G3 - arm converged at the manifold hover
            self.plc_gate(Phase.TRACK_PORT, label="G3 PORT_PERMIT")
        port_xy = self.track_descend(MANIFOLD_HOVER_XY, MANIFOLD_HOVER_Z, PORT_TRACK_FLOOR_Z,
                                     port_target_xy, self.detect_ports, PORT_MATCH_RADIUS_M,
                                     f"port_{port_num}", shape_kind="circle",
                                     active_color=HOLE_COLOR)
        self.filled_ports_xy.append(np.asarray(port_xy, dtype=float))
        self.clear_wrist_overlay()
        port = np.array([port_xy[0], port_xy[1], PORT_Z])
        diff_mm = np.linalg.norm(port - PORTS_HARDCODED[port_num]) * 1000
        self.status["diff_mm"] = diff_mm
        if self.sig:
            self.sig.note_track_diff(diff_mm)
        print(f"[vision] port_{port_num} final tracked position: {port} "
              f"(vs hardcoded, diff={diff_mm:.2f}mm)")

        # No centering fly_safe() here either: tracking already brought the arm
        # directly above the tracked port's X/Y, and TransportSequence flies
        # its own well-above-the-port waypoint.
        #
        # The pinch-to-thread-face offset is measured, not a hardcoded flange
        # constant, and is constant from here through insertion because the
        # orientation never changes in between.
        pinch_pos, _ = site_pose(self.env.data, self.arm.site_id)
        offset_m = pinch_pos[2] - self.thread_face_z(qd_name)
        transport = TransportSequence(self.arm, port, offset_m)
        transport.start(self.env.data, self.env.dt)

        def transport_step():
            self.set_phase(f"transport:{transport.state}")
            transport.tick(self.env.data, self.env.dt)

        steps = self.run_until(transport_step, transport.is_done, 20000)
        print(f"[{qd_name}] Transport done in {steps} steps, "
              f"tilt={tilt_deg(self.env.data, qd_name):.2f}deg")
        return port

    def _insert(self, qd_name, port):
        """Screw the QD in, decelerate to a stop, and hold still.

        Kinematic thread-advance: helical motion as the control law itself,
        not force-discovered - the welded QD cannot generate a force to
        discover anything from. See qd_sim/control/screw_controller.py.

        Turning starts from wherever transport's pre-insert point left off,
        well above the manifold, rather than from a separate approach move.
        That approach would be one big Cartesian jump, and at this
        stretched-out reach the arm settles into a few-mm steady-state droop
        under its own weight, so the jump overshoots before settling into it -
        visually a dive at the manifold, and a real risk of touching it. The screw controller
        already advances a fraction of a millimetre per tick, so starting it
        early removes the risky move rather than trying to make it safe. The
        turn count simply grows to cover the extra distance; the pitch is a
        property of the thread, not something scaled to fit.

        reverse=False verified in world coordinates, not by eye off an angled
        render: this direction turns clockwise seen from above while
        descending, the right-hand-screw tightening convention."""
        if self.args.plc:  # G4 - tight-converged at pre-insert (0.002m tol)
            self.plc_gate(Phase.TRANSPORT_DESCEND_PRE_INSERT, label="G4 INSERT_PERMIT")
        screw = ScrewController(self.arm, self.env.model, ENGAGEMENT_DEPTH_M, EXPECTED_TURNS,
                                SCREW_ANGULAR_SPEED_RAD_S,
                                angular_accel_rad_s2=SCREW_ANGULAR_ACCEL_RAD_S2, reverse=False,
                                unwind_speed_rad_s=UNWIND_ANGULAR_SPEED_RAD_S,
                                unwind_accel_rad_s2=UNWIND_ANGULAR_ACCEL_RAD_S2)
        screw.start(self.env.data)
        self.set_phase(f"screw_turn:{qd_name}")

        def screw_tick():
            # Publish depth every physics step, not only at gates.
            screw.tick(self.env.data, self.env.dt)
            if self.sig:
                self.sig.publish_depth(self.thread_face_z(qd_name), port[2], ENGAGEMENT_DEPTH_M)

        steps = self.run_until(screw_tick,
                               lambda: self.thread_face_z(qd_name) <= port[2], 40000)
        # Ramp rotation and descent down to a stop rather than freezing: an
        # abrupt stop is a real reaction-torque jolt on the arm, not just a
        # cosmetic one. The brief hold afterwards gives the stop a clear beat -
        # opening the gripper before rotation has visibly stopped looks like
        # it cuts the stop short.
        screw.begin_stop(self.env.data)
        self.set_phase(f"screw_decel:{qd_name}")
        decel_steps = self.run_until(
            lambda: screw.decelerate_tick(self.env.data, self.env.dt), screw.is_stopped, 4000)
        self._dead_hold(60)
        print(f"[{qd_name}] Insert done in {steps}+{decel_steps} steps, "
              f"thread_face_z={self.thread_face_z(qd_name):.4f} (target {port[2]:.4f}), "
              f"tilt={tilt_deg(self.env.data, qd_name):.2f}deg")
        if self.sig:
            seated = self.thread_face_z(qd_name) <= port[2] + 0.0005  # 0.5mm grace
            self.sig.publish_seated(seated)
            self.sig.publish_seat_shortfall(not seated)
        return screw

    def _dead_hold(self, ticks):
        """Step physics with the arm completely untouched.

        From the end of screwing until the gripper has fully opened, no
        arm.step() runs at all - ctrl just holds. Calling arm.step() here
        keeps the position servos live even against an unchanging target, and
        that residual correction is enough to visibly disturb the seated QD,
        which is still welded until the weld is released.

        Note the frame gating: calling record_frame() on every physics tick,
        at 500Hz physics against a 30fps writer, would silently turn 0.3s of
        motion into 5 seconds of slow-motion in the video."""
        for i in range(ticks):
            self.untouched_tick(i)

    def _release_and_retract(self, qd_name, screw):
        """Let go of the seated QD, rise clear, and unwind the tool joint."""
        # G5 is the interlock that matters: the supervisor decides the part is
        # seated deep enough before the gripper may let go. It sits right after
        # the dead hold that ends _insert(), the one window where the arm is
        # already deliberately motionless - and the gate pump never calls arm.step(), so waiting
        # here cannot disturb the seated QD.
        if self.args.plc:
            self.plc_gate(Phase.GRIPPER_RELEASE, label="G5 RELEASE_PERMIT")

        # Open the gripper on a ramp (open() snaps in one step, which looks
        # like a pop), arm still untouched, then release the weld. The QD stays
        # exactly where it is - mocap bodies never move on their own.
        start_ctrl = self.env.data.ctrl[self.gripper.actuator_id]
        for tick in range(GRIPPER_OPEN_RAMP_TICKS):
            frac = (tick + 1) / GRIPPER_OPEN_RAMP_TICKS
            self.gripper.close(self.env.data,
                               ctrl_value=start_ctrl + frac * (self.gripper.OPEN_CTRL - start_ctrl))
            self.untouched_tick(tick)
        if self.sig:
            self.sig.note_gripper_open(True)
        self.weld.release()
        if self.sig:
            self.sig.note_qd_grasped(False)
        disable_qd_collision(self.env.model, qd_name)

        if self.args.plc:  # G6 - gripper open, part released
            self.plc_gate(Phase.POST_SCREW_RISE, label="G6 RETRACT_PERMIT")

        # Rise clear BEFORE unwinding the tool joint, not after. Unwinding
        # down at the insertion height is harmless - the QD is released and it
        # is a pure rotation - but the gripper visibly spins at the height of
        # the other installed QDs, which reads as about to hit them.
        cur_pos, cur_quat = site_pose(self.env.data, self.arm.site_id)
        self.set_phase(f"post_screw_rise:{qd_name}")
        self.move_to([cur_pos[0], cur_pos[1], SAFE_Z], cur_quat)

        # Unwind to the nearest angle that looks the same as the pre-screw one,
        # not the full amount screwed.
        #
        # Deliberately sequential, not concurrent with the flight to the next
        # QD, which fails in two ways. Without compensating for the screw
        # adapter's contribution to the measured gripper pose, the arm
        # self-collides (12 pairs against the usual 0). Compensating
        # analytically leaves a subtler problem: the flight finishes long
        # before the unwind does, and ArmController only re-samples a clean
        # target_quat while its trajectory is still active - so afterwards the
        # correction compounds onto its own previous output instead of a fixed
        # baseline, and the target spins away without bound (past 70 radians,
        # far enough to physically drag the joint round with it).
        #
        # Doing it properly would mean ArmController tracking a separate
        # never-corrected baseline, a moderate change to code every motion here
        # depends on, for at most ~1s per QD. UNWIND_ANGULAR_ACCEL_RAD_S2 gets
        # most of that back instead: ~60% faster sequentially, no risk.
        screw.start_unwind(self.env.data)
        self.set_phase(f"unwind:{qd_name}")
        steps = self.run_until(lambda: screw.unwind_tick(self.env.data, self.env.dt),
                               lambda: screw.unwind_done(self.env.data), 8000)
        print(f"[{qd_name}] Unwound in {steps} steps")

    def return_to_ready(self):
        """All four placed - fly back to the exact pose the arm started from.

        Reorient to ready_quat while still up at the safe height, then descend
        already in that orientation. Reorienting down at ready_pos
        self-collides (gripper base mount vs wrist_2) - the same problem
        fly_safe routes every other transition around."""
        self.set_phase("fly_ready")
        self.fly_safe(self.ready_pos[:2], SAFE_Z, quat=GRIPPER_DOWN_QUAT)
        self.move_to([self.ready_pos[0], self.ready_pos[1], SAFE_Z], self.ready_quat)
        self.move_to(self.ready_pos, self.ready_quat)

        # Exact snap to the ready keyframe's own joint angles: IK reaching the
        # same Cartesian pose gives a visually identical result, not identical
        # joint angles.
        #
        # The 800-tick cap is measured. This loop's tolerance is never actually
        # reachable - the joint error plateaus at ~0.0107rad by tick ~750 and
        # never improves, the same steady-state droop the screw approach runs
        # into - so a longer cap only spends time achieving nothing (a
        # 3000-tick cap is 6 seconds per session).
        self.env.data.ctrl[self.arm.actuator_idx] = self.ready_arm_qpos
        for i in range(800):
            self.untouched_tick(i)
            if np.allclose(self.env.data.qpos[self.arm.qpos_idx],
                           self.ready_arm_qpos, atol=0.001):
                break
        print(f"Returned to ready pose in {i} steps")

    # ---- the leak-test phase ---------------------------------------------

    def leak_test_all(self):
        """Trace the seam of every installed flange, one port at a time.

        Starts only once the pick arm has installed all four QDs and parked.
        The two arms never move at once, for the same reasons the unwind
        doesn't overlap the next flight. The pick arm is
        not touched again from here. See qd_sim/tasks/leak_test_sequence.py
        for the per-port inspection FSM."""
        self.set_phase("leak_test")
        if self.sig:
            self.sig.note_active_arm(True)
        self.dashboard_cam = "leak"
        self.clear_wrist_overlay()
        for qd_name, _port in SESSION:
            set_qd_collision_for_inspection(self.env.model, qd_name)
        for round_num in range(len(SESSION)):
            self._leak_test_one(round_num)

    @property
    def _leak_survey_point(self):
        """One fixed pose on the port row, high enough to frame all four
        installed flanges - the same roster scheme the pick arm uses over the
        tray. Checked at four heights, all four found at every one to
        0.7-2.3mm, which is selection-grade: it only has to tell apart ports
        45mm apart. The per-port detect pass then refines the chosen one to
        0.41mm before any ring is built.

        The done-set matters more here than on the tray. A picked QD leaves
        the tray, so that survey thins out by itself; an inspected QD stays
        exactly where it is, and nothing in the image distinguishes it from a
        pending one."""
        x = ((PORTS_HARDCODED[1][0] + PORTS_HARDCODED[len(SESSION)][0]) / 2
             - LEAK_CAM_OPTICAL_OFFSET_XY_M[0])
        y = PORTS_HARDCODED[1][1] - LEAK_CAM_OPTICAL_OFFSET_XY_M[1]
        return np.array([x, y, PORT_Z + LEAK_SURVEY_HOVER_M])

    def _leak_test_one(self, round_num):
        """Fly to the survey pose, choose the next untested port, and trace
        its seam."""
        self.set_phase(f"survey:manifold:{round_num + 1}")
        self.leak_arm.set_smooth_target_paced(self.env.data, self._leak_survey_point,
                                              PROBE_DOWN_QUAT, self.env.dt)
        survey_tick = 0

        def survey_step():
            # The leak pane has to keep updating through this move. run_until
            # renders the third-person video every frame regardless, but
            # nothing else refreshes this pane - without it the pane freezes on
            # its last frame for the whole flight back up, which reads as the
            # camera having lagged or restarted.
            nonlocal survey_tick
            self.leak_arm.step(self.env.data, self.env.dt)
            if self.dash is not None and survey_tick % STEPS_PER_FRAME == 0:
                cam_id = self.env.model.camera("leak/leak_cam").id
                K = intrinsics_from_mj_camera(self.env.model, "leak/leak_cam",
                                              VISION_WIDTH, VISION_HEIGHT)
                sh, lb = self._leak_roster_shapes(K, self.env.data.cam_xpos[cam_id],
                                                  self.env.data.cam_xmat[cam_id],
                                                  self._last_seen, self.tested_ports_xy)
                self.wrist_view.update(img=self.render_leak_cam(), shapes=sh,
                                       labels=lb, coast_text=None)
            survey_tick += 1

        self.run_until(survey_step,
                       lambda: self.leak_arm.is_converged(self.env.data, pos_tol_m=0.002,
                                                          angle_tol_deg=1.0), 12000)
        seen = survey_installed_flanges(self.env.model, self.env.data, "leak/leak_cam",
                                        VISION_WIDTH, VISION_HEIGHT,
                                        self.render_leak_cam, PORT_Z)
        if seen:
            self._last_seen = list(seen)
        if self.sig:
            self.sig.vision_part_detected(len(seen))
        target_xy = self.choose_next(seen, self.tested_ports_xy)
        if target_xy is None:
            fallback = SESSION[round_num][1]
            print(f"[vision] manifold survey {round_num + 1}: saw {len(seen)} flange(s), "
                  f"none un-tested - falling back to hardcoded port {fallback}")
            target_xy = np.asarray(PORTS_HARDCODED[fallback][:2], dtype=float)
            if self.sig:
                self.sig.vision_fallback()
        port_num = min(PORTS_HARDCODED,
                       key=lambda k: np.linalg.norm(target_xy - PORTS_HARDCODED[k][:2]))
        print(f"[vision] manifold survey {round_num + 1}: saw {len(seen)} flange(s), "
              f"{len(self.tested_ports_xy)} already tested -> chose port {port_num} at "
              f"({target_xy[0]:.4f}, {target_xy[1]:.4f})")
        self.tested_ports_xy.append(np.asarray(target_xy, dtype=float))
        self.status["leak_port"] = port_num
        if self.sig:
            self.sig.note_port_id(port_num)

        if self.args.plc:
            # G9 - leak arm converged at the survey pose (0.002m/1deg),
            # guaranteed by the run_until() just above.
            self.plc_gate(Phase.LEAK_SURVEY, label="G9 LEAK_PORT_PERMIT")
        leak_seq = LeakTestSequence(self.leak_arm, self.env.model,
                                    np.array([target_xy[0], target_xy[1], PORT_Z]),
                                    cam_name="leak/leak_cam", cam_width=VISION_WIDTH,
                                    cam_height=VISION_HEIGHT, render_fn=self.render_leak_cam)
        leak_seq.start(self.env.data, self.env.dt)
        overlay_tick = 0

        def leak_step():
            nonlocal overlay_tick
            leak_seq.tick(self.env.data, self.env.dt)
            if self.sig:
                # The leak sequence never calls set_phase(), so this is the
                # only thing reporting where the arm is between G9 and G10.
                self.sig.publish_leak_state(leak_seq.state)
            if self.dash is not None and overlay_tick % STEPS_PER_FRAME == 0:
                self._draw_leak(leak_seq, port_num, self._last_seen,
                                self.tested_ports_xy, target_xy)
            overlay_tick += 1

        # 40000 rather than 20000: the longer-offset probe has a real
        # steady-state droop to settle out of, not a performance choice.
        steps = self.run_until(leak_step, leak_seq.is_done, 40000)
        print(f"[leak-test port {port_num}] done in {steps} steps, "
              f"final state={leak_seq.state}")
        if self.sig:
            self.sig.note_leak_port_complete(round_num + 1)
        if self.args.plc:  # G10 - next port, or teardown
            self.plc_gate(Phase.LEAK_PORT_DONE, label="G10 LEAK_PORT_END")

    # ---- finishing up ----------------------------------------------------

    def finish(self):
        """Flush the video, report the run, write the diagnostics, and (with
        --plc) stay powered up and idle."""
        # Hold the final frame for a beat so the video doesn't end abruptly.
        for _ in range(30):
            self.record_frame()
            if self.sig:
                # 30 renders is ~1.5s of wall time. Going quiet for that long
                # walks into the 2s comms watchdog just as the session is
                # trying to report success, and the fault would win.
                self.sig.tick()

        print(f"\nFrames recorded: {self.frame_count} "
              f"({self.frame_count/FPS:.1f}s at {FPS}fps)")
        print(f"Peak arm/gripper self-collision contacts anywhere in the run: "
              f"{self.self_collision_peak} "
              f"(pairs involved: {self.self_collision_pairs or 'none'})")
        print(f"Peak leak-arm/QD/manifold contacts anywhere in the run: "
              f"{self.leak_contact_peak} "
              f"(pairs involved: {self.leak_contact_pairs or 'none'})")
        if self.sig:
            self.sig.note_session_done()
        self.writer.release()
        print(f"Saved: {self.out_path}")
        self._write_diagnostics()
        if self.sig:
            self._idle_hold()

    def _write_diagnostics(self):
        """Per-tick position and target, plus every vision event, so a run can
        be analysed from real data rather than by eyeballing the video."""
        out_dir = self.out_path.parent
        with open(out_dir / "tracking_debug.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["tick", "phase", "ax", "ay", "az", "tx", "ty", "tz"])
            w.writerows(self.tick_log)
        with open(out_dir / "vision_events.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["tick", "phase", "status", "dist_mm", "x", "y"])
            w.writerows(self.vision_events)
        print(f"Diagnostic logs saved: {out_dir/'tracking_debug.csv'} "
              f"({len(self.tick_log)} rows), {out_dir/'vision_events.csv'} "
              f"({len(self.vision_events)} events)")

    def _idle_hold(self):
        """Stay powered up and idle, the way a real controller does at the end
        of a part.

        Writing the CSVs and releasing the video writer takes seconds during
        which nothing publishes, so exiting straight afterwards would have the
        supervisor's next poll find the device gone and report a comms fault
        instead of the COMPLETE it had just earned. Holds until the PLC clears
        RUN or the operator stops the cell."""
        print("\n[PLC] session complete; holding and still publishing "
              "(Ctrl-C to stop)")
        started = time.time()
        while time.time() - started < PLC_IDLE_HOLD_S:
            self.sig.tick()
            if not self.modbus.data.get("RUN"):
                break
            time.sleep(0.02)

    def _report_abort(self, headline, exc):
        print(f"\n{headline}: {exc}")
        print(f"Frames recorded before abort: {self.frame_count} "
              f"({self.frame_count/FPS:.1f}s at {FPS}fps)")
        self.writer.release()
        print(f"Partial video saved: {self.out_path}")

    def close(self):
        """Shut the window and release the Modbus port. A server thread left
        running is what makes the NEXT run fail with 'address already in
        use'."""
        if self.dash is not None:
            self.dash.close()
        if self.modbus is not None:
            self.modbus.data.set_raw("ABORT_COMPLETE", True)
            self.modbus.data.set_raw("SIM_STOPPED", True)
            self.modbus.stop()
