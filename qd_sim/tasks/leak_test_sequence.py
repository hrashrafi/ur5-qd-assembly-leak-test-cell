"""Leak-test inspection of one manifold port.

Fly the probe tip to a hover above the seam, detect the installed QD's real
position AND orientation from the camera, trace a ring around its actual hex
flange - offset outward for clearance, and hex-shaped rather than a plain
circle (see qd_sim/vision/seam_path.py) - then retract. States:
TO_PORT_ABOVE -> HOVER_DETECT -> ORBIT_SEAM -> RETRACT -> DONE.

Detecting an INSTALLED QD is a different vision problem from the pick arm's
detect_hexagons(), which works against the tray's blue floor but not the
manifold's similarly bright top face. This uses a dedicated detector,
shape_detector.detect_flange_hexagon(): one local contour fit, looking
straight down, that returns the flange's centre and its corner orientation
together. The corner RADIUS is a known CAD constant and needs no detecting.
Looking straight down is load-bearing - see LEAK_CAM_OPTICAL_OFFSET_XY_M.

No rotating tool joint is involved; the probe is rigidly mounted, because a
rotate-in-place design can only ever trace a circle. Every waypoint targets
the "leak/probe_tip" site through ordinary 6-DOF IK and
ArmController.set_smooth_path() - the same mechanism every other motion here
uses, with a hex-shaped set of waypoints.
"""

import numpy as np

from qd_sim.control.state_machine import StateMachine
from qd_sim.robot.arm_controller import DEFAULT_CRUISE_SPEED_MPS as CRUISE_SPEED_MPS
from qd_sim.robot.arm_controller import DEFAULT_MIN_MOVE_STEPS as MIN_MOVE_STEPS
from qd_sim import transforms as tf
from qd_sim.vision.camera_intrinsics import intrinsics_from_mj_camera
from qd_sim.vision.ray_cast import pixel_to_world_on_plane
from qd_sim.vision.seam_path import hex_ring_from_known_geometry, world_yaw_from_pixel_angle
from qd_sim.vision.shape_detector import detect_flange_hexagon, detect_flange_hexagons_all


def set_qd_collision_for_inspection(model, qd_name):
    """Put qd_name's collision geoms into the state that makes a leak-test
    contact check mean something, whatever state they were in before. This
    has to be idempotent and order-independent:

      - col_flange ENABLED: the seam surface the whole feature exists to
        inspect. In the real session flow disable_qd_collision() has already
        turned it off along with the other two - for a good reason of its
        own, a mocap body's reaction force fighting the pick arm during later
        installs - but that leaves a contact check with nothing to mean
        anything against: it would report a clean "0 contacts" purely
        because the collision is off.
      - col_far DISABLED: this is the QD's smooth carrying shaft, a
        pick-fixture handle that is never part of the surface being
        inspected. Left enabled, a full 360deg orbit genuinely swings the
        probe's trailing mass through its radius - which is not the failure
        these counters exist to catch. A real sniffer brushing an inert
        handle above the joint is not a leak-test failure.

    col_thread is left untouched: it sits inside the manifold's solid
    interior post-installation, which is what "installed" means, and any
    contact it generates is QD-vs-manifold - a different pair.

    The session calls this once per QD, right before running a
    LeakTestSequence against it."""
    flange_id = model.geom(f"{qd_name}_col_flange").id
    model.geom_contype[flange_id] = 1
    model.geom_conaffinity[flange_id] = 1
    far_id = model.geom(f"{qd_name}_col_far").id
    model.geom_contype[far_id] = 0
    model.geom_conaffinity[far_id] = 0


# Point the probe straight down - the same 180deg-about-X flip
# pick_sequence.py's GRIPPER_DOWN_QUAT uses. This is only correct because
# the probe CAD is a straight, coaxial rod, so the body's main axis and the
# tip's pointing direction are the same line. A bent probe would leave the
# tip aimed off-vertical by the bend's own angle and need a measured
# tip-alignment quat on top.
PROBE_DOWN_QUAT = tf.mat_to_quat(np.diag([1.0, -1.0, -1.0]))

# leak_cam's optical-centre offset from the "leak/probe_tip" site - the same
# idea as qd_cell/constants.py's WRIST_CAM_OPTICAL_OFFSET_XY_M: the camera is
# mounted off to one side of the site being positioned, so hovering the site
# over a target does not put the camera over it. It is the X/Y distance from
# probe_tip to the point on the seam-Z plane that the camera's centre pixel
# sees. Every hover target below subtracts it, so the CAMERA centres on the
# port.
#
# This is a rigid constant only because leak_cam looks straight down, along the
# probe's own axis. A vertical optical ray's X/Y does not depend on which
# Z-plane it is cast to, so the offset is the same at every hover position
# and height, and the port lands at the image centre - which
# detect_flange_hexagon()'s centre crop assumes. Aimed at the tip instead, the
# camera body would sit ~145mm off to the side of the port, giving a
# systematic ~4.7mm ring-centre error and a ~20deg oblique view of the
# flange.
LEAK_CAM_OPTICAL_OFFSET_XY_M = (0.07000, 0.03000)

# Height to hover and detect from: close enough for a clean flange contour,
# and high enough that the probe's mount body - its bulkiest part, right at
# the site origin - clears the neighbouring QDs.
HOVER_CLEARANCE_M = 0.15
RETRACT_CLEARANCE_M = 0.15

# How far outside the flange's real corner radius (15.6mm, see
# seam_path.FLANGE_CORNER_RADIUS_M) the probe tip traces.
#
# Swept against the real installed QD poses - not an idealised seating, which
# can look clean in isolation and then fail in the integrated session -
# counting real leak-arm contacts:
#
#     3.0mm  34259 contacts, worst -0.2177mm
#     4.0mm  14257 contacts, worst -0.0442mm
#     5.0mm   1136 contacts, worst -0.0048mm
#     5.5mm      0 contacts
#     6.0mm and above: 0 contacts
#
# Monotone with a clean edge at 5.5mm - a sign the geometry is deterministic
# rather than fighting detection error. 6.5mm gives a full millimetre past
# where contact stops while staying snug against the seam.
#
# The contact threshold sits higher than the tip's own 2mm capsule radius
# suggests because the flange's COLLISION geom is a cylinder circumscribed
# around a hexagon, so it bulges up to ~2.1mm past the real flats. Visual
# clearance from the actual part is the full ~6.5mm; the check is the
# conservative one.
OUTWARD_OFFSET_M = 0.0065

# The ring is traced this far ABOVE the true seam height, not at it. The tip's
# collision geometry has real thickness (a 2mm-radius capsule), so targeting
# seam_z exactly would put part of that capsule inside the manifold's solid
# interior (6 contacts per orbit). 4mm clears the capsule with margin and
# still reads as inspecting the seam; a real sniffer senses escaping gas near
# the joint, it does not touch it.
SEAM_HOVER_CLEARANCE_M = 0.004

# How many independent (centre, yaw) detections to average before locking the
# ring - the same "require a few consistent reads, not just the first" rule
# used during the pick descent, except that this waits once for a stable
# reading rather than tracking continuously, the port not being in motion.
# Yaw is averaged with the circular-mean-mod-60deg trick
# detect_flange_hexagon() uses internally: a hex's 6-fold symmetry means a
# naive average gets the wraparound wrong.
DETECT_SAMPLES_REQUIRED = 5
DETECT_STRIDE_TICKS = 16     # once per rendered frame (500Hz / 30fps), not
                              # every physics tick - rendering is the cost
DETECT_MAX_TICKS = 4000      # ~8s. If vision never locks, fall back to the
                              # assumed port pose rather than stalling
PORT_MATCH_RADIUS_M = 0.020  # a flange further than this from the assumed port
                              # XY is a different port - under half the 45mm
                              # port spacing

TRANSIT_LOOSE_POS_TOL_M = 0.03
TRANSIT_LOOSE_ANGLE_TOL_DEG = 8.0

# The orbit is deliberately much slower than every other move in the project.
# The ring's circumference is tiny - radius ~19-23mm, set by the real port
# spacing and flange size - so at the normal 0.30m/s transit speed the whole
# orbit finishes in under half a second - tracing the ring correctly, but too
# fast to see.
ORBIT_EDGE_SPEED_MPS = 0.05
# The move INTO the ring, from the hover pose down to the first corner, is a
# transit rather than an inspection stroke. Pacing it at the inspection speed
# would make it by far the largest cost in the phase and make the probe look
# like it had stalled - a ~146mm descent at 20mm/s takes over 7s, during which
# the camera creeps downward almost imperceptibly. Profiled per port with the
# descent at inspection speed:
#
#     TO_PORT_ABOVE   2446 ticks   4.89s   13.5%
#     HOVER_DETECT      65 ticks   0.13s    0.4%
#     ORBIT_SEAM     14771 ticks  29.54s   81.3%
#     RETRACT          879 ticks   1.76s    4.8%
#
#     edge 0 (the descent into the ring)  9.72s   <- nearly 3x any other edge
#     edges 1-6 (the real hex legs)       3.2-3.4s each
#
# Only the approach is sped up. Every hex edge still traces at
# ORBIT_EDGE_SPEED_MPS, that slowness being the point.
ORBIT_DESCENT_SPEED_MPS = 0.15
# Floor each edge's move at a real wall-clock minimum, so even the short 25mm
# legs between corners read as a deliberate traverse rather than a blink.
ORBIT_EDGE_MIN_DURATION_S = 0.35
# A safety net on waiting for one edge's is_converged(), not a tuning knob.
ORBIT_EDGE_MAX_TICKS = 6000
# How close is close enough before advancing to the next corner. This is what
# decides how exactly the traced path follows the planned ring, and it is
# nowhere near the arm's physical tracking limit - at a 10mm tolerance every
# short hex edge lands at very nearly exactly 10mm off, which is the
# tolerance accepting the move as done well before the trajectory's own
# deceleration would close the gap. Swept by measuring the worst inward
# deviation of the travelled path from the planned ring over a whole orbit:
#
#     tol        worst inward dev      ticks/port
#     3.00mm     1.67mm                ~16000
#     1.50mm     0.78mm                ~17900
#     1.00mm     0.50mm                ~18900
#     0.50mm     0.26mm                ~20800
#
# Dead linear, with no ORBIT_EDGE_MAX_TICKS fallback anywhere in the sweep.
# 1.0mm picked: half a millimetre of deviation on a 15.6mm-radius part is well
# under what is visible in the recording, for ~18% more ticks per port rather
# than the ~29% the next step down costs.
ORBIT_EDGE_POS_TOL_M = 0.001
ORBIT_EDGE_ANGLE_TOL_DEG = 0.4


def flat_lit_render(model, render_fn):
    """render_fn(), with the model's lighting temporarily flattened to
    ambient-only for just that one frame, then restored.

    Module-level (not a method) so the leak arm's SURVEY pass can render the
    same way a per-port detection does - see
    LeakTestSequence._flat_lit_render() for the full rationale and the
    measured effect, and survey_installed_flanges() below for the survey
    that also needs it."""
    n = model.nlight
    saved = (model.light_ambient[:n].copy(), model.light_diffuse[:n].copy(),
             model.light_specular[:n].copy(), model.vis.headlight.ambient.copy(),
             model.vis.headlight.diffuse.copy(), model.vis.headlight.specular.copy())
    try:
        for i in range(n):
            if model.light(i).name in ("spotlight", "leak/spotlight"):
                model.light_ambient[i], model.light_diffuse[i], model.light_specular[i] = \
                    [0, 0, 0], [0, 0, 0], [0, 0, 0]
            else:
                model.light_ambient[i], model.light_diffuse[i], model.light_specular[i] = \
                    [0.4, 0.4, 0.4], [0, 0, 0], [0, 0, 0]
        model.vis.headlight.ambient[:] = 0
        model.vis.headlight.diffuse[:] = 0
        model.vis.headlight.specular[:] = 0
        return render_fn()
    finally:
        (model.light_ambient[:n], model.light_diffuse[:n], model.light_specular[:n],
         model.vis.headlight.ambient[:], model.vis.headlight.diffuse[:],
         model.vis.headlight.specular[:]) = saved


def survey_installed_flanges(model, data, cam_name, cam_width, cam_height, render_fn, seam_z):
    """World (x, y) of EVERY installed QD flange currently in the leak
    camera's view, as a list.

    The roster counterpart to the per-port detection: fly to one fixed pose
    that frames the whole port row, see all four at once, and let the caller
    choose the next by excluding the ones it has already done, rather than
    walking a hardcoded order.

    Accuracy is 0.7-2.3mm, measured against all four ports at four survey
    heights - selection-grade, deliberately, not inspection-grade. It only has
    to tell apart four ports 45mm apart; the ring itself is built from the
    per-port HOVER_DETECT pass, which refines to 0.41mm once the arm is
    stationary above the chosen one.

    Uses the same flat lighting and the same detector as the per-port
    pass."""
    img = flat_lit_render(model, render_fn)
    cam_id = model.camera(cam_name).id
    K = intrinsics_from_mj_camera(model, cam_name, cam_width, cam_height)
    out = []
    for (px, py), _angle in detect_flange_hexagons_all(img):
        out.append(pixel_to_world_on_plane(px, py, K, data.cam_xpos[cam_id],
                                            data.cam_xmat[cam_id], seam_z)[:2])
    return out


class LeakTestSequence:
    def __init__(self, arm_ctrl, model, port_target_world, cam_name, cam_width, cam_height, render_fn):
        """
        arm_ctrl: ArmController for the leak-test arm (its site is the
            "leak/probe_tip").
        model: the compiled MjModel (for camera intrinsics lookups).
        port_target_world: (x, y, z) known/assumed port pose - the coarse
            hover seed AND the anchor every detection is checked against
            (PORT_MATCH_RADIUS_M) (same "assumed pose, vision only refines it"
            premise as everywhere else in this project's vision pipeline -
            never trusted directly for the final ring).
        cam_name/cam_width/cam_height: this arm's own camera identity, for
            intrinsics_from_mj_camera().
        render_fn: callable() -> a fresh BGR image from cam_name, called
            once per detection tick. Caller-supplied (not owned here) so
            this class doesn't need its own mujoco.Renderer - same division
            of responsibility as the session's own detect_qds() and
            detect_ports().
        """
        self.arm = arm_ctrl
        self.model = model
        self.port_target = np.asarray(port_target_world, dtype=float)
        self.port_xy = self.port_target[:2]
        self.seam_z = self.port_target[2]
        self.cam_name = cam_name
        self.cam_width = cam_width
        self.cam_height = cam_height
        self.render_fn = render_fn

        hover_xy = self.port_xy - np.array(LEAK_CAM_OPTICAL_OFFSET_XY_M)
        self.hover_point = np.array([hover_xy[0], hover_xy[1], self.seam_z + HOVER_CLEARANCE_M])
        self.retract_point = np.array([self.port_xy[0], self.port_xy[1], self.seam_z + RETRACT_CLEARANCE_M])

        self._dt = None
        self._K = None
        self._tick = 0
        self._samples = []  # list of (center_xy, yaw_rad)

        # Public, read-only-by-convention state for a caller's own --live
        # dashboard overlay (see the session's leak-test overlay) - nothing
        # INSIDE this class reads these back, they're purely for
        # external inspection after each tick(), same spirit as self.state
        # below. World coordinates, same frame as everything else here -
        # a caller wanting to actually draw them needs to project through
        # its own camera pose (qd_sim/vision/ray_cast.py's
        # world_to_pixel()), which is a dashboard-only concern this class
        # itself has no reason to know about.
        self.last_detection = None  # (center_xy, yaw_rad) - most recent
                                     # successful HOVER_DETECT sample, or
                                     # None before the first one lands.
        self.ring_points = None     # the locked ring, world XYZ, once
                                     # ORBIT_SEAM starts (None before that -
                                     # see _lock_ring_and_start_orbit()).

        self.fsm = StateMachine(
            handlers={
                "TO_PORT_ABOVE": self._h_to_port_above,
                "HOVER_DETECT": self._h_hover_detect,
                "ORBIT_SEAM": self._h_orbit,
                "RETRACT": self._h_retract,
            },
            start_state="TO_PORT_ABOVE",
        )

    def start(self, data, dt):
        self._dt = dt
        self._tick = 0
        self._samples = []
        self.last_detection = None
        self.ring_points = None
        self.arm.set_smooth_target_paced(data, self.hover_point, PROBE_DOWN_QUAT, dt,
                                          cruise_speed_mps=CRUISE_SPEED_MPS)
        self.fsm.reset()

    def _h_to_port_above(self, data):
        # is_converged(), NOT is_near_final() - unlike every OTHER loose
        # handoff in this project (which deliberately hand off while still
        # moving, since the next phase tolerates or even benefits from
        # carried-over velocity), vision sampling here needs a genuinely
        # STATIC camera. With is_near_final()'s loose 30mm/8deg tolerance,
        # HOVER_DETECT would start sampling while the arm is still drifting
        # the last few mm/deg into the hover pose, so early samples come
        # from a still-moving camera - one such sample can carry ~17mm of
        # error (vs. a fully-static detection's repeatable ~4mm), skewing
        # the 5-sample average enough to visibly shift the locked ring off
        # the true seam. With the arm actually stopped, detection is
        # perfectly deterministic (the same static frame gives the
        # identical pixel result on every call, no rendering noise at all)
        # - so waiting for a real stop here removes this failure mode at
        # the source rather than trying to filter/average around it.
        if self.arm.is_converged(data):
            self._K = intrinsics_from_mj_camera(self.model, self.cam_name, self.cam_width, self.cam_height)
            self._tick = 0
            self._samples = []
            return "HOVER_DETECT"
        return "TO_PORT_ABOVE"

    def _flat_lit_render(self):
        """render_fn(), but with the model's own lighting temporarily
        flattened to ambient-only for JUST this one frame. Under the
        scene's normal lighting, per-port yaw detection carries a ~1-9deg
        bias - consistent and repeatable to sub-degree precision run to
        run, since the whole sim is deterministic, i.e. a rendering
        artifact, not measurement noise.

        The cause: an edge detector finds a real 3D edge's boundary
        shifted by a few pixels depending on which way the scene's
        directional lights happen to hit that specific facet, since a
        Lambertian-lit surface's own brightness gradient runs right across
        real edges, not just flat regions - switching every light in the
        model to pure ambient (contributes the same regardless of a
        surface's own angle to anything) removes that gradient entirely,
        dropping the error under 1deg on all 4 ports. Also zeroes each
        arm's own per-body tracking spotlight ("spotlight" and
        "leak/spotlight", from the stock UR5e model) - these move with each
        arm and would otherwise reintroduce the exact same
        per-pose-dependent shading, especially the leak arm's own copy,
        which tracks its wrist right next to whatever's being inspected.

        It is scoped to this one internal, never-displayed render rather
        than applied scene-wide (flattening scene_real.xml's own lights
        directly): scene-wide, it would make the third-person video and
        the --live dashboard's feed look flat and unrealistic, and take
        away contrast the pick arm's tray-QD tracking (detect_hexagons())
        depends on. Here the lighting is restored immediately after, and
        never touches whatever render_wrist_cam()/render_leak_cam()
        produce for the actual video/dashboard, which call
        update_scene()+render() fresh and independently - the same
        accuracy win with zero visible side effect anywhere a human
        actually looks: the pick arm's own detection pipeline never calls
        this method at all (it has its own, completely separate render
        calls in the session), and this class's own render_fn is ONLY ever
        used for detection, never for anything recorded or displayed (see
        this class's own __init__ docstring)."""
        return flat_lit_render(self.model, self.render_fn)

    def _sample_once(self, data):
        """One detection attempt: find this port's flange in the frame,
        check it's really the one being inspected (not a neighbor), and
        read its real center AND orientation. Appends a (center_xy,
        yaw_rad) sample on success; does nothing otherwise (the caller's
        own tick/sample-count budget decides when to give up).

        Detection runs in a crop around the image's own CENTER, not the
        full frame: with all 4 ports occupied (the real leak-test
        scenario), a global search can return a DIFFERENT port entirely,
        or a merged blob spanning two adjacent occupied ports, and a
        "nearest to the assumed port_xy" filter would still pick whichever
        bad candidate happens to be nearest, up to 20-30mm off. A center
        crop works because the hover pose puts the camera directly over
        the assumed port (see LEAK_CAM_OPTICAL_OFFSET_XY_M above), so the
        right flange is always near image center and a neighbor 45mm away
        is ~78px off it - outside the crop, and in any case losing the
        nearest-to-center tie-break inside detect_flange_hexagon().

        The detector is a single detect_flange_hexagon() call: with the
        camera looking straight down (see that constant's own comment),
        the flange renders as a clean face-on hexagon, so fitting its real
        contour directly is both simpler and an order of magnitude more
        accurate than fitting a circle plus a few Hough line segments to
        an oblique blob. See detect_flange_hexagon()'s own docstring for
        the measured comparison.

        The PORT_MATCH_RADIUS_M check below is a sanity gate against
        locking the ring onto something that isn't this port at all."""
        img = self._flat_lit_render()
        cam_id = self.model.camera(self.cam_name).id
        found = detect_flange_hexagon(img, self.cam_width / 2, self.cam_height / 2)
        if found is None:
            return
        center_px, corner_angle_deg = found
        cam_pos, cam_mat = data.cam_xpos[cam_id], data.cam_xmat[cam_id]
        center_xy = pixel_to_world_on_plane(center_px[0], center_px[1], self._K,
                                             cam_pos, cam_mat, self.seam_z)[:2]
        if np.linalg.norm(center_xy - self.port_xy) >= PORT_MATCH_RADIUS_M:
            return
        yaw = world_yaw_from_pixel_angle(center_px, corner_angle_deg, self._K,
                                          cam_pos, cam_mat, self.seam_z)
        self._samples.append((center_xy, yaw))
        self.last_detection = (center_xy, yaw)

    def _lock_ring_and_start_orbit(self, data, center_xy, yaw_rad):
        # Just the 6 hex corners, not a subdivided ring - see _h_orbit()'s
        # own comment for why this is traversed edge-by-edge as discrete
        # straight-line moves rather than one continuous smoothed path.
        corners = hex_ring_from_known_geometry(center_xy, yaw_rad, OUTWARD_OFFSET_M)
        orbit_z = self.seam_z + SEAM_HOVER_CLEARANCE_M
        self._edge_targets = [np.array([p[0], p[1], orbit_z]) for p in corners]
        self._edge_targets.append(self._edge_targets[0])  # close the loop back to the start
        self.ring_points = list(self._edge_targets)  # public copy - see __init__'s own comment
        # Every corner uses the same fixed PROBE_DOWN_QUAT. Turning the
        # probe so its trailing mass points outward at every corner would
        # hit the same wrist singularity screwing avoids: pointing straight
        # down while sweeping yaw puts wrist_1/wrist_3 co-axial (see
        # qd_sim/control/screw_controller.py's module docstring, which is
        # why THAT motion uses a dedicated rotating joint, screw_adapter,
        # instead of the arm's own 6 joints). Through the arm's IK, 3 of 4
        # ports fail to converge, with angle error plateauing at 40-60deg
        # rather than closing - a reachability wall, not slow convergence
        # that more time would fix. Doing it properly would mean giving the
        # probe its own dedicated yaw joint, decoupled from the arm's 6-DOF
        # IK - a design addition, not a tuning change.
        self._edge_idx = 0
        self._edge_tick = 0
        self._advance_to_next_edge(data)
        return "ORBIT_SEAM"

    def _advance_to_next_edge(self, data):
        """Command a single straight-line move (position AND orientation)
        to the next hex corner - ArmController.set_smooth_target_paced()
        with only ONE destination traces a straight Cartesian line to it
        (no other waypoint for a spline to curve through). A single
        set_smooth_path() call through every corner would not work here:
        its Catmull-Rom curve doesn't ride exactly on the ring between
        control points and cuts each corner slightly inside it, enough for
        a small (sub-0.1mm) graze against the flange being inspected.
        Going corner-to-corner as discrete straight legs traces the
        hexagon's TRUE edges exactly, at the cost of a brief stop at each
        corner instead of one continuous sweep - a reasonable trade for a
        careful, deliberate inspection motion, not a fast production move."""
        target = self._edge_targets[self._edge_idx]
        # Edge 0 is the descent from the hover pose into the ring's first
        # corner - a transit, not a seam stroke - so it gets the faster
        # ORBIT_DESCENT_SPEED_MPS (see that constant's own comment for the
        # profile that motivated splitting these). Every subsequent edge is
        # a real hex leg and keeps the slow, deliberate inspection pace.
        descending = self._edge_idx == 0
        self.arm.set_smooth_target_paced(
            data, target, PROBE_DOWN_QUAT, self._dt,
            cruise_speed_mps=ORBIT_DESCENT_SPEED_MPS if descending else ORBIT_EDGE_SPEED_MPS,
            min_steps=max(MIN_MOVE_STEPS, int(ORBIT_EDGE_MIN_DURATION_S / self._dt)))

    def _h_hover_detect(self, data):
        if self._tick % DETECT_STRIDE_TICKS == 0:
            self._sample_once(data)
            if len(self._samples) >= DETECT_SAMPLES_REQUIRED:
                centers = np.array([c for c, _ in self._samples])
                yaws = np.array([y for _, y in self._samples])
                center = centers.mean(axis=0)
                # circular mean mod 60deg - same trick
                # detect_flange_hexagon() uses internally, applied again
                # here across independent samples for the same
                # 6-fold-symmetry reason.
                yaw = np.arctan2(np.mean(np.sin(6 * yaws)), np.mean(np.cos(6 * yaws))) / 6.0
                return self._lock_ring_and_start_orbit(data, center, yaw)
        self._tick += 1
        if self._tick >= DETECT_MAX_TICKS:
            # Vision never locked a stable reading - fall back to the
            # known/assumed port pose (yaw=0) rather than stalling this
            # port's inspection forever. A real miss here means the
            # inspection ring is only as accurate as the assumed geometry,
            # not vision-refined - worth flagging to the caller (e.g. a
            # printed warning), not silently treated as equivalent to a
            # real detection.
            print(f"[leak-test] WARNING: no stable hex detection at port "
                  f"{self.port_xy} after {DETECT_MAX_TICKS} ticks - "
                  f"falling back to the assumed pose, yaw=0.")
            return self._lock_ring_and_start_orbit(data, self.port_xy, 0.0)
        return "HOVER_DETECT"

    def _h_orbit(self, data):
        # is_converged(), NOT is_near_final(): is_near_final()'s default
        # pos_tol_m=0.03 (30mm) is fine for a big transit handoff (30mm is
        # a tiny fraction of a long move), but each EDGE here is itself
        # only ~20-25mm long - so "within 30mm of this edge's own corner"
        # is true almost as soon as the arm starts moving toward it, which
        # would skip corners and leave the probe hovering near the QD
        # instead of going around the flange. is_converged() instead ALWAYS
        # returns False while any trajectory is still active (see its own
        # docstring) and only fires once the current edge's move has
        # genuinely finished - exactly "must actually reach this corner"
        # behavior, unlike every OTHER loose is_near_final() handoff in
        # this project (those are deliberately fine to cut short - this one
        # is not, since stopping precisely at each corner is the whole
        # point of tracing the hexagon's true edges - see
        # _advance_to_next_edge()'s own comment).
        # ORBIT_EDGE_POS_TOL_M/ANGLE_TOL_DEG is TIGHTER than the arm's own
        # 3mm/2deg defaults here, not looser (see that constant's own
        # comment for the measured sweep).
        # Every other convergence check in this project can afford slop
        # because it's a handoff mid-transit; this one sets where the probe
        # physically traces the seam, and any slack it allows shows up
        # directly as the traced path cutting inside the planned ring - not
        # just at the corner it was allowed to stop short of, but along the
        # whole NEXT edge too, since that edge's straight line starts from
        # wherever this one actually stopped. ORBIT_EDGE_MAX_TICKS stays in
        # place regardless as a real safety net.
        self._edge_tick += 1
        reached = self.arm.is_converged(data, pos_tol_m=ORBIT_EDGE_POS_TOL_M,
                                         angle_tol_deg=ORBIT_EDGE_ANGLE_TOL_DEG)
        if not reached and self._edge_tick >= ORBIT_EDGE_MAX_TICKS:
            pos_err, angle_err = self.arm.final_pose_error(data)
            print(f"[leak-test] WARNING: edge {self._edge_idx} never reached full "
                  f"convergence after {ORBIT_EDGE_MAX_TICKS} ticks (pos_err="
                  f"{pos_err * 1000:.1f}mm, angle_err={angle_err:.2f}deg) - a real "
                  f"steady-state droop plateau, not slow convergence (see "
                  f"ORBIT_EDGE_MAX_TICKS's own comment) - moving on anyway.")
            reached = True
        if reached:
            self._edge_idx += 1
            self._edge_tick = 0
            if self._edge_idx >= len(self._edge_targets):
                self.arm.set_smooth_target_paced(data, self.retract_point, PROBE_DOWN_QUAT, self._dt,
                                                  cruise_speed_mps=CRUISE_SPEED_MPS)
                return "RETRACT"
            self._advance_to_next_edge(data)
        return "ORBIT_SEAM"

    def _h_retract(self, data):
        if self.arm.is_near_final(data, pos_tol_m=TRANSIT_LOOSE_POS_TOL_M,
                                   angle_tol_deg=TRANSIT_LOOSE_ANGLE_TOL_DEG):
            return "DONE"
        return "RETRACT"

    def tick(self, data, dt):
        self._dt = dt
        self.arm.step(data, dt)
        self.fsm.tick(data)
        return self.fsm.state

    @property
    def state(self):
        return self.fsm.state

    @property
    def current_edge_index(self):
        """Which entry of ring_points the arm is currently heading toward
        (0-based) - public alongside ring_points, for a caller's own
        overlay to highlight the active edge distinctly from the rest of
        the locked ring. Meaningless (0) before ring_points is set."""
        return getattr(self, "_edge_idx", 0)

    def is_done(self):
        return self.fsm.is_done()
