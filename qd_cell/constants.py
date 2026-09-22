"""Every tuning constant the session runs on, in one place.

Each value's comment says what it is and why it has the value it has. Nothing in this file imports
from the rest of the cell, so it can be read on its own.
"""

import mujoco
import numpy as np

# The QD's thread-side flange face, in the QD's own frame: its flange collision
# cylinder spans 34.9-42.9mm and its 12mm thread 42.9-54.9mm, so this is
# exactly where the thread starts. "Seated" means this face has reached the
# manifold's top surface - the whole thread is then inside the port.
THREAD_FACE_LOCAL_OFFSET = np.array([0, 0, 0.0429])

# Reference only: at runtime the real QD and port positions come from vision
# (CellSession.track_descend() in session.py). These are world.xml's authored
# positions, kept for three things - the coarse hover to fly to, the initial
# seed vision refines from, and a known-good number to print next to each
# detection. This being a simulation is the only reason that last one is
# possible.
QD_NEST_HARDCODED = {
    "qd_1": np.array([0.42, -0.23, 0.0559]),
    "qd_2": np.array([0.48, -0.23, 0.0559]),
    "qd_3": np.array([0.42, -0.17, 0.0559]),
    "qd_4": np.array([0.48, -0.17, 0.0559]),
}
PORTS_HARDCODED = {
    1: np.array([0.4575, 0.175, 0.050]),
    2: np.array([0.5025, 0.175, 0.050]),
    3: np.array([0.5475, 0.175, 0.050]),
    4: np.array([0.5925, 0.175, 0.050]),
}
# qd_N -> port N, in order
SESSION = [("qd_1", 1), ("qd_2", 2), ("qd_3", 3), ("qd_4", 4)]

# Body-name prefix for every leak-arm body.
# Its contact counter is kept separate from the pick arm's self-collision
# counter so the two stay independently comparable.
LEAK_ARM_PREFIX = "leak/"

# Grasp the flange, not the smooth shaft above it: a round shaft gives the
# closing pads nothing to lock against, so the QD can visibly twist in the grip
# while the gripper closes. This offset puts the pad's bottom tip 0.26mm above
# the flange's bottom edge - a little clearance, no thread under the pads.
GRASP_OFFSET = np.array([0.0, 0.0, -0.0125])

# 7/8"-14 UNF male straight thread (ISO 11926-3 / SAE J1926 ORB), from the
# connector's own datasheet. Its stated 12.0mm thread length matches
# ENGAGEMENT_DEPTH_M, and its hex dimensions match the QD model to within
# 0.02mm.
# See qd_sim/control/screw_controller.py for the control law itself.
QD_THREAD_TPI = 14
QD_THREAD_PITCH_M = 0.0254 / QD_THREAD_TPI  # 1.8143mm/turn
# Flange thread-side face (mesh z=42.9mm) to the true tip (z=54.9mm).
ENGAGEMENT_DEPTH_M = 0.0120
EXPECTED_TURNS = ENGAGEMENT_DEPTH_M / QD_THREAD_PITCH_M  # ~6.61 turns
# 1.25 rev/s. No real-world spec says how fast this fastener is driven, so
# this is a pacing choice; the turn count above is the part that is real.
SCREW_ANGULAR_SPEED_RAD_S = 2.5 * np.pi
# Ramp time to full speed = SCREW_ANGULAR_SPEED_RAD_S / this, i.e. ~2.5s.
# ScrewController's own 0.5s default is visibly abrupt on the arm.
SCREW_ANGULAR_ACCEL_RAD_S2 = np.pi

# The post-screw unwind can be faster than screwing: it is a cosmetic
# derotation of an empty tool, not constrained by any thread-engagement rate,
# and it never touches the arm's six joints (track_arm=False).
UNWIND_ANGULAR_SPEED_RAD_S = 2 * SCREW_ANGULAR_SPEED_RAD_S
# Acceleration, not the speed ceiling, is what actually shortens the unwind:
# the remainder is usually well under one turn, so the ramp is a pure
# triangular accel-limited profile that never reaches any ceiling. Raising
# only UNWIND_ANGULAR_SPEED_RAD_S makes no difference to the step count.
UNWIND_ANGULAR_ACCEL_RAD_S2 = 6 * SCREW_ANGULAR_ACCEL_RAD_S2

# Height every inter-task transition rises to before translating across.
# Deliberately equal to TRAY_HOVER_Z: any lower and the arm would translate
# below the tray hover, climb the last 100mm, and have the tracking descent
# immediately reverse it - and a vertical reversal has to pass through zero
# speed (from ~240mm/s to ~5mm/s at the apex). Matching the two heights makes
# every transit monotonic: rise, translate, descend.
SAFE_Z = 0.50

# --- Vision ---------------------------------------------------------------
# Both the QD and the port case fly to a coarse hover over the assembly area,
# then descend in steps re-detecting the real feature - a QD's hex flange, a
# manifold port hole - at each one. Both hovers use the ordinary
# GRIPPER_DOWN_QUAT: wrist_cam is mounted looking straight down at that same
# orientation, so no reorientation dance is needed.
#
# wrist_cam's mount has a real lateral offset, so even looking straight down
# its optical centre is not above the pinch site that the hover targets
# position. The offset is the same at every hover position, hover height and
# target Z (to within ~0.4mm) - a fixed property of the mount. Left
# unaccounted for, it skews every
# detection's framing: the port row sits ~69mm off the camera's true frame
# centre in Y, and port_4 alone shows a consistent ~3.2mm error. The hover
# targets below subtract it, so the camera's true optical centre lands on the
# feature centre.
WRIST_CAM_OPTICAL_OFFSET_XY_M = (0.0577, -0.0688)

_TRAY_CENTER_XY = (0.45, -0.20)  # average of QD_NEST_HARDCODED
TRAY_HOVER_XY = (_TRAY_CENTER_XY[0] - WRIST_CAM_OPTICAL_OFFSET_XY_M[0],
                  _TRAY_CENTER_XY[1] - WRIST_CAM_OPTICAL_OFFSET_XY_M[1])
# Height is an accuracy choice, not just framing. A QD's flange sits 35mm below
# its own shaft top, so seen off-vertical the shaft's bright circular top
# parallax-shifts over the flange and shadows part of it; the two merge into
# one asymmetric blob and pull detect_hexagons()' centroid off true centre,
# worst for the QDs furthest off-axis (~2-4mm at 0.40). Raising the camera to
# 0.50 shrinks that angle and roughly halves the mean error. Higher still
# (0.60+) doesn't work: adjacent QDs' contours merge into one oversized blob
# that the max-area filter throws out, losing 2 of 4 detections.
TRAY_HOVER_Z = 0.50
_MANIFOLD_CENTER_XY = (0.525, 0.175)  # average of PORTS_HARDCODED
MANIFOLD_HOVER_XY = (_MANIFOLD_CENTER_XY[0] - WRIST_CAM_OPTICAL_OFFSET_XY_M[0],
                      _MANIFOLD_CENTER_XY[1] - WRIST_CAM_OPTICAL_OFFSET_XY_M[1])
MANIFOLD_HOVER_Z = 0.30  # frames all 4 port holes with margin

# The world-Z planes vision ray-casts detected pixels onto. The scene doesn't
# leave these uncertain - QDs rest on a known floor, the manifold's top face is
# fixed - so only X/Y needs recovering from the image.
#
# This Z must be the height of the surface the camera actually sees. The
# ray-cast intersects the pixel's real camera ray, not a vertical line, so a
# wrong plane silently returns a wrong X/Y that grows with distance off-axis.
# Casting to QD_GRASP_Z (12.5mm below the flange top, for pad clearance), for
# example, would put ~2-4mm of error into every detection - which shows up
# downstream as a ~0.35deg grasp tilt and a visibly non-vertical screwing
# rotation. So detect_qds() casts to the flange's true top face and builds the
# grasp point at its own Z afterwards.
QD_FLANGE_TOP_Z = QD_NEST_HARDCODED["qd_1"][2] - 0.0349
QD_GRASP_Z = QD_NEST_HARDCODED["qd_1"][2] + GRASP_OFFSET[2]  # flange grasp height
PORT_Z = PORTS_HARDCODED[1][2]  # manifold top face height

# Closed-loop tracking descent - see CellSession.track_descend(). The floor
# heights are where detection stops being reliable, with margin: hexes are
# clean from Z=0.50 down to 0.25 and gone by 0.20 (the pixel area grows past
# detect_hexagons()' max_area_px); holes are clean and sub-mm from 0.30 to 0.18
# and gone by 0.15.
QD_TRACK_FLOOR_Z = 0.28
PORT_TRACK_FLOOR_Z = 0.20
QD_MATCH_RADIUS_M = 0.030   # under half the 60mm tray grid spacing
PORT_MATCH_RADIUS_M = 0.020  # under half the 45mm port row spacing

WIDTH, HEIGHT = 1920, 1440
# The vision renderer is deliberately smaller than the recorded video -
# smaller is faster, and it doesn't need to match. shape_detector.py's
# thresholds are calibrated against this resolution; keep them in sync.
VISION_WIDTH, VISION_HEIGHT = 960, 720
FPS = 30
PHYSICS_HZ = 500
STEPS_PER_FRAME = PHYSICS_HZ // FPS
GRIPPER_OPEN_RAMP_TICKS = 90   # 0.18s - a visible ramp, not an instant pop

# Cosmetic, third-person only: show the darker manifold overlay (geom group 5)
# and hide the vision-safe coloured one (group 1) for the recorded video and
# the dashboard's robot view. Group 2, where the arm and QDs live, is
# untouched - hiding that one instead would make the whole arm disappear.
#
# Hiding by group rather than occluding by a Z-nudge, because a vertical nudge
# doesn't move a sideways-facing surface any closer to the camera: it would
# look right from above and leave shiny slivers visible on the manifold's
# sides. Every vision render calls update_scene() with no scene_option, so it
# still sees group 1 - this swap cannot affect detection accuracy by
# construction, not just by testing.
THIRD_PERSON_SCENE_OPTION = mujoco.MjvOption()
THIRD_PERSON_SCENE_OPTION.geomgroup[1] = 0  # hide the vision-safe manifold color
THIRD_PERSON_SCENE_OPTION.geomgroup[5] = 1  # show the pretty overlay instead
