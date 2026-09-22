"""Publishes what the robot is doing into Map A, for the PLC to read.

Kept out of qd_cell/session.py so every Map A write lives in one place rather
than scattered through the motion code: the session calls a one-line hook
where it already has a value, and this module turns it into register writes -
units, sticky latches and phase codes included. None of it touches the
network; the Datastore it writes to is what the Modbus server serves.
"""

from __future__ import annotations

from .datastore import Datastore
from .gate import PhaseGate
from .tags import Phase


class SimSignals:
    """One object per session, holding the state the session's hooks update.

    The session calls these from the places it already computes the same
    values - check_self_collision(), check_leak_contacts(), the tracking
    descent - so each hook is a line or two, not a rewrite of the motion code.
    """

    def __init__(self, data: Datastore, gate: PhaseGate):
        self.data = data
        self.gate = gate
        self._last_self_coll_peak = 0
        self._last_leak_coll_peak = 0
        self._heartbeat = 0

    # -- the heartbeat: called from every stepping loop, and while held ------

    def tick(self) -> None:
        """Advance the heartbeat, and act on any command that cannot wait.

        The heartbeat the supervisor watches is SIM_TICK, a free-running
        count of calls that keeps advancing through gate waits and holds, and
        wraps at 16 bits - the PLC only asks whether it has changed. It is
        deliberately not the caller's own motion-tick index, which does NOT
        advance while parked at a gate, and a heartbeat that stalls at a gate
        is exactly the wrong signal for a comms watchdog.

        Nor is it a one-bit toggle, the obvious first choice, which aliases
        badly here: the sim renders a frame every STEPS_PER_FRAME (16) steps,
        that render dominates wall-clock time, and 16 is even - so a toggle
        has the same parity at every frame boundary and reads stuck to
        anything sampling on a slower clock. Measured against the real sim: 2
        of 40 polls saw it change. A counter cannot alias that way.
        """
        self._heartbeat = (self._heartbeat + 1) & 0xFFFF
        self.data.set_raw("SIM_TICK", self._heartbeat)
        self.gate.check_commands()

    # -- collisions: called after check_self_collision()/check_leak_contacts() --

    def note_self_collision_peak(self, peak: int) -> None:
        self.data.set("SELF_COLL_PEAK", peak)
        if peak > self._last_self_coll_peak:
            self.data.latch("SELF_COLLISION")
        self._last_self_coll_peak = peak

    def note_leak_contact_peak(self, peak: int) -> None:
        self.data.set("LEAK_CONTACT_PEAK", peak)
        if peak > self._last_leak_coll_peak:
            self.data.latch("LEAK_CONTACT")
        self._last_leak_coll_peak = peak

    # -- vision: called where the tracking descent logs its events -----------

    def vision_tracked(self) -> None:
        """A fresh detection unambiguously ends any coasting streak."""
        self.data.set_raw("VISION_COASTING", False)

    def vision_coasting(self) -> None:
        self.data.set_raw("VISION_COASTING", True)

    def vision_fallback(self) -> None:
        """A survey found nothing usable and fell back to the taught
        position. The session prints it; this makes it visible to the PLC."""
        self.data.latch("VISION_FALLBACK")
        self.data.set_raw("PART_DETECTED", False)

    def vision_part_detected(self, count: int) -> None:
        self.data.set_raw("PART_DETECTED", count > 0)

    # -- which step the robot is on ----------------------------------------
    #
    # A gate publishes PHASE_CODE when it is REACHED, and gates are sparse -
    # between G2 and G3 the arm picks the QD, surveys the manifold and flies
    # across to it, all of which would sit on "Pick QD" because no gate fires
    # until it arrives. The panel lagged a whole step behind the robot.
    #
    # Reporting and permission are different things. A gate is where the
    # supervisor DECIDES, and those stay exactly where they are. Where the
    # robot IS is something a real controller publishes continuously over the
    # fieldbus, and nothing about the handshake depends on it: the permit
    # compares against the gate's own remembered phase, and the supervisor
    # only reads PHASE_CODE for an interlock while GATED is set.
    #
    # set_phase() is already called at every phase boundary, for the per-tick
    # diagnostic log. Hanging this off it means there is one
    # list of phase names, not two that can drift.

    _SIMPLE = {
        "idle": Phase.WAIT_START,
        "screw_turn": Phase.SCREW_IN,
        "screw_decel": Phase.SCREW_DECEL,
        "post_screw_rise": Phase.POST_SCREW_RISE,
        "unwind": Phase.UNWIND,
        "fly_ready": Phase.FLY_READY,
        "leak_test": Phase.LEAK_PREP,
    }
    _SURVEY = {
        "tray": Phase.SURVEY_TRAY,
        "ports": Phase.SURVEY_PORTS,
        "manifold": Phase.LEAK_SURVEY,
    }
    _FSM = {
        "pick:APPROACH": Phase.PICK_APPROACH,
        "pick:DESCEND": Phase.PICK_DESCEND,
        "pick:GRASP": Phase.PICK_GRASP,
        "pick:LIFT": Phase.PICK_LIFT,
        "transport:TO_PORT_ABOVE": Phase.TRANSPORT_TO_PORT_ABOVE,
        "transport:DESCEND_TO_PRE_INSERT": Phase.TRANSPORT_DESCEND_PRE_INSERT,
    }
    #: keyed by (verb, is_the_manifold) - the same two verbs are used over
    #: the tray and over the manifold, and only the label says which
    _MOVE = {
        ("fly_hover", False): Phase.FLY_HOVER_TRAY,
        ("fly_hover", True): Phase.FLY_HOVER_PORT,
        ("track", False): Phase.TRACK_QD,
        ("track", True): Phase.TRACK_PORT,
    }

    @classmethod
    def phase_for_label(cls, label: str) -> Phase | None:
        """Map one of set_phase()'s labels to a Map A phase, or None if the
        label is not one the panel has a step for."""
        parts = label.split(":")
        head = parts[0]
        if head in ("fly_hover", "track"):
            at_manifold = len(parts) > 1 and parts[1].startswith("port")
            return cls._MOVE[(head, at_manifold)]
        if head == "survey":
            return cls._SURVEY.get(parts[1]) if len(parts) > 1 else None
        if head in ("pick", "transport"):
            return cls._FSM.get(label)
        return cls._SIMPLE.get(head)

    def publish_phase_label(self, label: str) -> None:
        phase = self.phase_for_label(label)
        if phase is not None:
            self.data.set("PHASE_CODE", int(phase))

    # -- the leak arm's own state -----------------------------------------
    #
    # LeakTestSequence runs its own FSM between G9 and G10 and never calls
    # set_phase() - so without this the column sat on whatever G9 published for the whole 36-second test, and "Retract" lit
    # only for the single scan G10 took, which is to say never. RETRACT is
    # 1.76s of real motion; it should look like it.
    #
    # Same split as everywhere else: the gates decide, this reports.

    LEAK_PHASES = {
        "TO_PORT_ABOVE": Phase.LEAK_TO_PORT_ABOVE,
        "HOVER_DETECT": Phase.LEAK_HOVER_DETECT,
        "ORBIT_SEAM": Phase.LEAK_ORBIT_SEAM,
        "RETRACT": Phase.LEAK_RETRACT,
        "DONE": Phase.LEAK_PORT_DONE,
    }

    def publish_leak_state(self, state: str) -> None:
        phase = self.LEAK_PHASES.get(state)
        if phase is not None:
            self.data.set("PHASE_CODE", int(phase))

    # -- insertion/seating -----------------------------------------------

    def publish_depth(self, thread_face_z: float, port_z: float,
                      engagement_depth_m: float) -> None:
        """How far the thread face has travelled into the port, clamped to
        [0, engagement_depth_m]. thread_face_z starts near
        port_z + engagement_depth_m (pre-insert) and descends toward port_z
        (fully seated) - the session's screw stop condition. Matches
        Map A's INSERT_DEPTH: 0.01mm units, 0..1200 for a 12.00mm target."""
        depth_m = engagement_depth_m - max(0.0, thread_face_z - port_z)
        depth_m = max(0.0, min(engagement_depth_m, depth_m))
        self.data.set("INSERT_DEPTH", depth_m * 1000.0)

    def publish_seated(self, seated: bool) -> None:
        self.data.set_raw("QD_SEATED", seated)

    def publish_seat_shortfall(self, shortfall: bool) -> None:
        if shortfall:
            self.data.latch("SEAT_SHORTFALL")

    def note_maxsteps_expired(self) -> None:
        self.data.latch("MAXSTEPS_EXPIRED")

    # -- cycle / session bookkeeping ----------------------------------------

    def note_qd_grasped(self, grasped: bool) -> None:
        self.data.set_raw("QD_GRASPED", grasped)

    def note_gripper_open(self, open_: bool) -> None:
        self.data.set_raw("GRIPPER_OPEN", open_)

    def note_cycle_complete(self, installed_count: int) -> None:
        self.data.set("QDS_INSTALLED", installed_count)

    def note_leak_port_complete(self, tested_count: int) -> None:
        self.data.set("LEAK_TESTED", tested_count)

    def note_session_done(self) -> None:
        self.data.set_raw("SESSION_DONE", True)

    def note_active_arm(self, leak: bool) -> None:
        self.data.set_raw("ACTIVE_ARM_LEAK", leak)

    def note_tilt(self, tilt_deg: float) -> None:
        self.data.set("TILT_DEG", tilt_deg)

    def note_track_diff(self, diff_mm: float) -> None:
        """Vision-tracked position vs the taught reference, in mm - the
        sim's own accuracy metric, which the PLC checks against its limit at
        gates G2 and G4."""
        self.data.set("TRACK_DIFF", diff_mm)

    def note_port_id(self, port_id: int) -> None:
        """Which port the current QD is going into, 0 for none yet."""
        self.data.set("PORT_ID", port_id)
