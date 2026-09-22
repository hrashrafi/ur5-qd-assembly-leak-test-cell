"""THE register maps - single source of truth for the whole project.

Everything that needs to agree about an address, a scale factor or an
enumeration reads it from here: the robot's publisher (qd_plc.signals), the
operator panel's backend, the OpenPLC slave-device table, the generated
Structured Text VAR block and the generated docs. Nothing else is allowed to
hard-code an address. That matters
more than usual here because OpenPLC derives its %-addresses from the ORDER
and SIZE of the slave-device fields you type into its web UI - resize one
block and every address after it shifts silently.

TWO MAPS, deliberately different:

  MAP A  sim <-> PLC     the machine interface: the gate handshake, and the
                         raw conditions and measurements the PLC judges.
                         OpenPLC is master.
  MAP B  PLC <-> HMI     the supervisory view: cell state, progress, the
                         active fault, operator pushbuttons. HMI is master.

A note on why Map B puts PLC->HMI status in COILS rather than discrete
inputs, which looks wrong to anyone used to real PLCs: an IEC 61131 program
cannot write to %IX or %IW - they are inputs, driven by the hardware. In
OpenPLC the %IX/%IW space is already taken by the master (slave-device)
mapping. So status the program computes has to live somewhere the program can
write, which means coils and holding registers, with a documented ownership
split instead of a hardware-enforced one. This is a real constraint of the
platform, not a shortcut.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


# --------------------------------------------------------------------------
# Tag model
# --------------------------------------------------------------------------

class Kind:
    """Modbus object type. The names match the classic 0x/1x/3x/4x tables."""

    COIL = "co"            # 0x  read/write bit   - commands
    DISCRETE_INPUT = "di"  # 1x  read-only bit    - status
    INPUT_REG = "ir"       # 3x  read-only word   - measured values
    HOLDING_REG = "hr"     # 4x  read/write word  - setpoints

    BITS = (COIL, DISCRETE_INPUT)


class DType:
    BOOL = "bool"
    UINT16 = "uint16"
    INT16 = "int16"
    ENUM = "enum"


@dataclass(frozen=True, slots=True)
class Tag:
    """One address in one map.

    scale is the engineering value of one raw count, so
    engineering = raw * scale (0.01 for hundredths, 1.0 for plain counts).
    sticky marks a bit the sim latches set-only: it is never cleared by the
    sim, only by an explicit CLEAR_STICKY command, so a condition cannot
    appear and vanish between two PLC polls.
    """

    name: str
    addr: int
    kind: str
    dtype: str = DType.UINT16
    scale: float = 1.0
    unit: str = ""
    desc: str = ""
    sticky: bool = False
    enum: str | None = None

    @property
    def is_bit(self) -> bool:
        return self.kind in Kind.BITS


@dataclass
class TagMap:
    """One side of one Modbus link."""

    name: str
    description: str
    tags: list[Tag] = field(default_factory=list)
    #: Declared block sizes per kind. These are DECLARED, not inferred from
    #: the highest address in use, so that adding a tag into the spare space
    #: later does not resize a block - which in OpenPLC would silently shift
    #: every %-address after it and quietly break a working configuration.
    sizes: dict[str, int] = field(default_factory=dict)
    _by_name: dict[str, Tag] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        seen: dict[tuple[str, int], str] = {}
        for tag in self.tags:
            key = (tag.kind, tag.addr)
            if key in seen:
                raise ValueError(
                    f"{self.name}: {tag.kind} address {tag.addr} used by both "
                    f"{seen[key]} and {tag.name}"
                )
            seen[key] = tag.name
            if tag.name in self._by_name:
                raise ValueError(f"{self.name}: duplicate tag name {tag.name}")
            self._by_name[tag.name] = tag
        for kind, size in self.sizes.items():
            used = [t for t in self.tags if t.kind == kind]
            if used and max(t.addr for t in used) >= size:
                raise ValueError(
                    f"{self.name}: {kind} block declared size {size} but "
                    f"address {max(t.addr for t in used)} is in use"
                )

    def __getitem__(self, name: str) -> Tag:
        try:
            return self._by_name[name]
        except KeyError:
            raise KeyError(
                f"{self.name} has no tag {name!r}. Known: "
                + ", ".join(sorted(self._by_name))
            ) from None

    def of_kind(self, kind: str) -> list[Tag]:
        return sorted((t for t in self.tags if t.kind == kind), key=lambda t: t.addr)

    def sticky_bits(self) -> list[Tag]:
        return [t for t in self.tags if t.sticky]

    def size(self, kind: str) -> int:
        """The declared block size - how many addresses of this kind exist."""
        return self.sizes[kind]

    def spare(self, kind: str) -> list[int]:
        """Unused addresses inside the declared block - the room to grow."""
        used = {t.addr for t in self.of_kind(kind)}
        return [a for a in range(self.size(kind)) if a not in used]


# --------------------------------------------------------------------------
# Enumerations carried in registers
# --------------------------------------------------------------------------

class Phase(IntEnum):
    """PHASE_CODE values: which step the robot is on.

    Decade-aligned on purpose, so a code reads as its stage at a glance: 10s
    the tray, 20s the pick, 30s the manifold survey, 40s transport, 50s the
    screw cycle, 60s back to ready, 70s the leak test.
    """

    IDLE = 0
    WAIT_START = 1

    SURVEY_TRAY = 10
    FLY_HOVER_TRAY = 11
    TRACK_QD = 12

    PICK_APPROACH = 20
    PICK_DESCEND = 21
    PICK_GRASP = 22
    PICK_LIFT = 23

    SURVEY_PORTS = 30
    FLY_HOVER_PORT = 31
    TRACK_PORT = 32

    TRANSPORT_TO_PORT_ABOVE = 40
    TRANSPORT_DESCEND_PRE_INSERT = 41

    SCREW_IN = 50
    SCREW_DECEL = 51
    GRIPPER_RELEASE = 53
    POST_SCREW_RISE = 55
    UNWIND = 56
    CYCLE_END = 57

    FLY_READY = 60

    LEAK_PREP = 70
    LEAK_SURVEY = 72
    LEAK_TO_PORT_ABOVE = 73
    LEAK_HOVER_DETECT = 74
    LEAK_ORBIT_SEAM = 75
    LEAK_RETRACT = 76
    LEAK_PORT_DONE = 77

    SESSION_COMPLETE = 90


#: Operator-facing text for each phase. The PLC never sends strings - ST has
#: no practical string support and OpenPLC's is worse - so the wire carries
#: the code and the HMI owns the words.
PHASE_TEXT: dict[int, str] = {
    Phase.IDLE: "Idle",
    Phase.WAIT_START: "Waiting for start",
    Phase.SURVEY_TRAY: "Surveying tray",
    Phase.FLY_HOVER_TRAY: "Moving to tray",
    Phase.TRACK_QD: "Tracking QD",
    Phase.PICK_APPROACH: "Approaching QD",
    Phase.PICK_DESCEND: "Descending to QD",
    Phase.PICK_GRASP: "Grasping",
    Phase.PICK_LIFT: "Lifting",
    Phase.SURVEY_PORTS: "Surveying ports",
    Phase.FLY_HOVER_PORT: "Moving to manifold",
    Phase.TRACK_PORT: "Tracking port",
    Phase.TRANSPORT_TO_PORT_ABOVE: "Moving above port",
    Phase.TRANSPORT_DESCEND_PRE_INSERT: "Descending to pre-insert",
    Phase.SCREW_IN: "Screwing in",
    Phase.SCREW_DECEL: "Stopping rotation",
    Phase.GRIPPER_RELEASE: "Opening gripper",
    Phase.POST_SCREW_RISE: "Retracting",
    Phase.UNWIND: "Unwinding tool",
    Phase.CYCLE_END: "Cycle complete",
    Phase.FLY_READY: "Returning to ready",
    Phase.LEAK_PREP: "Preparing leak test",
    Phase.LEAK_SURVEY: "Surveying flanges",
    Phase.LEAK_TO_PORT_ABOVE: "Moving above flange",
    Phase.LEAK_HOVER_DETECT: "Detecting seam",
    Phase.LEAK_ORBIT_SEAM: "Tracing seam",
    Phase.LEAK_RETRACT: "Retracting probe",
    Phase.LEAK_PORT_DONE: "Port tested",
    Phase.SESSION_COMPLETE: "Session complete",
}

#: The operator-panel SEQUENCE column: the cell's steps as a human thinks of
#: them, and which phases light each one. Order is display order.
#: The two arms run two different jobs, so the panel shows two step lists.
#: The pick arm installs four QDs; the leak arm then traces four seams. Each
#: list is the operator's view of its own arm - which is why a phase belongs
#: to exactly one of them.
ASSEMBLY_STEPS: list[tuple[str, tuple[Phase, ...]]] = [
    ("Survey tray", (Phase.SURVEY_TRAY, Phase.FLY_HOVER_TRAY)),
    ("Pick QD", (Phase.TRACK_QD, Phase.PICK_APPROACH, Phase.PICK_DESCEND,
                 Phase.PICK_GRASP, Phase.PICK_LIFT)),
    # TRACK_PORT belongs here rather than with the insert: G3 fires with the
    # arm hovering over the manifold having just surveyed it, so that is the
    # gate this step owns. Without it this step would have no gate at all and
    # could never light.
    ("Survey ports", (Phase.SURVEY_PORTS, Phase.FLY_HOVER_PORT,
                      Phase.TRACK_PORT)),
    ("Insert / screw", (Phase.TRANSPORT_TO_PORT_ABOVE,
                        Phase.TRANSPORT_DESCEND_PRE_INSERT, Phase.SCREW_IN,
                        Phase.SCREW_DECEL)),
    ("Release", (Phase.GRIPPER_RELEASE,)),
    ("Retract", (Phase.POST_SCREW_RISE, Phase.UNWIND, Phase.CYCLE_END,
                 Phase.FLY_READY)),
]

#: The leak arm's per-port cycle, repeated for each of the four ports:
#: survey the manifold and pick the next untested flange, trace its seam,
#: lift clear. G8 and G9 land on the first step, G10 on the last.
LEAK_STEPS: list[tuple[str, tuple[Phase, ...]]] = [
    ("Survey ports", (Phase.LEAK_PREP, Phase.LEAK_SURVEY)),
    ("Test port", (Phase.LEAK_TO_PORT_ABOVE, Phase.LEAK_HOVER_DETECT,
                   Phase.LEAK_ORBIT_SEAM)),
    ("Retract", (Phase.LEAK_RETRACT, Phase.LEAK_PORT_DONE)),
]


class CellState(IntEnum):
    """Map B CELL_STATE - what the PLC thinks the cell is doing."""

    IDLE = 0
    STARTING = 1
    RUNNING = 2
    HOLD = 3
    FAULTING = 4      # fault latched, sim still parking itself
    FAULTED = 5       # sim has parked, awaiting reset
    COMPLETE = 6
    ABORTING = 7


CELL_STATE_TEXT = {
    CellState.IDLE: "IDLE",
    CellState.STARTING: "STARTING",
    CellState.RUNNING: "RUNNING",
    CellState.HOLD: "HOLD",
    CellState.FAULTING: "FAULTING",
    CellState.FAULTED: "FAULTED",
    CellState.COMPLETE: "COMPLETE",
    CellState.ABORTING: "ABORTING",
}


class Fault(IntEnum):
    """Fault codes. The sim reports raw conditions; the PLC decides which of
    them is a fault and latches one of these."""

    NONE = 0
    SIM_COMM_LOSS = 1
    GATE_TIMEOUT = 2
    PHASE_WATCHDOG = 3
    MAXSTEPS_EXPIRED = 4
    VISION_FALLBACK = 5
    VISION_COAST_EXCESS = 6
    SELF_COLLISION = 7
    LEAK_ARM_CONTACT = 8
    SEAT_SHORTFALL = 9
    PERMIT_MISMATCH = 14
    LIVE_WINDOW_CLOSED = 17


FAULT_TEXT: dict[int, str] = {
    Fault.NONE: "",
    Fault.SIM_COMM_LOSS: "Lost communication with robot controller",
    Fault.GATE_TIMEOUT: "Robot did not release from gate within timeout",
    Fault.PHASE_WATCHDOG: "Step exceeded its maximum time",
    Fault.MAXSTEPS_EXPIRED: "Motion step hit its step limit without completing",
    Fault.VISION_FALLBACK: "Vision failed; fell back to taught position",
    Fault.VISION_COAST_EXCESS: "Vision lost for too long during tracking",
    Fault.SELF_COLLISION: "Arm self-collision detected",
    Fault.LEAK_ARM_CONTACT: "Leak-test arm contacted a part",
    Fault.SEAT_SHORTFALL: "Part not seated to depth - release refused",
    Fault.PERMIT_MISMATCH: "Permit did not match the requested step",
    Fault.LIVE_WINDOW_CLOSED: "Vision display was closed",
}


# --------------------------------------------------------------------------
# MAP A - the sim's Modbus slave. OpenPLC is the master.
# --------------------------------------------------------------------------
# Handshake and status only. No trajectory, no joint angles, no images: the
# robot controller keeps those to itself, exactly as a real one does.

_A = [
    # --- discrete inputs: sim -> PLC status bits ---------------------------
    Tag("SIM_READY", 1, Kind.DISCRETE_INPUT, DType.BOOL,
        desc="Model loaded, server up, ready to be started"),
    Tag("BUSY", 2, Kind.DISCRETE_INPUT, DType.BOOL,
        desc="Executing a phase (not sitting at a gate)"),
    Tag("GATED", 3, Kind.DISCRETE_INPUT, DType.BOOL,
        desc="Parked at a gate, waiting for PERMIT"),
    Tag("PHASE_DONE", 4, Kind.DISCRETE_INPUT, DType.BOOL, sticky=True,
        desc="Set on reaching any gate; cleared by PHASE_ACK"),
    Tag("SESSION_DONE", 7, Kind.DISCRETE_INPUT, DType.BOOL,
        desc="All QDs installed and all ports leak-tested"),
    Tag("QD_GRASPED", 8, Kind.DISCRETE_INPUT, DType.BOOL,
        desc="Part is held (kinematic weld engaged)"),
    Tag("QD_SEATED", 9, Kind.DISCRETE_INPUT, DType.BOOL,
        desc="Thread face has reached port depth right now"),
    Tag("GRIPPER_OPEN", 10, Kind.DISCRETE_INPUT, DType.BOOL),
    Tag("PART_DETECTED", 11, Kind.DISCRETE_INPUT, DType.BOOL,
        desc="Last survey returned at least one candidate"),
    Tag("VISION_FALLBACK", 12, Kind.DISCRETE_INPUT, DType.BOOL, sticky=True,
        desc="A survey gave up and used the taught position"),
    Tag("VISION_COASTING", 13, Kind.DISCRETE_INPUT, DType.BOOL,
        desc="Tracking with no current detection"),
    Tag("SELF_COLLISION", 15, Kind.DISCRETE_INPUT, DType.BOOL, sticky=True),
    Tag("LEAK_CONTACT", 16, Kind.DISCRETE_INPUT, DType.BOOL, sticky=True,
        desc="Leak arm touched a QD or the manifold"),
    Tag("MAXSTEPS_EXPIRED", 17, Kind.DISCRETE_INPUT, DType.BOOL, sticky=True,
        desc="A motion step hit its step cap - the otherwise silent hang"),
    Tag("SEAT_SHORTFALL", 19, Kind.DISCRETE_INPUT, DType.BOOL, sticky=True,
        desc="Screw phase ended short of the depth target"),
    Tag("ACTIVE_ARM_LEAK", 20, Kind.DISCRETE_INPUT, DType.BOOL,
        desc="0 = pick arm is active, 1 = leak arm is active"),
    Tag("ABORT_COMPLETE", 21, Kind.DISCRETE_INPUT, DType.BOOL,
        desc="Parked and video flushed after an ABORT"),
    Tag("SIM_STOPPED", 22, Kind.DISCRETE_INPUT, DType.BOOL),
    Tag("LIVE_WINDOW_OPEN", 23, Kind.DISCRETE_INPUT, DType.BOOL,
        desc="Vision display alive; 0 means the operator closed it"),
    Tag("PERMIT_MISMATCH", 24, Kind.DISCRETE_INPUT, DType.BOOL, sticky=True,
        desc="A permit arrived for a stale phase/sequence and was refused"),
    Tag("GATE_TIMEOUT", 25, Kind.DISCRETE_INPUT, DType.BOOL, sticky=True,
        desc="Sim-side safety: waited longer than GATE_TIMEOUT_S at a gate"),

    # --- coils: PLC -> sim commands ---------------------------------------
    # All LEVEL-sensed, never pulses. The sim can free-run far faster than
    # the PLC polls, so a one-scan pulse could be written and cleared between
    # two reads and never be seen. Edge detection lives in the PLC.
    Tag("PERMIT", 0, Kind.COIL, DType.BOOL,
        desc="Release the currently gated phase (validated against HR0/HR1)"),
    Tag("PHASE_ACK", 1, Kind.COIL, DType.BOOL, desc="Clears PHASE_DONE"),
    Tag("RUN", 2, Kind.COIL, DType.BOOL,
        desc="Master enable; dropping it holds the sim at the next gate"),
    Tag("HOLD", 3, Kind.COIL, DType.BOOL,
        desc="Freeze where it stands, mid-motion; dropping it resumes"),
    Tag("ABORT", 4, Kind.COIL, DType.BOOL,
        desc="Graceful: finish the tick, park, flush video, set ABORT_COMPLETE"),
    Tag("CLEAR_STICKY", 5, Kind.COIL, DType.BOOL,
        desc="The only way sticky status bits are ever cleared"),

    # --- input registers: sim -> PLC values --------------------------------
    Tag("PHASE_CODE", 0, Kind.INPUT_REG, DType.ENUM, enum="Phase"),
    Tag("PHASE_SEQ", 1, Kind.INPUT_REG, DType.UINT16,
        desc="Increments at every gate; the permit must echo it back"),
    Tag("PORT_ID", 5, Kind.INPUT_REG, DType.UINT16, desc="1..4, 0 = none"),
    Tag("INSERT_DEPTH", 6, Kind.INPUT_REG, DType.UINT16, 0.01, "mm",
        "How far the thread face has entered the port"),
    Tag("TILT_DEG", 9, Kind.INPUT_REG, DType.INT16, 0.01, "deg",
        "Grasp tilt from vertical - a genuine quality metric"),
    Tag("TRACK_DIFF", 10, Kind.INPUT_REG, DType.UINT16, 0.01, "mm",
        "Vision-tracked position vs the taught reference"),
    Tag("SELF_COLL_PEAK", 13, Kind.INPUT_REG, DType.UINT16),
    Tag("LEAK_CONTACT_PEAK", 14, Kind.INPUT_REG, DType.UINT16),
    Tag("SIM_TICK", 19, Kind.INPUT_REG, DType.UINT16, unit="ticks",
        desc="Free-running step counter, wraps at 65535; the comms heartbeat"),
    Tag("QDS_INSTALLED", 22, Kind.INPUT_REG, DType.UINT16),
    Tag("LEAK_TESTED", 23, Kind.INPUT_REG, DType.UINT16),

    # --- holding registers: PLC -> sim setpoints ---------------------------
    Tag("PERMIT_PHASE", 0, Kind.HOLDING_REG, DType.ENUM, enum="Phase",
        desc="Which phase the PLC believes it is permitting"),
    Tag("PERMIT_SEQ", 1, Kind.HOLDING_REG, DType.UINT16,
        desc="The anti-stale guard: must equal PHASE_SEQ or PERMIT is refused"),
    Tag("GATE_TIMEOUT_S", 7, Kind.HOLDING_REG, DType.UINT16, 1.0, "s",
        "0 = wait forever"),
]

MAP_A = TagMap(
    name="sim",
    description="Sim-side Modbus slave (OpenPLC polls this). The machine "
                "interface: gate handshake, status bits, measured values.",
    tags=_A,
    # Declared sizes, larger than the addresses in use: the OpenPLC
    # slave-device table is typed from these, so the spare space lets a tag be
    # added without re-entering that configuration.
    sizes={
        Kind.DISCRETE_INPUT: 32,
        Kind.COIL: 16,
        Kind.INPUT_REG: 32,
        Kind.HOLDING_REG: 16,
    },
)


# --------------------------------------------------------------------------
# MAP B - the PLC's Modbus slave. The HMI is the master.
# --------------------------------------------------------------------------
# The supervisory view, not a pass-through of Map A. Ownership split, since
# the platform cannot enforce it (see the module docstring):
#
#   coils   0..15   PLC writes, HMI reads     status lamps
#   coils  80..95   HMI writes, PLC consumes  pushbuttons
#   regs    0..35   PLC writes, HMI reads     readouts
#
# Anything at or above coil 800 / register 100 belongs to the sim link and is
# out of bounds for the HMI; qd_plc.hmi.tagmap enforces that at runtime.

#: First %-address owned by the OpenPLC master mapping. The HMI must stay below.
PLC_RESERVED_REG = 100
PLC_RESERVED_COIL = 100 * 8

_B = [
    # --- status coils: PLC -> HMI -----------------------------------------
    Tag("SIM_COMM_OK", 6, Kind.COIL, DType.BOOL,
        desc="The robot's heartbeat is advancing"),
    Tag("LEAK_PHASE_ACTIVE", 10, Kind.COIL, DType.BOOL,
        desc="The leak arm has the cell"),
    Tag("LEAK_ARMED", 11, Kind.COIL, DType.BOOL,
        desc="Operator has started the leak section; permits G8-G10"),
    Tag("SAFE_TO_START", 12, Kind.COIL, DType.BOOL,
        desc="Every interlock for a fresh start is satisfied"),

    # --- command coils: HMI -> PLC (momentary pushbuttons) -----------------
    # The PLC edge-detects each one and writes it back to 0 once consumed,
    # which is the standard OpenPLC pushbutton pattern.
    Tag("HMI_START", 80, Kind.COIL, DType.BOOL),
    Tag("HMI_STOP", 81, Kind.COIL, DType.BOOL),
    Tag("HMI_RESET", 82, Kind.COIL, DType.BOOL),
    Tag("HMI_START_LEAK", 89, Kind.COIL, DType.BOOL,
        desc="Momentary: arm the leak section"),
    Tag("HMI_STOP_LEAK", 90, Kind.COIL, DType.BOOL,
        desc="Momentary: disarm it, or stop the leak arm if it is moving"),

    # --- status registers: PLC -> HMI -------------------------------------
    Tag("CELL_STATE", 0, Kind.HOLDING_REG, DType.ENUM, enum="CellState"),
    Tag("CURRENT_STEP", 1, Kind.HOLDING_REG, DType.ENUM, enum="Phase"),
    Tag("ACTIVE_FAULT_CODE", 15, Kind.HOLDING_REG, DType.ENUM, enum="Fault"),
    Tag("LEAK_TESTED", 25, Kind.HOLDING_REG, DType.UINT16),
    Tag("QDS_INSTALLED", 32, Kind.HOLDING_REG, DType.UINT16),
]

MAP_B = TagMap(
    name="plc",
    description="PLC-side Modbus slave (the HMI polls this). The supervisory "
                "view: cell state, progress, the active fault, pushbuttons.",
    tags=_B,
    sizes={
        Kind.COIL: 96,          # 0..15 status, 80..95 commands
        Kind.HOLDING_REG: 36,   # 0..35 status
    },
)


# --------------------------------------------------------------------------
# OpenPLC %-address derivation
# --------------------------------------------------------------------------
# OpenPLC assigns located-variable addresses to a slave device from the ORDER
# and SIZE of the fields you type into its web UI. Change a size and every
# address after it moves. So the config table you type in and the ST VAR
# block you compile are generated together, from here, and cannot disagree.

#: OpenPLC starts master-mapped addresses at 100.
MASTER_BASE = 100

#: The order OpenPLC lists the slave-device fields in, and the kind each maps
#: to. Holding-read is deliberately sized 0: everything the PLC reads from the
#: sim is an input register already, and a second copy would be a second
#: source of truth for the same words.
OPENPLC_FIELD_ORDER = [
    ("Discrete Inputs", Kind.DISCRETE_INPUT, "%IX"),
    ("Coils", Kind.COIL, "%QX"),
    ("Input Registers", Kind.INPUT_REG, "%IW"),
    ("Holding Registers - Read", None, "%IW"),
    ("Holding Registers - Write", Kind.HOLDING_REG, "%QW"),
]


def openplc_address(tag: Tag, *, master: bool) -> str:
    """The IEC located-variable address for a tag.

    master=True  -> Map A, reached through OpenPLC's slave-device mapping,
                    which is offset by MASTER_BASE.
    master=False -> Map B, served by OpenPLC's own Modbus server, where the
                    Modbus address is the %-address directly.
    """
    base = MASTER_BASE if master else 0
    if tag.is_bit:
        prefix = "%IX" if tag.kind == Kind.DISCRETE_INPUT else "%QX"
        byte, bit = divmod(tag.addr, 8)
        return f"{prefix}{base + byte}.{bit}"
    prefix = "%IW" if tag.kind == Kind.INPUT_REG else "%QW"
    return f"{prefix}{base + tag.addr}"


def openplc_slave_config() -> list[tuple[str, int, int, str]]:
    """The slave-device table to type into OpenPLC, in its field order.

    Returns (field label, start address, size, resulting %-range).
    """
    rows = []
    for label, kind, prefix in OPENPLC_FIELD_ORDER:
        if kind is None:
            rows.append((label, 0, 0, "unused - leave zero"))
            continue
        size = MAP_A.size(kind)
        if kind in Kind.BITS:
            last_byte, last_bit = divmod(size - 1, 8)
            span = (f"{prefix}{MASTER_BASE}.0 - "
                    f"{prefix}{MASTER_BASE + last_byte}.{last_bit}")
        else:
            span = f"{prefix}{MASTER_BASE} - {prefix}{MASTER_BASE + size - 1}"
        rows.append((label, 0, size, span))
    return rows


_ST_TYPE = {
    DType.BOOL: "BOOL",
    DType.UINT16: "UINT",
    DType.INT16: "INT",
    DType.ENUM: "UINT",
}


def st_var_block() -> str:
    """The ST VAR declarations for both maps, addresses derived not typed.

    Signedness matters here and is easy to get wrong by hand: SIM_TICK counts
    up to 65535, which overflows INT, while TILT_DEG genuinely goes negative
    and must not be UINT.
    """
    lines = ["VAR"]
    for title, tmap, master in (
        ("Map A - robot controller (via slave-device mapping)", MAP_A, True),
        ("Map B - operator panel (OpenPLC's own Modbus server)", MAP_B, False),
    ):
        lines.append(f"    (* ---- {title} ---- *)")
        for kind in (Kind.DISCRETE_INPUT, Kind.COIL, Kind.INPUT_REG,
                     Kind.HOLDING_REG):
            tags = tmap.of_kind(kind)
            if not tags:
                continue
            for tag in tags:
                addr = openplc_address(tag, master=master)
                st_type = _ST_TYPE[tag.dtype]
                prefix = "a_" if master else "b_"
                name = f"{prefix}{tag.name.lower()}"
                comment = f"  (* {tag.desc} *)" if tag.desc else ""
                lines.append(
                    f"    {name:<28} AT {addr:<10} : {st_type};{comment}"
                )
        lines.append("")
    lines.append("END_VAR")
    return "\n".join(lines)
