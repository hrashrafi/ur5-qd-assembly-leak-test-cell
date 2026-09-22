# PLC / HMI / Modbus supervisory layer

## What this is

A supervisory control layer bolted on top of the finished QD-manifold
simulation, structured the way a real industrial cell is:

```
   browser (operator page)            OpenCV window (vision display)
            |                                    ^
            | HTTP, polls every 300ms            | owned by the sim process
            v                                    |
   panel server (qd_plc/hmi)                     |
            |                                    |
            | Modbus TCP, every 250ms (Map B)    |
            v                                    |
   OpenPLC (Structured Text, 50ms scan)          |
            |                                    |
            | Modbus TCP, ~100ms poll (Map A)    |
            v                                    |
   MuJoCo simulation (qd_cell/session.py) -------+
```

Motion and vision stay entirely inside the simulation - the same way a real
robot controller keeps its own control loop off the fieldbus. Modbus carries
handshake and status only: phase codes, a gate-permit protocol, status bits
and a handful of scalar readings. No trajectory or image data crosses it.

## Mapping to real cell hardware

| This project | Real equivalent |
|---|---|
| MuJoCo sim + its internal motion/vision code | Robot controller (a UR control box) |
| Sim's Modbus TCP slave (`qd_plc/server.py`) | Robot controller's fieldbus interface |
| OpenPLC running Structured Text | The cell PLC (CompactLogix, S7-1200, ...) |
| Browser operator panel (`qd_plc/hmi/`) | Operator panel (PanelView, Comfort Panel) |
| The OpenCV live window (`qd_sim/live_dashboard.py`) | Vision system's own display (Cognex, Keyence) |

The vision display is a separate window from the operator panel, as it is on a
real floor: the panel carries control and status, and the vision controller has
its own screen.

## The three processes

| Process | Command | Listens | Talks to |
|---|---|---|---|
| Sim + vision display | `venv/bin/python scripts/run_cell.py --plc` | `:5020` Modbus | - |
| OpenPLC | `docker compose -f plc/docker-compose.yml up` | `:8080` web, `:502` Modbus | sim's `:5020` |
| Operator panel | `venv/bin/python scripts/run_hmi.py` | `:8000` HTTP | OpenPLC's `:502` |

Bring-up order: OpenPLC first (`docs/OPENPLC_SETUP.md`), then the panel, then
the sim last - the sim is what actually moves.

## Two Modbus links, two different register maps

**Map A** (`qd_plc.tags.MAP_A`) - the sim's slave, OpenPLC as master. The
machine interface: the gate-permit handshake, and the raw conditions and
measurements the PLC judges.
Documented in full in the generated `docs/PLC_TAGS.md`.

**Map B** (`qd_plc.tags.MAP_B`) - OpenPLC's own slave, the panel as master.
The supervisory view: cell state, the current step, progress counts, the
active fault and the operator's pushbuttons. Not a pass-through of Map A -
interpreted.

Why they differ: an IEC 61131 program cannot write `%IX`/`%IW` (those are
inputs, and in OpenPLC that address space is already claimed by the slave-
device/master mapping), so everything the PLC publishes to the panel has to
live in coils and holding registers, with a documented (not
platform-enforced) address-range split. `qd_plc/hmi/tagmap.py` is where that
split is actually enforced, on the panel's side.

## Sim faster than the PLC scan - the real problem, and how it's handled

The simulation free-runs at whatever speed the CPU allows; OpenPLC polls its
slave device on a fixed schedule (~100ms). A whole phase can complete between
two polls. Three mechanisms handle this together:

1. **Gates.** A gated sim physically cannot outrun the PLC - it waits at each
   of 12 points until permitted. See "The twelve gates" below.
2. **Sticky bits.** Transient conditions (`VISION_FALLBACK`,
   `SELF_COLLISION`, `MAXSTEPS_EXPIRED`, ...) latch set-only in the sim and
   are cleared only by an explicit `CLEAR_STICKY` command from the PLC. A
   condition that appears and clears between two polls still reaches the
   supervisor.
3. **A monotonic sequence number.** `PHASE_SEQ` increments on every phase
   entry. The PLC's permit must echo back the exact phase AND sequence it is
   permitting (`qd_plc/gate.py`'s `PhaseGate._permitted()`); a permit left
   over from the previous gate cannot release the wrong one. A permit merely
   *behind* the current sequence (an ordinary leftover, since the sim can
   reach the next gate before the PLC's write lands) is treated as benign and
   ignored quietly - only a permit *ahead* of the sim, or one naming the
   wrong phase for its sequence, is a genuine disagreement and gets reported
   as `PERMIT_MISMATCH`. The distinction matters: latching a fault on every
   ordinary leftover permit would make a cell running faster than its PLC
   polls fault continuously.

## The twelve gates

Gate placement rule: **a gate is a stop**, and the session's motion is
built throughout to avoid stop-then-go behaviour (loose transit
tolerances, `is_near_final` vs `is_converged`,
velocity-matched target handoff). So every gate in `qd_cell/session.py` sits
only where the motion has already come to rest (a `run_until()` blocked on
`is_converged`/`is_done` completed just before the gate) or is deliberately
frozen (the post-screw dead hold) - never inside a loose handoff.

| Gate | Phase | Robot state | The PLC's interlock (in `plc/st/cell_control.st`) |
|---|---|---|---|
| G0 | `WAIT_START` | ready keyframe | `RUN & SAFE_TO_START & SIM_READY & no fault` |
| G1 | `SURVEY_TRAY` | converged | running & no fault (common to every gate) |
| G2 | `TRACK_QD` | at tray hover | part detected, not a vision fallback, fewer than 4 installed, tracking error within limit |
| G3 | `TRACK_PORT` | at manifold hover | part grasped, part detected, tilt within limit |
| G4 | `TRANSPORT_DESCEND_PRE_INSERT` | tight-converged (0.002m) | grasped, tracking error within limit, no collision |
| **G5** | `GRIPPER_RELEASE` | **frozen** (60-tick dead hold) | **seated AND depth >= target - tolerance** |
| G6 | `POST_SCREW_RISE` | frozen, gripper open | gripper open, part released |
| G7 | `CYCLE_END` | at safe height, unwound | always (the cycle boundary) |
| G8 | `LEAK_PREP` | pick arm parked at ready | leak section armed, all 4 installed |
| G9 | `LEAK_SURVEY` | leak arm converged (0.002m/1deg) | still armed, no leak-arm contact |
| G10 | `LEAK_PORT_DONE` | after retract | still armed |
| G11 | `SESSION_COMPLETE` | parked | always |

**G5 is the interlock that matters most**: "the supervisor decides the part
is seated deep enough before the gripper is allowed to let go." It sits
exactly where the code already deliberately freezes the arm for a beat after
screwing - so the wait costs nothing physically, and the interlock is real
(the sim has no force sensing; depth-vs-target is the only signal available,
and it is exactly what a real cell's proximity/limit-switch seating check
would also be standing in for).

## What the sim exposes

A motion loop on its own hides most of what a supervisor needs: a
`run_until()` whose `max_steps` expires simply returns, a vision fallback is
just a `print()`, and collision counters only mean something once the run is
over. `qd_cell/session.py` publishes each of them:

- `run_until()` latches `MAXSTEPS_EXPIRED` on every expiry - otherwise a hung
  phase looks identical to a finished one to the caller.
- The tracking descent reports `VISION_COASTING`, and every survey that falls
  back to a taught position latches `VISION_FALLBACK`.
- `check_self_collision()`/`check_leak_contacts()` latch
  `SELF_COLLISION`/`LEAK_CONTACT` whenever their peak counter increases.
- `INSERT_DEPTH` is published continuously during screwing (computed from
  `thread_face_z()`, the same value the screw's own stop condition uses);
  `SEAT_SHORTFALL` is evaluated once, right before G5.

**No thresholds live in the sim.** It reports `TILT_DEG = 51`; the PLC
decides whether 51 (0.51 degrees) is within limits. That split - raw numbers
from the robot, judgement in the supervisor - is the entire point of the
architecture: the robot side can be swapped, replayed or stubbed without the
supervisor's logic changing at all. The limits themselves are named
constants at the top of `plc/st/cell_control.st`.

## Running the session

`qd_cell/session.py` is the robot half of the cell, and
`scripts/run_cell.py` is the short launcher that starts it. The supervisory hooks are
additive: the gates decide *when* the next phase may begin, and the gate pump
never calls `arm.step()`, so a gated run and an ungated run produce the same
motion. That is verified end to end - same
frame count, same self-collision and contact peaks, same 4/4 installed and
leak-tested, same 0.0758deg mean grasp tilt and 0.36mm worst tracking error,
and byte-identical `tracking_debug.csv` (157,921 rows) and `vision_events.csv`
(413 events). Not merely within tolerance: every physics tick, commanded
target and arm position matched exactly.

It inserts the repo root onto `sys.path` itself, so it can be launched from
anywhere:

```bash
# Normal operation - waits at G0 for the operator panel's START
venv/bin/python scripts/run_cell.py --plc

# Every gate releases locally - the full gated session with no PLC at all
venv/bin/python scripts/run_cell.py --plc --no-plc-gating

# Headless: skip the vision display window
venv/bin/python scripts/run_cell.py --plc --no-vision-display

# No supervisory layer at all - just run the session and write the video
venv/bin/python scripts/run_cell.py --out out/full_session.mp4
```

## Gotchas specific to this setup

See `docs/OPENPLC_SETUP.md` for the OpenPLC bring-up gotchas (libmodbus not
resolving hostnames, MATIEC's located/plain `VAR` block rule, the
three-function-block segfault, `hostnamectl` in a container). The ones
specific to the sim side:

- **The three loops that bypass `run_until()`** (60-tick post-screw dead
  hold, 90-tick gripper-open ramp, 800-tick ready-pose snap) must still
  publish the heartbeat, or `SIM_TICK` stops advancing long enough to approach
  the PLC's comms-loss watchdog. All three step through `untouched_tick()`,
  which does.
- **Gate holds never call `arm.step()`.** `_gate_pump()` only calls
  `tick_once()` (which steps physics with whatever ctrl is already set) plus
  periodic frame recording - never anything that would move the arm's
  target. This matters most at G5/G6: the 60-tick dead hold's own comment
  documents that stepping the servo there visibly disturbs the just-seated
  QD.
- **Only OpenPLC writes Map A; only the operator panel writes Map B's
  pushbutton coils.** Writing to the sim's registers directly
  while the PLC is running produces a permit race indistinguishable from a
  PLC bug - stop the PLC (or use `--no-plc-gating`) first.
