"""Phase gates: the points where the robot asks the PLC for permission.

The cell runs "sim always running, PLC gates phases". The simulation keeps
stepping physics and recording video the whole time; what the PLC controls is
whether it may *progress* to the next phase. A gated robot holds position - it
does not freeze, and the vision display stays live and responsive.

Two constraints shaped this, and both are easy to get wrong:

**A gate is a stop.** The motion is built throughout to avoid stop-then-go
(loose transit tolerances, `is_near_final` vs `is_converged`,
velocity-matched target handoff). So gates are placed only where the motion
already comes to rest or is deliberately frozen, never inside a loose handoff.
The gate list lives in qd_cell/session.py, not here.

**The sim can outrun the PLC.** Physics free-runs far faster than OpenPLC's
~100ms poll, so a whole phase can elapse between two polls. A permit left high
from the previous phase would therefore release the next one instantly and
invisibly. The guard is a monotonic sequence number: the PLC must echo back
the exact PHASE_SEQ it is permitting, so a stale permit matches nothing and is
refused rather than obeyed.
"""

from __future__ import annotations

import time
from typing import Callable

from .datastore import Datastore
from .tags import Phase


class SimAborted(Exception):
    """Raised inside a gate wait when the PLC commands an abort.

    The session catches this the same way it catches the operator quitting the
    live window: flush the video writer, close the display, exit cleanly.
    """


class PhaseGate:
    """Owns PHASE_CODE/PHASE_SEQ and the permit handshake for one sim run."""

    def __init__(self, data: Datastore, *, auto_permit: bool = False):
        self.data = data
        #: --no-plc-gating: release every gate locally, so the whole gated
        #: session can be run before any PLC exists.
        self.auto_permit = auto_permit
        self._seq = 0
        self._phase = Phase.IDLE

    # -- the handshake -----------------------------------------------------

    def gate(self, phase: Phase, pump: Callable[[], None], *,
             label: str = "") -> None:
        """Ask for permission to enter `phase`, pumping the sim until granted.

        `pump` advances one physics tick and renders when due. It is supplied
        by the caller, and the session's never drives the arm servo: every gate
        sits where the arm is already at rest, and at the gates around the
        post-screw release, stepping the servo visibly disturbs the QD that was
        just seated.
        """
        self._phase = phase
        self._seq = (self._seq + 1) & 0xFFFF
        self.data.set("PHASE_CODE", int(phase))
        self.data.set("PHASE_SEQ", self._seq)
        self.data.latch("PHASE_DONE")
        self.data.set("BUSY", False)
        self.data.set("GATED", True)

        if self.auto_permit:
            self._release()
            return

        timeout_s = self.data.get("GATE_TIMEOUT_S")
        started = time.monotonic()
        timed_out = False

        while True:
            pump()

            # A deliberate hold is not an overrun. The watchdog below measures
            # time spent waiting for a permit the supervisor intends to give;
            # while HOLD is asserted it intends not to, so the clock restarts
            # instead of running a deliberately-stopped cell into a fault.
            # Without this, pressing Stop and walking away for two minutes
            # latches GATE_TIMEOUT, the PLC faults, commands abort, and the
            # robot exits - the machine killing itself for doing as it was
            # told.
            if self.data.get("HOLD"):
                started = time.monotonic()
                timed_out = False

            waited = time.monotonic() - started

            if self.data.get("ABORT"):
                self.data.set("GATED", False)
                raise SimAborted(f"PLC commanded abort at {phase.name}"
                                 + (f" ({label})" if label else ""))

            if self._permitted():
                break

            # Sim-side safety net. It does not release the gate - releasing on
            # a timeout would defeat the interlock the gate exists for - it
            # raises a flag the PLC can fault on while the robot keeps waiting.
            if timeout_s and waited > timeout_s and not timed_out:
                self.data.latch("GATE_TIMEOUT")
                timed_out = True

        self._release()

    #: How far behind the current sequence a permit can be and still count as
    #: an ordinary leftover rather than a logic error. Generous, because the
    #: gap is bounded by how many gates the sim can pass in one PLC scan.
    STALE_WINDOW = 64

    def _permitted(self) -> bool:
        """Every condition for releasing the current gate.

        The sequence echo is what makes this safe against a sim that reaches
        the next gate inside a single PLC poll: a permit for the phase we just
        left does not match the one we are now asking about.

        A permit that is merely BEHIND the current sequence is an ordinary
        leftover - the supervisor granted the previous gate and has not yet
        seen that we moved on - and is ignored quietly. Only a permit that is
        ahead of us, or that names the wrong phase for the sequence it does
        claim, indicates the supervisor and the robot genuinely disagree about
        where the cell is. Latching on the benign case would fault a healthy
        cell every time it ran faster than the PLC polled.
        """
        if not self.data.get("RUN"):
            return False
        if self.data.get("HOLD"):
            return False
        if not self.data.get("PERMIT"):
            return False

        permit_seq = self.data.get("PERMIT_SEQ")
        permit_phase = self.data.get("PERMIT_PHASE")
        if permit_seq == self._seq and permit_phase == int(self._phase):
            return True

        behind = (self._seq - permit_seq) & 0xFFFF
        if 0 < behind <= self.STALE_WINDOW:
            return False          # ordinary leftover; wait for the real one

        self.data.latch("PERMIT_MISMATCH")
        return False

    def _release(self) -> None:
        self.data.set("GATED", False)
        self.data.set("BUSY", True)
        if self.data.get("PHASE_ACK"):
            self.data.set_raw("PHASE_DONE", False)

    # -- commands the sim honours between gates ----------------------------

    def check_commands(self) -> None:
        """Call once per tick. Handles the commands that must take effect
        without waiting for the next gate."""
        if self.data.get("ABORT"):
            raise SimAborted("PLC commanded abort")
        if self.data.get("CLEAR_STICKY"):
            self.data.clear_sticky()
