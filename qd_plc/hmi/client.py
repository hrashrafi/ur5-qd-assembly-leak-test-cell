"""One persistent Modbus client polling the PLC for the whole panel.

Every open browser tab shares this, so the PLC sees one poll at a fixed rate
regardless of how many people are watching. That matters more than it sounds:
OpenPLC's master polls its slave devices synchronously inside the scan, so
piling concurrent requests onto its server stretches the scan that supervises
the robot.

Reads are whole blocks, not individual tags - two requests per cycle however
many tags the panel shows.
"""

from __future__ import annotations

import threading

from pymodbus.client import ModbusTcpClient

from ..tags import (ASSEMBLY_STEPS, CELL_STATE_TEXT, FAULT_TEXT, LEAK_STEPS,
                    MAP_B, CellState, Fault, Kind, Phase)
from .tagmap import check_writable


class PlcClient:
    """Polls Map B in the background; hands out the latest snapshot."""

    def __init__(self, host: str = "127.0.0.1", port: int = 502,
                 device_id: int = 1, interval: float = 0.25):
        self.host = host
        self.port = port
        self.device_id = device_id
        self.interval = interval
        self._client = ModbusTcpClient(host, port=port, timeout=2.0)
        self._lock = threading.Lock()
        self._values: dict[str, int | bool] = {}
        self._connected = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="hmi-poll",
                                        daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)
        self._client.close()

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                if not self._client.connected and not self._client.connect():
                    self._connected = False
                    continue
                values = {}
                for kind in (Kind.COIL, Kind.HOLDING_REG):
                    size = MAP_B.size(kind)
                    raw = self._read_block(kind, size)
                    if raw is None:
                        raise ConnectionError(f"{kind} block read failed")
                    for tag in MAP_B.of_kind(kind):
                        if tag.addr < len(raw):
                            values[tag.name] = raw[tag.addr]
                with self._lock:
                    self._values = values
                    self._connected = True
            except Exception:
                with self._lock:
                    self._connected = False
                self._client.close()

    def _read_block(self, kind: str, size: int):
        fn = (self._client.read_coils if kind == Kind.COIL
              else self._client.read_holding_registers)
        rr = fn(0, count=size, device_id=self.device_id)
        if rr.isError():
            return None
        return list(rr.bits[:size]) if kind == Kind.COIL else list(rr.registers)

    # -- writes ------------------------------------------------------------

    def command(self, name: str, value) -> None:
        """Press a panel pushbutton. Refuses anything the PLC publishes."""
        tag = check_writable(name)
        self._client.write_coil(tag.addr, bool(value), device_id=self.device_id)

    # -- the panel's view --------------------------------------------------

    def snapshot(self) -> dict:
        """Everything the page needs, in one object, with the text resolved.

        The PLC never sends strings - ST has no practical string support - so
        the wire carries codes and the words are attached here.
        """
        with self._lock:
            raw = dict(self._values)
            connected = self._connected

        def num(name):
            return int(raw.get(name, 0) or 0)

        step_code = num("CURRENT_STEP")
        fault_code = num("ACTIVE_FAULT_CODE")
        state_code = num("CELL_STATE")

        try:
            state_name = CELL_STATE_TEXT.get(CellState(state_code), "UNKNOWN")
        except ValueError:
            state_name = "UNKNOWN"

        step = self._step_index
        asm_i = step(ASSEMBLY_STEPS, step_code)
        leak_i = step(LEAK_STEPS, step_code)
        installed = num("QDS_INSTALLED")
        tested = num("LEAK_TESTED")
        running = state_code == int(CellState.RUNNING)
        # The leak arm becomes "active" only once G8 has actually been
        # permitted - the robot publishes LEAK_PREP the moment it PARKS at
        # G8, which is before the operator has pressed anything, so keying
        # the column off the phase code alone would show Running while the
        # arm sits still waiting to be started.
        leak_active = running and bool(raw.get("LEAK_PHASE_ACTIVE"))

        return {
            "connected": connected,
            "sim_comm_ok": bool(raw.get("SIM_COMM_OK")),
            "safe_to_start": bool(raw.get("SAFE_TO_START")),
            "holding": state_name == "HOLD",
            "idle": state_code == int(CellState.IDLE),
            "state": {"code": state_code, "name": state_name},
            # The two arms are shown as two independent jobs, because that is
            # what they are: the pick arm installs, the leak arm inspects, and
            # each has its own pushbuttons. "active" is which step of its own
            # list that arm is on, or -1 when the other arm has the cell.
            "assembly": {
                "steps": [label for label, _ in ASSEMBLY_STEPS],
                "active": asm_i if running and not leak_active else -1,
                "done": installed,
                "total": 4,
                "running": running and not leak_active and asm_i >= 0,
                "complete": installed >= 4,
            },
            "leak": {
                "steps": [label for label, _ in LEAK_STEPS],
                "active": leak_i if leak_active else -1,
                "done": tested,
                "total": 4,
                "armed": bool(raw.get("LEAK_ARMED")),
                "running": leak_active,
                "complete": tested >= 4,
            },
            "fault": {
                "code": fault_code,
                "text": (FAULT_TEXT.get(Fault(fault_code), "")
                         if fault_code in [f.value for f in Fault]
                         else f"Fault {fault_code}"),
            },
        }

    @staticmethod
    def _step_index(steps, step_code: int) -> int:
        """Which entry of one column's step list this phase lights, or -1 if
        the phase belongs to the other column."""
        try:
            phase = Phase(step_code)
        except ValueError:
            return -1
        for i, (_, phases) in enumerate(steps):
            if phase in phases:
                return i
        return -1
