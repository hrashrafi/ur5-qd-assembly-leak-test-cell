#!/usr/bin/env python3
"""Run one assembly + leak-test session.

    venv/bin/python scripts/run_cell.py --plc
        The normal way to run the cell. Parks at gate G0 until the operator
        panel says START, then steps through the session as the PLC permits
        each phase.
    venv/bin/python scripts/run_cell.py
        Standalone, no PLC: runs the whole session and writes the video.
    venv/bin/python scripts/run_cell.py --live
        Standalone with the interactive window and its own START button.

The session itself lives in qd_cell/session.py; this file only reads the
command line and starts it.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qd_cell.session import CellSession


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Run one assembly + leak-test session.")
    p.add_argument("--out", default="out/full_session.mp4",
                   help="Where to write the recorded session.")
    p.add_argument("--live", action="store_true",
                   help="Open the interactive window, gated behind its own "
                        "START button, while recording the same video.")
    p.add_argument("--auto-start", action="store_true",
                   help="With --live, skip the START button and begin at once.")
    p.add_argument("--live-out", default="out/live_dashboard.mp4",
                   help="With --live, also record the window itself - both "
                        "halves, overlays and telemetry - to this file.")
    p.add_argument("--plc", action="store_true",
                   help="Publish status to a Modbus TCP slave and wait at each "
                        "gate for the cell PLC's permit. Without this flag the "
                        "session simply runs start to finish.")
    p.add_argument("--modbus-host", default="0.0.0.0",
                   help="Bind address for the Modbus slave. 0.0.0.0 so a "
                        "Docker-hosted PLC can reach it - binding loopback is "
                        "the commonest cause of an offline slave device.")
    p.add_argument("--modbus-port", type=int, default=5020)
    p.add_argument("--no-plc-gating", action="store_true",
                   help="With --plc, release every gate locally instead of "
                        "waiting for a permit - the full tag map, no PLC.")
    p.add_argument("--no-vision-display", action="store_true",
                   help="Don't open the OpenCV vision-display window. It is on "
                        "by default with --plc.")
    return p.parse_args(argv)


if __name__ == "__main__":
    CellSession(parse_args()).run()
