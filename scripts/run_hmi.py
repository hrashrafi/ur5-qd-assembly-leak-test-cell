#!/usr/bin/env python3
"""Run the operator panel.

    venv/bin/python scripts/run_hmi.py

Then open http://127.0.0.1:8000. The panel polls the PLC; it never talks to
the robot directly - that link belongs to the supervisor.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qd_plc.hmi.app import serve


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plc", default="127.0.0.1:502",
                    help="host:port of the PLC's Modbus server")
    ap.add_argument("--http", type=int, default=8000)
    ap.add_argument("--interval", type=float, default=0.25,
                    help="seconds between PLC polls")
    args = ap.parse_args()

    host, _, port = args.plc.partition(":")
    print("\nQD manifold assembly cell - operator panel\n")
    serve(host, int(port or 502), args.http, args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
