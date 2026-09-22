#!/usr/bin/env python3
"""Configure OpenPLC's slave device and server settings from qd_plc.tags.

The slave-device register block SIZES are the most error-prone thing in the
whole setup: OpenPLC derives every %-address from them, a wrong size produces
no error at all, and the symptom is a panel full of plausible wrong numbers.
Typing them by hand from a table invites exactly that. This derives them from
the same source of truth the sim publishes from, so they cannot disagree.

    venv/bin/python scripts/plc_configure.py            # show what it would do
    venv/bin/python scripts/plc_configure.py --apply

docs/OPENPLC_SETUP.md documents the manual route as well - worth walking once
to understand what this is doing on your behalf.
"""

from __future__ import annotations

import argparse
import http.cookiejar
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qd_plc.hostenv import host_ipv4_from_container
from qd_plc.tags import MAP_A, Kind

BASE = "http://127.0.0.1:8080"
DEVICE_NAME = "qd_sim"
SIM_HOST = "host.docker.internal"   # how the container reaches the Mac
SIM_PORT = 5020
SLAVE_ID = 1


def opener_login(user: str, password: str):
    jar = http.cookiejar.CookieJar()
    op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    try:
        req = urllib.request.Request(
            BASE + "/login",
            data=urllib.parse.urlencode({"username": user, "password": password}).encode())
        body = op.open(req, timeout=10).read().decode("utf-8", "replace")
    except urllib.error.URLError as exc:
        raise SystemExit(f"error: cannot reach OpenPLC at {BASE}: {exc}\n"
                         f"  docker compose -f plc/docker-compose.yml up -d")
    if "Bad credentials" in body:
        raise SystemExit("error: OpenPLC rejected those credentials")
    return op


def post(op, path: str, fields: dict, timeout: float = 30) -> str:
    req = urllib.request.Request(BASE + path,
                                 data=urllib.parse.urlencode(fields).encode())
    return op.open(req, timeout=timeout).read().decode("utf-8", "replace")


def get(op, path: str, timeout: float = 30) -> str:
    return op.open(urllib.request.Request(BASE + path),
                   timeout=timeout).read().decode("utf-8", "replace")


def block_sizes() -> dict[str, int]:
    """Derived from the declared block sizes in qd_plc/tags.py.

    Holding-Read is zero on purpose: everything the PLC reads from the robot
    is already an input register, and a second copy would be a second source
    of truth for the same words.
    """
    return {
        "di_size": MAP_A.size(Kind.DISCRETE_INPUT),
        "do_size": MAP_A.size(Kind.COIL),
        "ai_size": MAP_A.size(Kind.INPUT_REG),
        "aor_size": 0,
        "aow_size": MAP_A.size(Kind.HOLDING_REG),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true",
                    help="actually write the configuration (default: preview)")
    ap.add_argument("--replace", action="store_true",
                    help="delete an existing device of the same name and "
                         "rebuild it (use after changing block sizes)")
    ap.add_argument("--user", default="openplc")
    ap.add_argument("--password", default="openplc")
    ap.add_argument("--sim-host", default=None,
                    help="address the CONTAINER uses to reach the sim; "
                         "auto-resolved to IPv4 by default")
    ap.add_argument("--sim-port", type=int, default=SIM_PORT)
    ap.add_argument("--polling-ms", type=int, default=100)
    args = ap.parse_args()

    sizes = block_sizes()

    # OpenPLC's master is libmodbus, and modbus_new_tcp() takes a dotted-quad
    # address - it does not resolve hostnames. Handing it
    # "host.docker.internal" fails with "Invalid argument", which reads like a
    # bad register block rather than a name that could not be looked up. So
    # resolve it here, from the container's own /etc/hosts.
    if args.sim_host:
        sim_host = args.sim_host
    else:
        sim_host = host_ipv4_from_container()
        if sim_host is None:
            print("error: could not resolve host.docker.internal to an IPv4\n"
                  "  address from inside the container. Is it running?\n"
                  "  Pass --sim-host explicitly if you know the address.",
                  file=sys.stderr)
            return 1
        print(f"\nResolved {SIM_HOST} -> {sim_host} "
              f"(libmodbus needs an IP, not a hostname)")

    print(f"\nSlave device '{DEVICE_NAME}'  ->  "
          f"{sim_host}:{args.sim_port}  (Generic Modbus TCP, slave id {SLAVE_ID})\n")
    print(f"  {'Discrete Inputs':<26} start 0  size {sizes['di_size']:<3} -> %IX100.0+")
    print(f"  {'Coils':<26} start 0  size {sizes['do_size']:<3} -> %QX100.0+")
    print(f"  {'Input Registers':<26} start 0  size {sizes['ai_size']:<3} -> %IW100+")
    print(f"  {'Holding Registers - Read':<26} start 0  size {sizes['aor_size']:<3} "
          f"(zero on purpose)")
    print(f"  {'Holding Registers - Write':<26} start 0  size {sizes['aow_size']:<3} -> %QW100+")
    print(f"\n  Modbus server for the HMI: port 502, polling {args.polling_ms}ms\n")

    if not args.apply:
        print("Preview only. Re-run with --apply to write it.\n")
        return 0

    op = opener_login(args.user, args.password)

    existing = get(op, "/modbus")
    if DEVICE_NAME in existing and args.replace:
        # The listing links each row as modbus-edit-device?table_id=N.
        ids = re.findall(r"table_id=(\d+)", existing)
        for dev_id in dict.fromkeys(ids):
            get(op, f"/delete-device?dev_id={dev_id}")
        print(f"Removed the existing '{DEVICE_NAME}' device.")
        existing = ""
    if DEVICE_NAME in existing:
        print(f"A device called '{DEVICE_NAME}' already exists - leaving it alone.")
        print("  Re-run with --replace to rebuild it.\n")
    else:
        post(op, "/add-modbus-device", {
            "device_name": DEVICE_NAME,
            "device_protocol": "TCP",     # Generic Modbus TCP Device
            "device_id": str(SLAVE_ID),
            "device_ip": sim_host,
            "device_port": str(args.sim_port),
            "device_cport": "", "device_baud": "", "device_parity": "None",
            "device_data": "8", "device_stop": "1", "device_pause": "0",
            "di_start": "0", "di_size": str(sizes["di_size"]),
            "do_start": "0", "do_size": str(sizes["do_size"]),
            "ai_start": "0", "ai_size": str(sizes["ai_size"]),
            "aor_start": "0", "aor_size": str(sizes["aor_size"]),
            "aow_start": "0", "aow_size": str(sizes["aow_size"]),
        })
        print(f"Added slave device '{DEVICE_NAME}'.")

    post(op, "/settings", {
        "modbus_server_port": "502",
        "dnp3_server_port": "disabled",
        "enip_server_port": "disabled",
        "pstorage_thread_poll": "disabled",
        "auto_run_text": "disabled",
        "snap7_run_text": "disabled",
        "slave_polling_period": str(args.polling_ms),
        "slave_timeout": "1000",
        "device_hostname": "openplc",
    })
    print("Enabled the Modbus server on :502 and set the slave polling period.")

    # Slave devices and settings are only picked up when the runtime starts,
    # so a restart here is not optional - without it nothing appears to change.
    get(op, "/stop_plc")
    time.sleep(2)
    get(op, "/start_plc")
    time.sleep(3)
    print("Restarted the PLC runtime (required - neither is picked up while "
          "it is running).\n")
    print("Now start the cell:  ./scripts/start_cell.sh\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
