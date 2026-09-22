#!/usr/bin/env python3
"""Upload a Structured Text program to OpenPLC, compile it, and report errors.

Doing this through the web UI is several clicks and a wait, and OpenPLC's
browser editor loses work when you navigate away - so programs are authored in
this repo and uploaded as files. This makes that a one-liner, and prints the
compiler's actual complaint instead of leaving you to hunt for it in the logs
page.

    venv/bin/python scripts/plc_upload.py plc/st/cell_control.st
    venv/bin/python scripts/plc_upload.py plc/st/cell_control.st --start

MATIEC's diagnostics point at the generated intermediate file, so a reported
line number is usually near, but not exactly at, the line in your source.
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
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = "http://127.0.0.1:8080"


def make_opener() -> urllib.request.OpenerDirector:
    jar = http.cookiejar.CookieJar()
    return urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))


def post(opener, path: str, fields: dict, timeout: float = 60) -> str:
    data = urllib.parse.urlencode(fields).encode()
    req = urllib.request.Request(BASE + path, data=data)
    with opener.open(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", "replace")


def get(opener, path: str, timeout: float = 60) -> str:
    with opener.open(urllib.request.Request(BASE + path), timeout=timeout) as resp:
        return resp.read().decode("utf-8", "replace")


def post_file(opener, path: str, filename: str, content: bytes,
              timeout: float = 60) -> str:
    """Minimal multipart/form-data, to avoid a dependency for one request."""
    boundary = uuid.uuid4().hex
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: application/octet-stream\r\n\r\n"
    ).encode() + content + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(BASE + path, data=body)
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    with opener.open(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", "replace")


def hidden_field(html: str, name: str) -> str | None:
    """Pull a hidden input's value out, regardless of attribute order."""
    for tag in re.findall(r"<input[^>]*>", html):
        if re.search(rf"name=['\"]{re.escape(name)}['\"]", tag):
            value = re.search(r"value=['\"]([^'\"]*)['\"]", tag)
            if value:
                return value.group(1)
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("st_file", type=Path)
    ap.add_argument("--user", default="openplc")
    ap.add_argument("--password", default="openplc")
    ap.add_argument("--start", action="store_true",
                    help="start the runtime after a successful compile")
    ap.add_argument("--timeout", type=float, default=120)
    args = ap.parse_args()

    path = args.st_file if args.st_file.is_absolute() else ROOT / args.st_file
    if not path.exists():
        print(f"error: no such file: {path}", file=sys.stderr)
        return 2
    content = path.read_bytes()

    opener = make_opener()
    try:
        body = post(opener, "/login",
                    {"username": args.user, "password": args.password})
    except urllib.error.URLError as exc:
        print(f"error: cannot reach OpenPLC at {BASE}: {exc}\n"
              f"  Is the container up?  "
              f"docker compose -f plc/docker-compose.yml up -d", file=sys.stderr)
        return 1
    if "Bad credentials" in body:
        print("error: OpenPLC rejected those credentials", file=sys.stderr)
        return 1

    print(f"uploading {path.relative_to(ROOT)} ({len(content)} bytes)")
    body = post_file(opener, "/upload-program", path.name, content)
    # OpenPLC writes these hidden fields with value= before name=, so match
    # the tag and pull attributes out of it rather than assuming an order.
    stored = hidden_field(body, "prog_file")
    if not stored:
        print("error: OpenPLC did not accept the upload (no prog_file in reply).",
              file=sys.stderr)
        print("  Is a compile already in progress? Check "
              "http://127.0.0.1:8080/programs", file=sys.stderr)
        return 1
    epoch = hidden_field(body, "epoch_time") or str(int(time.time()))
    post(opener, "/upload-program-action", {
        "prog_name": path.stem,
        "prog_descr": f"uploaded from {path.name} by scripts/plc_upload.py",
        "prog_file": stored,
        "epoch_time": epoch,
    })
    # That handler answers with a <meta http-equiv="refresh"> pointing at the
    # compile endpoint. A browser would follow it; urllib will not, so the
    # compile has to be kicked off explicitly or nothing ever happens and the
    # logs stay empty.
    get(opener, f"/compile-program?file={urllib.parse.quote(stored)}")

    print("compiling", end="", flush=True)
    deadline = time.time() + args.timeout
    logs = ""
    while time.time() < deadline:
        logs = get(opener, "/compilation-logs")
        if "Compilation finished" in logs or "error" in logs.lower():
            break
        print(".", end="", flush=True)
        time.sleep(1.5)
    print()

    text = re.sub(r"<[^>]+>", "", logs).strip()
    ok = "Compilation finished successfully" in text

    if ok:
        print("\nCompilation finished successfully.")
        if args.start:
            get(opener, "/start_plc")
            time.sleep(2)
            print("PLC runtime started.")
        return 0

    print("\nCompilation FAILED. OpenPLC said:\n")
    interesting = [ln for ln in text.splitlines()
                   if ln.strip() and not ln.startswith("Optimizing")]
    for line in interesting[-40:]:
        print("   " + line)
    print("\n  MATIEC points at its generated intermediate file, so the line\n"
          "  number is near - not exactly at - the line in your .st source.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
