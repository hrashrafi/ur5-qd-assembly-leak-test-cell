"""Finding Docker, and finding the Mac from inside the container.

Two small problems that both cost more time than they should.

Docker Desktop installs its CLI to ~/.docker/bin and adds that to your shell
profile, so a shell opened before the install - or a non-login shell - will not
find `docker` on PATH even though Docker is plainly running.

And OpenPLC's Modbus master is libmodbus, whose `modbus_new_tcp()` takes a
dotted-quad address and does not resolve hostnames. Configuring a slave device
as `host.docker.internal` therefore fails with "Connection failed ... Invalid
argument" - which reads like a malformed register block rather than a name
resolution problem. The address has to be resolved to IPv4 first.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess

DOCKER_CANDIDATES = [
    "docker",
    os.path.expanduser("~/.docker/bin/docker"),
    "/usr/local/bin/docker",
    "/opt/homebrew/bin/docker",
    "/Applications/Docker.app/Contents/Resources/bin/docker",
]

CONTAINER = "qd_cell_plc"
HOST_ALIAS = "host.docker.internal"


class DockerMissing(RuntimeError):
    pass


def docker_bin() -> str:
    for cand in DOCKER_CANDIDATES:
        if shutil.which(cand) or (os.path.isabs(cand) and os.access(cand, os.X_OK)):
            return cand
    raise DockerMissing(
        "The docker command is not on PATH.\n"
        "  Install Docker Desktop for Mac (Apple Silicon build) and start it.\n"
        "  If it is already installed, open a NEW terminal - Docker Desktop adds\n"
        "  ~/.docker/bin to your shell profile and existing shells lack it."
    )


def host_ipv4_from_container(container: str = CONTAINER,
                             alias: str = HOST_ALIAS) -> str | None:
    """The IPv4 address the container reaches the Mac on.

    Read from the container's /etc/hosts rather than via getent, because
    getent returns the IPv6 entry first and libmodbus wants IPv4.
    """
    try:
        proc = subprocess.run(
            [docker_bin(), "exec", container, "cat", "/etc/hosts"],
            capture_output=True, text=True, timeout=15)
    except (DockerMissing, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    for line in proc.stdout.splitlines():
        if alias in line:
            match = re.match(r"^(\d+\.\d+\.\d+\.\d+)\s", line.strip())
            if match:
                return match.group(1)
    return None
