# OpenPLC v3 built from source, so it runs natively on Apple Silicon.
#
# The published images are amd64 only; running one under emulation on an M-
# series Mac adds scan-time jitter that is very easy to misdiagnose as a
# Modbus problem. Building here costs a few minutes once and then the runtime
# is native arm64.
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential pkg-config bison flex autoconf automake libtool \
        make cmake python3 python3-pip python3-dev sudo curl ca-certificates \
        sqlite3 libssl-dev netcat-openbsd iproute2 \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/thiagoralves/OpenPLC_v3.git /opt/OpenPLC_v3
WORKDIR /opt/OpenPLC_v3
RUN ./install.sh docker

# OpenPLC's Settings page shells out to hostnamectl, which only exists under
# systemd - in a container that turns "Save changes" into a 500 error page.
# A no-op stub keeps the page working; the container's hostname is fixed by
# docker-compose anyway.
RUN printf '#!/bin/sh\nexit 0\n' > /usr/local/bin/hostnamectl \
    && chmod +x /usr/local/bin/hostnamectl

# 8080 web UI (programs, slave devices, monitoring)
# 502  Modbus TCP server - this is the HMI's link into the PLC
EXPOSE 8080 502

CMD ["./start_openplc.sh"]
