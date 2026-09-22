#!/usr/bin/env bash
# Bring up the whole cell and tell you what to do next.
#
# Starts whichever of the three processes is not already running, in the
# right order (PLC first, then the panel, then the robot - the robot is what
# actually moves, so it goes last). Safe to re-run; it skips anything that is
# already up.
#
# Opens the operator panel in your browser once the robot is parked and
# ready, so this is the only command needed to bring the cell up.
#
#   ./scripts/start_cell.sh                vision display on, panel opened
#   ./scripts/start_cell.sh --headless     no OpenCV window
#   ./scripts/start_cell.sh --no-browser   do not open the browser

set -u
cd "$(dirname "$0")/.."
export PATH="$HOME/.docker/bin:$PATH"
mkdir -p out    # logs, the recorded video and the supervisor's stop-file

VISION_FLAG=""
OPEN_BROWSER=1
for arg in "$@"; do
    case "$arg" in
        --headless)   VISION_FLAG="--no-vision-display" ;;
        --no-browser) OPEN_BROWSER=0 ;;
    esac
done

say()  { printf '\n\033[1m%s\033[0m\n' "$*"; }
ok()   { printf '  \033[32mok\033[0m   %s\n' "$*"; }
work() { printf '  \033[33m..\033[0m   %s\n' "$*"; }

say "1/3  Cell PLC (OpenPLC in Docker)"
if docker ps --filter name=qd_cell_plc --format '{{.Names}}' 2>/dev/null | grep -q qd_cell_plc; then
    ok "already running  ->  http://localhost:8080  (openplc/openplc)"
else
    work "starting container (first run builds the image, several minutes)"
    docker compose -f plc/docker-compose.yml up -d >/dev/null 2>&1
    until curl -s -m 2 http://127.0.0.1:8080 >/dev/null 2>&1; do sleep 2; done
    ok "up  ->  http://localhost:8080"
    work "loading the cell logic"
    venv/bin/python scripts/plc_upload.py plc/st/cell_control.st --start 2>&1 | tail -1
fi

say "2/3  Operator panel"
if curl -s -m 2 http://127.0.0.1:8000/api/state >/dev/null 2>&1; then
    ok "already running  ->  http://localhost:8000"
else
    work "starting"
    nohup venv/bin/python scripts/run_hmi.py > out/hmi.log 2>&1 &
    until curl -s -m 2 http://127.0.0.1:8000/api/state >/dev/null 2>&1; do sleep 1; done
    ok "up  ->  http://localhost:8000"
fi

say "3/3  Robot (MuJoCo)"
if pgrep -f "scripts/robot_loop.sh" >/dev/null; then
    ok "already running"
else
    work "starting - loads the scene, then parks at gate G0 and waits"
    nohup ./scripts/robot_loop.sh $VISION_FLAG > /dev/null 2>&1 &
    # The robot is ready once the supervisor can see it and every interlock
    # for a fresh start is satisfied - that is exactly what SAFE_TO_START is,
    # and the panel already publishes it. Asking the panel rather than the PLC
    # keeps this script to one interface instead of two.
    READY_NOW=0
    for _ in $(seq 1 90); do
        if curl -s -m 2 http://127.0.0.1:8000/api/state | grep -q '"safe_to_start": true'; then
            ok "parked at G0, PLC sees it, ready to start"
            READY_NOW=1
            break
        fi
        sleep 2
    done
    # The commonest reason for not getting there is a fault the PLC latched
    # when the PREVIOUS robot process exited - it saw the slave device
    # disappear and called it a comms loss, which is exactly right. That
    # fault outlives the process, so say so rather than timing out silently.
    if [ "$READY_NOW" = "0" ]; then
        FAULT=$(curl -s -m 2 http://127.0.0.1:8000/api/state \
                | sed -n 's/.*"text": "\([^"]*\)".*/\1/p')
        printf '  \033[31m!!\033[0m   not ready to start: %s\n' "${FAULT:-no fault reported}"
        echo "       Press RESET on the operator panel to clear it."
    fi
fi

# Opened last, once the robot is actually parked at G0 - a panel opened
# before then shows a cell with no robot, which reads as a fault rather than
# as "still starting".
if [ "$OPEN_BROWSER" = "1" ] && command -v open >/dev/null 2>&1; then
    open http://localhost:8000
fi

say "Ready."
cat <<'MSG'
  The operator panel is open at  http://localhost:8000  -  press START.

  The robot will not move until you do - it is parked at gate G0 waiting for
  the PLC's permission, and the PLC will not give it until you press the
  button. Past the panel's own web server, that whole path is real Modbus:

      browser -> HTTP -> panel server -> Modbus -> OpenPLC -> Modbus -> robot

  A full session is 4 QDs installed, then 4 leak tests, about 12 minutes.

  To stop everything:   ./scripts/stop_cell.sh
MSG
