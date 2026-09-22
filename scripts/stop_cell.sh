#!/usr/bin/env bash
# Stop the robot and the operator panel. Leaves the PLC container running -
# it is slow to start and holds no session state worth clearing.
#
#   ./scripts/stop_cell.sh            stop robot + panel
#   ./scripts/stop_cell.sh --all      also stop the PLC container
cd "$(dirname "$0")/.."
export PATH="$HOME/.docker/bin:$PATH"

# What is up right now. Sampled BEFORE anything is killed, because the
# supervisor takes the panel down with it on its way out (robot_loop.sh's own
# trap) - so by the time we get to the panel it is already gone, and asking
# pkill whether it matched anything would report a running cell as "not
# running".
pgrep -f "scripts/run_cell.py" >/dev/null && ROBOT=1 || ROBOT=0
pgrep -f "scripts/run_hmi.py"  >/dev/null && PANEL=1 || PANEL=0

# Tell the supervisor to stay down BEFORE killing the robot, or it just
# relaunches the process we are trying to stop.
mkdir -p out && touch out/.robot_stop
pkill -f "scripts/robot_loop.sh" 2>/dev/null
pkill -f "scripts/run_cell.py" 2>/dev/null
pkill -f "scripts/run_hmi.py"  2>/dev/null
[ "$ROBOT" = 1 ] && echo "  robot stopped" || echo "  robot was not running"
[ "$PANEL" = 1 ] && echo "  panel stopped" || echo "  panel was not running"
if [ "${1:-}" = "--all" ]; then
    docker compose -f plc/docker-compose.yml down >/dev/null 2>&1 && echo "  PLC container stopped"
fi
