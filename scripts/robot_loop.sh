#!/usr/bin/env bash
# Keeps a robot process alive for the cell, so Reset restarts the session.
#
# The robot exits whenever the supervisor aborts it, which is what Reset
# does. Relaunching IS the session reset: a fresh process loads the scene
# from its keyframe, so the QDs are back on the tray, the manifold is empty
# and the arms are at their home pose. Start then runs from the beginning.
#
# Why a new process rather than resetting in place: mj_resetData() restores
# DATA, and several things the session changes live in the MODEL -
# disable_qd_collision() and set_qd_collision_for_inspection() both rewrite
# geom contype/conaffinity. A data reset would leave a "fresh" scene whose QDs
# still carry the collision state of the previous run, the kind of bug that
# surfaces three cycles later as an impossible contact reading.
#
# A process restart cannot leave stale state behind, because there is no state
# to leave.
set -u
cd "$(dirname "$0")/.."
mkdir -p out
STOP=out/.robot_stop
rm -f "$STOP"

# Ctrl+C has to mean stop, not "restart once more". Without this the loop
# survives SIGINT - bash keeps going, the robot comes back, and the
# terminal looks like it has ignored you.
#
# The child is backgrounded and waited on rather than run in the foreground,
# because bash defers a trap until the current foreground command finishes -
# so a foreground `python run_cell.py` would swallow Ctrl+C for as long as
# the session lasted. `wait` is interruptible, so the trap runs at once.
CHILD=""

# EXIT in the vision window, Ctrl+C here and stop_cell.sh all mean "stop the
# cell", not just "stop the robot" - so the operator panel goes down too.
#
# The supervisor does this rather than the robot, because by the time the
# robot's exit has been noticed the robot is already gone; the supervisor is
# the only thing still alive that can take the rest down with it. A dying
# process reaching out to kill its siblings is how you get half-torn-down
# cells that need a manual sweep afterwards.
shutdown_cell() {
    pkill -f "scripts/run_hmi.py" 2>/dev/null \
        && echo "[robot_loop] operator panel stopped"
}

trap 'touch "$STOP"; [ -n "$CHILD" ] && kill "$CHILD" 2>/dev/null; \
      echo "[robot_loop] interrupted - staying down"; shutdown_cell; \
      exit 0' INT TERM

# Each launch records to its own file. A fixed name would mean the relaunch
# after a finished session opens a writer on the same path and truncates the
# recording that session just produced - so a completed run is destroyed by
# the reset that follows it, which is exactly backwards.
while true; do
    venv/bin/python scripts/run_cell.py --plc \
        --out "out/cell_run_$(date +%Y%m%d-%H%M%S).mp4" "$@" \
        >> out/cell_run.log 2>&1 &
    CHILD=$!
    wait "$CHILD"
    CHILD=""
    [ -f "$STOP" ] && break
    echo "[robot_loop] session ended; reloading the scene" >> out/cell_run.log
    sleep 2
done
shutdown_cell
rm -f "$STOP"
