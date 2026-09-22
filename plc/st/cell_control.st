(* =====================================================================
   QD MANIFOLD ASSEMBLY CELL - supervisory control

   Sits between the operator panel and the robot controller. It does not
   move anything: motion and vision stay inside the robot controller, the
   way a real one keeps its servo loop off the fieldbus. What this program
   owns is permission - whether the cell may proceed from one step to the
   next - plus the cycle state, the interlocks, and the fault table.

   THE DIVISION OF LABOUR, which is the point of the whole design:
   the robot reports raw numbers and raw conditions; this program decides
   which of them matter. The robot says TILT_DEG = 51 and
   SELF_COLL_PEAK = 3. This program decides that 51 is within limits and
   that 3 collisions is a fault. Move that judgement into the robot and
   you no longer have a supervisor, you have a logger.

   Addresses come from qd_plc/tags.py via plc/st/tags_generated.st. Do not
   retype them: OpenPLC derives them from the slave-device block sizes, so
   a resize silently shifts everything after it.

   Scan: 50ms. The robot free-runs far faster, so between gates this
   program is an observer, not a controller. Anything it must arbitrate
   has to be a gate.
   ===================================================================== *)

PROGRAM prog0
VAR
    (* ==== Map A - robot controller, via the slave-device mapping ==== *)
    a_sim_ready         AT %IX100.1 : BOOL;
    a_busy              AT %IX100.2 : BOOL;
    a_gated             AT %IX100.3 : BOOL;
    a_phase_done        AT %IX100.4 : BOOL;
    a_session_done      AT %IX100.7 : BOOL;
    a_qd_grasped        AT %IX101.0 : BOOL;
    a_qd_seated         AT %IX101.1 : BOOL;
    a_gripper_open      AT %IX101.2 : BOOL;
    a_part_detected     AT %IX101.3 : BOOL;
    a_vision_fallback   AT %IX101.4 : BOOL;
    a_vision_coasting   AT %IX101.5 : BOOL;
    a_self_collision    AT %IX101.7 : BOOL;
    a_leak_contact      AT %IX102.0 : BOOL;
    a_maxsteps_expired  AT %IX102.1 : BOOL;
    a_seat_shortfall    AT %IX102.3 : BOOL;
    a_active_arm_leak   AT %IX102.4 : BOOL;
    a_abort_complete    AT %IX102.5 : BOOL;
    a_sim_stopped       AT %IX102.6 : BOOL;
    a_live_window_open  AT %IX102.7 : BOOL;
    a_permit_mismatch   AT %IX103.0 : BOOL;
    a_gate_timeout      AT %IX103.1 : BOOL;

    a_phase_code        AT %IW100 : UINT;
    a_phase_seq         AT %IW101 : UINT;
    a_port_id           AT %IW105 : UINT;
    a_insert_depth      AT %IW106 : UINT;
    a_tilt_deg          AT %IW109 : INT;
    a_track_diff        AT %IW110 : UINT;
    a_self_coll_peak    AT %IW113 : UINT;
    a_leak_contact_peak AT %IW114 : UINT;
    a_sim_tick          AT %IW119 : UINT;
    a_qds_installed     AT %IW122 : UINT;
    a_leak_tested       AT %IW123 : UINT;

    a_permit            AT %QX100.0 : BOOL;
    a_phase_ack         AT %QX100.1 : BOOL;
    a_run               AT %QX100.2 : BOOL;
    a_hold              AT %QX100.3 : BOOL;
    a_abort             AT %QX100.4 : BOOL;
    a_clear_sticky      AT %QX100.5 : BOOL;
    a_permit_phase      AT %QW100 : UINT;
    a_permit_seq        AT %QW101 : UINT;
    a_gate_timeout_s    AT %QW107 : UINT;

    (* ==== Map B - operator panel, on OpenPLC's own Modbus server ====
       Status lives in coils and holding registers because an IEC program
       cannot write %IX or %IW, and here that space belongs to the master
       mapping. Ownership is by address range, documented, not enforced. *)
    b_sim_comm_ok       AT %QX0.6 : BOOL;
    b_leak_phase_active AT %QX1.2 : BOOL;
    b_leak_armed        AT %QX1.3 : BOOL;
    b_safe_to_start     AT %QX1.4 : BOOL;

    b_hmi_start         AT %QX10.0 : BOOL;
    b_hmi_stop          AT %QX10.1 : BOOL;
    b_hmi_reset         AT %QX10.2 : BOOL;
    b_hmi_start_leak    AT %QX11.1 : BOOL;
    b_hmi_stop_leak     AT %QX11.2 : BOOL;

    b_cell_state        AT %QW0 : UINT;
    b_current_step      AT %QW1 : UINT;
    b_active_fault_code AT %QW15 : UINT;
    b_leak_tested       AT %QW25 : UINT;
    b_qds_installed     AT %QW32 : UINT;
END_VAR

(* MATIEC will not accept located and plain variables in the same VAR block:
   once a block contains an AT %... declaration, every entry in it must be
   located, and the ones that are not are reported as "invalid located
   variable declaration". Internals therefore get their own block. *)
VAR
    (* One instance per line. Three or more function-block instances in a
       single comma-separated declaration segfaults MATIEC - the compiler
       crashes outright with no diagnostic, which costs an afternoon to find
       if you do not already know. Two is fine; three is not. *)
    trig_start          : R_TRIG;
    trig_stop           : R_TRIG;
    trig_reset          : R_TRIG;
    trig_start_leak     : R_TRIG;
    trig_stop_leak      : R_TRIG;
    t_comms             : TON;
    t_fault_park        : TON;

    (* The limits. They live here, in the supervisor, and nowhere in the
       robot: the robot reports raw numbers in the same units as its
       registers, and this program decides what counts as out of limits. *)
    depth_target        : UINT := 1200; (* 12.00mm full engagement *)
    depth_tol           : UINT := 20;   (* 0.20mm grace on the seating check *)
    tilt_limit          : INT := 500;   (* 5.00deg grasp tilt *)
    track_diff_limit    : UINT := 300;  (* 3.00mm tracked vs taught *)
    coast_scan_limit    : UINT := 400;  (* 20s of vision coasting *)
    phase_wdt_s         : UINT := 120;  (* per-phase watchdog, seconds *)

    phase_scans         : UINT := 0;
    phase_overrun       : BOOL := FALSE;
    tilt_abs            : INT := 0;
    alive_changed       : BOOL;
    alive_last_tick     : UINT := 0;
    permit_ok           : BOOL;
    cell_state          : UINT := 0;
    fault_code          : UINT := 0;
    last_seq            : UINT := 0;
    coast_scans         : UINT := 0;
    leak_armed          : BOOL := FALSE;
    clearing            : BOOL := FALSE;
    clear_wait          : UINT := 0;
END_VAR

(* =====================================================================
   1. HOUSEKEEPING
   ===================================================================== *)
(* Comms watchdog, keyed off the robot's free-running step COUNTER, not off
   a toggle bit.

   A one-bit toggle looks like the obvious heartbeat and is a trap here: the
   robot renders a video frame every 16 physics steps, that render dominates
   its wall-clock time, and 16 is even - so the bit carries the same parity
   at every frame boundary and reads stuck to anything sampling on a slower
   clock. Measured against the real simulation, only 2 of 40 polls ever saw
   it change. A counter cannot alias that way: if the robot is stepping at
   all, this word differs from the last one we read. *)
IF a_sim_tick <> alive_last_tick THEN
    alive_changed := TRUE;
    alive_last_tick := a_sim_tick;
END_IF;
t_comms(IN := NOT alive_changed, PT := T#2s);
b_sim_comm_ok := NOT t_comms.Q;
alive_changed := FALSE;

(* Operator pushbuttons. The panel writes a coil; this edge-detects it and
   writes it back to 0, which is the standard OpenPLC momentary pattern. *)
trig_start(CLK := b_hmi_start);
trig_stop(CLK := b_hmi_stop);
trig_reset(CLK := b_hmi_reset);
trig_start_leak(CLK := b_hmi_start_leak);
trig_stop_leak(CLK := b_hmi_stop_leak);
IF b_hmi_start THEN b_hmi_start := FALSE; END_IF;
IF b_hmi_stop THEN b_hmi_stop := FALSE; END_IF;
IF b_hmi_reset THEN b_hmi_reset := FALSE; END_IF;
IF b_hmi_start_leak THEN b_hmi_start_leak := FALSE; END_IF;
IF b_hmi_stop_leak THEN b_hmi_stop_leak := FALSE; END_IF;

(* The leak section is its own job with its own pushbuttons. Arming is
   latched rather than maintained: a maintained ENABLE switch left off
   would not skip the leak phase, it would park the arm at G8 until the
   gate watchdog faulted the cell. A latch
   the operator sets and clears cannot do that - an unarmed leak section
   simply never asks for the gate, and the cell finishes without it. *)
(* Start ARMS the leak section - except in HOLD, where every Start means
   "resume" and nothing else. Arming as a side effect of resuming is how a
   leak test ends up running that nobody asked for: hold the cell during
   assembly, press this to carry on, and the leak phase silently becomes
   permitted for later. *)
IF trig_start_leak.Q AND cell_state <> 3 THEN leak_armed := TRUE; END_IF;

(* Stop means two different things depending on whether the leak arm has the
   cell yet, and both are what the word ought to mean at the time:

     before  - the arm has not started, so cancel the request. Disarming is
               the only way back out, since Stop on its own holds rather
               than disarms.
     during  - the arm is tracing a seam, so stop it NOW. The latch is left
               alone so Start picks the same port back up where it froze,
               rather than abandoning a half-traced seam. *)
IF trig_stop_leak.Q AND NOT a_active_arm_leak THEN
    leak_armed := FALSE;
END_IF;
IF cell_state = 0 OR cell_state = 5 THEN leak_armed := FALSE; END_IF;

(* The robot's own gate-wait safety net gets the same time budget. *)
a_gate_timeout_s := phase_wdt_s;

(* Vision coasting is judged per step: a new PHASE_SEQ starts the count
   again, so a long run of short, harmless coasts cannot add up to a fault. *)
IF a_phase_seq <> last_seq THEN
    last_seq := a_phase_seq;
    coast_scans := 0;
END_IF;
IF a_vision_coasting THEN
    coast_scans := coast_scans + 1;
END_IF;

(* =====================================================================
   2. FAULT EVALUATION

   The robot reports conditions; this decides which are faults. First one
   wins and latches.
   ===================================================================== *)
(* Phase watchdog by scan count rather than a TON with a computed PT.
   MATIEC cannot build a TIME from a register at runtime, and the task
   interval is a fixed 50ms, so counting scans is both simpler and exact:
   20 scans per second. *)
(* cell_state 3 is HOLD. The robot freezes mid-phase rather than running
   on to its next gate, so it stays BUSY while held - and the scan counter
   has to stop with it, or a long pause spends the phase's whole budget and
   faults the moment the operator resumes. *)
IF a_busy AND NOT a_gated AND cell_state <> 3 THEN
    phase_scans := phase_scans + 1;
ELSE
    phase_scans := 0;
END_IF;
phase_overrun := phase_scans > (phase_wdt_s * 20);

IF cell_state >= 1 AND cell_state <= 3 AND fault_code = 0 THEN
    (* cell_state 3 is HOLD: the operator asked the cell to stop, so a
       robot sitting still is the intended outcome, not an overrun. The sim
       already stops accruing gate time while HOLD is set; this is the
       supervisor's half of the same rule. *)
    IF NOT b_sim_comm_ok THEN                       fault_code := 1;
    ELSIF a_gate_timeout AND cell_state <> 3 THEN   fault_code := 2;
    ELSIF phase_overrun AND cell_state <> 3 THEN    fault_code := 3;
    ELSIF a_maxsteps_expired THEN                   fault_code := 4;
    ELSIF a_self_collision OR a_self_coll_peak > 0 THEN  fault_code := 7;
    ELSIF a_leak_contact OR a_leak_contact_peak > 0 THEN    fault_code := 8;
    ELSIF a_seat_shortfall THEN                     fault_code := 9;
    ELSIF a_permit_mismatch THEN                    fault_code := 14;
    ELSIF NOT a_live_window_open THEN               fault_code := 17;
    ELSIF a_vision_fallback THEN                    fault_code := 5;
    ELSIF coast_scans > coast_scan_limit THEN       fault_code := 6;
    END_IF;
END_IF;

(* =====================================================================
   3. CYCLE STATE MACHINE

   0 IDLE  1 STARTING  2 RUNNING  3 HOLD  4 FAULTING  5 FAULTED
   6 COMPLETE  7 ABORTING

   FAULTING is separate from FAULTED because a physics loop cannot be
   stopped mid-step: the robot needs a moment to park itself and flush its
   recording. Collapsing the two would mean declaring the cell safe before
   it had actually stopped.
   ===================================================================== *)
b_safe_to_start := a_sim_ready AND b_sim_comm_ok AND fault_code = 0
                   AND a_live_window_open;

CASE cell_state OF
    0:  (* IDLE *)
        IF trig_start.Q AND b_safe_to_start THEN
            cell_state := 1;
        END_IF;

    1:  (* STARTING - wait for the robot to present its first gate *)
        IF fault_code <> 0 THEN
            cell_state := 4;
        ELSIF a_gated THEN
            cell_state := 2;
        END_IF;

    2:  (* RUNNING *)
        IF fault_code <> 0 THEN
            cell_state := 4;
        ELSIF trig_reset.Q THEN
            (* Reset is available in every state except IDLE, because an
               operator should never have to work out which button the cell
               will accept. From here it abandons the run rather than
               pausing it - Stop is the button that pauses. *)
            cell_state := 7;
        ELSIF trig_stop.Q OR (trig_stop_leak.Q AND a_active_arm_leak) THEN
            (* Either column's Stop holds the cell. They are two buttons on
               one machine, not two machines - whichever arm is moving, Stop
               has to stop it, and an operator should not have to work out
               which arm owns the cell right now. *)
            cell_state := 3;
        ELSIF a_session_done THEN
            cell_state := 6;
        END_IF;

    3:  (* HOLD - the robot stays at its gate; nothing is abandoned.
           Start resumes from here. Reset abandons the run instead: the
           robot is told to park and the cell returns to IDLE. *)
        IF fault_code <> 0 THEN
            cell_state := 4;
        ELSIF trig_start.Q OR trig_start_leak.Q THEN
            cell_state := 2;
        ELSIF trig_reset.Q THEN
            cell_state := 7;
        END_IF;

    4:  (* FAULTING - told the robot to park; waiting for it to finish *)
        t_fault_park(IN := TRUE, PT := T#5s);
        IF a_abort_complete OR a_sim_stopped OR t_fault_park.Q
           OR NOT b_sim_comm_ok THEN
            cell_state := 5;
            t_fault_park(IN := FALSE, PT := T#5s);
        END_IF;

    5:  (* FAULTED - reset clears sticky bits FIRST, then the fault.
           The other order re-latches the fault from a stale bit the robot
           has not been told to drop yet. *)
        IF trig_reset.Q AND NOT clearing THEN
            clearing := TRUE;
            clear_wait := 0;
        END_IF;
        IF clearing THEN
            a_clear_sticky := TRUE;
            clear_wait := clear_wait + 1;
            IF clear_wait > 10 THEN          (* half a second of polls *)
                a_clear_sticky := FALSE;
                clearing := FALSE;
                (* The sticky bits live in the robot. With the robot
                   unreachable there is nothing to verify and nothing to
                   protect - and a robot that is restarted comes back with
                   them clear anyway. Refusing to reset here would strand
                   the cell in FAULTED with no way out: the button would be
                   live, the operator would press it, and nothing would
                   happen. SAFE_TO_START still requires comms, so clearing
                   the alarm cannot start a cell whose robot is missing. *)
                IF (NOT b_sim_comm_ok)
                   OR NOT (a_self_collision OR a_leak_contact
                        OR a_maxsteps_expired OR a_seat_shortfall
                        OR a_vision_fallback OR a_permit_mismatch
                        OR a_gate_timeout) THEN
                    fault_code := 0;
                    coast_scans := 0;
                    cell_state := 0;
                END_IF;
            END_IF;
        END_IF;

    6:  (* COMPLETE *)
        IF trig_reset.Q OR trig_start.Q THEN
            cell_state := 0;
        END_IF;

    7:  (* ABORTING - reached either from a fault or from Reset in HOLD.
           Where it lands depends on which: a fault still has to be
           acknowledged, a deliberate abort has nothing to acknowledge. *)
        IF a_abort_complete OR a_sim_stopped OR NOT b_sim_comm_ok THEN
            IF fault_code <> 0 THEN
                cell_state := 5;
            ELSE
                cell_state := 0;
            END_IF;
        END_IF;
END_CASE;

(* =====================================================================
   4. GATE ARBITRATION - the interlock table

   The robot parks at a gate and publishes what it wants to do next. This
   decides whether it may. Permits echo back both the phase and the
   sequence number: the robot refuses anything that does not match, so a
   permit left over from the previous gate cannot release this one - which
   matters, because the robot can reach the next gate well inside one scan
   of this program.
   ===================================================================== *)
permit_ok := FALSE;

IF a_tilt_deg < 0 THEN
    tilt_abs := 0 - a_tilt_deg;
ELSE
    tilt_abs := a_tilt_deg;
END_IF;

IF cell_state = 2 AND a_gated AND fault_code = 0 THEN
    CASE a_phase_code OF
        1:      (* G0 WAIT_START *)
            permit_ok := b_safe_to_start;

        10:     (* G1 CYCLE_START - nothing to check beyond the conditions
                   every gate needs: running, parked here, no fault *)
            permit_ok := TRUE;

        12:     (* G2 PICK_PERMIT - do not reach for a part vision never saw *)
            permit_ok := a_part_detected
                         AND NOT a_vision_fallback
                         AND a_qds_installed < 4
                         AND a_track_diff <= track_diff_limit;

        32:     (* G3 PORT_PERMIT - a badly gripped part will not thread.
                   Tilt is signed, so compare its magnitude; written out
                   rather than with ABS() to keep the expression simple. *)
            permit_ok := a_qd_grasped
                         AND a_part_detected
                         AND tilt_abs <= tilt_limit;

        41:     (* G4 INSERT_PERMIT *)
            permit_ok := a_qd_grasped
                         AND a_port_id > 0
                         AND a_track_diff <= track_diff_limit
                         AND NOT a_self_collision;

        53:     (* G5 RELEASE_PERMIT - the interlock that matters.
                   The supervisor decides the part is seated deep enough
                   before the gripper is allowed to let go. It sits in the
                   window where the robot is already deliberately holding
                   still after screwing, so waiting here costs nothing. *)
            permit_ok := a_qd_seated
                         AND (a_insert_depth + depth_tol >= depth_target)
                         AND NOT a_seat_shortfall;

        55:     (* G6 RETRACT_PERMIT - do not lift while still holding it *)
            permit_ok := a_gripper_open AND NOT a_qd_grasped;

        57:     (* G7 CYCLE_END - the cycle boundary *)
            permit_ok := TRUE;

        70:     (* G8 LEAK_PHASE_PERMIT - the leak section's own start *)
            permit_ok := leak_armed AND a_qds_installed >= 4;

        72:     (* G9 LEAK_PORT_PERMIT
               leak_armed is re-checked here and at G10 so the section's STOP
               takes effect at the next port boundary rather than only before
               the first one. Stopping between ports is a clean place to
               stop; stopping mid-seam is not, and a gate is the only place
               the supervisor can stop anything at all. *)
            permit_ok := leak_armed AND NOT a_leak_contact;

        77:     (* G10 LEAK_PORT_END *)
            permit_ok := leak_armed;

        90:     (* G11 SESSION_END *)
            permit_ok := TRUE;
    ELSE
        (* An unknown gate is not permitted. Failing closed is the only
           safe default for a supervisor. *)
        permit_ok := FALSE;
    END_CASE;
END_IF;

(* Echo phase and sequence, then permit. Written every scan so that a
   permit is always consistent with the gate it refers to. *)
a_permit_phase := a_phase_code;
a_permit_seq := a_phase_seq;
a_permit := permit_ok;
a_phase_ack := a_phase_done;

a_run := (cell_state = 1) OR (cell_state = 2);
a_hold := (cell_state = 3);
a_abort := (cell_state = 4) OR (cell_state = 7);

(* =====================================================================
   5. PUBLISH TO THE OPERATOR PANEL
   ===================================================================== *)
b_cell_state := cell_state;
b_current_step := a_phase_code;
b_qds_installed := a_qds_installed;
b_leak_tested := a_leak_tested;
b_active_fault_code := fault_code;
b_leak_phase_active := a_active_arm_leak;
b_leak_armed := leak_armed;

END_PROGRAM


CONFIGURATION Config0
    RESOURCE Res0 ON PLC
        TASK task0(INTERVAL := T#50ms, PRIORITY := 0);
        PROGRAM instance0 WITH task0 : prog0;
    END_RESOURCE
END_CONFIGURATION
