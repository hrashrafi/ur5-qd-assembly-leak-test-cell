/* Operator panel front end.
 *
 * Polls one JSON endpoint and paints the result. No state of its own:
 * everything shown comes from the PLC, so a reload can never disagree with
 * the cell. Buttons POST a command and let the next poll confirm it, rather
 * than optimistically drawing a state the PLC has not actually reached - on a
 * machine panel, showing Running before the cell is running is worse than
 * showing it a quarter second late.
 */

const $ = (id) => document.getElementById(id);

const STATE_CLASS = {
  IDLE: "", STARTING: "run", RUNNING: "run", HOLD: "hold",
  FAULTING: "fault", FAULTED: "fault", COMPLETE: "done", ABORTING: "hold",
};

/* ---------- commands ---------- */
async function send(tag) {
  try {
    const r = await fetch("/api/cmd", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ tag, value: 1 }),
    });
    if (!r.ok) {
      const body = await r.json().catch(() => ({}));
      console.warn("command refused:", tag, body.error || r.status);
    }
  } catch (e) {
    console.warn("command failed:", tag, e);
  }
}

$("asm-start").onclick  = () => send("HMI_START");
$("asm-stop").onclick   = () => send("HMI_STOP");
$("leak-start").onclick = () => send("HMI_START_LEAK");
$("leak-stop").onclick  = () => send("HMI_STOP_LEAK");
$("btn-reset").onclick  = () => send("HMI_RESET");

/* ---------- painting ---------- */
const built = {};

function paintJob(key, job, els, label) {
  if (!built[key]) {
    $(els.steps).innerHTML = job.steps.map((s) => `<li>${s}</li>`).join("");
    built[key] = true;
  }
  document.querySelectorAll(`#${els.steps} li`).forEach((li, i) => {
    li.classList.toggle("active", i === job.active);
    li.classList.toggle("done", job.complete || (job.active > i));
  });

  $(els.done).textContent = job.done;
  $(els.bar).style.width = (100 * job.done / job.total) + "%";

  const st = $(els.state);
  st.textContent = label.text;
  st.className = "jobstate " + label.cls;
  $(els.card).classList.toggle("live", job.running);
}

function assemblyLabel(job, d) {
  if (job.running) return { text: "Running", cls: "run" };
  if (job.complete) return { text: "Complete", cls: "done" };
  if (d.state.name === "FAULTED" || d.state.name === "FAULTING")
    return { text: "Faulted", cls: "fault" };
  if (d.state.name === "HOLD") return { text: "Stopped", cls: "hold" };
  return { text: "Idle", cls: "" };
}

function leakLabel(job, d) {
  if (job.running) return { text: "Running", cls: "run" };
  if (job.complete) return { text: "Complete", cls: "done" };
  if (d.state.name === "FAULTED" || d.state.name === "FAULTING")
    return { text: "Faulted", cls: "fault" };
  // Armed but not yet running is the normal case while assembly is still
  // going: the leak arm is waiting its turn at gate G8, which is a real
  // state an operator should be able to see rather than guess at.
  if (job.armed) return { text: "Waiting", cls: "hold" };
  return { text: "Idle", cls: "" };
}

function paint(d) {
  const state = d.state.name || "----";
  const pill = $("statepill");
  pill.textContent = state;
  pill.className = "statepill " + (STATE_CLASS[state] || "");

  $("dot-plc").classList.toggle("on", !!d.connected);
  $("dot-sim").classList.toggle("on", !!d.sim_comm_ok);

  paintJob("asm", d.assembly, {
    steps: "asm-steps", done: "asm-done", bar: "asm-bar",
    state: "asm-state", card: "job-assembly",
  }, assemblyLabel(d.assembly, d));

  paintJob("leak", d.leak, {
    steps: "leak-steps", done: "leak-done", bar: "leak-bar",
    state: "leak-state", card: "job-leak",
  }, leakLabel(d.leak, d));

  /* Buttons follow the PLC, not the click. */
  // Either Stop holds the whole cell, and either Start resumes it, so in
  // HOLD both Starts are live and both Stops are spent.
  $("asm-start").disabled  = d.assembly.running || !d.safe_to_start
                             || d.assembly.complete;
  $("asm-stop").disabled   = !d.assembly.running;
  $("leak-start").disabled = d.leak.complete || d.fault.code !== 0
                             || (d.leak.armed && !d.holding);
  // Live whenever the section is armed: before the arm starts this cancels
  // the request, once it is running it stops the arm.
  $("leak-stop").disabled  = !d.leak.armed || (d.holding && !d.leak.running);

  const alarm = $("alarm");
  const active = d.fault.code !== 0;
  alarm.classList.toggle("active", active);
  let text = active ? `${d.fault.code} — ${d.fault.text}` : "— none —";
  // A fault the robot is not around to confirm cannot be cleared silently -
  // say so, rather than leaving the operator pressing a live button that
  // does nothing.
  if (active && !d.sim_comm_ok) text += "  ·  robot offline";
  $("alarm-text").textContent = text;

  // Reset returns the cell to Idle from anywhere it is not already there.
  // One rule rather than four: an operator should not have to work out
  // which button this cell will accept right now.
  $("btn-reset").disabled = d.idle;
  $("btn-reset").title = active
    ? "Clear the alarm and return to Idle"
    : "Abandon the run and return to Idle (Start resumes instead)";
}

async function poll() {
  try {
    paint(await (await fetch("/api/state")).json());
  } catch (e) {
    $("dot-plc").classList.remove("on");
    $("dot-sim").classList.remove("on");
    $("statepill").textContent = "NO LINK";
    $("statepill").className = "statepill fault";
  }
}

setInterval(poll, 300);
poll();
