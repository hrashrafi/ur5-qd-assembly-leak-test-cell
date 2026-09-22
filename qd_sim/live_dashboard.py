"""The interactive live window for scripts/run_cell.py --live - the way to run
and watch the robot; the recorded .mp4 is the way to review a run afterwards.
It is also the vision system's own display in the PLC cell.

ONE window, split in half: the left half shows the third-person camera
feed of the robot moving, live; the right half shows the active arm's camera
(the wrist camera during assembly, the leak camera during leak testing) with
detection results drawn on AS THEY HAPPEN - a QD's hex outline, a hole's
circle, which ports are filled - and a telemetry + control panel along the
bottom (current phase, which QD/port is being handled, tracked position vs.
reference, tilt, contact counts, and - before a standalone session starts - a
clickable START button, since the robot shouldn't move the instant the script
launches; the operator decides when via START or 's'). Press 'q'/Esc at any
point to abort.

This module only knows how to draw the window and track click/key state - it
has no idea what a QD or a port is. The session supplies already-rendered
frames, already-computed HUD text lines, and already-classified shapes/labels,
drawn with the shared convention in qd_sim/vision/annotate.py.
"""

import cv2
import numpy as np

from qd_sim.vision.annotate import COAST_TEXT_COLOR, annotate

WINDOW_NAME = "QD Sim - Live"

FONT = cv2.FONT_HERSHEY_SIMPLEX
PANEL_BG = (30, 30, 30)
PANEL_TEXT = (230, 230, 230)
PANEL_HEADING = (0, 220, 60)
BUTTON_FILL = (0, 150, 40)
BUTTON_FILL_HOVER = (0, 200, 60)
BUTTON_TEXT = (255, 255, 255)
BUTTON_BORDER = (255, 255, 255)
DIVIDER_COLOR = (70, 70, 70)


class LiveDashboard:
    """right_w/wrist_display_h size the right half's wrist-cam image;
    panel_h is the telemetry/control strip under it. The left half (the
    third-person robot view) is letterboxed to the same total height,
    preserving its own aspect ratio rather than stretching it, at
    left_w width - so half the window is the robot view and half is the
    camera + controls, without distorting either feed's real proportions."""

    def __init__(self, left_w=800, right_w=800, wrist_display_h=600, panel_h=260,
                 record_path=None, record_fps=30, exit_button=False):
        self.left_w = left_w
        self.right_w = right_w
        self.wrist_display_h = wrist_display_h
        self.panel_h = panel_h
        self.total_h = wrist_display_h + panel_h
        self.start_requested = False
        self.quit_requested = False
        # An on-screen EXIT, drawn over the third-person view so the machine
        # can be shut down from the window you are actually watching rather
        # than by remembering a keystroke. Off by default for a standalone
        # --live run; the PLC cell turns it on.
        self.exit_button = exit_button
        self.exit_requested = False
        self._exit_rect = None
        self._button_rect = None  # (x0, y0, x1, y1) in full-window pixel coords
        self._mouse_pos = (-1, -1)
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, self._on_mouse)
        # Optional: save this exact composed window (both halves, overlay,
        # telemetry panel - everything actually shown) to a video file as it
        # runs. This is separate from the plain third-person recording the
        # session's --out makes: it captures the LIVE DASHBOARD itself,
        # exactly as seen. Same frame size as the on-screen canvas, so what's
        # recorded is pixel-for-pixel what was shown.
        self._writer = (cv2.VideoWriter(str(record_path), cv2.VideoWriter_fourcc(*"avc1"),
                                         record_fps, (left_w + right_w, self.total_h))
                         if record_path is not None else None)

    def _on_mouse(self, event, x, y, flags, _userdata):
        self._mouse_pos = (x, y)
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if self._button_rect is not None:
            x0, y0, x1, y1 = self._button_rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                self.start_requested = True
                return
        if self._exit_rect is not None:
            x0, y0, x1, y1 = self._exit_rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                self.exit_requested = True
                self.quit_requested = True

    def _poll_keys(self, wait_ms):
        key = cv2.waitKey(wait_ms) & 0xFF
        if key in (27, ord("q"), ord("Q")):
            self.quit_requested = True
        return key

    def _render_left(self, third_person_img_rgb):
        """Letterbox the (large, video-resolution) third-person render
        into the left_w x total_h cell, preserving its own aspect ratio -
        centered vertically, black bars filling the rest."""
        bgr = cv2.cvtColor(third_person_img_rgb, cv2.COLOR_RGB2BGR)
        src_h, src_w = bgr.shape[:2]
        scale = self.left_w / src_w
        disp_h = min(self.total_h, int(src_h * scale))
        disp = cv2.resize(bgr, (self.left_w, disp_h), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((self.total_h, self.left_w, 3), dtype=np.uint8)
        y0 = (self.total_h - disp_h) // 2
        canvas[y0:y0 + disp_h, :, :] = disp
        return canvas

    def _render_right(self, wrist_bgr, hud_lines, button_label):
        wrist_disp = cv2.resize(wrist_bgr, (self.right_w, self.wrist_display_h),
                                 interpolation=cv2.INTER_NEAREST)
        panel = np.full((self.panel_h, self.right_w, 3), PANEL_BG, dtype=np.uint8)
        y = 24
        for line, color in hud_lines:
            cv2.putText(panel, line, (14, y), FONT, 0.55, color, 1, cv2.LINE_AA)
            y += 24
        button_rect_in_panel = None
        if button_label is not None:
            bw, bh = 220, 50
            bx0 = (self.right_w - bw) // 2
            by0 = self.panel_h - bh - 14
            bx1, by1 = bx0 + bw, by0 + bh
            mx, my = self._mouse_pos
            my_panel = my - self.wrist_display_h  # mouse coords are window-absolute; panel starts below the wrist image
            hovered = bx0 <= mx <= bx1 and by0 <= my_panel <= by1
            cv2.rectangle(panel, (bx0, by0), (bx1, by1),
                          BUTTON_FILL_HOVER if hovered else BUTTON_FILL, -1)
            cv2.rectangle(panel, (bx0, by0), (bx1, by1), BUTTON_BORDER, 2)
            (tw, th), _ = cv2.getTextSize(button_label, FONT, 0.85, 2)
            cv2.putText(panel, button_label, (bx0 + (bw - tw) // 2, by0 + (bh + th) // 2),
                        FONT, 0.85, BUTTON_TEXT, 2, cv2.LINE_AA)
            button_rect_in_panel = (bx0, by0, bx1, by1)
        right_col = np.vstack([wrist_disp, panel])
        return right_col, button_rect_in_panel

    def _compose(self, third_person_img_rgb, wrist_bgr, hud_lines, button_label=None):
        left = self._render_left(third_person_img_rgb)
        right, button_rect_in_panel = self._render_right(wrist_bgr, hud_lines, button_label)
        if button_rect_in_panel is not None:
            bx0, by0, bx1, by1 = button_rect_in_panel
            self._button_rect = (self.left_w + bx0, self.wrist_display_h + by0,
                                  self.left_w + bx1, self.wrist_display_h + by1)
        else:
            self._button_rect = None
        canvas = np.hstack([left, right])
        cv2.line(canvas, (self.left_w, 0), (self.left_w, self.total_h), DIVIDER_COLOR, 2)
        if self.exit_button:
            self._draw_exit(canvas)
        return canvas

    def _draw_exit(self, canvas):
        # Bottom-left of the third-person half, where it cannot be confused
        # with the camera overlays on the right.
        w, h = 150, 46
        x0, y0 = 22, self.total_h - h - 22
        x1, y1 = x0 + w, y0 + h
        self._exit_rect = (x0, y0, x1, y1)
        hovered = (x0 <= self._mouse_pos[0] <= x1
                   and y0 <= self._mouse_pos[1] <= y1)
        fill = (40, 40, 190) if hovered else (28, 28, 120)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), fill, -1)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (235, 235, 245), 2)
        cv2.putText(canvas, "EXIT", (x0 + 40, y1 - 15), FONT, 0.75,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "stops the cell", (x0, y0 - 9), FONT, 0.45,
                    (200, 200, 210), 1, cv2.LINE_AA)

    def wait_for_start(self, third_person_img_rgb, wrist_bgr, subtitle_lines):
        """Block, showing a static preview, until the user clicks START
        or presses 's'. Returns False instead if the user quits ('q'/Esc)
        before starting."""
        lines = [("Session ready - click START or press 's' to begin", PANEL_HEADING)]
        lines += [(t, PANEL_TEXT) for t in subtitle_lines]
        lines.append(("(press 'q' to quit without starting)", PANEL_TEXT))
        while True:
            canvas = self._compose(third_person_img_rgb, wrist_bgr, lines, button_label="START")
            cv2.imshow(WINDOW_NAME, canvas)
            key = self._poll_keys(30)
            if self.quit_requested:
                return False
            if self.start_requested or key in (ord("s"), ord("S")):
                return True

    def update(self, third_person_img_rgb, wrist_bgr, hud_lines, shapes=(), labels=(), coast_text=None):
        """Call once per rendered frame while the session is running.
        hud_lines: list of (text, color) tuples for the telemetry panel.
        shapes/labels: same format as qd_sim.vision.annotate.annotate() -
        pass empty to show the plain wrist feed with no overlay (between
        live-tracking steps). Returns False once the user has asked to
        quit ('q'/Esc) - the caller should stop the session."""
        annotated = annotate(wrist_bgr, shapes, labels) if (shapes or labels) else wrist_bgr.copy()
        if coast_text:
            cv2.putText(annotated, coast_text, (14, 24), FONT, 0.55, COAST_TEXT_COLOR, 2, cv2.LINE_AA)
        canvas = self._compose(third_person_img_rgb, annotated, hud_lines, button_label=None)
        cv2.imshow(WINDOW_NAME, canvas)
        if self._writer is not None:
            self._writer.write(canvas)
        self._poll_keys(1)
        return not self.quit_requested

    def close(self):
        if self._writer is not None:
            self._writer.release()
        cv2.destroyAllWindows()
        # A no-op waitKey right after destroy - on some platforms (macOS
        # included) the window doesn't actually disappear until the event
        # loop is pumped one more time.
        cv2.waitKey(1)
