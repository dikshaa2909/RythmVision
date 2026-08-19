"""
RhythmVision — Visualizer
All drawing/rendering lives here: the four pose-art modes, the metrics
dashboard, sparkline graphs, and the calibration / session-summary
screens. Keeping this separate from main.py means the "look" of the app
can be restyled without touching any tracking logic.
"""

import cv2
import numpy as np

import config as cfg

FONT = cv2.FONT_HERSHEY_SIMPLEX


# ---------------------------------------------------------------------
# Pose art modes
# ---------------------------------------------------------------------

class PoseArtist:
    """Renders the four pose visualization modes."""

    def __init__(self, mp_drawing, mp_pose):
        self.mp_drawing = mp_drawing
        self.mp_pose = mp_pose
        self.modes = ["Skeleton", "Cartoon", "Mesh", "Artistic Trails"]

    def draw(self, mode_index, frame, landmarks, w, h):
        if mode_index == 0:
            return self._skeleton(frame, landmarks)
        elif mode_index == 1:
            return self._cartoon(frame, landmarks, w, h)
        elif mode_index == 2:
            return self._mesh(frame, landmarks, w, h)
        else:
            return self._artistic(frame, landmarks, w, h)

    def _skeleton(self, image, landmarks):
        canvas = np.zeros_like(image)
        self.mp_drawing.draw_landmarks(
            canvas,
            landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(150, 220, 100), thickness=2, circle_radius=3),
            self.mp_drawing.DrawingSpec(color=(200, 130, 235), thickness=2),
        )
        return canvas

    def _cartoon(self, image, landmarks, w, h):
        canvas = np.full_like(image, 250)
        points = {i: (int(lm.x * w), int(lm.y * h)) for i, lm in enumerate(landmarks.landmark)}
        for start, end in self.mp_pose.POSE_CONNECTIONS:
            if start in points and end in points:
                cv2.line(canvas, points[start], points[end], (100, 90, 235), 6, cv2.LINE_AA)
        for idx in points:
            cv2.circle(canvas, points[idx], 4, (40, 40, 40), -1, cv2.LINE_AA)
        return canvas

    def _mesh(self, image, landmarks, w, h):
        canvas = np.zeros_like(image)
        points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks.landmark]
        for i, (p1, p2) in enumerate(self.mp_pose.POSE_CONNECTIONS):
            color = (i * 5 % 255, 255 - (i * 5 % 255), 180)
            cv2.line(canvas, points[p1], points[p2], color, 3, cv2.LINE_AA)
        return canvas

    def _artistic(self, image, landmarks, w, h):
        canvas = np.zeros_like(image)
        for lm in landmarks.landmark:
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(canvas, (x, y), 6, (235, 220, 60), -1, cv2.LINE_AA)
            cv2.circle(canvas, (x, y), 10, (235, 220, 60), 1, cv2.LINE_AA)
        return canvas


# ---------------------------------------------------------------------
# Dashboard / HUD
# ---------------------------------------------------------------------

def _rounded_panel(image, x1, y1, x2, y2, color, alpha=0.55):
    overlay = image.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)


def _progress_bar(image, x, y, width, height, value, label, color):
    cv2.rectangle(image, (x, y), (x + width, y + height), (60, 55, 55), -1, cv2.LINE_AA)
    fill_w = int(width * max(0, min(100, value)) / 100)
    if fill_w > 0:
        cv2.rectangle(image, (x, y), (x + fill_w, y + height), color, -1, cv2.LINE_AA)
    cv2.rectangle(image, (x, y), (x + width, y + height), (90, 85, 85), 1, cv2.LINE_AA)
    cv2.putText(image, f"{label} {value}%", (x, y - 6), FONT, 0.45, cfg.COLOR_TEXT, 1, cv2.LINE_AA)


def _sparkline(image, x, y, width, height, values, color):
    cv2.rectangle(image, (x, y), (x + width, y + height), (45, 40, 40), -1, cv2.LINE_AA)
    if len(values) < 2:
        return
    vmax = max(values) if max(values) > 0 else 1
    pts = []
    n = len(values)
    for i, v in enumerate(values):
        px = x + int(i / (n - 1) * width)
        py = y + height - int(min(1.0, v / vmax) * height)
        pts.append((px, py))
    for i in range(len(pts) - 1):
        cv2.line(image, pts[i], pts[i + 1], color, 2, cv2.LINE_AA)


def draw_dashboard(frame, stats, sparkline_data, mode_name, unity_enabled, fps):
    """
    stats: dict with energy, rhythm, balance, movement_count, move_label,
           move_icon, emotion_label, emotion_emoji, calibrating (bool),
           calibration_progress (0-1)
    """
    h, w = frame.shape[:2]
    panel_w = 260
    _rounded_panel(frame, 10, 10, 10 + panel_w, 230, cfg.COLOR_BG_PANEL, alpha=0.6)

    cv2.putText(frame, "RhythmVision", (24, 38), FONT, 0.75, cfg.COLOR_ACCENT, 2, cv2.LINE_AA)
    cv2.putText(frame, f"{fps:.0f} FPS  |  {mode_name}", (24, 58), FONT, 0.42, cfg.COLOR_TEXT_DIM, 1, cv2.LINE_AA)

    _progress_bar(frame, 24, 85, panel_w - 28, 14, stats["energy"], "Energy", cfg.COLOR_ACCENT)
    _progress_bar(frame, 24, 120, panel_w - 28, 14, stats["rhythm"], "Rhythm", cfg.COLOR_ACCENT_2)
    _progress_bar(frame, 24, 155, panel_w - 28, 14, stats["balance"], "Balance", cfg.COLOR_GOOD)

    cv2.putText(
        frame,
        f"{stats['move_icon']}  {stats['move_label']}",
        (24, 195), FONT, 0.55, cfg.COLOR_TEXT, 2, cv2.LINE_AA,
    )

    if stats.get("calibrating"):
        progress_txt = f"Calibrating emotion baseline... {int(stats['calibration_progress'] * 100)}%  (hold a neutral face)"
        cv2.putText(frame, progress_txt, (24, 216), FONT, 0.4, cfg.COLOR_WARN, 1, cv2.LINE_AA)
    else:
        cv2.putText(
            frame,
            f"Mood: {stats['emotion_emoji']}  {stats['emotion_label'].capitalize()}",
            (24, 216), FONT, 0.5, cfg.COLOR_TEXT, 1, cv2.LINE_AA,
        )

    # Sparkline of recent movement/energy, top-right
    spark_w, spark_h = 180, 50
    sx, sy = w - spark_w - 20, 20
    _rounded_panel(frame, sx - 8, sy - 8, sx + spark_w + 8, sy + spark_h + 24, cfg.COLOR_BG_PANEL, alpha=0.55)
    cv2.putText(frame, "Movement", (sx, sy + 12), FONT, 0.42, cfg.COLOR_TEXT_DIM, 1, cv2.LINE_AA)
    _sparkline(frame, sx, sy + 18, spark_w, spark_h, sparkline_data, cfg.COLOR_ACCENT)

    # Unity status pill, bottom-right of panel area
    status_color = cfg.COLOR_GOOD if unity_enabled else cfg.COLOR_TEXT_DIM
    status_txt = "Unity: ON" if unity_enabled else "Unity: OFF"
    cv2.putText(frame, status_txt, (w - 140, h - 16), FONT, 0.45, status_color, 1, cv2.LINE_AA)

    return frame


def draw_controls_footer(frame, controls_text):
    h, w = frame.shape[:2]
    cv2.putText(frame, controls_text, (16, h - 16), FONT, 0.42, cfg.COLOR_TEXT_DIM, 1, cv2.LINE_AA)
    return frame


def draw_no_pose(frame):
    h, w = frame.shape[:2]
    canvas = np.zeros_like(frame)
    text = "No pose detected — step into frame"
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.7, 2)
    cv2.putText(canvas, text, ((w - tw) // 2, h // 2), FONT, 0.7, (100, 100, 240), 2, cv2.LINE_AA)
    return canvas


def draw_startup_screen(w, h):
    canvas = np.full((h, w, 3), 20, dtype=np.uint8)
    title = "RhythmVision"
    sub = "Real-time dance & movement analysis"
    (tw, th), _ = cv2.getTextSize(title, FONT, 1.4, 3)
    cv2.putText(canvas, title, ((w - tw) // 2, h // 2 - 20), FONT, 1.4, cfg.COLOR_ACCENT, 3, cv2.LINE_AA)
    (sw, sh), _ = cv2.getTextSize(sub, FONT, 0.6, 1)
    cv2.putText(canvas, sub, ((w - sw) // 2, h // 2 + 20), FONT, 0.6, cfg.COLOR_TEXT_DIM, 1, cv2.LINE_AA)
    hint = "Stand back so your full body is visible, then hold a neutral face to calibrate."
    (hw, hh), _ = cv2.getTextSize(hint, FONT, 0.45, 1)
    cv2.putText(canvas, hint, ((w - hw) // 2, h // 2 + 60), FONT, 0.45, cfg.COLOR_TEXT_DIM, 1, cv2.LINE_AA)
    return canvas


def draw_session_summary(w, h, summary):
    canvas = np.full((h, w, 3), 20, dtype=np.uint8)
    lines = [
        ("Session Summary", 1.0, cfg.COLOR_ACCENT),
        (f"Duration: {summary['duration_seconds']}s", 0.6, cfg.COLOR_TEXT),
        (f"Peak Energy: {summary['peak_energy']}%", 0.6, cfg.COLOR_TEXT),
        (f"Average Rhythm: {summary['average_rhythm']}%", 0.6, cfg.COLOR_TEXT),
        (f"Final Balance: {summary['final_balance_score']}%", 0.6, cfg.COLOR_TEXT),
        (f"Movements Counted: {summary['movement_count']}", 0.6, cfg.COLOR_TEXT),
        (f"Most Common Move: {summary['most_common_move']}", 0.6, cfg.COLOR_TEXT),
        ("Press any key to exit", 0.45, cfg.COLOR_TEXT_DIM),
    ]
    y = h // 2 - 140
    for text, scale, color in lines:
        (tw, th), _ = cv2.getTextSize(text, FONT, scale, 2)
        cv2.putText(canvas, text, ((w - tw) // 2, y), FONT, scale, color, 2, cv2.LINE_AA)
        y += int(45 * scale) + 20
    return canvas
