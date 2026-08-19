"""
RhythmVision — Emotion Analyzer
Heuristic emotion detection from MediaPipe FaceMesh landmarks.

There's no ML classifier here — instead, we calibrate a short baseline of
the user's neutral face, then classify each frame by how far mouth
openness, mouth-corner lift, and eyebrow height deviate from that
baseline. It's simple, transparent, and works reasonably well for
"happy / surprised / sad / neutral" without extra model weights.
"""

import math
from collections import deque

import config as cfg


class EmotionAnalyzer:
    def __init__(self):
        self.calibrated = False
        self._calib_samples = []
        self.baseline = None

        self.emotion_history = deque(maxlen=cfg.EMOTION_HISTORY_LEN)
        self.current_emotion = "neutral"
        self.current_emoji = cfg.EMOTIONS["neutral"]

    def reset_calibration(self):
        self.calibrated = False
        self._calib_samples = []
        self.baseline = None
        self.emotion_history.clear()

    @property
    def calibration_progress(self):
        """0.0 - 1.0 progress through calibration, for UI display."""
        if self.calibrated:
            return 1.0
        return min(1.0, len(self._calib_samples) / cfg.EMOTION_CALIBRATION_FRAMES)

    def _extract_features(self, landmarks):
        def pt(idx):
            lm = landmarks[idx]
            return lm.x, lm.y

        lx, ly = pt(cfg.MOUTH_LEFT)
        rx, ry = pt(cfg.MOUTH_RIGHT)
        tx, ty = pt(cfg.MOUTH_TOP)
        bx, by = pt(cfg.MOUTH_BOTTOM)
        eye_l_x, eye_l_y = pt(cfg.LEFT_EYE_OUTER)
        eye_r_x, eye_r_y = pt(cfg.RIGHT_EYE_OUTER)
        brow_l_x, brow_l_y = pt(cfg.LEFT_EYEBROW)
        brow_r_x, brow_r_y = pt(cfg.RIGHT_EYEBROW)
        eye_top_l_x, eye_top_l_y = pt(cfg.LEFT_EYE_TOP)
        eye_top_r_x, eye_top_r_y = pt(cfg.RIGHT_EYE_TOP)

        interocular = math.hypot(eye_r_x - eye_l_x, eye_r_y - eye_l_y)
        mouth_width = math.hypot(rx - lx, ry - ly)
        mouth_height = math.hypot(bx - tx, by - ty)
        mouth_center_y = (ty + by) / 2
        corner_avg_y = (ly + ry) / 2
        # positive => corners raised above mouth center (smile-ish)
        corner_lift = mouth_center_y - corner_avg_y
        eyebrow_dist = (
            (eye_top_l_y - brow_l_y) + (eye_top_r_y - brow_r_y)
        ) / 2

        return {
            "interocular": max(interocular, 1e-6),
            "mouth_width": mouth_width,
            "mouth_height": mouth_height,
            "corner_lift": corner_lift,
            "eyebrow_dist": eyebrow_dist,
        }

    def calibrate_frame(self, landmarks):
        """Feed one frame during the calibration phase. Returns True once done."""
        feats = self._extract_features(landmarks)
        self._calib_samples.append(feats)
        if len(self._calib_samples) >= cfg.EMOTION_CALIBRATION_FRAMES:
            keys = self._calib_samples[0].keys()
            self.baseline = {
                k: sum(s[k] for s in self._calib_samples) / len(self._calib_samples)
                for k in keys
            }
            self.calibrated = True
        return self.calibrated

    def update(self, landmarks):
        """Classify emotion for the current frame. Requires calibration first."""
        if not self.calibrated:
            return self.current_emotion, self.current_emoji

        feats = self._extract_features(landmarks)
        interocular = feats["interocular"]

        mouth_width_delta = (feats["mouth_width"] - self.baseline["mouth_width"]) / interocular
        mouth_open_delta = (feats["mouth_height"] - self.baseline["mouth_height"]) / interocular
        eyebrow_delta = (feats["eyebrow_dist"] - self.baseline["eyebrow_dist"]) / interocular
        corner_delta = (feats["corner_lift"] - self.baseline["corner_lift"]) / interocular

        if mouth_open_delta > cfg.MOUTH_OPEN_THRESHOLD and eyebrow_delta > cfg.EYEBROW_RAISE_THRESHOLD:
            label = "surprised"
        elif mouth_width_delta > cfg.SMILE_RATIO_THRESHOLD or corner_delta > cfg.SMILE_RATIO_THRESHOLD:
            label = "happy"
        elif corner_delta < cfg.MOUTH_FROWN_THRESHOLD:
            label = "sad"
        else:
            label = "neutral"

        self.emotion_history.append(label)
        # Majority vote over recent frames to avoid flicker
        smoothed = max(set(self.emotion_history), key=self.emotion_history.count)
        self.current_emotion = smoothed
        self.current_emoji = cfg.EMOTIONS[smoothed]
        return self.current_emotion, self.current_emoji
