"""
RhythmVision — Pose Analyzer
Turns raw MediaPipe pose landmarks into movement metrics: energy, rhythm,
balance, and a dynamically classified "dance move" label.
"""

import math
import time
from collections import deque

import config as cfg


class PoseAnalyzer:
    """Computes movement metrics from consecutive pose landmark frames."""

    def __init__(self):
        self.previous_landmarks = None
        self.movement_history = deque(maxlen=cfg.MOVEMENT_HISTORY_LEN)

        self.energy_level = 0
        self.rhythm_score = 0
        self.balance_score = 0
        self.movement_count = 0

        self.current_move = "Waiting for pose..."
        self.current_move_icon = "🕺"

        # Session-wide stats
        self.energy_peak = 0
        self.rhythm_sum = 0
        self.rhythm_samples = 0
        self.move_counter = {}
        self.session_start = time.time()

    def reset(self):
        """Reset all counters (keeps session_start untouched)."""
        self.previous_landmarks = None
        self.movement_history.clear()
        self.energy_level = 0
        self.rhythm_score = 0
        self.balance_score = 0
        self.movement_count = 0
        self.energy_peak = 0
        self.rhythm_sum = 0
        self.rhythm_samples = 0
        self.move_counter = {}

    def update(self, landmarks):
        """
        landmarks: list of MediaPipe NormalizedLandmark objects (33 pose points)
        Returns True if metrics were updated (i.e. we had a previous frame
        to diff against).
        """
        if self.previous_landmarks is None:
            self.previous_landmarks = landmarks
            return False

        total_movement = 0.0
        left_movement = 0.0
        right_movement = 0.0

        for i, (curr, prev) in enumerate(zip(landmarks, self.previous_landmarks)):
            dx = curr.x - prev.x
            dy = curr.y - prev.y
            dz = curr.z - prev.z
            movement = math.sqrt(dx * dx + dy * dy + dz * dz)
            total_movement += movement

            if i in cfg.LEFT_SIDE_IDX:
                left_movement += movement
            elif i in cfg.RIGHT_SIDE_IDX:
                right_movement += movement

        self.previous_landmarks = landmarks
        self.movement_history.append(total_movement)

        # --- Energy ---
        self.energy_level = min(100, int(total_movement * cfg.ENERGY_SCALE))
        self.energy_peak = max(self.energy_peak, self.energy_level)

        # --- Rhythm (lower variance in recent movement => steadier rhythm) ---
        if len(self.movement_history) >= cfg.RHYTHM_WINDOW:
            recent = list(self.movement_history)[-cfg.RHYTHM_WINDOW:]
            avg = sum(recent) / len(recent)
            variance = sum((m - avg) ** 2 for m in recent) / len(recent)
            self.rhythm_score = max(
                0, min(100, int(100 - variance * cfg.RHYTHM_VARIANCE_SCALE))
            )
            self.rhythm_sum += self.rhythm_score
            self.rhythm_samples += 1

        # --- Balance ---
        if left_movement + right_movement > 0:
            balance_ratio = min(left_movement, right_movement) / max(
                left_movement, right_movement
            )
            self.balance_score = int(balance_ratio * 100)

        if total_movement > cfg.MOVEMENT_COUNT_THRESHOLD:
            self.movement_count += 1

        self._classify_move()
        return True

    def _classify_move(self):
        for threshold, label, icon in cfg.DANCE_MOVE_THRESHOLDS:
            if self.energy_level >= threshold:
                self.current_move = label
                self.current_move_icon = icon
                break
        self.move_counter[self.current_move] = (
            self.move_counter.get(self.current_move, 0) + 1
        )

    def energy_history(self):
        """Returns the raw movement history, useful for sparkline drawing."""
        return list(self.movement_history)

    def summary(self):
        """Session-level summary stats, used for the end-of-session report."""
        elapsed = time.time() - self.session_start
        avg_rhythm = (
            self.rhythm_sum / self.rhythm_samples if self.rhythm_samples else 0
        )
        most_common_move = (
            max(self.move_counter, key=self.move_counter.get)
            if self.move_counter
            else "N/A"
        )
        return {
            "duration_seconds": round(elapsed, 1),
            "peak_energy": self.energy_peak,
            "average_rhythm": round(avg_rhythm, 1),
            "final_balance_score": self.balance_score,
            "movement_count": self.movement_count,
            "most_common_move": most_common_move,
            "move_breakdown": self.move_counter,
        }
