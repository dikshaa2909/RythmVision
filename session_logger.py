"""
RhythmVision — Session Logger
Handles saving screenshots and exporting a JSON session report to disk,
so a practice session leaves behind something you can actually look at
afterwards.
"""

import json
import os
import time

import cv2

import config as cfg


class SessionLogger:
    def __init__(self, output_dir=cfg.OUTPUT_DIR):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def save_screenshot(self, frame):
        filename = f"screenshot_{time.strftime('%Y%m%d_%H%M%S')}.png"
        path = os.path.join(self.output_dir, filename)
        cv2.imwrite(path, frame)
        return path

    def export_report(self, summary_dict):
        filename = f"session_{time.strftime('%Y%m%d_%H%M%S')}.json"
        path = os.path.join(self.output_dir, filename)
        with open(path, "w") as f:
            json.dump(summary_dict, f, indent=2)
        return path
