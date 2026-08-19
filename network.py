"""
RhythmVision — Unity Network Streamer
Sends pose landmark data to Unity (or any UDP listener) as JSON, with
throttling and graceful error handling instead of a silent bare `except`.
"""

import json
import socket
import time
import logging

import config as cfg

logger = logging.getLogger("rhythmvision.network")


class UnityStreamer:
    def __init__(self, host=cfg.UNITY_HOST, port=cfg.UNITY_PORT,
                 enabled=cfg.UNITY_ENABLED_DEFAULT):
        self.host = host
        self.port = port
        self.enabled = enabled
        self._sock = None
        self._last_send = 0.0
        self._error_logged = False
        if self.enabled:
            self._open_socket()

    def _open_socket(self):
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        except OSError as e:
            logger.warning("Could not open UDP socket: %s", e)
            self._sock = None

    def toggle(self):
        self.enabled = not self.enabled
        if self.enabled and self._sock is None:
            self._open_socket()
        return self.enabled

    def send_pose(self, landmarks, emotion_label=None):
        """
        landmarks: list of MediaPipe NormalizedLandmark (has .x, .y, .z)
        Throttled to UNITY_SEND_INTERVAL and safe to call every frame.
        """
        if not self.enabled or self._sock is None:
            return

        now = time.time()
        if now - self._last_send < cfg.UNITY_SEND_INTERVAL:
            return
        self._last_send = now

        try:
            payload = {
                "landmarks": {
                    str(i): {"x": lm.x, "y": lm.y, "z": lm.z}
                    for i, lm in enumerate(landmarks)
                },
                "emotion": emotion_label or "neutral",
                "timestamp": now,
            }
            message = json.dumps(payload).encode("utf-8")
            self._sock.sendto(message, (self.host, self.port))
            self._error_logged = False
        except (OSError, TypeError) as e:
            # Log once, don't spam the console every frame if Unity isn't listening
            if not self._error_logged:
                logger.warning("UDP send failed (%s). Will keep retrying quietly.", e)
                self._error_logged = True

    def close(self):
        if self._sock is not None:
            self._sock.close()
            self._sock = None
