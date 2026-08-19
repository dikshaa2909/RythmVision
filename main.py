"""
RhythmVision — Main Application
Real-time dance/movement analysis: pose tracking, energy/rhythm/balance
metrics, heuristic emotion detection, four visualization modes, and
optional UDP streaming to Unity.

Run:
    python main.py

Controls (shown in-app too):
    q - quit                m - toggle metrics panel
    v - switch viz mode     r - reset counters + recalibrate emotion
    c - save screenshot     e - export session report (JSON)
    u - toggle Unity stream space - pause / resume tracking
"""

import logging
import sys
import time

import cv2
import mediapipe as mp

import config as cfg
from pose_analyzer import PoseAnalyzer
from emotion_analyzer import EmotionAnalyzer
from network import UnityStreamer
from session_logger import SessionLogger
import visualizer as viz

logging.basicConfig(level=logging.INFO, format="[%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("rhythmvision.main")

WINDOW_NAME = "RhythmVision"

CONTROLS_TEXT = (
    "q: quit   v: viz mode   m: metrics   r: reset+recalibrate   "
    "c: screenshot   e: export report   u: toggle Unity   space: pause"
)


class RhythmVisionApp:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.mp_face = mp.solutions.face_mesh

        self.pose = self.mp_pose.Pose(
            model_complexity=cfg.POSE_MODEL_COMPLEXITY,
            enable_segmentation=False,
            min_detection_confidence=cfg.POSE_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=cfg.POSE_MIN_TRACKING_CONFIDENCE,
        )
        self.face_mesh = self.mp_face.FaceMesh(
            static_image_mode=False,
            max_num_faces=cfg.FACE_MAX_NUM_FACES,
            refine_landmarks=cfg.FACE_REFINE_LANDMARKS,
            min_detection_confidence=cfg.FACE_MIN_DETECTION_CONFIDENCE,
        )

        self.pose_analyzer = PoseAnalyzer()
        self.emotion_analyzer = EmotionAnalyzer()
        self.streamer = UnityStreamer()
        self.session_logger = SessionLogger()
        self.artist = viz.PoseArtist(self.mp_drawing, self.mp_pose)

        self.visualization_mode = 0
        self.show_metrics = True
        self.paused = False

        self._fps_smoothed = 0.0
        self._last_frame_time = time.time()

    def _open_camera(self):
        cap = cv2.VideoCapture(cfg.CAMERA_INDEX)
        if not cap.isOpened():
            logger.error(
                "Could not open webcam at index %s. Check that no other app "
                "is using the camera, or try changing CAMERA_INDEX in config.py.",
                cfg.CAMERA_INDEX,
            )
            return None
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.FRAME_HEIGHT)
        return cap

    def _update_fps(self):
        now = time.time()
        dt = now - self._last_frame_time
        self._last_frame_time = now
        if dt > 0:
            instant_fps = 1.0 / dt
            self._fps_smoothed = (
                instant_fps if self._fps_smoothed == 0 else 0.9 * self._fps_smoothed + 0.1 * instant_fps
            )
        return self._fps_smoothed

    def _handle_key(self, key, frame):
        if key == ord("q"):
            return False
        elif key == ord("v"):
            self.visualization_mode = (self.visualization_mode + 1) % len(self.artist.modes)
        elif key == ord("m"):
            self.show_metrics = not self.show_metrics
        elif key == ord("r"):
            self.pose_analyzer.reset()
            self.emotion_analyzer.reset_calibration()
            logger.info("Counters reset, recalibrating emotion baseline.")
        elif key == ord("c"):
            path = self.session_logger.save_screenshot(frame)
            logger.info("Screenshot saved to %s", path)
        elif key == ord("e"):
            path = self.session_logger.export_report(self.pose_analyzer.summary())
            logger.info("Session report exported to %s", path)
        elif key == ord("u"):
            state = self.streamer.toggle()
            logger.info("Unity streaming %s", "enabled" if state else "disabled")
        elif key == ord(" "):
            self.paused = not self.paused
        return True

    def run(self):
        cap = self._open_camera()
        if cap is None:
            sys.exit(1)

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        h0, w0 = cfg.FRAME_HEIGHT, cfg.FRAME_WIDTH
        cv2.imshow(WINDOW_NAME, viz.draw_startup_screen(w0 * 2, h0))
        cv2.waitKey(1200)

        logger.info("RhythmVision running. Press 'q' in the window to quit.")

        running = True
        while running:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to read frame from camera; stopping.")
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            fps = self._update_fps()

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False

            if not self.paused:
                pose_results = self.pose.process(rgb)
                face_results = self.face_mesh.process(rgb)
            else:
                pose_results, face_results = None, None

            rgb.flags.writeable = True
            original = frame.copy()

            emotion_label, emotion_emoji = self.emotion_analyzer.current_emotion, self.emotion_analyzer.current_emoji
            calibrating = not self.emotion_analyzer.calibrated

            if face_results and face_results.multi_face_landmarks:
                face_landmarks = face_results.multi_face_landmarks[0].landmark
                if not self.emotion_analyzer.calibrated:
                    self.emotion_analyzer.calibrate_frame(face_landmarks)
                else:
                    emotion_label, emotion_emoji = self.emotion_analyzer.update(face_landmarks)

            if pose_results and pose_results.pose_landmarks:
                landmarks = pose_results.pose_landmarks
                self.pose_analyzer.update(landmarks.landmark)
                self.streamer.send_pose(landmarks.landmark, emotion_label)
                art = self.artist.draw(self.visualization_mode, frame, landmarks, w, h)
            else:
                art = viz.draw_no_pose(frame)

            if self.show_metrics:
                stats = {
                    "energy": self.pose_analyzer.energy_level,
                    "rhythm": self.pose_analyzer.rhythm_score,
                    "balance": self.pose_analyzer.balance_score,
                    "move_label": self.pose_analyzer.current_move,
                    "move_icon": self.pose_analyzer.current_move_icon,
                    "emotion_label": emotion_label,
                    "emotion_emoji": emotion_emoji,
                    "calibrating": calibrating,
                    "calibration_progress": self.emotion_analyzer.calibration_progress,
                }
                original = viz.draw_dashboard(
                    original, stats, self.pose_analyzer.energy_history(),
                    self.artist.modes[self.visualization_mode],
                    self.streamer.enabled, fps,
                )
            original = viz.draw_controls_footer(original, CONTROLS_TEXT)

            combined = cv2.hconcat([original, art])
            cv2.imshow(WINDOW_NAME, combined)

            key = cv2.waitKey(10) & 0xFF
            if key != 255:
                running = self._handle_key(key, combined)

        self._shutdown(cap)

    def _shutdown(self, cap):
        summary = self.pose_analyzer.summary()
        h0, w0 = cfg.FRAME_HEIGHT, cfg.FRAME_WIDTH
        cv2.imshow(WINDOW_NAME, viz.draw_session_summary(w0 * 2, h0, summary))
        cv2.waitKey(0)

        cap.release()
        cv2.destroyAllWindows()
        self.streamer.close()
        self.pose.close()
        self.face_mesh.close()
        logger.info("Session ended. %s", summary)


if __name__ == "__main__":
    RhythmVisionApp().run()
