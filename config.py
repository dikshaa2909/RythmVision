"""
RhythmVision — Configuration
All tunable constants live here so behavior can be adjusted without
touching logic code.
"""

# ---------- Camera ----------
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# ---------- MediaPipe ----------
POSE_MODEL_COMPLEXITY = 1
POSE_MIN_DETECTION_CONFIDENCE = 0.5
POSE_MIN_TRACKING_CONFIDENCE = 0.5

FACE_MAX_NUM_FACES = 1
FACE_MIN_DETECTION_CONFIDENCE = 0.5
FACE_REFINE_LANDMARKS = True

# ---------- Movement metrics ----------
MOVEMENT_HISTORY_LEN = 30
ENERGY_SCALE = 4000          # multiplier: raw displacement -> 0-100 energy
RHYTHM_WINDOW = 5            # frames used for rhythm variance calc
RHYTHM_VARIANCE_SCALE = 20000
MOVEMENT_COUNT_THRESHOLD = 0.01

# Landmark index groups (MediaPipe Pose, 33 landmarks) used for L/R balance
LEFT_SIDE_IDX = [11, 13, 15, 23, 25, 27, 29, 31]
RIGHT_SIDE_IDX = [12, 14, 16, 24, 26, 28, 30, 32]

# ---------- Dance move classification thresholds (energy %) ----------
DANCE_MOVE_THRESHOLDS = [
    (80, "High Energy Dance", "🔥"),
    (60, "Moderate Flow", "💃"),
    (30, "Gentle Movement", "🌊"),
    (10, "Subtle Motion", "🍃"),
    (0,  "Rest Pose", "🧘"),
]

# ---------- Emotion detection (heuristic, FaceMesh landmarks) ----------
# Key landmark indices (MediaPipe FaceMesh, 468 points)
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
MOUTH_TOP = 13
MOUTH_BOTTOM = 14
UPPER_LIP_CENTER = 0
LOWER_LIP_CENTER = 17
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374
LEFT_EYEBROW = 105
RIGHT_EYEBROW = 334
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263
NOSE_TIP = 1

EMOTION_HISTORY_LEN = 10          # frames to smooth emotion over (majority vote)
EMOTION_CALIBRATION_FRAMES = 30   # frames of "neutral face" used to build baseline

# All thresholds below are deltas from the calibrated neutral baseline,
# normalized by interocular (eye-to-eye) distance so they stay roughly
# scale-invariant regardless of distance from the camera. They are
# heuristic starting points — nudge them if detection feels off for you.
MOUTH_OPEN_THRESHOLD = 0.12       # mouth_height delta -> mouth opened (surprised)
EYEBROW_RAISE_THRESHOLD = 0.06    # eyebrow-to-eye gap delta -> eyebrows raised (surprised)
SMILE_RATIO_THRESHOLD = 0.06      # mouth width / corner-lift delta -> happy
MOUTH_FROWN_THRESHOLD = -0.02     # corner-lift delta (negative = corners drooping) -> sad

EMOTIONS = {
    "happy":     "😊",
    "surprised": "😮",
    "sad":       "😔",
    "neutral":   "😐",
}

# ---------- Networking (Unity UDP) ----------
UNITY_HOST = "127.0.0.1"
UNITY_PORT = 5050
UNITY_ENABLED_DEFAULT = True
UNITY_SEND_INTERVAL = 1 / 30      # throttle sends to ~30/sec max

# ---------- UI Theme ----------
COLOR_BG_PANEL = (28, 22, 20)         # BGR, dark plum
COLOR_ACCENT = (219, 112, 147)        # BGR, orchid/pink accent
COLOR_ACCENT_2 = (235, 170, 100)      # BGR, soft blue accent
COLOR_TEXT = (245, 245, 245)
COLOR_TEXT_DIM = (170, 170, 170)
COLOR_GOOD = (120, 220, 120)
COLOR_WARN = (100, 190, 250)
COLOR_BAD = (100, 100, 240)

FONT = "FONT_HERSHEY_SIMPLEX"

# ---------- Session ----------
OUTPUT_DIR = "rhythmvision_sessions"
