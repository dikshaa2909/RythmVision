# RhythmVision

Real-time dance and movement analysis using computer vision. RhythmVision
tracks your full-body pose through a webcam, computes live **energy**,
**rhythm**, and **balance** metrics, detects your **emotion** from facial
landmarks, classifies your movement into a dance-style label, and can
stream the raw pose data over UDP to Unity for 3D avatar animation.

This is a rebuilt, "actually usable" version of the original RhythmVision
prototype — same core idea, but fixed bugs, a real emotion detector
(the original had the FaceMesh model wired up but never used it), a
proper dashboard UI, session recording, and a project structure you can
extend without wading through one giant script.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

**Important:** `requirements.txt` pins `mediapipe==0.10.14`. Newer
mediapipe releases (0.10.2x+) dropped the legacy `mp.solutions` API this
project is built on — `pip install mediapipe` on its own will give you a
version that breaks with `AttributeError: module 'mediapipe' has no
attribute 'solutions'`. Stick to the pinned version unless you rewrite
`pose_analyzer.py` / `emotion_analyzer.py` against MediaPipe's newer
Tasks API.

## Controls

| Key | Action |
|-----|--------|
| `q` | Quit (shows a session summary first) |
| `v` | Cycle visualization mode (Skeleton / Cartoon / Mesh / Artistic Trails) |
| `m` | Toggle the metrics dashboard |
| `r` | Reset counters and recalibrate the emotion baseline |
| `c` | Save a screenshot to `rhythmvision_sessions/` |
| `e` | Export a JSON session report to `rhythmvision_sessions/` |
| `u` | Toggle UDP streaming to Unity on/off |
| `space` | Pause / resume tracking |

On launch, hold a neutral face for ~1 second while the emotion baseline
calibrates (progress shown in the dashboard) — after that, "happy",
"surprised", "sad", and "neutral" are detected relative to your own
neutral expression, not a generic average face.

## Project structure

```
RhythmVision/
├── main.py              # App loop: camera capture, orchestration, key handling
├── config.py             # All tunable constants (thresholds, colors, ports)
├── pose_analyzer.py      # Energy / rhythm / balance metrics + move classification
├── emotion_analyzer.py   # Calibrated, heuristic FaceMesh emotion detection
├── visualizer.py         # Dashboard UI, sparklines, 4 pose-art render modes
├── network.py            # UDP streamer to Unity (JSON payloads, throttled)
├── session_logger.py     # Screenshot + session report export
└── requirements.txt
```

## How the metrics work

- **Energy**: total Euclidean displacement of all 33 pose landmarks
  between consecutive frames, scaled to 0–100.
- **Rhythm**: inverse of the variance in movement over the last 5
  frames — steady, repeated motion scores higher than erratic motion.
- **Balance**: ratio of left-side vs. right-side body movement (closer
  to 100 = more symmetric).
- **Emotion**: heuristic, not ML. After a short calibration phase, each
  frame's mouth openness, mouth-corner lift, and eyebrow height are
  compared against your calibrated neutral baseline (all normalized by
  interocular distance so it holds up at different camera distances).
  The thresholds live in `config.py` — nudge them if detection feels
  off for your face/lighting.

## Unity integration

With streaming enabled (default, toggle with `u`), RhythmVision sends a
JSON UDP packet to `127.0.0.1:5050` roughly 30 times/second:

```json
{
  "landmarks": { "0": {"x": 0.51, "y": 0.32, "z": -0.02}, "...": "...33 total" },
  "emotion": "happy",
  "timestamp": 1755500000.123
}
```

To receive this in Unity, add a `UdpClient` listener on port 5050,
parse the JSON (e.g. with `Newtonsoft.Json` or Unity's `JsonUtility`
with a matching wrapper class), and map each landmark index to a bone
or IK target on your avatar rig. This repo doesn't include a Unity
project — the receiver side depends heavily on your specific avatar
rig, so it's left for you to wire up against your own character.

## Known limitations / future work

- Emotion detection is a hand-tuned heuristic, not a trained classifier
  — it's good for a demo but will misfire on strong head turns or
  unusual lighting.
- Rhythm scoring reacts to variance in raw movement, not actual beat
  detection — it doesn't listen to music. A real "sync to the beat"
  feature would need audio input and onset detection.
- Single-person tracking only (MediaPipe Pose's default mode).
- No persistent history across sessions beyond the exported JSON
  reports in `rhythmvision_sessions/`.

# RythmVision

# RythmVision
