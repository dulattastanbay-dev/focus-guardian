"""
Focus Guardian
==============
A webcam-based focus tracker. It watches for your face at your desk and,
if you wander off -- or stay put but look away for too long -- it nudges you
to get back to work. At the end of a session it prints a focus report.

No cloud, no uploads -- everything runs locally on your machine and the
video frames never leave it.

Features
--------
* YuNet DNN face detector (auto-downloaded once, ~230 KB) with a Haar
  cascade fallback, so detection stays solid through head turns and dim light.
* Gaze check: tells "looking away / down" apart from "left the desk".
* Pomodoro mode with work/break cycles.
* Real Windows desktop notifications (with a terminal-beep fallback).
* A stats dashboard over your logged sessions.

Usage
-----
    python focus_guardian.py
    python focus_guardian.py --grace 15 --goal 50 --no-window
    python focus_guardian.py --pomodoro --work 25 --break 5
    python focus_guardian.py --stats          # show the dashboard and exit

Press 'q' (with the preview window focused) or Ctrl+C to stop.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

try:
    import cv2
except ImportError:  # pragma: no cover - friendly message if dep missing
    sys.exit(
        "OpenCV is not installed. Run:  pip install -r requirements.txt"
    )

try:  # nicer beep on Windows; harmless to skip elsewhere
    import winsound
except ImportError:  # pragma: no cover - non-Windows
    winsound = None

try:  # optional: enables true eye-gaze tracking (are your eyes on the screen?)
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions as MpBaseOptions
    from mediapipe.tasks.python import vision as mp_vision
except ImportError:  # pragma: no cover - mediapipe is optional
    mp = None


SCRIPT_DIR = Path(__file__).resolve().parent
LOG_FILE = SCRIPT_DIR / "focus_log.json"
MODEL_FILE = SCRIPT_DIR / "face_detection_yunet_2023mar.onnx"
MODEL_URL = (
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
GAZE_MODEL_FILE = SCRIPT_DIR / "face_landmarker.task"
GAZE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
IS_WINDOWS = platform.system() == "Windows"


# --------------------------------------------------------------------------- #
#  Alerts: terminal beep + best-effort desktop notification
# --------------------------------------------------------------------------- #
def beep() -> None:
    """Make a short sound to get the user's attention. Best-effort only."""
    if winsound is not None:
        try:
            winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
            return
        except RuntimeError:
            pass
    print("\a", end="", flush=True)


def notify(title: str, message: str, enabled: bool = True) -> None:
    """Show a desktop notification (Windows toast), always beeping too."""
    beep()
    if not enabled or not IS_WINDOWS:
        return
    # Build a Windows toast with PowerShell's built-in WinRT APIs -- no extra
    # Python packages needed. Content is passed via env vars to dodge quoting.
    ps = (
        "$ErrorActionPreference='SilentlyContinue';"
        "[Windows.UI.Notifications.ToastNotificationManager,Windows.UI."
        "Notifications,ContentType=WindowsRuntime]|Out-Null;"
        "$t=[Windows.UI.Notifications.ToastNotificationManager]::"
        "GetTemplateContent([Windows.UI.Notifications.ToastTemplateType]"
        "::ToastText02);"
        "$x=$t.GetElementsByTagName('text');"
        "$x.Item(0).AppendChild($t.CreateTextNode($env:FG_TITLE))|Out-Null;"
        "$x.Item(1).AppendChild($t.CreateTextNode($env:FG_MSG))|Out-Null;"
        "$toast=[Windows.UI.Notifications.ToastNotification]::new($t);"
        "[Windows.UI.Notifications.ToastNotificationManager]::"
        "CreateToastNotifier('Focus Guardian').Show($toast);"
    )
    env = {**os.environ, "FG_TITLE": title, "FG_MSG": message}
    try:
        subprocess.Popen(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except OSError:
        pass  # toast is a nice-to-have; never let it crash the session


def fmt_duration(seconds: float) -> str:
    return str(timedelta(seconds=round(seconds)))


# --------------------------------------------------------------------------- #
#  Face detection: YuNet DNN (preferred) with a Haar cascade fallback
# --------------------------------------------------------------------------- #
def ensure_model() -> Path | None:
    """Return the YuNet model path, downloading it once if necessary."""
    if MODEL_FILE.exists() and MODEL_FILE.stat().st_size > 100_000:
        return MODEL_FILE
    try:
        print("Downloading face-detection model (one-time, ~230 KB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_FILE)
        if MODEL_FILE.stat().st_size > 100_000:
            print("  done.")
            return MODEL_FILE
    except Exception as exc:  # network/SSL/etc. -- fall back gracefully
        print(f"  could not download model ({exc}); using basic detector.")
    return None


class YuNetDetector:
    """Modern DNN detector. Returns boxes plus 5-point facial landmarks."""

    has_landmarks = True
    name = "YuNet DNN"

    def __init__(self, model_path: Path) -> None:
        self._net = cv2.FaceDetectorYN_create(
            str(model_path), "", (320, 320), 0.7, 0.3, 5000
        )
        self._size = (320, 320)

    def detect(self, frame) -> list[dict]:
        h, w = frame.shape[:2]
        if (w, h) != self._size:
            self._net.setInputSize((w, h))
            self._size = (w, h)
        _, faces = self._net.detect(frame)
        out: list[dict] = []
        if faces is None:
            return out
        for f in faces:
            x, y, bw, bh = f[0:4]
            out.append(
                {
                    "box": (int(x), int(y), int(bw), int(bh)),
                    "landmarks": {
                        "reye": (f[4], f[5]),
                        "leye": (f[6], f[7]),
                        "nose": (f[8], f[9]),
                        "rmouth": (f[10], f[11]),
                        "lmouth": (f[12], f[13]),
                    },
                    "score": float(f[14]),
                }
            )
        return out


class HaarDetector:
    """Fallback detector. Lighter, no landmarks, so no gaze info."""

    has_landmarks = False
    name = "Haar cascade"

    def __init__(self) -> None:
        cdir = Path(cv2.data.haarcascades)
        names = [
            "haarcascade_frontalface_alt2.xml",
            "haarcascade_frontalface_default.xml",
            "haarcascade_profileface.xml",
        ]
        self._cascades = []
        for n in names:
            c = cv2.CascadeClassifier(str(cdir / n))
            if not c.empty():
                self._cascades.append(c)
        if not self._cascades:
            raise RuntimeError("Could not load any face-detection model.")

    def detect(self, frame) -> list[dict]:
        gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        for cascade in self._cascades:
            faces = cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48)
            )
            if len(faces) > 0:
                return [
                    {"box": tuple(int(v) for v in f), "landmarks": None,
                     "score": 1.0}
                    for f in faces
                ]
        return []


def ensure_gaze_model() -> Path | None:
    """Return the MediaPipe FaceLandmarker model path, downloading if needed."""
    if GAZE_MODEL_FILE.exists() and GAZE_MODEL_FILE.stat().st_size > 1_000_000:
        return GAZE_MODEL_FILE
    try:
        print("Downloading eye-gaze model (one-time, ~3.8 MB)...")
        urllib.request.urlretrieve(GAZE_MODEL_URL, GAZE_MODEL_FILE)
        if GAZE_MODEL_FILE.stat().st_size > 1_000_000:
            print("  done.")
            return GAZE_MODEL_FILE
    except Exception as exc:  # network/SSL/etc.
        print(f"  could not download gaze model ({exc}).")
    return None


# Iris + eye-centre landmark indices used purely for the preview overlay.
_IRIS_IDS = tuple(range(468, 478))


class MediaPipeGazeEngine:
    """Tracks the face AND where the eyes point, via MediaPipe FaceLandmarker.

    Eye-look blendshapes tell us how far each eye is rotated up/down/in/out.
    We compare the live gaze against a calibrated "looking at the screen"
    baseline, so anything beyond a margin counts as eyes-off-screen.
    """

    has_landmarks = True
    has_eye_gaze = True
    name = "MediaPipe eye-gaze"

    def __init__(self, model_path: Path, away_threshold: float = 0.25) -> None:
        opts = mp_vision.FaceLandmarkerOptions(
            base_options=MpBaseOptions(model_asset_path=str(model_path)),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=True,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._lm = mp_vision.FaceLandmarker.create_from_options(opts)
        self.away_threshold = away_threshold
        self.blink_threshold = 0.5
        self._start = time.monotonic()
        self._last_ts = -1
        self._baseline: dict | None = None
        self._calib: list[dict] = []

    # -- gaze maths --------------------------------------------------------- #
    @staticmethod
    def _gaze_features(bs: dict) -> dict:
        g = bs.get
        # Average the left/right eye for each direction. "in" = toward the
        # nose, "out" = toward the ear, so sideways gaze lights up in+out.
        return {
            "left": (g("eyeLookOutLeft", 0.0) + g("eyeLookInRight", 0.0)) / 2,
            "right": (g("eyeLookInLeft", 0.0) + g("eyeLookOutRight", 0.0)) / 2,
            "up": (g("eyeLookUpLeft", 0.0) + g("eyeLookUpRight", 0.0)) / 2,
            "down": (g("eyeLookDownLeft", 0.0) + g("eyeLookDownRight", 0.0)) / 2,
        }

    def _looking_away(self, gaze: dict) -> bool:
        b = self._baseline or {"left": 0.0, "right": 0.0, "up": 0.0, "down": 0.0}
        deviation = max(gaze[k] - b[k] for k in ("left", "right", "up", "down"))
        return deviation > self.away_threshold

    # -- calibration -------------------------------------------------------- #
    def add_calibration_sample(self, gaze: dict | None) -> None:
        if gaze is not None:
            self._calib.append(gaze)

    def finalize_calibration(self) -> bool:
        if not self._calib:
            return False
        keys = ("left", "right", "up", "down")
        self._baseline = {k: sum(s[k] for s in self._calib) / len(self._calib)
                          for k in keys}
        self._calib = []
        return True

    def reset_calibration(self) -> None:
        self._calib = []
        self._baseline = None

    # -- detection ---------------------------------------------------------- #
    def detect(self, frame) -> list[dict]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = max(self._last_ts + 1, int((time.monotonic() - self._start) * 1000))
        self._last_ts = ts
        res = self._lm.detect_for_video(img, ts)
        if not res.face_landmarks:
            return []
        h, w = frame.shape[:2]
        lms = res.face_landmarks[0]
        xs = [p.x for p in lms]
        ys = [p.y for p in lms]
        box = (int(min(xs) * w), int(min(ys) * h),
               int((max(xs) - min(xs)) * w), int((max(ys) - min(ys)) * h))
        bs = {c.category_name: c.score for c in res.face_blendshapes[0]} \
            if res.face_blendshapes else {}
        gaze = self._gaze_features(bs)
        eyes_closed = (bs.get("eyeBlinkLeft", 0.0)
                       + bs.get("eyeBlinkRight", 0.0)) / 2 > self.blink_threshold
        draw_pts = [(int(lms[i].x * w), int(lms[i].y * h))
                    for i in _IRIS_IDS if i < len(lms)]
        return [{
            "box": box,
            "landmarks": None,
            "score": 1.0,
            "looking_away": self._looking_away(gaze),
            "eyes_closed": eyes_closed,
            "gaze": gaze,
            "draw_pts": draw_pts,
        }]


def build_detector(prefer_gaze: bool = True):
    """Pick the best available detector (MediaPipe eye-gaze > YuNet > Haar)."""
    if prefer_gaze and mp is not None:
        model = ensure_gaze_model()
        if model is not None:
            try:
                return MediaPipeGazeEngine(model)
            except Exception as exc:  # mediapipe init can fail various ways
                print(f"  eye-gaze engine unavailable ({exc}); falling back.")
    if hasattr(cv2, "FaceDetectorYN_create"):
        model = ensure_model()
        if model is not None:
            try:
                return YuNetDetector(model)
            except cv2.error:
                pass
    return HaarDetector()


# --------------------------------------------------------------------------- #
#  Session stats
# --------------------------------------------------------------------------- #
@dataclass
class SessionStats:
    """Tracks how a single focus session went."""

    started_at: datetime = field(default_factory=datetime.now)
    focused_seconds: float = 0.0
    distracted_seconds: float = 0.0  # present, but looking away
    away_seconds: float = 0.0
    nudges: int = 0
    pomodoros: int = 0  # completed work blocks
    longest_streak: float = 0.0  # longest unbroken focused stretch (seconds)

    @property
    def active_seconds(self) -> float:
        return self.focused_seconds + self.distracted_seconds + self.away_seconds

    @property
    def focus_ratio(self) -> float:
        if self.active_seconds == 0:
            return 0.0
        return self.focused_seconds / self.active_seconds

    def as_record(self) -> dict:
        return {
            "version": 2,
            "started_at": self.started_at.isoformat(timespec="seconds"),
            "duration_min": round(self.active_seconds / 60, 1),
            "focused_min": round(self.focused_seconds / 60, 1),
            "distracted_min": round(self.distracted_seconds / 60, 1),
            "away_min": round(self.away_seconds / 60, 1),
            "focus_ratio": round(self.focus_ratio, 3),
            "nudges": self.nudges,
            "pomodoros": self.pomodoros,
            "longest_streak_min": round(self.longest_streak / 60, 1),
        }


# --------------------------------------------------------------------------- #
#  The guardian
# --------------------------------------------------------------------------- #
# Detection state names
FOCUSED, DISTRACTED, AWAY, BREAK = "FOCUSED", "DISTRACTED", "AWAY", "BREAK"
CALIBRATE = "CALIBRATING"


class FocusGuardian:
    """Watches the webcam and decides whether you are focused or away."""

    def __init__(
        self,
        grace_seconds: float,
        goal_minutes: float | None,
        show_window: bool,
        camera_index: int,
        use_gaze: bool = True,
        notifications: bool = True,
        pomodoro: bool = False,
        work_minutes: float = 25.0,
        break_minutes: float = 5.0,
        calibrate: bool = True,
        gaze_threshold: float | None = None,
    ) -> None:
        self.grace_seconds = grace_seconds
        self.goal_seconds = goal_minutes * 60 if goal_minutes else None
        self.show_window = show_window
        self.camera_index = camera_index
        self.notifications = notifications

        self.detector = build_detector()
        self.use_gaze = use_gaze and self.detector.has_landmarks
        # True only when we have real eye-gaze (MediaPipe), not just head-yaw.
        self.has_eye_gaze = getattr(self.detector, "has_eye_gaze", False)
        if self.has_eye_gaze and gaze_threshold is not None:
            self.detector.away_threshold = gaze_threshold
        # Calibrate the "looking at screen" baseline at startup (eye-gaze only).
        self.calibrate = calibrate and self.has_eye_gaze and self.use_gaze
        self.calib_seconds = 3.0
        self._calib_until: float | None = None

        # Pomodoro config
        self.pomodoro = pomodoro
        self.work_seconds = work_minutes * 60
        self.break_seconds = break_minutes * 60
        self._phase = FOCUSED  # FOCUSED == "work phase"; BREAK == resting
        self._phase_start = time.monotonic()

        # Smoothing: a single dropped frame shouldn't flip the state.
        self.detection_grace = 1.5   # seconds a face may vanish, still "present"
        self.gaze_grace = 1.0        # seconds you may glance away, still "engaged"
        self.gaze_yaw_thresh = 0.5   # head-turn sensitivity (higher = stricter)
        self._last_seen: float | None = None
        self._engaged_seen: float | None = None

        self.stats = SessionStats()
        self._away_since: float | None = None
        self._distract_since: float | None = None
        self._nudged_away = False
        self._nudged_distract = False
        self._streak_start: float | None = None

    # -- detection helpers -------------------------------------------------- #
    @staticmethod
    def _largest(detections: list[dict]) -> dict:
        return max(detections, key=lambda d: d["box"][2] * d["box"][3])

    def _looking_away(self, landmarks: dict) -> bool:
        """Estimate head yaw from eye/nose landmarks."""
        rex = landmarks["reye"][0]
        lex = landmarks["leye"][0]
        nx = landmarks["nose"][0]
        iod = abs(lex - rex)  # inter-ocular distance in x
        if iod < 1:
            return False
        yaw = (nx - (rex + lex) / 2.0) / iod
        return abs(yaw) > self.gaze_yaw_thresh

    def _is_engaged(self, face: dict) -> bool:
        """Are the eyes on the screen? Real eye-gaze if we have it, else head-yaw."""
        looking_away = face.get("looking_away")
        if looking_away is None:  # legacy detector: approximate with head yaw
            if not face.get("landmarks"):
                return True
            looking_away = self._looking_away(face["landmarks"])
        return not (looking_away or face.get("eyes_closed", False))

    def _classify(self, detections: list[dict], now: float) -> str:
        """Turn raw detections into a smoothed FOCUSED/DISTRACTED/AWAY state."""
        if detections:
            self._last_seen = now
            if not self.use_gaze or self._is_engaged(self._largest(detections)):
                self._engaged_seen = now

        present = (
            self._last_seen is not None
            and now - self._last_seen <= self.detection_grace
        )
        if not present:
            return AWAY
        engaged = (
            self._engaged_seen is not None
            and now - self._engaged_seen <= self.gaze_grace
        )
        return FOCUSED if engaged else DISTRACTED

    # -- state accounting --------------------------------------------------- #
    def _update_state(self, state: str, dt: float, now: float) -> None:
        if state == FOCUSED:
            self.stats.focused_seconds += dt
            if self._streak_start is None:
                self._streak_start = now
            self.stats.longest_streak = max(
                self.stats.longest_streak, now - self._streak_start
            )
            self._reset_absence()
        elif state == DISTRACTED:
            self.stats.distracted_seconds += dt
            self._streak_start = None
            self._away_since = None
            self._nudged_away = False
            if self._distract_since is None:
                self._distract_since = now
            if (now - self._distract_since >= self.grace_seconds
                    and not self._nudged_distract):
                reason = ("looking away from the screen" if self.has_eye_gaze
                          else "looking away")
                self._nudge(reason, now - self._distract_since)
                self._nudged_distract = True
        else:  # AWAY
            self.stats.away_seconds += dt
            self._streak_start = None
            self._distract_since = None
            self._nudged_distract = False
            if self._away_since is None:
                self._away_since = now
            if (now - self._away_since >= self.grace_seconds
                    and not self._nudged_away):
                self._nudge("away from your desk", now - self._away_since)
                self._nudged_away = True

    def _reset_absence(self) -> None:
        self._away_since = None
        self._distract_since = None
        self._nudged_away = False
        self._nudged_distract = False

    def _nudge(self, reason: str, seconds: float) -> None:
        self.stats.nudges += 1
        msg = f"You've been {reason} for {fmt_duration(seconds)} -- back to it!"
        print(f"\n  [nudge #{self.stats.nudges}] {msg}")
        notify("Focus Guardian", msg, self.notifications)

    # -- pomodoro ----------------------------------------------------------- #
    def _tick_pomodoro(self, now: float) -> str | None:
        """Advance work/break phases. Returns BREAK while resting, else None."""
        if not self.pomodoro:
            return None
        elapsed = now - self._phase_start
        if self._phase == FOCUSED and elapsed >= self.work_seconds:
            self.stats.pomodoros += 1
            self._phase = BREAK
            self._phase_start = now
            mins = int(self.break_seconds // 60)
            print(f"\n  Pomodoro #{self.stats.pomodoros} done -- take a {mins} min break.")
            notify("Focus Guardian", f"Break time! Rest for {mins} min.",
                   self.notifications)
        elif self._phase == BREAK and elapsed >= self.break_seconds:
            self._phase = FOCUSED
            self._phase_start = now
            print("\n  Break over -- back to work!")
            notify("Focus Guardian", "Break over -- back to work!",
                   self.notifications)
            # Don't punish the first moments after a break.
            self._last_seen = now
            self._engaged_seen = now
            self._reset_absence()
        return BREAK if self._phase == BREAK else None

    def _phase_remaining(self, now: float) -> float:
        total = self.break_seconds if self._phase == BREAK else self.work_seconds
        return max(0.0, total - (now - self._phase_start))

    # -- drawing & status --------------------------------------------------- #
    @staticmethod
    def _state_color(state: str):
        return {
            FOCUSED: (0, 200, 0),
            DISTRACTED: (0, 165, 255),
            AWAY: (0, 0, 220),
            BREAK: (220, 160, 0),
            CALIBRATE: (220, 220, 0),
        }[state]

    def _draw(self, frame, detections: list[dict], state: str) -> None:
        color = self._state_color(state)
        for d in detections:
            x, y, w, h = d["box"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            if d.get("landmarks"):
                for px, py in d["landmarks"].values():
                    cv2.circle(frame, (int(px), int(py)), 2, color, -1)
            for px, py in d.get("draw_pts", ()):  # iris points (eye-gaze engine)
                cv2.circle(frame, (px, py), 2, (0, 220, 255), -1)
        cv2.putText(frame, state, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    color, 2)
        if state == DISTRACTED and self.has_eye_gaze:
            cv2.putText(frame, "eyes off screen", (12, 116),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        cv2.putText(
            frame,
            f"focus {fmt_duration(self.stats.focused_seconds)}",
            (12, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1,
        )
        if self.pomodoro:
            phase = "BREAK" if state == BREAK else "WORK"
            remaining = fmt_duration(self._phase_remaining(time.monotonic()))
            cv2.putText(frame, f"{phase} {remaining}", (12, 92),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1)

    def _print_status(self, state: str, now: float) -> None:
        bar_target = self.goal_seconds or max(self.stats.active_seconds, 1)
        filled = int(20 * min(self.stats.focused_seconds / bar_target, 1.0))
        bar = "#" * filled + "-" * (20 - filled)
        extra = ""
        if self.pomodoro:
            phase = "break" if state == BREAK else "work"
            extra = f"  [{phase} {fmt_duration(self._phase_remaining(now))}]"
        print(
            f"\r[{bar}] {state:<10} "
            f"focus {fmt_duration(self.stats.focused_seconds)}  "
            f"nudges {self.stats.nudges}{extra}   ",
            end="", flush=True,
        )

    # -- main loop ---------------------------------------------------------- #
    def run(self) -> None:
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open camera #{self.camera_index}. "
                "Is another app using it?"
            )

        gaze = ("eye-gaze" if self.has_eye_gaze
                else "head-pose" if self.use_gaze else "off")
        print(f"Focus Guardian is watching ({self.detector.name}, gaze {gaze}).")
        if self.calibrate:
            self._calib_until = time.monotonic() + self.calib_seconds
            print("Calibrating -- look straight at your screen for a few seconds.")
        print("Press Ctrl+C (or 'q' in the window) to stop"
              + ("; 'k' to recalibrate gaze.\n" if self.has_eye_gaze else ".\n"))
        last = time.monotonic()
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("\nLost the camera feed, stopping.")
                    break

                now = time.monotonic()
                dt, last = now - last, now

                detections = self.detector.detect(frame)
                on_break = self._tick_pomodoro(now)

                if self._calib_until is not None:
                    if now < self._calib_until:
                        state = CALIBRATE  # learning your on-screen baseline
                        if detections:
                            self.detector.add_calibration_sample(
                                self._largest(detections).get("gaze"))
                    else:
                        ok_cal = self.detector.finalize_calibration()
                        self._calib_until = None
                        print("\n  " + ("Gaze calibrated -- tracking your eyes now."
                              if ok_cal else
                              "No face seen while calibrating; using defaults."))
                        state = self._classify(detections, now)
                        self._update_state(state, dt, now)
                elif on_break:
                    state = BREAK  # rest; don't track focus or nudge
                else:
                    state = self._classify(detections, now)
                    self._update_state(state, dt, now)

                self._print_status(state, now)

                if (self._calib_until is None and self.goal_seconds
                        and not self.pomodoro
                        and self.stats.focused_seconds >= self.goal_seconds):
                    print("\n  Goal reached -- nice work! Wrapping up.")
                    break

                if self.show_window:
                    self._draw(frame, detections, state)
                    cv2.imshow("Focus Guardian (press q to quit)", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    if key == ord("k") and self.has_eye_gaze:
                        self.detector.reset_calibration()
                        self._calib_until = now + self.calib_seconds
                        print("\n  Recalibrating -- look at your screen...")
                else:
                    time.sleep(0.05)  # be gentle on the CPU in headless mode
        except KeyboardInterrupt:
            print("\nStopped by user.")
        finally:
            cap.release()
            if self.show_window:
                cv2.destroyAllWindows()
            self._report()

    # -- end-of-session report --------------------------------------------- #
    def _report(self) -> None:
        s = self.stats
        if s.active_seconds < 1:
            print("\nNo session recorded.")
            return
        print("\n\n" + "=" * 42)
        print("  SESSION REPORT")
        print("=" * 42)
        print(f"  Total time      {fmt_duration(s.active_seconds)}")
        print(f"  Focused         {fmt_duration(s.focused_seconds)}")
        if self.use_gaze:
            label = "Eyes off screen" if self.has_eye_gaze else "Looking away  "
            print(f"  {label} {fmt_duration(s.distracted_seconds)}")
        print(f"  Away            {fmt_duration(s.away_seconds)}")
        print(f"  Focus ratio     {s.focus_ratio:.0%}")
        print(f"  Longest streak  {fmt_duration(s.longest_streak)}")
        if self.pomodoro:
            print(f"  Pomodoros       {s.pomodoros}")
        print(f"  Nudges          {s.nudges}")
        print("=" * 42)
        self._save_record()

    def _save_record(self) -> None:
        history: list[dict] = []
        if LOG_FILE.exists():
            try:
                history = json.loads(LOG_FILE.read_text())
            except (json.JSONDecodeError, OSError):
                history = []
        history.append(self.stats.as_record())
        try:
            LOG_FILE.write_text(json.dumps(history, indent=2))
            print(f"  Saved to {LOG_FILE.name}  ({len(history)} sessions logged)")
        except OSError as exc:
            print(f"  Could not save log: {exc}")


# --------------------------------------------------------------------------- #
#  Stats dashboard
# --------------------------------------------------------------------------- #
def print_dashboard() -> int:
    if not LOG_FILE.exists():
        print("No sessions logged yet. Run a session first.")
        return 0
    try:
        history = json.loads(LOG_FILE.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        print(f"Could not read {LOG_FILE.name}: {exc}")
        return 1
    if not history:
        print("No sessions logged yet.")
        return 0

    # Aggregate by calendar day.
    days: dict[str, dict] = {}
    for rec in history:
        day = str(rec.get("started_at", ""))[:10] or "unknown"
        d = days.setdefault(
            day, {"focused": 0.0, "distracted": 0.0, "away": 0.0,
                  "nudges": 0, "pomodoros": 0, "sessions": 0}
        )
        d["focused"] += rec.get("focused_min", 0)
        d["distracted"] += rec.get("distracted_min", 0)
        d["away"] += rec.get("away_min", 0)
        d["nudges"] += rec.get("nudges", 0)
        d["pomodoros"] += rec.get("pomodoros", 0)
        d["sessions"] += 1

    print("\n" + "=" * 60)
    print("  FOCUS DASHBOARD")
    print("=" * 60)
    print(f"  {'Date':<12}{'Focus':>7}  {'Ratio':<22}{'Focused':>9}")
    print("  " + "-" * 56)
    tot_focused = tot_active = tot_nudges = tot_pomo = 0.0
    for day in sorted(days):
        d = days[day]
        active = d["focused"] + d["distracted"] + d["away"]
        ratio = d["focused"] / active if active else 0.0
        filled = round(ratio * 20)
        bar = "#" * filled + "-" * (20 - filled)
        print(f"  {day:<12}{ratio:>6.0%}  [{bar}]  {d['focused']:>6.0f} min")
        tot_focused += d["focused"]
        tot_active += active
        tot_nudges += d["nudges"]
        tot_pomo += d["pomodoros"]

    overall = tot_focused / tot_active if tot_active else 0.0
    print("  " + "-" * 56)
    print(f"  {len(history)} sessions over {len(days)} day(s)")
    print(f"  Total focused   {tot_focused / 60:.1f} h")
    print(f"  Overall ratio   {overall:.0%}")
    if tot_pomo:
        print(f"  Pomodoros       {int(tot_pomo)}")
    print(f"  Total nudges    {int(tot_nudges)}")
    print("=" * 60)
    return 0


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Webcam focus tracker that nudges you when you drift off."
    )
    p.add_argument("--grace", type=float, default=20.0,
                   help="Seconds away/looking-away before a nudge (default 20).")
    p.add_argument("--goal", type=float, default=None,
                   help="Focus goal in minutes; the session ends when you hit it.")
    p.add_argument("--camera", type=int, default=0,
                   help="Camera index (default 0).")
    p.add_argument("--no-window", action="store_true",
                   help="Run headless without the preview window.")
    p.add_argument("--no-gaze", action="store_true",
                   help="Disable eye/looking-away detection (presence only).")
    p.add_argument("--no-calibrate", action="store_true",
                   help="Skip the startup eye-gaze calibration.")
    p.add_argument("--gaze-threshold", type=float, default=None,
                   help="Eye-gaze strictness: 0.1 strict to 0.5 lenient "
                        "(default 0.25). Lower flags smaller eye movements.")
    p.add_argument("--no-notify", action="store_true",
                   help="Disable desktop notifications (terminal beep only).")
    p.add_argument("--pomodoro", action="store_true",
                   help="Enable Pomodoro work/break cycles.")
    p.add_argument("--work", type=float, default=None,
                   help="Pomodoro work minutes (default 25; implies --pomodoro).")
    p.add_argument("--break", dest="break_min", type=float, default=None,
                   help="Pomodoro break minutes (default 5; implies --pomodoro).")
    p.add_argument("--stats", action="store_true",
                   help="Print the focus dashboard and exit.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.stats:
        return print_dashboard()

    pomodoro = args.pomodoro or args.work is not None or args.break_min is not None
    try:
        guardian = FocusGuardian(
            grace_seconds=args.grace,
            goal_minutes=args.goal,
            show_window=not args.no_window,
            camera_index=args.camera,
            use_gaze=not args.no_gaze,
            notifications=not args.no_notify,
            pomodoro=pomodoro,
            work_minutes=args.work if args.work is not None else 25.0,
            break_minutes=args.break_min if args.break_min is not None else 5.0,
            calibrate=not args.no_calibrate,
            gaze_threshold=args.gaze_threshold,
        )
        guardian.run()
    except (RuntimeError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
